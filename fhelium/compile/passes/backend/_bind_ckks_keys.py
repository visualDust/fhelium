"""Match caller-supplied CKKS key values to Program resource roles."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
)

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects.builtin import IntegerAttr

from fhelium.backend.ckks.resources import (
    CKKS_KEY_RESOURCE_KINDS,
    ROTATION_KEY_RESOURCE_KIND,
    ckks_key_resource_kind,
)
from fhelium.backend.execution import ProgramDispatchTable
from fhelium.backend.resources import (
    BoundResource,
    ResourceBindings,
    ResourceRequirement,
)
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core
from fhelium.values import KeySwitchKey, PublicKey, RotationKey, SecretKey

from ._operations import program_resource_requirements


def _rotation_steps_by_symbol(program: Program) -> dict[str, int]:
    steps: dict[str, int] = {}
    for operation in program.single_block("main").walk():
        if not isinstance(operation, core.ResourceRefOp):
            continue
        if (
            operation.kind is None
            or operation.kind.data != ROTATION_KEY_RESOURCE_KIND
        ):
            continue
        if operation.symbol is None:
            raise ValueError("Rotation-key resource reference has no symbol")
        value_type = operation.value.type
        if not isinstance(value_type, ckks.EvaluationKeyType):
            raise TypeError("Rotation-key resource has another IR type")
        represented = value_type.state.data.get("rotation_step")
        if not isinstance(represented, IntegerAttr):
            raise ValueError(
                "Rotation-key resource requires a represented rotation step"
            )
        symbol = operation.symbol.data
        step = int(represented.value.data)
        previous = steps.get(symbol)
        if previous is not None and previous != step:
            raise ValueError(
                f"Rotation-key resource {symbol!r} represents two steps"
            )
        steps[symbol] = step
    return steps


@dataclass(frozen=True)
class BindCkksKeysPass:
    """Match ordinary CKKS key objects to the key roles required by a Program.

    Rotation keys match resource operands by their normalized rotation step.
    Other concrete key classes match when exactly one unbound Program resource
    role of that kind remains. Extra keys are ignored. Ambiguous same-kind roles
    remain available through caller-constructed low-level ``ResourceBindings``.
    """

    keys: tuple[PublicKey | SecretKey | KeySwitchKey, ...]
    name: str = field(default="bind-ckks-keys", init=False)

    def __init__(
        self,
        keys: Iterable[PublicKey | SecretKey | KeySwitchKey],
    ) -> None:
        selected = tuple(keys)
        for key in selected:
            ckks_key_resource_kind(key)
        object.__setattr__(self, "keys", selected)
        object.__setattr__(self, "name", "bind-ckks-keys")

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        dispatch_table = shared_data.get(ProgramDispatchTable)
        if not isinstance(dispatch_table, ProgramDispatchTable):
            raise RuntimeError(
                "BindCkksKeysPass requires ResolveBackendOperationsPass"
            )
        provided = shared_data.get(ResourceBindings)
        if not isinstance(provided, ResourceBindings):
            raise RuntimeError(
                "BindCkksKeysPass requires InitializeResourceBindingsPass"
            )
        bindings = provided
        _, requirements = program_resource_requirements(program, dispatch_table)
        bound_symbols = frozenset(bindings.symbols)
        unresolved = tuple(
            requirement
            for requirement in requirements
            if requirement.kind in CKKS_KEY_RESOURCE_KINDS
            and requirement.symbol not in bound_symbols
        )

        keys_by_kind: dict[str, list[object]] = defaultdict(list)
        for key in self.keys:
            keys_by_kind[ckks_key_resource_kind(key)].append(key)

        selected: list[BoundResource] = []
        rotation_steps = _rotation_steps_by_symbol(program)
        rotation_keys: dict[int, RotationKey] = {}
        for supplied in keys_by_kind[ROTATION_KEY_RESOURCE_KIND]:
            key = cast(RotationKey, supplied)
            previous = rotation_keys.get(key.rotation_step)
            if previous is not None and previous is not key:
                raise ValueError(
                    "Multiple caller rotation keys represent step "
                    f"{key.rotation_step}"
                )
            rotation_keys[key.rotation_step] = key

        remaining_by_kind: dict[str, list[ResourceRequirement]] = defaultdict(
            list
        )
        for requirement in unresolved:
            if requirement.kind == ROTATION_KEY_RESOURCE_KIND:
                step = rotation_steps.get(requirement.symbol)
                if step is None:
                    continue
                key = rotation_keys.get(step)
                if key is not None:
                    selected.append(
                        BoundResource(requirement.symbol, requirement.kind, key)
                    )
                continue
            remaining_by_kind[requirement.kind].append(requirement)

        for kind, kind_requirements in remaining_by_kind.items():
            candidates = keys_by_kind[kind]
            if not candidates:
                continue
            if len(kind_requirements) != 1 or len(candidates) != 1:
                raise ValueError(
                    f"Cannot infer {kind!r} resources: Program has "
                    f"{len(kind_requirements)} unbound roles and caller supplied "
                    f"{len(candidates)} keys; bind ambiguous roles by symbol"
                )
            requirement = kind_requirements[0]
            selected.append(
                BoundResource(
                    requirement.symbol, requirement.kind, candidates[0]
                )
            )

        if selected:
            bindings = bindings.overlay(ResourceBindings(tuple(selected)))
        shared_data[ResourceBindings] = bindings
        return PassResult.unchanged(
            program,
            matched=len(selected),
            skipped=len(unresolved) - len(selected),
        )


__all__ = ["BindCkksKeysPass"]
