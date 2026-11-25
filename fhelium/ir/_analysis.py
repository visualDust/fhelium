"""Read-only analyses over mixed-level Programs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from typing import cast

from xdsl.dialects.builtin import ArrayAttr, IntAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation, SSAValue

from fhelium.values import EvaluationKeyRequirements

from ._dialect import operation_name, value_role
from .dialects import ckks, core, logical, rns
from ._program import Program


@dataclass(frozen=True)
class ProgramInventory:
    """Operation counts and dialect/function names found in a Program.

    ``operation_counts`` counts every walked operation by its registered or
    textual name. ``dialects`` contains the first namespace component of those
    names. ``functions`` preserves top-level function order.
    """

    operation_counts: Mapping[str, int]
    dialects: frozenset[str]
    functions: tuple[str, ...]


@dataclass(frozen=True)
class ValueState:
    """Expose one SSA value's type, known role, and open metadata."""

    role: str | None
    type: Attribute
    metadata: Mapping[str, Attribute]


def inventory_program(program: Program) -> ProgramInventory:
    """Count operations and list dialect and function names in ``program``.

    The result contains operation counts, dialect namespace prefixes, and
    ordered top-level function names. This read-only inspection does not
    validate CKKS state, derive resource requirements, query backend coverage,
    or modify the Program.
    """

    if not isinstance(program, Program):
        raise TypeError("inventory_program expects a Program")
    counts = Counter(operation_name(operation) for operation in program.walk())
    dialects = frozenset(
        name.partition(".")[0] if "." in name else name for name in counts
    )
    return ProgramInventory(
        MappingProxyType(dict(sorted(counts.items()))),
        dialects,
        tuple(function.sym_name.data for function in program.functions),
    )


def analyze_value_states(
    program: Program,
    *,
    function: str = "main",
) -> Mapping[SSAValue, ValueState]:
    """Return represented state for one single-block function's SSA values."""

    block = program.single_block(function)
    values = (
        *block.args,
        *(result for operation in block.ops for result in operation.results),
    )
    result: dict[SSAValue, ValueState] = {}
    for value in values:
        state = getattr(value.type, "state", None)
        metadata = (
            MappingProxyType(dict(state.data))
            if state is not None and hasattr(state, "data")
            else MappingProxyType({})
        )
        result[value] = ValueState(
            role=value_role(value),
            type=value.type,
            metadata=metadata,
        )
    return MappingProxyType(result)


def _integer_attribute(operation: Operation, name: str) -> int | None:
    attribute = operation.attributes.get(name)
    if isinstance(attribute, IntegerAttr):
        return int(attribute.value.data)
    if isinstance(attribute, IntAttr):
        return int(attribute.data)
    return None


def _rotation_key_step(program: Program, value: SSAValue) -> int | None:
    """Return the step described for a lowered rotation-key material."""

    owner = value.owner
    if not isinstance(owner, core.MaterialRefOp) or owner.symbol is None:
        return None
    step = (program.material_description(owner.symbol.data) or {}).get(
        "rotation_step"
    )
    return step if type(step) is int else None


def analyze_evaluation_key_requirements(
    program: Program,
    *,
    entry: str = "main",
) -> EvaluationKeyRequirements:
    """List the evaluation-key capabilities requested by ``entry``.

    The analysis reads logical/CKKS operations and lowered evaluation-key
    operand uses from the selected single-block function and its nested regions. It returns
    symbolic requirements only; it does not generate, load, bind, or validate
    key objects. Re-run it after any transformation that may add or remove
    those operations. Unresolved logical rolls contribute their integer
    ``shift``; resolved rotation operations contribute their
    ``rotation_step``/``rotation_steps``/``baby_steps`` attributes, and lowered
    key products the step described for their key material. ``entry`` must name one
    single-block function. Generic key switching remains caller-named and is
    not representable by EvaluationKeyRequirements.
    """

    if not isinstance(program, Program):
        raise TypeError("analyze_evaluation_key_requirements expects a Program")
    if not isinstance(entry, str):
        raise TypeError("entry must be a string")

    rotation_steps: set[int] = set()
    requires_relinearization = False
    requires_conjugation = False
    for operation in program.single_block(entry).walk():
        if isinstance(operation, ckks.RotateOp):
            step = int(operation.rotation_step.value.data)
            if step:
                rotation_steps.add(step)
        elif isinstance(operation, ckks.RotationSumOp):
            rotation_steps.update(
                int(step.value.data)
                for step in operation.rotation_steps
                if step.value.data
            )
        elif isinstance(operation, ckks.RotateManyOp):
            rotation_steps.update(
                int(step.value.data)
                for step in cast(
                    ArrayAttr[IntegerAttr],
                    operation.attributes["rotation_steps"],
                ).data
                if step.value.data
            )
        elif isinstance(operation, ckks.GroupedRotationWeightedSumOp):
            rotation_steps.update(
                int(step.value.data)
                for step in operation.baby_steps
                if step.value.data
            )
        elif isinstance(operation, logical.RollEncryptedOp):
            step = _integer_attribute(operation, "shift")
            if step is None:
                raise ValueError(
                    "fhelium_logical.roll.encrypted requires an integer shift "
                    "before evaluation-key requirements can be derived"
                )
            if step:
                rotation_steps.add(step)
        elif isinstance(operation, ckks.RelinearizeOp):
            requires_relinearization = True
        elif isinstance(operation, ckks.ConjugateOp):
            requires_conjugation = True
        elif isinstance(operation, rns.KeySwitchDigitProductOp):
            role = operation.attributes.get("key_role")
            if role == StringAttr("relinearization"):
                requires_relinearization = True
            elif role == StringAttr("conjugation"):
                requires_conjugation = True
            elif role == StringAttr("rotation"):
                step = _rotation_key_step(program, operation.key)
                if step is None:
                    raise ValueError(
                        "Lowered rotation key operand needs a described step"
                    )
                if step:
                    rotation_steps.add(step)

    return EvaluationKeyRequirements(
        rotation_steps=frozenset(rotation_steps),
        requires_relinearization=requires_relinearization,
        requires_conjugation=requires_conjugation,
    )


__all__ = [
    "ProgramInventory",
    "ValueState",
    "analyze_evaluation_key_requirements",
    "analyze_value_states",
    "inventory_program",
]
