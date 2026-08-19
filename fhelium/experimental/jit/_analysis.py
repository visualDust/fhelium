"""Pure requirement and structural-state analyses for JIT Programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.dialects.builtin import IntAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Operation, SSAValue

from fhelium.core import EvaluationKeyRequirements

from ._dialect import (
    MaterialRefOp,
    ResourceRefOp,
    operation_name,
    value_role,
)
from ._program import Program

_STRUCTURAL_OPERATION_NAMES = frozenset(
    {"builtin.module", "func.func", "func.return"}
)
RUNTIME_CKKS_OPERATION_NAMES = frozenset(
    {
        "fhelium.ckks.negate",
        "fhelium.ckks.rotate",
        "fhelium.ckks.to_ntt",
        "fhelium.ckks.from_ntt",
        "fhelium.ckks.add",
        "fhelium.ckks.subtract",
        "fhelium.ckks.multiply",
        "fhelium.ckks.add_plaintext",
        "fhelium.ckks.multiply_plaintext",
        "fhelium.ckks.relinearize",
        "fhelium.ckks.rescale",
        *(
            f"fhelium.ckks.prepare.{operation}.{role}"
            for operation in ("add", "multiply")
            for role in ("message", "plaintext", "static")
        ),
    }
)
RUNTIME_OPERATION_NAMES = RUNTIME_CKKS_OPERATION_NAMES | {
    "fhelium.material.ref",
    "fhelium.resource.ref",
    "fhelium.constant",
    "torch.call",
}


@dataclass(frozen=True)
class ProgramRequirements:
    """Describe symbolic runtime capabilities referenced by one entry.

    ``analyze_requirements`` derives this immutable record from Program IR.
    Workspace comparison, materialization, parameter selection, and the final
    readiness decision are separate operations.
    """

    operations: frozenset[str]
    unknown_operations: frozenset[str]
    materials: frozenset[str]
    resources: frozenset[str]
    torch_targets: frozenset[str]
    malformed_references: tuple[str, ...]
    rotation_steps: frozenset[int]
    requires_relinearization: bool
    requires_engine: bool
    return_count: int | None


@dataclass(frozen=True)
class InferredValueState:
    """Expose one SSA value's structural role, type, and open metadata."""

    role: str | None
    type: Attribute
    metadata: Mapping[str, Attribute]


def _string_attribute(operation: Operation, *names: str) -> str | None:
    for name in names:
        attribute = operation.attributes.get(name)
        if isinstance(attribute, StringAttr) and attribute.data:
            return attribute.data
    return None


def _integer_attribute(operation: Operation, *names: str) -> int | None:
    for name in names:
        attribute = operation.attributes.get(name)
        if isinstance(attribute, IntegerAttr):
            return int(attribute.value.data)
        if isinstance(attribute, IntAttr):
            return int(attribute.data)
    return None


def analyze_requirements(
    program: Program,
    *,
    entry: str = "main",
) -> ProgramRequirements:
    """Collect runtime capabilities referenced by the selected entry block.

    The scan covers entry arguments and direct operations in ``entry``'s unique
    block. It records operation names, extension operations, Torch targets,
    material/resource symbols, malformed references, CKKS engine use, rotation
    steps, relinearization, and return arity. Unknown and not-yet-lowered
    operations remain requirements. ``return_count=None`` represents a
    missing entry, a multi-block entry, or a non-unique return for subsequent
    readiness diagnostics.

    Standard transformation passes may scan all top-level function blocks; this
    analysis intentionally describes only the entry selected for execution.
    """

    if not isinstance(program, Program):
        raise TypeError("analyze_requirements expects a Program")
    if not isinstance(entry, str):
        raise TypeError("entry must be a string")

    operations: set[str] = set()
    unknown_operations: set[str] = set()
    materials: set[str] = set()
    resources: set[str] = set()
    torch_targets: set[str] = set()
    malformed: list[str] = []
    rotation_steps: set[int] = set()
    requires_relinearization = False
    requires_engine = False

    try:
        block = program.entry_block(entry)
    except (KeyError, ValueError):
        block = None

    selected_operations = () if block is None else tuple(block.ops)
    requires_engine = bool(
        block is not None
        and any(value_role(argument) == "encrypted" for argument in block.args)
    )

    for operation in selected_operations:
        name = operation_name(operation)
        if name in _STRUCTURAL_OPERATION_NAMES:
            continue
        operations.add(name)
        if (
            isinstance(operation, MaterialRefOp)
            or name == "fhelium.material.ref"
        ):
            symbol = _string_attribute(operation, "symbol")
            if symbol is None:
                malformed.append("fhelium.material.ref")
            else:
                materials.add(symbol)
            continue
        if (
            isinstance(operation, ResourceRefOp)
            or name == "fhelium.resource.ref"
        ):
            symbol = _string_attribute(operation, "symbol")
            if symbol is None:
                malformed.append("fhelium.resource.ref")
            else:
                resources.add(symbol)
            continue
        if name == "torch.call":
            target = _string_attribute(
                operation,
                "fhelium.call.target",
            )
            if target is None:
                malformed.append("torch.call")
            else:
                torch_targets.add(target)
            continue
        if name in RUNTIME_CKKS_OPERATION_NAMES:
            requires_engine = True
            if name == "fhelium.ckks.rotate":
                step = _integer_attribute(operation, "shift")
                if step is None:
                    malformed.append("fhelium.ckks.rotate")
                elif step:
                    rotation_steps.add(step)
            elif name == "fhelium.ckks.relinearize":
                requires_relinearization = True
            continue
        if name not in RUNTIME_OPERATION_NAMES:
            unknown_operations.add(name)

    if block is None:
        return_count = None
    else:
        terminators = tuple(
            operation
            for operation in block.ops
            if isinstance(operation, ReturnOp)
        )
        return_count = (
            len(terminators[0].arguments) if len(terminators) == 1 else None
        )

    return ProgramRequirements(
        operations=frozenset(operations),
        unknown_operations=frozenset(unknown_operations),
        materials=frozenset(materials),
        resources=frozenset(resources),
        torch_targets=frozenset(torch_targets),
        malformed_references=tuple(malformed),
        rotation_steps=frozenset(rotation_steps),
        requires_relinearization=requires_relinearization,
        requires_engine=requires_engine,
        return_count=return_count,
    )


def analyze_evaluation_key_requirements(
    program: Program,
    *,
    entry: str = "main",
) -> EvaluationKeyRequirements:
    """Derive ``entry``'s key requirements from explicit CKKS operations."""

    requirements = analyze_requirements(program, entry=entry)
    return EvaluationKeyRequirements(
        rotation_steps=requirements.rotation_steps,
        requires_relinearization=requirements.requires_relinearization,
    )


def analyze_value_states(
    program: Program,
    *,
    entry: str = "main",
) -> Mapping[SSAValue, InferredValueState]:
    """Return structural type metadata for ``entry`` arguments and results.

    The mapping exposes open role/type attributes exactly as represented in IR.
    Specialized numerical state or parameter passes may publish richer analyses
    in the caller's Workspace while retaining this structural view.
    """

    block = program.entry_block(entry)
    values = (
        *block.args,
        *(result for op in block.ops for result in op.results),
    )
    result: dict[SSAValue, InferredValueState] = {}
    for value in values:
        state = getattr(value.type, "state", None)
        metadata = (
            MappingProxyType(dict(state.data))
            if state is not None and hasattr(state, "data")
            else MappingProxyType({})
        )
        result[value] = InferredValueState(
            role=value_role(value),
            type=value.type,
            metadata=metadata,
        )
    return MappingProxyType(result)


__all__ = [
    "RUNTIME_CKKS_OPERATION_NAMES",
    "RUNTIME_OPERATION_NAMES",
    "InferredValueState",
    "ProgramRequirements",
    "analyze_evaluation_key_requirements",
    "analyze_requirements",
    "analyze_value_states",
]
