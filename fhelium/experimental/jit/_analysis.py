"""Pure requirement and structural-state analyses used by runtime JIT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import (
    IntAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation, SSAValue

from fhelium.ir import operation_name, value_role
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core, semantic, torch

_STRUCTURAL_OPERATION_NAMES = frozenset(
    {"builtin.module", "func.func", "func.return"}
)
RUNTIME_CKKS_OPERATION_TYPES: tuple[type[Operation], ...] = (
    ckks.EncodeOp,
    ckks.DecodeOp,
    ckks.IntegerCoefficientsToRnsOp,
    ckks.EncryptOp,
    ckks.DecryptOp,
    ckks.NegateOp,
    ckks.RotateOp,
    ckks.RotateManyOp,
    ckks.ToNttOp,
    ckks.FromNttOp,
    ckks.ToMontgomeryResiduesOp,
    ckks.ToStandardResiduesOp,
    ckks.AddOp,
    ckks.SubtractOp,
    ckks.MultiplyOp,
    ckks.AddScalarOp,
    ckks.MultiplyScalarOp,
    ckks.MultiplyIntegerScalarOp,
    ckks.AddPlaintextOp,
    ckks.MultiplyPlaintextOp,
    ckks.AddCompressedPlaintextOp,
    ckks.MultiplyCompressedPlaintextOp,
    ckks.RelinearizeOp,
    ckks.SwitchKeyOp,
    ckks.ConjugateOp,
    ckks.RescaleOp,
    ckks.ModSwitchOp,
    ckks.ReinterpretScaleOp,
    ckks.PrepareAddMessageOp,
    ckks.PrepareAddPlaintextOp,
    ckks.PrepareAddStaticOp,
    ckks.PrepareMultiplyMessageOp,
    ckks.PrepareMultiplyPlaintextOp,
    ckks.PrepareMultiplyStaticOp,
)
RUNTIME_POINTWISE_OPERATION_TYPES: tuple[type[Operation], ...] = (
    semantic.AddOp,
    semantic.SubtractOp,
    semantic.MultiplyOp,
    semantic.NegateOp,
    semantic.RollOp,
)
RUNTIME_AUXILIARY_OPERATION_TYPES: tuple[type[Operation], ...] = (
    core.MaterialRefOp,
    core.ResourceRefOp,
    core.ConstantOp,
    torch.CallOp,
    UnrealizedConversionCastOp,
)
RUNTIME_OPERATION_TYPES = (
    *RUNTIME_CKKS_OPERATION_TYPES,
    *RUNTIME_POINTWISE_OPERATION_TYPES,
    *RUNTIME_AUXILIARY_OPERATION_TYPES,
)
RUNTIME_CKKS_OPERATION_NAMES = frozenset(
    operation_type.name for operation_type in RUNTIME_CKKS_OPERATION_TYPES
)
RUNTIME_OPERATION_NAMES = frozenset(
    operation_type.name for operation_type in RUNTIME_OPERATION_TYPES
)


@dataclass(frozen=True)
class ProgramRequirements:
    """Runtime operations and capabilities referenced by one Program entry.

    The record contains operation and handler names, material/resource symbols,
    malformed references, Eager Engine/key roles, and return arity.
    """

    operations: frozenset[str]
    unknown_operations: frozenset[str]
    materials: frozenset[str]
    resources: frozenset[str]
    torch_targets: frozenset[str]
    malformed_references: tuple[str, ...]
    rotation_steps: frozenset[int]
    requires_relinearization: bool
    requires_conjugation: bool
    key_switch_symbols: frozenset[str]
    rotation_key_symbols: tuple[tuple[str, int], ...]
    public_key_symbols: frozenset[str]
    secret_key_symbols: frozenset[str]
    requires_eager_engine: bool
    return_count: int | None


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


def rotation_key_reference(value: SSAValue) -> tuple[str, int]:
    """Return the resource symbol and represented step of one key operand."""

    owner = value.owner
    if not isinstance(owner, core.ResourceRefOp):
        raise ValueError(
            "rotation key operand must be defined by fhelium.resource.ref"
        )
    symbol = _string_attribute(owner, "symbol")
    if symbol is None:
        raise ValueError("rotation key resource requires a nonempty symbol")
    state = getattr(value.type, "state", None)
    data = getattr(state, "data", None)
    if not isinstance(data, Mapping):
        raise ValueError("rotation key operand has no represented state")
    represented = data.get("rotation_step")
    if isinstance(represented, IntegerAttr):
        return symbol, int(represented.value.data)
    if isinstance(represented, IntAttr):
        return symbol, int(represented.data)
    raise ValueError("rotation key operand has no represented rotation_step")


def analyze_requirements(
    program: Program,
    *,
    entry: str = "main",
) -> ProgramRequirements:
    """Collect runtime capabilities referenced by the selected entry block.

    The scan covers entry arguments and direct operations in ``entry``'s unique
    block. It records operation names, extension operations, Torch targets,
    material/resource symbols, malformed references, Eager Engine use, rotation
    steps, relinearization, and return arity. Unknown and not-yet-lowered
    operations remain requirements. ``return_count=None`` represents a
    missing entry, a multi-block entry, or a non-unique return for subsequent
    interpreter coverage diagnostics.

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
    requires_conjugation = False
    key_switch_symbols: set[str] = set()
    rotation_key_symbols: list[tuple[str, int]] = []
    public_key_symbols: set[str] = set()
    secret_key_symbols: set[str] = set()
    requires_eager_engine = False

    try:
        block = program.single_block(entry)
    except (KeyError, ValueError):
        block = None

    selected_operations = () if block is None else tuple(block.ops)
    requires_eager_engine = bool(
        block is not None
        and any(value_role(argument) == "encrypted" for argument in block.args)
    )

    for operation in selected_operations:
        name = operation_name(operation)
        if name in _STRUCTURAL_OPERATION_NAMES:
            continue
        operations.add(name)
        if isinstance(operation, core.MaterialRefOp):
            symbol = _string_attribute(operation, "symbol")
            if symbol is None:
                malformed.append(core.MaterialRefOp.name)
            else:
                materials.add(symbol)
            continue
        if isinstance(operation, core.ResourceRefOp):
            symbol = _string_attribute(operation, "symbol")
            if symbol is None:
                malformed.append(core.ResourceRefOp.name)
            else:
                resources.add(symbol)
            continue
        if isinstance(operation, torch.CallOp):
            target = _string_attribute(
                operation,
                "fhelium.call.target",
            )
            if target is None:
                malformed.append(torch.CallOp.name)
            else:
                torch_targets.add(target)
            continue
        if isinstance(operation, RUNTIME_CKKS_OPERATION_TYPES):
            requires_eager_engine = True
            if isinstance(operation, ckks.RotateOp):
                try:
                    rotation_key_symbols.append(
                        rotation_key_reference(operation.key)
                    )
                except ValueError:
                    malformed.append(ckks.RotateOp.name)
            elif isinstance(operation, ckks.RotateManyOp):
                try:
                    references = tuple(
                        rotation_key_reference(key) for key in operation.keys
                    )
                except ValueError:
                    malformed.append(ckks.RotateManyOp.name)
                else:
                    rotation_key_symbols.extend(references)
            elif isinstance(operation, ckks.RelinearizeOp):
                requires_relinearization = True
            elif isinstance(operation, ckks.ConjugateOp):
                requires_conjugation = True
            elif isinstance(operation, ckks.SwitchKeyOp):
                if operation.key_symbol.data:
                    key_switch_symbols.add(operation.key_symbol.data)
                else:
                    malformed.append(ckks.SwitchKeyOp.name)
            elif isinstance(operation, ckks.EncryptOp):
                public_key_symbols.add(operation.key_symbol.data)
            elif isinstance(operation, ckks.DecryptOp):
                secret_key_symbols.add(operation.key_symbol.data)
            continue
        if not isinstance(operation, RUNTIME_OPERATION_TYPES):
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
        requires_conjugation=requires_conjugation,
        key_switch_symbols=frozenset(key_switch_symbols),
        rotation_key_symbols=tuple(rotation_key_symbols),
        public_key_symbols=frozenset(public_key_symbols),
        secret_key_symbols=frozenset(secret_key_symbols),
        requires_eager_engine=requires_eager_engine,
        return_count=return_count,
    )


__all__ = [
    "RUNTIME_CKKS_OPERATION_NAMES",
    "RUNTIME_CKKS_OPERATION_TYPES",
    "RUNTIME_OPERATION_NAMES",
    "RUNTIME_OPERATION_TYPES",
    "ProgramRequirements",
    "analyze_requirements",
    "rotation_key_reference",
]
