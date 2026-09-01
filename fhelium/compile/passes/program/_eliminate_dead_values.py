"""Remove unreachable known-pure values while preserving effect roots."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.ir import Operation
from xdsl.rewriter import Rewriter

from fhelium.ir import Program

from fhelium.ir.dialects import ckks, core, logical, semantic
from .._operation_transforms import display_name, program_operations

_KNOWN_PURE_TYPES: tuple[type[Operation], ...] = (
    core.MaterialRefOp,
    core.ConstantOp,
    semantic.AddOp,
    semantic.SubtractOp,
    semantic.MultiplyOp,
    semantic.NegateOp,
    semantic.RollOp,
    logical.AddEncryptedEncryptedOp,
    logical.AddEncryptedPublicOp,
    logical.AddPublicEncryptedOp,
    logical.SubtractEncryptedEncryptedOp,
    logical.SubtractEncryptedPublicOp,
    logical.SubtractPublicEncryptedOp,
    logical.MultiplyEncryptedEncryptedOp,
    logical.MultiplyEncryptedPublicOp,
    logical.MultiplyPublicEncryptedOp,
    logical.NegateEncryptedOp,
    logical.RollEncryptedOp,
    ckks.NegateOp,
    ckks.RotateOp,
    ckks.RotateManyOp,
    ckks.ToNttOp,
    ckks.FromNttOp,
    ckks.AddOp,
    ckks.SubtractOp,
    ckks.MultiplyOp,
    ckks.AddPlaintextOp,
    ckks.MultiplyPlaintextOp,
    ckks.RelinearizeOp,
    ckks.RescaleOp,
    ckks.PrepareAddMessageOp,
    ckks.PrepareAddPlaintextOp,
    ckks.PrepareAddStaticOp,
    ckks.PrepareMultiplyMessageOp,
    ckks.PrepareMultiplyPlaintextOp,
    ckks.PrepareMultiplyStaticOp,
    UnrealizedConversionCastOp,
)


def _known_pure(operation: Operation) -> bool:
    """Return whether a registered operation class has closed pure semantics."""

    return isinstance(operation, _KNOWN_PURE_TYPES)


@dataclass(frozen=True)
class EliminateDeadValuesPass:
    """Delete dead registered operations from a closed pure class set.

    Every unknown or extension operation is an effectful liveness root. Known
    operations carrying properties, regions, or successors are also kept,
    so the pass preserves unclassified and structural effects.
    """

    name: str = "eliminate-dead-values"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Compute module-wide SSA liveness from returns and effect roots."""

        del workspace
        operations = program_operations(program)
        operation_set = frozenset(operations)
        root_operations = {
            operation
            for operation in program.walk(include_module=True)
            if operation not in operation_set
            or not _known_pure(operation)
            or bool(operation.properties)
        }
        live_values = {
            operand
            for operation in root_operations
            for operand in operation.operands
        }
        changed = True
        while changed:
            changed = False
            for operation in reversed(operations):
                if any(result in live_values for result in operation.results):
                    for operand in operation.operands:
                        if operand not in live_values:
                            live_values.add(operand)
                            changed = True

        removable = tuple(
            operation
            for operation in operations
            if _known_pure(operation)
            and not operation.properties
            and not operation.regions
            and not operation.successors
            and not any(result in live_values for result in operation.results)
        )
        if not removable:
            return PassResult.unchanged(program)
        names = tuple(display_name(operation) for operation in removable)
        for operation in reversed(removable):
            Rewriter.erase_op(operation)
        preview = ", ".join(names[:8])
        if len(names) > 8:
            preview += f", ... (+{len(names) - 8})"
        return PassResult(
            program,
            PassStats(
                matched=len(removable),
                transformed=len(removable),
                removed=len(removable),
            ),
            (f"removed unreachable pure values: {preview}",),
        )
