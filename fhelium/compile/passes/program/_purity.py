"""Classify effects using registered operation semantics, not trait hints."""

from __future__ import annotations

from xdsl.dialects import scf
from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.ir import Block, Operation
from xdsl.traits import IsTerminator

from fhelium.ir import DEFAULT_OPERATION_SPECS
from fhelium.ir.dialects import distributed, fusion


def known_region_blocks(operation: Operation) -> tuple[Block, ...]:
    """Enter regions with known execution semantics; leave extensions opaque."""

    if type(operation) not in (
        scf.ForOp,
        scf.IfOp,
        distributed.AllReduceOp,
        fusion.FusedOp,
    ):
        return ()
    return tuple(
        block for region in operation.regions for block in region.blocks
    )


def known_pure(operation: Operation) -> bool:
    """Recognize exact registered pure classes, including pure fusion bodies.

    A trait alone is insufficient: some operations consume randomness despite
    carrying a structural Pure hint. Unknown classes, effects, properties, and
    control flow are retained. Terminators are liveness roots even when their
    operation registration describes no external side effect.
    """

    if (
        operation.has_trait(IsTerminator)
        or set(operation.properties)
        - {"operandSegmentSizes", "resultSegmentSizes"}
        or operation.successors
    ):
        return False
    if type(operation) is UnrealizedConversionCastOp:
        return not operation.regions
    spec = DEFAULT_OPERATION_SPECS.get(operation.name)
    if (
        spec is None
        or spec.operation_type is not type(operation)
        or spec.effect != "pure"
    ):
        return False
    if operation.regions:
        if type(operation) is not fusion.FusedOp:
            return False
        return all(
            type(nested) is fusion.YieldOp or known_pure(nested)
            for nested in operation.body.block.ops
        )
    return True
