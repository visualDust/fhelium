"""Group compatible pure elementwise operations into visible fusion regions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass
from collections.abc import Sequence

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation, SSAValue, Block

from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import core, fusion
from fhelium.backend.implementation import FusionImplementation

from .._pipeline import DecisionRecord, PassResult, PassStats
from .program._purity import known_region_blocks


def _fuse(
    block: Block, operations: list[Operation], implementation: str
) -> fusion.FusedOp | None:
    selected = set(operations)
    body_ops = [
        op
        for op in operations
        if not isinstance(op, (core.ResourceRefOp, core.MaterialRefOp))
    ]
    results = [
        value
        for op in body_ops
        for value in op.results
        if any(use.operation not in selected for use in value.uses)
    ]
    if not results:
        return None

    inputs: list[SSAValue] = []
    canonical: dict[SSAValue, SSAValue] = {}
    resource_roles: dict[tuple[object, object], SSAValue] = {}
    for op in body_ops:
        for operand in op.operands:
            owner = operand.owner
            if owner in selected and not isinstance(
                owner, (core.ResourceRefOp, core.MaterialRefOp)
            ):
                continue
            value = operand
            if isinstance(owner, (core.ResourceRefOp, core.MaterialRefOp)):
                role = (owner.symbol, owner.kind)
                value = resource_roles.setdefault(role, operand)
            canonical[operand] = value
            if value not in inputs:
                inputs.append(value)

    # Resource references have no execution effect. Move the retained references
    # before the region so every explicit outer operand dominates its use.
    first = operations[0]
    resource_ops = [
        op
        for op in operations
        if isinstance(op, (core.ResourceRefOp, core.MaterialRefOp))
    ]
    anchor = first
    for op in resource_ops:
        if op is anchor:
            continue
        block.detach_op(op)
        block.insert_op_before(op, anchor)
    body = Block(arg_types=[value.type for value in inputs])
    mapper: dict[SSAValue, SSAValue] = dict(zip(inputs, body.args, strict=True))
    for alias, value in canonical.items():
        mapper[alias] = mapper[value]
    for op in body_ops:
        body.add_op(op.clone(value_mapper=mapper))
    body.add_op(fusion.YieldOp([mapper[value] for value in results]))
    fused = fusion.FusedOp(
        inputs,
        [value.type for value in results],
        body,
        attributes={
            EXECUTION_IMPLEMENTATION_ATTRIBUTE: StringAttr(implementation)
        },
    )
    # The original first operation may itself be a resource; all resources must
    # precede the fused operation, so insert at the first computational operation.
    block.insert_op_before(fused, body_ops[0])
    for old, new in zip(results, fused.outputs, strict=True):
        old.replace_all_uses_with(new)
    for op in reversed(body_ops):
        block.erase_op(op)
    return fused


def _fusion_segments(
    operations: Sequence[Operation],
    implementations: Sequence[FusionImplementation],
) -> list[tuple[list[Operation], FusionImplementation, int]]:
    """Find supported consecutive regions without changing operations or uses."""
    pending: list[Operation] = []
    segments: list[tuple[list[Operation], FusionImplementation, int]] = []

    def match(
        operations: list[Operation],
    ) -> tuple[FusionImplementation, int] | None:
        for implementation in implementations:
            count = implementation.match_fusion(operations)
            if count is not None:
                return implementation, count
        return None

    current: tuple[FusionImplementation, int] | None = None
    for operation in operations:
        assigned = EXECUTION_IMPLEMENTATION_ATTRIBUTE in operation.attributes
        candidate = None if assigned else match([*pending, operation])
        if candidate is not None:
            pending.append(operation)
            current = candidate
            continue
        if pending and current is not None:
            segments.append((pending, *current))
        pending = []
        current = None
        if not assigned:
            candidate = match([operation])
            if candidate is not None:
                pending = [operation]
                current = candidate
    if pending and current is not None:
        segments.append((pending, *current))
    return segments


@dataclass(frozen=True)
class FuseOperationsPass:
    """Select contiguous SSA regions using caller-supplied Backend support.

    Implementations own operation, resource and layout admissibility. The pass
    grows a window while at least one supplied implementation accepts it, then
    selects the first matching implementation in caller order. An incompatible
    operation ends the current window without
    discarding compatible operations on either side. Recorded implementation
    assignments remain barriers, including assignments on plumbing operations.
    """

    implementations: Sequence[FusionImplementation]
    min_ops: int = 2
    name: str = "fuse-operations"

    def __post_init__(self) -> None:
        object.__setattr__(self, "implementations", tuple(self.implementations))
        if self.min_ops < 1:
            raise ValueError("Fusion min_ops must be positive")

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        del shared_data
        blocks = []

        def visit(block):
            blocks.append(block)
            for operation in tuple(block.ops):
                if isinstance(operation, fusion.FusedOp):
                    continue
                for nested in known_region_blocks(operation):
                    visit(nested)

        visit(program.single_block("main"))
        matched = transformed = removed = 0
        decisions: list[DecisionRecord] = []
        for block in blocks:
            segments = _fusion_segments(tuple(block.ops), self.implementations)

            for operations, implementation, count in segments:
                if count < self.min_ops:
                    continue
                matched += 1
                fused = _fuse(block, operations, implementation.name)
                if fused is None:
                    continue
                transformed += 1
                removed += count
                decisions.append(
                    DecisionRecord(
                        "fusion-region",
                        implementation.name,
                        (implementation.name,),
                        (
                            f"operations={count}",
                            f"outputs={len(fused.outputs)}",
                        ),
                    )
                )
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=transformed,
                inserted=transformed,
                removed=removed,
            ),
            decisions=tuple(decisions),
        )


__all__ = ["FuseOperationsPass"]
