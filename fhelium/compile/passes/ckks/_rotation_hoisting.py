"""Share key-switch preparation across data-dependent rotation groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, cast

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Block, Operation
from xdsl.rewriter import Rewriter

from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import ckks, core

from ..._pipeline import DecisionRecord, PassResult, PassStats
from ..program._purity import known_pure, known_region_blocks

_REFERENCES = (core.MaterialRefOp, core.ResourceRefOp)
_Group = tuple[ckks.RotateOp, ...]


def _block_user(operation: Operation, block: Block) -> Operation | None:
    """Find the enclosing use in this block, including nested region uses."""
    while operation.parent_block() is not block:
        parent = operation.parent_op()
        if parent is None:
            return None
        operation = parent
    return operation


def _placement(
    group: _Group, positions: dict[Operation, int], start: int, end: int
) -> tuple[int, tuple[Operation, ...]] | None:
    """Find a common definition before all uses without moving computation.

    Only operand-free references in the same effect interval may move. Other
    operands must already be available at the insertion point. Results used in
    nested regions constrain placement at the enclosing operation.
    """
    block = group[0].parent_block()
    assert block is not None
    lower = max(start, positions[group[0]])
    upper = end
    references: dict[Operation, None] = {}
    for rotation in group:
        for operand in rotation.operands:
            owner = operand.owner
            if not isinstance(owner, Operation) or owner not in positions:
                continue
            index = positions[owner]
            if type(owner) in _REFERENCES and start <= index < end:
                references[owner] = None
            else:
                lower = max(lower, index + 1)
        for use in rotation.result.uses:
            user = _block_user(use.operation, block)
            if user is not None:
                upper = min(upper, positions[user])
    if lower > upper:
        return None
    return lower, tuple(ref for ref in references if positions[ref] >= lower)


@dataclass(frozen=True)
class RotationHoistingPass:
    r"""Share preparation among rotations of the same ciphertext data.

    A rotation has the form ``Finish(c0, Prepare(c1, tables), key, step)``.
    Rotations with the same input SSA value, parameter operands, and compatible
    attributes can share ``Prepare`` despite interleaved consumers or rotations
    of other inputs. The emitted ``RotateManyOp`` retains each key operand,
    result type, and rotation step. Its common output domain and execution
    attributes must agree across the group.

    Matching and placement stay inside a block's known-pure effect interval.
    Unknown operations, effects, control flow, and assigned implementations end
    that interval; known nested regions are processed independently. Every group
    is placed after its computed operands and before its
    first result use. Material references may move within the interval, but
    producer computations are not speculated or reordered.

    ``max_group_size`` bounds the number of results produced together. Grouping
    can extend result lifetimes.
    """

    max_group_size: int | None = None
    name: str = "rotation-hoisting"

    def __post_init__(self) -> None:
        if self.max_group_size is not None:
            if type(self.max_group_size) is not int:
                raise TypeError("max_group_size must be an integer or None")
            if self.max_group_size < 2:
                raise ValueError("max_group_size must be at least 2")

    @staticmethod
    def _step(rotation: ckks.RotateOp) -> int | None:
        step = cast(ckks.EvaluationKeyType, rotation.key.type).state.data.get(
            "rotation_step"
        )
        return int(step.value.data) if isinstance(step, IntegerAttr) else None

    @staticmethod
    def _signature(rotation: ckks.RotateOp) -> tuple[object, ...]:
        return (
            rotation.value,
            tuple(rotation.parameters),
            tuple(
                sorted(
                    (name, value)
                    for name, value in rotation.attributes.items()
                    if name != "rotation_step"
                )
            ),
        )

    @staticmethod
    def _rewrite(
        group: _Group,
        block: Block,
        position: int,
        references: tuple[Operation, ...],
    ) -> DecisionRecord:
        anchor = tuple(block.ops)[position]
        for reference in references:
            if reference is anchor:
                anchor = reference.next_op
                assert anchor is not None
            block.detach_op(reference)
            block.insert_op_before(reference, anchor)
        first = group[0]
        steps = tuple(
            cast(int, RotationHoistingPass._step(rotation))
            for rotation in group
        )
        hoisted = ckks.RotateManyOp(
            first.value,
            tuple(rotation.key for rotation in group),
            tuple(rotation.result.type for rotation in group),
            output_domain=cast(
                Literal["coefficient", "ntt"], first.output_domain.data
            ),
            parameters=tuple(first.parameters),
            attributes={
                **{
                    name: value
                    for name, value in first.attributes.items()
                    if name not in {"rotation_step", "input_domain"}
                },
                "rotation_steps": ArrayAttr(
                    IntegerAttr(step, 64) for step in steps
                ),
            },
        )
        block.insert_op_before(hoisted, anchor)
        for rotation, result in zip(group, hoisted.results, strict=True):
            result.name_hint = rotation.result.name_hint
            rotation.result.replace_all_uses_with(result)
        for rotation in group:
            Rewriter.erase_op(rotation)
        return DecisionRecord(
            subject="rotation-hoisting",
            selected="hoisted",
            candidates=("independent", "hoisted"),
            details=(
                f"offsets={steps}",
                "shared input and parameter Tensor operands",
            ),
        )

    def run(self, compilation: "Compilation") -> PassResult:
        """Identify compatible rotations and place legal multi-result groups."""
        program = compilation.program
        workspace = compilation.workspace
        del workspace
        decisions: list[DecisionRecord] = []
        diagnostics: list[str] = []
        removed = skipped = 0

        def interval(block: Block, operations: tuple[Operation, ...]) -> None:
            nonlocal removed, skipped
            if not operations:
                return
            candidates: dict[tuple[object, ...], list[ckks.RotateOp]] = (
                defaultdict(list)
            )
            for operation in operations:
                if type(operation) is not ckks.RotateOp:
                    continue
                state = cast(
                    ckks.CiphertextType, operation.value.type
                ).state.data
                if any(
                    state.get(field) != StringAttr(required)
                    for field, required in (
                        ("polynomial_domain", "coefficient"),
                        ("residue_representation", "standard"),
                        ("basis", "Q"),
                    )
                ):
                    skipped += 1
                    diagnostics.append(
                        "rotation hoisting requires coefficient/standard Q input"
                    )
                    continue
                step = self._step(operation)
                if step is None:
                    skipped += 1
                    diagnostics.append(
                        "rotation hoisting requires a represented rotation-key step"
                    )
                    continue
                if step != 0:
                    candidates[self._signature(operation)].append(operation)
            # Keep stable interval bounds as rewrites change operation positions.
            before = operations[0].prev_op
            after = operations[-1].next_op
            for rotations in candidates.values():
                positions = {op: i for i, op in enumerate(block.ops)}
                start = positions[before] + 1 if before is not None else 0
                end = positions[after] if after is not None else len(positions)
                groups: list[list[ckks.RotateOp]] = []
                for rotation in rotations:
                    for group in groups:
                        if (
                            self.max_group_size is not None
                            and len(group) >= self.max_group_size
                        ):
                            continue
                        if (
                            _placement(
                                (*group, rotation), positions, start, end
                            )
                            is not None
                        ):
                            group.append(rotation)
                            break
                    else:
                        # An unavailable key need not discard an earlier group:
                        # a later independent rotation may still join that group.
                        groups.append([rotation])
                if len(groups) > 1 and self.max_group_size is None:
                    diagnostics.append(
                        "rotation candidates require separate operand/use placement intervals"
                    )
                for group in groups:
                    if len(group) < 2:
                        continue
                    positions = {op: i for i, op in enumerate(block.ops)}
                    start = positions[before] + 1 if before is not None else 0
                    end = (
                        positions[after]
                        if after is not None
                        else len(positions)
                    )
                    placement = _placement(tuple(group), positions, start, end)
                    if placement is None:
                        diagnostics.append(
                            "rotation group retained: earlier scheduling changed operand availability"
                        )
                        continue
                    decisions.append(
                        self._rewrite(tuple(group), block, *placement)
                    )
                    removed += len(group)

        def visit(block: Block) -> None:
            nonlocal skipped
            pending: list[Operation] = []
            for operation in tuple(block.ops):
                if (
                    EXECUTION_IMPLEMENTATION_ATTRIBUTE
                    not in operation.attributes
                ):
                    for child in known_region_blocks(operation):
                        visit(child)
                if (
                    not known_pure(operation)
                    or EXECUTION_IMPLEMENTATION_ATTRIBUTE
                    in operation.attributes
                ):
                    interval(block, tuple(pending))
                    pending.clear()
                    if (
                        type(operation) is ckks.RotateOp
                        and EXECUTION_IMPLEMENTATION_ATTRIBUTE
                        in operation.attributes
                    ):
                        skipped += 1
                        diagnostics.append(
                            "rotation has a caller-assigned implementation"
                        )
                else:
                    pending.append(operation)
            interval(block, tuple(pending))

        for function in program.functions:
            for block in function.body.blocks:
                visit(block)
        count = len(decisions)
        return PassResult(
            program,
            PassStats(
                matched=count + skipped,
                transformed=count,
                inserted=count,
                removed=removed,
                skipped=skipped,
            ),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            decisions=tuple(decisions),
        )


__all__ = ["RotationHoistingPass"]
