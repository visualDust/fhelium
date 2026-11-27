"""Share key-switch preparation across data-dependent rotation groups."""

from __future__ import annotations

from fhelium.backend.implementation import OperationImplementationRegistry
from ..._material_requirements import resolve_material_requirements
from .._operation_transforms import program_operations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


import math
from collections import defaultdict

import torch
from dataclasses import dataclass
from typing import Literal, cast

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Block, Operation, SSAValue
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
    group: _Group,
    positions: dict[Operation, int],
    start: int,
    end: int,
    late: bool = False,
) -> tuple[int, tuple[Operation, ...]] | None:
    """Find a common definition before all uses without moving computation.

    Only operand-free references in the same effect interval may move. Other
    operands must already be available at the insertion point. Results used in
    nested regions constrain placement at the enclosing operation. ``late``
    places the group before its first use instead of after its operands.
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
    position = upper if late else lower
    return position, tuple(
        ref for ref in references if positions[ref] >= position
    )


def _integers(attribute: object) -> tuple[int, ...] | None:
    if not isinstance(attribute, ArrayAttr) or not all(
        isinstance(item, IntegerAttr) for item in attribute
    ):
        return None
    return tuple(cast(IntegerAttr, item).value.data for item in attribute)


def _row_bytes(value: SSAValue) -> int:
    """Bytes of one residue row of every batched polynomial of a ciphertext, else 0."""
    state = getattr(getattr(value.type, "state", None), "data", {})
    batch, ring = (
        _integers(state.get("batch_shape")),
        state.get("ring_dimension"),
    )
    dtype = state.get("dtype")
    if (
        batch is None
        or not isinstance(ring, IntegerAttr)
        or not isinstance(dtype, StringAttr)
    ):
        return 0
    element = getattr(torch, dtype.data.removeprefix("torch."), None)
    if not isinstance(element, torch.dtype):
        return 0
    return math.prod(batch) * ring.value.data * element.itemsize


def _bytes(value: SSAValue) -> int:
    """Bytes of a ciphertext from its components, batch shape, primes and ring dimension, else 0."""
    state = getattr(getattr(value.type, "state", None), "data", {})
    components, primes = (
        state.get("components"),
        _integers(state.get("prime_ids")),
    )
    if not isinstance(components, IntegerAttr) or primes is None:
        return 0
    return components.value.data * len(primes) * _row_bytes(value)


def _device_bytes(group: _Group) -> int | None:
    """Free memory of the CUDA device holding the group's input, including memory cached by the allocator, else None."""
    state = cast(ckks.CiphertextType, group[0].value.type).state.data
    device = state.get("device")
    if not isinstance(device, StringAttr) or not device.data.startswith("cuda"):
        return None
    if not torch.cuda.is_available():
        return None
    index = torch.device(device.data)
    free, _ = torch.cuda.mem_get_info(index)
    cached = torch.cuda.memory_reserved(index) - torch.cuda.memory_allocated(
        index
    )
    # Kernel launches, the output region and buffers outside the arena need
    # the rest.
    return (free + cached) * 9 // 10


def _live_bytes(block: Block) -> list[int]:
    """Return the bytes of the ciphertexts the block computes that are live at each operation.

    Arguments and materials are already allocated when the program runs; a
    value used in a nested region is live until the enclosing operation.
    """
    ops = tuple(block.ops)
    positions = {op: i for i, op in enumerate(ops)}
    change = [0] * (len(ops) + 1)

    def add(value: SSAValue, first: int, last: int) -> None:
        size = _bytes(value)
        if size:
            change[first] += size
            change[last + 1] -= size

    for index, op in enumerate(ops):
        if type(op) in _REFERENCES:
            continue
        last = index
        for result in op.results:
            for use in result.uses:
                user = _block_user(use.operation, block)
                if user is not None:
                    last = max(last, positions[user])
        for result in op.results:
            add(result, index, last)
    live, total = [], 0
    for delta in change[:-1]:
        total += delta
        live.append(total)
    return live


def _hoisted_bytes(
    chunks: list[tuple[_Group, int]],
    live: list[int],
    positions: dict[Operation, int],
) -> int:
    """Peak live bytes when each chunk produces its results at its position.

    Each result is live from the chunk position to its last use. At that
    position the chunk's workspace is live as well: the input's ModUp to the
    key basis for every key digit and, with several digits, one pair of
    key-basis accumulators per rotation (one pair with a single digit).
    """
    change = [0] * (len(live) + 1)
    for chunk, position in chunks:
        first = chunk[0]
        key = cast(ckks.EvaluationKeyType, first.key.type).state.data
        rows, digits = key.get("prime_ids"), key.get("digit_prime_ids")
        if isinstance(rows, ArrayAttr) and isinstance(digits, ArrayAttr):
            pairs = len(chunk) if len(digits) > 1 else 1
            workspace = (len(digits) + 2 * pairs) * len(rows)
            change[position] += workspace * _row_bytes(first.value)
            change[position + 1] -= workspace * _row_bytes(first.value)
        for rotation in chunk:
            if positions[rotation] > position:
                change[position] += _bytes(rotation.result)
                change[positions[rotation]] -= _bytes(rotation.result)
    peak = total = 0
    for held, delta in zip(live, change):
        total += delta
        peak = max(peak, held + total)
    return peak


_MARGINAL_SAVING = 0.05


def _chunk_size(group: _Group) -> int:
    r"""Chunk size at which doubling saves less than ``_MARGINAL_SAVING`` per rotation.

    Traffic is counted in ring rows of the input. A chunk shares one
    preparation: reading the input's $L$ rows, ModUp of every key digit to the
    $K$ key-basis rows and their forward NTT (two passes), $F = L + 5DK$ for $D$
    digits. Each rotation reads its key ($2DK$) and the shared digits ($DK$),
    writes and ModDowns a key-basis accumulator pair ($4K$) and finishes its
    $L$-row result ($6L$), $r = 3DK + 4K + 6L$. Doubling the chunk size $k$
    saves $F/(2k)$ of the per-rotation cost $r + F/k$; the chunk grows while
    that share is at least ``_MARGINAL_SAVING``.
    """
    first = group[0]
    key = cast(ckks.EvaluationKeyType, first.key.type).state.data
    rows, digits = key.get("prime_ids"), key.get("digit_prime_ids")
    primes = _integers(
        getattr(getattr(first.value.type, "state", None), "data", {}).get(
            "prime_ids"
        )
    )
    if not (
        isinstance(rows, ArrayAttr)
        and isinstance(digits, ArrayAttr)
        and primes is not None
    ):
        return len(group)
    d, k, l = len(digits), len(rows), len(primes)
    fixed = l + 5 * d * k
    rotation = 3 * d * k + 4 * k + 6 * l
    size = 1
    while fixed / (2 * (size * rotation + fixed)) >= _MARGINAL_SAVING:
        size *= 2
    return size


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

    ``max_group_size`` bounds the number of results produced together. The
    groups of an interval are split into chunks of one common size, each placed
    before its first use: the size grows while doubling it would still save a
    meaningful share of per-rotation traffic (:func:`_chunk_size`) and shrinks
    while the estimated live bytes of the block with every chunk hoisted exceed
    the free memory of their CUDA device.
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
        return int(rotation.rotation_step.value.data)

    @staticmethod
    def _key_layout(rotation: ckks.RotateOp) -> tuple[object, ...]:
        state = cast(ckks.EvaluationKeyType, rotation.key.type).state.data
        rows, digits = state.get("prime_ids"), state.get("digit_prime_ids")
        if rows is None or digits is None:
            return (rotation.key,)
        return (rows, digits, state.get("seed_rounds"))

    @staticmethod
    def _signature(rotation: ckks.RotateOp) -> tuple[object, ...]:
        return (
            rotation.value,
            RotationHoistingPass._key_layout(rotation),
            rotation.moddown_rounding,
            rotation.basis_conversion,
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
        compilation: Compilation,
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
            key_seeds=tuple(
                seed for rotation in group for seed in rotation.key_seeds
            ),
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
                "moddown_rounding": first.moddown_rounding,
                "basis_conversion": first.basis_conversion,
                "rotation_steps": ArrayAttr(
                    IntegerAttr(step, 64) for step in steps
                ),
            },
        )
        block.insert_op_before(hoisted, anchor)
        if OperationImplementationRegistry in compilation.workspace:
            resolve_material_requirements(compilation, (hoisted,))
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

    @staticmethod
    def _chunks(
        groups: list[_Group],
        block: Block,
        positions: dict[Operation, int],
        start: int,
        end: int,
    ) -> list[tuple[_Group, bool]]:
        """Split the interval's groups into chunks of one common size.

        The size is the smallest at which doubling it would save less than
        ``_MARGINAL_SAVING`` of the per-rotation traffic (:func:`_chunk_size`),
        reduced while the hoisted results of all chunks, each placed before its
        first use, exceed the free memory of their CUDA device. Groups are not
        hoisted when no chunk of two rotations fits.
        """
        if not groups:
            return []
        longest = max(len(group) for group in groups)
        size = min(longest, _chunk_size(groups[0]))
        limit = _device_bytes(groups[0])
        if limit is not None:
            live = _live_bytes(block)

            def fits(size: int) -> bool:
                chunks = []
                for group in groups:
                    for index in range(0, len(group), size):
                        chunk = group[index : index + size]
                        if len(chunk) < 2:
                            continue
                        placement = _placement(
                            chunk, positions, start, end, True
                        )
                        if placement is None:
                            return False
                        chunks.append((chunk, placement[0]))
                return _hoisted_bytes(chunks, live, positions) <= limit

            low, high = 1, size
            while low < high:
                middle = (low + high + 1) // 2
                if fits(middle):
                    low = middle
                else:
                    high = middle - 1
            size = low
        return [
            (group[index : index + size], True)
            for group in groups
            for index in range(0, len(group), size)
            if size >= 2 and len(group[index : index + size]) >= 2
        ]

    def run(self, compilation: "Compilation") -> PassResult:
        """Identify compatible rotations and place legal multi-result groups."""
        program = compilation.program
        workspace = compilation.workspace
        requirement_stats = PassStats()
        if OperationImplementationRegistry in workspace:
            rotations = tuple(
                op
                for op in program_operations(program)
                if type(op) is ckks.RotateOp
            )
            requirement_stats = resolve_material_requirements(
                compilation, rotations
            ).stats
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
            positions = {op: i for i, op in enumerate(block.ops)}
            start = positions[before] + 1 if before is not None else 0
            end = positions[after] if after is not None else len(positions)
            formed: list[_Group] = []
            for rotations in candidates.values():
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
                formed.extend(
                    tuple(group) for group in groups if len(group) > 1
                )
            for chunk, late in self._chunks(
                formed, block, positions, start, end
            ):
                positions = {op: i for i, op in enumerate(block.ops)}
                start = positions[before] + 1 if before is not None else 0
                end = positions[after] if after is not None else len(positions)
                placement = _placement(chunk, positions, start, end, late)
                if placement is None:
                    diagnostics.append(
                        "rotation group retained: earlier scheduling changed operand availability"
                    )
                    continue
                decisions.append(
                    self._rewrite(chunk, block, *placement, compilation)
                )
                removed += len(chunk)

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
                matched=count + skipped + requirement_stats.matched,
                transformed=count + requirement_stats.transformed,
                inserted=count + requirement_stats.inserted,
                removed=removed,
                skipped=skipped + requirement_stats.skipped,
            ),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            decisions=tuple(decisions),
        )


__all__ = ["RotationHoistingPass"]
