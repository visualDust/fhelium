"""Plan the storage and CUDA Graph segments of a Runner schedule.

Planning rewrites a translated :class:`~fhelium.backend.runner.Schedule` before it is built into a Runner:

- *Storage classes.* A buffer allocated by ``alloc`` becomes an ``arena`` buffer when neither it nor a view of it is returned, carried through structured control flow, or returned by an operation that may alias its arguments. Arena buffers live at fixed byte offsets of one buffer per device that the Runner allocates once or shares with other Runners; their ``alloc`` and ``release`` instructions are dropped. Other allocations stay per-call allocations; those that reach the schedule results become ``output`` buffers. With Graph segments, an ``output`` buffer with one allocation and a static CUDA layout is placed in the Runner's output region with ``place_output``: its address stays fixed across executions, and the Runner moves a result that the caller still references to fresh storage before the next execution writes the buffer.
- *Graph segments* (``graphs=True``). Maximal runs of capture-safe nodes of the top-level block, where a loop or branch is one node, are enclosed in ``begin_segment``/``end_segment``; the Runner captures each segment once and replays it. Segments stay out of loop bodies, because a segment that reads a changing induction variable would be captured again on every iteration. The Runner stages a segment's other live-in Tensors into storage it owns.
- *Arena offsets.* Each arena buffer is live from its first to its last instruction; a buffer used inside a loop and first used before it stays live until the loop ends. Buffers are placed largest first at the lowest offset that does not overlap a buffer with an intersecting lifetime, and ``place`` instructions record the offsets.

Capture safety is a static property of an instruction. Kernel launches, copies, views, zeroing, schedule calls whose callee is safe, and operator calls on Tensors of one CUDA device are safe, unless the operator carries ``torch.Tag.cudagraph_unsafe`` or reads device values on the host. Collectives on CUDA buffers are safe; the Runner runs a segment directly when its process group cannot be captured. Python calls and CPU operations are unsafe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch

ARENA_ALIGNMENT = 256
"""Byte alignment of arena and output region offsets."""


def _aligned(size: int) -> int:
    return -(-size // ARENA_ALIGNMENT) * ARENA_ALIGNMENT


_HOST_READS = frozenset(
    {
        "aten::item",
        "aten::_local_scalar_dense",
        "aten::nonzero",
        "aten::nonzero_static",
        "aten::masked_select",
        "aten::unique",
        "aten::_unique",
        "aten::_unique2",
        "aten::unique_consecutive",
        "aten::unique_dim",
        "aten::equal",
        "aten::is_nonzero",
        "aten::allclose",
        "aten::_assert_async",
    }
)
"""ATen operators that read device values on the host."""

_BUFFER = 1
_BUFFER_LIST = 2
_MARKERS = frozenset(
    {"begin_for", "end_for", "begin_if", "begin_else", "end_if"}
)
_DEVICE_WORK = frozenset(
    {"call_op", "launch", "call", "copy", "zero", "collective"}
)
_ORDERING = frozenset(
    {
        "stream",
        "event",
        "use_stream",
        "record",
        "wait",
        "host_wait",
        "record_stream",
        "task",
        "begin_task",
        "end_task",
        "join",
    }
)
"""Stream, event and CPU worker lane instructions; Graph segments end at them."""


# Buffer access of one instruction --------------------------------------------------


@dataclass(frozen=True)
class _Access:
    defines: tuple[int, ...] = ()
    """Buffers that the instruction (re)binds to a new Tensor."""
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    """Buffers whose storage the instruction may write."""
    aliases: tuple[tuple[int, int], ...] = ()
    """(defined buffer, argument buffer) pairs whose storage may be shared."""


def _operator(schema: str) -> Any:
    from .lowering import resolve_operator

    return resolve_operator(schema)


def _buffer_arguments(
    kinds: Sequence[int], values: Sequence[object]
) -> Iterator[tuple[int, int]]:
    """Yield (argument position, buffer) of buffer and buffer-list arguments."""
    for position, (kind, value) in enumerate(zip(kinds, values, strict=True)):
        if kind == _BUFFER:
            yield position, cast(int, value)
        elif kind == _BUFFER_LIST:
            for buffer in cast(list, value):
                if buffer >= 0:
                    yield position, buffer


def _call_op_access(instruction: tuple[object, ...]) -> _Access:
    _, schema, kinds, values, results = instruction
    arguments = _operator(cast(str, schema))._schema.arguments
    returns = _operator(cast(str, schema))._schema.returns
    reads, writes = [], []
    by_alias: dict[str, list[int]] = {}
    for position, buffer in _buffer_arguments(
        cast(list, kinds), cast(list, values)
    ):
        info = arguments[position].alias_info
        (writes if info is not None and info.is_write else reads).append(buffer)
        if info is not None:
            for name in info.before_set:
                by_alias.setdefault(name, []).append(buffer)
    # Results list the returns that do not alias a written argument; a
    # returned Tensor list provides all remaining results.
    listed = [
        r for r in returns if not (r.alias_info and r.alias_info.is_write)
    ]
    aliases = []
    for index, result in enumerate(cast(list, results)):
        returned = listed[min(index, len(listed) - 1)] if listed else None
        info = returned.alias_info if returned is not None else None
        if info is not None:
            for name in info.before_set:
                aliases.extend(
                    (result, buffer) for buffer in by_alias.get(name, ())
                )
    return _Access(
        tuple(cast(list, results)), tuple(reads), tuple(writes), tuple(aliases)
    )


def _access(
    instruction: tuple[object, ...],
    passthrough: Callable[[object], bool],
    destinations: Callable[[object], int],
) -> _Access:
    name = instruction[0]
    if name == "call_op":
        return _call_op_access(instruction)
    if name == "launch":
        buffers = tuple(
            buffer
            for _, buffer in _buffer_arguments(
                cast(list, instruction[4]), cast(list, instruction[5])
            )
        )
        return _Access(reads=buffers, writes=buffers)
    if name == "view":
        target, source = cast(int, instruction[1]), cast(int, instruction[2])
        return _Access((target,), (source,), aliases=((target, source),))
    if name == "alloc":
        return _Access((cast(int, instruction[1]),))
    if name == "copy":
        return _Access(
            reads=(cast(int, instruction[2]),),
            writes=(cast(int, instruction[1]),),
        )
    if name == "zero":
        return _Access(writes=(cast(int, instruction[1]),))
    if name == "collective":
        return _Access(
            reads=tuple(cast(list, instruction[3])),
            writes=tuple(cast(list, instruction[4])),
        )
    if name == "assign":
        target, source = cast(int, instruction[1]), cast(int, instruction[2])
        return _Access((target,), (source,), aliases=((target, source),))
    if name in ("call", "python_call"):
        arguments = tuple(cast(list, instruction[2]))
        results = tuple(cast(list, instruction[5]))
        # A Python function may return its arguments; a callee may return
        # its inputs or views of them.
        shared = name == "python_call" or passthrough(instruction[1])
        aliases = (
            tuple(
                (result, buffer) for result in results for buffer in arguments
            )
            if shared
            else ()
        )
        # A called schedule writes its results into the trailing destination
        # arguments.
        written = destinations(instruction[1]) if name == "call" else 0
        return _Access(
            results,
            arguments,
            arguments[len(arguments) - written :] if written else (),
            aliases,
        )
    return _Access()


# Structure ---------------------------------------------------------------------


@dataclass
class _Node:
    """An instruction, or a structured operation with its nested nodes."""

    start: int
    end: int
    """Index of the last instruction of the node (its end_for/end_if)."""
    children: list[_Node] = field(default_factory=list)


def _nodes(
    instructions: Sequence[tuple[object, ...]], start: int = 0
) -> tuple[list[_Node], int]:
    """Parse one block from ``start``; return its nodes and the closing index."""
    nodes: list[_Node] = []
    index = start
    while index < len(instructions):
        name = instructions[index][0]
        if name in ("end_for", "end_if", "begin_else"):
            return nodes, index
        if name in ("begin_for", "begin_if"):
            node = _Node(index, index)
            inner, index = _nodes(instructions, index + 1)
            node.children.extend(inner)
            while instructions[index][0] == "begin_else":
                inner, index = _nodes(instructions, index + 1)
                node.children.extend(inner)
            node.end = index
            nodes.append(node)
        else:
            nodes.append(_Node(index, index))
        index += 1
    return nodes, index


def _loops(instructions: Sequence[tuple[object, ...]]) -> list[tuple[int, int]]:
    """Return the (begin_for, end_for) index pairs of every loop."""
    loops, stack = [], []
    for index, instruction in enumerate(instructions):
        if instruction[0] == "begin_for":
            stack.append(index)
        elif instruction[0] == "end_for":
            loops.append((stack.pop(), index))
    return loops


# Planning ----------------------------------------------------------------------


_ELEMENT_SIZES: dict[torch.dtype, int] = {}


def _extent(spec: tuple) -> int:
    _, sizes, strides, dtype, _ = spec
    if any(size == 0 for size in sizes):
        return 0
    elements = 1 + sum(
        (s - 1) * st for s, st in zip(sizes, strides, strict=True)
    )
    size = _ELEMENT_SIZES.get(dtype)
    if size is None:
        size = _ELEMENT_SIZES[dtype] = torch.empty(
            (), dtype=dtype, device="meta"
        ).element_size()
    return elements * size


def _static_layout(spec: tuple) -> bool:
    return all(item is not None for item in spec[1:])


class _Planner:
    def __init__(self, schedule: Any, graphs: bool) -> None:
        self.schedule = schedule
        self.graphs = graphs
        self.buffers = list(schedule.buffers)
        self.instructions = list(schedule.instructions)
        self._safe_callees: dict[int, bool] = {}
        self._passthrough: dict[int, bool] = {}
        self._accesses: dict[object, tuple[tuple[object, ...], _Access]] = {}
        self.side: set[int] = set()
        """Instructions that run on a copy or communication stream."""
        self.guards: dict[int, _Access] = {}
        """Buffers an ordering instruction keeps live: the transfers its event completes."""

    # Callees ---------------------------------------------------------------

    def _callee_schedule(self, reference: object) -> Any:
        from .runner import CalledSchedule

        callee = self.schedule.objects[cast(Any, reference).index]
        if not isinstance(callee, CalledSchedule):
            return None
        return callee.schedule

    def destinations(self, reference: object) -> int:
        """Number of trailing arguments a called schedule writes its results into."""
        from .runner import CalledSchedule

        callee = self.schedule.objects[cast(Any, reference).index]
        return (
            len(callee.destinations)
            if isinstance(callee, CalledSchedule)
            else 0
        )

    def passthrough(self, reference: object) -> bool:
        """Whether a called schedule may return its inputs or views of them."""
        index = cast(Any, reference).index
        if index not in self._passthrough:
            callee = self._callee_schedule(reference)
            if callee is None:
                self._passthrough[index] = True
            else:
                planner = _Planner(callee, False)
                groups = planner.groups()
                inputs = {
                    groups(buffer)
                    for kind, buffer in callee.inputs
                    if kind == "buffer"
                }
                self._passthrough[index] = any(
                    groups(buffer) in inputs for buffer in callee.output_buffers
                )
        return self._passthrough[index]

    def callee_safe(self, reference: object) -> bool:
        index = cast(Any, reference).index
        if index not in self._safe_callees:
            callee = self._callee_schedule(reference)
            if callee is None:
                self._safe_callees[index] = False
            else:
                planner = _Planner(callee, False)
                self._safe_callees[index] = all(
                    planner.safe(i) for i in range(len(callee.instructions))
                )
        return self._safe_callees[index]

    # Alias groups ------------------------------------------------------------

    def access(self, index: int) -> _Access:
        return self.instruction_access(self.instructions[index])

    def instruction_access(self, instruction: tuple[object, ...]) -> _Access:
        guarded = self.guards.get(id(instruction))
        if guarded is not None:
            return guarded
        # Instructions with equal contents have equal accesses; the overlap
        # planner emits many equal instructions while it searches.
        # Unhashable instructions are keyed by identity; the entry keeps the
        # instruction alive, so the id stays unique.
        key: object = instruction
        try:
            hash(instruction)
        except TypeError:
            key = id(instruction)
        found = self._accesses.get(key)
        if found is None:
            access = _access(instruction, self.passthrough, self.destinations)
            self._accesses[key] = (instruction, access)
            return access
        return found[1]

    def groups(self) -> Callable[[int], int]:
        """Return the alias-group representative function of the schedule's buffers."""
        parent = list(range(len(self.buffers)))

        def find(buffer: int) -> int:
            while parent[buffer] != buffer:
                parent[buffer] = parent[parent[buffer]]
                buffer = parent[buffer]
            return buffer

        for index in range(len(self.instructions)):
            for target, source in self.access(index).aliases:
                parent[find(target)] = find(source)
        return find

    def arena_roots(self) -> set[int]:
        """Return the allocated buffers that can live in the arena."""
        return self._allocation_roots(outputs=False)

    def output_roots(self) -> set[int]:
        """Return the allocated schedule results that can live in the Runner's output region."""
        return {
            buffer
            for buffer in self._allocation_roots(outputs=True)
            if self._device(buffer) is not None
        }

    def _allocation_roots(self, outputs: bool) -> set[int]:
        """Return buffers whose group has one allocation, a static layout and no other storage definition; ``outputs`` selects groups reaching the schedule results instead of excluding them."""
        find = self.groups()
        allocated: dict[int, list[int]] = {}
        excluded: set[int] = set()
        for index, instruction in enumerate(self.instructions):
            name = instruction[0]
            if name == "alloc":
                buffer = cast(int, instruction[1])
                allocated.setdefault(find(buffer), []).append(buffer)
            elif name == "assign":
                excluded.add(find(cast(int, instruction[1])))
            elif name not in ("view", "release"):
                for buffer in self.access(index).defines:
                    # A group whose storage another instruction defines, such
                    # as a call returning a view of its argument.
                    excluded.add(find(buffer))
        results = {find(buffer) for buffer in self.schedule.output_buffers}
        if not outputs:
            excluded |= results
        for kind, buffer in self.schedule.inputs:
            if kind == "buffer":
                excluded.add(find(buffer))
        return {
            buffers[0]
            for group, buffers in allocated.items()
            if len(buffers) == 1
            and group not in excluded
            and (group in results or not outputs)
            and _static_layout(self.buffers[buffers[0]])
            and self.buffers[buffers[0]][0] != "pinned"
        }

    # Capture safety ----------------------------------------------------------

    def _device(self, buffer: int) -> torch.device | None:
        device = self.buffers[buffer][4]
        return device if device is not None and device.type == "cuda" else None

    def safe(self, index: int) -> bool:
        instruction = self.instructions[index]
        name = instruction[0]
        if (
            name in _ORDERING
            or name == "python_call"
            or id(instruction) in self.side
        ):
            return False
        if name == "call" and not self.callee_safe(instruction[1]):
            return False
        if name == "call_op":
            schema = cast(str, instruction[1])
            if schema.split(".")[0] in _HOST_READS:
                return False
            if torch.Tag.cudagraph_unsafe in _operator(schema).tags:
                return False
        access = self.access(index)
        devices = {
            self._device(buffer)
            for buffer in (*access.defines, *access.reads, *access.writes)
        }
        if name == "launch":
            devices.add(torch.device("cuda", cast(int, instruction[2])))
        return None not in devices and len(devices) <= 1

    # Segments ---------------------------------------------------------------

    def _segment_device(self, instructions: list[tuple[object, ...]]) -> int:
        """Return the CUDA device index of a safe run, or -1 for the current device."""
        for instruction in instructions:
            if instruction[0] in _MARKERS:
                continue
            if instruction[0] == "launch":
                return cast(int, instruction[2])
            access = self.instruction_access(instruction)
            for buffer in (*access.defines, *access.reads, *access.writes):
                device = self._device(buffer)
                if device is not None and device.index is not None:
                    return device.index
        return -1

    def segment(self) -> list[tuple[object, ...]]:
        """Return the instructions with segment markers around maximal safe runs."""
        nodes, _ = _nodes(self.instructions)
        output: list[tuple[object, ...]] = []
        current: list[tuple[object, ...]] = []
        work = 0

        def close() -> None:
            nonlocal current, work
            if work:
                output.append(("begin_segment", self._segment_device(current)))
                output.extend(current)
                output.append(("end_segment",))
            else:
                # A run without device work is not worth capturing.
                output.extend(current)
            current, work = [], 0

        for node in nodes:
            indices = range(node.start, node.end + 1)
            if not all(
                self.safe(i)
                for i in indices
                if self.instructions[i][0] not in _MARKERS
            ):
                close()
                output.extend(self.instructions[i] for i in indices)
                continue
            for i in indices:
                current.append(self.instructions[i])
                if self.instructions[i][0] in _DEVICE_WORK:
                    work += 1
        close()
        return output

    # Offsets ---------------------------------------------------------------

    def place(
        self, instructions: list[tuple[object, ...]], arena: set[int]
    ) -> tuple[list[tuple[object, ...]], dict[int, int]]:
        """Drop arena alloc/release instructions and assign arena offsets."""
        kept = [
            instruction
            for instruction in instructions
            if not (
                instruction[0] in ("alloc", "release")
                and instruction[1] in arena
            )
        ]
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        find = self.groups()
        by_group = {find(buffer): buffer for buffer in arena}
        members = {
            buffer: by_group[find(buffer)]
            for buffer in range(len(self.buffers))
            if find(buffer) in by_group
        }
        for index, instruction in enumerate(kept):
            access = self.instruction_access(instruction)
            for buffer in (*access.defines, *access.reads, *access.writes):
                root = members.get(buffer)
                if root is not None:
                    first.setdefault(root, index)
                    last[root] = index
        loops = _loops(kept)
        for root in first:
            changed = True
            while changed:
                changed = False
                for begin, end in loops:
                    if first[root] < begin <= last[root] < end:
                        last[root] = end
                        changed = True
        placed: dict[torch.device, list[tuple[int, int, int, int]]] = {}
        offsets: dict[int, int] = {}
        sizes = {buffer: _extent(self.buffers[buffer]) for buffer in first}
        order = sorted(
            first, key=lambda buffer: (-sizes[buffer], first[buffer])
        )
        for buffer in order:
            size = sizes[buffer]
            device = cast(torch.device, self.buffers[buffer][4])
            busy = sorted(
                (offset, end)
                for offset, end, start, stop in placed.get(device, ())
                if start <= last[buffer] and first[buffer] <= stop
            )
            offset = 0
            for begin, end in busy:
                if offset + size <= begin:
                    break
                offset = max(offset, _aligned(end))
            offsets[buffer] = offset
            placed.setdefault(device, []).append(
                (offset, offset + size, first[buffer], last[buffer])
            )
        # Arena buffers never touched by an instruction need no storage.
        for buffer in arena - set(offsets):
            offsets[buffer] = 0
        return kept, offsets


@dataclass(frozen=True)
class PlanReport:
    """Arena bytes per device, arena buffer count, Graph segment count and output region bytes of a planned schedule."""

    arena_bytes: tuple[tuple[str, int], ...]
    arena_buffers: int
    segments: int
    output_bytes: int = 0
    overlap: Any = None
    """The :class:`~fhelium.backend._overlap.OverlapReport` of the schedule."""


def plan_schedule(
    schedule: Any, *, graphs: bool = False
) -> tuple[Any, PlanReport]:
    """Return ``schedule`` with overlapped transfers, arena storage, offsets and, with ``graphs``, Graph segments.

    Transfers between page-locked host buffers and CUDA buffers run on copy streams and runs of CPU instructions on the CPU worker lane (:mod:`fhelium.backend._overlap`); Graph segments end at their stream and task instructions.
    """
    from ._overlap import overlap as plan_overlap

    planner = _Planner(schedule, graphs)
    arena = planner.arena_roots()
    planner.instructions, overlapped = plan_overlap(planner, arena)
    instructions = planner.segment() if graphs else list(planner.instructions)
    instructions, offsets = planner.place(instructions, arena)
    # With Graph segments, results live at fixed addresses in the Runner's
    # output region, which it detaches from a result the caller still holds
    # before overwriting it; otherwise each execution allocates its results.
    fixed = planner.output_roots() if graphs else set()
    instructions = [
        instruction
        for instruction in instructions
        if not (
            instruction[0] in ("alloc", "release") and instruction[1] in fixed
        )
    ]
    output_offsets: dict[int, int] = {}
    output_bytes = 0
    for buffer in sorted(fixed):
        output_offsets[buffer] = output_bytes
        output_bytes += _aligned(_extent(planner.buffers[buffer]))
    buffers = list(planner.buffers)
    outputs = set(schedule.output_buffers)
    find = planner.groups()
    output_groups = {find(buffer) for buffer in outputs}
    for index, instruction in enumerate(instructions):
        if instruction[0] == "alloc":
            buffer = cast(int, instruction[1])
            if (
                buffer < len(schedule.buffers)
                and find(buffer) in output_groups
                and buffers[buffer][0] != "pinned"
            ):
                buffers[buffer] = ("output", *buffers[buffer][1:])
    for buffer in offsets:
        buffers[buffer] = ("arena", *buffers[buffer][1:])
    for buffer in fixed:
        buffers[buffer] = ("output", *buffers[buffer][1:])
    extent: dict[str, int] = {}
    for buffer, offset in offsets.items():
        device = str(buffers[buffer][4])
        extent[device] = max(
            extent.get(device, 0), offset + _extent(buffers[buffer])
        )
    placements = [
        ("place", buffer, offset) for buffer, offset in sorted(offsets.items())
    ] + [
        ("place_output", buffer, offset)
        for buffer, offset in sorted(output_offsets.items())
    ]
    planned = replace(
        schedule,
        buffers=tuple(buffers),
        instructions=(*placements, *instructions),
        graphs=graphs,
    )
    report = PlanReport(
        tuple(sorted(extent.items())),
        len(offsets),
        sum(1 for i in instructions if i[0] == "begin_segment"),
        output_bytes,
        overlapped,
    )
    return planned, report


def destination_results(schedule: Any) -> tuple[int, ...]:
    """Return the positions of results that a caller can provide the storage of.

    Such a result is the buffer of its schedule's only allocation in its storage group, with a static CUDA layout, no other result in the group and no input sharing it.
    """
    planner = _Planner(schedule, False)
    roots = planner.output_roots()
    find = planner.groups()
    groups = [find(buffer) for buffer in schedule.output_buffers]
    return tuple(
        index
        for index, (kind, item) in enumerate(schedule.outputs)
        if kind == "buffer"
        and schedule.output_buffers[cast(int, item)] in roots
        and groups.count(groups[cast(int, item)]) == 1
        and sum(1 for k, i in schedule.outputs if (k, i) == (kind, item)) == 1
    )


def with_destinations(schedule: Any, positions: Sequence[int]) -> Any:
    """Return ``schedule`` taking the storage of the results at ``positions`` as trailing Tensor inputs.

    The schedule writes those results into the given Tensors instead of allocating them and returns the remaining results.
    """
    written = [
        schedule.output_buffers[cast(int, schedule.outputs[index][1])]
        for index in positions
    ]
    kept = [
        position
        for position, buffer in enumerate(schedule.output_buffers)
        if buffer not in written
    ]
    renumbered = {position: index for index, position in enumerate(kept)}
    return replace(
        schedule,
        instructions=tuple(
            instruction
            for instruction in schedule.instructions
            if not (
                instruction[0] in ("alloc", "release")
                and instruction[1] in written
            )
        ),
        inputs=(*schedule.inputs, *(("buffer", buffer) for buffer in written)),
        output_buffers=tuple(schedule.output_buffers[p] for p in kept),
        outputs=tuple(
            (kind, renumbered[cast(int, item)] if kind == "buffer" else item)
            for index, (kind, item) in enumerate(schedule.outputs)
            if index not in positions
        ),
    )


__all__ = [
    "ARENA_ALIGNMENT",
    "PlanReport",
    "destination_results",
    "plan_schedule",
    "with_destinations",
]
