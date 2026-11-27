"""Schedule host/device transfers on copy streams and CPU operations on the CPU worker lane.

Overlap planning rewrites the top-level instructions of a schedule without adding or removing a transfer or collective or changing the device of any operation. Each instruction keeps its operands and its order relative to every instruction it depends on, so results are bitwise equal to the synchronous schedule.

- *Transfers.* A ``copy`` or ``aten::_to_copy`` between a CPU buffer and a CUDA buffer of one dtype runs without blocking on a stream of its direction (role ``h2d`` or ``d2h``). The copy stream first waits for an event recorded on the compute stream when the copy is issued, which orders it after earlier device work on its buffers, including work on arena storage that the destination reuses. Device work that writes the device buffer, or reads the destination of a host-to-device copy, waits for the copy's event. The arena source of a device-to-host copy stays live until the compute stream waits for the copy, at the latest position the transfer budget allows. Copies of one direction keep their program order, since their stream runs them in issue order. The host thread waits for the event before it accesses the host buffer, and before the schedule returns.
- *Collectives.* A collective whose buffers are on one CUDA device runs on a stream of role ``comm`` in the same way: the stream waits for the compute stream when the collective is issued, and device work that accesses its outputs or writes its inputs waits for its event. Graph segments end at the stream instructions around it, so it runs outside CUDA Graphs.
- *Page-locked host buffers.* A host buffer that the schedule allocates is allocated page-locked. A transfer that uses a caller-owned pageable host buffer remains correct: CUDA stages a pageable source before the copy call returns and completes a copy into pageable memory before it returns, so such transfers overlap only the device work already queued.
- *Early issue.* A transfer is issued after the last earlier instruction it depends on: the writers of its source and every earlier access of its destination. The allocation of its destination moves with it. Transfers are issued, in order, at the earliest such position that does not raise the memory a CUDA device needs, its arena plus the other buffers the schedule allocates, by more than ``TRANSFER_BUDGET``, by default the size of the largest transfer of the schedule.
- *CPU worker lane.* A run of CPU instructions becomes a task on the process's CPU lane, one persistent worker thread that runs tasks in issue order, and starts after the events of the transfers producing its inputs. Host-to-device transfers of the task's results are issued at the end of the task. The host thread joins the task before it accesses a buffer the task accesses, and before the schedule returns.

Only the top-level block is rewritten; loops and branches are planned as single instructions that keep their synchronous transfers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._planning import _Planner


TRANSFER_BUDGET: int | str = "largest"
"""Device bytes that early-issued transfers and deferred device-to-host waits may add to the memory a CUDA device needs without moves; ``"largest"``, the default, allows the size of the largest transfer of the schedule. Read when a schedule is planned, at its first execution."""


@dataclass(frozen=True)
class OverlapReport:
    """What overlap planning changed in one schedule.

    ``moved`` lists early-issued transfers as (original, issued-after) top-level positions, where -1 is the start of the schedule; ``transfers`` counts asynchronous transfers by direction (``h2d``, ``d2h``, and ``comm`` for collectives);
    """

    moved: tuple[tuple[int, int], ...] = ()
    transfers: tuple[tuple[str, int], ...] = ()
    tasks: int = 0


@dataclass
class _Item:
    """One top-level node with its buffer access by alias group."""

    instructions: list[tuple[object, ...]]
    reads: set[int]
    writes: set[int]
    defines: tuple[int, ...]
    cpu: bool
    """CPU work that the worker lane can run."""
    transfer: tuple[str, int, int] | None = None
    """Direction, target and source of an asynchronous transfer."""
    collective: bool = False
    """The node is or contains a collective."""
    touched: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self.touched = self.reads | self.writes


_LANE = frozenset({"call_op", "copy", "zero", "alloc", "release", "view"})


def _transfer(
    planner: _Planner, instruction: tuple[object, ...]
) -> tuple[str, int, int] | None:
    from ._planning import _static_layout

    name = instruction[0]
    if name == "copy" and len(instruction) == 3:
        target, source = cast(int, instruction[1]), cast(int, instruction[2])
    elif (
        name == "call_op"
        and instruction[1] == "aten::_to_copy.default"
        and cast(list, instruction[2])[0] == 1
        and len(cast(list, instruction[4])) == 1
    ):
        source = cast(int, cast(list, instruction[3])[0])
        target = cast(int, cast(list, instruction[4])[0])
    else:
        return None
    target_layout, source_layout = (
        planner.buffers[target],
        planner.buffers[source],
    )
    devices = (target_layout[4], source_layout[4])
    if None in devices or target_layout[3] != source_layout[3]:
        return None
    kinds = {cast(torch.device, device).type for device in devices}
    if kinds != {"cpu", "cuda"} or not _static_layout(target_layout):
        return None
    direction = "h2d" if cast(torch.device, devices[1]).type == "cpu" else "d2h"
    return direction, target, source


def _items(planner: _Planner, find: Callable[[int], int]) -> list[_Item]:
    from ._planning import _nodes

    nodes, _ = _nodes(planner.instructions)
    items = []
    for node in nodes:
        instructions = list(planner.instructions[node.start : node.end + 1])
        reads: set[int] = set()
        writes: set[int] = set()
        defines: list[int] = []
        devices: set[str] = set()
        for instruction in instructions:
            access = planner.instruction_access(instruction)
            reads.update(find(b) for b in access.reads)
            writes.update(find(b) for b in (*access.writes, *access.defines))
            defines.extend(access.defines)
            buffers = (*access.reads, *access.writes, *access.defines)
            if instruction[0] == "release":
                buffers = (cast(int, instruction[1]),)
                writes.add(find(buffers[0]))
            for buffer in buffers:
                device = planner.buffers[buffer][4]
                devices.add("none" if device is None else device.type)
        cpu = (
            len(instructions) == 1
            and instructions[0][0] in _LANE
            and devices == {"cpu"}
        )
        transfer = (
            _transfer(planner, instructions[0])
            if len(instructions) == 1
            else None
        )
        if (
            len(instructions) == 1
            and instructions[0][0] == "collective"
            and len(
                {
                    planner.buffers[b][4]
                    for b in (
                        *cast(list, instructions[0][3]),
                        *cast(list, instructions[0][4]),
                    )
                }
            )
            == 1
            and devices == {"cuda"}
        ):
            transfer = ("comm", -1, -1)
        items.append(
            _Item(
                instructions,
                reads,
                writes,
                tuple(defines),
                cpu,
                transfer,
                any(
                    instruction[0] in ("collective", "call", "python_call")
                    for instruction in instructions
                ),
            )
        )
    return items


def _dependencies(
    items: list[_Item], allocations: dict[int, int]
) -> dict[int, int]:
    """Return, for each transfer, the last item it must follow (-1 for none)."""
    found = {}
    for k, item in enumerate(items):
        if item.transfer is None:
            continue
        position = -1
        for j in range(k - 1, -1, -1):
            other = items[j]
            # The arena allocation of the destination moves with the transfer.
            if allocations.get(j) in item.writes:
                continue
            # Collectives keep their order, which every rank of a group
            # must share.
            if (
                other.writes & item.touched
                or other.reads & item.writes
                or (other.collective and item.collective)
            ):
                position = j
                break
        found[k] = position
    return found


def _issue_positions(
    items: list[_Item], limits: dict[int, int], dependencies: dict[int, int]
) -> dict[int, int]:
    """Return the item after which each movable transfer is issued (-1 for the start).

    ``limits`` maps each movable transfer to the earliest item it may follow.
    """
    issued = {}
    # Copies of one direction share a stream, which runs them in issue order;
    # a copy issued ahead of an earlier one delays it.
    queued: dict[str, int] = {}
    for k, position in dependencies.items():
        stream = cast(tuple, items[k].transfer)[0]
        if k not in limits:
            queued[stream] = k
            continue
        position = max(position, limits[k], queued.get(stream, -1))
        if position < k - 1:
            issued[k] = position
        queued[stream] = position if position < k - 1 else k
    return issued


def overlap(
    planner: _Planner, arena: set[int]
) -> tuple[list[tuple[object, ...]], OverlapReport]:
    """Return the planner's instructions with asynchronous transfers and CPU tasks, and the report.

    Host buffers that become page-locked leave ``arena``.
    """
    # Group declarations stay ahead of the instructions that use them.
    count = 0
    while count < len(planner.instructions) and planner.instructions[count] == (
        "group",
    ):
        count += 1
    declarations = planner.instructions[:count]
    planner.instructions = planner.instructions[count:]
    instructions, report = _overlap(planner, arena)
    return [*declarations, *instructions], report


def _overlap(
    planner: _Planner, arena: set[int]
) -> tuple[list[tuple[object, ...]], OverlapReport]:
    find = planner.groups()
    items = _items(planner, find)
    if not any(item.transfer or item.cpu for item in items):
        return list(planner.instructions), OverlapReport()
    members: dict[int, int] = {}
    for buffer in range(len(planner.buffers)):
        members[find(buffer)] = members.get(find(buffer), 0) + 1
    for item in items:
        if item.transfer is None:
            continue
        direction, target, source = item.transfer
        if direction == "comm":
            continue
        host = source if direction == "h2d" else target
        storage = planner.buffers[host][0]
        if (storage in ("internal", "arena") or host in arena) and members[
            find(host)
        ] == 1:
            planner.buffers[host] = ("pinned", *planner.buffers[host][1:])
            arena.discard(host)
    arena_groups = {find(b) for b in arena}
    allocations = {
        j: find(cast(int, item.instructions[0][1]))
        for j, item in enumerate(items)
        if item.instructions[0][0] == "alloc"
        and find(cast(int, item.instructions[0][1])) in arena_groups
    }
    transfers = {k for k, item in enumerate(items) if item.transfer}
    dependencies = _dependencies(items, allocations)

    def emitter(
        limits: dict[int, int], deadlines: dict[int, int] | None = None
    ) -> _Emitter:
        return _Emitter(
            planner,
            items,
            find,
            _issue_positions(items, limits, dependencies),
            allocations,
            arena,
            deadlines,
        )

    def emit(
        limits: dict[int, int], deadlines: dict[int, int] | None = None
    ) -> tuple[list[tuple[object, ...]], OverlapReport]:
        return emitter(limits, deadlines).run()

    # Without moves, transfers keep their positions.
    peak = _device_peaks(planner, emit({})[0], arena)
    budget = TRANSFER_BUDGET
    if budget == "largest":
        from ._planning import _extent

        sizes = [0]
        for item in items:
            if item.transfer is not None and item.transfer[0] != "comm":
                direction, target, source = item.transfer
                device = target if direction == "h2d" else source
                sizes.append(_extent(planner.buffers[device]))
        budget = max(sizes)
    budget = cast(int, budget)

    def neutral(instructions: list[tuple[object, ...]]) -> bool:
        return all(
            size <= peak.get(device, 0) + budget
            for device, size in _device_peaks(
                planner, instructions, arena
            ).items()
            if device.startswith("cuda")
        )

    # A device-to-host copy from the arena, or a collective reading arena
    # buffers, keeps its sources live until compute waits for it; the wait
    # otherwise precedes the next arena definition on the device.
    probe = emitter({})
    probe.run()
    held = probe.arena_waits
    emitted, report = emit(
        dict.fromkeys(transfers, -1), dict.fromkeys(held, len(items))
    )
    if not neutral(emitted):
        # In program order, issue each transfer as early as it can be issued,
        # and move each such wait as late as it can be moved, without raising
        # the memory a device needs above the budget. Issuing later and
        # waiting earlier never need more memory, so each position is found
        # by bisection.
        limits: dict[int, int] = {}
        deadlines: dict[int, int] = {}
        for k in sorted(transfers):
            # Positions before the one dependencies and stream order allow
            # emit the same schedule.
            low = _issue_positions(items, {**limits, k: -1}, dependencies).get(
                k, k - 1
            )
            high = k - 1
            while low < high:
                middle = (low + high) // 2
                if neutral(emit({**limits, k: middle}, deadlines)[0]):
                    high = middle
                else:
                    low = middle + 1
            if high < k - 1:
                limits[k] = high
            if k in held:
                direction = cast(tuple, items[k].transfer)[0]
                # The stream starts the next copy of the direction once this
                # one completes, so waiting later holds the source without
                # gain.
                low = k - 1
                high = next(
                    (
                        j
                        for j in range(k + 1, len(items))
                        if (transfer := items[j].transfer) is not None
                        and transfer[0] == direction
                    ),
                    len(items),
                )
                while low < high:
                    middle = (low + high + 1) // 2
                    if neutral(emit(limits, {**deadlines, k: middle})[0]):
                        low = middle
                    else:
                        high = middle - 1
                if low >= k:
                    deadlines[k] = low
        emitted, report = emit(limits, deadlines)
    return emitted, report


def _device_peaks(
    planner: _Planner, instructions: list[tuple[object, ...]], arena: set[int]
) -> dict[str, int]:
    """Bytes each device needs: its arena plus the peak of the other buffers the schedule allocates."""
    from ._planning import _extent

    _, offsets = planner.place(instructions, arena)
    arenas: dict[str, int] = {}
    for buffer, offset in offsets.items():
        device = str(planner.buffers[buffer][4])
        arenas[device] = max(
            arenas.get(device, 0), offset + _extent(planner.buffers[buffer])
        )
    live: dict[str, int] = {}
    peaks = dict(arenas)
    for instruction in instructions:
        if instruction[0] not in ("alloc", "release"):
            continue
        buffer = cast(int, instruction[1])
        layout = planner.buffers[buffer]
        device = str(layout[4])
        # Only device memory bounds the moves; host buffers may have sizes
        # known only when the schedule runs.
        if (
            buffer in arena
            or layout[0] not in ("internal", "output")
            or not device.startswith("cuda")
        ):
            continue
        change = (
            _extent(layout) if instruction[0] == "alloc" else -_extent(layout)
        )
        live[device] = live.get(device, 0) + change
        peaks[device] = max(
            peaks.get(device, 0), arenas.get(device, 0) + live[device]
        )
    return peaks


class _Emitter:
    def __init__(
        self,
        planner: _Planner,
        items: list[_Item],
        find: Callable[[int], int],
        issued: dict[int, int],
        allocations: dict[int, int],
        arena: set[int],
        deadlines: dict[int, int] | None = None,
    ) -> None:
        self.planner = planner
        # Guards are keyed by instruction identity, which only this
        # emission's instructions keep unique.
        planner.guards = {}
        self.deadlines = {} if deadlines is None else deadlines
        self.position = -1
        self.waited: dict[int, int] = {}
        self.arena_waits: set[int] = set()
        self.items = items
        self.find = find
        self.issued = issued
        self.allocations = allocations
        self.arena = arena
        self.after: dict[int, list[int]] = {}
        for k, j in issued.items():
            self.after.setdefault(j, []).append(k)
        self.carried = {
            a
            for k in issued
            for a, group in allocations.items()
            if group in items[k].writes and a < k
        }
        self.streams: dict[tuple[int, str], int] = {}
        self.declarations: list[tuple[object, ...]] = []
        self.events = 0
        self.tasks = 0
        self.out: list[tuple[object, ...]] = []
        # Pending device waits: (groups the waiting work accesses, groups it
        # writes, wait instruction, device whose arena definitions wait for it
        # or None, transfer item).
        self.device_waits: list[
            tuple[set[int], set[int], tuple[object, ...], object, int]
        ] = []
        # Pending host waits: (groups, event).
        self.host_waits: list[tuple[set[int], int]] = []
        self.joins: list[tuple[set[int], int]] = []
        self.recorded_in: dict[int, int] = {}
        self.moved: list[tuple[int, int]] = []
        self.transfers: dict[str, int] = {}

    def stream(self, device: int, role: str) -> int:
        if (device, role) not in self.streams:
            self.streams[device, role] = len(self.streams)
            self.declarations.append(("stream", device, role))
        return self.streams[device, role]

    def event(self) -> int:
        self.events += 1
        self.declarations.append(("event",))
        return self.events - 1

    def guard(
        self, instruction: tuple[object, ...], buffers: tuple[int, ...]
    ) -> tuple[object, ...]:
        from ._planning import _Access

        # The instruction keeps the buffers live in arena planning.
        self.planner.guards[id(instruction)] = _Access(reads=buffers)
        return instruction

    def join(self, task: int) -> None:
        for entry in [entry for entry in self.joins if entry[1] == task]:
            self.joins.remove(entry)
            # The join keeps the task's arena buffers live in arena planning.
            self.out.append(
                self.guard(("join", task), self.task_buffers.get(task, ()))
            )

    def settle(self, item: _Item, *, host: bool = True) -> None:
        """Emit the joins and waits that order ``item`` after pending work it depends on."""
        touched = item.touched
        for groups, task in list(self.joins):
            if groups & touched:
                self.join(task)
        # The caching host allocator keeps released page-locked memory until
        # the copies that use it complete.
        released = item.instructions[0][0] == "release" and (
            self.planner.buffers[cast(int, item.instructions[0][1])][0]
            == "pinned"
        )
        if host and not released:
            for entry in list(self.host_waits):
                if entry[0] & touched:
                    self.host_wait(entry)
        defines_arena = {
            self.planner.buffers[b][4] for b in item.defines if b in self.arena
        }
        arena_release = (
            item.instructions[0][0] == "release"
            and item.instructions[0][1] in self.arena
        )
        for entry in list(self.device_waits):
            accessed, written, wait, device, k = entry
            # Device work may read the source of a device-to-host copy and
            # the inputs of a collective. An arena source stays live until
            # its wait, so arena storage defined before the wait never
            # overlaps it; without a deadline, the next arena definition on
            # the device waits.
            deadline = self.deadlines.get(k) if device is not None else None
            if deadline is None:
                due = device is not None and device in defines_arena
            else:
                due = self.position >= deadline
            if (
                accessed & touched
                or (
                    written & item.writes
                    and not (deadline is not None and arena_release)
                )
                or due
            ):
                if wait[1] in self.recorded_in:
                    self.join(self.recorded_in[cast(int, wait[1])])
                self.out.append(wait)
                self.waited[k] = self.position
                self.device_waits.remove(entry)

    def host_wait(self, entry: tuple[set[int], int]) -> None:
        self.host_waits.remove(entry)
        if entry[1] in self.recorded_in:
            self.join(self.recorded_in[entry[1]])
        self.out.append(self.guard(("host_wait", entry[1]), ()))

    def transfer(self, k: int, task: int | None = None) -> None:
        item = self.items[k]
        direction, target, source = cast(tuple[str, int, int], item.transfer)
        self.settle(item, host=task is None)
        if direction == "comm":
            self.collective(item, k)
            return
        device_buffer, host_buffer = (
            (target, source) if direction == "h2d" else (source, target)
        )
        device = cast(torch.device, self.planner.buffers[device_buffer][4])
        index = -1 if device.index is None else device.index
        compute = self.stream(index, "compute")
        side = self.stream(index, direction)
        ready, done = self.event(), self.event()
        if item.instructions[0][0] == "call_op":
            self.out.append(("alloc", target))
        self.out.extend(
            (
                ("record", ready, compute),
                ("use_stream", side),
                ("wait", ready, side),
                ("copy", target, source, True),
                ("record", done, side),
                ("use_stream", compute),
            )
        )
        self.transfers[direction] = self.transfers.get(direction, 0) + 1
        wait = self.guard(("wait", done, compute), (device_buffer,))
        group = {self.find(device_buffer)}
        self.device_wait(
            (
                group if direction == "h2d" else set(),
                set() if direction == "h2d" else group,
                wait,
                device
                if direction == "d2h" and device_buffer in self.arena
                else None,
                k,
            )
        )
        self.host_waits.append(({self.find(host_buffer)}, done))
        if task is not None:
            self.recorded_in[done] = task

    def device_wait(
        self,
        entry: tuple[set[int], set[int], tuple[object, ...], object, int],
    ) -> None:
        self.device_waits.append(entry)
        if entry[3] is not None:
            self.arena_waits.add(entry[4])

    def collective(self, item: _Item, k: int) -> None:
        instruction = item.instructions[0]
        inputs = tuple(cast(list, instruction[3]))
        outputs = tuple(cast(list, instruction[4]))
        device = cast(torch.device, self.planner.buffers[outputs[0]][4])
        index = -1 if device.index is None else device.index
        compute = self.stream(index, "compute")
        side = self.stream(index, "comm")
        ready, done = self.event(), self.event()
        self.out.extend(
            (
                ("record", ready, compute),
                ("use_stream", side),
                ("wait", ready, side),
                instruction,
                ("record", done, side),
                ("use_stream", compute),
            )
        )
        self.planner.side.add(id(instruction))
        self.transfers["comm"] = self.transfers.get("comm", 0) + 1
        wait = self.guard(("wait", done, compute), (*inputs, *outputs))
        self.device_wait(
            (
                {self.find(b) for b in outputs},
                {self.find(b) for b in inputs},
                wait,
                device if any(b in self.arena for b in inputs) else None,
                k,
            )
        )

    def issue_after(self, j: int, task: int | None = None) -> None:
        for k in sorted(self.after.get(j, ())):
            self.moved.append((k, j))
            for a, group in self.allocations.items():
                if a in self.carried and group in self.items[k].writes:
                    self.out.extend(self.items[a].instructions)
            self.transfer(k, task)

    def lane(self, index: int) -> int:
        """Emit the CPU run starting at ``index`` as a task; return the next item."""
        items = self.items
        run = []
        while (
            index + len(run) < len(items)
            and items[index + len(run)].cpu
            and items[index + len(run)].transfer is None
            and index + len(run) not in self.issued
            and index + len(run) not in self.carried
        ):
            run.append(index + len(run))
        if not any(
            items[j].instructions[0][0] in ("call_op", "copy", "zero")
            for j in run
        ):
            return index
        touched = set().union(*(items[j].touched for j in run))
        for groups, task in list(self.joins):
            if groups & touched:
                self.join(task)
        # The worker waits for the transfers that produce its inputs.
        waits = []
        for entry in list(self.host_waits):
            if entry[0] & touched:
                self.host_waits.remove(entry)
                if entry[1] in self.recorded_in:
                    self.join(self.recorded_in[entry[1]])
                waits.append(entry[1])
        task = self.tasks
        self.tasks += 1
        self.declarations.append(("task",))
        self.out.append(("begin_task", task, waits))
        buffers: set[int] = set()
        for j in run:
            self.out.extend(items[j].instructions)
            for instruction in items[j].instructions:
                access = self.planner.instruction_access(instruction)
                buffers.update((*access.reads, *access.writes, *access.defines))
            self.issue_after(j, task)
        follow = index + len(run)
        # Host-to-device transfers of the task's results run in the task.
        while (
            follow < len(items)
            and items[follow].transfer is not None
            and cast(tuple, items[follow].transfer)[0] == "h2d"
            and items[follow].reads & touched
            and follow not in self.issued
            and items[follow].instructions[0][0] == "copy"
        ):
            self.transfer(follow, task)
            self.issue_after(follow, task)
            touched |= items[follow].touched
            buffers.update(items[follow].instructions[0][1:3])  # type: ignore[arg-type]
            follow += 1
        self.out.append(("end_task",))
        self.joins.append((touched, task))
        self.task_buffers[task] = tuple(sorted(buffers))
        return follow

    task_buffers: dict[int, tuple[int, ...]]

    def run(self) -> tuple[list[tuple[object, ...]], OverlapReport]:
        self.task_buffers = {}
        items = self.items
        self.issue_after(-1)
        index = 0
        while index < len(items):
            if index in self.issued or index in self.carried:
                self.issue_after(index)
                index += 1
                continue
            following = self.lane(index)
            if following != index:
                index = following
                continue
            item = items[index]
            self.position = index
            if item.transfer is not None:
                self.transfer(index)
            else:
                self.settle(item)
                self.out.extend(item.instructions)
            self.issue_after(index)
            index += 1
        for _, task in list(self.joins):
            self.join(task)
        for _, _, wait, _, k in self.device_waits:
            self.out.append(wait)
            self.waited[k] = len(items)
        # The caller may access its host buffers once the schedule returns.
        visible = {
            self.find(buffer)
            for buffer, layout in enumerate(self.planner.buffers)
            if layout[0] not in ("pinned", "internal", "arena")
        }
        for entry in list(self.host_waits):
            if entry[0] & visible:
                self.host_wait(entry)
        report = OverlapReport(
            tuple(sorted(self.moved)),
            tuple(sorted(self.transfers.items())),
            self.tasks,
        )
        return [*self.declarations, *self.out], report


__all__ = ["OverlapReport", "overlap"]
