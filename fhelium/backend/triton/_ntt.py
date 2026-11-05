"""Schedule NTT endpoints, pure expressions, and synchronized native interiors."""

from __future__ import annotations

import hashlib
import linecache
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from typing import Any, cast

import torch

from fhelium.backend.ntt._stages import NttStages, endpoint_stages
from fhelium.config.ntt import NttBackendPolicy
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Operation, SSAValue

from fhelium.ir.dialects import ntt, rns

from ._expressions import Anchor, BoundGraph, Emitter, Expr, ValueExpr
from ._launch import PreparedKernel
from ._layout import check_ntt_layouts
from ._ntt_codegen import StageEmitter, compile_endpoint


def _inverse(anchor: Anchor) -> bool:
    return isinstance(
        anchor.operation,
        (
            ntt.NttMontgomeryToCoefficientStandardOp,
            ntt.NttMontgomeryToCoefficientMontgomeryOp,
        ),
    )


def _anchor_leaves(expressions: tuple[Expr, ...]) -> tuple[Expr, ...]:
    leaves: dict[Expr, None] = {}
    visited: set[int] = set()

    def visit(expr):
        if id(expr) in visited:
            return
        visited.add(id(expr))
        if expr.op == "anchor":
            leaves[expr] = None
        else:
            for argument in expr.args:
                visit(argument)

    for expr in expressions:
        visit(expr)
    return tuple(leaves)


@dataclass(frozen=True)
class _Endpoint:
    anchor: int
    start: int
    end: int
    tiled: bool
    prologue: bool
    epilogue: bool
    transform: bool = True

    def partition(self, n: int, inverse: bool) -> tuple[int, int | None]:
        if not self.transform:
            return 1, 1
        log_n = n.bit_length() - 1
        width = 1 << (self.end - self.start)
        return width, None if self.tiled else 1 << (
            self.start if inverse else log_n - self.end
        )


@dataclass(frozen=True)
class _Step:
    endpoints: tuple[_Endpoint, ...]
    native: bool = False


def _steps(
    graph: BoundGraph, schedules: tuple[NttStages, ...]
) -> tuple[_Step, ...]:
    """Schedule stage dependencies and combine ready coefficient partitions."""
    return _schedule(
        graph.anchors[0].source.shape[-1],
        tuple(_inverse(anchor) for anchor in graph.anchors),
        tuple(
            frozenset(
                e.data[0] for e in _anchor_leaves(anchor.source.components)
            )
            for anchor in graph.anchors
        ),
        schedules,
    )


def _schedule(
    n: int,
    inverse: tuple[bool, ...],
    predecessors: tuple[frozenset[int], ...],
    schedules: tuple[NttStages, ...],
) -> tuple[_Step, ...]:
    nodes: list[_Endpoint] = []
    chains: list[list[int]] = []
    dependencies: dict[int, set[int]] = {}
    for ordinal, schedule in enumerate(schedules):
        chain = []
        for part, (start, end, tiled) in enumerate(schedule.intervals):
            index = len(nodes)
            nodes.append(
                _Endpoint(
                    ordinal,
                    start,
                    end,
                    tiled,
                    part == 0,
                    part == len(schedule.intervals) - 1,
                    not schedule.whole,
                )
            )
            dependencies[index] = {chain[-1]} if chain else set()
            chain.append(index)
        chains.append(chain)
    for ordinal, chain in enumerate(chains):
        dependencies[chain[0]].update(
            chains[producer][-1] for producer in predecessors[ordinal]
        )
    generated = {
        index
        for index, node in enumerate(nodes)
        if node.prologue or node.epilogue
    }
    partitions = {
        index: node.partition(n, inverse[node.anchor])
        for index, node in enumerate(nodes)
        if index in generated
    }
    completed: set[int] = set()
    steps: list[_Step] = []

    def cluster(consumer: int, ready: set[int]) -> list[int]:
        partition = partitions[consumer]
        selected: list[int] = []
        available = set(completed)

        def extend(index):
            missing = dependencies[index] - available
            if not missing or not missing <= ready:
                return False
            if any(
                not nodes[p].epilogue
                or p not in generated
                or partitions[p] != partition
                for p in missing
            ):
                return False
            for producer in sorted(missing):
                selected.append(producer)
                available.add(producer)
            selected.append(index)
            available.add(index)
            return True

        if not extend(consumer):
            return []
        while True:
            extended = False
            for index, node in enumerate(nodes):
                if (
                    index in available
                    or not node.prologue
                    or partitions[index] != partition
                ):
                    continue
                missing = dependencies[index] - available
                if not (dependencies[index] & set(selected)):
                    continue
                if not missing:
                    selected.append(index)
                    available.add(index)
                    extended = True
                elif extend(index):
                    extended = True
            if not extended:
                return selected

    while len(completed) < len(nodes):
        ready = {
            index
            for index in range(len(nodes))
            if index not in completed and dependencies[index] <= completed
        }
        choices = [
            cluster(index, ready)
            for index, node in enumerate(nodes)
            if index not in completed and node.prologue
        ]
        selected = max(choices, key=len, default=[])
        if selected:
            steps.append(_Step(tuple(nodes[index] for index in selected)))
            completed.update(selected)
            continue
        # Advance unfinished branches before materializing their ready ends,
        # allowing a join to consume several producer endpoints together.
        index = min(ready, key=lambda index: (nodes[index].epilogue, index))
        node = nodes[index]
        steps.append(_Step((node,), index not in generated))
        completed.add(index)
    return tuple(steps)


def _limit_steps(
    steps: tuple[_Step, ...], limit: int | None
) -> tuple[_Step, ...]:
    if limit is None:
        return steps
    return tuple(
        part
        for step in steps
        for part in (
            (step,)
            if step.native
            else tuple(
                _Step(step.endpoints[start : start + limit])
                for start in range(0, len(step.endpoints), limit)
            )
        )
    )


def _fusion_qualified(
    operations: Sequence[Operation], max_endpoints: int | None
) -> bool:
    """Require a generated expression group when whole or unresolved anchors occur.

    Semantic matching is independent of this qualification. A complete native
    transform contributes no generated operation, and wrapping separate single
    expressions around it does not establish fusion. Known stage partitions use
    the execution scheduler; unresolved geometry proves only local expression
    groups. This is an eligibility check, not a latency estimate.
    """
    from fhelium.backend.ntt._selection import select_operation_policy
    from ._expressions import NTT_OPS, SUPPORTED_OPERATIONS
    from ._matching import RegionMatcher, _outer, _state

    anchors = tuple(op for op in operations if isinstance(op, NTT_OPS))
    if not anchors:
        return True
    layout = RegionMatcher(include_ntt=True).layout
    schedules = []
    sizes = []
    inverse = []
    for op in anchors:
        shape, _ = layout(op.operands[0])
        dimension = _state(_outer(op.operands[0])).get("ring_dimension")
        n = (
            shape[-1]
            if shape is not None
            else int(dimension.value.data)
            if isinstance(dimension, IntegerAttr)
            else None
        )
        log_n = n.bit_length() - 1 if n is not None else None
        policy = select_operation_policy(op, log_n=log_n)
        direction = isinstance(
            op,
            (
                ntt.NttMontgomeryToCoefficientStandardOp,
                ntt.NttMontgomeryToCoefficientMontgomeryOp,
            ),
        )
        schedules.append(
            endpoint_stages(policy, log_n, direction)
            if policy is not None
            else None
        )
        sizes.append(n)
        inverse.append(direction)
    if all(plan is not None and not plan.whole for plan in schedules):
        return True

    selected = set(operations)
    anchor_ids: dict[Operation, int] = {
        op: index for index, op in enumerate(anchors)
    }

    @lru_cache(maxsize=None)
    def expression(
        value: SSAValue,
    ) -> tuple[frozenset[Operation], frozenset[int]]:
        owner = value.owner
        if owner not in selected:
            return frozenset(), frozenset()
        owner = cast(Operation, owner)
        if owner in anchor_ids:
            return frozenset(), frozenset((anchor_ids[owner],))
        nodes: set[Operation] = set()
        dependencies: set[int] = set()
        for operand in owner.operands:
            pure, prior = expression(operand)
            nodes.update(pure)
            dependencies.update(prior)
        # Component extraction is a storage view rather than a generated compute.
        if isinstance(owner, SUPPORTED_OPERATIONS) and not isinstance(
            owner, rns.ExtractComponentOp
        ):
            nodes.add(owner)
        return frozenset(nodes), frozenset(dependencies)

    inputs = tuple(expression(op.operands[0]) for op in anchors)
    prefix_nodes = frozenset().union(*(nodes for nodes, _ in inputs))
    detached = all(op.parent_block() is None for op in operations)
    outputs = tuple(
        expression(value)
        for op in operations
        for value in op.results
        if (detached and not value.uses)
        or any(use.operation not in selected for use in value.uses)
    )
    if (
        all(plan is not None and plan.intervals for plan in schedules)
        and len(set(sizes)) == 1
    ):
        steps = _limit_steps(
            _schedule(
                sizes[0],
                tuple(inverse),
                tuple(deps for _, deps in inputs),
                tuple(schedules),
            ),
            max_endpoints,
        )
        completed = {
            endpoint.anchor: order
            for order, endpoint in enumerate(
                e for step in steps for e in step.endpoints if e.epilogue
            )
        }
        first = min(completed, key=completed.__getitem__)
        stores: list[set[Operation]] = [set() for _ in anchors]
        for nodes, dependencies in outputs:
            ready = max(dependencies, key=completed.__getitem__, default=first)
            # Recomputing an already materialized prologue for an external use
            # must not turn two separate singleton expressions into a gain.
            stores[ready].update(nodes - prefix_nodes)
        for step in steps:
            if step.native:
                continue
            group: set[Operation] = set()
            for endpoint in step.endpoints:
                if endpoint.transform:
                    group.add(anchors[endpoint.anchor])
                if endpoint.prologue:
                    group.update(inputs[endpoint.anchor][0])
                if endpoint.epilogue:
                    group.update(stores[endpoint.anchor])
            if len(group) >= 2:
                return True
        return False

    # Unknown schedules never count as generated transforms. Pointwise chains
    # remain provable without concrete tables, shapes or stage partitions.
    decomposed = {
        i
        for i, plan in enumerate(schedules)
        if plan is not None and not plan.whole
    }
    if any(
        len(nodes) >= 2 or bool(nodes) and i in decomposed
        for i, (nodes, _) in enumerate(inputs)
    ):
        return True
    groups: dict[frozenset[int], set[Operation]] = {}
    for nodes, dependencies in outputs:
        group = groups.setdefault(dependencies, set())
        group.update(nodes - prefix_nodes)
        if (
            len(group) >= 2
            or group
            and dependencies
            and dependencies <= decomposed
        ):
            return True
    return False


def _prior_leaves(
    graph: BoundGraph, step: _Step, stores: tuple[tuple[ValueExpr, ...], ...]
) -> tuple[Expr, ...]:
    available = {endpoint.anchor for endpoint in step.endpoints}
    expressions = []
    for endpoint in step.endpoints:
        if endpoint.prologue:
            expressions.extend(graph.anchors[endpoint.anchor].source.components)
        if endpoint.epilogue:
            expressions.extend(
                e for value in stores[endpoint.anchor] for e in value.components
            )
    return tuple(
        e
        for e in _anchor_leaves(tuple(expressions))
        if e.data[0] not in available
    )


def _source(
    graph: BoundGraph,
    schedules: tuple[NttStages, ...],
    step: _Step,
    stores: tuple[tuple[ValueExpr, ...], ...],
    writes: tuple[int, ...],
) -> str:
    rows, n = graph.anchors[0].source.shape[-2:]
    batches = prod(graph.anchors[0].source.batch_shape)
    leaves = _prior_leaves(graph, step, stores)
    prior_ids = tuple(sorted({expr.data[0] for expr in leaves}))
    scratch_ids = tuple(
        sorted(
            set(prior_ids)
            | {e.anchor for e in step.endpoints if not e.prologue}
            | set(writes)
        )
    )
    output_values = tuple(
        value
        for e in step.endpoints
        if e.epilogue
        for value in stores[e.anchor]
    )
    arguments = [
        *(f"x{i}" for i in range(len(graph.layouts))),
        *(f"scratch{i}" for i in scratch_ids),
        *(f"out{v.root}" for v in output_values),
        *(
            name
            for e in step.endpoints
            if e.transform
            for name in (
                f"twiddles{e.anchor}",
                *((f"roots{e.anchor}",) if schedules[e.anchor].radix else ()),
                f"TSTRIDE{e.anchor}: tl.constexpr",
                f"TESTRIDE{e.anchor}: tl.constexpr",
                *(
                    (f"TWSTRIDE{e.anchor}: tl.constexpr",)
                    if schedules[e.anchor].twiddle_encoding == "shoup32"
                    else ()
                ),
                *(
                    (
                        f"RSTRIDE{e.anchor}: tl.constexpr",
                        f"RESTRIDE{e.anchor}: tl.constexpr",
                    )
                    if schedules[e.anchor].radix
                    else ()
                ),
            )
        ),
        "rns_parameters",
        "PARAM_STRIDE: tl.constexpr",
        "PARAM_ROW_STRIDE: tl.constexpr",
        "RADIX: tl.constexpr",
        "N: tl.constexpr",
        "BLOCK: tl.constexpr",
    ]
    lines = [
        "row = tl.program_id(1).to(tl.int64)",
        "batch = tl.program_id(2).to(tl.int64)",
        "q = (tl.load(rns_parameters + row * PARAM_ROW_STRIDE).to(tl.int64) >> 1)",
        "k = tl.load(rns_parameters + 3 * PARAM_STRIDE + row * PARAM_ROW_STRIDE).to(tl.int64) | (tl.load(rns_parameters + 4 * PARAM_STRIDE + row * PARAM_ROW_STRIDE).to(tl.int64) << (RADIX // 2))",
        "r2 = tl.load(rns_parameters + 5 * PARAM_STRIDE + row * PARAM_ROW_STRIDE).to(tl.int64)",
        "ninv = tl.load(rns_parameters + 7 * PARAM_STRIDE + row * PARAM_ROW_STRIDE).to(tl.int64)",
    ]
    first = step.endpoints[0]
    if first.transform:
        stage = StageEmitter(
            n=n,
            start=first.start,
            end=first.end,
            inverse=_inverse(graph.anchors[first.anchor]),
            tiled=first.tiled,
        )
        indices = stage.indices()
        lines.extend(stage.lines)
    else:
        # Whole transforms own every butterfly and representation transition.
        # Their generated ends evaluate only coefficient-wise expressions.
        indices = ("index",)
        lines.extend(
            (
                "index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
                "mask = index < N",
            )
        )
    # Each lane carries the completed anchors in this kernel. Earlier transforms
    # that still have consumers are read from their prepared storage instead.
    overrides: list[dict[Expr, str]] = [{} for _ in indices]
    for lane, index in enumerate(indices):
        for expr in leaves:
            ordinal, component = expr.data
            count = prod(graph.anchors[ordinal].result.batch_shape)
            name = f"prior_{lane}_{ordinal}_{component}"
            lines.append(
                f"{name} = tl.load(scratch{ordinal} + (({component * count} + batch) * {rows} + row) * N + {index}, mask=mask, other=0).to(tl.int64)"
            )
            overrides[lane][expr] = name

    for endpoint in step.endpoints:
        ordinal = endpoint.anchor
        anchor = graph.anchors[ordinal]
        stage = None
        if endpoint.transform:
            stage = StageEmitter(
                n=n,
                start=endpoint.start,
                end=endpoint.end,
                inverse=_inverse(anchor),
                tiled=endpoint.tiled,
                radix=schedules[ordinal].radix,
                twiddle_encoding=schedules[ordinal].twiddle_encoding,
                prefix=f"a{ordinal}_",
            )
            lines.extend(
                (
                    f"twiddles = twiddles{ordinal}",
                    f"TWIDDLE_STRIDE = TSTRIDE{ordinal}",
                    f"TWIDDLE_ELEMENT_STRIDE = TESTRIDE{ordinal}",
                )
            )
            if schedules[ordinal].twiddle_encoding == "shoup32":
                lines.append(f"TWIDDLE_WORD_STRIDE = TWSTRIDE{ordinal}")
            if schedules[ordinal].radix:
                lines.extend(
                    (
                        f"roots = roots{ordinal}",
                        f"ROOT_STRIDE = RSTRIDE{ordinal}",
                        f"ROOT_ELEMENT_STRIDE = RESTRIDE{ordinal}",
                    )
                )
        emitter = Emitter(graph.layouts)
        cursor = 0

        def emit(expr, lane):
            nonlocal cursor
            value = emitter.emit(
                expr, indices[lane], "row", "batch", overrides[lane]
            )
            lines.extend(emitter.lines[cursor:])
            cursor = len(emitter.lines)
            return value

        transformed = []
        for component, expression in enumerate(anchor.source.components):
            loaded = []
            for lane, index in enumerate(indices):
                if endpoint.prologue:
                    value = emit(expression, lane)
                    if stage is not None and isinstance(
                        anchor.operation,
                        ntt.CoefficientStandardToNttMontgomeryOp,
                    ):
                        converted = stage.name("montgomery")
                        lines.append(
                            f"{converted} = montgomery_mul({value}, r2, q, k, RADIX)"
                        )
                        value = converted
                else:
                    value = (
                        stage.name("input")
                        if stage is not None
                        else f"a{ordinal}_input_{component}_{lane}"
                    )
                    lines.append(
                        f"{value} = tl.load(scratch{ordinal} + (({component * batches} + batch) * {rows} + row) * N + {index}, mask=mask, other=0).to(tl.int64)"
                    )
                loaded.append(value)
            if stage is None:
                transformed.append(tuple(loaded))
            else:
                transformed.append(stage.transform(tuple(loaded)))
                lines.extend(stage.lines)
                stage.lines.clear()
        for lane, index in enumerate(indices):
            for component, values in enumerate(transformed):
                value = values[lane]
                if endpoint.epilogue and stage is not None and _inverse(anchor):
                    normalized = stage.name("normalized")
                    lines.append(
                        f"{normalized} = montgomery_mul({value}, ninv, q, k, RADIX)"
                    )
                    value = normalized
                    if isinstance(
                        anchor.operation,
                        ntt.NttMontgomeryToCoefficientStandardOp,
                    ):
                        standard = stage.name("standard")
                        lines.extend(
                            (
                                f"{standard} = montgomery_reduce({value}, q, k, RADIX)",
                                f"{standard} = tl.where({standard} < q, {standard}, {standard} - q)",
                            )
                        )
                        value = standard
                if endpoint.epilogue:
                    overrides[lane][anchor.result.components[component]] = value
                if ordinal in writes:
                    lines.append(
                        f"tl.store(scratch{ordinal} + (({component * batches} + batch) * {rows} + row) * N + {index}, {value}, mask=mask)"
                    )
        if endpoint.epilogue:
            # Rebuild the cache after the transform, so its input expressions do
            # not bypass a completed anchor when an output shares a subexpression.
            emitter = Emitter(graph.layouts)
            cursor = 0
            for lane, index in enumerate(indices):
                for output in stores[ordinal]:
                    for component, expression in enumerate(output.components):
                        value = emit(expression, lane)
                        lines.append(
                            f"tl.store(out{output.root} + (({component * batches} + batch) * {rows} + row) * N + {index}, {value}, mask=mask)"
                        )
    return (
        "def endpoint("
        + ", ".join(arguments)
        + "):\n"
        + "\n".join("    " + line for line in lines)
        + "\n"
    )


@lru_cache(maxsize=128)
def _kernel(source: str) -> Any:
    return compile_endpoint(source)


@dataclass(frozen=True)
class NttExecution:
    """Execute dependency-selected endpoints with live tables and planned storage."""

    graph: BoundGraph
    schedules: tuple[NttStages, ...]
    steps: tuple[_Step, ...]
    output_roots: tuple[ValueExpr, ...]
    stores: tuple[tuple[ValueExpr, ...], ...]
    dtype: torch.dtype
    device: torch.device
    table_strides: tuple[tuple[tuple[int, ...], ...], ...]
    complete: tuple[Callable | None, ...]
    _call: Callable = field(init=False, repr=False, compare=False)
    execution_source: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        graph = self.graph
        rows, n = graph.anchors[0].source.shape[-2:]
        batches = prod(graph.anchors[0].source.batch_shape)
        aliases = {
            i: value.root
            for i, anchor in enumerate(graph.anchors)
            for value in self.output_roots
            if value.root == anchor.result.root
        }
        writes = []
        required = []
        prior_ids = []
        for position, step in enumerate(self.steps):
            prior = tuple(
                sorted(
                    {e.data[0] for e in _prior_leaves(graph, step, self.stores)}
                )
            )
            prior_ids.append(prior)
            emitted = []
            for endpoint in step.endpoints:
                ordinal = endpoint.anchor
                if not endpoint.epilogue:
                    emitted.append(ordinal)
                    continue
                future = tuple(
                    expr
                    for later in self.steps[position + 1 :]
                    for expr in _prior_leaves(graph, later, self.stores)
                )
                if any(e.data[0] == ordinal for e in future):
                    emitted.append(ordinal)
            # An escaped anchor is stored through its output alias at this step.
            emitted = tuple(
                i
                for i in emitted
                if i not in aliases
                or not any(e.anchor == i and e.epilogue for e in step.endpoints)
            )
            writes.append(emitted)
            required.append(
                set(prior)
                | {e.anchor for e in step.endpoints if not e.prologue}
                | set(emitted)
                | {
                    i
                    for i in aliases
                    if any(e.anchor == i and e.epilogue for e in step.endpoints)
                }
            )
        last_use = {
            i: position for position, ids in enumerate(required) for i in ids
        }
        namespace = {
            "empty": torch.empty,
            "dtype": self.dtype,
            "device": self.device,
            "device_scope": torch.cuda.device,
            "current_stream": torch.cuda.current_stream,
        }
        lines = ["def execute(values, tables):"]
        for value in self.output_roots:
            lines.append(
                f"    output_{value.root} = empty({value.shape!r}, dtype=dtype, device=device)"
            )
        for ordinal, root in aliases.items():
            lines.append(f"    scratch{ordinal} = output_{root}")
        lines.extend(
            (
                "    with device_scope(device):",
                "        stream = current_stream(device).cuda_stream",
            )
        )
        allocated = set(aliases)
        for position, step in enumerate(self.steps):
            for ordinal in sorted(required[position] - allocated):
                lines.append(
                    f"        scratch{ordinal} = empty({graph.anchors[ordinal].result.shape!r}, dtype=dtype, device=device)"
                )
                allocated.add(ordinal)
            if step.native:
                endpoint = step.endpoints[0]
                ordinal = endpoint.anchor
                anchor = graph.anchors[ordinal]
                table = anchor.table_indices
                symbol = f"native_{position}"
                if self.schedules[ordinal].whole:
                    namespace[symbol] = self.complete[ordinal]
                    operands = [
                        f"scratch{ordinal}",
                        *(f"tables[{i}]" for i in table),
                    ]
                    args = ["(" + ", ".join(operands) + ")"]
                elif self.schedules[ordinal].radix:
                    namespace[symbol] = (
                        torch.ops.fhelium_ntt_ops.fixed_radix_ntt_stage_range_
                    )
                    args = [
                        f"scratch{ordinal}",
                        f"tables[{table[1]}]",
                        f"tables[{table[2]}]",
                        f"tables[{table[0]}]",
                        repr(_inverse(anchor)),
                        str(endpoint.start),
                        str(endpoint.end),
                    ]
                else:
                    namespace[symbol] = (
                        torch.ops.fhelium_ntt_ops.compact_ntt_stage_range_
                    )
                    args = [
                        f"scratch{ordinal}",
                        f"tables[{table[1]}]",
                        f"tables[{table[0]}]",
                        repr(_inverse(anchor)),
                        str(endpoint.start),
                        str(endpoint.end),
                        str(self.schedules[ordinal].group),
                    ]
                lines.append(f"        {symbol}({', '.join(args)})")
            else:
                first = step.endpoints[0]
                width, spacing = first.partition(
                    n, _inverse(graph.anchors[first.anchor])
                )
                grid = (
                    n // width
                    if spacing is None
                    else (n // width + 127) // 128,
                    rows,
                    batches,
                )
                symbol = f"launch_{position}"
                namespace[symbol] = PreparedKernel(
                    _kernel(
                        _source(
                            graph,
                            self.schedules,
                            step,
                            self.stores,
                            writes[position],
                        )
                    ),
                    grid,
                )
                scratch_ids = tuple(
                    sorted(
                        set(prior_ids[position])
                        | {e.anchor for e in step.endpoints if not e.prologue}
                        | set(writes[position])
                    )
                )
                outputs = tuple(
                    v
                    for e in step.endpoints
                    if e.epilogue
                    for v in self.stores[e.anchor]
                )
                args = [
                    "*values",
                    *(f"scratch{i}" for i in scratch_ids),
                    *(f"output_{v.root}" for v in outputs),
                ]
                for endpoint in step.endpoints:
                    if not endpoint.transform:
                        continue
                    ordinal = endpoint.anchor
                    table = graph.anchors[ordinal].table_indices
                    args.append(f"tables[{table[1]}]")
                    if self.schedules[ordinal].radix:
                        args.append(f"tables[{table[2]}]")
                    args.extend(
                        str(stride)
                        for strides in self.table_strides[ordinal][1:]
                        for stride in strides
                    )
                table = graph.anchors[first.anchor].table_indices
                args.extend(
                    (
                        f"tables[{table[0]}]",
                        *(
                            str(stride)
                            for stride in self.table_strides[first.anchor][0]
                        ),
                        str(32 if self.dtype == torch.int32 else 62),
                        str(n),
                        "128",
                        "stream=stream",
                    )
                )
                lines.append(f"        {symbol}({', '.join(args)})")
            for ordinal in sorted(required[position]):
                if last_use[ordinal] == position and ordinal not in aliases:
                    lines.append(f"        del scratch{ordinal}")
                    allocated.remove(ordinal)
        results = []
        for value in graph.outputs:
            result = (
                f"values[{-value.root - 1}]"
                if value.root < 0
                else f"output_{value.root}"
            )
            for index in value.view:
                name = f"view_{len(namespace)}"
                namespace[name] = index
                result += f"[{name}]"
            results.append(result)
        lines.append(
            "    return ("
            + ", ".join(results)
            + ("," if len(results) == 1 else "")
            + ")"
        )
        source = "\n".join(lines) + "\n"
        filename = f"<fhelium-ntt-schedule-{hashlib.sha256(source.encode()).hexdigest()}>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        exec(compile(source, filename, "exec"), namespace)
        object.__setattr__(self, "_call", namespace["execute"])
        object.__setattr__(self, "execution_source", source)

    @classmethod
    def build(
        cls,
        graph: BoundGraph,
        values: tuple[torch.Tensor, ...],
        tables: tuple[torch.Tensor, ...],
        *,
        max_endpoints: int | None = None,
    ) -> NttExecution:
        first = values[0]
        if first.device.type != "cuda" or first.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "Generated NTT endpoints require CUDA int32/int64 operands"
            )
        if any(
            v.device != first.device or v.dtype != first.dtype for v in values
        ):
            raise ValueError("NTT operands must share dtype and device")
        root_ids = tuple(
            dict.fromkeys(v.root for v in graph.outputs if v.root >= 0)
        )
        outputs = tuple(graph.roots[i] for i in root_ids)
        check_ntt_layouts(
            [
                (v.shape, v.component_axis)
                for v in (*[a.source for a in graph.anchors], *outputs)
            ]
        )
        rows, n = graph.anchors[0].source.shape[-2:]
        from fhelium.backend.ntt._selection import (
            select_operation_policy,
            select_policy,
        )
        from fhelium.backend.ntt.operations import (
            _TRANSITIONS,
            prepare_transition,
        )

        schedules = []
        strides = []
        complete = []
        for anchor in graph.anchors:
            log_n = n.bit_length() - 1
            policy = select_operation_policy(anchor.operation, log_n=log_n)
            if policy is None:
                # The selection helper defers only when its operation facts are
                # insufficient. Bound placement now supplies the missing fact;
                # table ranks or contents never select an algorithm or encoding.
                policy = select_policy(
                    device_type=first.device.type, log_n=log_n
                )
            policy = cast(NttBackendPolicy, policy)
            schedule = endpoint_stages(policy, log_n, _inverse(anchor))
            if schedule is None:
                raise ValueError(f"NTT schedule is incompatible with N={n}")
            selected = tuple(tables[i] for i in anchor.table_indices)
            if schedule.whole:
                expected = (
                    ((8, rows), first.dtype),
                    ((rows, log_n, n // 2), first.dtype),
                    ((log_n, n // 2), torch.int32),
                    ((log_n, n // 2), torch.int32),
                )
            elif schedule.radix:
                expected = (
                    ((8, rows), first.dtype),
                    ((rows, n - 1), first.dtype),
                    ((rows, schedule.radix), first.dtype),
                )
            else:
                pair = schedule.twiddle_encoding == "shoup32"
                if pair and (
                    first.dtype != torch.int32 or torch.version.hip is not None
                ):
                    raise ValueError(
                        "Shoup32 stage tables require CUDA int32 execution"
                    )
                expected = (
                    ((8, rows), first.dtype),
                    ((rows, n, 2) if pair else (rows, n), first.dtype),
                )
            if len(selected) != len(expected) or any(
                t.shape != shape or t.dtype != dtype or t.device != first.device
                for t, (shape, dtype) in zip(selected, expected, strict=True)
            ):
                raise ValueError(
                    "NTT table layout, dtype or device differs from the selected algorithm; "
                    "supply consistent NTT selection facts and table operands"
                )
            schedules.append(schedule)
            strides.append(tuple(tuple(t.stride()) for t in selected))
            complete.append(
                prepare_transition(
                    _TRANSITIONS[type(anchor.operation)],
                    policy.name,
                    in_place=True,
                )
                if schedule.whole
                else None
            )
        schedules = tuple(schedules)
        steps = _steps(graph, schedules)
        steps = _limit_steps(steps, max_endpoints)
        completed_order = {
            endpoint.anchor: position
            for position, endpoint in enumerate(
                e for step in steps for e in step.endpoints if e.epilogue
            )
        }
        first_completed = min(completed_order, key=completed_order.__getitem__)
        stores: list[list[ValueExpr]] = [[] for _ in graph.anchors]
        for value in outputs:
            ready = max(
                (e.data[0] for e in _anchor_leaves(value.components)),
                key=completed_order.__getitem__,
                default=first_completed,
            )
            stores[ready].append(value)
        return cls(
            graph,
            schedules,
            steps,
            outputs,
            tuple(tuple(v) for v in stores),
            first.dtype,
            first.device,
            tuple(strides),
            tuple(complete),
        )

    def execute(
        self, values: tuple[torch.Tensor, ...], tables: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return self._call(values, tables)
