"""Compose generated NTT endpoints and synchronized native middle stages.

A schedule fuses component expressions into transform loads and stores. Each
interior stage group retains its native synchronization and scratch storage.
Successive transforms materialize the preceding transform's live result while
its pointwise consumers can become the next transform's input expressions.
"""

from __future__ import annotations

import hashlib
import linecache
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from typing import Any

import torch

from fhelium.config.ntt import CompactRadix2Policy, resolve_ntt_backend_policy
from fhelium.ir.dialects import ntt

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

    def visit(expr: Expr) -> None:
        if expr.op == "anchor":
            leaves[expr] = None
        else:
            for argument in expr.args:
                visit(argument)

    for expr in expressions:
        visit(expr)
    return tuple(leaves)


def _endpoint_leaves(
    anchor: Anchor, ordinal: int, prologue: bool, outputs: tuple[ValueExpr, ...]
) -> tuple[Expr, ...]:
    expressions = (
        *(anchor.source.components if prologue else ()),
        *(expr for value in outputs for expr in value.components),
    )
    return tuple(
        expr for expr in _anchor_leaves(expressions) if expr.data[0] != ordinal
    )


@lru_cache(maxsize=128)
def _source(
    layouts: tuple,
    anchors: tuple[Anchor, ...],
    anchor_index: int,
    start: int,
    end: int,
    *,
    tiled: bool,
    prologue: bool,
    epilogue: bool,
    outputs: tuple[ValueExpr, ...],
) -> str:
    graph = BoundGraph(layouts, outputs, {}, anchors)
    anchor = graph.anchors[anchor_index]
    rows, n = anchor.source.shape[-2:]
    batches = prod(anchor.source.batch_shape)
    stage = StageEmitter(
        n=n, start=start, end=end, inverse=_inverse(anchor), tiled=tiled
    )
    leaves = _endpoint_leaves(anchor, anchor_index, prologue, outputs)
    prior_ids = tuple(sorted({expr.data[0] for expr in leaves}))
    arguments = [
        *(f"x{i}" for i in range(len(graph.layouts))),
        *(f"a{i}" for i in prior_ids),
        "scratch",
        *(f"out{i}" for i in range(len(outputs))),
        "twiddles",
        "rns_parameters",
        "PARAM_STRIDE: tl.constexpr",
        "TWIDDLE_STRIDE: tl.constexpr",
        "RADIX: tl.constexpr",
        "N: tl.constexpr",
        "BLOCK: tl.constexpr",
    ]
    lines = [
        "row = tl.program_id(1).to(tl.int64)",
        "batch = tl.program_id(2).to(tl.int64)",
        "q = (tl.load(rns_parameters + row).to(tl.int64) >> 1)",
        "k = tl.load(rns_parameters + 3 * PARAM_STRIDE + row).to(tl.int64) | (tl.load(rns_parameters + 4 * PARAM_STRIDE + row).to(tl.int64) << (RADIX // 2))",
        "r2 = tl.load(rns_parameters + 5 * PARAM_STRIDE + row).to(tl.int64)",
        "ninv = tl.load(rns_parameters + 7 * PARAM_STRIDE + row).to(tl.int64)",
    ]
    indices = stage.indices()
    lines.extend(stage.lines)
    stage.lines.clear()
    emitter = Emitter(graph.layouts)
    cursor = 0

    def emit(expr: Expr, index: str, overrides: dict[Expr, str]) -> str:
        nonlocal cursor
        result = emitter.emit(expr, index, "row", "batch", overrides)
        lines.extend(emitter.lines[cursor:])
        cursor = len(emitter.lines)
        return result

    def prior_overrides(index: str) -> dict[Expr, str]:
        result: dict[Expr, str] = {}
        for expr in leaves:
            ordinal, component = expr.data
            count = prod(graph.anchors[ordinal].result.batch_shape)
            name = stage.name("prior")
            lines.append(
                f"{name} = tl.load(a{ordinal} + (({component * count} + batch) * {rows} + row) * N + {index}, mask=mask, other=0).to(tl.int64)"
            )
            result[expr] = name
        return result

    loaded: list[tuple[str, ...]] = []
    for component, expr in enumerate(anchor.source.components):
        values = []
        for index in indices:
            if prologue:
                value = emit(expr, index, prior_overrides(index))
                if isinstance(
                    anchor.operation, ntt.CoefficientStandardToNttMontgomeryOp
                ):
                    converted = stage.name("montgomery")
                    lines.append(
                        f"{converted} = montgomery_mul({value}, r2, q, k, RADIX)"
                    )
                    value = converted
            else:
                value = stage.name("input")
                lines.append(
                    f"{value} = tl.load(scratch + (({component * batches} + batch) * {rows} + row) * N + {index}, mask=mask, other=0).to(tl.int64)"
                )
            values.append(value)
        loaded.append(tuple(values))
    transformed = []
    for values in loaded:
        transformed.append(stage.transform(values))
        lines.extend(stage.lines)
        stage.lines.clear()
    if not epilogue:
        for component, values in enumerate(transformed):
            for index, value in zip(indices, values, strict=True):
                lines.append(
                    f"tl.store(scratch + (({component * batches} + batch) * {rows} + row) * N + {index}, {value}, mask=mask)"
                )
    else:
        # A fresh expression cache prevents prologue values from bypassing the
        # transform when the same expression appears in both endpoint graphs.
        emitter = Emitter(graph.layouts)
        cursor = 0
        for lane, index in enumerate(indices):
            overrides = prior_overrides(index)
            for component, values in enumerate(transformed):
                value = values[lane]
                if _inverse(anchor):
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
                overrides[anchor.result.components[component]] = value
            for ordinal, output in enumerate(outputs):
                for component, expr in enumerate(output.components):
                    result = emit(expr, index, overrides)
                    lines.append(
                        f"tl.store(out{ordinal} + (({component * batches} + batch) * {rows} + row) * N + {index}, {result}, mask=mask)"
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


def _intervals(
    log_n: int, group: int, inverse: bool
) -> tuple[tuple[int, int, bool], ...]:
    if log_n <= 8:
        return ((0, log_n, True),)
    if inverse:
        last = max(8, log_n - group)
        return ((0, 8, True), (8, last, False), (last, log_n, False))
    last = log_n - 8
    first = min(group, last)
    return ((0, first, False), (first, last, False), (last, log_n, True))


@dataclass(frozen=True)
class _Stage:
    kernel: Any
    grid: tuple[int, int, int]
    prior_ids: tuple[int, ...]
    epilogue: bool
    start: int
    end: int


@dataclass(frozen=True)
class _Transform:
    shape: tuple[int, ...]
    inverse: bool
    group: int
    table_indices: tuple[int, int]
    constants: tuple[int, ...]
    stages: tuple[_Stage, ...]
    release_after: tuple[int, ...]


@dataclass(frozen=True)
class CompactNttExecution:
    """Prepared compact radix-2 execution for one input layout.

    Preparation records table operand positions and emits kernels. Execution accepts current
    Tensor operands, allocates result/scratch storage and launches those kernels;
    no input or evaluation-key Tensor is retained by the plan.
    """

    output_roots: tuple[ValueExpr, ...]
    outputs: tuple[ValueExpr, ...]
    transforms: tuple[_Transform, ...]
    dtype: torch.dtype
    device: torch.device

    _call: Callable = field(init=False, repr=False, compare=False)
    execution_source: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        namespace = {
            "empty": torch.empty,
            "dtype": self.dtype,
            "device": self.device,
            "device_scope": torch.cuda.device,
            "current_stream": torch.cuda.current_stream,
            "native_stage": torch.ops.fhelium_ntt_ops.compact_ntt_stage_range_,
        }
        lines = ["def execute(values, tables):"]
        for value in self.output_roots:
            lines.append(
                f"    output_{value.root} = empty({value.shape!r}, dtype=dtype, device=device)"
            )
        lines.extend(
            (
                "    with device_scope(device):",
                "        stream = current_stream(device).cuda_stream",
            )
        )
        outputs = tuple(f"output_{value.root}" for value in self.output_roots)
        for ordinal, transform in enumerate(self.transforms):
            scratch = f"scratch_{ordinal}"
            lines.append(
                f"        {scratch} = empty({transform.shape!r}, dtype=dtype, device=device)"
            )
            parameters, twiddles = (
                f"tables[{index}]" for index in transform.table_indices
            )
            destinations = (
                outputs if ordinal == len(self.transforms) - 1 else (scratch,)
            )
            for index, stage in enumerate(transform.stages):
                if stage.kernel is None:
                    lines.append(
                        f"        native_stage({scratch}, {twiddles}, {parameters}, {transform.inverse!r}, {stage.start}, {stage.end}, {transform.group})"
                    )
                else:
                    symbol = f"launch_{ordinal}_{index}"
                    namespace[symbol] = PreparedKernel(stage.kernel, stage.grid)
                    arguments = [
                        "*values",
                        *(f"scratch_{prior}" for prior in stage.prior_ids),
                        scratch,
                    ]
                    if stage.epilogue:
                        arguments.extend(destinations)
                    arguments.extend(
                        (
                            twiddles,
                            parameters,
                            *(repr(c) for c in transform.constants),
                            "stream=stream",
                        )
                    )
                    lines.append(f"        {symbol}({', '.join(arguments)})")
            for expired in transform.release_after:
                lines.append(f"        del scratch_{expired}")
        results = []
        for value in self.outputs:
            expression = (
                f"values[{-value.root - 1}]"
                if value.root < 0
                else f"output_{value.root}"
            )
            for index in value.view:
                name = f"view_{len(namespace)}"
                namespace[name] = index
                expression += f"[{name}]"
            results.append(expression)
        lines.append(
            "    return ("
            + ", ".join(results)
            + ("," if len(results) == 1 else "")
            + ")"
        )
        source = "\n".join(lines) + "\n"
        filename = f"<fhelium-compact-ntt-{hashlib.sha256(source.encode()).hexdigest()}>"
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
    ) -> CompactNttExecution:
        if not graph.anchors:
            raise ValueError(
                "An NTT fusion schedule requires a transform anchor"
            )
        first = values[0]
        if first.device.type != "cuda" or first.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "Triton NTT fusion requires CUDA int32/int64 operands"
            )
        if any(
            value.device != first.device or value.dtype != first.dtype
            for value in values
        ):
            raise ValueError("Triton NTT operands must share dtype and device")
        root = graph.anchors[0].source
        rows, n = root.shape[-2:]
        batches = prod(root.batch_shape)
        root_ids = tuple(
            dict.fromkeys(
                value.root for value in graph.outputs if value.root >= 0
            )
        )
        materialized = tuple(graph.roots[root_id] for root_id in root_ids)
        check_ntt_layouts(
            [
                (value.shape, value.component_axis)
                for value in (
                    *[anchor.source for anchor in graph.anchors],
                    *materialized,
                )
            ]
        )
        transforms: list[_Transform] = []
        retained: set[int] = set()
        for ordinal, anchor in enumerate(graph.anchors):
            parameters, twiddles = (
                tables[index] for index in anchor.table_indices
            )
            policy_name = getattr(
                anchor.operation.attributes.get("ntt_backend"), "data", ""
            )
            policy = resolve_ntt_backend_policy(policy_name)
            if not isinstance(policy, CompactRadix2Policy):
                raise ValueError(
                    "Selected Triton NTT fusion requires compact radix-2 tables"
                )
            if any(
                t.device != first.device or t.dtype != first.dtype
                for t in (parameters, twiddles)
            ):
                raise ValueError(
                    "NTT tables must share operand dtype and device"
                )
            if parameters.shape != (8, rows) or twiddles.shape != (rows, n):
                raise ValueError(
                    "NTT parameter or twiddle extent differs from its operands"
                )
            inverse = _inverse(anchor)
            group = policy.grouped_radix2_stage_count
            intervals = _intervals(n.bit_length() - 1, group, inverse)
            final_anchor = ordinal == len(graph.anchors) - 1
            destination_exprs = (
                materialized if final_anchor else (anchor.result,)
            )
            stages = []
            block = 128
            for part, (start, end, tiled) in enumerate(intervals):
                if start == end:
                    continue
                prologue = part == 0
                epilogue = part == len(intervals) - 1
                if not prologue and not epilogue:
                    stages.append(
                        _Stage(None, (0, 0, 0), (), False, start, end)
                    )
                    continue
                source = _source(
                    graph.layouts,
                    graph.anchors,
                    ordinal,
                    start,
                    end,
                    tiled=tiled,
                    prologue=prologue,
                    epilogue=epilogue,
                    outputs=destination_exprs if epilogue else (),
                )
                width = 1 << (end - start)
                grid = (
                    n // width if tiled else (n // width + block - 1) // block,
                    rows,
                    batches,
                )
                leaves = _endpoint_leaves(
                    anchor,
                    ordinal,
                    prologue,
                    destination_exprs if epilogue else (),
                )
                prior_ids = tuple(sorted({expr.data[0] for expr in leaves}))
                stages.append(
                    _Stage(
                        _kernel(source), grid, prior_ids, epilogue, start, end
                    )
                )
            if not final_anchor:
                retained.add(ordinal)
            future_expressions = (
                *(
                    expr
                    for later in graph.anchors[ordinal + 1 :]
                    for expr in later.source.components
                ),
                *(expr for value in materialized for expr in value.components),
            )
            live = {expr.data[0] for expr in _anchor_leaves(future_expressions)}
            release_after = tuple(sorted(retained - live))
            retained.intersection_update(live)
            transforms.append(
                _Transform(
                    anchor.result.shape,
                    inverse,
                    group,
                    (anchor.table_indices[0], anchor.table_indices[1]),
                    (
                        parameters.stride(0),
                        twiddles.stride(0),
                        32 if first.dtype == torch.int32 else 62,
                        n,
                        block,
                    ),
                    tuple(stages),
                    release_after,
                )
            )
        return cls(
            materialized,
            graph.outputs,
            tuple(transforms),
            first.dtype,
            first.device,
        )

    def execute(
        self, values: tuple[torch.Tensor, ...], tables: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Execute the prepared compact schedule using fresh result and scratch storage."""
        return self._call(values, tables)


__all__ = ["CompactNttExecution"]
