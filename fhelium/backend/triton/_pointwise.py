"""Generate and execute component-aware RNS expression kernels."""

from __future__ import annotations

import hashlib
import linecache
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING, Any

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import FusionMatcher, OperationInvocation
from fhelium.backend.lowering import ExecBuilder
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir._structure import structural_key
from fhelium.ir.dialects import fusion

from ._expressions import (
    BoundGraph,
    Emitter,
    ExpressionGraph,
    InputLayout,
    KernelArguments,
    ValueExpr,
    component_axis,
    material_view_owners,
)
from ._budget import (
    DEFAULT_KERNEL_BUDGET,
    DEFAULT_POINTWISE_BUDGET,
    KernelBudget,
    indivisible_kernel,
    within_budget,
)
from ._launch import PreparedKernel
from ._layout import polynomial_shape
from ._ntt_codegen import _parameter_policy

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=1024)
def compile_source(source: str) -> Any:
    """Compile generated Python to a Triton function, retaining inspectable source."""
    import triton
    import triton.language as tl

    from . import _arithmetic

    filename = f"<fhelium-triton-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )
    namespace: dict[str, Any] = {
        "tl": tl,
        "__name__": __name__,
        **{
            name: getattr(_arithmetic, name)
            for name in (
                "montgomery_mul",
                "montgomery_reduce",
                "add_lazy",
                "subtract_lazy",
                "canonicalize",
                "seeded_key_residue",
                "seeded_key_residue8",
            )
        },
    }
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    function = namespace.pop("fused")
    alignment_independent, scalars = _parameter_policy(function)
    return triton.jit(
        do_not_specialize=scalars,
        do_not_specialize_on_alignment=alignment_independent,
    )(function)


@dataclass(frozen=True)
class _Value:
    """Result root, extent and view of a restored plan output."""

    root: int
    shape: tuple[int, ...]
    view: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Outputs:
    outputs: tuple[_Value, ...]


def _restore_pointwise(outputs, materialized, *fields):
    return PointwisePlan(_Outputs(outputs), materialized, *fields)


@dataclass(frozen=True)
class PointwisePlan:
    """Execute complete output reductions using bounded generated kernels.

    ``inputs`` lists the region input numbers a leaf kernel reads in parameter
    order; ``scalars`` are the runtime layout arguments appended after the
    compile constants. Extents and strides are non-specialized int64 arguments.
    """

    graph: BoundGraph | _Outputs
    materialized: tuple[ValueExpr | _Value, ...]
    kernel: PreparedKernel | None
    length: int
    inputs: tuple[int, ...]
    scalars: tuple[int, ...]
    dtype: torch.dtype = torch.int64
    parts: tuple[PointwisePlan, ...] = ()
    scalar_names: tuple[str, ...] = ()
    seed_inputs: frozenset[int] = frozenset()

    @classmethod
    def build(
        cls,
        graph: BoundGraph,
        *,
        dtype: torch.dtype = torch.int64,
        kernel_budget: KernelBudget = DEFAULT_POINTWISE_BUDGET,
    ) -> PointwisePlan:
        if graph.anchors:
            raise ValueError(
                "NTT anchors require the transform execution implementation"
            )
        root_ids = tuple(
            dict.fromkeys(v.root for v in graph.outputs if v.root >= 0)
        )
        materialized = tuple(graph.roots[i] for i in root_ids)
        rows, n = (
            materialized[0].shape if materialized else graph.outputs[0].shape
        )[-2:]
        length = max(
            (prod(v.batch_shape) * rows * n for v in materialized), default=0
        )
        names = KernelArguments(graph.layouts)
        length_name = names.scalar("length", length)
        rows_name = names.scalar("rows", rows)
        plane = names.scalar("plane", rows * n)
        lines = [
            "offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
            f"mask = offset < {length_name}",
            f"index = offset % {n}",
            f"row = (offset // {n}) % {rows_name}",
            f"batch = offset // {plane}",
            "q = tl.load(rns_parameters + row * PROW_STRIDE, mask=mask, other=2).to(tl.int64) // 2",
            "klo = tl.load(rns_parameters + 3 * PSTRIDE + row * PROW_STRIDE, mask=mask, other=0).to(tl.uint64)",
            "khi = tl.load(rns_parameters + 4 * PSTRIDE + row * PROW_STRIDE, mask=mask, other=0).to(tl.uint64)",
            "k = klo | (khi << (RADIX // 2))",
            "r2 = tl.load(rns_parameters + 5 * PSTRIDE + row * PROW_STRIDE, mask=mask, other=0).to(tl.int64)",
        ]
        emitter = Emitter(graph.layouts, names, segment_index=True)
        stores: list[str] = []
        for number, value in enumerate(materialized):
            count = prod(value.batch_shape) * rows * n
            count_name = names.scalar(f"count{number}", count)
            output_mask = f"(offset < {count_name})"
            stride_name = (
                names.scalar(f"component_stride{number}", count)
                if len(value.components) > 1
                else count_name
            )
            for component, expr in enumerate(value.components):
                result = emitter.emit(
                    expr, "index", "row", "batch", mask=output_mask
                )
                component_offset = (
                    f" + {component} * {stride_name}" if component else ""
                )
                stores.append(
                    f"tl.store(out{number} + offset{component_offset}, {result}, mask={output_mask})"
                )
        parameters = [
            *names.pointer_parameters,
            *(f"out{i}" for i in range(len(materialized))),
            "rns_parameters",
            "PSTRIDE: tl.int64",
            "PROW_STRIDE: tl.int64",
            "RADIX: tl.constexpr",
            "BLOCK: tl.constexpr",
            *names.scalar_parameters,
        ]
        source = (
            "def fused("
            + ", ".join(parameters)
            + "):\n"
            + "".join(
                "    " + line + "\n"
                for line in (*lines, *emitter.lines, *stores)
            )
        )
        if not within_budget(source, kernel_budget):
            if len(materialized) < 2:
                raise indivisible_kernel(source, kernel_budget)
            middle = len(materialized) // 2
            parts = tuple(
                cls.build(
                    replace(graph, outputs=outputs),
                    dtype=dtype,
                    kernel_budget=kernel_budget,
                )
                for outputs in (materialized[:middle], materialized[middle:])
            )
            return cls(graph, materialized, None, length, (), (), dtype, parts)
        return cls(
            graph,
            materialized,
            PreparedKernel(
                compile_source(source),
                ((length + 255) // 256,),
                source=(compile_source, source),
            ),
            length,
            tuple(names.inputs),
            names.scalar_values,
            dtype,
            scalar_names=tuple(name for name, _, _ in names.scalars),
            seed_inputs=graph.seed_inputs,
        )

    def __reduce__(self):
        return (
            _restore_pointwise,
            (
                tuple(
                    _Value(v.root, v.shape, v.view) for v in self.graph.outputs
                ),
                tuple(_Value(v.root, v.shape) for v in self.materialized),
                self.kernel,
                self.length,
                self.inputs,
                self.scalars,
                self.dtype,
                self.parts,
                self.scalar_names,
                self.seed_inputs,
            ),
        )

    @property
    def kernel_count(self) -> int:
        return (
            sum(part.kernel_count for part in self.parts)
            if self.parts
            else int(bool(self.length))
        )

    def execute(
        self,
        inputs: tuple[torch.Tensor, ...],
        parameters: torch.Tensor,
        radix: int,
        out: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Evaluate the outputs; ``out`` maps output roots to caller storage.

        Caller storage has the output shape with contiguous batch, row and
        coefficient axes; its leading component axis may have any stride.
        """
        out = out or {}
        if self.parts:
            storage = {
                value.root: tensor
                for part in self.parts
                for value, tensor in zip(
                    part.graph.outputs,
                    part.execute(inputs, parameters, radix, out),
                    strict=True,
                )
            }
        else:
            storage = {
                v.root: out[v.root]
                if v.root in out
                else torch.empty(
                    v.shape, dtype=inputs[0].dtype, device=inputs[0].device
                )
                for v in self.materialized
            }
        if self.length and self.kernel is not None:
            scalars = self.scalars
            if out and len(self.scalar_names) == len(scalars):
                strides = {
                    f"component_stride{number}": storage[v.root].stride(0)
                    for number, v in enumerate(self.materialized)
                    if v.root in out
                }
                scalars = tuple(
                    strides.get(name, value)
                    for name, value in zip(self.scalar_names, scalars)
                )
            with torch.cuda.device(inputs[0].device):
                self.kernel(
                    *(inputs[i] for i in self.inputs),
                    *storage.values(),
                    parameters,
                    parameters.stride(0),
                    parameters.stride(1),
                    radix,
                    256,
                    *scalars,
                    stream=torch.cuda.current_stream(
                        inputs[0].device
                    ).cuda_stream,
                )
        outputs: list[torch.Tensor] = []
        for value in self.graph.outputs:
            tensor = (
                inputs[-value.root - 1]
                if value.root < 0
                else storage[value.root]
            )
            for index in value.view:
                tensor = tensor[index]
            outputs.append(tensor)
        return tuple(outputs)


def execution_parameters(
    invocation: OperationInvocation | None,
    inputs: tuple[torch.Tensor, ...],
    parameters: torch.Tensor,
    *,
    in_place: bool,
) -> tuple[torch.Tensor, int]:
    if in_place:
        raise ValueError("Triton fusion uses out-of-place execution")
    if not inputs or inputs[0].device.type != "cuda":
        raise ValueError(
            "The selected Triton RNS implementation requires CUDA tensors"
        )
    first = inputs[0]
    if first.dtype not in (torch.int32, torch.int64) or first.ndim < 2:
        raise ValueError(
            "Triton RNS tensors require int32/int64 [..., prime, coefficient] layout"
        )
    if any(
        v.device != first.device
        or v.dtype != first.dtype
        or v.ndim < 2
        or v.shape[-2] != first.shape[-2]
        for v in inputs
    ):
        raise ValueError(
            "Fusion tensors require matching device, dtype and prime extents"
        )
    if (
        parameters.device != first.device
        or parameters.dtype != first.dtype
        or parameters.ndim != 2
        or parameters.size(0) != 8
        or parameters.size(1) != first.size(-2)
    ):
        raise ValueError(
            "Triton RNS parameters require an [8, prime] Tensor matching operand dtype and device"
        )
    return parameters, 32 if first.dtype == torch.int32 else 62


_InputDescription = tuple[
    tuple[int, ...], tuple[int, ...], torch.dtype, torch.device
]
"""Extents, strides, dtype and device of one fusion input."""


def declared_layouts(
    operation: fusion.FusedOp,
) -> tuple[tuple[_InputDescription, ...], bool] | None:
    """Read input extents, strides, dtype and device from fusion input types.

    Returns ``(descriptions, fixed)``, or None when an extent, dtype or device is unknown. Missing shapes are reconstructed from polynomial state and missing strides are assumed contiguous; either makes the layout not fixed.
    """
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

    descriptions: list[_InputDescription] = []
    fixed = True
    for value in operation.inputs:
        state = getattr(getattr(value.type, "state", None), "data", {})
        shape, strides = state.get("shape"), state.get("strides")
        if shape is None:
            shape = polynomial_shape(value.type)
            fixed = False
        dtype, device = state.get("dtype"), state.get("device")
        if (
            not isinstance(shape, ArrayAttr)
            or not all(
                isinstance(x, IntegerAttr) and x.value.data >= 0 for x in shape
            )
            or not isinstance(dtype, StringAttr)
            or not isinstance(device, StringAttr)
            or device.data == "unknown"
        ):
            return None
        extents = tuple(int(x.value.data) for x in shape)
        if isinstance(strides, ArrayAttr) and all(
            isinstance(x, IntegerAttr) and x.value.data >= 0 for x in strides
        ):
            layout = tuple(int(x.value.data) for x in strides)
        else:
            fixed = False
            layout = tuple(
                prod(extents[axis + 1 :]) for axis in range(len(extents))
            )
        descriptions.append(
            (
                extents,
                layout,
                getattr(torch, dtype.data.removeprefix("torch.")),
                torch.device(device.data),
            )
        )
    return tuple(descriptions), fixed


def _layout_signature(values) -> tuple[object, ...]:
    return tuple(
        (tuple(v.shape), tuple(v.stride()), v.dtype, v.device) for v in values
    )


def _key_row_split(
    stored_rows: int, rows: int, start: int, p_start: int
) -> tuple[int, int] | None:
    """Return the ``(Q rows, unused rows)`` split of a two-slice key digit."""
    if p_start < 0:
        return None
    q_rows = rows - (stored_rows - p_start)
    gap = p_start - start - q_rows
    return (q_rows, gap) if gap else None


class _Payloads:
    """Select data operands and key-digit views from fusion operands."""

    __slots__ = ("data", "tables", "keys", "rows")

    def __init__(self, data, tables, keys, rows):
        self.data, self.tables, self.keys, self.rows = data, tables, keys, rows

    def __reduce__(self):
        return (_Payloads, (self.data, self.tables, self.keys, self.rows))

    def __call__(self, values):
        data = tuple(values[index] for index in self.data)
        if self.keys:
            # The view starts at the first active Q row; layouts with a row
            # split address the P rows past the unused stored rows. Seed rows
            # of seeded keys follow the key views.
            rows = self.rows
            data += tuple(
                values[self.tables[table]][digit, :, start : start + rows, :]
                for table, digit, start, *_ in self.keys
            )
            data += tuple(
                values[self.tables[seed]][digit]
                for _, digit, _, _, seed, _ in self.keys
                if seed >= 0
            )
        return data


class _NttRun:
    __slots__ = ("payloads", "plan")

    def __init__(self, payloads, plan):
        self.payloads, self.plan = payloads, plan

    def __reduce__(self):
        return (_NttRun, (self.payloads, self.plan))

    def __call__(self, values):
        return self.plan.execute(
            self.payloads(values),
            tuple(values[index] for index in self.payloads.tables),
        )


class _PointwiseRun:
    __slots__ = ("payloads", "plan", "parameter", "radix")

    def __init__(self, payloads, plan, parameter, radix):
        self.payloads, self.plan = payloads, plan
        self.parameter, self.radix = parameter, radix

    def __reduce__(self):
        return (
            _PointwiseRun,
            (self.payloads, self.plan, self.parameter, self.radix),
        )

    def __call__(self, values):
        return self.plan.execute(
            self.payloads(values), values[self.parameter], self.radix
        )


def _restore_fusion(state):
    prepared = object.__new__(PreparedFusion)
    prepared.__dict__.update(state, graph=None, _region=None)
    return prepared


@dataclass
class PreparedFusion:
    """Kernel plans of one fusion body, keyed by operand layout.

    One instance serves every dispatch of alpha-equivalent regions; each operand layout signature selects or builds its own plan.
    """

    graph: ExpressionGraph
    name: str = "triton-rns-fused"
    max_ntt_endpoints: int | None = None
    kernel_budget: KernelBudget = DEFAULT_KERNEL_BUDGET
    pointwise_kernel_budget: KernelBudget = DEFAULT_POINTWISE_BUDGET
    operation_types: tuple[type[Operation], ...] = (fusion.FusedOp,)
    supports_in_place: bool = False
    _plans: dict[tuple[object, ...], Any] = field(
        default_factory=dict, repr=False
    )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def __reduce__(self):
        # Lowered schedules bind the prepared kernels; plans are rebuilt on demand.
        state = dict(self.__dict__)
        state.pop("graph")
        state.pop("_region", None)
        state["_plans"] = {}
        return (_restore_fusion, (state,))

    def attach_region(self, linked: Callable[[], PreparedFusion]) -> None:
        """Obtain the region from ``linked`` when a restored fusion meets a new layout."""
        if self.graph is None:
            self._region = linked

    def _prepare(self, all_inputs: tuple[torch.Tensor, ...]):
        signature = _layout_signature(all_inputs)
        run = self._plans.get(signature)
        if run is None:
            run = self._build(all_inputs)
            self._plans[signature] = run
        return run

    def _build(self, all_inputs: tuple[torch.Tensor, ...]):
        from ._ntt import NttExecution

        data_indices, table_indices = self.graph.input_indices
        tables = tuple(all_inputs[index] for index in table_indices)
        inputs = tuple(all_inputs[index] for index in data_indices)
        rows = inputs[0].size(-2)
        key_views = tuple(self.graph.key_inputs)

        payloads = _Payloads(data_indices, table_indices, key_views, rows)
        inputs = payloads(all_inputs)
        layouts = tuple(
            InputLayout(
                tuple(value.shape), tuple(value.stride()), component_axis(typ)
            )
            for value, typ in zip(
                inputs[: len(data_indices)],
                self.graph.tensor_types,
                strict=True,
            )
        )
        key_count = len(key_views)
        layouts += tuple(
            InputLayout(
                tuple(value.shape),
                tuple(value.stride()),
                True,
                row_split=_key_row_split(
                    all_inputs[table_indices[table]].size(-2),
                    rows,
                    start,
                    p_start,
                ),
            )
            for value, (table, _, start, p_start, *_) in zip(
                inputs[len(data_indices) : len(data_indices) + key_count],
                key_views,
                strict=True,
            )
        )
        layouts += tuple(
            InputLayout(tuple(value.shape), tuple(value.stride()))
            for value in inputs[len(data_indices) + key_count :]
        )
        graph = self.graph.bind(layouts)
        if graph.anchors:
            limit = self.max_ntt_endpoints
            if limit is None and torch.version.hip:
                # Bound generated IR growth in the HIP coalescing compiler pass.
                limit = 8
            plan = NttExecution.build(
                graph,
                inputs,
                tables,
                max_endpoints=limit,
                kernel_budget=self.kernel_budget,
            )
            return _NttRun(payloads, plan)
        parameters = tables[self.graph.parameter_index]
        # Reuse the selected implementation's native Tensor ABI check.
        parameters, radix = execution_parameters(
            None, inputs, parameters, in_place=False
        )
        pointwise = PointwisePlan.build(
            graph,
            dtype=inputs[0].dtype,
            kernel_budget=self.pointwise_kernel_budget,
        )
        return _PointwiseRun(
            payloads,
            pointwise,
            table_indices[self.graph.parameter_index],
            radix,
        )

    def prepare_layout(self, operation: fusion.FusedOp) -> None:
        """Prepare kernels from declared Program layouts without numerical storage.

        Polynomial extents follow either the declared shape or the component,
        batch, prime-row and ring-dimension state; missing strides are assumed
        contiguous. Lowering prepares any other operand layout it meets.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        declared = declared_layouts(operation)
        if declared is None:
            return
        descriptions, _ = declared
        with FakeTensorMode():
            values = tuple(
                torch.empty_strided(shape, strides, dtype=dtype, device=device)
                for shape, strides, dtype, device in descriptions
            )
            self._prepare(values)

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        if invocation.in_place:
            raise ValueError("Triton fusion uses out-of-place execution")
        run = self._plans.get(_layout_signature(inputs))
        if run is None:
            with builder.suspended():
                if self.graph is None:
                    self.graph = self._region().graph
                run = self._prepare(inputs)
        return run(inputs)


@dataclass(frozen=True)
class TritonFusionImplementation:
    """Generate indexed integer expression kernels from visible fusion regions.

    Component extraction and packing become expression selection and output
    indexing. Arithmetic chains retain intermediates in registers and preserve
    every external result and identity-view alias. ``include_ntt`` admits transform
    dependencies; their execution plans compose generated endpoints, native stage
    ranges or complete transforms according to the selected algorithm.
    Match checks consume known Program facts, while missing layout/resource facts
    are checked when binding execution. Device compilation is lazy.

    ``max_ntt_endpoints`` bounds transform endpoints per generated kernel while
    preserving their stage order. ``None`` selects eight endpoints on HIP to
    limit compiler IR growth and leaves CUDA endpoint merging unbounded.

    ``max_ntt_anchors`` bounds the NTT transforms of one fusion region. An
    output that combines the results of many transforms, such as a weighted sum
    over rotated ciphertexts, is computed in the epilogue of its last transform
    and reads every earlier transform result there; above about eight results
    that epilogue's time per result grows with their number. Larger
    combinations continue in the next region from a stored partial result.
    ``None`` leaves regions unbounded.

    ``kernel_budget`` bounds static Montgomery calls, pointer arguments and
    Python source lines for NTT endpoint kernels. ``pointwise_kernel_budget``
    applies the same metrics to ordinary expression kernels. Complete output
    reductions or NTT endpoints are partitioned without changing arithmetic
    order. ``None`` disables the respective limit; a single oversized endpoint
    requires a larger budget or an application-selected decomposition.
    """

    name: str = "triton-rns-fused"
    include_ntt: bool = False
    max_ntt_endpoints: int | None = None
    max_ntt_anchors: int | None = 8
    kernel_budget: KernelBudget = DEFAULT_KERNEL_BUDGET
    pointwise_kernel_budget: KernelBudget = DEFAULT_POINTWISE_BUDGET
    operation_types: tuple[type[Operation], ...] = (fusion.FusedOp,)
    supports_in_place: bool = False

    def __post_init__(self):
        for name in ("max_ntt_endpoints", "max_ntt_anchors"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("kernel_budget", "pointwise_kernel_budget"):
            budget = getattr(self, name)
            if budget is not None and (
                len(budget) != 3 or any(v <= 0 for v in budget)
            ):
                raise ValueError(f"{name} requires three positive limits")

    def select_ntt_schedule(self, **facts):
        """Select an NTT schedule independently of endpoint decomposition."""
        from fhelium.backend.ntt._selection import (
            select_policy,
            table_count,
            table_layout,
        )

        if not self.include_ntt or facts.get("device_type") not in (
            None,
            "cuda",
        ):
            return None
        if not any(
            facts.get(name) is not None
            for name in (
                "selected",
                "algorithm",
                "tables",
                "radix",
                "device_type",
            )
        ):
            return None
        policy = select_policy(**facts)
        if policy is None:
            return None
        return {
            "ntt_backend": policy.name,
            "ntt_table_layout": table_layout(policy),
            "table_count": table_count(policy),
        }

    def new_fusion_matcher(self) -> FusionMatcher:
        """Create candidate analysis scoped to a read-only scheduling window."""
        from ._matching import RegionMatcher

        return RegionMatcher(
            include_ntt=self.include_ntt, max_anchors=self.max_ntt_anchors
        )

    def match_fusion(self, operations: Sequence[Operation]) -> int | None:
        """Match Program facts against this implementation's RNS/NTT support."""
        from ._matching import match_region

        count = match_region(
            operations,
            include_ntt=self.include_ntt,
            max_anchors=self.max_ntt_anchors,
        )
        if count and self.include_ntt:
            from ._ntt import _fusion_qualified

            if not _fusion_qualified(operations, self.max_ntt_endpoints):
                return None
        return count

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def preparation_key(self, operation: Operation) -> Hashable | None:
        """Return the key under which equal fusion regions share one preparation.

        Alpha-equivalent bodies with equal input types and material-view grouping generate the same kernels for each operand layout.
        """
        if not isinstance(operation, fusion.FusedOp):
            return None
        return (
            structural_key(operation),
            material_view_owners(
                operation, tuple(range(len(operation.inputs)))
            ),
        )

    def prepare_operation(self, operation: Operation) -> PreparedFusion:
        if not isinstance(operation, fusion.FusedOp):
            raise TypeError(
                "Triton fusion preparation requires a fusion region"
            )
        from ._matching import match_region

        if (
            match_region(
                tuple(
                    op
                    for op in operation.body.block.ops
                    if not isinstance(op, fusion.YieldOp)
                ),
                include_ntt=self.include_ntt,
                max_anchors=self.max_ntt_anchors,
            )
            is None
        ):
            raise ValueError(
                "The selected fusion implementation does not support this region"
            )
        prepared = PreparedFusion(
            ExpressionGraph(operation),
            name=self.name,
            max_ntt_endpoints=self.max_ntt_endpoints,
            kernel_budget=self.kernel_budget,
            pointwise_kernel_budget=self.pointwise_kernel_budget,
        )
        prepared.prepare_layout(operation)
        return prepared

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        raise RuntimeError("Resolve the fusion operation before lowering")
