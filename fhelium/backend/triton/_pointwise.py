"""Generate and execute component-aware RNS expression kernels."""

from __future__ import annotations

import hashlib
import linecache
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING, Any

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks, fusion, rns

from ._expressions import (
    ARITHMETIC_OPS,
    BoundGraph,
    Emitter,
    Expr,
    ExpressionGraph,
    InputLayout,
    ValueExpr,
    arithmetic_value,
    component_axis,
    compact_value,
)
from ._launch import PreparedKernel

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=128)
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
            )
        },
    }
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    function = namespace["fused"]
    return triton.jit(
        do_not_specialize_on_alignment=[
            name
            for name in function.__code__.co_varnames[
                : function.__code__.co_argcount
            ]
            if function.__annotations__.get(name) is not tl.constexpr
        ],
    )(function)


@dataclass(frozen=True)
class PointwisePlan:
    graph: BoundGraph
    materialized: tuple[ValueExpr, ...]
    kernel: Any
    length: int

    @classmethod
    def build(cls, graph: BoundGraph) -> PointwisePlan:
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
        parameters = [
            *(f"x{i}" for i in range(len(graph.layouts))),
            *(f"out{i}" for i in range(len(materialized))),
            "rns_parameters",
            "PSTRIDE: tl.constexpr",
            "RADIX: tl.constexpr",
            "BLOCK: tl.constexpr",
        ]
        lines = [
            "def fused(" + ", ".join(parameters) + "):",
            "    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
            f"    mask = offset < {length}",
            f"    index = offset % {n}",
            f"    row = (offset // {n}) % {rows}",
            f"    batch = offset // {rows * n}",
            "    q = tl.load(rns_parameters + row, mask=mask, other=2).to(tl.int64) // 2",
            "    klo = tl.load(rns_parameters + 3 * PSTRIDE + row, mask=mask, other=0).to(tl.uint64)",
            "    khi = tl.load(rns_parameters + 4 * PSTRIDE + row, mask=mask, other=0).to(tl.uint64)",
            "    k = klo | (khi << (RADIX // 2))",
            "    r2 = tl.load(rns_parameters + 5 * PSTRIDE + row, mask=mask, other=0).to(tl.int64)",
        ]
        emitter = Emitter(graph.layouts)
        stores: list[str] = []
        for number, value in enumerate(materialized):
            count = prod(value.batch_shape) * rows * n
            output_mask = f"(offset < {count})"
            for component, expr in enumerate(value.components):
                result = emitter.emit(
                    expr, "index", "row", "batch", mask=output_mask
                )
                stores.append(
                    f"tl.store(out{number} + offset + {component * count}, {result}, mask={output_mask})"
                )
        lines.extend("    " + line for line in (*emitter.lines, *stores))
        return cls(
            graph,
            materialized,
            PreparedKernel(
                compile_source("\n".join(lines) + "\n"),
                ((length + 255) // 256,),
            ),
            length,
        )

    def execute(
        self,
        inputs: tuple[torch.Tensor, ...],
        parameters: torch.Tensor,
        radix: int,
    ) -> tuple[torch.Tensor, ...]:
        storage = {
            v.root: torch.empty(
                v.shape, dtype=inputs[0].dtype, device=inputs[0].device
            )
            for v in self.materialized
        }
        if self.length:
            with torch.cuda.device(inputs[0].device):
                self.kernel(
                    *inputs,
                    *storage.values(),
                    parameters,
                    parameters.stride(0),
                    radix,
                    256,
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


@dataclass
class PreparedFusion:
    graph: ExpressionGraph
    name: str = "triton-rns-fused"
    operation_types: tuple[type[Operation], ...] = (fusion.FusedOp,)
    supports_in_place: bool = False
    _last_match: Any = field(default=None, repr=False)
    _last_call: Any = field(default=None, repr=False)
    _fixed: bool = False

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def _prepare(self, all_inputs: tuple[torch.Tensor, ...]):
        from ._ntt import CompactNttExecution

        data_indices, table_indices = self.graph.input_indices
        tables = tuple(all_inputs[index] for index in table_indices)
        inputs = tuple(all_inputs[index] for index in data_indices)
        rows = inputs[0].size(-2)
        key_views = tuple(self.graph.key_inputs)

        def payloads(values):
            data = tuple(values[index] for index in data_indices)
            if key_views:
                data += tuple(
                    values[table_indices[table]][
                        digit, :, start : start + rows, :
                    ]
                    for table, digit, start in key_views
                )
            return data

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
        layouts += tuple(
            InputLayout(tuple(value.shape), tuple(value.stride()), True)
            for value in inputs[len(data_indices) :]
        )
        graph = self.graph.bind(layouts)
        if graph.anchors:
            plan = CompactNttExecution.build(graph, inputs, tables)

            def run(values):
                return plan.execute(
                    payloads(values),
                    tuple(values[index] for index in table_indices),
                )
        else:
            parameters = tables[self.graph.parameter_index]
            # Reuse the selected implementation's native Tensor ABI check.
            parameters, radix = execution_parameters(
                None, inputs, parameters, in_place=False
            )
            pointwise = PointwisePlan.build(graph)
            parameter_index = table_indices[self.graph.parameter_index]

            def run(values):
                return pointwise.execute(
                    payloads(values), values[parameter_index], radix
                )

        namespace: dict[str, Any] = {}
        conditions = []
        for index, value in enumerate(all_inputs):
            namespace[f"shape{index}"] = tuple(value.shape)
            namespace[f"stride{index}"] = tuple(value.stride())
            namespace[f"dtype{index}"] = value.dtype
            namespace[f"device{index}"] = value.device
            conditions.append(
                f"values[{index}].shape == shape{index} and values[{index}].stride() == stride{index} and values[{index}].dtype == dtype{index} and values[{index}].device == device{index}"
            )
        source = (
            "def matches(values):\n    return "
            + " and ".join(conditions)
            + "\n"
        )
        exec(compile(source, "<fhelium-fusion-layout>", "exec"), namespace)
        self._last_match = namespace["matches"]
        self._last_call = run

    def prepare_layout(self, operation: fusion.FusedOp) -> None:
        """Prepare known physical layouts without allocating numerical storage."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

        descriptions = []
        for value in operation.inputs:
            state = getattr(getattr(value.type, "state", None), "data", {})
            shape, strides = state.get("shape"), state.get("strides")
            dtype, device = state.get("dtype"), state.get("device")
            if (
                not isinstance(shape, ArrayAttr)
                or not isinstance(strides, ArrayAttr)
                or not all(
                    isinstance(x, IntegerAttr) and x.value.data >= 0
                    for x in (*shape, *strides)
                )
                or not isinstance(dtype, StringAttr)
                or not isinstance(device, StringAttr)
                or device.data == "unknown"
            ):
                return
            descriptions.append(
                (
                    tuple(int(x.value.data) for x in shape),
                    tuple(int(x.value.data) for x in strides),
                    getattr(torch, dtype.data.removeprefix("torch.")),
                    torch.device(device.data),
                )
            )
        with FakeTensorMode():
            values = tuple(
                torch.empty_strided(shape, strides, dtype=dtype, device=device)
                for shape, strides, dtype, device in descriptions
            )
            self._prepare(values)
        self._fixed = True

    def execute(self, invocation, inputs, resources, *, in_place):
        if in_place:
            raise ValueError("Triton fusion uses out-of-place execution")
        if self._last_match is None or not self._last_match(inputs):
            if self._fixed:
                raise ValueError(
                    "Fusion operands differ from the prepared layout/dtype/device"
                )
            self._prepare(inputs)
        return self._last_call(inputs)


@dataclass(frozen=True)
class TritonFusionImplementation:
    """Generate indexed integer expression kernels from visible fusion regions.

    Component extraction and packing become expression selection and output
    indexing. Arithmetic chains retain intermediates in registers and preserve
    every external result and identity-view alias. ``include_ntt`` admits compact
    radix-2 transform anchors; implementation names do not determine support.
    Match checks consume known Program facts, while missing layout/resource facts
    are checked when binding execution. Device compilation is lazy.
    """

    name: str = "triton-rns-fused"
    include_ntt: bool = False
    operation_types: tuple[type[Operation], ...] = (fusion.FusedOp,)
    supports_in_place: bool = False

    def select_ntt_schedule(self, **facts):
        """Offer compact schedules when this region implementation admits NTT."""
        from fhelium.backend.ntt._selection import (
            select_policy,
            table_count,
            table_layout,
        )

        if not self.include_ntt or facts.get("algorithm") not in (
            None,
            "radix2_compact",
        ):
            return None
        if (
            facts.get("device_type") not in (None, "cuda")
            or facts.get("radix") is not None
        ):
            return None
        if not any(
            facts.get(name) is not None
            for name in (
                "selected",
                "algorithm",
                "tables",
                "group_width",
                "device_type",
            )
        ):
            return None
        try:
            policy = select_policy(**{**facts, "algorithm": "radix2_compact"})
        except ValueError:
            return None
        if policy is None:
            return None
        return {
            "ntt_backend": policy.name,
            "ntt_table_layout": table_layout(policy),
            "table_count": table_count(policy),
        }

    def match_fusion(self, operations: Sequence[Operation]) -> int | None:
        """Match Program facts against this implementation's RNS/NTT support."""
        from ._matching import match_region

        return match_region(operations, include_ntt=self.include_ntt)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def prepare_operation(self, operation: Operation) -> PreparedFusion:
        if not isinstance(operation, fusion.FusedOp):
            raise TypeError(
                "Triton fusion preparation requires a fusion region"
            )
        if (
            self.match_fusion(
                tuple(
                    op
                    for op in operation.body.block.ops
                    if not isinstance(op, fusion.YieldOp)
                )
            )
            is None
        ):
            raise ValueError(
                "The selected fusion implementation does not support this region"
            )
        prepared = PreparedFusion(ExpressionGraph(operation), name=self.name)
        prepared.prepare_layout(operation)
        return prepared

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        /,
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        raise RuntimeError("Resolve the fusion operation before execution")


@dataclass
class TritonRnsImplementation:
    """Execute individual modular arithmetic operations using the expression generator.

    Inputs use the native lazy interval [0,2q); Montgomery products preserve the
    native R32 canonical and R62 lazy representatives. Selection requires CUDA.
    """

    name: str = "triton-rns-pointwise"
    operation_types: tuple[type[Operation], ...] = ARITHMETIC_OPS
    supports_in_place: bool = False
    plans: dict[tuple[object, ...], PointwisePlan] = field(default_factory=dict)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        /,
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        inputs, parameters = inputs[:-1], inputs[-1]
        parameters, radix = execution_parameters(
            invocation, inputs, parameters, in_place=in_place
        )
        plaintext = invocation.operation_type in (
            rns.MultiplyPlaintextOp,
            rns.AddPlaintextOp,
            ckks.AddCompressedPlaintextOp,
            ckks.MultiplyCompressedPlaintextOp,
        )
        layouts = tuple(
            InputLayout(tuple(v.shape), tuple(v.stride()), plaintext and i == 0)
            for i, v in enumerate(inputs)
        )
        ntt_plaintext = invocation.attributes.get("polynomial_domain") == "ntt"
        compression = invocation.attributes.get("compression_layout")
        key = (invocation.operation_type, layouts, ntt_plaintext, compression)
        plan = self.plans.get(key)
        if plan is None:
            values = [
                ValueExpr(
                    tuple(
                        Expr("input", data=(i, c))
                        for c in range(layout.components)
                    ),
                    layout.shape,
                    layout.component_axis,
                    -i - 1,
                )
                for i, layout in enumerate(layouts)
            ]
            output = (
                compact_value(
                    invocation.operation_type,
                    values,
                    0,
                    layout=compression,
                    ntt_plaintext=ntt_plaintext,
                )
                if compression is not None
                else arithmetic_value(
                    invocation.operation_type,
                    values,
                    0,
                    ntt_plaintext=ntt_plaintext,
                )
            )
            roots = {v.root: v for v in (*values, output)}
            plan = PointwisePlan.build(BoundGraph(layouts, (output,), roots))
            self.plans[key] = plan
        return plan.execute(inputs, parameters, radix)
