"""Generate ordinary floating-point elementwise and cyclic-indexing kernels."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation, SSAValue

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import ResourceRequirement
from fhelium.backend.torch import decode_argument
from fhelium.ir import operation_dependencies
from fhelium.ir.dialects import core, fusion
from fhelium.ir.dialects.torch import TensorCallOp as CallOp

_SUPPORTED = {
    "torch.add",
    "torch.sub",
    "torch.mul",
    "torch.div",
    "torch.neg",
    "torch.roll",
}


def _facts(value: SSAValue):
    state = getattr(getattr(value.type, "state", None), "data", {})
    requires_grad = state.get("requires_grad")
    if isinstance(requires_grad, IntegerAttr) and requires_grad.value.data:
        return None
    shape = state.get("shape")
    if not isinstance(shape, ArrayAttr) or any(
        not isinstance(n, IntegerAttr) or n.value.data < 0 for n in shape
    ):
        return None
    dtype, device = state.get("dtype"), state.get("device")
    if (
        not isinstance(dtype, StringAttr)
        or dtype.data not in {"torch.float32", "torch.float64"}
        or not isinstance(device, StringAttr)
        or device.data == "unknown"
    ):
        return None
    strides = state.get("strides")
    return (
        tuple(int(n.value.data) for n in shape),
        dtype.data,
        device.data,
        tuple(int(n.value.data) for n in strides)
        if isinstance(strides, ArrayAttr)
        and all(isinstance(n, IntegerAttr) for n in strides)
        else None,
    )


def _remap(index: str, output: tuple[int, ...], source: tuple[int, ...]) -> str:
    if source == output:
        return index
    offset = len(output) - len(source)
    terms = []
    for axis, extent in enumerate(source):
        if extent != 1:
            terms.append(
                f"(({index}) // {prod(output[offset + axis + 1 :])} % {extent}) * {prod(source[axis + 1 :])}"
            )
    return " + ".join(terms) or "0"


@dataclass(frozen=True)
class TritonTensorFusionImplementation:
    """Fuse supported ordinary Tensor expressions without introducing CKKS.

    Automatic matching requires known CUDA layouts. Unknown placement remains
    unassigned; CPU execution and unsupported operations retain ordinary Torch
    implementations. Each output has separate storage, and floating-point
    contraction is disabled to retain the source operation rounding sequence.
    """

    name: str = "triton-tensor-fused"
    operation_types: tuple[type[Operation], ...] = (fusion.FusedOp,)
    supports_in_place: bool = False

    def match_fusion(self, operations: Sequence[Operation]) -> int | None:
        count = 0
        common = None
        for op in operations:
            if isinstance(op, core.MaterialRefOp):
                continue
            if (
                isinstance(op, UnrealizedConversionCastOp)
                and len(op.inputs) == len(op.outputs) == 1
            ):
                continue
            if (
                not isinstance(op, CallOp)
                or op.kind != StringAttr("function")
                or op.target is None
                or op.target.data not in _SUPPORTED
            ):
                return None
            dependencies = operation_dependencies(op)
            if any(
                dependencies.kind(0, index, "tensor_position")
                not in {"element", "reindexed"}
                for index in range(len(op.operands))
            ):
                return None
            descriptor = json.loads(
                cast(StringAttr, op.argument_descriptor).data
            )
            args = decode_argument(descriptor["args"], tuple(op.operands))
            kwargs = decode_argument(descriptor["kwargs"], tuple(op.operands))
            if op.target.data == "torch.roll":
                shift = kwargs.get("shifts", args[1] if len(args) > 1 else None)
                dim = kwargs.get("dims", args[2] if len(args) > 2 else None)
                if type(shift) is not int or dim != -1:
                    return None
            elif kwargs:
                return None
            for value in (*op.operands, *op.results):
                facts = _facts(value)
                if facts is None:
                    return None
                shape, dtype, device, strides = facts
                if torch.device(device).type != "cuda" or not all(shape):
                    return None
                identity = (dtype, torch.device(device))
                if common is not None and identity != common:
                    return None
                common = identity
                if strides is None or any(stride < 0 for stride in strides):
                    return None
            count += 1
        return count

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def prepare_operation(self, operation: Operation):
        if not isinstance(operation, fusion.FusedOp):
            raise TypeError("Tensor fusion requires a fusion region")
        body = tuple(
            op
            for op in operation.body.block.ops
            if not isinstance(op, fusion.YieldOp)
        )
        if self.match_fusion(body) is None:
            raise ValueError(
                "The selected Tensor fusion implementation does not support this region"
            )
        return _PreparedTensorFusion.build(operation, self.name)

    def execute(self, invocation, inputs, resources, *, in_place):
        raise RuntimeError("Prepare the Tensor fusion region before execution")


@dataclass
class _PreparedTensorFusion:
    name: str
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    kernel: Any
    length: int
    operation_types = (fusion.FusedOp,)
    supports_in_place = False

    @classmethod
    def build(cls, operation: fusion.FusedOp, name: str):
        from ._launch import PreparedKernel
        from ._pointwise import compile_source

        block = operation.body.block
        arguments: dict[SSAValue, int] = {
            value: index for index, value in enumerate(block.args)
        }
        terminator = block.last_op
        assert isinstance(terminator, fusion.YieldOp)
        outputs = tuple(terminator.operands)
        output_facts = tuple(cast(tuple, _facts(v)) for v in outputs)
        lines = []
        expressions = {}

        def emit(value, index, domain, mask):
            if not isinstance(value, SSAValue):
                return repr(value)
            facts = cast(tuple, _facts(value))
            shape = facts[0]
            index = _remap(index, domain, shape)
            key = (value, index, mask)
            if key in expressions:
                return expressions[key]
            if value in arguments:
                strides = facts[3]
                address = (
                    " + ".join(
                        f"(({index}) // {prod(shape[a + 1 :])} % {extent}) * {strides[a]}"
                        for a, extent in enumerate(shape)
                    )
                    or "0"
                )
                if address == "0":
                    address = "tl.full((BLOCK,), 0, tl.int64)"
                expression = f"tl.load(arg{arguments[value]} + ({address}), mask={mask}, other=0)"
            else:
                owner = value.owner
                if isinstance(owner, UnrealizedConversionCastOp):
                    return emit(owner.inputs[0], index, shape, mask)
                assert isinstance(owner, CallOp)
                descriptor = json.loads(
                    cast(StringAttr, owner.argument_descriptor).data
                )
                args = decode_argument(
                    descriptor["args"], tuple(owner.operands)
                )
                kwargs = decode_argument(
                    descriptor["kwargs"], tuple(owner.operands)
                )
                target = cast(StringAttr, owner.target).data
                if target == "torch.roll":
                    shift = (
                        kwargs.get("shifts", args[1] if len(args) > 1 else 0)
                        % shape[-1]
                    )
                    shifted = f"(({index}) // {shape[-1]} * {shape[-1]} + (({index}) % {shape[-1]} + {shape[-1] - shift}) % {shape[-1]})"
                    return emit(args[0], shifted, shape, mask)
                lhs = emit(args[0], index, shape, mask)
                if target == "torch.neg":
                    expression = f"-{lhs}"
                else:
                    rhs = emit(args[1], index, shape, mask)
                    sign = {
                        "torch.add": "+",
                        "torch.sub": "-",
                        "torch.mul": "*",
                        "torch.div": "/",
                    }[target]
                    expression = f"{lhs} {sign} {rhs}"
            variable = f"v{len(expressions)}"
            # Store each operation result in its represented dtype before reuse.
            dtype = facts[1].removeprefix("torch.")
            lines.append(f"{variable} = ({expression}).to(tl.{dtype})")
            expressions[key] = variable
            return variable

        stores = []
        for i, value in enumerate(outputs):
            shape = output_facts[i][0]
            mask = f"offset < {prod(shape)}"
            result = emit(value, "offset", shape, mask)
            stores.append(f"tl.store(out{i} + offset, {result}, mask={mask})")
        parameters = (
            [f"arg{i}" for i in range(len(arguments))]
            + [f"out{i}" for i in range(len(outputs))]
            + ["BLOCK: tl.constexpr"]
        )
        source = (
            "def fused("
            + ", ".join(parameters)
            + "):\n    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)\n"
            + "".join("    " + line + "\n" for line in (*lines, *stores))
        )
        return cls(
            name,
            tuple(_facts(v) for v in block.args),
            output_facts,
            PreparedKernel(
                compile_source(source),
                ((max(prod(f[0]) for f in output_facts) + 255) // 256,),
                enable_fp_fusion=False,
            ),
            max(prod(f[0]) for f in output_facts),
        )

    def resource_requirements(self, invocation):
        return ()

    def execute(self, invocation, inputs, resources, *, in_place):
        for tensor, facts in zip(inputs, self.inputs, strict=True):
            shape, dtype, device, strides = facts
            if (
                tensor.requires_grad
                or tuple(tensor.shape) != shape
                or tuple(tensor.stride()) != strides
                or str(tensor.dtype) != dtype
                or tensor.device != torch.device(device)
            ):
                raise ValueError(
                    "Tensor fusion operand does not match the prepared execution layout/device"
                )
        outputs = tuple(
            torch.empty(
                f[0],
                dtype=getattr(torch, f[1].removeprefix("torch.")),
                device=f[2],
            )
            for f in self.outputs
        )
        with torch.cuda.device(inputs[0].device):
            self.kernel(
                *inputs,
                *outputs,
                256,
                stream=torch.cuda.current_stream(inputs[0].device).cuda_stream,
            )
        return outputs
