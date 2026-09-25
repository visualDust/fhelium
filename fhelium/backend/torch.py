"""Execute registered public Tensor calls without changing their numerical role.

The function table is the executable scope of the Torch-call implementation.
Serialized target strings never trigger arbitrary Python imports or evaluation.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation

from fhelium.ir.dialects.torch import CallOp, TensorCallOp
from .implementation import OperationInvocation
from .resources import BoundResource, ResourceRequirement

TORCH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "torch.add": torch.add,
    "torch.sub": torch.sub,
    "torch.mul": torch.mul,
    "torch.div": torch.div,
    "torch.neg": torch.neg,
    "torch.pow": torch.pow,
    "torch.roll": torch.roll,
    "torch.matmul": torch.matmul,
    "torch.linalg.vector_norm": torch.linalg.vector_norm,
    "torch.sum": torch.sum,
    "torch.mean": torch.mean,
    "torch.sqrt": torch.sqrt,
    "torch.rsqrt": torch.rsqrt,
    "torch.exp": torch.exp,
    "torch.log": torch.log,
    "torch.sin": torch.sin,
    "torch.cos": torch.cos,
    "torch.tanh": torch.tanh,
    "torch.sigmoid": torch.sigmoid,
    "torch.relu": torch.relu,
    "torch.reshape": torch.reshape,
    "torch.transpose": torch.transpose,
    "torch.permute": torch.permute,
    "torch.clone": torch.clone,
    "torch.zeros_like": torch.zeros_like,
    "torch.stack": torch.stack,
    "torch.cat": torch.cat,
    "torch.getitem": operator.getitem,
    "torch.select": torch.select,
    "torch.narrow": torch.narrow,
    "torch.Tensor.to": torch.Tensor.to,
    "torch.Tensor.contiguous": torch.Tensor.contiguous,
}
_FUNCTION_SYMBOLS = {
    function: name for name, function in TORCH_FUNCTIONS.items()
}
_FUNCTION_SYMBOLS.update(
    {
        operator.add: "torch.add",
        operator.sub: "torch.sub",
        operator.mul: "torch.mul",
        operator.truediv: "torch.div",
        operator.neg: "torch.neg",
        operator.pow: "torch.pow",
        torch.multiply: "torch.mul",
        torch.subtract: "torch.sub",
        torch.divide: "torch.div",
        torch.negative: "torch.neg",
        torch.Tensor.select: "torch.select",
        torch.Tensor.narrow: "torch.narrow",
    }
)


def torch_function_symbol(function: Callable[..., Any]) -> str:
    """Name a supported ordinary Tensor operation using its public target."""
    try:
        return _FUNCTION_SYMBOLS[function]
    except KeyError:
        raise NotImplementedError(
            f"No ordinary Tensor implementation is registered for {function}"
        ) from None


def decode_argument(descriptor: Any, operands: tuple[Any, ...]) -> Any:
    """Decode the existing Torch-call argument format against live operands."""
    kind = descriptor["kind"]
    if kind == "ssa":
        return operands[descriptor["operand"]]
    if kind in {"tuple", "list"}:
        values = [
            decode_argument(item, operands) for item in descriptor["items"]
        ]
        return tuple(values) if kind == "tuple" else values
    if kind == "mapping":
        return {
            decode_literal(key): decode_argument(value, operands)
            for key, value in descriptor["entries"]
        }
    if kind == "slice":
        return slice(
            *(
                decode_argument(descriptor[field], operands)
                for field in ("start", "stop", "step")
            )
        )
    if kind == "literal":
        return decode_literal(descriptor["value"])
    raise ValueError(f"Unsupported Torch argument descriptor {kind!r}")


def decode_literal(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "complex":
        return complex(value["real"], value["imag"])
    if kind == "ellipsis":
        return Ellipsis
    if kind == "torch.device":
        return torch.device(value["value"])
    if kind in {"torch.dtype", "torch.layout"}:
        return getattr(torch, value["value"].removeprefix("torch."))
    raise ValueError(f"Unsupported Torch literal {kind!r}")


def prepare_tensor_call(
    function: Callable[..., Any], descriptor: Any
) -> Callable:
    """Resolve nested argument descriptors into a direct Torch invocation."""
    namespace: dict[str, Any] = {"function": function}

    def literal(value):
        name = f"literal{len(namespace)}"
        namespace[name] = decode_literal(value)
        return name

    def expression(value):
        if not isinstance(value, dict):
            return literal(value)
        kind = value["kind"]
        if kind == "ssa":
            return f"inputs[{value['operand']}]"
        if kind in {"tuple", "list"}:
            items = [expression(item) for item in value["items"]]
            if kind == "list":
                return "[" + ", ".join(items) + "]"
            return (
                "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
            )
        if kind == "mapping":
            return (
                "{"
                + ", ".join(
                    f"{literal(key)}: {expression(item)}"
                    for key, item in value["entries"]
                )
                + "}"
            )
        if kind == "slice":
            return (
                "slice("
                + ", ".join(
                    expression(value[field])
                    for field in ("start", "stop", "step")
                )
                + ")"
            )
        if kind == "literal":
            return literal(value["value"])
        raise ValueError(f"Unsupported Torch argument descriptor {kind!r}")

    arguments = descriptor["args"]
    kwargs = descriptor["kwargs"]
    positional = [expression(item) for item in arguments["items"]]
    keyword = [
        f"{decode_literal(key)}={expression(value)}"
        for key, value in kwargs["entries"]
    ]
    # Keyword names are validated Python identifiers, not executable source.
    import keyword as keywords

    if any(
        not isinstance(decode_literal(key), str)
        or not decode_literal(key).isidentifier()
        or keywords.iskeyword(decode_literal(key))
        for key, _ in kwargs["entries"]
    ):
        raise ValueError("Torch keyword names must be Python identifiers")
    source = (
        "def call(inputs):\n    return (function("
        + ", ".join((*positional, *keyword))
        + "),)\n"
    )
    exec(compile(source, "<fhelium-torch-call>", "exec"), namespace)
    return namespace["call"]


@dataclass(frozen=True)
class TorchCallImplementation:
    """Run one registered Tensor function with caller-supplied Tensor operands."""

    name: str = "torch-tensor"
    operation_types: tuple[type[Operation], ...] = (CallOp, TensorCallOp)
    supports_in_place: bool = False
    _function: Callable[..., Any] | None = None
    _descriptor: Any = None
    _call: Callable | None = None

    def supports_operation(self, operation: Operation) -> bool:
        target = operation.attributes.get("fhelium.call.target")
        kind = operation.attributes.get("fhelium.call.kind")
        return (
            isinstance(target, StringAttr)
            and target.data in TORCH_FUNCTIONS
            and kind == StringAttr("function")
        )

    def prepare_operation(
        self, operation: Operation
    ) -> TorchCallImplementation:
        if not self.supports_operation(operation):
            raise NotImplementedError(
                f"Unsupported ordinary Tensor call: {operation.attributes.get('fhelium.call.target')}"
            )
        target = operation.attributes["fhelium.call.target"]
        descriptor = operation.attributes["fhelium.call.arguments"]
        assert isinstance(target, StringAttr) and isinstance(
            descriptor, StringAttr
        )
        arguments = json.loads(descriptor.data)
        function = TORCH_FUNCTIONS[target.data]
        return TorchCallImplementation(
            _function=function,
            _descriptor=arguments,
            _call=prepare_tensor_call(function, arguments),
        )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        assert self._call is not None
        return self._call(inputs)


__all__ = ["TorchCallImplementation"]
