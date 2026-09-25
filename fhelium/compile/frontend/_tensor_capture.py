"""Capture ordinary Tensor calls and role-local encrypted numeric expressions."""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, cast

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import SSAValue

from fhelium.backend.torch import torch_function_symbol, TORCH_FUNCTIONS
from fhelium.ir import value_role
from fhelium.ir.dialects import core, semantic
from fhelium.ir.dialects import torch as torch_dialect
from fhelium.ir.dialects._common import OpenStateType

from .._errors import CaptureError
from ._eager_values import state_attributes
from ._pytorch_encoding import encode_literal

if TYPE_CHECKING:
    from ._eager_capture import _Capture, _Value


def numeric_call(capture: _Capture, function, args, kwargs) -> _Value:
    from ._eager_capture import _Value

    def values(item):
        if isinstance(item, _Value):
            return [item]
        if isinstance(item, (tuple, list)):
            return [value for child in item for value in values(child)]
        if isinstance(item, dict):
            return values(tuple(item.values()))
        return []

    function = TORCH_FUNCTIONS[torch_function_symbol(function)]
    symbolic = values((args, kwargs))
    if any(value_role(item.value) == "encrypted" for item in symbolic):
        return _encrypted_call(capture, function, args, kwargs)
    symbol = torch_function_symbol(function)
    operands: list[SSAValue] = []

    def encode(item):
        if isinstance(item, torch.Tensor):
            item = capture.substitute(item)
        if isinstance(item, _Value):
            index = len(operands)
            operands.append(item.value)
            return {"kind": "ssa", "operand": index}
        if isinstance(item, (tuple, list)):
            return {
                "kind": "tuple" if isinstance(item, tuple) else "list",
                "items": [encode(child) for child in item],
            }
        if isinstance(item, dict):
            return {
                "kind": "mapping",
                "entries": [
                    [encode_literal(key), encode(value)]
                    for key, value in item.items()
                ],
            }
        if isinstance(item, slice):
            return {
                "kind": "slice",
                "start": encode(item.start),
                "stop": encode(item.stop),
                "step": encode(item.step),
            }
        return {"kind": "literal", "value": encode_literal(item)}

    descriptor = {"args": encode(args), "kwargs": encode(kwargs)}

    # FakeTensor propagation runs metadata kernels only. No payload is read,
    # transferred, or cached, and an unknown device is never filled in here.
    def fake(item):
        if isinstance(item, _Value):
            state = cast(OpenStateType, item.value.type).state.data
            shape, strides = state.get("shape"), state.get("strides")
            dtype, device = state.get("dtype"), state.get("device")
            if (
                not isinstance(shape, ArrayAttr)
                or not isinstance(dtype, StringAttr)
                or not isinstance(device, StringAttr)
                or device.data == "unknown"
            ):
                raise LookupError("incomplete physical Tensor facts")
            if any(
                not isinstance(n, IntegerAttr) or n.value.data < 0
                for n in shape
            ):
                raise LookupError("symbolic shape")
            sizes = tuple(int(cast(IntegerAttr, n).value.data) for n in shape)
            requires_grad = state.get("requires_grad")
            options = dict(
                dtype=getattr(torch, dtype.data.removeprefix("torch.")),
                device=torch.device(device.data),
                requires_grad=isinstance(requires_grad, IntegerAttr)
                and bool(requires_grad.value.data),
            )
            if isinstance(strides, ArrayAttr) and all(
                isinstance(n, IntegerAttr) and n.value.data >= 0
                for n in strides
            ):
                return torch.empty_strided(
                    sizes,
                    tuple(
                        int(cast(IntegerAttr, n).value.data) for n in strides
                    ),
                    **options,
                )
            return torch.empty(sizes, **options)
        if isinstance(item, torch.Tensor):
            return fake(capture.substitute(item))
        if isinstance(item, tuple):
            return tuple(fake(child) for child in item)
        if isinstance(item, list):
            return [fake(child) for child in item]
        if isinstance(item, dict):
            return {key: fake(child) for key, child in item.items()}
        return item

    result_type = core.MessageType()
    try:
        with FakeTensorMode():
            result = function(*fake(args), **fake(kwargs))
            if not isinstance(result, torch.Tensor):
                raise CaptureError(
                    "This Tensor-call frontend requires a Tensor result"
                )
            facts = state_attributes(result)
            if function in {
                operator.getitem,
                torch.select,
                torch.narrow,
                torch.reshape,
                torch.transpose,
                torch.permute,
                torch.clone,
                torch.Tensor.to,
            } and any(
                "strides" not in cast(OpenStateType, item.value.type).state.data
                for item in symbolic
            ):
                facts.pop("strides", None)
            result_type = result_type.with_state(facts)
    except LookupError:
        pass
    operation = torch_dialect.TensorCallOp(
        operands,
        result_type,
        kind="function",
        target=symbol,
        argument_descriptor=json.dumps(descriptor),
        role="message",
    )
    return capture.emit(operation)


def _encrypted_call(capture: _Capture, function, args, kwargs) -> _Value:
    from ._eager_capture import _Value

    operation_types = {
        torch.add: semantic.AddOp,
        torch.sub: semantic.SubtractOp,
        torch.mul: semantic.MultiplyOp,
        torch.neg: semantic.NegateOp,
        torch.roll: semantic.RollOp,
    }
    operation_type = operation_types.get(function)
    if operation_type is None:
        raise CaptureError(
            f"No encrypted numerical semantics are registered for {function}"
        )
    attributes = {}
    if function is torch.roll:
        shift = kwargs.get("shifts", args[1] if len(args) > 1 else None)
        dimension = kwargs.get("dims", args[2] if len(args) > 2 else None)
        if type(shift) is not int or dimension != -1:
            raise CaptureError(
                "Encrypted roll requires a static shift along the last logical slot axis (dims=-1)"
            )
        attributes = {
            "shift": IntegerAttr(shift, 64),
            "dimension": IntegerAttr(-1, 64),
        }
        args = args[:1]
    elif kwargs:
        raise CaptureError(
            "Encrypted arithmetic currently accepts positional operands without Torch options"
        )
    operands: list[SSAValue] = []
    for item in args:
        if isinstance(item, torch.Tensor):
            item = capture.substitute(item)
        if not isinstance(item, _Value):
            constant = core.ConstantOp.create(
                result_types=(
                    semantic.PublicType().with_state(
                        {"role": StringAttr("static")}
                    ),
                ),
                attributes={
                    "fhelium.literal": StringAttr(
                        json.dumps(encode_literal(item))
                    )
                },
            )
            item = capture.emit(constant)
        value = item.value
        encrypted = value_role(value) == "encrypted"
        state = dict(cast(OpenStateType, value.type).state.data)
        typ = (
            semantic.SecretType if encrypted else semantic.PublicType
        )().with_state(state)
        if value.type != typ:
            operation, value = UnrealizedConversionCastOp.cast_one(value, typ)
            capture.block.add_op(operation)
        operands.append(value)
    # Mathematical state is selected by FHE scheduling, not by executing the
    # Python operator on an RNS payload during capture.
    result_type = semantic.SecretType()
    return capture.emit(
        operation_type.create(
            operands=operands,
            result_types=(result_type,),
            attributes=attributes,
        )
    )
