"""Registered CKKS codec implementations over numerical Tensor operands."""

from __future__ import annotations

from typing import cast

import torch

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.random.stream import advance_
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops, rns_ops


class NativeEncodeImplementation:
    """Encode slots into stochastically rounded integer coefficients.

    The libfhelium ``encode`` operation pads the message to the slot count,
    applies the inverse slot embedding, and rounds the scaled coefficients
    with the rounding state, which then advances.
    """

    name = "native-ckks-encode"
    operation_types = (ckks.EncodeOp,)
    supports_in_place = False

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
        del resources, in_place
        message, pre, twister, rounding_state = inputs
        coefficients = ckks_ops.encode(
            message,
            pre,
            twister,
            rounding_state,
            float(cast(float, invocation.attributes["scale"])),
        )
        advance_(rounding_state)
        return (coefficients,)


class NativeDecodeImplementation:
    """Decode coefficients on the operation input device."""

    name = "native-ckks-decode"
    operation_types = (ckks.DecodeOp,)
    supports_in_place = False

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
        del resources, in_place
        return (
            ckks_ops.decode(
                inputs[0],
                inputs[1],
                inputs[2],
                float(cast(float, invocation.attributes["scale"])),
                bool(invocation.attributes["is_real"]),
            ),
        )


class NativeIntegerCoefficientsToRnsImplementation:
    """Reduce integer coefficients using the supplied twice-modulus rows."""

    name = "native-ckks-integer-coefficients-to-rns"
    operation_types = (ckks.IntegerCoefficientsToRnsOp,)
    supports_in_place = False

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
        del resources, in_place
        return (rns_ops.lift_integer_coefficients(inputs[0], inputs[1]),)


__all__ = [
    "NativeDecodeImplementation",
    "NativeEncodeImplementation",
    "NativeIntegerCoefficientsToRnsImplementation",
]
