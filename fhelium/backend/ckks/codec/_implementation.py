"""Registered CKKS codec implementations over numerical Tensor operands."""

from __future__ import annotations

from typing import cast

import torch

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks

from ._codec import decode_tensor, encode_tensor


class NativeEncodeImplementation:
    """Encode slots with supplied embedding tables and advancing rounding state."""

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
        return (
            encode_tensor(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                scale=float(cast(float, invocation.attributes["scale"])),
            ),
        )


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
            decode_tensor(
                inputs[0],
                inputs[1],
                inputs[2],
                scale=float(cast(float, invocation.attributes["scale"])),
                is_real=bool(invocation.attributes["is_real"]),
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
        from fhelium.backend.rns.context import lift_integer_coefficients_exact

        del resources, in_place
        minimum = invocation.attributes.get("min_modulus")
        minimum = (
            int((inputs[1] // 2).min().item())
            if minimum is None
            else int(cast(int, minimum))
        )
        return (
            lift_integer_coefficients_exact(
                inputs[0], inputs[1], min_modulus=minimum
            ),
        )


__all__ = [
    "NativeDecodeImplementation",
    "NativeEncodeImplementation",
    "NativeIntegerCoefficientsToRnsImplementation",
]
