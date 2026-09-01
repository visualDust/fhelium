"""Registered CKKS codec implementations and their execution resources."""

from __future__ import annotations

from typing import cast

import torch

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.config import CkksConfig
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.resources import RNS_RESOURCE_KIND
from fhelium.values import ModulusBasis
from fhelium.ir.dialects import ckks
from fhelium.rng import Csprng
from ..crypto._resources import RANDOM_STREAM_RESOURCE_KIND

from ._codec import decode_tensor, encode_tensor

CKKS_CONFIG_RESOURCE_KIND = "ckks-config"
CKKS_CONFIG_RESOURCE_SYMBOL = "ckks-config"
RANDOM_STREAM_RESOURCE_SYMBOL = "ckks-random-stream"


class NativeEncodeImplementation:
    """Encode slots using resources selected for this execution placement."""

    name = "native-ckks-encode"
    operation_types = (ckks.EncodeOp,)
    supports_in_place = False

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                CKKS_CONFIG_RESOURCE_SYMBOL,
                CKKS_CONFIG_RESOURCE_KIND,
            ),
            ResourceRequirement(
                RANDOM_STREAM_RESOURCE_SYMBOL,
                RANDOM_STREAM_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        config = cast(CkksConfig, resources[0].value)
        random_stream = cast(Csprng, resources[1].value)
        return (
            encode_tensor(
                inputs[0],
                config=config,
                rng=random_stream,
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
        del invocation
        return (
            ResourceRequirement(
                CKKS_CONFIG_RESOURCE_SYMBOL,
                CKKS_CONFIG_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        config = cast(CkksConfig, resources[0].value)
        return (
            decode_tensor(
                inputs[0],
                config=config,
                scale=float(cast(float, invocation.attributes["scale"])),
                is_real=bool(int(cast(int, invocation.attributes["is_real"]))),
            ),
        )


class NativeIntegerCoefficientsToRnsImplementation:
    """Reduce integer coefficients through the bound RNS context."""

    name = "native-ckks-integer-coefficients-to-rns"
    operation_types = (ckks.IntegerCoefficientsToRnsOp,)
    supports_in_place = False

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement("active-rns-parameters", RNS_RESOURCE_KIND),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        context = cast(RnsContext, resources[0].value)
        basis = cast(ModulusBasis, invocation.attributes["modulus_basis"])
        return (
            context.lift_integer_coefficients_exact(
                inputs[0],
                int(cast(int, invocation.attributes["level"])),
                include_p=basis == "QP",
            ),
        )


__all__ = [
    "CKKS_CONFIG_RESOURCE_KIND",
    "CKKS_CONFIG_RESOURCE_SYMBOL",
    "RANDOM_STREAM_RESOURCE_KIND",
    "RANDOM_STREAM_RESOURCE_SYMBOL",
    "NativeDecodeImplementation",
    "NativeEncodeImplementation",
    "NativeIntegerCoefficientsToRnsImplementation",
]
