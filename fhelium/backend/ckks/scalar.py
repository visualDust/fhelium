"""Native CKKS arithmetic with real and integer scalar constants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns._operand_state import _active_depth, _operand_basis
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.resources import RNS_RESOURCE_KIND
from fhelium.ir.dialects import ckks
from fhelium.rng import Csprng

from .codec import RANDOM_STREAM_RESOURCE_KIND, RANDOM_STREAM_RESOURCE_SYMBOL

_RNS_RESOURCE_SYMBOL = "active-rns-parameters"


def _quantized_real_coefficients(
    ciphertext: torch.Tensor,
    context: RnsContext,
    random_stream: Csprng,
    *,
    scalar: float,
    scalar_scale: float,
    depth: int,
    include_p: bool,
) -> torch.Tensor:
    """Return one standard-RNS coefficient for a scaled real scalar."""

    scaled = scalar * scalar_scale
    coefficient = random_stream.randround(
        torch.tensor(
            [scaled],
            dtype=torch.float64,
            device=ciphertext.device,
        )
    )
    return context.lift_integer_coefficients_exact(
        coefficient,
        depth,
        include_p=include_p,
        max_abs=int(math.ceil(abs(scaled))) + 1,
    )


def _integer_montgomery_rows(
    ciphertext: torch.Tensor,
    context: RnsContext,
    *,
    scalar: int,
    depth: int,
    include_p: bool,
) -> torch.Tensor:
    """Return one Montgomery scalar in every active prime row."""

    del ciphertext
    basis = context.basis_parameters(depth, include_p=include_p)
    return context.integer_scalar_parameters(scalar)[
        basis.parameter_row_start:basis.parameter_row_stop
    ]



@dataclass(frozen=True)
class NativeScalarArithmeticImplementation:
    """Execute CKKS scalar arithmetic with existing RNS primitives."""

    name: str = "native-ckks-scalar-arithmetic"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        ckks.AddScalarOp,
        ckks.MultiplyScalarOp,
        ckks.MultiplyIntegerScalarOp,
    )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        requirements = [
            ResourceRequirement(_RNS_RESOURCE_SYMBOL, RNS_RESOURCE_KIND)
        ]
        if invocation.operation_type in {
            ckks.AddScalarOp,
            ckks.MultiplyScalarOp,
        }:
            requirements.append(
                ResourceRequirement(
                    RANDOM_STREAM_RESOURCE_SYMBOL,
                    RANDOM_STREAM_RESOURCE_KIND,
                )
            )
        return tuple(requirements)

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        ciphertext = inputs[0]
        context = cast(RnsContext, resources[0].value)
        include_p = _operand_basis(invocation) == "QP"
        depth = _active_depth(ciphertext, context, include_p=include_p)

        if invocation.operation_type is ckks.MultiplyIntegerScalarOp:
            row_scalars = _integer_montgomery_rows(
                ciphertext,
                context,
                scalar=int(cast(int, invocation.attributes["scalar"])),
                depth=depth,
                include_p=include_p,
            )
            return (
                context.montgomery_mul_row_scalars_standard(
                    ciphertext,
                    row_scalars,
                    include_p=include_p,
                ),
            )

        random_stream = cast(Csprng, resources[1].value)
        residues = _quantized_real_coefficients(
            ciphertext,
            context,
            random_stream,
            scalar=float(cast(float, invocation.attributes["scalar"])),
            scalar_scale=float(
                cast(float, invocation.attributes["scalar_scale"])
            ),
            depth=depth,
            include_p=include_p,
        )
        if invocation.operation_type is ckks.AddScalarOp:
            output = ciphertext.clone()
            context.add_standard_(
                output[0, ..., :1],
                residues,
                include_p=include_p,
            )
            return (output,)

        context.to_montgomery_(residues, include_p=include_p)
        return (
            context.montgomery_mul_row_scalars_standard(
                ciphertext,
                residues.squeeze(-1),
                include_p=include_p,
            ),
        )


__all__ = ["NativeScalarArithmeticImplementation"]
