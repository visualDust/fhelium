"""Native CKKS scalar arithmetic over parameter and random-state Tensors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns.context import RnsContext
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import rns_ops
from fhelium.rng.csprng import stochastic_round_


def scalar_operands(
    context: RnsContext,
    prime_ids: tuple[int, ...],
    scalar: int | float,
    scalar_scale: float | None,
    rounding_state: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Supply row parameters or live rounding state without sampling noise."""
    parameters = context.rns_parameters_for_prime_ids(prime_ids)
    if scalar_scale is None:
        rows = context.integer_scalar_parameters(int(scalar))[
            prime_ids[0] : prime_ids[-1] + 1
        ]
        return (parameters, rows), {}
    if rounding_state is None:
        raise ValueError(
            "Real-scalar preparation requires a rounding-state Tensor"
        )
    centered = int(math.ceil(abs(float(scalar) * scalar_scale))) + 1 < min(
        context.config.moduli[i] for i in prime_ids
    )
    return (parameters, rounding_state), {"centered_lift": centered}


@dataclass(frozen=True)
class NativeScalarArithmeticImplementation:
    """Execute scalar multiplication and addition using shared RNS primitives."""

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
        ciphertext, parameters, auxiliary = inputs
        if invocation.operation_type is ckks.MultiplyIntegerScalarOp:
            return (
                rns_ops.montgomery_mul_row_scalars_standard(
                    ciphertext, auxiliary, parameters
                ),
            )
        scaled = float(cast(float, invocation.attributes["scalar"])) * float(
            cast(float, invocation.attributes["scalar_scale"])
        )
        coefficient = stochastic_round_(
            torch.tensor(
                [scaled], dtype=torch.float64, device=ciphertext.device
            ),
            auxiliary,
        ).to(parameters.dtype)
        if invocation.attributes["centered_lift"]:
            residues = rns_ops.lift_centered_coefficients(
                coefficient, parameters[0]
            )
        else:
            residues = torch.remainder(
                coefficient.unsqueeze(-2), (parameters[0] // 2).view(-1, 1)
            )
        if invocation.operation_type is ckks.AddScalarOp:
            output = ciphertext.clone()
            rns_ops.add_standard_(output[0, ..., :1], residues, parameters)
            return (output,)
        rns_ops.to_montgomery_(residues, parameters)
        return (
            rns_ops.montgomery_mul_row_scalars_standard(
                ciphertext, residues.squeeze(-1), parameters
            ),
        )


__all__ = ["NativeScalarArithmeticImplementation"]
