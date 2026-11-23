"""Native CKKS scalar arithmetic over parameter and random-state Tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns.context import RnsContext
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops, rns_ops
from fhelium.backend.random.stream import advance_


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
    return (parameters, rounding_state), {}


@dataclass(frozen=True)
class NativeScalarArithmeticImplementation:
    r"""Add or multiply CKKS ciphertexts by scalars.

    Integer scalars multiply every component by their row residues. A real
    scalar $a$ at scale $\Delta$ is rounded stochastically to
    $\lfloor a\Delta \rceil$ with the rounding state, which then advances;
    addition adds it to the constant coefficient of component zero, and
    multiplication multiplies every component by it.
    """

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
        output = ckks_ops.scalar_arithmetic(
            ciphertext,
            parameters,
            auxiliary,
            float(cast(float, invocation.attributes["scalar"])),
            float(cast(float, invocation.attributes["scalar_scale"])),
            invocation.operation_type is ckks.AddScalarOp,
        )
        advance_(auxiliary)
        return (output,)


__all__ = ["NativeScalarArithmeticImplementation"]
