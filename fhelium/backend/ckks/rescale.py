"""Native CKKS rescaling implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.ckks.resources import RescaleExecutionResource
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ntt.resources import NTT_RESOURCE_KIND
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns._operand_state import _operand_basis
from fhelium.backend.rns.resources import RNS_RESOURCE_KIND
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops


@dataclass(frozen=True)
class NativeRescaleImplementation:
    """Execute logical divide-round-drop rescaling on CPU or CUDA."""

    name: str = "native-rescale"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.RescaleDropLeadingPrimeOp,
    )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        if invocation.attributes.get("input_domain", "coefficient") == "ntt":
            return (
                ResourceRequirement("active-rns-parameters", RNS_RESOURCE_KIND),
                ResourceRequirement("active-ntt-plan", NTT_RESOURCE_KIND),
            )
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        source = values[0]
        resource = cast(RescaleExecutionResource, resources[0].value)
        include_p = _operand_basis(invocation) == "QP"
        remaining_parameters, inverse, dropped_prime = resource.active_views(
            source.size(-2),
            include_p=include_p,
        )
        remaining = source[..., 1:, :]
        dropped = source[..., 0, :]
        rounding = cast(str, invocation.attributes["rounding"])
        if invocation.attributes.get("input_domain", "coefficient") == "ntt":
            rns_context = cast(RnsContext, resources[1].value)
            ntt_context = cast(NttContext, resources[2].value)
            level = rns_context.q_row_stop - source.size(-2)
            dropped_coefficients = dropped.unsqueeze(-2).clone()
            ntt_context.inverse_to_standard_(
                dropped_coefficients,
                parameter_row_start=level,
            )
            if rounding == "nearest":
                correction = ckks_ops.rescale_drop_leading_prime_nearest(
                    torch.zeros_like(remaining),
                    inverse,
                    dropped_coefficients.squeeze(-2),
                    remaining_parameters,
                    dropped_prime // 2,
                )
            else:
                correction = ckks_ops.rescale_drop_leading_prime_truncate(
                    torch.zeros_like(remaining),
                    inverse,
                    dropped_coefficients.squeeze(-2),
                    remaining_parameters,
                )
            ntt_context.forward_to_montgomery_(
                correction,
                parameter_row_start=level + 1,
            )
            scaled = rns_context.montgomery_mul_row_scalars_standard(
                remaining,
                inverse,
                include_p=False,
            )
            output_data = rns_context.add_lazy(
                scaled,
                correction,
                include_p=False,
            )
            if in_place:
                remaining.copy_(output_data)
                output_data = remaining
            return (output_data,)
        if in_place:
            if rounding == "nearest":
                ckks_ops.rescale_drop_leading_prime_nearest_(
                    remaining,
                    inverse,
                    dropped,
                    remaining_parameters,
                    dropped_prime // 2,
                )
            else:
                ckks_ops.rescale_drop_leading_prime_truncate_(
                    remaining,
                    inverse,
                    dropped,
                    remaining_parameters,
                )
            output_data = remaining
        elif rounding == "nearest":
            output_data = ckks_ops.rescale_drop_leading_prime_nearest(
                remaining,
                inverse,
                dropped,
                remaining_parameters,
                dropped_prime // 2,
            )
        else:
            output_data = ckks_ops.rescale_drop_leading_prime_truncate(
                remaining,
                inverse,
                dropped,
                remaining_parameters,
            )
        return (output_data,)


__all__ = ["NativeRescaleImplementation"]
