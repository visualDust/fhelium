"""Native CKKS rescaling implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.ckks.resources import RescaleExecutionResource
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns._operand_state import _operand_basis
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
        del invocation
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
