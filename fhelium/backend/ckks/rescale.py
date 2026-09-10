"""Prime-group quotient execution for represented RNS polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.ckks.resources import RescaleExecutionResource
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ntt.executors.compact_radix2 import CompactRadix2NttBackend
from fhelium.backend.ntt.resources import NTT_RESOURCE_KIND
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns._operand_state import _operand_basis
from fhelium.backend.rns.resources import RNS_RESOURCE_KIND
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops, rns_ops


def _drop_coefficient_rows(
    source: torch.Tensor,
    resource: RescaleExecutionResource,
    drop_count: int,
    *,
    include_p: bool,
    nearest: bool,
    in_place: bool,
) -> torch.Tensor:
    """Apply successive rounded quotients, reusing owned intermediate storage."""

    for step in range(drop_count):
        parameters, inverse, prime = resource.active_views(
            source.size(-2), include_p=include_p
        )
        remaining, dropped = source[..., 1:, :], source[..., 0, :]
        if in_place or step:
            if nearest:
                ckks_ops.rescale_drop_leading_prime_nearest_(
                    remaining, inverse, dropped, parameters, prime // 2
                )
            else:
                ckks_ops.rescale_drop_leading_prime_truncate_(
                    remaining, inverse, dropped, parameters
                )
            source = remaining
        elif nearest:
            source = ckks_ops.rescale_drop_leading_prime_nearest(
                remaining, inverse, dropped, parameters, prime // 2
            )
        else:
            source = ckks_ops.rescale_drop_leading_prime_truncate(
                remaining, inverse, dropped, parameters
            )
    return source


@dataclass(frozen=True)
class NativeRescaleImplementation:
    """Execute a leading-prime-group quotient in coefficient or NTT form."""

    name: str = "native-rescale"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.RescaleDropLeadingPrimesOp,
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
        drop_count = cast(int, invocation.attributes.get("drop_count", 1))
        nearest = invocation.attributes.get("rounding", "nearest") == "nearest"
        if invocation.attributes.get("input_domain", "coefficient") != "ntt":
            return (
                _drop_coefficient_rows(
                    source, resource, drop_count,
                    include_p=include_p, nearest=nearest, in_place=in_place,
                ),
            )

        rns_context = cast(RnsContext, resources[1].value)
        ntt_context = cast(NttContext, resources[2].value)
        row_stop = rns_context.qp_row_stop if include_p else rns_context.q_row_stop
        row_start = row_stop - source.size(-2)
        remaining = source[..., drop_count:, :]
        dropped = source[..., :drop_count, :].clone()
        ntt_context.inverse_to_standard_(dropped, parameter_row_start=row_start)

        # The dropped residues determine the rounded correction modulo every
        # surviving prime. Keep this computation in coefficient form throughout
        # the group, then combine its single NTT with M^{-1} times the input.
        correction = torch.zeros_like(source)
        correction[..., :drop_count, :].copy_(dropped)
        correction = _drop_coefficient_rows(
            correction, resource, drop_count,
            include_p=include_p, nearest=nearest, in_place=True,
        )
        inverse = resource.prefix_inverse(
            source.size(-2), drop_count, include_p=include_p
        )
        backend = ntt_context.ntt_backend
        if isinstance(backend, CompactRadix2NttBackend):
            backend.forward_to_montgomery_add_scaled_(
                correction, remaining, inverse, row_start + drop_count
            )
        else:
            ntt_context.forward_to_montgomery_(
                correction, parameter_row_start=row_start + drop_count
            )
            parameters = rns_context.rns_parameters_for_prime_ids(
                tuple(range(row_start + drop_count, row_stop))
            )
            scaled = rns_ops.montgomery_mul_row_scalars_standard(
                remaining, inverse, parameters
            )
            correction = rns_ops.add_lazy(scaled, correction, parameters)
        if in_place:
            remaining.copy_(correction)
            correction = remaining
        return (correction,)


__all__ = ["NativeRescaleImplementation"]
