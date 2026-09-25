"""Prime-group quotient execution for represented RNS polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from collections.abc import Mapping

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops, rns_ops, ntt_ops
from fhelium.config.ntt import CompactRadix2Policy, resolve_ntt_backend_policy
from fhelium.backend.ntt.operations import execute_named_transform


def _drop_coefficient_rows(
    source: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    half_primes: tuple[int, ...],
    drop_count: int,
    *,
    nearest: bool,
    in_place: bool,
) -> torch.Tensor:
    """Apply successive rounded quotients, reusing owned intermediate storage."""

    for step in range(drop_count):
        parameters, inverse = (
            tensors[f"step{step}_parameters"],
            tensors[f"step{step}_inverse"],
        )
        half_prime = half_primes[step]
        remaining, dropped = source[..., 1:, :], source[..., 0, :]
        if in_place or step:
            if nearest:
                ckks_ops.rescale_drop_leading_prime_nearest_(
                    remaining, inverse, dropped, parameters, half_prime
                )
            else:
                ckks_ops.rescale_drop_leading_prime_truncate_(
                    remaining, inverse, dropped, parameters
                )
            source = remaining
        elif nearest:
            source = ckks_ops.rescale_drop_leading_prime_nearest(
                remaining, inverse, dropped, parameters, half_prime
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
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        source = values[0]
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                values[1:],
                strict=True,
            )
        )
        count = cast(int, invocation.attributes["drop_count"])
        halves = cast(tuple[int, ...], invocation.attributes["half_primes"])
        nearest = invocation.attributes.get("rounding", "nearest") == "nearest"
        if invocation.attributes.get("input_domain", "coefficient") != "ntt":
            return (
                _drop_coefficient_rows(
                    source,
                    tensors,
                    halves,
                    count,
                    nearest=nearest,
                    in_place=in_place,
                ),
            )
        remaining = source[..., count:, :]
        dropped = source[..., :count, :].clone()
        execute_named_transform(
            dropped,
            tensors,
            invocation.attributes,
            "dropped",
            "inverse_to_standard_",
        )
        correction = torch.zeros_like(source)
        correction[..., :count, :].copy_(dropped)
        correction = _drop_coefficient_rows(
            correction, tensors, halves, count, nearest=nearest, in_place=True
        )
        inverse = tensors["prefix_inverse"]
        parameters = tensors["remaining_forward_0"]
        policy = resolve_ntt_backend_policy(
            str(invocation.attributes["ntt_backend"])
        )
        if isinstance(policy, CompactRadix2Policy):
            ntt_ops.forward_ntt_to_montgomery_compact_add_scaled_(
                correction,
                remaining,
                inverse,
                tensors["remaining_forward_1"],
                parameters,
                policy.grouped_radix2_stage_count,
            )
        else:
            execute_named_transform(
                correction,
                tensors,
                invocation.attributes,
                "remaining",
                "forward_to_montgomery_",
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
