"""Prime-group quotient execution for represented RNS polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.backend._native_operands import named_tables
from fhelium.native.wrapper import rns_ops


@dataclass(frozen=True)
class NativeRescaleImplementation:
    r"""Divide by the leading prime group in coefficient or NTT form.

    Each dropped prime $q$ maps the remaining rows to
    $\lfloor (x + h)/q \rfloor$ with $h = (q - 1)/2$ for nearest rounding
    and $h = 0$ for truncation. NTT inputs invert only the dropped rows and
    add the forward transform of the coefficient correction to the remaining
    rows multiplied by the inverse prefix product.
    """

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
        attributes = invocation.attributes
        arguments = (
            values[0],
            *named_tables(attributes["parameter_names"], values[1:]),
            int(cast(int, attributes["drop_count"])),
            list(cast(tuple[int, ...], attributes["half_primes"])),
            str(attributes.get("rounding", "nearest")),
            str(attributes.get("input_domain", "coefficient")),
            str(attributes.get("ntt_backend", "")),
        )
        if in_place:
            return (rns_ops.rescale_drop_leading_primes_(*arguments),)
        return (rns_ops.rescale_drop_leading_primes(*arguments),)


__all__ = ["NativeRescaleImplementation"]
