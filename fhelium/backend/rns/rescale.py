"""Rational rescale execution for represented RNS polynomials."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.backend._native_operands import named_tables
from fhelium.native.wrapper import rns_ops
from fhelium.backend.lowering import ExecBuilder


@dataclass(frozen=True)
class NativeRescaleImplementation:
    r"""Rescale from basis $B$ to $B'$ in coefficient or NTT form.

    With $D=B\setminus B'$ and $R=B'\setminus B$, each coefficient maps to
    $\lfloor (xR+H)/D \rfloor$ through one mixed-radix division by $D$,
    with $H=(D-1)/2$ for nearest rounding and $H=0$ for floor rounding.
    NTT inputs invert only the dropped rows and add the forward transform of
    the coefficient correction to the target rows multiplied by $D^{-1}$.
    """

    name: str = "native-rescale"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.RescaleOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def lower(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        attributes = invocation.attributes
        return (
            rns_ops.rational_rescale(
                values[0],
                named_tables(
                    "rescale", attributes["parameter_names"], values[1:]
                ),
                str(attributes.get("input_domain", "coefficient")),
                str(attributes.get("ntt_backend", "")),
            ),
        )


__all__ = ["NativeRescaleImplementation"]
