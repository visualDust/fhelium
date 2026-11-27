"""Hybrid-digit ModUp implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend._native_operands import flat_digits, named_tables
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import rns_ops
from fhelium.backend.lowering import ExecBuilder


@dataclass(frozen=True)
class NativeHybridModUpImplementation:
    """Extend one coefficient digit to QP Montgomery residues.

    The libfhelium operation ``hybrid_modup_digit`` decomposes the selected
    source rows into Garner mixed-radix digits and evaluates them modulo each
    QP prime with one Montgomery factor.
    """

    name: str = "native-hybrid-modup"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.HybridModUpDigitOp,)

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
            rns_ops.hybrid_modup_digit(
                values[0],
                named_tables(
                    "key_switch", attributes["parameter_names"], values[1:]
                ),
                int(attributes["digit_index"]),
                flat_digits(attributes["digits"]),
            ),
        )
