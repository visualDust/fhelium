"""Montgomery products with indexed key-digit Tensor operands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops


@dataclass(frozen=True)
class NativeKeySwitchDigitProductImplementation:
    """Multiply one NTT QP digit by an evaluation-key digit."""

    name: str = "native-key-switch-digit-product"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.KeySwitchDigitProductOp,
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
        del resources, in_place
        digit, key, parameters = values
        index = cast(int, invocation.attributes["key_digit_index"])
        accumulator0, accumulator1 = (
            torch.zeros_like(digit),
            torch.zeros_like(digit),
        )
        ckks_ops.keyswitch_accumulate_digit_products_(
            accumulator0,
            accumulator1,
            digit,
            key[index],
            parameters,
            cast(int, invocation.attributes["key_row_start"]),
        )
        return (torch.stack((accumulator0, accumulator1)),)
