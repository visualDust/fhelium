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
from fhelium.backend.lowering import ExecBuilder


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

    def lower(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        digit, key, parameters, *seed = values
        index = cast(int, invocation.attributes["key_digit_index"])
        rounds = int(cast(int, invocation.attributes.get("key_rounds", 0)))
        digit_seed = seed[0][index] if seed else None
        accumulators = digit.new_zeros((2, *digit.shape))
        accumulator0, accumulator1 = accumulators[0], accumulators[1]
        p_start = int(
            cast(int, invocation.attributes.get("key_p_row_start", -1))
        )
        rows = tuple(
            cast(tuple[int, ...], invocation.attributes.get("key_q_rows", ()))
        )
        if not rows:
            runs = (
                (
                    0,
                    digit.size(-2),
                    cast(int, invocation.attributes["key_row_start"]),
                    p_start,
                ),
            )
        else:
            # Each run of consecutive key rows, then the auxiliary rows, is one product.
            runs, start = [], 0
            for row in range(1, len(rows) + 1):
                if row == len(rows) or rows[row] != rows[row - 1] + 1:
                    runs.append((start, row - start, rows[start], -1))
                    start = row
            runs.append((len(rows), digit.size(-2) - len(rows), p_start, -1))
        for start, count, key_start, key_p_start in runs:
            ckks_ops.keyswitch_accumulate_digit_products_(
                accumulator0.narrow(-2, start, count),
                accumulator1.narrow(-2, start, count),
                digit.narrow(-2, start, count),
                key[index],
                parameters.narrow(1, start, count),
                key_start,
                key_p_start,
                None,
                digit_seed,
                rounds,
            )
        return (accumulators,)
