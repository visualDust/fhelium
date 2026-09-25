"""Hybrid-digit decomposition and RNS basis extension."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import rns_ops


def modup_digit(
    source: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    digit_index: int,
    span: tuple[int, int, int],
) -> torch.Tensor:
    """Extend one represented coefficient digit using supplied arithmetic tables."""
    start, stop, _ = span
    mixed = source[..., start:stop, :]
    if stop - start > 1:
        mixed = rns_ops.mixed_radix_decompose(
            mixed,
            tensors[f"digit{digit_index}_normalizers"],
            tensors[f"digit{digit_index}_propagation"],
            *(
                tensors[f"digit{digit_index}_reduction{part}"]
                for part in range(4)
            ),
        )
    parameters = tensors["qp_parameters"]
    return rns_ops.mixed_radix_basis_extend_to_montgomery(
        mixed,
        tensors[f"digit{digit_index}_extension"],
        parameters,
        parameters.size(1),
    )


@dataclass(frozen=True)
class NativeHybridModUpImplementation:
    """Execute one represented hybrid-digit ModUp using concrete RNS tables."""

    name: str = "native-hybrid-modup"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.HybridModUpDigitOp,)

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
        del resources, in_place
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                values[1:],
                strict=True,
            )
        )
        index = cast(int, invocation.attributes["digit_index"])
        spans = cast(
            tuple[tuple[int, int, int], ...], invocation.attributes["digits"]
        )
        return (modup_digit(values[0], tensors, index, spans[index]),)
