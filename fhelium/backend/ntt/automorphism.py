"""Galois permutations of bit-reversed NTT evaluations."""

from __future__ import annotations
from functools import cache
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects.ntt import AutomorphismOp
import torch


@cache
def ntt_galois_indices(
    n: int, galois_element: int, device: torch.device
) -> torch.Tensor:
    """Map stored bit-reversed NTT positions through the Galois automorphism."""
    source = torch.arange(n, dtype=torch.int64, device=device)
    remaining = source.clone()
    reversed_indices = torch.zeros_like(source)
    for _ in range(n.bit_length() - 1):
        reversed_indices = (reversed_indices << 1) | (remaining & 1)
        remaining >>= 1
    element = galois_element
    positions = (((2 * reversed_indices + 1) * element) % (2 * n) - 1) // 2
    return reversed_indices.index_select(0, positions.to(torch.long)).to(
        torch.int32
    )


class NativeNttAutomorphismImplementation:
    """Execute the supplied NTT index gather without modifying its operands."""

    name = "native-ntt-automorphism"
    supports_in_place = False
    operation_types: tuple[type[Operation], ...] = (AutomorphismOp,)

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
        from fhelium.native.wrapper import ckks_ops

        value, indices = values
        return (ckks_ops.apply_ntt_galois_automorphism(value, indices),)
