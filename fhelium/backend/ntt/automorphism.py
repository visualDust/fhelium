"""Galois permutations of bit-reversed NTT evaluations."""

from __future__ import annotations
from functools import cache
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects.ntt import AutomorphismOp
import torch

from fhelium.native.wrapper import ntt_ops


@cache
def ntt_galois_indices(
    n: int, galois_element: int, device: torch.device
) -> torch.Tensor:
    """Map stored bit-reversed NTT positions through the Galois automorphism."""
    return ntt_ops.galois_indices(n.bit_length() - 1, galois_element).to(device)


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
