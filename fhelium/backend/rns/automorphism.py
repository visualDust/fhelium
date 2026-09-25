r"""Coefficient indexing and RNS execution of polynomial automorphisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops

from functools import cache


@cache
def coefficient_galois_gather_indices(
    ring_dimension: int,
    galois_element: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return gathers implementing $\sigma_g:X\mapsto X^g$.

    The returned device tensors have shapes ``[coefficient]`` with dtypes
    ``torch.int32`` and ``torch.int8``. For an integral coefficient-domain
    residue tensor ``[..., coefficient]``, destination ``j`` reads
    ``source_indices[j]`` and multiplies by ``source_sign[j]``. No RNS limb or
    batch axis is represented in the tables, so the gather broadcasts across
    both without mutation.
    """

    modulus = 2 * ring_dimension
    galois_element %= modulus
    if galois_element % 2 == 0:
        raise ValueError("galois_element must be odd")
    source = torch.arange(ring_dimension, device=device, dtype=torch.int64)
    mapped = (galois_element * source) % modulus
    destination = mapped % ring_dimension
    sign = torch.where(
        ((mapped // ring_dimension) & 1) != 0,
        torch.tensor(-1, dtype=torch.int8, device=device),
        torch.tensor(1, dtype=torch.int8, device=device),
    )
    source_indices = torch.empty(
        ring_dimension, dtype=torch.int32, device=device
    )
    source_indices[destination] = source.to(torch.int32)
    source_sign = torch.empty(ring_dimension, dtype=torch.int8, device=device)
    source_sign[destination] = sign
    return source_indices, source_sign


def apply_coefficient_galois_automorphism(
    residues: torch.Tensor,
    galois_element: int,
    moduli: torch.Tensor,
) -> torch.Tensor:
    r"""Return non-aliasing coefficient residues for $\sigma_g$.

    ``residues`` has integral ``[*batch, limb, coefficient]`` layout on one
    device in either standard or Montgomery representation. The gather and
    sign change preserve that representation and prime-row order.
    ``moduli`` is a same-dtype/device vector containing the modulus for
    each limb; reducing against it maps signed gather results into
    $[0,q_i)$ representatives before any NTT or key-switch consumer. The
    output has identical shape, dtype, and device, owns independent storage,
    and neither input is mutated.
    """

    if (
        moduli.ndim != 1
        or moduli.numel() != residues.size(-2)
        or moduli.dtype != residues.dtype
        or moduli.device != residues.device
    ):
        raise ValueError(
            "moduli must be a same-dtype/device vector aligned with the "
            "residue limb axis"
        )
    source_indices, source_sign = coefficient_galois_gather_indices(
        residues.size(-1), galois_element, residues.device
    )
    transformed = (
        residues.index_select(-1, source_indices.to(torch.long)) * source_sign
    )
    return transformed.remainder(
        moduli.view(*([1] * (residues.ndim - 2)), moduli.numel(), 1)
    )


@dataclass(frozen=True)
class NativeCoefficientAutomorphismImplementation:
    """Apply one coefficient-domain Galois automorphism through the native op."""

    name: str = "native-coefficient-automorphism"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.CoefficientAutomorphismOp,
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
        del in_place
        if len(values) != 2:
            raise ValueError(
                "Coefficient automorphism requires a value and parameter Tensor"
            )
        source = values[0]
        parameters = values[1]
        gather, source_sign = coefficient_galois_gather_indices(
            source.size(-1),
            cast(int, invocation.attributes["galois_element"]),
            source.device,
        )
        twice_modulus = parameters[0]
        return (
            ckks_ops.apply_coefficient_galois_automorphism(
                source,
                gather,
                source_sign,
                twice_modulus,
            ),
        )
