"""CKKS Galois-automorphism indexing utilities."""

from __future__ import annotations

from functools import cache

import torch


def rotation_galois_element(
    ring_dimension: int, rotation_step: int, generator: int = 3
) -> int:
    r"""Map signed ``rotation_step`` to odd polynomial ``galois_element``.

    The slot convention is ``torch.roll(slots, shifts=rotation_step)``. The
    result $g$ identifies automorphism $\sigma_g:X\mapsto X^g$ modulo
    $X^N+1$. The step and element are related but intentionally retain distinct
    names.
    """

    exponent = -int(rotation_step) if generator == 5 else int(rotation_step)
    return pow(
        generator,
        exponent % ring_dimension,
        2 * ring_dimension,
    )


@cache
def forward_slot_generator_positions(
    ring_dimension: int,
    generator: int,
) -> tuple[int, ...]:
    """Enumerate canonical coefficient positions by generator powers."""

    if ring_dimension <= 0 or ring_dimension & (ring_dimension - 1):
        raise ValueError("ring_dimension must be a positive power of two")
    if generator not in {3, 5}:
        raise ValueError("generator must be 3 or 5")
    modulus = 2 * ring_dimension
    value = 1
    positions = []
    for _ in range(ring_dimension // 2):
        positions.append((value - 1) // 2)
        value = (value * generator) % modulus
    return tuple(positions)


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
    r"""Return canonical non-aliasing coefficient residues for $\sigma_g$.

    ``residues`` has integral ``[*batch, limb, coefficient]`` layout on one
    device in either standard or Montgomery representation. The gather and
    sign change preserve that representation and exact prime-row order.
    ``moduli`` is a same-dtype/device vector containing the exact modulus for
    each limb; reducing against it maps signed gather results into canonical
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
