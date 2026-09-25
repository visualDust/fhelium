"""Galois permutations of bit-reversed NTT evaluations."""

from __future__ import annotations
from functools import cache
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
