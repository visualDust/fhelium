"""CKKS slot-embedding tables for the generator-3 and generator-5 slot orders.

``_permutations`` returns the slot permutations ``pre`` and ``post`` and
``_twister`` and ``_skewer`` the negacyclic twists consumed by the libfhelium
``encode`` and ``decode`` operations, cached per ring dimension and device.
"""

from __future__ import annotations

from functools import cache

import torch

from fhelium.native.wrapper import ckks_ops


@cache
def _permutations(
    ring_dimension: int,
    device: str,
    generator: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pre, post = ckks_ops.slot_permutations(
        ring_dimension.bit_length() - 1, generator
    )
    return pre.to(device), post.to(device)


@cache
def _twister(ring_dimension: int, device: str) -> torch.Tensor:
    return ckks_ops.slot_twist(ring_dimension, False, torch.device(device))


@cache
def _skewer(ring_dimension: int, device: str) -> torch.Tensor:
    return ckks_ops.slot_twist(ring_dimension, True, torch.device(device))
