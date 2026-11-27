"""CKKS slot-embedding tables for the generator-3 and generator-5 slot orders.

``embedding_tables`` returns the tables of ``fhelium_ckks_ops::embedding_table_set`` by name, cached per ring dimension, generator and device: the slot permutations ``pre`` and ``post`` and the negacyclic twists ``twister`` and ``skewer`` consumed by the libfhelium ``encode`` and ``decode`` operations.
"""

from __future__ import annotations

from functools import cache

import torch

from fhelium.native.wrapper import ckks_ops


@cache
def embedding_tables(
    ring_dimension: int, generator: int, device: str
) -> dict[str, torch.Tensor]:
    names, tensors = ckks_ops.embedding_table_set(
        ring_dimension.bit_length() - 1, generator, torch.device(device)
    )
    return dict(zip(names, tensors, strict=True))
