"""Typed, family-specific NTT table materialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    CompactRadix2Policy,
    IndexedRadix2Policy,
    NttBackendPolicy,
)
from fhelium.native.wrapper import ntt_ops


@dataclass(frozen=True)
class IndexedRadix2Tables:
    """Schedules and stage-expanded twiddles for indexed execution.

    Index tensors are ``torch.int32`` on the backend device. Preparation validates
    their per-stage coefficient partitions. The index structure stays fixed
    throughout execution; changing it requires preparing the schedule again.
    Twiddle tensors are integral ``[limb, stage, butterfly]`` Montgomery
    rows aligned with QP parameter columns.
    """

    forward_even_indices: torch.Tensor
    forward_odd_indices: torch.Tensor
    forward_twiddles: torch.Tensor
    inverse_even_indices: torch.Tensor
    inverse_odd_indices: torch.Tensor
    inverse_twiddles: torch.Tensor


@dataclass(frozen=True)
class CompactRadix2Tables:
    """Compact execution tables aligned with QP parameter columns.

    Montgomery tables have shape ``[limb,N]``. Shoup32 tables have shape
    ``[limb,N,2]`` and contain standard-root/quotient-word pairs.
    """

    forward_twiddles: torch.Tensor
    inverse_twiddles: torch.Tensor


@dataclass(frozen=True)
class CompactPowerOfTwoRadixTables:
    """Packed outer twists and roots for strict fixed-radix digits.

    Outer shapes are ``[limb, N - 1]`` and root shapes are
    ``[limb, radix]`` in QP row order, with Montgomery residues.
    """

    forward_outer_twiddles: torch.Tensor
    inverse_outer_twiddles: torch.Tensor
    forward_radix_root_powers: torch.Tensor
    inverse_radix_root_powers: torch.Tensor


NttTables = (
    IndexedRadix2Tables | CompactRadix2Tables | CompactPowerOfTwoRadixTables
)


def prepare_ntt_tables(
    policy: NttBackendPolicy,
    moduli: Sequence[int],
    log_n: int,
    *,
    rns_params: torch.Tensor,
) -> NttTables:
    """Build execution-ready Tensor tables in the selected numerical layout.

    libfhelium builds the tables of the primes ``moduli``, whose order matches
    the columns of ``rns_params``, and places them on the device of
    ``rns_params``. Montgomery tables are converted with those parameters;
    Shoup32 pairs retain their standard roots and quotient words.
    """
    forward, inverse = ntt_ops.prepare_tables(
        rns_params, list(moduli), log_n, policy.name
    )
    if isinstance(policy, CompactRadix2Policy):
        return CompactRadix2Tables(
            forward_twiddles=forward[0], inverse_twiddles=inverse[0]
        )
    if isinstance(policy, IndexedRadix2Policy):
        return IndexedRadix2Tables(
            forward_twiddles=forward[0],
            forward_even_indices=forward[1],
            forward_odd_indices=forward[2],
            inverse_twiddles=inverse[0],
            inverse_even_indices=inverse[1],
            inverse_odd_indices=inverse[2],
        )
    if isinstance(policy, CompactFixedRadixPolicy):
        return CompactPowerOfTwoRadixTables(
            forward_outer_twiddles=forward[0],
            inverse_outer_twiddles=inverse[0],
            forward_radix_root_powers=forward[1],
            inverse_radix_root_powers=inverse[1],
        )
    raise AssertionError(f"Unhandled NTT policy {policy}")
