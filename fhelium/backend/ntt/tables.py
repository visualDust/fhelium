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
from fhelium.native.wrapper import ntt_ops, rns_ops


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
    device: torch.device,
    rns_params: torch.Tensor,
    dtype: torch.dtype,
) -> NttTables:
    """Build execution-ready Tensor tables in the selected numerical layout.

    libfhelium builds the standard-residue tables on the host with prime rows
    in ``moduli`` order, which matches the columns of ``rns_params``.
    Montgomery tables are converted on ``device`` with those parameters.
    Shoup32 pairs retain their standard roots and quotient words.
    """
    moduli = list(moduli)
    shoup32 = (
        isinstance(policy, CompactRadix2Policy)
        and policy.twiddle_encoding == "shoup32"
    )
    if shoup32 and torch.version.hip is not None:
        raise ValueError(
            "Shoup32 NTT tables are not supported by the HIP backend"
        )

    def materialize_twiddles(table: torch.Tensor) -> torch.Tensor:
        result = table.to(device)
        if not shoup32:
            rns_ops.to_montgomery_(result.view(result.size(0), -1), rns_params)
        return result

    if isinstance(policy, CompactRadix2Policy):
        forward, inverse = ntt_ops.compact_twiddles(
            moduli, log_n, dtype, shoup32
        )
        return CompactRadix2Tables(
            forward_twiddles=materialize_twiddles(forward),
            inverse_twiddles=materialize_twiddles(inverse),
        )

    if isinstance(policy, IndexedRadix2Policy):
        (
            forward_indices,
            forward_twiddles,
            inverse_indices,
            inverse_twiddles,
        ) = ntt_ops.indexed_radix2_plan(moduli, log_n, dtype)
        for indices in (forward_indices, inverse_indices):
            ntt_ops.validate_indexed_schedule(
                indices[0], indices[1], 1 << log_n
            )
        forward_indices = forward_indices.to(device)
        inverse_indices = inverse_indices.to(device)
        return IndexedRadix2Tables(
            forward_even_indices=forward_indices[0],
            forward_odd_indices=forward_indices[1],
            forward_twiddles=materialize_twiddles(forward_twiddles),
            inverse_even_indices=inverse_indices[0],
            inverse_odd_indices=inverse_indices[1],
            inverse_twiddles=materialize_twiddles(inverse_twiddles),
        )

    if isinstance(policy, CompactFixedRadixPolicy):
        tables = ntt_ops.power_of_two_radix_twiddles(
            moduli, log_n, policy.radix, dtype
        )
        return CompactPowerOfTwoRadixTables(
            *(materialize_twiddles(table) for table in tables)
        )

    raise AssertionError(f"Unhandled NTT policy {policy}")
