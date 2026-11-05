"""Typed, family-specific NTT table materialization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from fhelium.config import CkksConfig
from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    CompactRadix2Policy,
    IndexedRadix2Policy,
    NttBackendPolicy,
)
from fhelium.backend.ntt.plans import (
    CompactPowerOfTwoRadixNttPlan,
    CompactRadix2NttPlan,
    IndexedRadix2NttPlan,
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
MaterializeParameterRows = Callable[[torch.Tensor], torch.Tensor]


def _copy_indices_to_device(
    indices: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    if indices.dtype != torch.int32:
        raise TypeError(
            f"Indexed NTT schedules require torch.int32, got {indices.dtype}"
        )
    return indices.detach().clone().to(device=device)


def prepare_ntt_tables(
    policy: NttBackendPolicy,
    ckks_config: CkksConfig,
    *,
    materialize_parameter_rows: MaterializeParameterRows,
    device: torch.device,
    rns_params: torch.Tensor,
    dtype: torch.dtype,
) -> NttTables:
    """Build execution-ready Tensor tables in the selected numerical layout.

    Prime rows follow depth-zero QP order. Montgomery tables are converted
    during preparation using the supplied RNS parameters. Shoup32 pairs retain
    their standard roots and quotient words without Montgomery conversion.
    """
    shoup32 = (
        isinstance(policy, CompactRadix2Policy)
        and policy.twiddle_encoding == "shoup32"
    )
    if shoup32 and torch.version.hip is not None:
        raise ValueError(
            "Shoup32 NTT tables are not supported by the HIP backend"
        )

    def materialize_twiddles(table: torch.Tensor) -> torch.Tensor:
        result = materialize_parameter_rows(table)
        if not shoup32:
            rns_ops.to_montgomery_(result.view(result.size(0), -1), rns_params)
        return result

    if isinstance(policy, CompactRadix2Policy):
        plan = CompactRadix2NttPlan(
            ckks_config,
            device="cpu",
            dtype=dtype,
            twiddle_encoding=policy.twiddle_encoding,
        )
        return CompactRadix2Tables(
            forward_twiddles=materialize_twiddles(plan.forward_twiddles),
            inverse_twiddles=materialize_twiddles(plan.inverse_twiddles),
        )

    if isinstance(policy, IndexedRadix2Policy):
        plan = IndexedRadix2NttPlan(ckks_config, device="cpu", dtype=dtype)
        for indices in (plan.forward_indices, plan.inverse_indices):
            ntt_ops.validate_indexed_schedule(
                indices[0], indices[1], ckks_config.N
            )
        return IndexedRadix2Tables(
            forward_even_indices=_copy_indices_to_device(
                plan.forward_indices[0], device=device
            ),
            forward_odd_indices=_copy_indices_to_device(
                plan.forward_indices[1], device=device
            ),
            forward_twiddles=materialize_twiddles(plan.forward_twiddles),
            inverse_even_indices=_copy_indices_to_device(
                plan.inverse_indices[0], device=device
            ),
            inverse_odd_indices=_copy_indices_to_device(
                plan.inverse_indices[1], device=device
            ),
            inverse_twiddles=materialize_twiddles(plan.inverse_twiddles),
        )

    if isinstance(policy, CompactFixedRadixPolicy):
        plan = CompactPowerOfTwoRadixNttPlan(
            ckks_config, policy, device="cpu", dtype=dtype
        )
        return CompactPowerOfTwoRadixTables(
            forward_outer_twiddles=materialize_twiddles(
                plan.forward_outer_twiddles
            ),
            inverse_outer_twiddles=materialize_twiddles(
                plan.inverse_outer_twiddles
            ),
            forward_radix_root_powers=materialize_twiddles(
                plan.forward_radix_root_powers
            ),
            inverse_radix_root_powers=materialize_twiddles(
                plan.inverse_radix_root_powers
            ),
        )

    raise AssertionError(f"Unhandled NTT policy {policy}")
