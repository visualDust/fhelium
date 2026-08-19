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
from fhelium.engine.ntt.plans import (
    CompactPowerOfTwoRadixNttPlan,
    CompactRadix2NttPlan,
    IndexedRadix2NttPlan,
)
from fhelium.native.wrapper import rns_ops


@dataclass(frozen=True)
class IndexedRadix2Tables:
    """Schedules and stage-expanded twiddles for indexed execution.

    Index tensors are ``torch.int32`` on the backend device. Twiddle tensors
    are integral ``[limb, stage, butterfly]`` rows aligned with canonical QP
    parameter columns; they begin in standard form and are converted in place
    to Montgomery form without changing shape, dtype, device, or storage.
    """

    forward_even_indices: torch.Tensor
    forward_odd_indices: torch.Tensor
    forward_twiddles: torch.Tensor
    inverse_even_indices: torch.Tensor
    inverse_odd_indices: torch.Tensor
    inverse_twiddles: torch.Tensor

    def convert_twiddles_to_montgomery_(self, rns_params: torch.Tensor) -> None:
        for twiddles in (self.forward_twiddles, self.inverse_twiddles):
            rns_ops.to_montgomery_(
                twiddles.view(twiddles.size(0), -1), rns_params
            )


@dataclass(frozen=True)
class CompactRadix2Tables:
    """Canonical ``[limb, N]`` twiddle rows for production kernels.

    Rows align exactly with canonical QP parameter columns. Conversion mutates
    each integral table from standard to Montgomery residues in place.
    """

    forward_twiddles: torch.Tensor
    inverse_twiddles: torch.Tensor

    def convert_twiddles_to_montgomery_(self, rns_params: torch.Tensor) -> None:
        for twiddles in (self.forward_twiddles, self.inverse_twiddles):
            rns_ops.to_montgomery_(twiddles, rns_params)


@dataclass(frozen=True)
class CompactPowerOfTwoRadixTables:
    """Packed outer twists and roots for strict fixed-radix digits.

    Outer shapes are ``[limb, N - 1]`` and root shapes are
    ``[limb, radix]`` in canonical QP row order. Conversion mutates each
    integral table from standard to Montgomery residues in place.
    """

    forward_outer_twiddles: torch.Tensor
    inverse_outer_twiddles: torch.Tensor
    forward_radix_root_powers: torch.Tensor
    inverse_radix_root_powers: torch.Tensor

    def convert_twiddles_to_montgomery_(self, rns_params: torch.Tensor) -> None:
        for twiddles in (
            self.forward_outer_twiddles,
            self.inverse_outer_twiddles,
            self.forward_radix_root_powers,
            self.inverse_radix_root_powers,
        ):
            rns_ops.to_montgomery_(twiddles, rns_params)


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
) -> NttTables:
    """Build a plan and return separately allocated backend table tensors.

    Prime rows follow the config's canonical level-zero QP order exactly.
    Returned tensors use ``ckks_config.torch_dtype`` on ``device`` and remain
    in standard representation until ``RnsRuntime`` converts them in place.
    """

    if isinstance(policy, CompactRadix2Policy):
        plan = CompactRadix2NttPlan(ckks_config, device="cpu")
        return CompactRadix2Tables(
            forward_twiddles=materialize_parameter_rows(plan.forward_twiddles),
            inverse_twiddles=materialize_parameter_rows(plan.inverse_twiddles),
        )

    if isinstance(policy, IndexedRadix2Policy):
        plan = IndexedRadix2NttPlan(ckks_config, device="cpu")
        return IndexedRadix2Tables(
            forward_even_indices=_copy_indices_to_device(
                plan.forward_indices[0], device=device
            ),
            forward_odd_indices=_copy_indices_to_device(
                plan.forward_indices[1], device=device
            ),
            forward_twiddles=materialize_parameter_rows(plan.forward_twiddles),
            inverse_even_indices=_copy_indices_to_device(
                plan.inverse_indices[0], device=device
            ),
            inverse_odd_indices=_copy_indices_to_device(
                plan.inverse_indices[1], device=device
            ),
            inverse_twiddles=materialize_parameter_rows(plan.inverse_twiddles),
        )

    if isinstance(policy, CompactFixedRadixPolicy):
        plan = CompactPowerOfTwoRadixNttPlan(ckks_config, policy, device="cpu")
        return CompactPowerOfTwoRadixTables(
            forward_outer_twiddles=materialize_parameter_rows(
                plan.forward_outer_twiddles
            ),
            inverse_outer_twiddles=materialize_parameter_rows(
                plan.inverse_outer_twiddles
            ),
            forward_radix_root_powers=materialize_parameter_rows(
                plan.forward_radix_root_powers
            ),
            inverse_radix_root_powers=materialize_parameter_rows(
                plan.inverse_radix_root_powers
            ),
        )

    raise AssertionError(f"Unhandled NTT policy {policy}")
