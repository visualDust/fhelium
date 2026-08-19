"""Compact-table grouped radix-2 CUDA NTT production backend."""

from __future__ import annotations

import torch

from fhelium.config.ntt import CompactRadix2Policy
from fhelium.engine.ntt.interface import slice_ntt_parameter_rows
from fhelium.engine.ntt.tables import CompactRadix2Tables
from fhelium.native.wrapper import ntt_ops


class CompactRadix2NttBackend:
    """Compute butterfly indices in CUDA from canonical compact twiddles."""

    def __init__(
        self,
        *,
        policy: CompactRadix2Policy,
        ntt_tables: CompactRadix2Tables,
        rns_params: torch.Tensor,
    ) -> None:
        self.policy = policy
        self.name = policy.name
        self.grouped_radix2_stage_count = policy.grouped_radix2_stage_count
        self.rns_params = rns_params
        self.forward_twiddles = ntt_tables.forward_twiddles
        self.inverse_twiddles = ntt_tables.inverse_twiddles
        self._native_input_cache: dict[
            tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def __str__(self) -> str:
        return (
            f"CompactRadix2NttBackend(name={self.name!r}, "
            f"group_width={self.policy.group_width}, "
            f"forward_twiddle_shape={tuple(self.forward_twiddles.shape)})"
        )

    __repr__ = __str__

    def _active_native_inputs(
        self,
        operand: torch.Tensor,
        twiddles: torch.Tensor,
        parameter_row_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if operand.ndim < 2:
            return slice_ntt_parameter_rows(
                operand,
                twiddles,
                self.rns_params,
                parameter_row_start,
            )
        cache_key = (
            id(twiddles),
            id(self.rns_params),
            parameter_row_start,
            operand.size(-2),
        )
        cached = self._native_input_cache.get(cache_key)
        if cached is not None:
            return cached
        cached = slice_ntt_parameter_rows(
            operand,
            twiddles,
            self.rns_params,
            parameter_row_start,
        )
        self._native_input_cache[cache_key] = cached
        return cached

    def forward_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.forward_twiddles, parameter_row_start
        )
        ntt_ops.forward_ntt_montgomery_compact_grouped_smem_(
            operand,
            twiddles,
            params,
            self.grouped_radix2_stage_count,
        )

    def forward_to_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.forward_twiddles, parameter_row_start
        )
        ntt_ops.forward_ntt_to_montgomery_compact_grouped_smem_(
            operand,
            twiddles,
            params,
            self.grouped_radix2_stage_count,
        )

    def forward_to_montgomery(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> torch.Tensor:
        twiddles, params = self._active_native_inputs(
            operand, self.forward_twiddles, parameter_row_start
        )
        return ntt_ops.forward_ntt_to_montgomery_compact_grouped_smem(
            operand,
            twiddles,
            params,
            self.grouped_radix2_stage_count,
        )

    def inverse_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_montgomery_compact_grouped_smem_(
            operand,
            twiddles,
            params,
            self.grouped_radix2_stage_count,
        )

    def inverse_to_standard_lazy_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_to_standard_lazy_compact_grouped_smem_(
            operand,
            twiddles,
            params,
            self.grouped_radix2_stage_count,
        )

    def inverse_to_standard_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_to_standard_compact_grouped_smem_(
            operand,
            twiddles,
            params,
            self.grouped_radix2_stage_count,
        )

    def inverse_to_centered_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_to_centered_compact_grouped_smem_(
            operand,
            twiddles,
            params,
            self.grouped_radix2_stage_count,
        )
