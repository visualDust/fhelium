"""Indexed radix-2 NTT execution."""

from __future__ import annotations

import torch

from fhelium.config.ntt import IndexedRadix2Policy
from fhelium.engine.ntt.interface import slice_ntt_parameter_rows
from fhelium.engine.ntt.tables import IndexedRadix2Tables
from fhelium.native.wrapper import ntt_ops


class IndexedRadix2NttBackend:
    """Execute the indexed radix-2 schedule on CPU or CUDA."""

    def __init__(
        self,
        *,
        policy: IndexedRadix2Policy,
        ntt_tables: IndexedRadix2Tables,
        rns_params: torch.Tensor,
    ) -> None:
        self.policy = policy
        self.name = policy.name
        self.rns_params = rns_params
        self.forward_even_indices = ntt_tables.forward_even_indices
        self.forward_odd_indices = ntt_tables.forward_odd_indices
        self.forward_twiddles = ntt_tables.forward_twiddles
        self.inverse_even_indices = ntt_tables.inverse_even_indices
        self.inverse_odd_indices = ntt_tables.inverse_odd_indices
        self.inverse_twiddles = ntt_tables.inverse_twiddles
        self._native_input_cache: dict[
            tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def __str__(self) -> str:
        return (
            f"IndexedRadix2NttBackend(name={self.name!r}, "
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
        ntt_ops.forward_ntt_montgomery_indexed_(
            operand,
            self.forward_even_indices,
            self.forward_odd_indices,
            twiddles,
            params,
        )

    def forward_to_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.forward_twiddles, parameter_row_start
        )
        ntt_ops.forward_ntt_to_montgomery_indexed_(
            operand,
            self.forward_even_indices,
            self.forward_odd_indices,
            twiddles,
            params,
        )

    def forward_to_montgomery(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> torch.Tensor:
        twiddles, params = self._active_native_inputs(
            operand, self.forward_twiddles, parameter_row_start
        )
        return ntt_ops.forward_ntt_to_montgomery_indexed(
            operand,
            self.forward_even_indices,
            self.forward_odd_indices,
            twiddles,
            params,
        )

    def inverse_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_montgomery_indexed_(
            operand,
            self.inverse_even_indices,
            self.inverse_odd_indices,
            twiddles,
            params,
        )

    def inverse_to_standard_lazy_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_to_standard_lazy_indexed_(
            operand,
            self.inverse_even_indices,
            self.inverse_odd_indices,
            twiddles,
            params,
        )

    def inverse_to_standard_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_to_standard_indexed_(
            operand,
            self.inverse_even_indices,
            self.inverse_odd_indices,
            twiddles,
            params,
        )

    def inverse_to_centered_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        twiddles, params = self._active_native_inputs(
            operand, self.inverse_twiddles, parameter_row_start
        )
        ntt_ops.inverse_ntt_to_centered_indexed_(
            operand,
            self.inverse_even_indices,
            self.inverse_odd_indices,
            twiddles,
            params,
        )
