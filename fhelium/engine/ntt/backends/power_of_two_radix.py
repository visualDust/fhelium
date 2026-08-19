"""Compact-table power-of-two-radix CUDA NTT backend."""

from __future__ import annotations

import torch

from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    validate_ntt_backend_for_log_n,
)
from fhelium.engine.ntt.interface import slice_ntt_parameter_rows
from fhelium.engine.ntt.tables import CompactPowerOfTwoRadixTables
from fhelium.native.wrapper import ntt_ops


class CompactPowerOfTwoRadixNttBackend:
    """Execute radix-4/8/16 digits with radix-specific CUDA butterflies."""

    def __init__(
        self,
        *,
        policy: CompactFixedRadixPolicy,
        ntt_tables: CompactPowerOfTwoRadixTables,
        rns_params: torch.Tensor,
    ) -> None:
        root_widths = {
            ntt_tables.forward_radix_root_powers.size(1),
            ntt_tables.inverse_radix_root_powers.size(1),
        }
        if root_widths != {policy.radix}:
            raise TypeError(
                f"Policy {policy.name!r} requires radix "
                f"{policy.radix}, got root widths {root_widths!r}"
            )
        self.policy = policy
        self.name = policy.name
        self.radix = policy.radix
        coefficient_count = ntt_tables.forward_outer_twiddles.size(1) + 1
        log_ring_dimension = coefficient_count.bit_length() - 1
        if 1 << log_ring_dimension != coefficient_count:
            raise ValueError(
                "Power-of-two radix outer tables do not encode a power-of-two ring"
            )
        validate_ntt_backend_for_log_n(policy, log_ring_dimension)
        self.rns_params = rns_params
        self.forward_outer_twiddles = ntt_tables.forward_outer_twiddles
        self.inverse_outer_twiddles = ntt_tables.inverse_outer_twiddles
        self.forward_radix_root_powers = ntt_tables.forward_radix_root_powers
        self.inverse_radix_root_powers = ntt_tables.inverse_radix_root_powers
        self._native_input_cache: dict[
            tuple[int, int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def __str__(self) -> str:
        return (
            f"CompactPowerOfTwoRadixNttBackend(name={self.name!r}, "
            f"radix={self.radix}, "
            f"outer_twiddle_shape={tuple(self.forward_outer_twiddles.shape)})"
        )

    __repr__ = __str__

    def _active_native_inputs(
        self,
        operand: torch.Tensor,
        outer_twiddles: torch.Tensor,
        radix_root_powers: torch.Tensor,
        parameter_row_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if operand.ndim < 2:
            slice_ntt_parameter_rows(
                operand,
                outer_twiddles,
                self.rns_params,
                parameter_row_start,
            )
            raise AssertionError("unreachable after NTT operand validation")
        cache_key = (
            id(outer_twiddles),
            id(radix_root_powers),
            id(self.rns_params),
            parameter_row_start,
            operand.size(-2),
        )
        cached = self._native_input_cache.get(cache_key)
        if cached is not None:
            return cached
        active_outer_twiddles, params = slice_ntt_parameter_rows(
            operand,
            outer_twiddles,
            self.rns_params,
            parameter_row_start,
        )
        row_stop = parameter_row_start + operand.size(-2)
        if row_stop > radix_root_powers.size(0):
            raise ValueError(
                "NTT operand rows exceed the prepared power-of-two radix root range"
            )
        cached = (
            active_outer_twiddles,
            radix_root_powers[parameter_row_start:row_stop],
            params,
        )
        self._native_input_cache[cache_key] = cached
        return cached

    def forward_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        outer, roots, params = self._active_native_inputs(
            operand,
            self.forward_outer_twiddles,
            self.forward_radix_root_powers,
            parameter_row_start,
        )
        ntt_ops.forward_ntt_montgomery_power_of_two_radix_compact_(
            operand,
            outer,
            roots,
            params,
        )

    def forward_to_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        outer, roots, params = self._active_native_inputs(
            operand,
            self.forward_outer_twiddles,
            self.forward_radix_root_powers,
            parameter_row_start,
        )
        ntt_ops.forward_ntt_to_montgomery_power_of_two_radix_compact_(
            operand,
            outer,
            roots,
            params,
        )

    def forward_to_montgomery(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> torch.Tensor:
        outer, roots, params = self._active_native_inputs(
            operand,
            self.forward_outer_twiddles,
            self.forward_radix_root_powers,
            parameter_row_start,
        )
        return ntt_ops.forward_ntt_to_montgomery_power_of_two_radix_compact(
            operand,
            outer,
            roots,
            params,
        )

    def inverse_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        outer, roots, params = self._active_native_inputs(
            operand,
            self.inverse_outer_twiddles,
            self.inverse_radix_root_powers,
            parameter_row_start,
        )
        ntt_ops.inverse_ntt_montgomery_power_of_two_radix_compact_(
            operand,
            outer,
            roots,
            params,
        )

    def inverse_to_standard_lazy_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        outer, roots, params = self._active_native_inputs(
            operand,
            self.inverse_outer_twiddles,
            self.inverse_radix_root_powers,
            parameter_row_start,
        )
        ntt_ops.inverse_ntt_to_standard_lazy_power_of_two_radix_compact_(
            operand,
            outer,
            roots,
            params,
        )

    def inverse_to_standard_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        outer, roots, params = self._active_native_inputs(
            operand,
            self.inverse_outer_twiddles,
            self.inverse_radix_root_powers,
            parameter_row_start,
        )
        ntt_ops.inverse_ntt_to_standard_power_of_two_radix_compact_(
            operand,
            outer,
            roots,
            params,
        )

    def inverse_to_centered_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        outer, roots, params = self._active_native_inputs(
            operand,
            self.inverse_outer_twiddles,
            self.inverse_radix_root_powers,
            parameter_row_start,
        )
        ntt_ops.inverse_ntt_to_centered_power_of_two_radix_compact_(
            operand,
            outer,
            roots,
            params,
        )
