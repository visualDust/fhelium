"""Indexed radix-2 plan construction."""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig
from fhelium.engine.ntt.plans.twiddles import (
    build_compact_twiddles,
    expand_stage_twiddles,
)


class IndexedRadix2NttPlan:
    """Radix-2 schedules used by the indexed backend.

    Forward and inverse indices have shape ``[2, logN, N/2]``. Expanded
    twiddles contain only the nontrivial odd lane and have shape
    ``[prime, logN, N/2]``. No all-one even-lane tensor is allocated.
    Schedules use ``torch.int32``; standard-residue twiddles use the config's
    integral dtype. All tensors are on ``device`` and prime rows follow
    ``ckks_config.moduli`` exactly. These immutable plan tables do not alias.
    """

    def __init__(
        self,
        ckks_config: CkksConfig,
        *,
        device: str | int | torch.device | None = None,
    ) -> None:
        self.cfg = ckks_config
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        (
            forward_indices,
            forward_twiddle_indices,
            inverse_indices,
            inverse_twiddle_indices,
        ) = self._build_indexed_schedule()
        self.forward_indices = forward_indices.to(self.device)
        self.inverse_indices = inverse_indices.to(self.device)

        compact_forward, compact_inverse = build_compact_twiddles(
            self.cfg.moduli,
            self.cfg.logN,
            self.cfg.torch_dtype,
            device=self.device,
        )
        self.forward_twiddles = expand_stage_twiddles(
            compact_forward, forward_twiddle_indices
        )
        self.inverse_twiddles = expand_stage_twiddles(
            compact_inverse, inverse_twiddle_indices
        )

    def _build_indexed_schedule(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build decimation-in-frequency/in-time schedules."""

        stage_count = self.cfg.logN
        ring_dimension = self.cfg.N
        butterfly_count = ring_dimension // 2

        forward_even = torch.empty(
            (stage_count, butterfly_count), dtype=torch.int32
        )
        forward_odd = torch.empty_like(forward_even)
        forward_twiddle_indices = torch.empty_like(forward_even)

        half_span = ring_dimension
        for stage in range(stage_count):
            group_count = 1 << stage
            half_span //= 2
            even_indices: list[int] = []
            odd_indices: list[int] = []
            twiddle_indices: list[int] = []
            for group in range(group_count):
                first = 2 * group * half_span
                last = first + half_span
                twiddle_index = group_count + group
                for even_index in range(first, last):
                    even_indices.append(even_index)
                    odd_indices.append(even_index + half_span)
                    twiddle_indices.append(twiddle_index)
            forward_even[stage] = torch.tensor(even_indices, dtype=torch.int32)
            forward_odd[stage] = torch.tensor(odd_indices, dtype=torch.int32)
            forward_twiddle_indices[stage] = torch.tensor(
                twiddle_indices, dtype=torch.int32
            )

        inverse_even = torch.empty_like(forward_even)
        inverse_odd = torch.empty_like(forward_even)
        inverse_twiddle_indices = torch.empty_like(forward_even)

        half_span = 1
        for remaining_stage_count in range(stage_count, 0, -1):
            stage = stage_count - remaining_stage_count
            group_count = 1 << (remaining_stage_count - 1)
            even_indices = []
            odd_indices = []
            twiddle_indices = []
            first = 0
            for group in range(group_count):
                last = first + half_span
                twiddle_index = group_count + group
                for even_index in range(first, last):
                    even_indices.append(even_index)
                    odd_indices.append(even_index + half_span)
                    twiddle_indices.append(twiddle_index)
                first += 2 * half_span
            half_span *= 2
            inverse_even[stage] = torch.tensor(even_indices, dtype=torch.int32)
            inverse_odd[stage] = torch.tensor(odd_indices, dtype=torch.int32)
            inverse_twiddle_indices[stage] = torch.tensor(
                twiddle_indices, dtype=torch.int32
            )

        return (
            torch.stack((forward_even, forward_odd)),
            forward_twiddle_indices,
            torch.stack((inverse_even, inverse_odd)),
            inverse_twiddle_indices,
        )

    def __str__(self) -> str:
        return (
            f"IndexedRadix2NttPlan(logN={self.cfg.logN}, N={self.cfg.N}, "
            f"primes={len(self.cfg.moduli)}, device={self.device}, "
            f"forward_indices_shape={tuple(self.forward_indices.shape)}, "
            f"forward_twiddles_shape={tuple(self.forward_twiddles.shape)}, "
            f"inverse_indices_shape={tuple(self.inverse_indices.shape)}, "
            f"inverse_twiddles_shape={tuple(self.inverse_twiddles.shape)})"
        )

    __repr__ = __str__
