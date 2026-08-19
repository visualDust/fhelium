"""Compact radix-2 NTT plan construction."""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig
from fhelium.engine.ntt.plans.twiddles import build_compact_twiddles


class CompactRadix2NttPlan:
    """Canonical compact forward and inverse standard-residue twiddle rows.

    Compact CUDA kernels compute butterfly indices in-kernel and consume one
    integral ``[prime, coefficient]`` twiddle table per transform direction.
    Prime rows follow ``ckks_config.moduli`` exactly; final extent is $N$ and
    both tables use ``ckks_config.torch_dtype`` on ``device``. Construction is
    functional and the forward and inverse tables do not alias.
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
        self.forward_twiddles, self.inverse_twiddles = build_compact_twiddles(
            self.cfg.moduli,
            self.cfg.logN,
            self.cfg.torch_dtype,
            device=self.device,
        )

    def __str__(self) -> str:
        return (
            f"CompactRadix2NttPlan(logN={self.cfg.logN}, "
            f"N={self.cfg.N}, primes={len(self.cfg.moduli)}, "
            f"device={self.device}, "
            f"forward_twiddles_shape={tuple(self.forward_twiddles.shape)}, "
            f"inverse_twiddles_shape={tuple(self.inverse_twiddles.shape)})"
        )

    __repr__ = __str__
