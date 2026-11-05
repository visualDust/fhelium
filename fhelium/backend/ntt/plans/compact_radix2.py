"""Compact radix-2 NTT plan construction."""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig
from fhelium.backend.rns.format import RnsExecutionFormat
from fhelium.backend.ntt.plans.twiddles import (
    build_compact_twiddles,
    encode_compact_shoup32,
)


class CompactRadix2NttPlan:
    """Construct bit-reversed compact roots in the selected table encoding.

    Montgomery-policy plans provide standard ``[prime,N]`` roots for later
    resource preparation. Shoup32 plans provide ready ``[prime,N,2]`` root and
    quotient pairs. Rows follow the configuration's prime order.
    """

    def __init__(
        self,
        ckks_config: CkksConfig,
        *,
        device: str | int | torch.device | None = None,
        dtype: torch.dtype | None = None,
        twiddle_encoding: str = "montgomery",
    ) -> None:
        self.cfg = ckks_config
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.forward_twiddles, self.inverse_twiddles = build_compact_twiddles(
            self.cfg.moduli,
            self.cfg.logN,
            RnsExecutionFormat.select(self.cfg.moduli, dtype).dtype,
            device=self.device,
        )

        if twiddle_encoding == "shoup32":
            self.forward_twiddles = encode_compact_shoup32(
                self.forward_twiddles, self.cfg.moduli
            )
            self.inverse_twiddles = encode_compact_shoup32(
                self.inverse_twiddles, self.cfg.moduli
            )
        elif twiddle_encoding != "montgomery":
            raise ValueError("Unsupported compact twiddle encoding")

    def __str__(self) -> str:
        return (
            f"CompactRadix2NttPlan(logN={self.cfg.logN}, "
            f"N={self.cfg.N}, primes={len(self.cfg.moduli)}, "
            f"device={self.device}, "
            f"forward_twiddles_shape={tuple(self.forward_twiddles.shape)}, "
            f"inverse_twiddles_shape={tuple(self.inverse_twiddles.shape)})"
        )

    __repr__ = __str__
