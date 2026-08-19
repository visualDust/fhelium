"""Fixed power-of-two-radix NTT plan construction."""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig
from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    validate_ntt_backend_for_log_n,
)
from fhelium.engine.ntt.plans.twiddles import build_power_of_two_radix_twiddles


class CompactPowerOfTwoRadixNttPlan:
    """Radix-specific outer twists and fixed cyclic-root powers.

    A radix-``R`` digit evaluates one twisted cyclic ``R``-point NTT. Radix-4,
    radix-8, and radix-16 use dedicated butterflies. Every digit has the exact
    policy radix, and incompatible ring dimensions are rejected. This does not
    alias grouped radix-2 execution.

    Outer tables have integral shape ``[prime, N - 1]`` and root-power tables
    have shape ``[prime, radix]`` on ``device`` in standard representation.
    Prime rows follow ``ckks_config.moduli`` exactly; the four tables are
    separately allocated and do not alias.
    """

    def __init__(
        self,
        ckks_config: CkksConfig,
        policy: CompactFixedRadixPolicy,
        *,
        device: str | int | torch.device | None = None,
    ) -> None:
        self.cfg = ckks_config
        self.policy = policy
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        validate_ntt_backend_for_log_n(policy, self.cfg.logN)
        (
            self.forward_outer_twiddles,
            self.inverse_outer_twiddles,
            self.forward_radix_root_powers,
            self.inverse_radix_root_powers,
        ) = build_power_of_two_radix_twiddles(
            self.cfg.moduli,
            self.cfg.logN,
            self.policy.radix,
            self.cfg.torch_dtype,
            device=self.device,
        )

    def __str__(self) -> str:
        return (
            f"CompactPowerOfTwoRadixNttPlan(logN={self.cfg.logN}, "
            f"radix={self.policy.radix}, "
            f"outer_twiddle_shape={tuple(self.forward_outer_twiddles.shape)}, "
            f"device={self.device})"
        )

    __repr__ = __str__
