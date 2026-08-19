"""Deterministic synthetic inputs shared by benchmark workloads."""

from __future__ import annotations

import torch

from fhelium import CkksEngine


def ckks_message(engine: CkksEngine, *, phase: float = 0.0) -> torch.Tensor:
    """Create a bounded real message spanning every available CKKS slot."""

    slots = torch.linspace(
        -0.02,
        0.02,
        engine.num_slots,
        dtype=torch.float64,
    )
    return torch.sin(slots * 31.0 + phase) * 0.02
