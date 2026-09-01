"""Deterministic synthetic inputs shared by benchmark workloads."""

from __future__ import annotations

from fhelium.legacy.engine import CkksEngine

import torch


def ckks_message(engine: CkksEngine, *, phase: float = 0.0) -> torch.Tensor:
    """Create a real message with magnitudes at most 0.02 in every CKKS slot."""

    slots = torch.linspace(
        -0.02,
        0.02,
        engine.num_slots,
        dtype=torch.float64,
    )
    return torch.sin(slots * 31.0 + phase) * 0.02
