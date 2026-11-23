"""Deterministic synthetic inputs and device-placed engines for benchmark workloads."""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig, Preset
from fhelium.eager import Engine


def ckks_engine(
    config: CkksConfig | Preset,
    *,
    device: torch.device | str,
    ntt_backend: str | None = None,
) -> Engine:
    """Create an Eager Engine whose secret key, and therefore every automatically generated key, resides on ``device``."""

    engine = Engine(config, ntt_backend=ntt_backend)
    engine.set_secret_key(engine.create_secret_key(device=device))
    return engine


def ckks_message(engine: Engine, *, phase: float = 0.0) -> torch.Tensor:
    """Create a real CPU message with magnitudes at most 0.02 in every CKKS slot."""

    slots = torch.linspace(
        -0.02,
        0.02,
        engine.num_slots,
        dtype=torch.float64,
    )
    return torch.sin(slots * 31.0 + phase) * 0.02
