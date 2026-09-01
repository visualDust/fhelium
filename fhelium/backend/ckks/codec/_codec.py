"""CKKS message, coefficient, and RNS plaintext conversion algorithms.

Registered operation implementations supply the execution placement, random
stream, and RNS resources for each call.
"""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig
from fhelium.rng import Csprng

from . import _embedding


def encode_tensor(
    message: torch.Tensor,
    *,
    config: CkksConfig,
    rng: Csprng,
    scale: float,
) -> torch.Tensor:
    """Encode a message Tensor on its current device into coefficients."""

    slots = _embedding.make_slot_tensor(
        message,
        config.num_slots,
        message.device,
    )
    return _embedding.encode_slots(
        slots,
        rng=rng,
        scale=scale,
        device=message.device,
        generator=config.galois_generator,
    )


def decode_tensor(
    coefficients: torch.Tensor,
    *,
    config: CkksConfig,
    scale: float,
    is_real: bool,
) -> torch.Tensor:
    """Decode coefficients into slots on the coefficient Tensor device."""

    decoded = _embedding.decode_slots(
        coefficients,
        scale=scale,
        generator=config.galois_generator,
    )[..., : config.num_slots]
    return decoded.real if is_real else decoded


__all__ = ["decode_tensor", "encode_tensor"]
