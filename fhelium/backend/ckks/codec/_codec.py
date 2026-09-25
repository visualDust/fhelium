"""CKKS coefficient quantization and slot reconstruction over supplied tables."""

from __future__ import annotations

import torch
from . import _embedding


def encode_tensor(
    message: torch.Tensor,
    pre: torch.Tensor,
    twister: torch.Tensor,
    rounding_state: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    slots = _embedding.make_slot_tensor(message, pre.numel(), message.device)
    return _embedding.encode_slots(
        slots,
        pre=pre,
        twister=twister,
        rounding_state=rounding_state,
        scale=scale,
    )


def decode_tensor(
    coefficients: torch.Tensor,
    post: torch.Tensor,
    skewer: torch.Tensor,
    *,
    scale: float,
    is_real: bool,
) -> torch.Tensor:
    decoded = _embedding.decode_slots(
        coefficients, post=post, skewer=skewer, scale=scale
    )[..., : coefficients.size(-1) // 2]
    return decoded.real if is_real else decoded


__all__ = ["decode_tensor", "encode_tensor"]
