"""Capture the Torch execution modes that specialize a Python invocation."""

from __future__ import annotations

import torch


def execution_context() -> tuple[object, ...]:
    """Describe grad, inference, and autocast modes without Tensor contents."""
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cpu"),
        str(torch.get_autocast_dtype("cpu")),
        torch.is_autocast_enabled("cuda"),
        str(torch.get_autocast_dtype("cuda")),
        torch.is_autocast_cache_enabled(),
        torch.cuda.current_device() if torch.cuda.is_initialized() else None,
    )
