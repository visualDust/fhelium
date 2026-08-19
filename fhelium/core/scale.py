"""Validation helpers for per-value CKKS scales."""

from __future__ import annotations

import math
from typing import Any

from fhelium.errors import InvalidScaleError


def coerce_scale(value: Any, *, value_name: str) -> float:
    r"""Validate and convert one actual CKKS scale to binary64.

    The result is a Python :class:`float` satisfying
    $0 < \Delta(v) < \infty$. This conversion does not normalize the scale to
    ``default_scale`` or alter any value payload. Centralizing the conversion
    prevents construction and operation paths from assigning different
    meanings to zero, infinities, NaNs, or booleans.
    """

    if isinstance(value, (bool, str, bytes)):
        raise InvalidScaleError(value_name=value_name, scale=value)
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidScaleError(value_name=value_name, scale=value) from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise InvalidScaleError(value_name=value_name, scale=value)
    return scale


__all__ = ["coerce_scale"]
