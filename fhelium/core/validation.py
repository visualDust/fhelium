"""Structural validation shared by CKKS value classes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch


def validate_nonnegative_level(level: object, *, value_name: str) -> int:
    """Require a non-bool, non-negative Python integer level."""

    if type(level) is not int:
        raise TypeError(
            f"{value_name} level must be an integer, got {type(level).__name__}"
        )
    if level < 0:
        raise ValueError(f"{value_name} level must be non-negative: {level}")
    return level


def validate_context_id(context_id: object, *, value_name: str) -> str:
    """Require a non-empty context identifier for a context-bound value."""

    if not isinstance(context_id, str) or not context_id:
        raise ValueError(f"{value_name} requires a non-empty context_id")
    return context_id


def validate_prime_ids(
    prime_ids: Iterable[object],
    *,
    value_name: str,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    """Require non-bool, non-negative, strictly ordered prime IDs."""

    try:
        values = tuple(prime_ids)
    except TypeError as error:
        raise TypeError(f"{value_name} prime_ids must be iterable") from error
    if not allow_empty and not values:
        raise ValueError(f"{value_name} prime_ids cannot be empty")
    if any(type(prime_id) is not int for prime_id in values):
        raise TypeError(
            f"{value_name} prime_ids must contain non-bool integers"
        )
    integer_values = cast(tuple[int, ...], values)
    if any(prime_id < 0 for prime_id in integer_values):
        raise ValueError(f"{value_name} prime_ids must be non-negative")
    if any(
        current >= following
        for current, following in zip(integer_values, integer_values[1:])
    ):
        raise ValueError(
            f"{value_name} prime_ids must be strictly increasing: {values}"
        )
    return integer_values


def validate_integral_tensor(
    tensor: object, *, value_name: str
) -> torch.Tensor:
    """Require a dense tensor with an integral, non-boolean scalar dtype."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{value_name} data must be a torch.Tensor")
    if tensor.layout != torch.strided:
        raise TypeError(f"{value_name} data must use dense strided storage")
    if (
        tensor.dtype == torch.bool
        or tensor.is_floating_point()
        or tensor.is_complex()
    ):
        raise TypeError(f"{value_name} data must use an integral scalar dtype")
    return tensor


__all__ = [
    "validate_context_id",
    "validate_integral_tensor",
    "validate_nonnegative_level",
    "validate_prime_ids",
]
