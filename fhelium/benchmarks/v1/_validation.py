"""Validation helpers for the fixed Benchmark v1 data formats."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Any

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class _FrozenJsonList(tuple[Any, ...]):
    """Immutable JSON array that compares by sequence contents."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(
            other, (str, bytes, bytearray)
        ):
            return tuple(self) == tuple(other)
        return False

    __hash__ = None  # type: ignore[assignment]


def strict_string(value: Any, field: str) -> str:
    """Return a non-empty string without silently coercing another type."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def strict_optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return strict_string(value, field)


def strict_timestamp(value: Any, field: str) -> str:
    """Return one timezone-aware ISO-8601 timestamp without coercion."""

    timestamp = strict_string(value, field)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return timestamp


def strict_optional_timestamp(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return strict_timestamp(value, field)


def strict_name(value: Any, field: str) -> str:
    value = strict_string(value, field)
    if _NAME.fullmatch(value) is None:
        raise ValueError(
            f"{field} must contain only letters, digits, '.', '_', and '-'"
        )
    return value


def normalize_json(value: Any, *, path: str = "value") -> Any:
    """Copy JSON data into canonical Python containers and reject extensions.

    The accepted value domain is deliberately narrower than ``json.dumps``:
    mapping keys must already be strings, sequences become lists, and every
    floating-point number must be finite.
    """

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            normalized[key] = normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def strict_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    normalized = normalize_json(value, path=field)
    assert isinstance(normalized, dict)
    return normalized


def freeze_json(value: Any, *, path: str = "value") -> Any:
    """Return a recursively immutable snapshot of finite JSON data."""

    normalized = normalize_json(value, path=path)

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType(
                {name: freeze(child) for name, child in item.items()}
            )
        if isinstance(item, list):
            return _FrozenJsonList(freeze(child) for child in item)
        return item

    return freeze(normalized)


def require_fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    model: str,
) -> None:
    """Require declared fields and reject unknown fields."""

    optional = optional or set()
    missing = required.difference(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{model} is missing required field(s): {names}")
    unknown = set(payload).difference(required | optional)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{model} contains unknown field(s): {names}")
