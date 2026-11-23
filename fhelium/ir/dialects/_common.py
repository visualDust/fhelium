"""Shared IRDL building blocks for FHElium dialects."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Literal, Self, cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    f64,
)
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import param_def

ValueRole = Literal["encrypted", "message", "plaintext", "static"]


def attribute_key(value: object) -> tuple[object, ...] | None:
    """Return the typed key of a metadata value, or None when it has no attribute form.

    The key separates ``bool``, ``int``, ``float`` and ``str`` values that compare equal in Python, and records floats by their IEEE-754 bits so signed zeros and NaN payloads remain distinct.
    """

    if isinstance(value, str):
        return ("s", str.__str__(value))
    if isinstance(value, bool):
        return ("b", int(value))
    if isinstance(value, int):
        return ("i", int(value))
    if isinstance(value, float):
        return ("f", struct.pack("<d", value))
    if isinstance(value, tuple):
        items = tuple(attribute_key(item) for item in value)
        if any(item is None for item in items):
            return None
        return ("t", items)
    return None


@lru_cache(maxsize=1 << 16)
def interned_attribute(key: tuple[object, ...]) -> Attribute:
    """Construct the attribute for an ``attribute_key`` once and share it.

    xDSL attributes are immutable values, so equal metadata may share one object, as attribute uniquing does in an MLIR context. Strings map to ``StringAttr``, booleans to ``i1``, integers to ``i64``, floats to ``f64`` and tuples to ``ArrayAttr``.
    """

    kind, data = key
    if kind == "s":
        return StringAttr(cast(str, data))
    if kind == "b":
        return IntegerAttr(cast(int, data), 1)
    if kind == "i":
        return IntegerAttr(cast(int, data), 64)
    if kind == "f":
        return FloatAttr(struct.unpack("<d", cast(bytes, data))[0], f64)
    return ArrayAttr(
        interned_attribute(item)
        for item in cast(tuple[tuple[object, ...], ...], data)
    )


def state_attribute(value: object) -> Attribute | None:
    """Return the shared builtin attribute for one supported metadata field."""

    key = attribute_key(value)
    return None if key is None else interned_attribute(key)


def dictionary_state(
    state: DictionaryAttr | Mapping[str, Attribute] | None,
) -> DictionaryAttr:
    """Return an immutable xDSL dictionary for partial value state."""

    if state is None:
        return DictionaryAttr({})
    if isinstance(state, DictionaryAttr):
        return state
    return DictionaryAttr(state)


@dataclass(frozen=True, init=False)
class OpenStateType(ParametrizedAttribute, TypeAttribute):
    """Base type for a value whose compile-time state may be incomplete."""

    ROLE: ClassVar[ValueRole | None] = None
    state: DictionaryAttr = cast(DictionaryAttr, param_def())

    def __init__(
        self,
        state: DictionaryAttr | Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(dictionary_state(state))

    def with_state(
        self,
        updates: Mapping[str, Attribute | None] | None = None,
        /,
        **fields: Attribute | None,
    ) -> Self:
        """Return the same concrete type with copied, updated state fields.

        An update value of ``None`` removes that field. The source type and its
        immutable ``DictionaryAttr`` remain unchanged.
        """

        state = dict(self.state.data)
        for name, value in {**dict(updates or {}), **fields}.items():
            if value is None:
                state.pop(name, None)
            else:
                state[name] = value
        return cast(Self, type(self).new((DictionaryAttr(state),)))


__all__ = [
    "OpenStateType",
    "ValueRole",
    "attribute_key",
    "dictionary_state",
    "interned_attribute",
    "state_attribute",
]
