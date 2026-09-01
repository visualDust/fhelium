"""Shared IRDL building blocks for FHElium dialects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Literal, Self, cast

from xdsl.dialects.builtin import DictionaryAttr
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import param_def

ValueRole = Literal["encrypted", "message", "plaintext", "static"]


def dictionary_state(
    state: DictionaryAttr | Mapping[str, Attribute] | None,
) -> DictionaryAttr:
    """Return an immutable xDSL dictionary for partial value state."""

    if state is None:
        return DictionaryAttr({})
    if isinstance(state, DictionaryAttr):
        return state
    return DictionaryAttr(state)


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


__all__ = ["OpenStateType", "ValueRole", "dictionary_state"]
