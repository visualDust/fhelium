"""Mutable data shared across one Compile request."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

_T = TypeVar("_T")


class WorkspaceKey(str, Generic[_T]):
    """Name a workspace entry whose value has type ``_T``.

    The key compares and hashes as its string name.
    """

    __slots__ = ()


class CompileWorkspace(dict[object, object]):
    """Store shared entries for one compilation request.

    Each consumer defines the format of its entries. Capture and every pass in the selected pipeline receive the same workspace. A class key stores an instance of that class; a :class:`WorkspaceKey` stores a value of its declared type.
    """

    def __init__(
        self,
        values: Mapping[Any, Any] | None = None,
        /,
        **named_values: object,
    ) -> None:
        super().__init__(values or (), **named_values)

    if TYPE_CHECKING:

        @overload
        def __getitem__(self, key: type[_T] | WorkspaceKey[_T], /) -> _T: ...
        @overload
        def __getitem__(self, key: object, /) -> object: ...
        def __getitem__(self, key: object, /) -> object: ...

        @overload
        def get(self, key: type[_T] | WorkspaceKey[_T], /) -> _T | None: ...
        @overload
        def get(
            self, key: type[_T] | WorkspaceKey[_T], default: _T, /
        ) -> _T: ...
        @overload
        def get(self, key: object, default: object = None, /) -> object: ...
        def get(self, key: object, default: object = None, /) -> object: ...

        @overload
        def setdefault(
            self, key: type[_T] | WorkspaceKey[_T], default: _T, /
        ) -> _T: ...
        @overload
        def setdefault(
            self, key: object, default: object = None, /
        ) -> object: ...
        def setdefault(
            self, key: object, default: object = None, /
        ) -> object: ...


__all__ = ["CompileWorkspace", "WorkspaceKey"]
