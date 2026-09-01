"""Mutable data shared across one Compile request."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CompileWorkspace(dict[object, object]):
    """Store arbitrary entries shared across one Compile request.

    Keys and values have no framework-defined schema. Code that uses an entry
    defines its format. Capture and every pass in the selected pipeline receive
    the same workspace.
    """

    def __init__(
        self,
        values: Mapping[Any, Any] | None = None,
        /,
        **named_values: object,
    ) -> None:
        super().__init__(values or (), **named_values)


__all__ = ["CompileWorkspace"]
