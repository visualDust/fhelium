"""Mutable data shared across one Compile request."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CompileWorkspace(dict[object, object]):
    """Store shared entries for one compilation request.

    Each consumer defines the format of its entries. Capture and every pass in the selected pipeline receive the same workspace.
    """

    def __init__(
        self,
        values: Mapping[Any, Any] | None = None,
        /,
        **named_values: object,
    ) -> None:
        super().__init__(values or (), **named_values)


__all__ = ["CompileWorkspace"]
