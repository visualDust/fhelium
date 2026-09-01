"""Graph-external constants stored during source capture and compilation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from types import MappingProxyType


class ConstantBundle(MutableMapping[str, object]):
    """Own graph-external constants captured by compiler frontends."""

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        self._values: dict[str, object] = {}
        for key, value in (values or {}).items():
            self[key] = value

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __setitem__(self, key: str, value: object) -> None:
        if not isinstance(key, str):
            raise TypeError("ConstantBundle keys must be strings")
        if not key:
            raise ValueError("ConstantBundle keys must be non-empty")
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def view(self) -> Mapping[str, object]:
        """Return a read-only view of the current symbolic constants."""

        return MappingProxyType(self._values)

    def clone(self) -> ConstantBundle:
        """Clone the symbol map without copying caller-owned payload objects."""

        return ConstantBundle(self._values)


__all__ = ["ConstantBundle"]
