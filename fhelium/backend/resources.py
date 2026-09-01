"""Link symbolic resource requirements to live Backend objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ResourceRequirement:
    """Identify one named resource required by a logical operation."""

    symbol: str
    kind: str


@runtime_checkable
class ResourceMaterializer(Protocol):
    """Create missing live resources for one whole Program."""

    def materialize(
        self,
        requirements: tuple[ResourceRequirement, ...],
        /,
    ) -> ResourceBindings:
        """Return bindings that satisfy the requested missing resources."""

        ...


@dataclass(frozen=True, eq=False)
class BoundResource:
    """Associate one Program resource symbol with its live object."""

    symbol: str
    kind: str
    value: object = field(repr=False)


class ResourceBindings:
    """Hold low-level live bindings after resource-role matching.

    Compile callers normally supply ordinary domain objects to a binding pass
    or a resource materializer. Direct construction remains available for
    named roles that cannot be inferred from Program structure.
    """

    def __init__(self, resources: Sequence[BoundResource] = ()) -> None:
        entries: dict[str, BoundResource] = {}
        for resource in resources:
            if resource.symbol in entries:
                raise ValueError(
                    f"Resource symbol {resource.symbol!r} is bound more than once"
                )
            entries[resource.symbol] = resource
        self._entries: Mapping[str, BoundResource] = MappingProxyType(entries)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return bound symbols in declaration order."""

        return tuple(self._entries)

    @property
    def resources(self) -> tuple[BoundResource, ...]:
        """Return live bindings in declaration order."""

        return tuple(self._entries.values())

    def select(self, symbols: Sequence[str]) -> ResourceBindings:
        """Return selected existing bindings in caller order."""

        try:
            return ResourceBindings(
                tuple(self._entries[symbol] for symbol in symbols)
            )
        except KeyError as error:
            raise KeyError(
                f"Resource symbol {error.args[0]!r} is not bound"
            ) from None

    def overlay(
        self,
        resources: ResourceBindings | Sequence[BoundResource],
        *,
        override: bool = False,
    ) -> ResourceBindings:
        """Return this table extended by another set of bindings."""

        additions = (
            resources.resources
            if isinstance(resources, ResourceBindings)
            else tuple(resources)
        )
        entries = dict(self._entries)
        for resource in additions:
            if resource.symbol in entries and not override:
                raise ValueError(
                    f"Resource symbol {resource.symbol!r} already exists; "
                    "pass override=True to replace it"
                )
            entries[resource.symbol] = resource
        return ResourceBindings(tuple(entries.values()))

    @classmethod
    def merge(
        cls,
        bindings: Sequence[ResourceBindings],
        *,
        override: bool = False,
    ) -> ResourceBindings:
        """Merge binding tables in order."""

        merged = cls()
        for binding in bindings:
            merged = merged.overlay(binding, override=override)
        return merged

    def resolve(self, requirement: ResourceRequirement) -> BoundResource:
        """Return the live object bound to one required symbol and kind."""

        try:
            resource = self._entries[requirement.symbol]
        except KeyError:
            raise KeyError(
                f"Required resource {requirement.symbol!r} is not bound"
            ) from None
        if resource.kind != requirement.kind:
            raise ValueError(
                f"Resource {requirement.symbol!r} has kind {resource.kind!r}; "
                f"required {requirement.kind!r}"
            )
        return resource

    def resolve_all(
        self,
        requirements: Sequence[ResourceRequirement],
    ) -> tuple[BoundResource, ...]:
        """Resolve requirements in operation declaration order."""

        return tuple(self.resolve(requirement) for requirement in requirements)


__all__ = [
    "BoundResource",
    "ResourceBindings",
    "ResourceMaterializer",
    "ResourceRequirement",
]
