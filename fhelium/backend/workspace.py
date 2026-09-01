"""Live inputs owned by one operation Backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from fhelium.values import KeySwitchKey, PublicKey, SecretKey

from .resources import ResourceBindings, ResourceMaterializer


@dataclass(frozen=True)
class BackendWorkspace:
    """Hold one Backend instance's keys, resources, and material choices."""

    named_resources: ResourceBindings = field(default_factory=ResourceBindings)
    keys: tuple[PublicKey | SecretKey | KeySwitchKey, ...] = ()
    materializer: ResourceMaterializer | None = None
    material_overrides: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(
            self,
            "material_overrides",
            MappingProxyType(dict(self.material_overrides)),
        )

    def with_named_resources(
        self,
        resources: ResourceBindings,
        *,
        override: bool = True,
    ) -> BackendWorkspace:
        """Return this workspace with an overlaid named-resource table."""

        return replace(
            self,
            named_resources=self.named_resources.overlay(
                resources,
                override=override,
            ),
        )


__all__ = ["BackendWorkspace"]
