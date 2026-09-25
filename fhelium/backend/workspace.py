"""Live inputs owned by one operation Backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


from .resources import ResourceBindings, ResourceMaterializer


@dataclass(frozen=True)
class BackendWorkspace:
    """Hold one Backend instance's non-Tensor execution handles."""

    named_resources: ResourceBindings = field(default_factory=ResourceBindings)
    materializer: ResourceMaterializer | None = None

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
