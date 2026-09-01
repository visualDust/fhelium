"""Start one Backend build with a fresh live-resource table."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
)

from dataclasses import dataclass, field

from fhelium.backend.resources import ResourceBindings
from fhelium.ir import Program


@dataclass(frozen=True)
class InitializeResourceBindingsPass:
    """Replace resource state left by an earlier build with this build's base."""

    resources: ResourceBindings = field(default_factory=ResourceBindings)
    name: str = "initialize-resource-bindings"

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        shared_data[ResourceBindings] = self.resources
        return PassResult.unchanged(program)


__all__ = ["InitializeResourceBindingsPass"]
