"""Start one Backend build with a fresh live-resource table."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    PassResult,
)

from dataclasses import dataclass, field

from fhelium.backend.resources import ResourceBindings


@dataclass(frozen=True)
class InitializeResourceBindingsPass:
    """Replace resource state left by an earlier build with this build's base."""

    resources: ResourceBindings = field(default_factory=ResourceBindings)
    name: str = "initialize-resource-bindings"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        shared_data[ResourceBindings] = self.resources
        return PassResult.unchanged(program)


__all__ = ["InitializeResourceBindingsPass"]
