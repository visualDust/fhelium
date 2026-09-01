"""Add constructible resources required by resolved Backend operations."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
)

from dataclasses import dataclass

from fhelium.backend.execution import ProgramDispatchTable
from fhelium.backend.resources import (
    ResourceBindings,
    ResourceMaterializer,
)
from fhelium.ir import Program

from ._operations import program_resource_requirements


@dataclass(frozen=True)
class MaterializeResourcesPass:
    """Create missing constructible resources once for the whole Program."""

    materializer: ResourceMaterializer
    name: str = "materialize-backend-resources"

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        dispatch_table = shared_data.get(ProgramDispatchTable)
        if not isinstance(dispatch_table, ProgramDispatchTable):
            raise RuntimeError(
                "MaterializeResourcesPass requires ResolveBackendOperationsPass"
            )

        provided = shared_data.get(ResourceBindings)
        if not isinstance(provided, ResourceBindings):
            raise RuntimeError(
                "MaterializeResourcesPass requires "
                "InitializeResourceBindingsPass"
            )
        bindings = provided
        _, requirements = program_resource_requirements(
            program,
            dispatch_table,
        )
        bound_symbols = frozenset(bindings.symbols)
        missing = [
            requirement
            for requirement in requirements
            if requirement.symbol not in bound_symbols
        ]

        if missing:
            created = self.materializer.materialize(
                tuple(missing),
            )
            bindings = bindings.overlay(created)

        shared_data[ResourceBindings] = bindings
        return PassResult.unchanged(
            program,
            matched=len(missing),
        )


__all__ = ["MaterializeResourcesPass"]
