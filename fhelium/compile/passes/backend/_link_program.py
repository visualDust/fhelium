"""Link Program material and resource references to live objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    PassResult,
)

from dataclasses import dataclass
from types import MappingProxyType


from fhelium.backend.execution import ProgramDispatchTable, ProgramExecutable
from fhelium.backend.resources import ResourceBindings

from ._operations import (
    program_material_values,
    program_resource_requirements,
)


@dataclass(frozen=True)
class LinkProgramPass:
    """Match all external references and produce the Program executable."""

    name: str = "link-backend-program"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        dispatch_table = shared_data.get(ProgramDispatchTable)
        if not isinstance(dispatch_table, ProgramDispatchTable):
            raise RuntimeError(
                "LinkProgramPass requires ResolveBackendOperationsPass"
            )

        provided = shared_data.get(ResourceBindings)
        if not isinstance(provided, ResourceBindings):
            raise RuntimeError(
                "LinkProgramPass requires InitializeResourceBindingsPass"
            )
        bindings = provided
        material_values = program_material_values(
            program, compilation.material_bindings
        )
        resource_references, requirements = program_resource_requirements(
            program,
            dispatch_table,
        )
        resources = bindings.resolve_all(requirements)
        indices = {
            requirement: index for index, requirement in enumerate(requirements)
        }
        resource_indices = {
            operation: tuple(
                indices[requirement] for requirement in dispatch.requirements
            )
            for operation, dispatch in dispatch_table.operations.items()
        }
        bound_resources = {
            operation: resources[indices[requirement]]
            for operation, requirement in resource_references.items()
        }
        shared_data[ResourceBindings] = bindings
        shared_data[ProgramExecutable] = ProgramExecutable(
            program,
            dispatch_table,
            resources,
            MappingProxyType(resource_indices),
            MappingProxyType(bound_resources),
            MappingProxyType(material_values),
        )
        return PassResult.unchanged(
            program,
            matched=len(requirements) + len(material_values),
        )


__all__ = ["LinkProgramPass"]
