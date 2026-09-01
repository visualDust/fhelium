"""Link Program material and resource references to live objects."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from fhelium.backend.execution import ProgramDispatchTable, ProgramExecutable
from fhelium.backend.resources import ResourceBindings
from fhelium.ir import Program

from ..._constants import ConstantBundle
from ._operations import (
    program_material_values,
    program_resource_requirements,
)


@dataclass(frozen=True)
class LinkProgramPass:
    """Match all external references and produce the Program executable."""

    material_overrides: Mapping[str, object] = field(default_factory=dict)
    name: str = "link-backend-program"

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
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
        captured = shared_data.get(ConstantBundle)
        if captured is None:
            materials: dict[str, object] = {}
        elif isinstance(captured, ConstantBundle):
            materials = dict(captured.view())
        else:
            raise TypeError("Backend ConstantBundle entry has another type")
        materials.update(self.material_overrides)
        material_values = program_material_values(program, materials)
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
