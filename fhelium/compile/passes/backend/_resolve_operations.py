"""Resolve every executable Program operation to a Backend call."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    DecisionRecord,
    PassResult,
)

from dataclasses import dataclass

from fhelium.backend.execution import (
    OperationDispatch,
    ProgramDispatchTable,
)
from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    PreparingOperationImplementation,
    operation_invocation,
    requested_implementation,
)

from ._operations import executable_operations
from ._validate_representations import ValidateExecutionRepresentationsPass


@dataclass(frozen=True)
class ResolveBackendOperationsPass:
    """Require Backend support and build the Program dispatch table."""

    registry: OperationImplementationRegistry
    in_place: bool = False
    name: str = "resolve-backend-operations"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        ValidateExecutionRepresentationsPass().run(compilation)
        operations = executable_operations(program)
        if self.in_place and len(operations) != 1:
            raise ValueError(
                "In-place execution requires a one-operation Program"
            )

        dispatches = {}
        decisions: list[DecisionRecord] = []
        for operation in operations:
            invocation = operation_invocation(operation)
            implementation = self.registry.resolve(
                operation,
                requested=requested_implementation(operation),
                in_place=self.in_place,
            )
            prepared_regions = False
            if isinstance(implementation, PreparingOperationImplementation):
                implementation = implementation.prepare_operation(operation)
                prepared_regions = bool(operation.regions)
            dispatches[operation] = OperationDispatch(
                invocation,
                implementation,
                implementation.resource_requirements(invocation),
                self.registry.effect(type(operation)),
                self.in_place,
                prepared_regions,
            )
            decisions.append(
                DecisionRecord(
                    operation.name,
                    implementation.name,
                    self.registry.available(type(operation)),
                )
            )

        shared_data[ProgramDispatchTable] = ProgramDispatchTable(dispatches)
        return PassResult.unchanged(
            program,
            matched=len(operations),
            decisions=tuple(decisions),
        )


__all__ = ["ResolveBackendOperationsPass"]
