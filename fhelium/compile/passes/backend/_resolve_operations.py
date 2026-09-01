"""Resolve every executable Program operation to a Backend call."""

from __future__ import annotations

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
    operation_invocation,
    requested_implementation,
)
from fhelium.ir import Program

from ._operations import executable_operations


@dataclass(frozen=True)
class ResolveBackendOperationsPass:
    """Require Backend support and build the Program dispatch table."""

    registry: OperationImplementationRegistry
    in_place: bool = False
    name: str = "resolve-backend-operations"

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
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
            dispatches[operation] = OperationDispatch(
                invocation,
                implementation,
                implementation.resource_requirements(invocation),
                self.registry.effect(type(operation)),
                self.in_place,
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
