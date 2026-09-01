"""Provider-local region compilers used by source-operation assignment."""

from __future__ import annotations


from collections.abc import Sequence
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation

from fhelium.ir import (
    OperationSpec,
    OperationSpecRegistry,
    Program,
    operation_name,
)

from .._contracts import CoverageDiagnostic
from .._execution import ExecutionInputs

from .._bindings import RuntimeBindings
from ..planning._records import CompiledRegion, LoweredRegion, RegionProposal


@runtime_checkable
class RegionCompiler(Protocol):
    """Check, lower, verify, and build provider-owned source regions."""

    @property
    def name(self) -> str:
        """Return the implementation identity recorded in plans."""

        ...

    @property
    def operation_types(self) -> tuple[type[Operation], ...]:
        """Return the operation classes implemented by this object."""

        ...

    def diagnostics(
        self,
        operation: Operation,
        specification: OperationSpec,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        """Check provider-local requirements for one source operation."""

        ...

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        """Check one adjacent assignment before provider selection."""

        ...

    def lower(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> LoweredRegion:
        """Lower one proposed source operation."""

        ...

    def verify(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        """Verify one lowered operation implementation."""

        ...

    def build(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> CompiledRegion:
        """Build one verified operation implementation."""

        ...


class RegionCompilerRegistry:
    """Map source operation classes and plan names to provider region compilers."""

    def __init__(self, compilers: Sequence[RegionCompiler] = ()) -> None:
        by_type: dict[type[Operation], list[RegionCompiler]] = {}
        by_name: dict[str, RegionCompiler] = {}
        for compiler in compilers:
            if not isinstance(compiler, RegionCompiler):
                raise TypeError("Region compiler does not satisfy its protocol")
            existing = by_name.get(compiler.name)
            if existing is not None and existing is not compiler:
                raise ValueError(
                    f"Region compiler {compiler.name!r} is duplicated"
                )
            by_name[compiler.name] = compiler
            for operation_type in compiler.operation_types:
                by_type.setdefault(operation_type, []).append(compiler)
        self._by_type = MappingProxyType(
            {
                operation_type: tuple(entries)
                for operation_type, entries in by_type.items()
            }
        )
        self._by_name = MappingProxyType(by_name)

    @property
    def operation_types(self) -> tuple[type[Operation], ...]:
        """Return registered operation classes in insertion order."""

        return tuple(self._by_type)

    @property
    def names(self) -> tuple[str, ...]:
        """Return distinct implementation names in insertion order."""

        return tuple(self._by_name)

    def get_for_operation(
        self, operation: Operation
    ) -> tuple[RegionCompiler, ...]:
        """Return region compilers registered for an operation class."""

        return self._by_type.get(type(operation), ())

    def require_name(self, name: str) -> RegionCompiler:
        """Return one named region compiler."""

        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"Region compiler {name!r} is not registered; "
                f"available={self.names}"
            ) from None


class BackendProvider:
    """Assign registered operation implementations for one execution target."""

    def __init__(
        self,
        name: str,
        target: str,
        *,
        regions: RegionCompilerRegistry | None = None,
    ) -> None:
        if not name:
            raise ValueError("BackendProvider name must be non-empty")
        if target not in {"cpu", "cuda"}:
            raise ValueError("BackendProvider target must be cpu or cuda")
        self.name = name
        self.target = target
        self.regions = regions or RegionCompilerRegistry()

    def propose(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        specifications: OperationSpecRegistry,
    ) -> tuple[RegionProposal, ...]:
        """Propose the registered implementation for each covered operation."""

        operations = tuple(
            operation
            for operation in program.single_block("main").ops
            if not isinstance(operation, ReturnOp)
        )
        proposals: list[RegionProposal] = []
        for index, operation in enumerate(operations):
            compilers = self.regions.get_for_operation(operation)
            if not compilers:
                continue
            specification = specifications.get(operation_name(operation))
            for compiler in compilers:
                diagnostics: list[CoverageDiagnostic] = []
                if specification is None:
                    diagnostics.append(
                        CoverageDiagnostic(
                            "unknown-operation-spec",
                            f"Operation {operation_name(operation)!r} has no semantic specification.",
                            f"main:{index}",
                        )
                    )
                else:
                    diagnostics.extend(
                        CoverageDiagnostic(
                            "operation-spec-mismatch",
                            message,
                            f"main:{index}",
                        )
                        for message in specification.diagnostics(operation)
                    )
                    diagnostics.extend(
                        compiler.diagnostics(
                            operation,
                            specification,
                            execution=execution,
                            bindings=bindings,
                        )
                    )
                if execution.device.type != self.target:
                    diagnostics.append(
                        CoverageDiagnostic(
                            "provider-rejected-operation",
                            f"Provider {self.name!r} requires target {self.target!r}.",
                            self.name,
                        )
                    )
                proposals.append(
                    RegionProposal(
                        self.name,
                        compiler.name,
                        (f"main:{index}",),
                        (index,),
                        priority=getattr(compiler, "priority", 0),
                        diagnostics=tuple(diagnostics),
                        metadata={
                            "execution": "provider-region-compiler",
                            "merge_adjacent": getattr(
                                compiler, "merge_adjacent", True
                            ),
                        },
                    )
                )
        return tuple(proposals)

    def _compiler(self, proposal: RegionProposal) -> RegionCompiler:
        if proposal.provider != self.name:
            raise ValueError(
                f"Proposal provider {proposal.provider!r} differs from {self.name!r}"
            )
        return self.regions.require_name(proposal.implementation)

    def lower(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> LoweredRegion:
        """Delegate lowering to the selected registered implementation."""

        return self._compiler(proposal).lower(
            program, proposal, execution=execution, bindings=bindings
        )

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        """Check one adjacent candidate without lowering or execution."""

        return self._compiler(proposal).region_diagnostics(
            program, proposal, execution=execution, bindings=bindings
        )

    def verify(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        """Delegate verification to the selected registered implementation."""

        return self._compiler(lowered.proposal).verify(
            lowered, execution=execution, bindings=bindings
        )

    def build(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> CompiledRegion:
        """Delegate build to the selected registered implementation."""

        return self._compiler(lowered.proposal).build(
            lowered, execution=execution, bindings=bindings
        )


__all__ = [
    "BackendProvider",
    "RegionCompiler",
    "RegionCompilerRegistry",
]
