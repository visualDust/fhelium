"""Internal backend-assignment candidates and selected-region records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block, Operation, SSAValue

from fhelium.ir import Program

from .._contracts import CoverageDiagnostic, Executable


@dataclass(frozen=True)
class ValueRequirement:
    """Describe state a provider needs or produces for one SSA value.

    ``fields`` contains known or symbolic state properties such as level,
    scale, basis, polynomial domain, residue representation, component count,
    dtype, device, shape, and layout. ``runtime_guards`` names properties that
    must be checked against live objects before execution.
    """

    value_id: str
    fields: Mapping[str, object] = field(default_factory=dict)
    runtime_guards: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.value_id:
            raise ValueError("ValueRequirement value_id must be non-empty")
        if any(not name for name in self.runtime_guards):
            raise ValueError("Runtime guard names must be non-empty")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


@dataclass(frozen=True)
class RegionProposal:
    """Offer one provider implementation for a contiguous source interval."""

    provider: str
    implementation: str
    operation_ids: tuple[str, ...]
    operation_indices: tuple[int, ...]
    input_requirements: tuple[ValueRequirement, ...] = ()
    output_requirements: tuple[ValueRequirement, ...] = ()
    required_bindings: tuple[str, ...] = ()
    lowering_depth: str = "source"
    priority: int = 0
    diagnostics: tuple[CoverageDiagnostic, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not self.implementation:
            raise ValueError(
                "Proposal provider and implementation must be non-empty"
            )
        if not self.operation_ids:
            raise ValueError(
                "A region proposal must own at least one operation"
            )
        if len(self.operation_ids) != len(self.operation_indices):
            raise ValueError("Proposal operation IDs and indices must align")
        if tuple(sorted(set(self.operation_indices))) != self.operation_indices:
            raise ValueError(
                "Proposal operation indices must be unique and ordered"
            )
        start = self.operation_indices[0]
        if self.operation_indices != tuple(
            range(start, start + len(self.operation_indices))
        ):
            raise ValueError("Initial region proposals must be contiguous")
        if any(not name for name in self.required_bindings):
            raise ValueError("Required binding names must be non-empty")
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    @property
    def start(self) -> int:
        """Return the first owned operation index."""

        return self.operation_indices[0]

    @property
    def stop(self) -> int:
        """Return one past the last owned operation index."""

        return self.operation_indices[-1] + 1

    @property
    def supported(self) -> bool:
        """Whether the proposal contains no error diagnostic."""

        return not any(item.severity == "error" for item in self.diagnostics)


@dataclass(frozen=True)
class ProposalDecision:
    """Record whether the planner selected one backend assignment and why."""

    proposal: RegionProposal
    selected: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("ProposalDecision reason must be non-empty")


@dataclass(frozen=True)
class ProposalSelection:
    """Return selected proposals and decisions for every considered proposal."""

    proposals: tuple[RegionProposal, ...]
    decisions: tuple[ProposalDecision, ...]


@dataclass(frozen=True)
class LoweredRegion:
    """Record an inspectable provider-owned lowering of one source proposal."""

    proposal: RegionProposal
    program: Program
    source_input_is_entry: tuple[bool, ...] = ()
    required_bindings: tuple[str, ...] = ()
    decisions: tuple[object, ...] = ()
    generated_assets: tuple[object, ...] = ()
    manifest: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.program, Program):
            raise TypeError("LoweredRegion program must be a Program")
        object.__setattr__(
            self, "manifest", MappingProxyType(dict(self.manifest))
        )


@dataclass(frozen=True)
class CompiledRegion:
    """Pair a verified lowered region with its executable implementation."""

    lowered: LoweredRegion
    executable: Executable

    def __post_init__(self) -> None:
        if not isinstance(self.lowered, LoweredRegion):
            raise TypeError(
                "CompiledRegion lowered value must be LoweredRegion"
            )
        if not isinstance(self.executable, Executable):
            raise TypeError(
                "CompiledRegion executable must implement Executable"
            )


def proposal_operations(
    program: Program, proposal: RegionProposal
) -> tuple[Operation, ...]:
    """Return the source operations owned by one indexed proposal."""

    operations = tuple(
        operation
        for operation in program.single_block("main").ops
        if not isinstance(operation, ReturnOp)
    )
    if proposal.stop > len(operations):
        raise ValueError("Region proposal exceeds the source operation range")
    return tuple(operations[index] for index in proposal.operation_indices)


def proposal_inputs_outputs(
    program: Program,
    proposal: RegionProposal,
) -> tuple[tuple[SSAValue, ...], tuple[SSAValue, ...]]:
    """Derive source SSA inputs and escaping results for one proposal."""

    operations = proposal_operations(program, proposal)
    selected = set(operations)
    produced = {
        result for operation in operations for result in operation.results
    }
    inputs: list[SSAValue] = []
    outputs: list[SSAValue] = []
    for operation in operations:
        for operand in operation.operands:
            if operand not in produced and operand not in inputs:
                inputs.append(operand)
        for result in operation.results:
            if any(use.operation not in selected for use in result.uses):
                outputs.append(result)
    return tuple(inputs), tuple(outputs)


def extract_proposal_program(
    program: Program, proposal: RegionProposal
) -> Program:
    """Clone a proposal into a single-block function with derived interfaces."""

    operations = proposal_operations(program, proposal)
    inputs, outputs = proposal_inputs_outputs(program, proposal)
    block = Block(arg_types=[value.type for value in inputs])
    mapper: dict[SSAValue, SSAValue] = dict(
        zip(inputs, block.args, strict=True)
    )
    for operation in operations:
        cloned = operation.clone(value_mapper=mapper)
        block.add_op(cloned)
        mapper.update(zip(operation.results, cloned.results, strict=True))
    block.add_op(ReturnOp(*(mapper[value] for value in outputs)))
    return Program.from_function(block, [value.type for value in outputs])


__all__ = [
    "CompiledRegion",
    "LoweredRegion",
    "ProposalDecision",
    "ProposalSelection",
    "RegionProposal",
    "ValueRequirement",
    "extract_proposal_program",
    "proposal_inputs_outputs",
    "proposal_operations",
]
