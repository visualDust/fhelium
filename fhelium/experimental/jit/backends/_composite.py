"""Backend assignment, adjacent-region formation, and mixed execution."""

from __future__ import annotations


from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation, SSAValue

from fhelium.ir import (
    DEFAULT_OPERATION_SPECS,
    OperationSpecRegistry,
    Program,
    operation_name,
    value_role,
)

from .._contracts import (
    CoverageDiagnostic,
    CoverageReport,
    Executable,
    ProviderDecision,
    ProviderPolicy,
)
from .._execution import ExecutionInputs

from .._bindings import RuntimeBindings
from ..planning._plan import (
    ExecutionPlan,
    OperationSupport,
    PlanRegion,
    PlanValue,
)
from ..planning._planner import form_adjacent_regions, select_region_proposals
from ..planning._records import (
    LoweredRegion,
    ProposalDecision,
    proposal_inputs_outputs,
)
from ..providers._defaults import default_backend_providers
from ..providers._registry import BackendProvider
from ._interpreter_runtime import (
    _bind_program_inputs_raw,
    _reconstruct_program_output,
)


@dataclass(frozen=True)
class _RegionBuild:
    plan: PlanRegion
    executable: Executable
    inputs: tuple[SSAValue, ...]
    outputs: tuple[SSAValue, ...]


@dataclass
class CompositeExecutable:
    """Connect compiled provider regions through Program SSA values."""

    _program: Program
    _plan: ExecutionPlan
    _regions: tuple[_RegionBuild, ...]
    _bindings: RuntimeBindings
    _execution: ExecutionInputs
    _decisions: tuple[ProposalDecision, ...]

    @property
    def backend(self) -> str:
        return "composite"

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "backend": self.backend,
                "execution_inputs": self._execution.as_dict(),
                "plan": dict(self._plan.manifest()),
                "assignment_decisions": [
                    {
                        "provider": decision.proposal.provider,
                        "implementation": decision.proposal.implementation,
                        "operations": list(decision.proposal.operation_ids),
                        "selected": decision.selected,
                        "reason": decision.reason,
                    }
                    for decision in self._decisions
                ],
                "region_executables": [
                    dict(region.executable.manifest) for region in self._regions
                ],
                "connection_abi": "python-object-ssa-v1",
                "cuda_stream": "current-pytorch-stream",
                "fallback": False,
            }
        )

    def run(self, *args: object, **kwargs: object) -> Any:
        block = self._program.single_block("main")
        raw_inputs = _bind_program_inputs_raw(self._program, args, kwargs)
        slots: dict[SSAValue, object] = dict(
            zip(block.args, raw_inputs, strict=True)
        )
        for region in self._regions:
            region_args = tuple(slots[value] for value in region.inputs)
            result = region.executable.run(*region_args)
            if len(region.outputs) == 0:
                values: tuple[object, ...] = ()
            elif len(region.outputs) == 1:
                values = (result,)
            elif isinstance(result, tuple) and len(result) == len(
                region.outputs
            ):
                values = result
            else:
                raise RuntimeError(
                    f"Region {region.plan.region_id!r} returned an invalid "
                    "output tuple"
                )
            slots.update(zip(region.outputs, values, strict=True))

        return_op = tuple(block.ops)[-1]
        assert isinstance(return_op, ReturnOp)
        returned = tuple(slots[value] for value in return_op.arguments)
        return _reconstruct_program_output(self._program, returned)


@dataclass(frozen=True)
class _AnalyzedPlan:
    program: Program
    plan: ExecutionPlan
    diagnostics: tuple[CoverageDiagnostic, ...]
    providers: tuple[BackendProvider, ...]
    lowered_regions: tuple[LoweredRegion, ...]
    region_inputs: tuple[tuple[SSAValue, ...], ...]
    region_outputs: tuple[tuple[SSAValue, ...], ...]
    decisions: tuple[ProposalDecision, ...]


class CompositeBackend:
    """Assign operations by caller policy and build adjacent backend regions."""

    name = "composite"

    def __init__(
        self,
        providers: Sequence[BackendProvider] | None = None,
        *,
        specifications: OperationSpecRegistry = DEFAULT_OPERATION_SPECS,
    ) -> None:
        selected: tuple[BackendProvider, ...] = (
            default_backend_providers()
            if providers is None
            else tuple(providers)
        )
        entries: dict[str, BackendProvider] = {}
        for provider in selected:
            if not isinstance(provider, BackendProvider):
                raise TypeError(
                    "Composite providers must be BackendProvider objects"
                )
            if provider.name in entries:
                raise ValueError(
                    f"Region provider {provider.name!r} is duplicated"
                )
            entries[provider.name] = provider
        self._providers = MappingProxyType(entries)
        self._specifications = specifications

    def coverage(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> CoverageReport:
        analyzed = self._analyze(program, execution, bindings, policy)
        return CoverageReport(
            backend=self.name,
            diagnostics=analyzed.diagnostics,
            operations=frozenset(
                item.operation for item in analyzed.plan.operations
            ),
            operation_support=analyzed.plan.operations,
            regions=analyzed.plan.regions,
            provider_decisions=tuple(
                ProviderDecision(
                    decision.proposal.provider,
                    decision.proposal.implementation,
                    decision.proposal.operation_ids,
                    decision.selected,
                    decision.reason,
                    decision.proposal.metadata,
                )
                for decision in analyzed.decisions
            ),
        )

    def build(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> Executable:
        executable_program = program.clone()
        analyzed = self._analyze(
            executable_program, execution, bindings, policy
        )
        if any(item.severity == "error" for item in analyzed.diagnostics):
            detail = "; ".join(item.message for item in analyzed.diagnostics)
            raise RuntimeError(
                f"Composite backend does not cover Program: {detail}"
            )
        builds: list[_RegionBuild] = []
        for index, region in enumerate(analyzed.plan.regions):
            provider = analyzed.providers[index]
            compiled = provider.build(
                analyzed.lowered_regions[index],
                execution=execution,
                bindings=bindings,
            )
            builds.append(
                _RegionBuild(
                    region,
                    compiled.executable,
                    analyzed.region_inputs[index],
                    analyzed.region_outputs[index],
                )
            )
        return CompositeExecutable(
            executable_program,
            analyzed.plan,
            tuple(builds),
            bindings,
            execution,
            analyzed.decisions,
        )

    def _analyze(
        self,
        program: Program,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None,
    ) -> _AnalyzedPlan:
        diagnostics: list[CoverageDiagnostic] = []
        try:
            program.verify_structure()
            block = program.single_block("main")
        except (KeyError, ValueError) as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-entry-structure",
                    f"Composite execution requires one @main block: {error}",
                    "main",
                )
            )
            return self._empty(program, diagnostics)
        body = tuple(block.ops)
        if (
            not body
            or not isinstance(body[-1], ReturnOp)
            or any(isinstance(operation, ReturnOp) for operation in body[:-1])
        ):
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-entry-structure",
                    "Composite @main requires one final func.return.",
                    "main",
                )
            )
            return self._empty(program, diagnostics)
        operations = body[:-1]
        return_op = body[-1]
        assert isinstance(return_op, ReturnOp)

        if policy is None:
            diagnostics.append(
                CoverageDiagnostic(
                    "missing-provider-policy",
                    "Composite execution requires an ordered provider policy.",
                    "policy",
                )
            )
            return self._empty(program, diagnostics)
        diagnostics.extend(policy.validate(execution))
        for provider_name in policy.providers:
            if provider_name not in self._providers:
                diagnostics.append(
                    CoverageDiagnostic(
                        "provider-unavailable",
                        f"Provider {provider_name!r} is not registered.",
                        provider_name,
                    )
                )

        operation_ids = {
            operation: f"main:{index}"
            for index, operation in enumerate(operations)
        }
        value_ids: dict[SSAValue, str] = {
            argument: f"arg:{index}"
            for index, argument in enumerate(block.args)
        }
        for operation in operations:
            for result_index, result in enumerate(operation.results):
                value_ids[result] = (
                    f"{operation_ids[operation]}:result:{result_index}"
                )

        proposals_list = []
        for provider in self._providers.values():
            provider_proposals = provider.propose(
                program,
                execution=execution,
                bindings=bindings,
                specifications=self._specifications,
            )
            assignment_region_diagnostics: dict[
                int, tuple[CoverageDiagnostic, ...]
            ] = {}
            for region in form_adjacent_regions(provider_proposals):
                if not region.supported:
                    continue
                diagnostics_for_region = provider.region_diagnostics(
                    program,
                    region,
                    execution=execution,
                    bindings=bindings,
                )
                for operation_index in region.operation_indices:
                    assignment_region_diagnostics[operation_index] = (
                        diagnostics_for_region
                    )
            proposals_list.extend(
                replace(
                    proposal,
                    diagnostics=(
                        *proposal.diagnostics,
                        *assignment_region_diagnostics.get(proposal.start, ()),
                    ),
                )
                for proposal in provider_proposals
            )
        proposals = tuple(proposals_list)
        if policy.preserve_ckks_operations:
            proposals = tuple(
                replace(proposal, priority=max(proposal.priority, 10_000))
                if proposal.implementation == "preserved-ckks-operation"
                else proposal
                for proposal in proposals
            )
        selection = select_region_proposals(
            proposals, provider_order=policy.providers
        )
        selected_by_index = {
            index: proposal
            for proposal in selection.proposals
            for index in proposal.operation_indices
        }

        support: list[OperationSupport] = []
        for index, operation in enumerate(operations):
            operation_id = operation_ids[operation]
            specification = self._specifications.get(operation_name(operation))
            item_diagnostics: list[CoverageDiagnostic] = []
            if specification is None:
                item_diagnostics.append(
                    CoverageDiagnostic(
                        "unknown-operation-spec",
                        f"Operation {operation_name(operation)!r} has no "
                        "semantic specification.",
                        operation_id,
                    )
                )
            else:
                item_diagnostics.extend(
                    CoverageDiagnostic(
                        "operation-spec-mismatch",
                        message,
                        operation_id,
                    )
                    for message in specification.diagnostics(operation)
                )
            proposal = selected_by_index.get(index)
            if proposal is None:
                rejected_diagnostics = tuple(
                    diagnostic
                    for candidate in proposals
                    if candidate.provider in policy.providers
                    and index in candidate.operation_indices
                    and not candidate.supported
                    for diagnostic in candidate.diagnostics
                )
                item_diagnostics.append(
                    CoverageDiagnostic(
                        "operation-uncovered",
                        "No selected backend assignment owns this operation.",
                        operation_id,
                    )
                )
                item_diagnostics.extend(rejected_diagnostics)
            elif not proposal.supported:
                item_diagnostics.extend(proposal.diagnostics)
            item_supported = not any(
                item.severity == "error" for item in item_diagnostics
            )
            support.append(
                OperationSupport(
                    operation_id,
                    operation_name(operation),
                    specification.family if specification is not None else None,
                    proposal.provider if proposal is not None else None,
                    proposal.implementation if proposal is not None else None,
                    item_supported,
                    tuple(item_diagnostics),
                )
            )
            diagnostics.extend(item_diagnostics)

        plan_regions: list[PlanRegion] = []
        region_providers: list[BackendProvider] = []
        lowered_regions: list[LoweredRegion] = []
        region_inputs: list[tuple[SSAValue, ...]] = []
        region_outputs: list[tuple[SSAValue, ...]] = []
        for index, proposal in enumerate(selection.proposals):
            provider = self._providers[proposal.provider]
            inputs, outputs = proposal_inputs_outputs(program, proposal)
            try:
                lowered = provider.lower(
                    program,
                    proposal,
                    execution=execution,
                    bindings=bindings,
                )
                region_diagnostics = provider.verify(
                    lowered, execution=execution, bindings=bindings
                )
            except Exception as error:
                lowered = LoweredRegion(proposal, program.clone())
                region_diagnostics = (
                    CoverageDiagnostic(
                        "provider-lowering-failed",
                        str(error),
                        proposal.provider,
                    ),
                )
            diagnostics.extend(region_diagnostics)
            plan_regions.append(
                PlanRegion(
                    f"region:{index}",
                    proposal.provider,
                    proposal.implementation,
                    proposal.operation_ids,
                    tuple(value_ids[value] for value in inputs),
                    tuple(value_ids[value] for value in outputs),
                    not any(
                        item.severity == "error" for item in region_diagnostics
                    ),
                    tuple((*proposal.diagnostics, *region_diagnostics)),
                )
            )
            region_providers.append(provider)
            lowered_regions.append(lowered)
            region_inputs.append(inputs)
            region_outputs.append(outputs)

        support_by_id = {
            item.operation_id: index for index, item in enumerate(support)
        }
        for region in plan_regions:
            region_errors = tuple(
                diagnostic
                for diagnostic in region.diagnostics
                if diagnostic.severity == "error"
            )
            if not region_errors:
                continue
            for operation_id in region.operation_ids:
                support_index = support_by_id[operation_id]
                item = support[support_index]
                support[support_index] = replace(
                    item,
                    supported=False,
                    diagnostics=tuple((*item.diagnostics, *region_errors)),
                )

        plan_values = _plan_values(
            block.args, operations, return_op, operation_ids, value_ids
        )
        plan = ExecutionPlan(
            "main",
            tuple(value_ids[value] for value in block.args),
            tuple(value_ids[value] for value in return_op.arguments),
            plan_values,
            tuple(support),
            tuple(plan_regions),
        )
        return _AnalyzedPlan(
            program,
            plan,
            tuple(diagnostics),
            tuple(region_providers),
            tuple(lowered_regions),
            tuple(region_inputs),
            tuple(region_outputs),
            selection.decisions,
        )

    @staticmethod
    def _empty(
        program: Program, diagnostics: list[CoverageDiagnostic]
    ) -> _AnalyzedPlan:
        return _AnalyzedPlan(
            program,
            ExecutionPlan("main", (), (), (), (), ()),
            tuple(diagnostics),
            (),
            (),
            (),
            (),
            (),
        )


def _plan_values(
    arguments: Sequence[SSAValue],
    operations: tuple[Operation, ...],
    return_op: ReturnOp,
    operation_ids: Mapping[Operation, str],
    value_ids: Mapping[SSAValue, str],
) -> tuple[PlanValue, ...]:
    values = (
        *arguments,
        *(result for op in operations for result in op.results),
    )
    result: list[PlanValue] = []
    for value in values:
        producer = getattr(value, "owner", None)
        producer_id = (
            operation_ids.get(producer)
            if isinstance(producer, Operation)
            else None
        )
        consumers = tuple(
            "return"
            if use.operation is return_op
            else operation_ids[use.operation]
            for use in value.uses
            if use.operation is return_op or use.operation in operation_ids
        )
        result.append(
            PlanValue(
                value_ids[value],
                value_role(value),
                f"python-object/{value_role(value) or 'opaque'}",
                producer_id,
                consumers,
            )
        )
    return tuple(result)


__all__ = [
    "CompositeBackend",
    "CompositeExecutable",
    "ExecutionPlan",
    "OperationSupport",
    "PlanRegion",
    "PlanValue",
]
