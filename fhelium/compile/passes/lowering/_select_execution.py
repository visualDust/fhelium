"""Select registered lowerings independently at each execution operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from typing import cast
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Block, Operation
from xdsl.dialects.func import FuncOp, ReturnOp
from fhelium.compile._compilation import Compilation
from fhelium.compile._workspace import CompileWorkspace

from fhelium.backend.implementation import (
    FusionImplementation,
    OperationImplementationRegistry,
    requested_implementation,
)
from fhelium.config import CkksConfig
from fhelium.config._derived import config_from_text
from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE, Program

from ..._local_plans import local_plan_scope
from ..._pipeline import DecisionRecord, PassResult, PassStats
from ..._selection_plan import (
    GuardMemo,
    SelectionPlans,
    graph_guard,
    plan_event,
    selection_policy,
)
from .._operation_transforms import program_operations
from ..backend._operations import (
    _STRUCTURAL_OPERATION_TYPES,
    executable_operations,
)
from ..fusion import FuseOperationsPass, _fusion_segments
from ._core import CkksLoweringRegistry
from ._driver import DEFAULT_CKKS_LOWERINGS, apply_lowering
from ._shared_bodies import KEY_SWITCH_OPERATIONS, lower_shared_key_switches
from ..._material_requirements import (
    implementation_registry,
    key_reference_scope,
    resolve_material_requirements,
)


@dataclass(frozen=True)
class SelectExecutionLoweringsPass:
    """Keep applicable whole operations or expose a supported fusion expression.

    A supplied implementation assignment is local to its operation. Unassigned
    operations without whole-operation coverage require lowering. Where both
    routes exist, an isolated lowering probe is accepted only when a fusion
    implementation supports the complete expression. With partial fusion enabled,
    supported subregions may instead be combined with independent implementations
    for the remaining operations.

    ``execution_cost`` may compare the whole operation with the isolated lowered
    candidate after this pass's fusion scheduling. It returns ``(whole, lowered)``
    costs in one caller-selected unit, or None when the comparison is unknown.
    Lowering is selected only for a strictly smaller supplied cost; ties retain
    the whole operation. Missing estimates retain the fusion-coverage decision.
    Implementation assignments always take precedence over this callback.

    With the default lowering registry, accepted key switches in function bodies that agree in operation type, attributes other than ``rotation_step``, and operand and result types are lowered once into a private function and replaced by calls carrying their own Galois tables. Set ``workspace['shared_key_switch_bodies']`` to False to lower each key switch in place.
    """

    implementations: OperationImplementationRegistry
    fusion_implementations: Sequence[FusionImplementation] = ()
    lowerings: CkksLoweringRegistry = DEFAULT_CKKS_LOWERINGS
    allow_partial_fusion: bool = True
    name: str = "select-execution-lowerings"
    execution_cost: (
        Callable[[Operation, Sequence[Operation]], tuple[float, float] | None]
        | None
    ) = None

    def _independent_support(self, operation: Operation) -> bool:
        if not self.implementations.supports(operation):
            return False
        implementation = self.implementations.resolve(
            operation, requested=requested_implementation(operation)
        )
        supports = getattr(implementation, "supports_operation", None)
        return cast(bool, supports(operation)) if callable(supports) else True

    def run(self, compilation: "Compilation") -> PassResult:
        previous = compilation.workspace.get(OperationImplementationRegistry)
        compilation.workspace[OperationImplementationRegistry] = (
            self.implementations
        )
        try:
            with key_reference_scope(
                compilation, enabled=self.lowerings is DEFAULT_CKKS_LOWERINGS
            ):
                return self._run(compilation)
        finally:
            if previous is not None:
                compilation.workspace[OperationImplementationRegistry] = (
                    previous
                )

    def _run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        matched = transformed = inserted = skipped = 0
        decisions: list[DecisionRecord] = []
        diagnostics: list[str] = []
        implementation_registry(compilation, self.implementations)
        fallback_config = shared_data.get(CkksConfig)
        plans = shared_data.get(SelectionPlans)
        policy = None
        policy_reason = ""
        if isinstance(plans, SelectionPlans):
            try:
                policy = selection_policy(
                    self.implementations,
                    self.fusion_implementations,
                    self.lowerings,
                    self.execution_cost,
                    self.allow_partial_fusion,
                )
            except ValueError as error:
                policy_reason = str(error)
        local_plans = local_plan_scope(shared_data, policy)
        guard_memo = GuardMemo()
        share = (
            self.lowerings is DEFAULT_CKKS_LOWERINGS
            and shared_data.get("shared_key_switch_bodies") is not False
        )
        deferred: list[tuple[Operation, CkksConfig]] = []
        for operation in program_operations(program):
            requirements = resolve_material_requirements(
                compilation, (operation,), self.implementations
            )
            matched += requirements.stats.matched
            transformed += requirements.stats.transformed
            inserted += requirements.stats.inserted
            skipped += requirements.stats.skipped
            if not self.lowerings.supports(operation):
                continue
            matched += 1
            candidates = self.lowerings.available(type(operation))
            if EXECUTION_IMPLEMENTATION_ATTRIBUTE in operation.attributes:
                skipped += 1
                decisions.append(
                    DecisionRecord(
                        operation.name,
                        requested_implementation(operation),
                        candidates,
                        ("caller-assigned implementation retained",),
                    )
                )
                continue
            configuration = operation.attributes.get("ckks_config")
            config = (
                config_from_text(configuration.data)
                if isinstance(configuration, StringAttr)
                else fallback_config
                if isinstance(fallback_config, CkksConfig)
                else None
            )
            whole = self.implementations.supports(operation)
            if whole:
                implementation = self.implementations.resolve(
                    operation, requested=None
                )
                supports = getattr(implementation, "supports_operation", None)
                if callable(supports):
                    whole = supports(operation)
            # A probe owns its uses independently of the Program. Discarding it
            # cannot leave speculative SSA consumers attached to live values.
            probe_block = Block(
                arg_types=[value.type for value in operation.operands]
            )
            probe = operation.clone(
                value_mapper=dict(zip(operation.operands, probe_block.args))
            )
            probe_block.add_ops((probe, ReturnOp(*probe.results)))
            probe_program = Program.from_function(
                probe_block, tuple(v.type for v in probe.results)
            )
            probe_workspace = CompileWorkspace(shared_data)
            probe_workspace.pop(SelectionPlans, None)
            probe_workspace.pop("selection_plan_stats", None)
            probe_workspace.pop("local_plan_stats", None)
            probe_compilation = Compilation(probe_program, probe_workspace)
            plan_key = None
            cached = None
            if isinstance(plans, SelectionPlans):
                if policy is not None:
                    try:
                        plan_key = "selection:" + graph_guard(
                            probe_block,
                            stage="selection",
                            policy=policy,
                            config=config,
                            memo=guard_memo,
                        )
                        cached = plans.entries.get(plan_key)
                        plan_event(
                            shared_data,
                            "selection",
                            cached is not None,
                            "guard matched"
                            if cached is not None
                            else "operation guard miss",
                        )
                    except ValueError as error:
                        plan_event(shared_data, "selection", False, str(error))
                else:
                    plan_event(shared_data, "selection", False, policy_reason)
            expression = None
            reusable = True
            accepted = not whole
            reason = "no whole-operation coverage"
            cost_details: tuple[str, ...] = ()
            attached = False
            cost_eligible = False
            try:
                if cached is not None:
                    accepted = cached["accepted"]
                    reason = cached["reason"]
                else:
                    expression = apply_lowering(
                        probe,
                        config,
                        probe_compilation,
                        registry=self.lowerings,
                    )
                    attached = True
                    if whole:
                        accepted = any(
                            (
                                count := candidate.match_fusion(
                                    expression.operations
                                )
                            )
                            is not None
                            and count > 0
                            for candidate in self.fusion_implementations
                        )
                        reason = (
                            "registered fusion supports the lowered expression"
                            if accepted
                            else "whole operation retained"
                        )
                        if not accepted and self.allow_partial_fusion:
                            regions = [
                                region
                                for region in _fusion_segments(
                                    expression.operations,
                                    self.fusion_implementations,
                                    local_plans=local_plans,
                                )
                                if region[2] >= 2
                                and (count := region[1].match_fusion(region[0]))
                                is not None
                                and count >= 2
                            ]
                            covered = {
                                op for ops, _, _ in regions for op in ops
                            }
                            remaining = [
                                nested
                                for op in expression.operations
                                if op not in covered
                                for nested in op.walk()
                                if not isinstance(
                                    nested, _STRUCTURAL_OPERATION_TYPES
                                )
                            ]
                            accepted = bool(regions) and all(
                                self._independent_support(op)
                                for op in remaining
                            )
                            if accepted:
                                reason = (
                                    f"lowering covered by {len(regions)} fusion regions "
                                    f"and {len(remaining)} independent operations"
                                )
                        cost_eligible = self.execution_cost is not None and (
                            accepted
                            or all(
                                self._independent_support(nested)
                                for item in expression.operations
                                for nested in item.walk()
                                if not isinstance(
                                    nested, _STRUCTURAL_OPERATION_TYPES
                                )
                            )
                        )
            except ValueError as error:
                # A partial Program need not yet have the mathematical facts
                # required by a lowering. Linking still checks execution readiness.
                reusable = False
                accepted = False
                reason = str(error)
                diagnostics.append(f"{operation.name}: {reason}")
            else:
                if cost_eligible and self.execution_cost is not None:
                    FuseOperationsPass(self.fusion_implementations).run(
                        probe_compilation
                    )
                    costs = self.execution_cost(
                        operation, executable_operations(probe_program)
                    )
                    if costs is None:
                        cost_details = (
                            "execution costs unknown; fusion-coverage decision retained",
                        )
                    else:
                        whole_cost, lowered_cost = costs
                        accepted = lowered_cost < whole_cost
                        reason = (
                            "lowered execution has smaller caller-provided cost"
                            if accepted
                            else "whole execution has no larger caller-provided cost"
                        )
                        cost_details = (
                            f"whole_cost={whole_cost}; lowered_cost={lowered_cost}; unit=caller-selected shared unit",
                        )
            finally:
                if expression is not None and not attached:
                    for item in reversed(expression.operations):
                        item.drop_all_references()
                probe_program.module.drop_all_references()
            if plan_key is not None and cached is None and reusable:
                plans.entries[plan_key] = {
                    "accepted": accepted,
                    "reason": reason,
                }
            if not accepted:
                skipped += 1
                decisions.append(
                    DecisionRecord(
                        operation.name,
                        "preserve",
                        candidates,
                        (reason, *cost_details),
                    )
                )
                continue
            transformed += 1
            if (
                share
                and config is not None
                and isinstance(operation, KEY_SWITCH_OPERATIONS)
                and isinstance(operation.parent_op(), FuncOp)
            ):
                deferred.append((operation, config))
            else:
                lowered = apply_lowering(
                    operation, config, compilation, registry=self.lowerings
                )
                inserted += len(lowered.operations)
            decisions.append(
                DecisionRecord(
                    operation.name,
                    self.lowerings.resolve(type(operation)).name,
                    candidates,
                    (reason, *cost_details),
                )
            )
        if deferred:
            inserted += lower_shared_key_switches(
                compilation, deferred, self.lowerings
            )
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=transformed,
                inserted=inserted,
                skipped=skipped,
            ),
            tuple(diagnostics),
            tuple(decisions),
        )


__all__ = ["SelectExecutionLoweringsPass"]
