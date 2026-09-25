"""Select registered lowerings independently at each execution operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


import json
from typing import cast
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.rewriter import Rewriter
from xdsl.ir import Block, Operation
from xdsl.dialects.func import ReturnOp
from fhelium.compile._compilation import Compilation
from fhelium.compile._workspace import CompileWorkspace

from fhelium.backend.implementation import (
    FusionImplementation,
    OperationImplementationRegistry,
    requested_implementation,
)
from fhelium.config import CkksConfig
from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE, Program

from ..._pipeline import DecisionRecord, PassResult, PassStats
from .._operation_transforms import program_operations
from ..backend._operations import _STRUCTURAL_OPERATION_TYPES
from ..fusion import _fusion_segments
from ._core import CkksLoweringRegistry
from ._driver import DEFAULT_CKKS_LOWERINGS


@dataclass(frozen=True)
class SelectExecutionLoweringsPass:
    """Keep applicable whole operations or expose a supported fusion expression.

    A supplied implementation assignment is local to its operation. Unassigned
    operations without whole-operation coverage require lowering. Where both
    routes exist, a detached lowering probe is accepted only when a fusion
    implementation supports the complete expression. With partial fusion enabled,
    supported subregions may instead be combined with independent implementations
    for the remaining operations.
    """

    implementations: OperationImplementationRegistry
    fusion_implementations: Sequence[FusionImplementation] = ()
    lowerings: CkksLoweringRegistry = DEFAULT_CKKS_LOWERINGS
    allow_partial_fusion: bool = True
    name: str = "select-execution-lowerings"

    def _independent_support(self, operation: Operation) -> bool:
        if not self.implementations.supports(operation):
            return False
        implementation = self.implementations.resolve(
            operation, requested=requested_implementation(operation)
        )
        supports = getattr(implementation, "supports_operation", None)
        return cast(bool, supports(operation)) if callable(supports) else True

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        matched = transformed = inserted = skipped = 0
        decisions: list[DecisionRecord] = []
        diagnostics: list[str] = []
        fallback_config = shared_data.get(CkksConfig)
        for operation in program_operations(program):
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
                CkksConfig.parse(json.loads(configuration.data))
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
            probe_compilation = Compilation(
                probe_program, CompileWorkspace(shared_data)
            )
            expression = None
            accepted = not whole
            reason = "no whole-operation coverage"
            try:
                expression = self.lowerings.lower(
                    probe, config, probe_compilation
                )
                if whole:
                    accepted = any(
                        (count := candidate.match_fusion(expression.operations))
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
                            )
                            if region[2] >= 2
                        ]
                        covered = {op for ops, _, _ in regions for op in ops}
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
                            self._independent_support(op) for op in remaining
                        )
                        if accepted:
                            reason = (
                                f"lowering covered by {len(regions)} fusion regions "
                                f"and {len(remaining)} independent operations"
                            )
            except ValueError as error:
                # A partial Program need not yet have the mathematical facts
                # required by a lowering. Linking still checks execution readiness.
                accepted = False
                reason = str(error)
                diagnostics.append(f"{operation.name}: {reason}")
            finally:
                if expression is not None:
                    for item in reversed(expression.operations):
                        item.drop_all_references()
                probe.drop_all_references()
            if not accepted:
                skipped += 1
                decisions.append(
                    DecisionRecord(
                        operation.name,
                        "preserve",
                        candidates,
                        (reason,),
                    )
                )
                continue
            lowered = self.lowerings.lower(operation, config, compilation)
            Rewriter.replace_op(
                operation, lowered.operations, new_results=(lowered.result,)
            )
            for symbol, description in lowered.material_descriptions.items():
                if symbol not in program.material_descriptions:
                    program.set_material_description(symbol, description)
            transformed += 1
            inserted += len(lowered.operations)
            decisions.append(
                DecisionRecord(
                    operation.name,
                    self.lowerings.resolve(type(operation)).name,
                    candidates,
                    (reason,),
                )
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
