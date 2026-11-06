"""Register default CKKS lowerings and transform complete Programs."""

from __future__ import annotations

from ..._pipeline import (
    DecisionRecord,
)

from fhelium.config import CkksConfig

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.rewriter import Rewriter

from ....ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from ....ir._program import Program

from .._operation_transforms import program_operations
from ._arithmetic import ARITHMETIC_LOWERINGS
from ._core import CkksLoweringRegistry, LoweredCkksOperation
from fhelium.backend.implementation import OperationImplementationRegistry
from ..._material_requirements import resolve_material_requirements
from ._keyswitch import KEY_SWITCH_LOWERINGS
from ._rotation_sum import ROTATION_SUM_LOWERINGS
from ._modulus_raise import MODULUS_RAISE_LOWERINGS
from ._representation import REPRESENTATION_LOWERINGS

DEFAULT_CKKS_LOWERINGS = CkksLoweringRegistry(
    (
        *ARITHMETIC_LOWERINGS,
        *REPRESENTATION_LOWERINGS,
        *KEY_SWITCH_LOWERINGS,
        *ROTATION_SUM_LOWERINGS,
        *MODULUS_RAISE_LOWERINGS,
    )
)


@dataclass(frozen=True)
class _LoweringStats:
    """Count patterns handled by one neutral lowering call."""

    matched: int = 0
    transformed: int = 0
    inserted: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class _LoweringResult:
    """Return a lowered Program and neutral transformation counts."""

    program: Program
    stats: _LoweringStats = _LoweringStats()
    decisions: tuple[DecisionRecord, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether the lowering replaced at least one operation."""

        return self.stats.transformed > 0


def apply_lowering(operation, config, compilation, *, registry, requested=None):
    """Replace an operation and complete its replacement material declarations."""
    block = operation.parent_block()
    if block is None:
        raise ValueError(
            "Lowering material requirements needs an operation attached to a Program"
        )
    before, after = operation.prev_op, operation.next_op
    lowered = registry.lower(
        operation, config, compilation, requested=requested
    )
    Rewriter.replace_op(
        operation, lowered.operations, new_results=(lowered.result,)
    )
    for symbol, description in lowered.material_descriptions.items():
        if symbol not in compilation.program.material_descriptions:
            compilation.program.set_material_description(symbol, description)
    if OperationImplementationRegistry in compilation.workspace:
        resolve_material_requirements(compilation, lowered.operations)
    replacements = []
    current = block.first_op if before is None else before.next_op
    while current is not after:
        if current is None:
            break
        replacements.append(current)
        current = current.next_op
    return LoweredCkksOperation(
        tuple(replacements), lowered.result, lowered.material_descriptions
    )


def lower_ckks_program(
    compilation,
    config: CkksConfig | None,
    *,
    registry: CkksLoweringRegistry = DEFAULT_CKKS_LOWERINGS,
    selections: Mapping[str, str] | None = None,
    preserve: Collection[str] = (),
) -> _LoweringResult:
    """Apply selected CKKS lowerings while preserving other mixed-level IR."""

    program = compilation.program
    requested = dict(selections or {})
    if any(
        not isinstance(operation_name, str)
        or not operation_name
        or not isinstance(lowering_name, str)
        or not lowering_name
        for operation_name, lowering_name in requested.items()
    ):
        raise ValueError("CKKS lowering selections require non-empty names")
    preserved = frozenset(preserve)
    if any(not isinstance(name, str) or not name for name in preserved):
        raise ValueError("Preserved CKKS operation names must be non-empty")
    overlap = preserved.intersection(requested)
    if overlap:
        raise ValueError(
            "CKKS operations cannot be both preserved and assigned a "
            f"lowering: {tuple(sorted(overlap))}"
        )

    matched = transformed = inserted = skipped = 0
    decisions: list[DecisionRecord] = []
    for operation in program_operations(program):
        if not registry.supports(operation):
            continue
        matched += 1
        candidates = registry.available(type(operation))
        if operation.name in preserved:
            material_changes = False
            if OperationImplementationRegistry in compilation.workspace:
                requirements = resolve_material_requirements(
                    compilation, (operation,)
                )
                material_changes = requirements.changed
                inserted += requirements.stats.inserted
            transformed += int(material_changes)
            skipped += int(not material_changes)
            decisions.append(
                DecisionRecord(
                    operation.name,
                    selected="preserve",
                    candidates=candidates,
                    details=("operation preserved at CKKS depth",),
                )
            )
            continue
        assigned = operation.attributes.get(EXECUTION_IMPLEMENTATION_ATTRIBUTE)
        if isinstance(assigned, StringAttr):
            raise ValueError(
                f"CKKS operation {operation.name!r} records Backend "
                f"implementation {assigned.data!r}; preserve the "
                "operation or remove the assignment before lowering"
            )
        definition = registry.resolve(
            type(operation),
            requested.get(operation.name),
        )
        represented = operation.attributes.get("ckks_config")
        selected_config = (
            CkksConfig.parse(json.loads(represented.data))
            if isinstance(represented, StringAttr)
            else config
        )
        lowered = apply_lowering(
            operation,
            selected_config,
            compilation,
            registry=registry,
            requested=definition.name,
        )
        transformed += 1
        inserted += len(lowered.operations)
        decisions.append(
            DecisionRecord(
                operation.name,
                selected=definition.name,
                candidates=candidates,
                details=("operation replaced by lower-depth IR",),
            )
        )
    return _LoweringResult(
        program,
        _LoweringStats(
            matched=matched,
            transformed=transformed,
            inserted=inserted,
            skipped=skipped,
        ),
        tuple(decisions),
    )
