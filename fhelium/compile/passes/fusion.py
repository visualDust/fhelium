"""Group compatible pure elementwise operations into visible fusion regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass
from collections.abc import Sequence

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation, SSAValue, Block

from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import core, fusion
from fhelium.backend.implementation import FusionImplementation, FusionMatcher

from .._pipeline import DecisionRecord, PassResult, PassStats
from .program._purity import known_pure, known_region_blocks


def _fuse(
    block: Block, operations: list[Operation], implementation: str
) -> fusion.FusedOp | None:
    selected = set(operations)
    body_ops = [
        op
        for op in operations
        if not isinstance(op, (core.ResourceRefOp, core.MaterialRefOp))
    ]
    results = [
        value
        for op in body_ops
        for value in op.results
        if any(use.operation not in selected for use in value.uses)
    ]
    if not results:
        return None

    inputs: list[SSAValue] = []
    input_set: set[SSAValue] = set()
    canonical: dict[SSAValue, SSAValue] = {}
    resource_roles: dict[tuple[object, object], SSAValue] = {}
    for operation in body_ops:
        for operand in operation.operands:
            if not isinstance(operand, SSAValue):
                continue
            owner = operand.owner
            if owner in selected and not isinstance(
                owner, (core.ResourceRefOp, core.MaterialRefOp)
            ):
                continue
            value = operand
            if isinstance(owner, (core.ResourceRefOp, core.MaterialRefOp)):
                role = (owner.symbol, owner.kind)
                value = resource_roles.setdefault(role, operand)
            canonical[operand] = value
            if value not in input_set:
                input_set.add(value)
                inputs.append(value)

    # Resource references have no execution effect. Move the retained references
    # before the region so every outer operand dominates its use.
    first = operations[0]
    resource_ops = [
        op
        for op in operations
        if isinstance(op, (core.ResourceRefOp, core.MaterialRefOp))
    ]
    anchor = first
    for op in resource_ops:
        if op is anchor:
            continue
        block.detach_op(op)
        block.insert_op_before(op, anchor)
    body = Block(arg_types=[value.type for value in inputs])
    mapper: dict[SSAValue, SSAValue] = dict(zip(inputs, body.args, strict=True))
    for alias, value in canonical.items():
        mapper[alias] = mapper[value]
    for op in body_ops:
        body.add_op(op.clone(value_mapper=mapper))
    body.add_op(fusion.YieldOp([mapper[value] for value in results]))
    fused = fusion.FusedOp(
        inputs,
        [value.type for value in results],
        body,
        attributes={
            EXECUTION_IMPLEMENTATION_ATTRIBUTE: StringAttr(implementation)
        },
    )
    # The original first operation may itself be a resource; all resources must
    # precede the fused operation, so insert at the first computational operation.
    block.insert_op_before(fused, body_ops[0])
    for old, new in zip(results, fused.outputs, strict=True):
        old.replace_all_uses_with(new)
    for op in reversed(body_ops):
        block.erase_op(op)
    return fused


class _CandidateSet:
    """Index necessary static facts while leaving escaping-use checks to probes."""

    def __init__(self, matcher, operations):
        from fhelium.backend.triton._matching import RegionMatcher
        from fhelium.backend.triton._tensor import (
            TensorRegionMatcher,
            _operation_facts,
        )

        self.matcher = matcher
        self.all = (1 << len(operations)) - 1
        self.valid = 0
        self.fields = {}
        self.previous = None
        self.allowed = self.all
        analysis = None
        include_ntt = False
        tensor_facts = {}
        if type(matcher) is RegionMatcher and matcher.analysis is not None:
            self.kind = "rns"
            analysis, include_ntt = matcher.analysis, matcher.include_ntt
        elif type(matcher) is TensorRegionMatcher:
            self.kind = "tensor"
            tensor_facts = matcher.facts
        else:
            self.kind = None
            return
        for index, operation in enumerate(operations):
            if analysis is not None:
                facts = analysis.operation(operation)
                if not facts.valid or (facts.anchors and not include_ntt):
                    continue
                row = dict(facts.known)
                if facts.parameter is not None:
                    row["_parameter"] = facts.parameter
            else:
                if operation not in tensor_facts:
                    tensor_facts[operation] = _operation_facts(operation)
                facts = tensor_facts[operation]
                if facts is None:
                    continue
                row = {} if facts[1] is None else {"_tensor": facts[1]}
            bit = 1 << index
            self.valid |= bit
            for field, value in row.items():
                present, groups = self.fields.setdefault(field, (0, {}))
                groups[value] = groups.get(value, 0) | bit
                self.fields[field] = (present | bit, groups)

    def candidates(self):
        if self.kind is None:
            return self.all
        matcher = self.matcher
        if matcher.invalid:
            return 0
        if self.kind == "rns":
            state = tuple(matcher.known.items())
            if matcher.parameter is not None:
                state += (("_parameter", matcher.parameter),)
        else:
            state = (
                () if matcher.common is None else (("_tensor", matcher.common),)
            )
        if state != self.previous:
            allowed = self.valid
            for field, value in state:
                entry = self.fields.get(field)
                if entry is not None:
                    present, groups = entry
                    allowed &= (self.all ^ present) | groups.get(value, 0)
            self.previous, self.allowed = state, allowed
        return self.allowed


def _first_position(mask):
    return (mask & -mask).bit_length() - 1


def _fusion_schedule(
    operations: Sequence[Operation],
    implementations: Sequence[FusionImplementation],
) -> tuple[
    list[Operation], list[tuple[list[Operation], FusionImplementation, int]]
]:
    """Schedule registered pure dependencies while preserving effect and assignment order."""
    scheduled: list[Operation] = []
    segments: list[tuple[list[Operation], FusionImplementation, int]] = []

    def window(items):
        matchers: list[FusionMatcher | None] = [
            cast(FusionMatcher, factory())
            if callable(
                factory := getattr(implementation, "new_fusion_matcher", None)
            )
            else None
            for implementation in implementations
        ]

        from fhelium.backend.triton._matching import (
            OperationAnalysis,
            RegionMatcher,
        )

        analysis = OperationAnalysis()
        for matcher in matchers:
            if type(matcher) is RegionMatcher:
                matcher.use_analysis(analysis)

        def match(operation):
            for implementation, matcher in zip(
                implementations, matchers, strict=True
            ):
                count = (
                    matcher.probe(operation)
                    if matcher is not None
                    else implementation.match_fusion([*pending, operation])
                )
                if count is not None:
                    return implementation, count
            return None

        def reset():
            for matcher in matchers:
                if matcher is not None:
                    matcher.reset()

        def append(operation):
            for matcher in matchers:
                if matcher is not None:
                    matcher.append(operation)

        selected = set(items)
        position = {operation: index for index, operation in enumerate(items)}
        dependencies = {
            operation: {
                operand.owner
                for operand in operation.operands
                if operand.owner in selected
            }
            for operation in items
        }
        consumers = {operation: [] for operation in items}
        for operation, producers in dependencies.items():
            for producer in producers:
                consumers[producer].append(operation)
        filters = [_CandidateSet(matcher, items) for matcher in matchers]
        ready = sum(
            1 << index
            for index, operation in enumerate(items)
            if not dependencies[operation]
        )
        pending: list[Operation] = []
        current = None
        while ready:
            choice = None
            eligible = 0
            for candidates in filters:
                eligible |= candidates.candidates()
            eligible &= ready
            while eligible:
                index = _first_position(eligible)
                eligible &= eligible - 1
                operation = items[index]
                candidate = match(operation)
                if candidate is not None:
                    choice = operation, candidate
                    break
            if choice is None:
                if pending and current is not None:
                    segments.append((pending, *current))
                pending = []
                current = None
                operation = items[_first_position(ready)]
                reset()
                candidate = match(operation)
                if candidate is not None:
                    pending = [operation]
                    current = candidate
                    append(operation)
            else:
                operation, current = choice
                pending.append(operation)
                append(operation)
            ready &= ~(1 << position[operation])
            scheduled.append(operation)
            for consumer in consumers[operation]:
                dependencies[consumer].remove(operation)
                if not dependencies[consumer]:
                    ready |= 1 << position[consumer]
        if pending and current is not None:
            segments.append((pending, *current))

    pending = []
    for operation in operations:
        if (
            known_pure(operation)
            and EXECUTION_IMPLEMENTATION_ATTRIBUTE not in operation.attributes
            and not isinstance(operation, fusion.FusedOp)
        ):
            pending.append(operation)
        else:
            window(pending)
            pending = []
            scheduled.append(operation)
    window(pending)
    return scheduled, segments


def _fusion_segments(
    operations: Sequence[Operation],
    implementations: Sequence[FusionImplementation],
) -> list[tuple[list[Operation], FusionImplementation, int]]:
    return _fusion_schedule(operations, implementations)[1]


@dataclass(frozen=True)
class FuseOperationsPass:
    """Group pure SSA dependencies using caller-supplied Backend support.

    Independent pure work can be scheduled around a supported region. Effects,
    opaque operations and caller implementation assignments preserve their order.
    Implementations determine operation, parameter and layout support.
    """

    implementations: Sequence[FusionImplementation]
    min_ops: int = 2
    name: str = "fuse-operations"

    def __post_init__(self) -> None:
        object.__setattr__(self, "implementations", tuple(self.implementations))
        if self.min_ops < 1:
            raise ValueError("Fusion min_ops must be positive")

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        blocks = []

        def visit(block):
            blocks.append(block)
            for operation in tuple(block.ops):
                if isinstance(operation, fusion.FusedOp):
                    continue
                for nested in known_region_blocks(operation):
                    visit(nested)

        for function in program.functions:
            for block in function.body.blocks:
                visit(block)
        matched = transformed = removed = 0
        decisions: list[DecisionRecord] = []
        for block in blocks:
            original = tuple(block.ops)
            scheduled, segments = _fusion_schedule(
                original, self.implementations
            )
            if scheduled != list(block.ops):
                for operation in tuple(block.ops):
                    block.detach_op(operation)
                block.add_ops(scheduled)

            for operations, implementation, count in segments:
                if count < self.min_ops:
                    continue
                eligible = implementation.match_fusion(operations)
                if eligible is None or eligible < self.min_ops:
                    continue
                matched += 1
                fused = _fuse(block, operations, implementation.name)
                if fused is None:
                    continue
                transformed += 1
                removed += count
                decisions.append(
                    DecisionRecord(
                        "fusion-region",
                        implementation.name,
                        (implementation.name,),
                        (
                            f"operations={count}",
                            f"outputs={len(fused.outputs)}",
                        ),
                    )
                )
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=transformed,
                inserted=transformed,
                removed=removed,
            ),
            decisions=tuple(decisions),
        )


__all__ = ["FuseOperationsPass"]
