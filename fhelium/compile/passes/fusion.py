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
from fhelium.ir.dialects import ckks, core, fusion
from fhelium.backend.implementation import FusionImplementation, FusionMatcher

from .._local_plans import LocalPlanScope, local_plan_scope
from .._pipeline import DecisionRecord, PassResult, PassStats
from .._selection_plan import (
    GuardMemo,
    SelectionPlans,
    fusion_policy,
    graph_guard,
    plan_event,
)
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
    from fhelium.backend.triton._repeat import graph_nodes

    for node in graph_nodes(body_ops, preserve_carried=True):
        for operand in node.operands:
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
        from fhelium.backend.triton._matching import RegionMatcher, NTT_OPS
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
        if type(matcher) is RegionMatcher and matcher.analysis is not None:
            self.kind = "rns"
        elif type(matcher) is TensorRegionMatcher:
            self.kind = "tensor"
        else:
            self.kind = None
            return
        for index, operation in enumerate(operations):
            if self.kind == "rns":
                if isinstance(operation, ckks.RepeatOp):
                    row = {}
                else:
                    facts = matcher.analysis.operation(operation)
                    if not facts.valid or (
                        facts.anchors and NTT_OPS[0] not in matcher.supported
                    ):
                        continue
                    row = dict(facts.known)
                    if facts.parameter is not None:
                        row["_parameter"] = facts.parameter
            else:
                if operation not in matcher.facts:
                    matcher.facts[operation] = _operation_facts(operation)
                facts = matcher.facts[operation]
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
    *,
    local_plans: LocalPlanScope | None = None,
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

        has_mac = any(
            isinstance(operation, ckks.RepeatOp)
            and any(
                isinstance(nested, ckks.RepeatOp) and nested.carried
                for nested in operation.walk()
            )
            for operation in items
        )
        for matcher in matchers:
            if type(matcher) is RegionMatcher:
                matcher.check_mac = has_mac

        if local_plans is not None:
            analysis = OperationAnalysis(
                local_plans.matcher_recipe, local_plans.stats
            )
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
    *,
    local_plans: LocalPlanScope | None = None,
) -> list[tuple[list[Operation], FusionImplementation, int]]:
    return _fusion_schedule(
        operations, implementations, local_plans=local_plans
    )[1]


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
        from .._repeat import expand_repeats
        from .._repeat_rotation import is_rotation_repeat

        # Unsupported endpoint policies use the ordinary expanded scheduler.
        for operation in tuple(program.walk()):
            if (
                isinstance(operation, ckks.RepeatOp)
                and not is_rotation_repeat(operation)
                and not any(
                    implementation.match_fusion((operation,)) is not None
                    for implementation in self.implementations
                )
            ):
                expand_repeats(compilation, roots=(operation,))
        shared_data = compilation.workspace
        plans = shared_data.get(SelectionPlans)
        policy = None
        policy_reason = ""
        if isinstance(plans, SelectionPlans):
            try:
                policy = [fusion_policy(self.implementations), self.min_ops]
            except ValueError as error:
                policy_reason = str(error)
        local_plans = local_plan_scope(shared_data, policy)
        blocks = []

        def visit(block):
            blocks.append(block)
            for operation in tuple(block.ops):
                if isinstance(operation, fusion.FusedOp) or (
                    isinstance(operation, ckks.RepeatOp)
                    and not is_rotation_repeat(operation)
                ):
                    continue
                for nested in known_region_blocks(operation):
                    visit(nested)

        for function in program.functions:
            for block in function.body.blocks:
                visit(block)
        matched = transformed = removed = 0
        decisions: list[DecisionRecord] = []
        guard_memo = GuardMemo()
        for block in blocks:
            original = tuple(block.ops)
            plan_key = None
            cached = None
            if isinstance(plans, SelectionPlans):
                if policy is not None:
                    try:
                        plan_key = "fusion:" + graph_guard(
                            block,
                            stage="fusion",
                            policy=policy,
                            memo=guard_memo,
                        )
                        cached = plans.entries.get(plan_key)
                        plan_event(
                            shared_data,
                            "fusion",
                            cached is not None,
                            "guard matched"
                            if cached is not None
                            else "graph guard miss",
                        )
                    except ValueError as error:
                        plan_event(shared_data, "fusion", False, str(error))
                else:
                    plan_event(shared_data, "fusion", False, policy_reason)
            positions = {op: index for index, op in enumerate(original)}
            recorded_groups = []
            if cached is None:
                scheduled, segments = _fusion_schedule(
                    original, self.implementations, local_plans=local_plans
                )
            else:
                scheduled = [original[index] for index in cached["schedule"]]
                segments = [
                    (
                        [original[index] for index in indices],
                        self.implementations[implementation],
                        count,
                    )
                    for indices, implementation, count in cached["groups"]
                ]
            if scheduled != list(block.ops):
                for operation in tuple(block.ops):
                    block.detach_op(operation)
                block.add_ops(scheduled)

            for operations, implementation, count in segments:
                if cached is None:
                    if count < self.min_ops:
                        continue
                    eligible = implementation.match_fusion(operations)
                    if eligible is None or eligible < self.min_ops:
                        continue
                    if plan_key is not None:
                        recorded_groups.append(
                            [
                                [positions[op] for op in operations],
                                next(
                                    index
                                    for index, item in enumerate(
                                        self.implementations
                                    )
                                    if item is implementation
                                ),
                                count,
                            ]
                        )
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
            if plan_key is not None and cached is None:
                plans.entries[plan_key] = {
                    "schedule": [positions[op] for op in scheduled],
                    "groups": recorded_groups,
                }
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
