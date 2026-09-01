"""Place CKKS relinearization from ciphertext structure and SSA use."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.ir import Attribute, Operation, SSAValue, Use
from xdsl.dialects.builtin import StringAttr, UnrealizedConversionCastOp

from fhelium.ir import Program
from fhelium.ir.dialects import ckks
from .._operation_transforms import (
    ciphertext_type,
    display_name,
    insert_after_and_replace_uses,
)
from ._transition_state import (
    compatible_states,
    exclusive_to,
    is_three_component,
    is_ckks_value_bridge,
    is_same_dialect_ckks_cast,
    reconcile_transition_paths,
    represented_state,
    retype_path,
    retype_result,
    source_through_casts,
)


_TRANSPARENT_TYPES = (
    ckks.RescaleOp,
    ckks.ToNttOp,
    ckks.FromNttOp,
    ckks.ToMontgomeryResiduesOp,
    ckks.ToStandardResiduesOp,
)
_MANUAL_SATISFACTION_PATH_TYPES = (
    *_TRANSPARENT_TYPES,
    ckks.AddOp,
    ckks.SubtractOp,
    ckks.NegateOp,
)


def _classify_relinearization_use(use: Use) -> tuple[bool, tuple[Use, ...]]:
    """Find manual satisfaction and unsatisfied frontiers below one edge."""

    operation = use.operation
    if isinstance(operation, ckks.RelinearizeOp):
        return True, ()
    if is_same_dialect_ckks_cast(operation):
        assert isinstance(operation, UnrealizedConversionCastOp)
        output = operation.outputs[0]
        if not is_three_component(output):
            return False, (use,)
        child_uses = tuple(output.uses)
        if not child_uses:
            return False, (use,)
        classified = tuple(
            _classify_relinearization_use(child) for child in child_uses
        )
        if not any(satisfied for satisfied, _ in classified):
            return False, (use,)
        return True, tuple(
            frontier for _, pending in classified for frontier in pending
        )
    traversable = isinstance(
        operation, _MANUAL_SATISFACTION_PATH_TYPES
    ) or is_ckks_value_bridge(operation)
    if not traversable or len(operation.results) != 1:
        return False, (use,)
    child_uses = tuple(operation.results[0].uses)
    if not child_uses:
        return False, (use,)
    classified = tuple(
        _classify_relinearization_use(child) for child in child_uses
    )
    if not any(satisfied for satisfied, _ in classified):
        return False, (use,)
    frontiers = tuple(
        frontier for _, pending in classified for frontier in pending
    )
    return True, frontiers


def _materialize_relinearization(
    operation: Operation,
    *,
    edge: Use | None = None,
) -> tuple[int, SSAValue]:
    opaque_cast = (
        edge.operation
        if edge is not None and is_same_dialect_ckks_cast(edge.operation)
        else None
    )
    if opaque_cast is not None:
        assert isinstance(opaque_cast, UnrealizedConversionCastOp)
        source = opaque_cast.outputs[0]
        if not is_three_component(source):
            raise ValueError(
                "relinearization cannot cross a same-dialect cast whose "
                "result is not a three-component ciphertext"
            )
    else:
        source = (
            operation.results[0]
            if edge is None
            else edge.operation.operands[edge.index]
        )
    state = represented_state(source) or {}
    domain = state.get("polynomial_domain")
    block = operation.parent_block()
    if block is None:
        raise ValueError("relinearization carrier is not attached to a block")
    inserted = 0
    if edge is None or opaque_cast is not None:
        insertion_point = operation if opaque_cast is None else opaque_cast

        def insert(created: Operation) -> None:
            nonlocal insertion_point
            insert_after_and_replace_uses(insertion_point, created)
            insertion_point = created

    else:
        consumer = edge.operation
        operand_index = edge.index

        def insert(created: Operation) -> None:
            block.insert_op_before(created, consumer)

    if not (isinstance(domain, StringAttr) and domain.data == "ntt"):
        ntt = ckks.ToNttOp(
            source,
            ciphertext_type(
                source,
                domain="ntt",
                residues="montgomery",
            ),
        )
        ntt.result.name_hint = f"{display_name(operation)}_ntt"
        insert(ntt)
        source = ntt.result
        inserted += 1
    result_type = ciphertext_type(
        source,
        domain="coefficient",
        residues="standard",
        components=2,
    )
    relinearized = ckks.RelinearizeOp(source, result_type)
    relinearized.result.name_hint = f"{display_name(operation)}_relinearized"
    insert(relinearized)
    if edge is not None and opaque_cast is None:
        consumer.operands[operand_index] = relinearized.result
    return inserted + 1, relinearized.result


def _place_relinearizations(program: Program, *, late: bool) -> PassResult:
    if len(program.functions) != 1:
        raise ValueError("relinearization placement requires one function")
    block = program.single_block("main")
    if any(
        operation.regions or operation.successors for operation in block.ops
    ):
        raise ValueError(
            "relinearization placement supports a flat single-block SSA DAG"
        )

    pending: set[Operation] = {
        operation
        for operation in block.ops
        if isinstance(operation, ckks.MultiplyOp)
        and len(operation.results) == 1
        and is_three_component(operation.results[0])
    }
    matched = len(pending)
    moved = consumed = 0

    changed = True
    while changed:
        changed = False
        for operation in tuple(block.ops):
            if isinstance(operation, ckks.RelinearizeOp):
                source, source_value, _ = source_through_casts(operation.value)
                if (
                    source in pending
                    and source is not None
                    and exclusive_to(source_value, operation)
                ):
                    pending.remove(source)
                    consumed += 1
                    changed = True
                continue

            if isinstance(operation, _TRANSPARENT_TYPES):
                source, source_value, casts = source_through_casts(
                    operation.operands[0]
                )
                if (
                    source not in pending
                    or source is None
                    or not exclusive_to(source_value, operation)
                ):
                    continue
                state = represented_state(source_value)
                if state is None or not is_three_component(source_value):
                    continue
                retype_path(casts, state)
                pending.remove(source)
                pending.add(operation)
                moved += 1
                changed = True
                continue

            if not late:
                continue
            if isinstance(operation, ckks.NegateOp):
                source, source_value, casts = source_through_casts(
                    operation.value
                )
                if (
                    source not in pending
                    or source is None
                    or not exclusive_to(source_value, operation)
                ):
                    continue
                state = represented_state(source_value)
                if state is None or not is_three_component(source_value):
                    continue
                retype_path(casts, state)
                retype_result(operation.result, state)
                pending.remove(source)
                pending.add(operation)
                moved += 1
                changed = True
                continue
            if not isinstance(operation, (ckks.AddOp, ckks.SubtractOp)):
                continue
            sources: list[Operation] = []
            cast_paths = []
            states: list[dict[str, Attribute]] = []
            for operand in operation.operands:
                source, source_value, casts = source_through_casts(operand)
                state = represented_state(source_value)
                if (
                    source not in pending
                    or source is None
                    or not exclusive_to(source_value, operation)
                    or state is None
                    or not is_three_component(source_value)
                ):
                    break
                sources.append(source)
                cast_paths.append(casts)
                states.append(state)
            if len(sources) != len(operation.operands) or not states:
                continue
            if any(
                not compatible_states(states[0], state) for state in states[1:]
            ):
                continue
            for casts in cast_paths:
                retype_path(casts, states[0])
            retype_result(operation.result, states[0])
            pending.difference_update(sources)
            pending.add(operation)
            moved += 1
            changed = True

    inserted = 0
    rewritten_roots = []
    for operation in tuple(block.ops):
        if operation not in pending:
            continue
        classified = tuple(
            _classify_relinearization_use(use)
            for use in tuple(operation.results[0].uses)
        )
        frontiers = tuple(
            frontier
            for _, pending_uses in classified
            for frontier in pending_uses
        )
        if any(satisfied for satisfied, _ in classified) or any(
            is_same_dialect_ckks_cast(frontier.operation)
            for frontier in frontiers
        ):
            for use in frontiers:
                created, result = _materialize_relinearization(
                    operation, edge=use
                )
                inserted += created
                rewritten_roots.append(result)
            continue
        created, result = _materialize_relinearization(operation)
        inserted += created
        rewritten_roots.append(result)

    if inserted:
        inserted += reconcile_transition_paths(program, tuple(rewritten_roots))
        program.function("main").update_function_type()
    transformed = moved + consumed + inserted
    if transformed == 0:
        return PassResult.unchanged(program, matched=matched)
    return PassResult(
        program,
        PassStats(
            matched=max(matched, transformed),
            transformed=transformed,
            inserted=inserted,
        ),
    )


@dataclass(frozen=True)
class InsertRelinearizationPass:
    """Relinearize each CT3 multiplication before its first non-transition use."""

    name: str = "insert-relinearization"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        del workspace
        return _place_relinearizations(program, late=False)


@dataclass(frozen=True)
class LateRelinearizationPass:
    """Coalesce compatible CT3 add/sub/negate regions before relinearization."""

    name: str = "late-relinearization"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        del workspace
        return _place_relinearizations(program, late=True)


__all__ = ["InsertRelinearizationPass", "LateRelinearizationPass"]
