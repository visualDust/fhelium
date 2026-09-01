"""Reconcile represented CKKS value state after transition placement."""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, SSAValue, Use
from xdsl.rewriter import Rewriter

from fhelium.ir import Program
from fhelium.ir.dialects import ckks
from fhelium.ir.dialects._common import OpenStateType


_CKKS_VALUE_BRIDGE_TYPES = frozenset(
    {
        "fhelium_ckks.ciphertext",
        "fhelium_logical.encrypted",
        "fhelium_semantic.secret",
    }
)


def is_ckks_value_bridge(operation: object) -> bool:
    """Identify a cross-dialect cast that carries one encrypted value.

    Same-dialect casts may intentionally refine represented CKKS state and are
    never transparent to transition placement.
    """

    if not isinstance(operation, UnrealizedConversionCastOp):
        return False
    if len(operation.inputs) != 1 or len(operation.outputs) != 1:
        return False
    source_name = operation.inputs[0].type.name
    result_name = operation.outputs[0].type.name
    return (
        source_name != result_name
        and source_name in _CKKS_VALUE_BRIDGE_TYPES
        and result_name in _CKKS_VALUE_BRIDGE_TYPES
    )


def is_same_dialect_ckks_cast(operation: object) -> bool:
    """Identify an opaque cast between two CKKS ciphertext states."""

    return (
        isinstance(operation, UnrealizedConversionCastOp)
        and len(operation.inputs) == 1
        and len(operation.outputs) == 1
        and operation.inputs[0].type.name == "fhelium_ckks.ciphertext"
        and operation.outputs[0].type.name == "fhelium_ckks.ciphertext"
    )


def source_through_casts(
    value: SSAValue,
) -> tuple[Operation | None, SSAValue, tuple[UnrealizedConversionCastOp, ...]]:
    """Return the producer reached through one-to-one state casts."""

    casts: list[UnrealizedConversionCastOp] = []
    current = value
    while is_ckks_value_bridge(current.owner):
        cast = current.owner
        assert isinstance(cast, UnrealizedConversionCastOp)
        casts.append(cast)
        current = cast.inputs[0]
    owner = current.owner
    return (
        owner if isinstance(owner, Operation) else None,
        current,
        tuple(casts),
    )


def terminal_uses(value: SSAValue) -> tuple[Use, ...]:
    """Return uses reached through one-to-one state casts."""

    terminal: list[Use] = []
    work = [value]
    visited: set[SSAValue] = set()
    while work:
        current = work.pop()
        if current in visited:
            continue
        visited.add(current)
        for use in current.uses:
            operation = use.operation
            if is_ckks_value_bridge(operation):
                assert isinstance(operation, UnrealizedConversionCastOp)
                work.append(operation.outputs[0])
            else:
                terminal.append(use)
    return tuple(terminal)


def exclusive_to(value: SSAValue, consumer: Operation) -> bool:
    """Return whether every terminal use reaches one consumer."""

    uses = terminal_uses(value)
    return bool(uses) and all(use.operation is consumer for use in uses)


def represented_state(value: SSAValue) -> dict[str, Attribute] | None:
    """Return a mutable copy of represented open state when present."""

    state = getattr(getattr(value.type, "state", None), "data", None)
    return dict(state) if isinstance(state, Mapping) else None


def compatible_states(
    lhs: dict[str, Attribute], rhs: dict[str, Attribute]
) -> bool:
    """Return whether two values can share a linear CKKS transition region."""

    represented = (
        "basis",
        "polynomial_domain",
        "residue_representation",
        "components",
    )
    if any(lhs.get(field) != rhs.get(field) for field in represented):
        return False
    for field in ("level", "scale", "prime_ids"):
        if field in lhs and field in rhs and lhs[field] != rhs[field]:
            return False
    return True


def is_three_component(value: SSAValue) -> bool:
    """Return whether a value represents a Q-basis CT3 ciphertext."""

    state = represented_state(value)
    if state is None:
        return False
    components = state.get("components")
    basis = state.get("basis")
    return (
        isinstance(components, IntegerAttr)
        and components.value.data == 3
        and not (isinstance(basis, StringAttr) and basis.data != "Q")
    )


def retype_path(
    casts: tuple[UnrealizedConversionCastOp, ...],
    state: dict[str, Attribute],
) -> None:
    """Make transparent cast results represent the propagated state."""

    for cast in reversed(casts):
        output = cast.outputs[0]
        if not isinstance(output.type, OpenStateType):
            raise TypeError("CKKS transition path cast must carry open state")
        Rewriter.replace_value_with_new_type(
            output,
            output.type.with_state(state),  # type: ignore[arg-type]
        )


def retype_result(value: SSAValue, state: dict[str, Attribute]) -> SSAValue:
    """Replace one result with the same value kind and new represented state."""

    if not isinstance(value.type, OpenStateType):
        raise TypeError("CKKS transition result must carry open state")
    return Rewriter.replace_value_with_new_type(
        value,
        value.type.with_state(state),  # type: ignore[arg-type]
    )


_PATH_OPERATION_TYPES = (
    UnrealizedConversionCastOp,
    ckks.AddOp,
    ckks.SubtractOp,
    ckks.NegateOp,
    ckks.RelinearizeOp,
    ckks.RescaleOp,
    ckks.ToNttOp,
    ckks.FromNttOp,
    ckks.ToMontgomeryResiduesOp,
    ckks.ToStandardResiduesOp,
)


def _reachable_path_operations(roots: tuple[SSAValue, ...]) -> set[Operation]:
    """Return known CKKS state carriers downstream from rewritten values."""

    reachable: set[Operation] = set()
    work = list(roots)
    visited: set[SSAValue] = set()
    while work:
        value = work.pop()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            operation = use.operation
            if isinstance(
                operation, UnrealizedConversionCastOp
            ) and not is_ckks_value_bridge(operation):
                continue
            if not isinstance(operation, _PATH_OPERATION_TYPES):
                continue
            reachable.add(operation)
            work.extend(operation.results)
    return reachable


def _refresh_reachable_states(
    program: Program, reachable: set[Operation]
) -> None:
    """Refresh represented state only on affected, known CKKS paths."""

    block = program.single_block("main")
    for operation in tuple(block.ops):
        if operation not in reachable:
            continue
        if isinstance(operation, UnrealizedConversionCastOp):
            if not is_ckks_value_bridge(operation):
                continue
            state = represented_state(operation.inputs[0])
            if state is not None and isinstance(
                operation.outputs[0].type, OpenStateType
            ):
                retype_result(operation.outputs[0], state)
            continue
        if not operation.results:
            continue
        state: dict[str, Attribute] | None = None
        if isinstance(operation, (ckks.AddOp, ckks.SubtractOp)):
            lhs = represented_state(operation.lhs)
            rhs = represented_state(operation.rhs)
            if (
                lhs is not None
                and rhs is not None
                and compatible_states(lhs, rhs)
            ):
                state = lhs
        elif isinstance(operation, ckks.NegateOp):
            state = represented_state(operation.value)
        elif isinstance(operation, ckks.RelinearizeOp):
            state = represented_state(operation.value)
            if state is not None:
                state.update(
                    {
                        "components": IntegerAttr(2, 64),
                        "polynomial_domain": StringAttr("coefficient"),
                        "residue_representation": StringAttr("standard"),
                    }
                )
        elif isinstance(operation, ckks.RescaleOp):
            source_state = represented_state(operation.value)
            result_state = represented_state(operation.result)
            if source_state is not None:
                state = dict(source_state)
                if result_state is not None:
                    for field in ("level", "scale", "prime_ids"):
                        if field in result_state:
                            state[field] = result_state[field]
                state.update(
                    {
                        "polynomial_domain": StringAttr("coefficient"),
                        "residue_representation": StringAttr("standard"),
                    }
                )
        elif isinstance(operation, ckks.ToNttOp):
            state = represented_state(operation.value)
            if state is not None:
                state.update(
                    {
                        "polynomial_domain": StringAttr("ntt"),
                        "residue_representation": StringAttr("montgomery"),
                    }
                )
        elif isinstance(operation, ckks.FromNttOp):
            state = represented_state(operation.value)
            if state is not None:
                state.update(
                    {
                        "polynomial_domain": StringAttr("coefficient"),
                        "residue_representation": StringAttr("standard"),
                    }
                )
        elif isinstance(operation, ckks.ToMontgomeryResiduesOp):
            state = represented_state(operation.value)
            if state is not None:
                state["residue_representation"] = StringAttr("montgomery")
        elif isinstance(operation, ckks.ToStandardResiduesOp):
            state = represented_state(operation.value)
            if state is not None:
                state["residue_representation"] = StringAttr("standard")
        if state is not None and isinstance(
            operation.results[0].type, OpenStateType
        ):
            retype_result(operation.results[0], state)


def _normalize_reachable_inputs(
    program: Program, reachable: set[Operation]
) -> int:
    """Satisfy representation inputs changed on affected paths."""

    block = program.single_block("main")
    inserted = 0
    for operation in tuple(block.ops):
        if operation not in reachable:
            continue
        needs_ntt = isinstance(operation, (ckks.RelinearizeOp, ckks.FromNttOp))
        if not needs_ntt:
            continue
        source = operation.operands[0]
        state = represented_state(source) or {}
        domain = state.get("polynomial_domain")
        if isinstance(domain, StringAttr) and domain.data == "ntt":
            continue
        if not isinstance(source.type, OpenStateType):
            raise TypeError("CKKS transition input must carry open state")
        converted = ckks.ToNttOp(
            source,
            source.type.with_state(
                {
                    **state,
                    "polynomial_domain": StringAttr("ntt"),
                    "residue_representation": StringAttr("montgomery"),
                }
            ),
        )
        converted.result.name_hint = "transition_input_ntt"
        block.insert_op_before(converted, operation)
        operation.operands[0] = converted.result
        inserted += 1
    return inserted


def reconcile_transition_paths(
    program: Program, roots: tuple[SSAValue, ...]
) -> int:
    """Reconcile only paths whose values a placement pass rewrote."""

    if not roots or len(program.functions) != 1:
        return 0
    reachable = _reachable_path_operations(roots)
    _refresh_reachable_states(program, reachable)
    inserted = _normalize_reachable_inputs(program, reachable)
    if inserted:
        _refresh_reachable_states(program, reachable)
    return inserted


__all__ = [
    "compatible_states",
    "exclusive_to",
    "is_ckks_value_bridge",
    "is_same_dialect_ckks_cast",
    "is_three_component",
    "reconcile_transition_paths",
    "represented_state",
    "retype_path",
    "retype_result",
    "source_through_casts",
    "terminal_uses",
]
