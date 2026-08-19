"""xDSL operation helpers shared by JIT passes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.rewriter import Rewriter

from .._dialect import (
    create_ir_context,
    create_operation,
    operation_name,
    value_role,
)
from .._program import Program

OBLIGATIONS_ATTRIBUTE = "fhelium.scheduling_obligations"


def program_operations(program: Program) -> tuple[Operation, ...]:
    """Return direct local-rewrite candidates across the entire Program.

    Candidates include non-terminator operations directly contained in every
    block of every top-level registered function. Operations that own regions
    or successors remain outside this local flat-rewrite surface, preserving
    their structural and control-flow effects for specialized passes.
    """

    return tuple(
        operation
        for function in program.functions
        for block in function.body.blocks
        for operation in block.ops
        if not isinstance(operation, ReturnOp)
        and not operation.regions
        and not operation.successors
    )


def display_name(operation: Operation) -> str:
    """Return a stable diagnostic name for one operation."""

    if operation.results and operation.results[0].name_hint:
        return operation.results[0].name_hint
    return operation_name(operation)


def result_role(operation: Operation) -> str | None:
    """Return the known role of one single-result operation."""

    if len(operation.results) != 1:
        return None
    return value_role(operation.results[0])


def operand_role(operand: SSAValue) -> str | None:
    """Return the known role of one operand value."""

    return value_role(operand)


def bool_attribute(
    attributes: Mapping[str, Attribute], name: str, default: bool = False
) -> bool:
    """Read one bool-like pass trait from xDSL attributes."""

    value = attributes.get(name)
    if isinstance(value, IntegerAttr):
        return bool(value.value.data)
    if isinstance(value, StringAttr):
        if value.data == "true":
            return True
        if value.data == "false":
            return False
    return default


def string_attribute(
    attributes: Mapping[str, Attribute], name: str, default: str | None = None
) -> str | None:
    """Read one string attribute without coercing unknown attribute kinds."""

    value = attributes.get(name)
    return value.data if isinstance(value, StringAttr) else default


def with_bool_attribute(
    attributes: Mapping[str, Attribute], name: str, value: bool
) -> dict[str, Attribute]:
    """Copy attributes and set one bool trait."""

    result = dict(attributes)
    result[name] = IntegerAttr.from_bool(value)
    return result


def obligations(operation: Operation) -> frozenset[str]:
    """Return scheduling obligations attached to an operation."""

    value = operation.attributes.get(OBLIGATIONS_ATTRIBUTE)
    if not isinstance(value, ArrayAttr):
        return frozenset()
    return frozenset(
        item.data for item in value if isinstance(item, StringAttr)
    )


def with_obligations(
    attributes: Mapping[str, Attribute], values: Iterable[str]
) -> dict[str, Attribute]:
    """Copy attributes and set or remove scheduling obligations."""

    result = dict(attributes)
    ordered = tuple(sorted(frozenset(values)))
    if ordered:
        result[OBLIGATIONS_ATTRIBUTE] = ArrayAttr(
            StringAttr(item) for item in ordered
        )
    else:
        result.pop(OBLIGATIONS_ATTRIBUTE, None)
    return result


def replacement_operation(
    operation: Operation,
    name: str,
    *,
    operands: Sequence[SSAValue] | None = None,
    result_types: Sequence[Attribute] | None = None,
    attributes: Mapping[str, Attribute] | None = None,
) -> Operation:
    """Create a same-arity replacement and preserve SSA name hints."""

    replacement = create_operation(
        create_ir_context(),
        name,
        operands=(tuple(operation.operands) if operands is None else operands),
        result_types=(
            tuple(result.type for result in operation.results)
            if result_types is None
            else result_types
        ),
        attributes={
            **dict(operation.attributes),
            **dict(attributes or {}),
        },
        properties=operation.properties,
        location=operation.location,
    )
    for old_result, new_result in zip(
        operation.results, replacement.results, strict=True
    ):
        new_result.name_hint = old_result.name_hint
    return replacement


def replace_operation(operation: Operation, replacement: Operation) -> None:
    """Replace one operation and map each old result to the new result."""

    Rewriter.replace_op(
        operation,
        replacement,
        new_results=tuple(replacement.results),
    )


def insert_after_and_replace_uses(
    operation: Operation, inserted: Operation
) -> None:
    """Insert one unary result after ``operation`` and redirect prior uses."""

    if len(operation.results) != 1 or len(inserted.results) != 1:
        raise ValueError("use redirection requires single-result operations")
    block = operation.parent_block()
    if block is None:
        raise ValueError("operation is not attached to a block")
    prior_uses = tuple(
        use
        for use in operation.results[0].uses
        if use.operation is not inserted
    )
    block.insert_op_after(inserted, operation)
    for use in prior_uses:
        use.operation.operands[use.index] = inserted.results[0]


def users(value: SSAValue) -> tuple[Operation, ...]:
    """Return stable unique users of one SSA value."""

    result: list[Operation] = []
    for use in value.uses:
        if use.operation not in result:
            result.append(use.operation)
    return tuple(result)


__all__ = [
    "OBLIGATIONS_ATTRIBUTE",
    "bool_attribute",
    "display_name",
    "insert_after_and_replace_uses",
    "obligations",
    "operand_role",
    "program_operations",
    "replace_operation",
    "replacement_operation",
    "result_role",
    "string_attribute",
    "users",
    "with_bool_attribute",
    "with_obligations",
]
