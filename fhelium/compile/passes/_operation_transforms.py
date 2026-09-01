"""xDSL operation helpers shared by source-oriented transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects import scf
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.ir import operation_name, value_role
from fhelium.ir import Program
from fhelium.ir.dialects import ckks

_KNOWN_REGION_OPERATION_NAMES = frozenset(
    {
        scf.ForOp.name,
        scf.IfOp.name,
        "fhelium_dist.all_reduce",
    }
)


def _is_terminator(operation: Operation) -> bool:
    return isinstance(operation, (ReturnOp, scf.YieldOp)) or operation.name == (
        "fhelium_dist.yield"
    )


def _known_region_blocks(operation: Operation) -> tuple[Block, ...]:
    """Return blocks whose semantics are known to local Compile transforms."""

    if operation.name not in _KNOWN_REGION_OPERATION_NAMES:
        return ()
    return tuple(
        block for region in operation.regions for block in region.blocks
    )


def _block_operations(
    block: Block,
    *,
    include_region_owners: bool,
) -> tuple[Operation, ...]:
    result: list[Operation] = []
    for operation in tuple(block.ops):
        if _is_terminator(operation):
            continue
        blocks = _known_region_blocks(operation)
        if blocks:
            if include_region_owners:
                result.append(operation)
            for nested in blocks:
                result.extend(
                    _block_operations(
                        nested,
                        include_region_owners=include_region_owners,
                    )
                )
            continue
        if operation.regions or operation.successors:
            if include_region_owners:
                result.append(operation)
            continue
        result.append(operation)
    return tuple(result)


def program_operations(program: Program) -> tuple[Operation, ...]:
    """Return local leaf operations in known structured regions.

    The traversal enters ``scf.for``, ``scf.if``, and the registered generic
    distributed reduction region. Unknown region owners remain opaque so a
    local FHElium pass cannot accidentally rewrite vendor control flow.
    """

    return tuple(
        operation
        for function in program.functions
        for block in function.body.blocks
        for operation in _block_operations(
            block,
            include_region_owners=False,
        )
    )


def program_operations_and_known_region_owners(
    program: Program,
) -> tuple[Operation, ...]:
    """Return local operations plus owners of known structured regions."""

    return tuple(
        operation
        for function in program.functions
        for block in function.body.blocks
        for operation in _block_operations(
            block,
            include_region_owners=True,
        )
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


def ciphertext_type(
    value: SSAValue,
    *,
    domain: str | None = None,
    residues: str | None = None,
    components: int | None = None,
) -> ckks.CiphertextType:
    """Refine one encrypted value into an explicit CKKS ciphertext type."""

    state_attribute = getattr(value.type, "state", None)
    state = dict(getattr(state_attribute, "data", {}))
    state.pop("role", None)
    state.setdefault("basis", StringAttr("Q"))
    if domain is not None:
        state["polynomial_domain"] = StringAttr(domain)
    if residues is not None:
        state["residue_representation"] = StringAttr(residues)
    if components is not None:
        state["components"] = IntegerAttr(components, 64)
    return ckks.CiphertextType().with_state(state)


def cast_before(
    operation: Operation,
    value: SSAValue,
    result_type: Attribute,
    *,
    name_hint: str,
) -> SSAValue:
    """Materialize a visible type-changing edge immediately before an op."""

    if value.type == result_type:
        return value
    block = operation.parent_block()
    if block is None:
        raise ValueError("operation is not attached to a block")
    cast, result = UnrealizedConversionCastOp.cast_one(value, result_type)
    result.name_hint = name_hint
    block.insert_op_before(cast, operation)
    return result


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


def replacement_operation(
    operation: Operation,
    operation_type: type[Operation],
    *,
    operands: Sequence[SSAValue] | None = None,
    result_types: Sequence[Attribute] | None = None,
    attributes: Mapping[str, Attribute] | None = None,
) -> Operation:
    """Create a same-arity replacement and preserve SSA name hints."""

    replacement = operation_type.create(
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
    "bool_attribute",
    "cast_before",
    "ciphertext_type",
    "display_name",
    "insert_after_and_replace_uses",
    "operand_role",
    "program_operations",
    "replace_operation",
    "replacement_operation",
    "result_role",
    "string_attribute",
    "users",
    "with_bool_attribute",
]
