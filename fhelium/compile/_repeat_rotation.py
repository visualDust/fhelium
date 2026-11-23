"""Represent keyed rotation tails and their ordered accumulation once per layout.

The operand table retains each original key reference and coefficient-domain input. Consecutive rows with the same key layout share one body; slot steps and their Galois elements remain per-round attributes. Each round's Galois tables are declared materials in its operand row, so material binding before linking covers every round. Backend planning expands the selected body without repeating lowering or fusion matching.
"""

from __future__ import annotations


from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Block
from xdsl.rewriter import Rewriter

from fhelium.backend.ckks.rotation._galois import rotation_galois_element
from fhelium.config import CkksConfig
from fhelium.config._derived import config_from_text
from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import ckks, core, fusion, rns

from .passes.program._purity import known_pure


def is_rotation_repeat(operation):
    return (
        isinstance(operation, ckks.RepeatOp)
        and "rotation_steps" in operation.attributes
    )


def _tail(marker):
    uses = tuple(marker.outputs[0].uses)
    if len(uses) != 1 or not isinstance(uses[0].operation, ckks.FromNttOp):
        return None
    inverse = uses[0].operation
    uses = tuple(inverse.result.uses)
    if len(uses) != 1 or not isinstance(uses[0].operation, ckks.RotateOp):
        return None
    rotation = uses[0].operation
    uses = tuple(rotation.result.uses)
    if len(uses) != 1 or not isinstance(uses[0].operation, ckks.AddOp):
        return None
    add = uses[0].operation
    if add.operands[1] is not rotation.result:
        return None
    if not isinstance(rotation.key.owner, core.MaterialRefOp):
        return None
    if (
        rotation.input_domain.data != "coefficient"
        or rotation.output_domain.data != "coefficient"
    ):
        return None
    return marker, inverse, rotation, add


def _signature(row):
    _, inverse, rotation, add = row
    return (
        tuple(
            sorted(
                (k, v)
                for k, v in rotation.attributes.items()
                if k != "rotation_step"
            )
        ),
        tuple(rotation.parameters),
        tuple(add.operands[2:]),
        tuple(sorted(rotation.properties.items())),
        rotation.key.type,
        tuple(
            v.type
            for v in (
                inverse.result,
                rotation.result,
                *add.operands,
                *add.results,
            )
        ),
        tuple(sorted(add.attributes.items())),
        tuple(sorted(add.properties.items())),
    )


def _fold(rows, compilation, replacements):
    first = rows[0][2]
    block = first.parent_block()
    count = len(rows)
    carried = rows[0][3].operands[0]
    shared = tuple(dict.fromkeys((*first.parameters, *rows[0][3].operands[2:])))
    body = Block(
        arg_types=(
            IndexType(),
            *(value.type for value in shared),
            rows[0][1].result.type,
            first.key.type,
            carried.type,
        )
    )
    loop = ckks.RepeatOp(
        shared,
        tuple(
            value
            for _, inverse, rotation, _ in rows
            for value in (inverse.result, rotation.key)
        ),
        (carried,),
        (rows[0][3].result.type,) * (count + 1),
        body,
        count=count,
        yield_count=1,
    )
    configuration = first.attributes.get("ckks_config")
    config = (
        config_from_text(configuration.data)
        if configuration is not None
        else compilation.workspace[CkksConfig]
    )
    steps = tuple(
        int(rotation.rotation_step.value.data) for _, _, rotation, _ in rows
    )
    loop.attributes["rotation_steps"] = ArrayAttr(
        IntegerAttr(step, 64) for step in steps
    )
    loop.attributes["rotation_galois_elements"] = ArrayAttr(
        IntegerAttr(
            rotation_galois_element(config.N, step, config.galois_generator), 64
        )
        for step in steps
    )
    block.insert_op_before(loop, first)
    selected = {op for _, _, rotation, add in rows for op in (rotation, add)}
    for i, (_, _, _, add) in enumerate(rows):
        result = loop.outputs[i]
        replacements[add.result] = result
        for use in tuple(add.result.uses):
            if use.operation not in selected:
                use.operation.operands[use.index] = result
    rotation, add = rows[0][2:]
    block.detach_op(rotation)
    block.detach_op(add)
    parameters = dict(zip(shared, body.args[1 : 1 + len(shared)], strict=True))
    offset = 1 + len(shared)
    rotation.operands = (
        body.args[offset],
        body.args[offset + 1],
        *(parameters[v] for v in rotation.parameters),
    )
    add.operands = (
        body.args[offset + 2],
        rotation.result,
        *(parameters[v] for v in add.operands[2:]),
    )
    body.add_ops((rotation, add, ckks.RepeatYieldOp((add.result, add.result))))
    for _, _, rotation, add in reversed(rows[1:]):
        Rewriter.erase_op(add, safe_erase=False)
        Rewriter.erase_op(rotation, safe_erase=False)


def fold_rotation_tails(compilation, *, observed=()):
    """Fold the nonzero keyed tails of a captured independent accumulation."""
    if compilation.workspace.get("ckks_repeat_rotation", True) is False:
        return {}
    replacements = {}
    for outer in tuple(compilation.program.walk()):
        if (
            not isinstance(outer, ckks.RepeatOp)
            or outer.carried
            or outer.yield_count.value.data != 1
        ):
            continue
        rows = []
        for value in outer.outputs[1:]:
            uses = tuple(value.uses)
            if len(uses) != 1 or not isinstance(
                uses[0].operation, ckks.RepeatIterationOp
            ):
                break
            row = _tail(uses[0].operation)
            if row is None or row[2].result in observed:
                break
            rows.append(row)
        else:
            if len(rows) < 2:
                continue
            if any(
                row[3].operands[0] is not previous[3].result
                for previous, row in zip(rows, rows[1:])
            ):
                continue
            block = outer.parent_block()
            if any(
                op.parent_block() is not block for row in rows for op in row
            ):
                continue
            operations = tuple(block.ops)
            start, end = operations.index(outer), operations.index(rows[-1][3])
            if any(
                not known_pure(op)
                or EXECUTION_IMPLEMENTATION_ATTRIBUTE in op.attributes
                for op in operations[start : end + 1]
            ):
                continue
            # These inverse endpoints are already independent. Moving them ahead
            # exposes exactly the same large endpoint fusion as the arithmetic loop.
            anchor = rows[0][2]
            for marker, inverse, rotation, _ in rows:
                for op in (marker, inverse, rotation.key.owner):
                    if not op.is_before_in_block(anchor):
                        block.detach_op(op)
                        block.insert_op_before(op, anchor)
            groups = []
            for row in rows:
                if groups and _signature(groups[-1][0]) == _signature(row):
                    groups[-1].append(row)
                else:
                    groups.append([row])
            for group in groups:
                if len(group) >= 2:
                    _fold(group, compilation, replacements)
    return replacements


def specialize_round(compilation, loop, cloned, iteration):
    """Bind the selected round's step, Galois tables and fused argument types.

    ``cloned`` is attached; new table references are inserted before it.
    """
    if not is_rotation_repeat(loop):
        return
    from .passes.lowering._keyswitch import rebind_galois_tables

    for operation in tuple(cloned.walk()):
        if isinstance(operation, ckks.RotateOp):
            operation.attributes["rotation_step"] = loop.attributes[
                "rotation_steps"
            ].data[iteration]
        elif isinstance(operation, rns.CoefficientAutomorphismOp):
            element = loop.attributes["rotation_galois_elements"].data[
                iteration
            ]
            for reference in rebind_galois_tables(
                compilation, operation, int(element.value.data)
            ):
                cloned.parent_block().insert_op_before(reference, cloned)
        elif isinstance(operation, fusion.FusedOp):
            for argument, value in zip(
                tuple(operation.body.block.args), operation.inputs, strict=True
            ):
                if argument.type != value.type:
                    Rewriter.replace_value_with_new_type(argument, value.type)


def round_galois_tables(compilation, loop):
    """Move the body's Galois tables into the operand table, one declaration per round."""
    from .passes.lowering._keyswitch import declare_galois_table

    body = loop.body.block
    tables = dict.fromkeys(
        operation.operands[index]
        for operation in body.walk()
        if isinstance(operation, rns.CoefficientAutomorphismOp)
        for index in (2, 3)
    )
    if not tables:
        return
    block = loop.parent_block()
    shared, carried = loop.shared, loop.carried
    width, count = loop.table_width.value.data, loop.count.value.data
    rows = [list(loop.row(index)) for index in range(count)]
    elements = tuple(
        int(element.value.data)
        for element in loop.attributes["rotation_galois_elements"].data
    )
    inside = set(body.walk())
    for table in tables:
        if table.owner in inside:
            table.owner.parent_block().detach_op(table.owner)
            block.insert_op_before(table.owner, loop)
        argument = body.insert_arg(table.type, 1 + len(shared) + width)
        width += 1
        for row, element in zip(rows, elements, strict=True):
            references, value = declare_galois_table(
                compilation, loop, table, element
            )
            for reference in references:
                block.insert_op_before(reference, loop)
            row.append(value)
        for use in tuple(table.uses):
            if use.operation in inside:
                use.operation.operands[use.index] = argument
        if not table.uses and table not in (v for row in rows for v in row):
            Rewriter.erase_op(table.owner)
    loop.operands = (*shared, *(v for row in rows for v in row), *carried)
    loop.attributes["table_width"] = IntegerAttr(width, 64)


def hoist_materials(loop):
    """Expose immutable numerical declarations to the enclosing reuse pass."""
    block = loop.parent_block()
    body = loop.body.block
    for operation in tuple(body.ops):
        if isinstance(operation, core.MaterialRefOp):
            body.detach_op(operation)
            block.insert_op_before(operation, loop)
    external = dict.fromkeys(
        value
        for operation in body.ops
        for value in operation.operands
        if value.owner is not body and value.owner.parent_block() is not body
    )
    for value in external:
        shared = loop.shared_count.value.data
        argument = body.insert_arg(value.type, 1 + shared)
        for use in tuple(value.uses):
            if use.operation.parent_block() is body:
                use.operation.operands[use.index] = argument
        loop.operands = (*loop.inputs[:shared], value, *loop.inputs[shared:])
        loop.attributes["shared_count"] = IntegerAttr(shared + 1, 64)


def normalize_rotation_ports(loop):
    """Cancel representation casts across a homogeneous rotation body interface."""
    body = loop.body.block
    shared, width, count = (
        loop.shared_count.value.data,
        loop.table_width.value.data,
        loop.count.value.data,
    )
    for index, argument in enumerate(tuple(body.args)[1:]):
        uses = tuple(argument.uses)
        if not uses or any(
            type(use.operation) is not UnrealizedConversionCastOp
            for use in uses
        ):
            continue
        typ = uses[0].operation.outputs[0].type
        if any(
            len(use.operation.inputs) != 1
            or len(use.operation.outputs) != 1
            or use.operation.outputs[0].type != typ
            for use in uses
        ):
            continue
        indices = (
            (index,)
            if index < shared
            else tuple(
                shared + i * width + index - shared for i in range(count)
            )
            if index < shared + width
            else (shared + count * width + index - shared - width,)
        )
        sources = []
        for operand_index in indices:
            value = loop.inputs[operand_index]
            if (
                type(value.owner) is UnrealizedConversionCastOp
                and value.owner.inputs[0].type == typ
            ):
                value = value.owner.inputs[0]
            if value.type != typ:
                break
            sources.append(value)
        else:
            replacement = Rewriter.replace_value_with_new_type(argument, typ)
            for use in uses:
                Rewriter.replace_op(
                    use.operation, (), new_results=(replacement,)
                )
            for operand_index, value in zip(indices, sources, strict=True):
                loop.operands[operand_index] = value
    terminator = body.last_op
    value = terminator.operands[0]
    cast = value.owner
    if (
        type(cast) is not UnrealizedConversionCastOp
        or len(cast.inputs) != 1
        or any(other is not value for other in terminator.operands)
        or loop.carried[0].type != cast.inputs[0].type
    ):
        return
    source = cast.inputs[0]
    terminator.operands = (source,) * len(terminator.operands)
    for old in tuple(loop.outputs):
        old_type = old.type
        replacement = Rewriter.replace_value_with_new_type(old, source.type)
        restored = None
        for use in tuple(replacement.uses):
            operation = use.operation
            if (
                type(operation) is UnrealizedConversionCastOp
                and len(operation.inputs) == 1
                and len(operation.outputs) == 1
                and operation.outputs[0].type == source.type
            ):
                Rewriter.replace_op(operation, (), new_results=(replacement,))
            else:
                if restored is None:
                    restored = UnrealizedConversionCastOp.get(
                        (replacement,), (old_type,)
                    )
                    loop.parent_block().insert_op_after(restored, loop)
                operation.operands[use.index] = restored.outputs[0]
    if not cast.outputs[0].uses:
        Rewriter.erase_op(cast)


def supports_rotation_policy(program_pass):
    """Keep step-varying rows represented only for the built-in selection rules."""
    from fhelium.backend.ckks.rotation.operations import (
        NativeRotateImplementation,
    )
    from fhelium.backend.triton._pointwise import TritonFusionImplementation
    from fhelium.backend.triton._tensor import TritonTensorFusionImplementation
    from .passes.fusion import FuseOperationsPass
    from .passes.lowering._select_execution import SelectExecutionLoweringsPass

    if type(program_pass) is SelectExecutionLoweringsPass:
        if any(
            ckks.RotateOp in implementation.operation_types
            and type(implementation) is not NativeRotateImplementation
            for implementation in program_pass.implementations.implementations
        ):
            return False
        implementations = program_pass.fusion_implementations
    elif type(program_pass) is FuseOperationsPass:
        implementations = program_pass.implementations
    else:
        return True
    return all(
        type(implementation)
        in (TritonFusionImplementation, TritonTensorFusionImplementation)
        for implementation in implementations
    )
