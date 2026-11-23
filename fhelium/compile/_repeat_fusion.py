"""Retain ordered MAC reductions and independent CKKS endpoint bodies.

Carried multiply-add regions expose one final accumulator. Independent repeats expose their initial terms to the scheduler and absorb matching inverse-transform successors. Both forms retain one arithmetic body inside their selected fusion region.
"""

from __future__ import annotations

from xdsl.dialects.builtin import IndexType, UnrealizedConversionCastOp
from xdsl.ir import Block
from xdsl.rewriter import Rewriter

from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import ckks, core
from fhelium.backend.triton._expressions import SUPPORTED_OPERATIONS
from fhelium.backend.triton._repeat import _INVERSE

from ._repeat import expand_repeats
from .passes.program._purity import known_pure


def _endpoint(loop):
    """Prove single-use, same-state inverse successors for every iteration."""
    if (
        loop.carried
        or loop.yield_count.value.data != 1
        or loop.count.value.data < 4
    ):
        return None
    body = tuple(loop.body.block.ops)
    inner = [op for op in body if isinstance(op, ckks.RepeatOp)]
    if len(inner) != 1 or len(inner[0].carried) != 1:
        return None
    prefix = body[: body.index(inner[0])]
    if not any(isinstance(op, SUPPORTED_OPERATIONS) for op in prefix):
        return None
    if any(op.regions for op in prefix):
        return None
    rows = []
    signatures = []
    for value in loop.outputs:
        uses = tuple(value.uses)
        if len(uses) != 1 or not isinstance(
            uses[0].operation, ckks.RepeatIterationOp
        ):
            return None
        marker = uses[0].operation
        if len(marker.outputs) != 1:
            return None
        chain = []
        current = marker.outputs[0]
        while True:
            uses = tuple(current.uses)
            if len(uses) != 1:
                return None
            op = uses[0].operation
            if op.parent_block() is not loop.parent_block():
                return None
            if (
                type(op) is UnrealizedConversionCastOp
                and len(op.inputs) == len(op.outputs) == 1
            ):
                chain.append(op)
                current = op.outputs[0]
                continue
            if isinstance(op, _INVERSE) and op.operands[0] is current:
                chain.append(op)
                break
            return None
        if any(
            not isinstance(v.owner, core.MaterialRefOp)
            for v in chain[-1].operands[1:]
        ):
            return None
        signatures.append(
            tuple(
                (
                    type(op),
                    tuple(sorted(op.attributes.items())),
                    tuple(sorted(op.properties.items())),
                    tuple(v.type for v in op.results),
                    tuple((v.owner.symbol, v.type) for v in op.operands[1:]),
                )
                for op in chain
            )
        )
        rows.append((marker, tuple(chain)))
    if any(signature != signatures[0] for signature in signatures[1:]):
        return None
    # Preserve the scheduler's effect and caller-assignment windows when moving
    # successors into the body. Their original positions remain the fallback.
    operations = tuple(loop.parent_block().ops)
    start = operations.index(loop)
    stop = max(operations.index(chain[-1]) for _, chain in rows)
    if any(
        not known_pure(op)
        or EXECUTION_IMPLEMENTATION_ATTRIBUTE in op.attributes
        for op in operations[start : stop + 1]
    ):
        return None
    return inner[0], prefix, rows


def _partition(loop, inner, prefix, rows, compilation):
    block = loop.parent_block()
    # Expand the ordered accumulation once, not once per independent group.
    expand_repeats(compilation, roots=(inner,))
    body = loop.body.block
    from .passes.program._reuse_intermediates import reuse_blocks

    reuse_blocks(compilation, (body,))
    terminator = body.last_op
    for op in reversed(tuple(body.ops)[:-1]):
        if not any(result.uses for result in op.results):
            Rewriter.erase_op(op)
    prefix = set(prefix)
    for op in tuple(body.ops):
        if (
            type(op) is UnrealizedConversionCastOp
            and op.inputs[0].owner in prefix
        ):
            prefix.add(op)
    references = [op for op in body.ops if isinstance(op, core.MaterialRefOp)]
    # Numerical declarations are ordinary immutable input bindings. Move them to
    # the enclosing block; the following global reuse pass resolves duplicates.
    for op in references:
        body.detach_op(op)
        block.insert_op_before(op, loop)
    for _, chain in rows:
        for value in chain[-1].operands[1:]:
            op = value.owner
            if not op.is_before_in_block(loop):
                block.detach_op(op)
                block.insert_op_before(op, loop)
    prefix.difference_update(references)
    prefix = tuple(op for op in body.ops if op in prefix)
    count = loop.count.value.data
    columns = {
        argument: tuple(
            (*loop.shared, *loop.row(i), *loop.carried)[j] for i in range(count)
        )
        for j, argument in enumerate(tuple(body.args)[1:])
    }
    for reference in references:
        columns[reference.value] = (reference.value,) * count
    for i in range(count):
        mapper = {arg: values[i] for arg, values in columns.items()}
        for op in prefix:
            cloned = op.clone(value_mapper=mapper)
            block.insert_op_before(cloned, loop)
            for old, new in zip(op.results, cloned.results, strict=True):
                if old not in columns:
                    columns[old] = [None] * count
                columns[old][i] = new
    selected = set(prefix)
    remaining = tuple(
        op for op in body.ops if op not in selected and op is not terminator
    )
    # All nested loops have been expanded; the represented body is a pure DAG.
    new_body = Block(arg_types=[IndexType()])
    shared, tables = [], []
    mapper = {}

    def bind(value):
        if value in mapper:
            return mapper[value]
        values = columns.get(value)
        if values is None:
            values = (value,) * count
        if all(v is values[0] for v in values):
            index = 1 + len(shared)
            argument = new_body.insert_arg(value.type, index)
            shared.append(values[0])
        else:
            argument = new_body.insert_arg(value.type, len(new_body.args))
            tables.append(tuple(values))
        mapper[value] = argument
        return argument

    for op in remaining:
        for value in op.operands:
            bind(value)
        new_body.add_op(op.clone(value_mapper=mapper))
    # The first successor chain is the representative. All other chains have
    # identical types, policies and material roles, as checked above.
    marker, chain = rows[0]
    mapper[marker.outputs[0]] = mapper[terminator.values[0]]
    for op in chain:
        for value in op.operands:
            bind(value)
        new_body.add_op(op.clone(value_mapper=mapper))
    yielded = mapper[chain[-1].results[0]]
    new_body.add_op(ckks.RepeatYieldOp((yielded,)))
    replacement = ckks.RepeatOp(
        shared,
        tuple(column[i] for i in range(count) for column in tables),
        (),
        (yielded.type,) * count,
        new_body,
        count=count,
        yield_count=1,
    )
    block.insert_op_before(replacement, loop)
    for i, (marker, chain) in enumerate(rows):
        chain[-1].results[0].replace_all_uses_with(replacement.outputs[i])
        for op in reversed(chain):
            Rewriter.erase_op(op)
        Rewriter.erase_op(marker)
    Rewriter.erase_op(loop)
    return replacement


def _carried(loop):
    """Remove unused round observations while preserving the final ordered carry."""
    from fhelium.backend.triton._mac import mac_body
    from ._repeat_rotation import hoist_materials, normalize_rotation_ports

    if len(loop.carried) != 1:
        return False
    # Only the last observation can escape a runtime accumulator.
    width, count = loop.yield_count.value.data, loop.count.value.data
    markers = []
    for index, value in enumerate(loop.outputs[:-1]):
        for use in tuple(value.uses):
            op = use.operation
            if isinstance(op, ckks.RepeatIterationOp):
                if index < (count - 1) * width and any(
                    v.uses for v in op.outputs
                ):
                    return False
                markers.append(op)
            elif index < (count - 1) * width:
                return False
    hoist_materials(loop)
    normalize_rotation_ports(loop)
    if mac_body(loop) is None:
        return False
    body = loop.body.detach_block(0)
    body.last_op.operands = (body.last_op.operands[-1],)
    replacement = ckks.RepeatOp(
        loop.shared,
        tuple(v for i in range(count) for v in loop.row(i)),
        loop.carried,
        (loop.outputs[-1].type,),
        body,
        count=count,
        yield_count=0,
    )
    block = loop.parent_block()
    block.insert_op_before(replacement, loop)
    for value in loop.outputs:
        value.replace_all_uses_with(replacement.outputs[0])
    for marker in dict.fromkeys(markers):
        for value, source in zip(marker.outputs, marker.inputs, strict=True):
            value.replace_all_uses_with(source)
        Rewriter.erase_op(marker)
    Rewriter.erase_op(loop)
    return True


def prepare_fusion_repeats(compilation):
    """Retain supported independent endpoint bodies and expand other repeats."""
    loops = [
        op
        for op in compilation.program.walk()
        if isinstance(op, ckks.RepeatOp)
        and not isinstance(op.parent_op(), ckks.RepeatOp)
    ]
    retained = 0
    from ._repeat_rotation import (
        hoist_materials,
        is_rotation_repeat,
        normalize_rotation_ports,
        round_galois_tables,
    )

    for loop in loops:
        if is_rotation_repeat(loop):
            normalize_rotation_ports(loop)
            round_galois_tables(compilation, loop)
            hoist_materials(loop)
            retained += 1
            continue
        if compilation.workspace.get("ckks_repeat_mac", True) and _carried(
            loop
        ):
            retained += 1
            continue
        plan = _endpoint(loop)
        if plan is None:
            expand_repeats(compilation, roots=(loop,))
            continue
        _partition(loop, *plan, compilation)
        retained += 1
    return retained
