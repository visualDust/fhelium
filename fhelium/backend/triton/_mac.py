"""Recognize ordered carried multiply-add bodies without expanding their rounds."""

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from fhelium.ir.dialects import ckks, rns


def uncast(value):
    while type(value.owner) is UnrealizedConversionCastOp:
        op = value.owner
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            break
        value = op.inputs[0]
    return value


def round_rows(loop):
    """Return the body's round-indexed row selections, or None for other round-index uses."""
    selections = []
    for use in loop.body.block.args[0].uses:
        operation = use.operation
        source = operation.operands[0] if use.index == 1 else None
        if (
            not isinstance(operation, ckks.SelectBatchOp)
            or source is None
            or source.owner is not loop.body.block
            or not 1 <= source.index <= loop.shared_count.value.data
        ):
            return None
        selections.append(operation)
    return tuple(selections)


def mac_body(loop):
    if not isinstance(loop, ckks.RepeatOp) or len(loop.carried) != 1:
        return None
    rows = round_rows(loop)
    if rows is None:
        return None
    operations = tuple(
        op
        for op in tuple(loop.body.block.ops)[:-1]
        if type(op) is not UnrealizedConversionCastOp and op not in rows
    )
    if len(operations) != 2:
        return None
    multiply, add = operations
    if not isinstance(
        multiply, (ckks.MultiplyCompressedPlaintextOp, rns.MultiplyPlaintextOp)
    ) or not isinstance(add, (rns.AddStandardOp, rns.AddMontgomeryLazyOp)):
        return None
    carry = loop.body.block.args[-1]
    if (
        uncast(add.operands[0]) is not carry
        or uncast(add.operands[1]) is not multiply.results[0]
        or any(
            uncast(v) is not add.results[0]
            for v in loop.body.block.last_op.operands
        )
        or uncast(multiply.operands[-1]) is not uncast(add.operands[-1])
    ):
        return None
    parameter = uncast(multiply.operands[-1])
    if (
        parameter.owner is not loop.body.block
        or not 1 <= parameter.index <= loop.shared_count.value.data
    ):
        return None
    return multiply, add
