"""JSON-compatible structural encoding of Program modules.

The encoding stores every distinct attribute once as IR text and stores operations as tables of integer references, so decoding parses the attribute table in one parser call and constructs operations directly. Values are numbered in definition order and keep their SSA name hints: the arguments of every block in a region are defined when the region is entered, and operation results after the operation's regions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.parser import Parser

from ._structure import _AttributeText

ENCODING_VERSION = 1


def encode_module(module: ModuleOp) -> dict[str, Any]:
    """Encode ``module``; operands must be defined before their uses in preorder."""

    printed = _AttributeText()
    attributes: dict[bytes, int] = {}
    names: dict[str, int] = {}
    values: dict[SSAValue, int] = {}
    blocks: dict[Block, int] = {}

    def attribute(item: Attribute) -> int:
        text = printed(item)
        number = attributes.get(text)
        if number is None:
            number = attributes[text] = len(attributes)
        return number

    def name(text: str) -> int:
        number = names.get(text)
        if number is None:
            number = names[text] = len(names)
        return number

    def hints(items: Sequence[SSAValue]) -> list[str | None] | None:
        names = [item.name_hint for item in items]
        return names if any(name is not None for name in names) else None

    def reference(value: SSAValue) -> int:
        number = values.get(value)
        if number is None:
            raise ValueError("an operand is used before its definition")
        return number

    def region(item: Region) -> list[object]:
        encoded = []
        for block in item.blocks:
            blocks[block] = len(blocks)
            for argument in block.args:
                values[argument] = len(values)
            encoded.append(
                [
                    [attribute(argument.type) for argument in block.args],
                    hints(block.args),
                ]
            )
        return [
            [*arguments, [operation(op) for op in block.ops]]
            for arguments, block in zip(encoded, item.blocks, strict=True)
        ]

    def operation(op: Operation) -> list[object]:
        encoded = [
            name(op.name),
            [reference(value) for value in op.operands],
            [
                [name(key), attribute(value)]
                for key, value in op.attributes.items()
            ],
            [
                [name(key), attribute(value)]
                for key, value in op.properties.items()
            ],
            [region(item) for item in op.regions],
            [blocks[successor] for successor in op.successors],
        ]
        for result in op.results:
            values[result] = len(values)
        encoded.insert(2, [attribute(result.type) for result in op.results])
        encoded.append(hints(op.results))
        return encoded

    root = operation(module)
    return {
        "version": ENCODING_VERSION,
        "names": list(names),
        "attributes": [text.decode() for text in attributes],
        "module": root,
    }


def decode_module(payload: Mapping[str, Any], context: Context) -> ModuleOp:
    """Construct the module encoded by :func:`encode_module`."""

    if payload["version"] != ENCODING_VERSION:
        raise ValueError("unsupported Program encoding version")
    names: list[str] = payload["names"]
    parsed = Parser(
        context, "[" + ", ".join(payload["attributes"]) + "]", "<encoded>"
    ).parse_attribute()
    assert isinstance(parsed, ArrayAttr)
    attributes = parsed.data
    types: dict[str, type[Operation]] = {}
    values: list[SSAValue] = []
    blocks: list[Block] = []

    def name_values(items: Sequence[SSAValue], hints: list[str | None] | None):
        if hints is not None:
            for item, hint in zip(items, hints, strict=True):
                item.name_hint = hint

    def region(encoded: list[list[Any]]) -> Region:
        created = []
        for arguments, argument_hints, _ in encoded:
            block = Block(
                arg_types=[attributes[number] for number in arguments]
            )
            name_values(block.args, argument_hints)
            blocks.append(block)
            values.extend(block.args)
            created.append(block)
        for block, (_, _, operations) in zip(created, encoded, strict=True):
            block.add_ops(operation(item) for item in operations)
        return Region(created)

    def operation(encoded: list[Any]) -> Operation:
        (
            name,
            operands,
            results,
            items,
            properties,
            regions,
            successors,
            result_hints,
        ) = encoded
        op_name = names[name]
        op_type = types.get(op_name)
        if op_type is None:
            op_type = types[op_name] = context.get_op(op_name)
        op = op_type.create(
            operands=[values[number] for number in operands],
            result_types=[attributes[number] for number in results],
            attributes={names[key]: attributes[value] for key, value in items},
            properties={
                names[key]: attributes[value] for key, value in properties
            },
            regions=[region(item) for item in regions],
            successors=[blocks[number] for number in successors],
        )
        name_values(op.results, result_hints)
        values.extend(op.results)
        return op

    module = operation(payload["module"])
    if not isinstance(module, ModuleOp):
        raise ValueError("encoded Program root is not a module")
    return module


__all__ = ["decode_module", "encode_module"]
