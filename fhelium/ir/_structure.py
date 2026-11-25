"""Structural keys for alpha-equivalent IR operations and Programs."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Hashable, Mapping
from enum import Enum
from io import StringIO
from typing import cast

from xdsl.ir import (
    Attribute,
    Block,
    Data,
    Operation,
    ParametrizedAttribute,
    SSAValue,
)
from xdsl.printer import Printer


def structural_key(operation: Operation) -> Hashable:
    """Return a hashable key shared by alpha-equivalent operations.

    Two operations receive equal keys when they have the same name, attributes, properties and result types, and their nested regions contain blocks with equal argument types and operation sequences whose operands refer to the same local definitions. Values defined outside ``operation`` are numbered in first-use order and contribute their types, so repeated use of one outer value remains distinguishable from uses of two different values.
    """

    local: dict[SSAValue, int] = {}
    outer: dict[SSAValue, int] = {}
    blocks: dict[Block, int] = {}

    def reference(value: SSAValue) -> object:
        number = local.get(value)
        if number is not None:
            return number
        number = outer.get(value)
        if number is None:
            number = outer[value] = len(outer)
        return ("outer", number, value.type)

    def encode_block(block: Block) -> tuple[object, ...]:
        blocks.setdefault(block, len(blocks))
        for argument in block.args:
            local[argument] = len(local)
        return (
            tuple(argument.type for argument in block.args),
            tuple(encode(item) for item in block.ops),
        )

    def encode(item: Operation) -> tuple[object, ...]:
        operands = tuple(reference(value) for value in item.operands)
        regions = tuple(
            tuple(encode_block(block) for block in region.blocks)
            for region in item.regions
        )
        for result in item.results:
            local[result] = len(local)
        return (
            item.name,
            operands,
            tuple(result.type for result in item.results),
            tuple(sorted(item.attributes.items())),
            tuple(sorted(item.properties.items())),
            regions,
            tuple(
                blocks.setdefault(block, len(blocks))
                for block in item.successors
            ),
        )

    return encode(operation)


class AttributeDigests:
    """Stable content digests of attributes, memoized by object identity.

    A digest covers the attribute class, its IR name and its parameters or data. Python payloads are encoded by type: integers, strings and booleans by value, floats by their IEEE-754 bits, bytes by content, tuples and mappings element-wise, enumerations by class and member. Other payloads use the attribute's printed text, or raise :class:`ValueError` when ``strict`` is set. Digests do not depend on the process, so they may key persisted data.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        # Retaining each attribute keeps its identity from being reused.
        self.by_identity: dict[int, tuple[Attribute, str]] = {}
        self.kinds: dict[type, str] = {}

    def __call__(self, attribute: Attribute) -> str:
        found = self.by_identity.get(id(attribute))
        if found is not None:
            return found[1]
        cls = type(attribute)
        kind = self.kinds.get(cls)
        if kind is None:
            kind = self.kinds[cls] = (
                "parameters"
                if isinstance(attribute, ParametrizedAttribute)
                else "data"
                if isinstance(attribute, Data)
                else "text"
            )
        identity = (cls.__module__, cls.__qualname__, attribute.name)
        if kind == "parameters":
            content: object = (
                identity,
                tuple(
                    (name, self(getattr(attribute, name)))
                    for name, _ in cast(ParametrizedAttribute, attribute)
                    .get_irdl_definition()
                    .parameters
                ),
            )
        elif kind == "data":
            try:
                content = (
                    identity,
                    self._data(cast(Data[object], attribute).data),
                )
            except _UnsupportedPayload:
                if self.strict:
                    raise ValueError("unsupported attribute payload") from None
                content = (identity, "text", _printed(attribute))
        else:
            if self.strict:
                raise ValueError("unsupported attribute representation")
            content = (identity, "text", _printed(attribute))
        digest = hashlib.blake2b(
            repr(content).encode(), digest_size=16
        ).hexdigest()
        self.by_identity[id(attribute)] = (attribute, digest)
        return digest

    def _data(self, value: object) -> object:
        if isinstance(value, Attribute):
            return ("attribute", self(value))
        cls = type(value)
        if value is None or cls in (bool, int, str):
            return (cls.__name__, value)
        if cls is float:
            return ("float64-bits", struct.pack("!d", value).hex())
        if cls is bytes:
            return ("bytes", cast(bytes, value).hex())
        if isinstance(value, tuple):
            return ("tuple", tuple(self._data(item) for item in value))
        if isinstance(value, Mapping):
            return (
                "mapping",
                tuple(
                    (key, self._data(item))
                    for key, item in sorted(value.items())
                ),
            )
        if isinstance(value, Enum):
            return (
                "enum",
                cls.__module__,
                cls.__qualname__,
                value.name,
                self._data(value.value),
            )
        raise _UnsupportedPayload


class _UnsupportedPayload(Exception):
    pass


def _printed(attribute: Attribute) -> str:
    stream = StringIO()
    Printer(stream=stream).print_attribute(attribute)
    return stream.getvalue()


class _AttributeText:
    """Print each distinct attribute once; repeated objects are found by identity."""

    def __init__(self) -> None:
        self.by_identity: dict[int, tuple[Attribute, bytes]] = {}
        self.by_value: dict[Attribute, bytes] = {}

    def __call__(self, attribute: Attribute) -> bytes:
        found = self.by_identity.get(id(attribute))
        if found is not None:
            return found[1]
        text = self.by_value.get(attribute)
        if text is None:
            stream = StringIO()
            Printer(stream=stream).print_attribute(attribute)
            text = self.by_value[attribute] = stream.getvalue().encode()
        # Retaining the object keeps its identity from being reused.
        self.by_identity[id(attribute)] = (attribute, text)
        return text


def program_digest(module: Operation) -> str:
    """Return a digest of ``module`` that is stable across processes.

    The digest covers operation names, operand definitions numbered in definition order, result and block-argument types, attributes, properties, regions and successors. Attributes contribute their :class:`AttributeDigests` content digests. Equal digests identify Programs with equal structure and attribute content up to SSA names.
    """
    digests = AttributeDigests()

    def text(attribute: Attribute) -> bytes:
        return digests(attribute).encode()

    values: dict[SSAValue, int] = {}
    blocks: dict[Block, int] = {}
    digest = hashlib.blake2b(digest_size=16)
    update = digest.update

    def visit(operation: Operation) -> None:
        update(operation.name.encode())
        update(b"(")
        for operand in operation.operands:
            update(str(values.get(operand, -1)).encode())
            update(b",")
        update(b")")
        for name, attribute in sorted(operation.attributes.items()):
            update(name.encode())
            update(b"=")
            update(text(attribute))
        for name, attribute in sorted(operation.properties.items()):
            update(b"<" + name.encode() + b">")
            update(text(attribute))
        for result in operation.results:
            values[result] = len(values)
            update(b":")
            update(text(result.type))
        for successor in operation.successors:
            update(
                b"^" + str(blocks.setdefault(successor, len(blocks))).encode()
            )
        for region in operation.regions:
            update(b"{")
            for block in region.blocks:
                update(
                    b"^" + str(blocks.setdefault(block, len(blocks))).encode()
                )
                for argument in block.args:
                    values[argument] = len(values)
                    update(b"%")
                    update(text(argument.type))
                for item in block.ops:
                    visit(item)
            update(b"}")
        update(b";")

    visit(module)
    return digest.hexdigest()


__all__ = ["AttributeDigests", "program_digest", "structural_key"]
