"""Core references and open value roles shared by FHElium dialects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Dialect, Operation
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure

from .._operation_catalog import (
    OperationSpec,
    decode_json_attribute,
    flat_operation,
    literal_diagnostics,
    registered_operation_spec,
    required_string_attribute,
    unsupported_attributes,
)
from ._common import OpenStateType, ValueRole


@irdl_attr_definition
class EncryptedType(OpenStateType):
    """Encrypted value with partial scheme and representation state."""

    name = "fhelium.encrypted"
    ROLE: ClassVar[ValueRole] = "encrypted"


@irdl_attr_definition
class MessageType(OpenStateType):
    """Public value with optional frontend and tensor metadata."""

    name = "fhelium.message"
    ROLE: ClassVar[ValueRole] = "message"


@irdl_attr_definition
class PlaintextType(OpenStateType):
    """Encoded plaintext with partial representation state."""

    name = "fhelium.plaintext"
    ROLE: ClassVar[ValueRole] = "plaintext"


@irdl_attr_definition
class MaterialType(OpenStateType):
    """Symbolic reference type for graph-external material."""

    name = "fhelium.material"


@irdl_attr_definition
class ResourceType(OpenStateType):
    """Symbolic reference type for graph-external execution resources."""

    name = "fhelium.resource"


def _string(value: str | StringAttr | None) -> StringAttr | None:
    if value is None or isinstance(value, StringAttr):
        return value
    return StringAttr(value)


@irdl_op_definition
class MaterialRefOp(IRDLOperation):
    """Introduce graph-external material by symbolic identity."""

    name = "fhelium.material.ref"

    value = result_def()
    symbol = opt_attr_def(StringAttr)
    kind = opt_attr_def(StringAttr)

    def __init__(
        self,
        result_type: Attribute | None = None,
        *,
        symbol: str | StringAttr | None = None,
        kind: str | StringAttr | None = None,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["symbol"] = _string(symbol)
        attrs["kind"] = _string(kind)
        super().__init__(
            result_types=[result_type or MaterialType()], attributes=attrs
        )


@irdl_op_definition
class ResourceRefOp(IRDLOperation):
    """Introduce a graph-external execution resource by symbolic identity."""

    name = "fhelium.resource.ref"

    value = result_def()
    symbol = opt_attr_def(StringAttr)
    kind = opt_attr_def(StringAttr)

    def __init__(
        self,
        result_type: Attribute | None = None,
        *,
        symbol: str | StringAttr | None = None,
        kind: str | StringAttr | None = None,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["symbol"] = _string(symbol)
        attrs["kind"] = _string(kind)
        super().__init__(
            result_types=[result_type or ResourceType()], attributes=attrs
        )


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """Introduce one immutable scalar or structured literal descriptor."""

    name = "fhelium.constant"

    value = result_def()
    literal = opt_attr_def(StringAttr, attr_name="fhelium.literal")
    traits = traits_def(Pure())


def _reference_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(
        operation,
        ("symbol", "kind", "fhelium.material.descriptor"),
    )
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    if required_string_attribute(operation, "symbol") is None:
        diagnostics.append("requires nonempty string 'symbol' attribute")
    kind = operation.attributes.get("kind")
    if kind is not None and not isinstance(kind, StringAttr):
        diagnostics.append("optional 'kind' attribute must be a string")
    return tuple(diagnostics)


def _constant_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(operation, ("fhelium.literal",))
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    literal, error = decode_json_attribute(operation, "fhelium.literal")
    if error is not None:
        diagnostics.append(error)
    else:
        diagnostics.extend(literal_diagnostics(literal))
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = (
    registered_operation_spec(
        MaterialRefOp,
        "auxiliary",
        effect="opaque",
        validator=_reference_specification,
    ),
    registered_operation_spec(
        ResourceRefOp,
        "auxiliary",
        effect="opaque",
        validator=_reference_specification,
    ),
    registered_operation_spec(
        ConstantOp,
        "auxiliary",
        validator=_constant_specification,
    ),
)
"""Semantic specifications owned by the core FHElium dialect."""


FHElium = Dialect(
    "fhelium",
    [MaterialRefOp, ResourceRefOp, ConstantOp],
    [EncryptedType, MessageType, PlaintextType, MaterialType, ResourceType],
)
"""Core FHElium dialect for open values and external references."""


__all__ = [
    "ConstantOp",
    "EncryptedType",
    "FHElium",
    "MaterialRefOp",
    "MaterialType",
    "MessageType",
    "OPERATION_SPECS",
    "PlaintextType",
    "ResourceRefOp",
    "ResourceType",
]
