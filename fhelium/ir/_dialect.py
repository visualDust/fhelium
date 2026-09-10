"""Permissive context and value helpers for multi-depth FHElium IR.

First-party operations and types live in distinct registered dialect modules.
The context loads all registered first-party dialects while preserving unknown operations and
types for mixed-level research and external extensions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
from xdsl.context import Context as XdslContext
from xdsl.dialects.builtin import (
    Builtin,
    DictionaryAttr,
    LocationAttr,
    StringAttr,
)
from xdsl.dialects.arith import Arith
from xdsl.dialects.func import Func
from xdsl.dialects.scf import Scf
from xdsl.ir import Attribute, Operation, SSAValue

from .dialects import REGISTERED_DIALECTS
from .dialects._common import OpenStateType, ValueRole, dictionary_state
from .dialects.core import (
    EncryptedType,
    FHElium,
    MaterialRefOp,
    MaterialType,
    MessageType,
    PlaintextType,
    ResourceRefOp,
    ResourceType,
)

SCHEMA_VERSION = "1"
DIALECT_VERSION = "0.2"
SCHEMA_VERSION_ATTRIBUTE = "fhelium.schema_version"
DIALECT_VERSION_ATTRIBUTE = "fhelium.dialect_version"


def create_dialect_context() -> XdslContext:
    """Create an xDSL context containing every registered IR dialect.

    Structural registration and preservation are independent of lowering
    readiness. Unknown application and vendor vocabulary remains legal and can
    coexist with operations from any registered abstraction level.
    """

    context = XdslContext(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    context.load_dialect(Arith)
    context.load_dialect(Scf)
    for dialect in REGISTERED_DIALECTS:
        context.load_dialect(dialect)
    return context


def value_type(
    role: ValueRole,
    state: DictionaryAttr | Mapping[str, Attribute] | None = None,
) -> Attribute:
    """Construct the preserved open core type for a known value role."""

    if role == "encrypted":
        return EncryptedType.new((dictionary_state(state),))
    if role == "plaintext":
        return PlaintextType.new((dictionary_state(state),))
    if role in {"message", "static"}:
        role_state = (
            dict(state.data)
            if isinstance(state, DictionaryAttr)
            else dict(state or {})
        )
        role_state.setdefault("role", StringAttr(role))
        return MessageType.new((dictionary_state(role_state),))
    raise ValueError(f"Unsupported value role: {role!r}")


def value_role(value_or_type: SSAValue | Attribute) -> ValueRole | None:
    """Return a registered value role, or ``None`` for non-value types."""

    attribute = (
        value_or_type.type
        if isinstance(value_or_type, SSAValue)
        else value_or_type
    )
    if not isinstance(attribute, OpenStateType):
        return None
    represented_role = attribute.state.data.get("role")
    if represented_role is not None:
        if not isinstance(represented_role, StringAttr):
            raise ValueError(
                f"{attribute.name} state 'role' must be a StringAttr"
            )
        if represented_role.data not in {
            "encrypted",
            "message",
            "plaintext",
            "static",
        }:
            raise ValueError(
                f"{attribute.name} state has unsupported role "
                f"{represented_role.data!r}"
            )
        return cast(ValueRole, represented_role.data)
    role = getattr(type(attribute), "ROLE", None)
    if role in {"encrypted", "message", "plaintext", "static"}:
        return role
    return None


def operation_name(operation: Operation) -> str:
    """Return the textual name of a registered or unknown operation."""

    op_name = getattr(operation, "op_name", None)
    if isinstance(op_name, StringAttr):
        return op_name.data
    return operation.name


def create_operation(
    context: XdslContext,
    name: str,
    *,
    operands: Sequence[SSAValue] = (),
    result_types: Sequence[Attribute] = (),
    attributes: Mapping[str, Attribute] | None = None,
    properties: Mapping[str, Attribute] | None = None,
    location: LocationAttr | None = None,
) -> Operation:
    """Construct a registered or preserved unknown operation by textual name."""

    if not isinstance(context, XdslContext):
        raise TypeError("dialect_context must be an xDSL Context")
    if not isinstance(name, str):
        raise TypeError("operation name must be a string")
    if not name.strip():
        raise ValueError("operation name must be non-empty")
    operation_type = context.get_op(name)
    return operation_type.create(
        operands=list(operands),
        result_types=list(result_types),
        attributes=dict(attributes or {}),
        properties=dict(properties or {}),
        location=location,
    )


__all__ = [
    "DIALECT_VERSION",
    "DIALECT_VERSION_ATTRIBUTE",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_ATTRIBUTE",
    "EncryptedType",
    "FHElium",
    "MaterialRefOp",
    "MaterialType",
    "MessageType",
    "PlaintextType",
    "ResourceRefOp",
    "ResourceType",
    "ValueRole",
    "create_dialect_context",
    "create_operation",
    "operation_name",
    "value_role",
    "value_type",
]
