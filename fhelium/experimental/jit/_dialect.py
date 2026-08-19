"""Open xDSL vocabulary for canonical mixed-dialect JIT Programs.

The registered FHElium types carry partial structural metadata. The permissive
IR context represents unknown dialects, operations, attributes, and types as
unregistered xDSL objects, preserving structurally valid extension IR during
parse, print, clone, and targeted rewrites. CKKS numerical state, material
availability, target support, and executable schemas are evaluated by analyses
and readiness gates rather than IRDL construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from xdsl.context import Context as IRContext
from xdsl.dialects.builtin import (
    Builtin,
    DictionaryAttr,
    LocationAttr,
    StringAttr,
)
from xdsl.dialects.func import Func
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    param_def,
    result_def,
)

SCHEMA_VERSION = "1"
DIALECT_VERSION = "0.1"
SCHEMA_VERSION_ATTRIBUTE = "fhelium.schema_version"
DIALECT_VERSION_ATTRIBUTE = "fhelium.dialect_version"


def _dictionary(
    state: DictionaryAttr | Mapping[str, Attribute] | None,
) -> DictionaryAttr:
    if state is None:
        return DictionaryAttr({})
    if isinstance(state, DictionaryAttr):
        return state
    return DictionaryAttr(state)


class _OpenStateType(ParametrizedAttribute, TypeAttribute):
    """Base for a value type carrying open-ended structural state metadata."""

    state: DictionaryAttr = cast(DictionaryAttr, param_def())

    def __init__(
        self,
        state: DictionaryAttr | Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(_dictionary(state))


@irdl_attr_definition
class EncryptedType(_OpenStateType):
    """Encrypted value with partial or unknown scheme/representation state."""

    name = "fhelium.encrypted"


@irdl_attr_definition
class MessageType(_OpenStateType):
    """Public value with optional tensor shape, dtype, and frontend metadata."""

    name = "fhelium.message"


@irdl_attr_definition
class PlaintextType(_OpenStateType):
    """Encoded plaintext with partial or unknown representation state."""

    name = "fhelium.plaintext"


@irdl_attr_definition
class MaterialType(_OpenStateType):
    """Type a symbolic reference to a graph-external material value."""

    name = "fhelium.material"


@irdl_attr_definition
class ResourceType(_OpenStateType):
    """Type a symbolic reference to a graph-external execution resource."""

    name = "fhelium.resource"


def _string(value: str | StringAttr | None) -> StringAttr | None:
    if value is None or isinstance(value, StringAttr):
        return value
    return StringAttr(value)


@irdl_op_definition
class MaterialRefOp(IRDLOperation):
    """Introduce one graph-external material through a symbolic identifier."""

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
    """Introduce one graph-external execution resource symbolically."""

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


FHElium = Dialect(
    "fhelium",
    [MaterialRefOp, ResourceRefOp],
    [EncryptedType, MessageType, PlaintextType, MaterialType, ResourceType],
)
"""Registered structural vocabulary for FHElium values and references."""


def create_ir_context() -> IRContext:
    """Create a permissive parser/rewrite context for mixed-dialect programs.

    Builtin, func, and the small FHElium structural vocabulary are registered.
    Every other dialect, operation, attribute, and type is represented as an
    unregistered xDSL object, including Torch and application extensions. This
    is the structural import representation; execution support is decided for one
    selected entry by bound handlers and readiness validation.
    """

    ir_context = IRContext(allow_unregistered=True)
    ir_context.load_dialect(Builtin)
    ir_context.load_dialect(Func)
    ir_context.load_dialect(FHElium)
    return ir_context


def value_type(
    role: str,
    state: DictionaryAttr | Mapping[str, Attribute] | None = None,
) -> Attribute:
    """Construct the open structural type for a JIT capture value role.

    ``encrypted``, ``message``, ``plaintext``, and ``static`` define capture
    and entry-binding semantics. The mixed-dialect representation also accepts
    extension types, which the permissive parser preserves independently of
    this frontend role vocabulary.
    """

    if role == "encrypted":
        return EncryptedType.new((_dictionary(state),))
    if role == "plaintext":
        return PlaintextType.new((_dictionary(state),))
    if role in {"message", "static"}:
        role_state = (
            dict(state.data)
            if isinstance(state, DictionaryAttr)
            else dict(state or {})
        )
        role_state.setdefault("role", StringAttr(role))
        return MessageType.new((_dictionary(role_state),))
    raise ValueError(f"Unsupported capture value role: {role!r}")


def value_role(value_or_type: SSAValue | Attribute) -> str | None:
    """Return a known JIT capture role, or ``None`` for an extension type."""

    attribute = (
        value_or_type.type
        if isinstance(value_or_type, SSAValue)
        else value_or_type
    )
    if isinstance(attribute, EncryptedType):
        return "encrypted"
    if isinstance(attribute, PlaintextType):
        return "plaintext"
    if isinstance(attribute, MessageType):
        role = attribute.state.data.get("role")
        if isinstance(role, StringAttr) and role.data == "static":
            return "static"
        return "message"
    return None


def operation_name(operation: Operation) -> str:
    """Return the textual operation name for registered or unknown ops."""

    op_name = getattr(operation, "op_name", None)
    if isinstance(op_name, StringAttr):
        return op_name.data
    return operation.name


def create_operation(
    ir_context: IRContext,
    name: str,
    *,
    operands: Sequence[SSAValue] = (),
    result_types: Sequence[Attribute] = (),
    attributes: Mapping[str, Attribute] | None = None,
    properties: Mapping[str, Attribute] | None = None,
    location: LocationAttr | None = None,
) -> Operation:
    """Create a registered or unregistered operation by stable textual name."""

    if not isinstance(ir_context, IRContext):
        raise TypeError("ir_context must be an xDSL IRContext")
    if not isinstance(name, str):
        raise TypeError("operation name must be a string")
    if not name.strip():
        raise ValueError("operation name must be non-empty")
    operation_type = ir_context.get_op(name)
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
    "create_ir_context",
    "create_operation",
    "operation_name",
    "value_role",
    "value_type",
]
