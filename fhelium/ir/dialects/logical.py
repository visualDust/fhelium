"""Operand-role logical arithmetic between semantic and CKKS IR."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure

from .._operation_catalog import (
    OperationSpec,
    registered_operation_spec,
)
from ._common import OpenStateType, ValueRole


@irdl_attr_definition
class EncryptedType(OpenStateType):
    """Logical encrypted value before CKKS representation planning."""

    name = "fhelium_logical.encrypted"
    ROLE: ClassVar[ValueRole] = "encrypted"


@irdl_attr_definition
class PublicType(OpenStateType):
    """Logical public value classified for an encrypted consumer."""

    name = "fhelium_logical.public"
    ROLE: ClassVar[ValueRole] = "message"


class _BinaryLogicalOp(IRDLOperation):
    lhs = operand_def()
    rhs = operand_def()
    result = result_def()
    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type or SSAValue.get(lhs).type],
            attributes=attributes,
        )


class _UnaryLogicalOp(IRDLOperation):
    value = operand_def()
    result = result_def()
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type or SSAValue.get(value).type],
            attributes=attributes,
        )


@irdl_op_definition
class AddEncryptedEncryptedOp(_BinaryLogicalOp):
    """Add two logically encrypted values."""

    name = "fhelium_logical.add.encrypted_encrypted"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class AddEncryptedPublicOp(_BinaryLogicalOp):
    """Add a public right operand to an encrypted left operand."""

    name = "fhelium_logical.add.encrypted_public"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(PublicType)
    result = result_def(EncryptedType)


@irdl_op_definition
class AddPublicEncryptedOp(_BinaryLogicalOp):
    """Add an encrypted right operand to a public left operand."""

    name = "fhelium_logical.add.public_encrypted"
    lhs = operand_def(PublicType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class SubtractEncryptedEncryptedOp(_BinaryLogicalOp):
    """Subtract one encrypted value from another."""

    name = "fhelium_logical.subtract.encrypted_encrypted"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class SubtractEncryptedPublicOp(_BinaryLogicalOp):
    """Subtract a public right operand from an encrypted left operand."""

    name = "fhelium_logical.subtract.encrypted_public"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(PublicType)
    result = result_def(EncryptedType)


@irdl_op_definition
class SubtractPublicEncryptedOp(_BinaryLogicalOp):
    """Subtract an encrypted right operand from a public left operand."""

    name = "fhelium_logical.subtract.public_encrypted"
    lhs = operand_def(PublicType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class MultiplyEncryptedEncryptedOp(_BinaryLogicalOp):
    """Multiply two logically encrypted values."""

    name = "fhelium_logical.multiply.encrypted_encrypted"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class MultiplyEncryptedPublicOp(_BinaryLogicalOp):
    """Multiply an encrypted left operand by a public right operand."""

    name = "fhelium_logical.multiply.encrypted_public"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(PublicType)
    result = result_def(EncryptedType)


@irdl_op_definition
class MultiplyPublicEncryptedOp(_BinaryLogicalOp):
    """Multiply a public left operand by an encrypted right operand."""

    name = "fhelium_logical.multiply.public_encrypted"
    lhs = operand_def(PublicType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class NegateEncryptedOp(_UnaryLogicalOp):
    """Negate one logically encrypted value."""

    name = "fhelium_logical.negate.encrypted"
    value = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class RollEncryptedOp(_UnaryLogicalOp):
    """Roll one logically encrypted value."""

    name = "fhelium_logical.roll.encrypted"
    value = operand_def(EncryptedType)
    result = result_def(EncryptedType)


_LOGICAL_OPERATION_TYPES = (
    AddEncryptedEncryptedOp,
    AddEncryptedPublicOp,
    AddPublicEncryptedOp,
    SubtractEncryptedEncryptedOp,
    SubtractEncryptedPublicOp,
    SubtractPublicEncryptedOp,
    MultiplyEncryptedEncryptedOp,
    MultiplyEncryptedPublicOp,
    MultiplyPublicEncryptedOp,
    NegateEncryptedOp,
    RollEncryptedOp,
)

OPERATION_SPECS: tuple[OperationSpec, ...] = tuple(
    registered_operation_spec(operation_type, "logical")
    for operation_type in _LOGICAL_OPERATION_TYPES
)
"""Semantic specifications owned by the operand-role logical dialect."""


FHEliumLogical = Dialect(
    "fhelium_logical",
    [
        *_LOGICAL_OPERATION_TYPES,
    ],
    [EncryptedType, PublicType],
)
"""Operand-role logical FHElium dialect."""


__all__ = [
    "AddEncryptedEncryptedOp",
    "AddEncryptedPublicOp",
    "AddPublicEncryptedOp",
    "EncryptedType",
    "FHEliumLogical",
    "MultiplyEncryptedEncryptedOp",
    "MultiplyEncryptedPublicOp",
    "MultiplyPublicEncryptedOp",
    "NegateEncryptedOp",
    "OPERATION_SPECS",
    "PublicType",
    "RollEncryptedOp",
    "SubtractEncryptedEncryptedOp",
    "SubtractEncryptedPublicOp",
    "SubtractPublicEncryptedOp",
]
