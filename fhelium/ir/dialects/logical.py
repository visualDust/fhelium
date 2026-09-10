"""Operand-role logical arithmetic between semantic and CKKS IR.

Cheon-Kim-Kim-Song (CKKS) intermediate representation (IR) state is assigned
after these operations record whether each value is encrypted or public.
"""

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
    r"""Return the logical pointwise sum of encrypted values.

    For logical values representing slot tensors $x$ and $y$, the result
    represents $z_i=x_i+y_i$.  This depth records operand roles only; CKKS
    depths, scales, residue rows, and polynomial representation are assigned by
    later passes.  ``LowerLogicalToCkksPass`` maps this operation to
    ``ckks.AddOp`` once both operands have usable CKKS state."""

    name = "fhelium_logical.add.encrypted_encrypted"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class AddEncryptedPublicOp(_BinaryLogicalOp):
    r"""Return the logical pointwise sum of encrypted $x$ and public $y$.

    The result represents $z_i=x_i+y_i$.  The public operand remains a message
    at this depth so later preparation can encode it at the encrypted operand's
    depth and scale before lowering to CKKS plaintext addition."""

    name = "fhelium_logical.add.encrypted_public"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(PublicType)
    result = result_def(EncryptedType)


@irdl_op_definition
class AddPublicEncryptedOp(_BinaryLogicalOp):
    r"""Return the logical pointwise sum of public $x$ and encrypted $y$.

    The result represents $z_i=x_i+y_i$.  Addition is commutative, so lowering
    prepares the public left operand as a CKKS plaintext and adds it to the
    encrypted right operand."""

    name = "fhelium_logical.add.public_encrypted"
    lhs = operand_def(PublicType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class SubtractEncryptedEncryptedOp(_BinaryLogicalOp):
    r"""Return the logical pointwise difference of encrypted values.

    For slot tensors $x$ and $y$, the result represents $z_i=x_i-y_i$.
    Later CKKS passes establish compatible depth, scale, basis, and residue state
    before lowering this operation to ``ckks.SubtractOp``."""

    name = "fhelium_logical.subtract.encrypted_encrypted"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class SubtractEncryptedPublicOp(_BinaryLogicalOp):
    r"""Subtract public $y$ pointwise from encrypted $x$.

    The result represents $z_i=x_i-y_i$.  Later passes encode and prepare the
    public right operand for the encrypted operand's CKKS state."""

    name = "fhelium_logical.subtract.encrypted_public"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(PublicType)
    result = result_def(EncryptedType)


@irdl_op_definition
class SubtractPublicEncryptedOp(_BinaryLogicalOp):
    r"""Subtract encrypted $y$ pointwise from public $x$.

    The result represents $z_i=x_i-y_i$.  The operand order is significant:
    lowering must form the public-minus-encrypted result rather than reuse the
    encrypted-minus-public path."""

    name = "fhelium_logical.subtract.public_encrypted"
    lhs = operand_def(PublicType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class MultiplyEncryptedEncryptedOp(_BinaryLogicalOp):
    r"""Return the logical pointwise product of encrypted values.

    For slot tensors $x$ and $y$, the result represents $z_i=x_i y_i$.
    Later passes choose CKKS depths and scales, move polynomial payloads to the
    number-theoretic-transform representation, and lower to ``ckks.MultiplyOp``."""

    name = "fhelium_logical.multiply.encrypted_encrypted"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class MultiplyEncryptedPublicOp(_BinaryLogicalOp):
    r"""Multiply encrypted $x$ pointwise by public $y$.

    The result represents $z_i=x_i y_i$.  The public operand is still a message
    here; CKKS preparation later encodes it at a selected scale and converts it to
    an operation-ready plaintext."""

    name = "fhelium_logical.multiply.encrypted_public"
    lhs = operand_def(EncryptedType)
    rhs = operand_def(PublicType)
    result = result_def(EncryptedType)


@irdl_op_definition
class MultiplyPublicEncryptedOp(_BinaryLogicalOp):
    r"""Multiply public $x$ pointwise by encrypted $y$.

    The result represents $z_i=x_i y_i$.  Multiplication is commutative, so the
    public left operand may be prepared as the plaintext operand of the eventual
    CKKS multiplication."""

    name = "fhelium_logical.multiply.public_encrypted"
    lhs = operand_def(PublicType)
    rhs = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class NegateEncryptedOp(_UnaryLogicalOp):
    r"""Return the logical pointwise additive inverse of an encrypted value.

    For a slot tensor $x$, the result represents $z_i=-x_i$.  CKKS
    representation state remains open until lowering selects ``ckks.NegateOp``."""

    name = "fhelium_logical.negate.encrypted"
    value = operand_def(EncryptedType)
    result = result_def(EncryptedType)


@irdl_op_definition
class RollEncryptedOp(_UnaryLogicalOp):
    r"""Apply a cyclic displacement to the encrypted slot axis.

    For $S$ slots and the inherited integer ``shift`` attribute $r$, the
    result represents ``torch.roll(x, r)`` on the selected logical dimension.
    CKKS lowering normalizes $r$ modulo $S$, resolves the corresponding
    Galois automorphism and rotation key, and replaces this operation with
    ``ckks.RotateOp``.  A zero normalized displacement is the identity."""

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
