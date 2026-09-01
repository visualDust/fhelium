"""Provider-neutral tensor semantics for encrypted and public values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyOf,
    BaseAttr,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure

from .._operation_catalog import (
    OperationSpec,
    flat_operation,
    flat_without_attributes,
    registered_operation_spec,
    unsupported_attributes,
)
from ._common import OpenStateType, ValueRole


@irdl_attr_definition
class SecretType(OpenStateType):
    """Secret semantic tensor before selection of an encryption scheme state."""

    name = "fhelium_semantic.secret"
    ROLE: ClassVar[ValueRole] = "encrypted"


@irdl_attr_definition
class PublicType(OpenStateType):
    """Public semantic tensor before encoding or specialization."""

    name = "fhelium_semantic.public"
    ROLE: ClassVar[ValueRole] = "message"


_SEMANTIC_VALUE = AnyOf((BaseAttr(SecretType), BaseAttr(PublicType)))


class _BinarySemanticOp(IRDLOperation):
    lhs = operand_def(_SEMANTIC_VALUE)
    rhs = operand_def(_SEMANTIC_VALUE)
    result = result_def(_SEMANTIC_VALUE)
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


class _UnarySemanticOp(IRDLOperation):
    value = operand_def(_SEMANTIC_VALUE)
    result = result_def(_SEMANTIC_VALUE)
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
class AddOp(_BinarySemanticOp):
    """Add two semantic values pointwise."""

    name = "fhelium_semantic.add"


@irdl_op_definition
class SubtractOp(_BinarySemanticOp):
    """Subtract the right semantic value pointwise from the left."""

    name = "fhelium_semantic.subtract"


@irdl_op_definition
class MultiplyOp(_BinarySemanticOp):
    """Multiply two semantic values pointwise."""

    name = "fhelium_semantic.multiply"


@irdl_op_definition
class NegateOp(_UnarySemanticOp):
    """Negate one semantic value pointwise."""

    name = "fhelium_semantic.negate"


@irdl_op_definition
class RollOp(_UnarySemanticOp):
    """Roll one semantic tensor along a statically identified dimension."""

    name = "fhelium_semantic.roll"

    shift = opt_attr_def(IntegerAttr)
    dimension = opt_attr_def(IntegerAttr)


def _roll_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(
        operation,
        ("fhelium.frontend.target", "shift", "dimension"),
    )
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    if not isinstance(operation.attributes.get("shift"), IntegerAttr):
        diagnostics.append("roll requires integer 'shift' attribute")
    dimension = operation.attributes.get("dimension")
    if dimension is not None and not isinstance(dimension, IntegerAttr):
        diagnostics.append("roll 'dimension' must be integer when present")
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = tuple(
    registered_operation_spec(
        operation_type,
        "pointwise",
        validator=flat_without_attributes,
    )
    for operation_type in (AddOp, SubtractOp, MultiplyOp, NegateOp)
) + (
    registered_operation_spec(
        RollOp,
        "pointwise",
        validator=_roll_specification,
    ),
)
"""Semantic specifications owned by the provider-neutral tensor dialect."""


FHEliumSemantic = Dialect(
    "fhelium_semantic",
    [AddOp, SubtractOp, MultiplyOp, NegateOp, RollOp],
    [SecretType, PublicType],
)
"""Semantic FHElium dialect independent of CKKS and execution providers."""


__all__ = [
    "AddOp",
    "FHEliumSemantic",
    "MultiplyOp",
    "NegateOp",
    "OPERATION_SPECS",
    "PublicType",
    "RollOp",
    "SecretType",
    "SubtractOp",
]
