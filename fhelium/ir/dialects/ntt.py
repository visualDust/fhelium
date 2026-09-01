"""Logical number-theoretic-transform operations used below CKKS lowering.

The four operations distinguish the complete source and destination residue
representations needed by ciphertext and plaintext transitions. Their plan
operand carries context-specialized NTT resources; concrete radix, device, and
kernel choices remain backend implementation details.
"""

from __future__ import annotations

from collections.abc import Mapping

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
    flat_without_attributes,
    registered_operation_spec,
)
from ._common import OpenStateType
from .rns import RnsBundleType


@irdl_attr_definition
class NttPlanType(OpenStateType):
    """CKKS-parameter-specific NTT tables and their logical row mapping."""

    name = "fhelium_ntt.plan"


class _NttOp(IRDLOperation):
    value = operand_def(RnsBundleType)
    plan = operand_def(NttPlanType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        plan: SSAValue | Operation,
        result_type: Attribute,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[value, plan],
            result_types=[result_type],
            attributes=attributes,
        )


@irdl_op_definition
class CoefficientStandardToNttMontgomeryOp(_NttOp):
    """Map coefficient/standard residues to NTT/Montgomery residues."""

    name = "fhelium_ntt.coefficient_standard_to_ntt_montgomery"


@irdl_op_definition
class CoefficientMontgomeryToNttMontgomeryOp(_NttOp):
    """Map coefficient/Montgomery residues to NTT/Montgomery residues."""

    name = "fhelium_ntt.coefficient_montgomery_to_ntt_montgomery"


@irdl_op_definition
class NttMontgomeryToCoefficientStandardOp(_NttOp):
    """Map NTT/Montgomery residues to coefficient/standard residues."""

    name = "fhelium_ntt.ntt_montgomery_to_coefficient_standard"


@irdl_op_definition
class InverseMontgomeryOp(_NttOp):
    """Map NTT/Montgomery residues to coefficient/Montgomery residues."""

    name = "fhelium_ntt.inverse_montgomery"


_NTT_OPERATION_TYPES = (
    CoefficientStandardToNttMontgomeryOp,
    CoefficientMontgomeryToNttMontgomeryOp,
    NttMontgomeryToCoefficientStandardOp,
    InverseMontgomeryOp,
)

OPERATION_SPECS: tuple[OperationSpec, ...] = tuple(
    registered_operation_spec(
        operation_type,
        "ntt",
        validator=flat_without_attributes,
    )
    for operation_type in _NTT_OPERATION_TYPES
)
"""Semantic specifications owned by the NTT dialect."""


FHEliumNtt = Dialect(
    "fhelium_ntt",
    [
        *_NTT_OPERATION_TYPES,
    ],
    [NttPlanType],
)
"""Logical NTT operations and plan type."""


__all__ = [
    "FHEliumNtt",
    "CoefficientMontgomeryToNttMontgomeryOp",
    "CoefficientStandardToNttMontgomeryOp",
    "InverseMontgomeryOp",
    "NttMontgomeryToCoefficientStandardOp",
    "NttPlanType",
    "OPERATION_SPECS",
]
