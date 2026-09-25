r"""Logical number-theoretic-transform operations used below CKKS lowering.

The number-theoretic transform (NTT) evaluates a polynomial in
$\mathbb{Z}_{q_i}[X]/(X^N+1)$ at roots of unity modulo each active prime
$q_i$, turning negacyclic multiplication into pointwise multiplication.
Residue number system (RNS) rows hold those per-prime polynomials.  Montgomery
representation stores a residue $x_i$ as
$x_iR_i\bmod q_i$ for radix $R_i$.

The four operations distinguish the complete source and destination residue
representations needed by ciphertext and plaintext transitions. The parameter operand carries the RNS arithmetic parameters. Additional
Tensor operands carry transform tables after their layout is selected. An
operation can leave its algorithm and execution implementation unassigned;
selection does not change its residue transitions or frequency ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import Pure

from .._operation_catalog import (
    OperationSpec,
    flat_with_attributes,
    registered_operation_spec,
)
from .rns import RnsBundleType
from .._dependencies import OperationDependencies, operand_relations


class _NttOp(IRDLOperation):
    value = operand_def(RnsBundleType)
    parameters = operand_def()
    tables = var_operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        tables: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[value, parameters, tables],
            result_types=[result_type],
            attributes=attributes,
        )


@irdl_op_definition
class CoefficientStandardToNttMontgomeryOp(_NttOp):
    r"""Compute the forward negacyclic NTT from standard residues.

    For every active prime $q_i$, the input row contains coefficient residues
    $a_j \bmod q_i$ for a polynomial in
    $R_{q_i}=\mathbb{Z}_{q_i}[X]/(X^N+1)$.  The result contains its
    number-theoretic transform $\operatorname{NTT}_{q_i}(a)$, multiplied by the
    Montgomery radix $R_i$ modulo $q_i$.  Prime rows, CKKS depth, component
    axes, and scale do not change."""

    name = "fhelium_ntt.coefficient_standard_to_ntt_montgomery"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class CoefficientMontgomeryToNttMontgomeryOp(_NttOp):
    r"""Compute the forward negacyclic NTT of Montgomery coefficients.

    Each input coefficient row represents $a_j R_i \bmod q_i$.  The transform
    is applied independently for every active prime and returns
    $\operatorname{NTT}_{q_i}(a)R_i \bmod q_i$.  The operation preserves prime
    rows, depth, component axes, and CKKS scale."""

    name = "fhelium_ntt.coefficient_montgomery_to_ntt_montgomery"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class NttMontgomeryToCoefficientStandardOp(_NttOp):
    r"""Compute the inverse negacyclic NTT and leave standard residues.

    For each active prime $q_i$, an NTT/Montgomery row
    $\operatorname{NTT}_{q_i}(a)R_i$ is mapped to coefficient residues
    $a_j \bmod q_i$.  The inverse transform includes the Montgomery and
    transform normalization factors.  Prime rows, depth, components, and scale
    are preserved."""

    name = "fhelium_ntt.ntt_montgomery_to_coefficient_standard"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class NttMontgomeryToCoefficientMontgomeryOp(_NttOp):
    r"""Compute the inverse negacyclic NTT and retain Montgomery residues.

    For every active prime $q_i$, the input
    $\operatorname{NTT}_{q_i}(a)R_i$ becomes coefficient data
    $a_jR_i \bmod q_i$.  The prime-row set, depth, component axes, and CKKS
    scale remain unchanged."""

    name = "fhelium_ntt.ntt_montgomery_to_coefficient_montgomery"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


_NTT_OPERATION_TYPES = (
    CoefficientStandardToNttMontgomeryOp,
    CoefficientMontgomeryToNttMontgomeryOp,
    NttMontgomeryToCoefficientStandardOp,
    NttMontgomeryToCoefficientMontgomeryOp,
)

OPERATION_SPECS: tuple[OperationSpec, ...] = tuple(
    registered_operation_spec(
        operation_type,
        "ntt",
        validator=flat_with_attributes(
            "ntt_backend",
            "ntt_algorithm",
            "ntt_group_width",
            "ntt_radix",
            "ntt_table_layout",
            "ckks_config",
        ),
    )
    for operation_type in _NTT_OPERATION_TYPES
)
"""Semantic specifications owned by the NTT dialect."""


FHEliumNtt = Dialect(
    "fhelium_ntt",
    [
        *_NTT_OPERATION_TYPES,
    ],
    [],
)
"""Logical NTT operations with Tensor parameter operands."""


__all__ = [
    "FHEliumNtt",
    "CoefficientMontgomeryToNttMontgomeryOp",
    "CoefficientStandardToNttMontgomeryOp",
    "NttMontgomeryToCoefficientMontgomeryOp",
    "NttMontgomeryToCoefficientStandardOp",
    "OPERATION_SPECS",
]
