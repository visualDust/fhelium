r"""Logical residue-number-system operations used below CKKS lowering.

A residue number system (RNS) stores a polynomial row modulo each active
ciphertext prime in Q.  The QP basis appends special P primes used by hybrid
key switching.  A hybrid digit is one selected group of Q rows lifted together
during that procedure.  Montgomery representation stores residue
$x_i$ as $x_iR_i\bmod q_i$; number-theoretic-transform (NTT) form
stores the negacyclic transform of each polynomial row.

Each operation defines one complete transformation of an RNS bundle. The
operations contain no CPU, CUDA, Triton, kernel, radix, or fusion selection.
Their resource operands identify context-specialized parameter and rescale
assets that an eager or JIT executable binds before running a backend.
"""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.dialects.builtin import (
    Float64Type,
    FloatAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from .._operation_catalog import (
    OperationSpec,
    flat_with_attributes,
    flat_without_attributes,
    registered_operation_spec,
)
from ._common import OpenStateType


@irdl_attr_definition
class RnsBundleType(OpenStateType):
    """RNS components with partial arithmetic, layout, and placement state."""

    name = "fhelium_rns.bundle"


@irdl_attr_definition
class RnsParametersType(OpenStateType):
    """CKKS-parameter-specific RNS moduli and Montgomery parameters."""

    name = "fhelium_rns.parameters"


@irdl_attr_definition
class RescalePlanType(OpenStateType):
    """Level-specific dropped-prime, inverse, and surviving-row resources."""

    name = "fhelium_rns.rescale_plan"


@irdl_attr_definition
class KeySwitchPlanType(OpenStateType):
    """Level-specialized hybrid decomposition and ModUp/ModDown resources."""

    name = "fhelium_rns.key_switch_plan"


@irdl_attr_definition
class EvaluationKeyResourceType(OpenStateType):
    """Adapter-validated evaluation-key material used by logical RNS ops."""

    name = "fhelium_rns.evaluation_key_resource"


@irdl_op_definition
class AddStandardOp(IRDLOperation):
    r"""Add two standard-residue polynomial bundles row by row.

    An RNS (residue number system) row stores one polynomial modulo an active
    prime $q_i$.  For every component, coefficient, and row, this operation
    computes $z_i=(x_i+y_i)\bmod q_i$.  Both operands use the same Q or QP
    prime rows and store values in the standard range $[0,q_i)$.  Component
    shape, level, basis, polynomial domain, Montgomery factor, and scale are
    unchanged."""

    name = "fhelium_rns.add_standard"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
            attributes=attributes,
        )


@irdl_op_definition
class SubtractStandardOp(IRDLOperation):
    r"""Subtract standard-residue polynomial bundles row by row.

    For each active prime $q_i$, component, and coefficient, the result is
    $z_i=(x_i-y_i)\bmod q_i$.  The operation preserves component shape, prime
    rows, level, modulus basis, polynomial domain, and scale."""

    name = "fhelium_rns.subtract_standard"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class NegateStandardOp(IRDLOperation):
    r"""Negate a standard-residue polynomial bundle row by row.

    For every active prime $q_i$, component, and coefficient, the result is
    $z_i=(-x_i)\bmod q_i$.  Component shape and all represented CKKS state are
    preserved."""

    name = "fhelium_rns.negate_standard"
    value = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
        )


class _CiphertextPlaintextOp(IRDLOperation):
    ciphertext = operand_def(RnsBundleType)
    plaintext = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        plaintext: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[ciphertext, plaintext, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class AddPlaintextOp(_CiphertextPlaintextOp):
    r"""Add a prepared plaintext polynomial to ciphertext component zero.

    For ciphertext $(c_0,\ldots,c_{k-1})$ and plaintext polynomial $p$, each
    active-prime row computes $c'_0=c_0+p\pmod {q_i}$ and
    $c'_j=c_j$ for $j>0$.  The prepared plaintext uses coefficient-domain
    Montgomery residues in the same Q or QP rows; Montgomery reduction produces
    the standard-residue component.  Level, component count, and ciphertext scale
    are preserved."""

    name = "fhelium_rns.add_plaintext"


@irdl_op_definition
class MultiplyPlaintextOp(_CiphertextPlaintextOp):
    r"""Multiply every ciphertext component by a prepared plaintext.

    The ciphertext and plaintext are NTT-domain Montgomery bundles on matching
    prime rows.  For each component $j$, prime $q_i$, and transform index
    $k$, Montgomery multiplication computes
    $c'_{j,i,k}=c_{j,i,k}p_{i,k}R_i^{-1}\bmod q_i$.  Thus the polynomial
    meaning is $c'_j=c_jp$ in $R_{q_i}$.  Component count and level remain;
    the CKKS result scale is the product of operand scales."""

    name = "fhelium_rns.multiply_plaintext"


@irdl_op_definition
class RescaleDropLeadingPrimeOp(IRDLOperation):
    r"""Divide, round, and remove the leading active Q prime.

    Let $q_d$ be the first active Q prime, $d=x\bmod q_d$ in
    $[0,q_d)$, and $x_i=x\bmod q_i$ for a surviving prime.  Truncating
    quotient uses $t=d$; nearest quotient uses $t=d-q_d$ when
    $d>q_d/2$, otherwise $t=d$.  Each surviving row is
    $y_i=(x_i-t)q_d^{-1}\bmod q_i$, representing the selected rounded
    quotient of $x/q_d$.  The $q_d$ row is removed, the public level advances
    by one, and actual scale $\Delta$ becomes $\Delta/q_d$.  Component count
    and coefficient/standard representation are preserved."""

    name = "fhelium_rns.rescale_drop_leading_prime"
    value = operand_def(RnsBundleType)
    plan = operand_def(RescalePlanType)
    result = result_def(RnsBundleType)
    rounding = opt_attr_def(StringAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        plan: SSAValue | Operation,
        result_type: Attribute,
        *,
        rounding: str | StringAttr = "nearest",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["rounding"] = (
            StringAttr(rounding) if isinstance(rounding, str) else rounding
        )
        super().__init__(
            operands=[value, plan],
            result_types=[result_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        """Accept only the represented CKKS quotient-rounding rules."""

        if self.rounding is not None and self.rounding.data not in {
            "nearest",
            "floor",
        }:
            raise VerifyException(
                "RNS rescale rounding must be 'nearest' or 'floor'"
            )


@irdl_op_definition
class ExtractComponentOp(IRDLOperation):
    r"""Select one ciphertext-component polynomial without arithmetic.

    For an RNS bundle $(c_0,\ldots,c_{k-1})$, ``component=j`` returns $c_j$
    with the same coefficient values, active prime rows, polynomial domain,
    Montgomery state, level, and scale."""

    name = "fhelium_rns.extract_component"
    value = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    component = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute,
        *,
        component: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            attributes={
                "component": (
                    IntegerAttr(component, 64)
                    if isinstance(component, int)
                    else component
                )
            },
        )

    def verify_(self) -> None:
        if int(self.component.value.data) < 0:
            raise VerifyException("RNS component index must be nonnegative")


@irdl_op_definition
class PackTwoComponentsOp(IRDLOperation):
    r"""Stack two polynomial bundles as $(c_0,c_1)$.

    No residue arithmetic is performed.  Both inputs retain their active prime
    rows and representation; a new leading component axis forms a two-component
    ciphertext-shaped bundle."""

    name = "fhelium_rns.pack_two_components"
    component0 = operand_def(RnsBundleType)
    component1 = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        component0: SSAValue | Operation,
        component1: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[component0, component1],
            result_types=[result_type],
        )


@irdl_op_definition
class PackThreeComponentsOp(IRDLOperation):
    r"""Stack three polynomial bundles as $(c_0,c_1,c_2)$.

    No residue arithmetic is performed.  Matching prime rows and representation
    are preserved while a new leading component axis forms a three-component
    ciphertext-shaped bundle."""

    name = "fhelium_rns.pack_three_components"
    component0 = operand_def(RnsBundleType)
    component1 = operand_def(RnsBundleType)
    component2 = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        component0: SSAValue | Operation,
        component1: SSAValue | Operation,
        component2: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[component0, component1, component2],
            result_types=[result_type],
        )


@irdl_op_definition
class MontgomeryMultiplyOp(IRDLOperation):
    r"""Multiply NTT/Montgomery polynomial bundles pointwise.

    For active prime $q_i$, transform index $k$, and Montgomery radix $R_i$,
    the operation computes $z_{i,k}=x_{i,k}y_{i,k}R_i^{-1}\bmod q_i$.
    The result remains NTT/Montgomery and represents polynomial multiplication in
    $\mathbb{Z}_{q_i}[X]/(X^N+1)$.  Prime rows and structural axes are
    preserved."""

    name = "fhelium_rns.montgomery_multiply"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class HybridModUpDigitOp(IRDLOperation):
    r"""Extend one hybrid-RNS digit from active Q rows to active QP rows.

    A hybrid digit is a selected group of Q-prime residues used in key switching.
    For each coefficient, the selected rows are reconstructed as one integer
    class $d$ modulo their product and basis-extended so every active Q and
    special P row stores $dR_i\bmod q_i$ or $dR_i\bmod p_i$.  The result is
    coefficient-domain Montgomery data; it does not change the source ciphertext
    scale."""

    name = "fhelium_rns.hybrid_modup_digit"
    source = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    plan = operand_def(KeySwitchPlanType)
    result = result_def(RnsBundleType)
    digit_index = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        source: SSAValue | Operation,
        parameters: SSAValue | Operation,
        plan: SSAValue | Operation,
        result_type: Attribute,
        *,
        digit_index: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[source, parameters, plan],
            result_types=[result_type],
            attributes={
                "digit_index": (
                    IntegerAttr(digit_index, 64)
                    if isinstance(digit_index, int)
                    else digit_index
                )
            },
        )

    def verify_(self) -> None:
        if int(self.digit_index.value.data) < 0:
            raise VerifyException(
                "Hybrid ModUp digit index must be nonnegative"
            )


@irdl_op_definition
class KeySwitchDigitProductOp(IRDLOperation):
    r"""Multiply one lifted digit by its evaluation-key pair.

    For NTT/Montgomery digit $d$ and key digit $(k_{d,0},k_{d,1})$, the result
    has two components
    $(d k_{d,0},d k_{d,1})$ in every active QP prime row.  Products use
    Montgomery pointwise multiplication and remain NTT/Montgomery.  The operation
    forms one summand of hybrid key switching and leaves CKKS level and scale
    unchanged."""

    name = "fhelium_rns.key_switch_digit_product"
    digit = operand_def(RnsBundleType)
    key = operand_def(EvaluationKeyResourceType)
    parameters = operand_def(RnsParametersType)
    plan = operand_def(KeySwitchPlanType)
    result = result_def(RnsBundleType)
    key_digit_index = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        digit: SSAValue | Operation,
        key: SSAValue | Operation,
        parameters: SSAValue | Operation,
        plan: SSAValue | Operation,
        result_type: Attribute,
        *,
        key_digit_index: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[digit, key, parameters, plan],
            result_types=[result_type],
            attributes={
                "key_digit_index": (
                    IntegerAttr(key_digit_index, 64)
                    if isinstance(key_digit_index, int)
                    else key_digit_index
                )
            },
        )

    def verify_(self) -> None:
        if int(self.key_digit_index.value.data) < 0:
            raise VerifyException(
                "Key-switch key digit index must be nonnegative"
            )


@irdl_op_definition
class AddMontgomeryLazyOp(IRDLOperation):
    r"""Accumulate two Montgomery bundles without full standard reduction.

    For each active prime $q_i$ or $p_i$, the represented result is
    $z=x+y\pmod {q_i}$ while storage may remain in the implementation's lazy
    range, currently bounded modulo $2q_i$.  Polynomial domain, Montgomery
    factor, prime rows, component axes, level, and scale are preserved."""

    name = "fhelium_rns.add_montgomery_lazy"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class ModDownQpToQOp(IRDLOperation):
    r"""Remove the special-prime product P from a key-switch accumulator.

    Let $P=\prod_j p_j$ for the special P rows.  For each coefficient of each
    accumulator component, one P prime $p_d$ is removed at a time.  If
    $d=x\bmod p_d$, every surviving row becomes
    $(x_i-d)p_d^{-1}\bmod q_i$ or
    $(x_i-d)p_d^{-1}\bmod p_i$.  Repeating this residue-algebra division removes
    all P rows and returns active-Q residues.  This is the ModDown stage of hybrid
    key switching; key construction supplies the factor P, so CKKS message scale
    and level do not change."""

    name = "fhelium_rns.moddown_qp_to_q"
    value = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    plan = operand_def(KeySwitchPlanType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        plan: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[value, parameters, plan],
            result_types=[result_type],
        )


@irdl_op_definition
class CoefficientAutomorphismOp(IRDLOperation):
    r"""Apply the negacyclic ring automorphism $\sigma_g$.

    For odd ``galois_element`` $g$, $\sigma_g$ substitutes
    $X\mapsto X^g$ in $R_q=\mathbb{Z}_q[X]/(X^N+1)$.  A coefficient
    $a_jX^j$ moves to index $gj\bmod N$ and changes sign when reduction of
    $gj$ modulo $2N$ crosses $N$.  The same permutation is applied to
    every component and prime row.  Level, scale, basis, and residue form remain
    unchanged."""

    name = "fhelium_rns.coefficient_automorphism"
    value = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    galois_element = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        galois_element: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes={
                "galois_element": (
                    IntegerAttr(galois_element, 64)
                    if isinstance(galois_element, int)
                    else galois_element
                )
            },
        )

    def verify_(self) -> None:
        if int(self.galois_element.value.data) % 2 == 0:
            raise VerifyException("RNS Galois element must be odd")


class _ResidueConversionOp(IRDLOperation):
    value = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class StandardToMontgomeryOp(_ResidueConversionOp):
    r"""Convert coefficient residues to Montgomery representation.

    For every active prime $q_i$ and stored residue $x_i$, the result is
    $x_iR_i\bmod q_i$, where $R_i$ is the Montgomery radix.  Polynomial
    coefficients, prime rows, level, basis, and CKKS scale otherwise do not
    change."""

    name = "fhelium_rns.standard_to_montgomery"


@irdl_op_definition
class MontgomeryToStandardOp(_ResidueConversionOp):
    r"""Remove the Montgomery factor from coefficient residues.

    For every active prime $q_i$, the operation maps stored
    $x_iR_i\bmod q_i$ to $x_i\bmod q_i$ by Montgomery reduction.  Polynomial
    coefficients, prime rows, level, basis, and CKKS scale are preserved."""

    name = "fhelium_rns.montgomery_to_standard"


@irdl_op_definition
class RestrictLevelOp(IRDLOperation):
    r"""Select the suffix of prime rows belonging to a later level.

    If level $\ell$ uses active rows $(q_\ell,\ldots,q_L)$, restriction to
    $t\ge\ell$ discards $(q_\ell,\ldots,q_{t-1})$ and copies the remaining
    rows.  No division or rounding occurs, so the actual scale and every surviving
    residue are unchanged.  P rows remain when the modulus basis is QP."""

    name = "fhelium_rns.restrict_level"
    value = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    target_level = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        target_level: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes={
                "target_level": (
                    IntegerAttr(target_level, 64)
                    if isinstance(target_level, int)
                    else target_level
                )
            },
        )


@irdl_op_definition
class ReinterpretScaleOp(IRDLOperation):
    r"""Replace scale metadata without changing an RNS payload.

    The result refers to the same residue classes and polynomial representation
    but records the supplied actual scale $\Delta'$.  Consequently decoding
    interprets the represented coefficients as $x/\Delta'$ rather than
    $x/\Delta$; no modular arithmetic, row selection, or level transition is
    performed."""

    name = "fhelium_rns.reinterpret_scale"
    value = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    scale = attr_def(FloatAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute,
        *,
        scale: float | FloatAttr,
    ) -> None:
        scale_attribute: Attribute
        if isinstance(scale, FloatAttr):
            scale_attribute = scale
        else:
            scale_attribute = FloatAttr(float(scale), Float64Type())
        super().__init__(
            operands=[value],
            result_types=[result_type],
            attributes={"scale": scale_attribute},
        )


_RNS_OPERATION_TYPES = (
    AddStandardOp,
    SubtractStandardOp,
    NegateStandardOp,
    AddPlaintextOp,
    MultiplyPlaintextOp,
    RescaleDropLeadingPrimeOp,
    ExtractComponentOp,
    PackTwoComponentsOp,
    PackThreeComponentsOp,
    MontgomeryMultiplyOp,
    HybridModUpDigitOp,
    KeySwitchDigitProductOp,
    AddMontgomeryLazyOp,
    ModDownQpToQOp,
    CoefficientAutomorphismOp,
    StandardToMontgomeryOp,
    MontgomeryToStandardOp,
    RestrictLevelOp,
    ReinterpretScaleOp,
)

_RNS_ATTRIBUTES: dict[type[Operation], tuple[str, ...]] = {
    RescaleDropLeadingPrimeOp: ("rounding",),
    ExtractComponentOp: ("component",),
    HybridModUpDigitOp: ("digit_index",),
    KeySwitchDigitProductOp: ("key_digit_index",),
    CoefficientAutomorphismOp: ("galois_element",),
    RestrictLevelOp: ("target_level",),
    ReinterpretScaleOp: ("scale",),
}

OPERATION_SPECS: tuple[OperationSpec, ...] = tuple(
    registered_operation_spec(
        operation_type,
        "rns",
        validator=(
            flat_with_attributes(*_RNS_ATTRIBUTES[operation_type])
            if operation_type in _RNS_ATTRIBUTES
            else flat_without_attributes
        ),
    )
    for operation_type in _RNS_OPERATION_TYPES
)
"""Semantic specifications owned by the RNS dialect."""


FHEliumRns = Dialect(
    "fhelium_rns",
    [
        *_RNS_OPERATION_TYPES,
    ],
    [
        RnsBundleType,
        RnsParametersType,
        RescalePlanType,
        KeySwitchPlanType,
        EvaluationKeyResourceType,
    ],
)
"""Logical RNS operations and resource types."""


__all__ = [
    "AddStandardOp",
    "AddPlaintextOp",
    "AddMontgomeryLazyOp",
    "CoefficientAutomorphismOp",
    "EvaluationKeyResourceType",
    "ExtractComponentOp",
    "NegateStandardOp",
    "HybridModUpDigitOp",
    "KeySwitchDigitProductOp",
    "KeySwitchPlanType",
    "ModDownQpToQOp",
    "MontgomeryToStandardOp",
    "MontgomeryMultiplyOp",
    "MultiplyPlaintextOp",
    "OPERATION_SPECS",
    "PackTwoComponentsOp",
    "PackThreeComponentsOp",
    "SubtractStandardOp",
    "FHEliumRns",
    "RescaleDropLeadingPrimeOp",
    "RescalePlanType",
    "RestrictLevelOp",
    "ReinterpretScaleOp",
    "RnsBundleType",
    "RnsParametersType",
    "StandardToMontgomeryOp",
]
