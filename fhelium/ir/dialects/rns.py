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
from typing import Literal

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
    var_operand_def,
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
    """Depth-specific dropped-prime, inverse, and surviving-row resources."""

    name = "fhelium_rns.rescale_plan"


@irdl_attr_definition
class KeySwitchPlanType(OpenStateType):
    """Depth-specialized hybrid decomposition and ModUp/ModDown resources."""

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
    shape, depth, basis, polynomial domain, Montgomery factor, and scale are
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
    rows, depth, modulus basis, polynomial domain, and scale."""

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
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext, plaintext, parameters],
            result_types=[result_type],
            attributes=attributes,
        )


@irdl_op_definition
class AddPlaintextOp(_CiphertextPlaintextOp):
    r"""Add a prepared plaintext polynomial to ciphertext component zero.

    For ciphertext $(c_0,\ldots,c_{k-1})$ and plaintext polynomial $p$, each
    active-prime row computes $c'_0=c_0+p\pmod {q_i}$ and
    $c'_j=c_j$ for $j>0$. With coefficient/standard ciphertext input, the
    prepared plaintext uses coefficient-domain Montgomery residues and the
    operation returns standard residues. With NTT/Montgomery input, both
    operands and the result remain NTT/Montgomery. Depth, component count, and
    ciphertext scale are preserved."""

    name = "fhelium_rns.add_plaintext"
    polynomial_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )


@irdl_op_definition
class MultiplyPlaintextOp(_CiphertextPlaintextOp):
    r"""Multiply every ciphertext component by a prepared plaintext.

    The ciphertext and plaintext are NTT-domain Montgomery bundles on matching
    prime rows.  For each component $j$, prime $q_i$, and transform index
    $k$, Montgomery multiplication computes
    $c'_{j,i,k}=c_{j,i,k}p_{i,k}R_i^{-1}\bmod q_i$.  Thus the polynomial
    meaning is $c'_j=c_jp$ in $R_{q_i}$.  Component count and depth remain;
    the CKKS result scale is the product of operand scales."""

    name = "fhelium_rns.multiply_plaintext"


@irdl_op_definition
class MontgomeryWeightedSumOp(IRDLOperation):
    r"""Sum ciphertext/plaintext products in NTT/Montgomery representation.

    For $T$ ciphertext bundles $c_t$ followed by $T$ plaintext bundles $p_t$,
    each component, batch element, active-prime row, and NTT index computes

    $$
    y=\sum_{t=0}^{T-1}c_tp_t\pmod {q_i}.
    $$

    Every term uses the same component and batch shape, depth, prime rows,
    ciphertext scale, and plaintext scale.  ``term_count`` separates the two
    equal operand ranges.  The result keeps the ciphertext shape, depth, and
    NTT/Montgomery representation; its CKKS scale is the product of the common
    ciphertext and plaintext scales.  The operation chooses no grouping,
    platform, kernel, or execution schedule.
    """

    name = "fhelium_rns.montgomery_weighted_sum"
    parameters = operand_def(RnsParametersType)
    terms = var_operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    term_count = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        parameters: SSAValue | Operation,
        ciphertexts: tuple[SSAValue | Operation, ...],
        plaintexts: tuple[SSAValue | Operation, ...],
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=(parameters, (*ciphertexts, *plaintexts)),
            result_types=[result_type],
            attributes={"term_count": IntegerAttr(len(ciphertexts), 64)},
        )

    def verify_(self) -> None:
        count = int(self.term_count.value.data)
        if count <= 0 or len(self.terms) != 2 * count:
            raise VerifyException(
                "RNS Montgomery weighted sum requires equally sized, "
                "non-empty ciphertext and plaintext operand ranges"
            )


@irdl_op_definition
class MontgomeryWeightedSumsOp(IRDLOperation):
    r"""Apply a dense plaintext matrix to shared ciphertext terms.

    For $T$ ciphertext bundles $c_t$ and $G T$ plaintext bundles $p_{g,t}$,
    the operation returns $G$ independent bundles

    $$
    y_g=\sum_{t=0}^{T-1}c_t p_{g,t}\pmod {q_i}.
    $$

    Operands after ``parameters`` contain the $T$ ciphertexts followed by the
    group-major plaintext matrix. Every ciphertext has one NTT/Montgomery
    component, batch, depth, and prime-row layout; every plaintext has the
    matching row and batch layout and one common scale. The results keep the
    ciphertext state and have scale $\Delta_c\Delta_p$. ``term_count`` and
    ``group_count`` describe the supplied matrix; the operation does not
    choose its groups, platform, kernel, or execution schedule.
    """

    name = "fhelium_rns.montgomery_weighted_sums"
    parameters = operand_def(RnsParametersType)
    terms = var_operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    term_count = attr_def(IntegerAttr)
    group_count = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        parameters: SSAValue | Operation,
        ciphertexts: tuple[SSAValue | Operation, ...],
        plaintexts: tuple[SSAValue | Operation, ...],
        result_type: Attribute,
        *,
        group_count: int,
    ) -> None:
        super().__init__(
            operands=(parameters, (*ciphertexts, *plaintexts)),
            result_types=[result_type],
            attributes={
                "term_count": IntegerAttr(len(ciphertexts), 64),
                "group_count": IntegerAttr(group_count, 64),
            },
        )

    def verify_(self) -> None:
        term_count = int(self.term_count.value.data)
        group_count = int(self.group_count.value.data)
        if term_count <= 0 or group_count <= 0:
            raise VerifyException(
                "RNS weighted sums require positive term and group counts"
            )
        if len(self.terms) != term_count * (group_count + 1):
            raise VerifyException(
                "RNS weighted-sums operands must contain term_count "
                "ciphertexts followed by group_count * term_count plaintexts"
            )


@irdl_op_definition
class RescaleDropLeadingPrimesOp(IRDLOperation):
    r"""Divide-round by the product of a leading prime group and remove its rows.

    For the first ``drop_count`` primes, let $M$ be their product and
    $r=x\bmod M$ in $[0,M)$. Floor quotient uses $t=r$; nearest quotient
    uses $t=r-M$ when $r>M/2$, otherwise $t=r$. Each surviving row is
    $y_i=(x_i-t)M^{-1}\bmod q_i$. Component count and polynomial domain
    are preserved. NTT/Montgomery input may retain surviving evaluations
    while constructing the rounded coefficient correction from dropped rows.
    The caller supplies resulting CKKS depth and scale; this RNS operation
    describes a row-group quotient, not a count of CKKS transitions."""

    name = "fhelium_rns.rescale_drop_leading_primes"
    value = operand_def(RnsBundleType)
    plan = operand_def(RescalePlanType)
    result = result_def(RnsBundleType)
    drop_count = attr_def(IntegerAttr, default_value=IntegerAttr(1, 64))
    rounding = opt_attr_def(StringAttr)
    input_domain = attr_def(StringAttr, default_value=StringAttr("coefficient"))
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        plan: SSAValue | Operation,
        result_type: Attribute,
        *,
        drop_count: int = 1,
        rounding: str | StringAttr = "nearest",
        polynomial_domain: Literal["coefficient", "ntt"] = "coefficient",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["drop_count"] = IntegerAttr(drop_count, 64)
        attrs["rounding"] = (
            StringAttr(rounding) if isinstance(rounding, str) else rounding
        )
        attrs["input_domain"] = StringAttr(polynomial_domain)
        attrs["output_domain"] = StringAttr(polynomial_domain)
        super().__init__(
            operands=[value, plan],
            result_types=[result_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        """Accept only the represented CKKS quotient-rounding rules."""

        if self.drop_count.value.data < 1:
            raise VerifyException("RNS rescale drop_count must be positive")
        if self.rounding is not None and self.rounding.data not in {
            "nearest",
            "floor",
        }:
            raise VerifyException(
                "RNS rescale rounding must be 'nearest' or 'floor'"
            )
        if (
            self.input_domain.data not in {"coefficient", "ntt"}
            or self.output_domain != self.input_domain
        ):
            raise VerifyException(
                "RNS rescale must preserve coefficient or NTT domain"
            )


@irdl_op_definition
class ExtractComponentOp(IRDLOperation):
    r"""Select one ciphertext-component polynomial without arithmetic.

    For an RNS bundle $(c_0,\ldots,c_{k-1})$, ``component=j`` returns $c_j$
    with the same coefficient values, active prime rows, polynomial domain,
    Montgomery state, depth, and scale."""

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
    forms one summand of hybrid key switching and leaves CKKS depth and scale
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
    factor, prime rows, component axes, depth, and scale are preserved."""

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
    and depth do not change."""

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
class ModDownNttQpToQOp(IRDLOperation):
    r"""Remove P from two QP accumulators and retain NTT/Montgomery Q rows.

    The input has shape ``[2, *batch, |Q|+|P|, N]``; the result has shape
    ``[2, *batch, |Q|, N]``. The NTT-plan operand supplies the transforms
    for the same RNS context as the parameters and key-switch plan.

    Let $r\in[0,P)$ be the coefficient representative reconstructed from
    $\operatorname{INTT}_P(\widehat{x}_P)$. The result is
    $\widehat{x}_Q P^{-1}+\operatorname{NTT}_Q(-rP^{-1})\bmod q_i$.
    This equals coefficient-domain ModDown followed by forward NTT, without
    inverting the Q rows. Output drops P, preserves active Q rows and N,
    and remains in Montgomery representation. It does not rescale CKKS values.
    """

    name = "fhelium_rns.moddown_ntt_qp_to_q"
    value = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    ntt_plan = operand_def()
    plan = operand_def(KeySwitchPlanType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        ntt_plan: SSAValue | Operation,
        plan: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[value, parameters, ntt_plan, plan],
            result_types=[result_type],
        )


@irdl_op_definition
class CoefficientAutomorphismOp(IRDLOperation):
    r"""Apply the negacyclic ring automorphism $\sigma_g$.

    For odd ``galois_element`` $g$, $\sigma_g$ substitutes
    $X\mapsto X^g$ in $R_q=\mathbb{Z}_q[X]/(X^N+1)$.  A coefficient
    $a_jX^j$ moves to index $gj\bmod N$ and changes sign when reduction of
    $gj$ modulo $2N$ crosses $N$.  The same permutation is applied to
    every component and prime row.  Depth, scale, basis, and residue form remain
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
    coefficients, prime rows, depth, basis, and CKKS scale otherwise do not
    change."""

    name = "fhelium_rns.standard_to_montgomery"


@irdl_op_definition
class MontgomeryToStandardOp(_ResidueConversionOp):
    r"""Remove the Montgomery factor from coefficient residues.

    For every active prime $q_i$, the operation maps stored
    $x_iR_i\bmod q_i$ to $x_i\bmod q_i$ by Montgomery reduction.  Polynomial
    coefficients, prime rows, depth, basis, and CKKS scale are preserved."""

    name = "fhelium_rns.montgomery_to_standard"


@irdl_op_definition
class RestrictDepthOp(IRDLOperation):
    r"""Select the suffix of prime rows belonging to a later depth.

    If depth $\ell$ uses active rows $(q_\ell,\ldots,q_L)$, restriction to
    $t\ge\ell$ discards $(q_\ell,\ldots,q_{t-1})$ and copies the remaining
    rows.  No division or rounding occurs, so the actual scale and every surviving
    residue are unchanged.  P rows remain when the modulus basis is QP."""

    name = "fhelium_rns.restrict_depth"
    value = operand_def(RnsBundleType)
    parameters = operand_def(RnsParametersType)
    result = result_def(RnsBundleType)
    target_depth = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        target_depth: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes={
                "target_depth": (
                    IntegerAttr(target_depth, 64)
                    if isinstance(target_depth, int)
                    else target_depth
                )
            },
        )


@irdl_op_definition
class ReinterpretScaleOp(IRDLOperation):
    r"""Replace scale metadata without changing an RNS payload.

    The result refers to the same residue classes and polynomial representation
    but records the supplied actual scale $\Delta'$.  Consequently decoding
    interprets the represented coefficients as $x/\Delta'$ rather than
    $x/\Delta$; no modular arithmetic, row selection, or depth transition is
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
    MontgomeryWeightedSumOp,
    MontgomeryWeightedSumsOp,
    RescaleDropLeadingPrimesOp,
    ExtractComponentOp,
    PackTwoComponentsOp,
    PackThreeComponentsOp,
    MontgomeryMultiplyOp,
    HybridModUpDigitOp,
    KeySwitchDigitProductOp,
    AddMontgomeryLazyOp,
    ModDownQpToQOp,
    ModDownNttQpToQOp,
    CoefficientAutomorphismOp,
    StandardToMontgomeryOp,
    MontgomeryToStandardOp,
    RestrictDepthOp,
    ReinterpretScaleOp,
)

_RNS_ATTRIBUTES: dict[type[Operation], tuple[str, ...]] = {
    AddPlaintextOp: ("polynomial_domain",),
    MontgomeryWeightedSumOp: ("term_count",),
    MontgomeryWeightedSumsOp: ("term_count", "group_count"),
    RescaleDropLeadingPrimesOp: (
        "drop_count",
        "rounding",
        "input_domain",
        "output_domain",
    ),
    ExtractComponentOp: ("component",),
    HybridModUpDigitOp: ("digit_index",),
    KeySwitchDigitProductOp: ("key_digit_index",),
    CoefficientAutomorphismOp: ("galois_element",),
    RestrictDepthOp: ("target_depth",),
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
    "ModDownNttQpToQOp",
    "MontgomeryToStandardOp",
    "MontgomeryMultiplyOp",
    "MontgomeryWeightedSumOp",
    "MontgomeryWeightedSumsOp",
    "MultiplyPlaintextOp",
    "OPERATION_SPECS",
    "PackTwoComponentsOp",
    "PackThreeComponentsOp",
    "SubtractStandardOp",
    "FHEliumRns",
    "RescaleDropLeadingPrimesOp",
    "RescalePlanType",
    "RestrictDepthOp",
    "ReinterpretScaleOp",
    "RnsBundleType",
    "RnsParametersType",
    "StandardToMontgomeryOp",
]
