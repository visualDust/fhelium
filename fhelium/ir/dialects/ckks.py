r"""Cheon-Kim-Kim-Song (CKKS) evaluator operations and value state.

A two-component ciphertext (CT2) has phase $c_0+c_1s$; a
three-component ciphertext (CT3) has phase $c_0+c_1s+c_2s^2$ for secret
polynomial $s$.  The residue number system (RNS) stores each polynomial
modulo active ciphertext primes Q; QP appends special P primes used during key
switching.  The number-theoretic transform (NTT) changes negacyclic polynomial
multiplication into pointwise multiplication.  Every plaintext and ciphertext
may carry its own actual scale $\Delta$.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from xdsl.dialects.builtin import FloatAttr, IntegerAttr, StringAttr
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
    var_result_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from .._operation_catalog import (
    OperationSpec,
    OperationValidator,
    flat_operation,
    flat_with_attributes,
    registered_operation_spec,
    required_string_attribute,
    unsupported_attributes,
)
from ._common import OpenStateType, ValueRole
from .core import MessageType


@irdl_attr_definition
class CiphertextType(OpenStateType):
    """CKKS ciphertext with partial level, scale, basis, and domain state."""

    name = "fhelium_ckks.ciphertext"
    ROLE: ClassVar[ValueRole] = "encrypted"


@irdl_attr_definition
class PlaintextType(OpenStateType):
    """CKKS plaintext with partial encoding and RNS representation state."""

    name = "fhelium_ckks.plaintext"
    ROLE: ClassVar[ValueRole] = "plaintext"


@irdl_attr_definition
class EvaluationKeyType(OpenStateType):
    """Caller-bound CKKS evaluation-key resource and compatibility state."""

    name = "fhelium_ckks.evaluation_key"


@irdl_attr_definition
class CompressedPlaintextType(OpenStateType):
    """Represent operation-ready compressed RNS plaintext storage."""

    name = "fhelium_ckks.compressed_plaintext"

    @property
    def value_role(self) -> ValueRole:
        return "plaintext"


@irdl_op_definition
class EncodeOp(IRDLOperation):
    r"""Encode ordered complex slots as a scaled integer polynomial.

    For ring $R=\mathbb{Z}[X]/(X^N+1)$, let $\sigma$ be the configured CKKS
    embedding in the configured generator's slot order.  Given slots $m$ and
    actual scale $\Delta$, the operation returns
    $a=\operatorname{RandRound}(\Delta\,\sigma^{-1}(m))\in R$.
    ``level`` records the later modulus-chain placement but does not reduce the
    integer coefficients.  The registered ``native-ckks-encode`` implementation
    performs the embedding and stochastic rounding."""

    name = "fhelium_ckks.encode"
    message = operand_def(MessageType)
    result = result_def(PlaintextType)
    level = attr_def(IntegerAttr)
    scale = attr_def(FloatAttr)


@irdl_op_definition
class DecodeOp(IRDLOperation):
    r"""Decode a scaled coefficient polynomial into ordered CKKS slots.

    For coefficient data $a$ with actual scale $\Delta$, the operation returns
    $m'=\sigma(a/\Delta)$, truncated to the configured slot count and ordered by
    the configured Galois generator.  ``is_real`` selects the real part after the
    complex embedding.  The input may be freshly encoded integers or bounded
    approximate coefficients reconstructed by decryption."""

    name = "fhelium_ckks.decode"
    plaintext = operand_def(PlaintextType)
    result = result_def(MessageType)
    is_real = attr_def(IntegerAttr)
    traits = traits_def(Pure())


@irdl_op_definition
class IntegerCoefficientsToRnsOp(IRDLOperation):
    r"""Reduce integer polynomial coefficients into active RNS rows.

    RNS means residue number system.  At ``level`` $\ell$, every coefficient
    $a_j$ is mapped to $a_j\bmod q_i$ for each active Q prime and, for basis
    ``QP``, to $a_j\bmod p_i$ for each special P prime.  The result is
    coefficient-domain standard-residue data.  Scale and polynomial meaning do
    not change."""

    name = "fhelium_ckks.integer_coefficients_to_rns"
    plaintext = operand_def(PlaintextType)
    result = result_def(PlaintextType)
    modulus_basis = attr_def(StringAttr)
    level = attr_def(IntegerAttr)
    traits = traits_def(Pure())


@irdl_op_definition
class EncryptOp(IRDLOperation):
    r"""Encrypt an integer plaintext polynomial as a CT2 ciphertext.

    Let the public key be $(k_0,k_1)$, satisfying
    $k_0+k_1s\approx0$, let $v$ be a sampled binary polynomial, and let
    $e_0,e_1$ be sampled error polynomials.  For plaintext $a$, the result is
    $c_0=k_0v+a+e_0$ and $c_1=k_1v+e_1$ modulo every active prime.  The output
    has two components in coefficient-domain standard Q or QP residues and keeps
    the plaintext level and actual scale.  ``key_symbol`` identifies the bound
    public-key resource."""

    name = "fhelium_ckks.encrypt"
    plaintext = operand_def(PlaintextType)
    result = result_def(CiphertextType)
    key_symbol = attr_def(StringAttr)


@irdl_op_definition
class DecryptOp(IRDLOperation):
    r"""Evaluate a ciphertext phase and reconstruct coefficient data.

    For CT2 $(c_0,c_1)$, the phase is $c_0+c_1s$; for CT3 it is
    $c_0+c_1s+c_2s^2$, computed modulo each active prime.  The registered
    decryption implementation converts products to coefficient-domain standard
    RNS, then reconstructs a centered bounded integer from the trailing active Q
    rows.  The result is an approximate coefficient plaintext at the ciphertext's
    level and actual scale.  ``key_symbol`` selects the secret key."""

    name = "fhelium_ckks.decrypt"
    ciphertext = operand_def(CiphertextType)
    result = result_def(PlaintextType)
    key_symbol = attr_def(StringAttr)
    traits = traits_def(Pure())


class _UnaryCiphertextOp(IRDLOperation):
    value = operand_def(CiphertextType)
    result = result_def(CiphertextType)
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


class _BinaryCiphertextOp(IRDLOperation):
    lhs = operand_def(CiphertextType)
    rhs = operand_def(CiphertextType)
    result = result_def(CiphertextType)
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


@irdl_op_definition
class NegateOp(_UnaryCiphertextOp):
    r"""Negate every component of a CKKS ciphertext.

    For $c=(c_0,\ldots,c_{k-1})$, the result is
    $(-c_0,\ldots,-c_{k-1})$ modulo each active prime and therefore decrypts to
    the negated message.  Component count, level, actual scale, prime rows,
    polynomial domain, and Montgomery state are unchanged."""

    name = "fhelium_ckks.negate"


@irdl_op_definition
class RotateOp(IRDLOperation):
    r"""Rotate CKKS slots through a Galois automorphism and key switch.

    The input is a coefficient-domain standard-Q CT2 ciphertext.  A rotation key
    records a signed slot displacement $r$ and its odd Galois element $g$.
    Applying $\sigma_g:X\mapsto X^g$ rotates the encoded slots and changes the
    secret relation from $s$ to $\sigma_g(s)$; the key-switch stage returns
    that relation to $s$.  The result is a coefficient/standard CT2 ciphertext
    with level, actual scale, and active Q rows preserved."""

    name = "fhelium_ckks.rotate"
    value = operand_def(CiphertextType)
    key = operand_def(EvaluationKeyType)
    result = result_def(CiphertextType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        key: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[value, key],
            result_types=[result_type or SSAValue.get(value).type],
            attributes=attributes,
        )


@irdl_op_definition
class RotateManyOp(IRDLOperation):
    r"""Produce several slot rotations from one shared key-switch preparation.

    The input is a coefficient-domain standard-Q CT2 ciphertext.  Each key operand
    defines one Galois automorphism and signed slot displacement.  For every key
    the mathematical result equals an independent ``RotateOp`` of the same input.
    Hoisting shares hybrid digit decomposition and Q-to-QP basis-extension work
    across outputs but does not change their order or values.  Every result
    preserves the input level, actual scale, and active Q rows."""

    name = "fhelium_ckks.hoisted_rotate_many"
    value = operand_def(CiphertextType)
    keys = var_operand_def(EvaluationKeyType)
    outputs = var_result_def(CiphertextType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        keys: tuple[SSAValue | Operation, ...],
        result_types: tuple[CiphertextType, ...],
    ) -> None:
        super().__init__(
            operands=(value, keys),
            result_types=(result_types,),
        )

    def verify_(self) -> None:
        count = len(self.outputs)
        if count == 0:
            raise VerifyException(
                "CKKS rotate-many requires at least one result"
            )
        if len(self.keys) != count:
            raise VerifyException(
                "CKKS rotate-many results and key operands must have equal "
                "lengths"
            )


class _RepresentationOp(IRDLOperation):
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

    def verify_(self) -> None:
        """Require matching CKKS ciphertext or plaintext value kinds."""

        source = self.value.type
        result = self.result.type
        if type(source) is not type(result) or not isinstance(
            source, (CiphertextType, PlaintextType)
        ):
            raise VerifyException(
                "CKKS NTT transition requires matching ciphertext or plaintext types"
            )


@irdl_op_definition
class ToNttOp(_RepresentationOp):
    r"""Transform CKKS residue polynomials to NTT/Montgomery form.

    For each component and active prime $q_i$, coefficient residues are mapped
    to $\operatorname{NTT}_{q_i}(c_j)R_i\bmod q_i$.  The represented ring
    element, component count, prime rows, level, basis, and actual scale remain
    unchanged.  Lowering selects the corresponding operation from the NTT
    dialect according to the input residue representation."""

    name = "fhelium_ckks.to_ntt"


@irdl_op_definition
class FromNttOp(_RepresentationOp):
    r"""Transform CKKS NTT/Montgomery polynomials to coefficient form.

    An inverse negacyclic transform is applied independently to every component
    and prime row.  Ciphertexts end in standard residues; plaintext lowering may
    retain Montgomery residues.  The represented ring element, component count,
    prime rows, level, basis, and actual scale are unchanged."""

    name = "fhelium_ckks.from_ntt"


@irdl_op_definition
class ToMontgomeryResiduesOp(_RepresentationOp):
    r"""Multiply coefficient-domain plaintext residues by the Montgomery radix.

    For each active prime $q_i$, the stored row changes from $x_i$ to
    $x_iR_i\bmod q_i$.  This representation transition preserves the
    plaintext polynomial, Q or QP rows, level, and actual scale."""

    name = "fhelium_ckks.to_montgomery_residues"


@irdl_op_definition
class ToStandardResiduesOp(_RepresentationOp):
    r"""Remove the Montgomery factor from coefficient-domain plaintext rows.

    For each active prime $q_i$, Montgomery reduction maps $x_iR_i$ to
    $x_i\bmod q_i$.  The plaintext polynomial, Q or QP rows, level, and actual
    scale are preserved."""

    name = "fhelium_ckks.to_standard_residues"


@irdl_op_definition
class AddOp(_BinaryCiphertextOp):
    r"""Add two state-compatible ciphertexts component by component.

    For each component $j$, active prime $q_i$, and coefficient, the operation
    computes $c'_j=c^{(a)}_j+c^{(b)}_j\pmod {q_i}$.  Inputs have the same
    component count, level, prime rows, polynomial domain, residue representation,
    and actual scale.  The result retains that state and represents the sum of the
    decoded slot values."""

    name = "fhelium_ckks.add"


@irdl_op_definition
class SubtractOp(_BinaryCiphertextOp):
    r"""Subtract ciphertexts component by component.

    For each component $j$ and active prime $q_i$, the operation computes
    $c'_j=c^{(a)}_j-c^{(b)}_j\pmod {q_i}$.  Compatible level, prime rows,
    representation, component count, and actual scale are preserved, and the
    result represents the slotwise difference."""

    name = "fhelium_ckks.subtract"


@irdl_op_definition
class MultiplyOp(_BinaryCiphertextOp):
    r"""Multiply two CT2 ciphertexts by polynomial convolution.

    For $a=(a_0,a_1)$ and $b=(b_0,b_1)$, the CT3 result is
    $(a_0b_0,\ a_0b_1+a_1b_0,\ a_1b_1)$.  Products are computed pointwise in
    NTT/Montgomery form modulo every active prime and correspond to negacyclic
    polynomial products.  If input actual scales are $\Delta_a$ and
    $\Delta_b$, the result scale is $\Delta_a\Delta_b$; no rescale or
    relinearization is implicit."""

    name = "fhelium_ckks.multiply"


class _RealScalarCiphertextOp(IRDLOperation):
    ciphertext = operand_def(CiphertextType)
    result = result_def(CiphertextType)
    scalar = attr_def(FloatAttr)
    scalar_scale = attr_def(FloatAttr)

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        scalar: FloatAttr,
        scalar_scale: FloatAttr,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext],
            result_types=[result_type or SSAValue.get(ciphertext).type],
            attributes={
                **({} if attributes is None else attributes),
                "scalar": scalar,
                "scalar_scale": scalar_scale,
            },
        )


@irdl_op_definition
class AddScalarOp(_RealScalarCiphertextOp):
    r"""Add a stochastically quantized real constant to component zero.

    The input uses coefficient-domain standard residues.  For scalar $u$ and
    ``scalar_scale`` $\delta$, the implementation samples
    $k=\operatorname{RandRound}(u\delta)$, reduces $k$ into each active prime,
    and adds it to the constant coefficient of $c_0$.  Other coefficients and
    components do not change.  The ciphertext actual scale $\Delta$ is retained,
    so the decoded increment is approximately $k/\Delta$; choosing
    $\delta=\Delta$ represents addition by $u$."""

    name = "fhelium_ckks.add_scalar"


@irdl_op_definition
class MultiplyScalarOp(_RealScalarCiphertextOp):
    r"""Multiply all ciphertext components by a quantized real constant.

    For scalar $u$ at scalar scale $\delta$, the implementation samples
    $k=\operatorname{RandRound}(u\delta)$ and computes $c'_j=kc_j$ modulo
    every active prime.  The output actual scale is $\Delta\delta$ for input
    scale $\Delta$, so it represents multiplication by approximately $u$.
    Level, component count, prime rows, and polynomial representation are
    preserved; rescaling is separate."""

    name = "fhelium_ckks.multiply_scalar"


@irdl_op_definition
class MultiplyIntegerScalarOp(IRDLOperation):
    r"""Multiply every ciphertext component by an integer scalar.

    For integer $k$, each polynomial becomes $c'_j=kc_j$ modulo every active
    prime.  Because $k$ is not a newly scaled encoding, the ciphertext actual
    scale, level, component count, prime rows, polynomial domain, and Montgomery
    state remain unchanged."""

    name = "fhelium_ckks.multiply_integer_scalar"
    ciphertext = operand_def(CiphertextType)
    result = result_def(CiphertextType)
    scalar = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        scalar: IntegerAttr,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext],
            result_types=[result_type or SSAValue.get(ciphertext).type],
            attributes={
                **({} if attributes is None else attributes),
                "scalar": scalar,
            },
        )


class _CiphertextPlaintextOp(IRDLOperation):
    ciphertext = operand_def(CiphertextType)
    plaintext = operand_def(PlaintextType)
    result = result_def(CiphertextType)
    traits = traits_def(Pure())

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        plaintext: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext, plaintext],
            result_types=[result_type or SSAValue.get(ciphertext).type],
            attributes=attributes,
        )


@irdl_op_definition
class AddPlaintextOp(_CiphertextPlaintextOp):
    r"""Add a prepared plaintext polynomial to ciphertext component zero.

    For ciphertext $(c_0,\ldots,c_{k-1})$ and plaintext $p$, the result is
    $(c_0+p,c_1,\ldots,c_{k-1})$ modulo each active prime.  Matching level,
    prime rows, and actual scale make this a slotwise message addition.  Component
    count and ciphertext representation remain unchanged."""

    name = "fhelium_ckks.add_plaintext"


@irdl_op_definition
class MultiplyPlaintextOp(_CiphertextPlaintextOp):
    r"""Multiply each ciphertext component by a prepared plaintext polynomial.

    For ciphertext components $c_j$ and plaintext $p$, the result components
    are $c'_j=c_jp$ in every active-prime negacyclic ring.  Operands use
    NTT/Montgomery form so execution is pointwise.  The level and component count
    remain unchanged, while output actual scale is $\Delta_c\Delta_p$; no
    rescale is implicit."""

    name = "fhelium_ckks.multiply_plaintext"


class _CiphertextCompressedPlaintextOp(IRDLOperation):
    ciphertext = operand_def(CiphertextType)
    plaintext = operand_def(CompressedPlaintextType)
    result = result_def(CiphertextType)
    inplace = opt_attr_def(IntegerAttr)


@irdl_op_definition
class AddCompressedPlaintextOp(_CiphertextCompressedPlaintextOp):
    r"""Add a compressed plaintext to ciphertext component zero.

    The compressed layout supplies selected or repeated prepared plaintext values
    without materializing a full polynomial bundle.  The implementation computes
    the same active-prime modular addition as ``AddPlaintextOp`` for $c_0$ and
    leaves later components unchanged.  Level, actual scale, domain, and residue
    representation are preserved."""

    name = "fhelium_ckks.add_compressed_plaintext"


@irdl_op_definition
class MultiplyCompressedPlaintextOp(_CiphertextCompressedPlaintextOp):
    r"""Multiply ciphertext components by compressed NTT plaintext data.

    The compressed layout expands logically to a prepared NTT/Montgomery
    plaintext $p$.  Each result component represents $c_jp$ modulo every
    active prime, computed without materializing that expansion.  Component count
    and level remain; output actual scale is $\Delta_c\Delta_p$."""

    name = "fhelium_ckks.multiply_compressed_plaintext"


@irdl_op_definition
class RelinearizeOp(_UnaryCiphertextOp):
    r"""Convert a three-component product ciphertext to two components.

    The input is an NTT/Montgomery CT3 product.  For phase
    $c_0+c_1s+c_2s^2$, hybrid key switching maps the $c_2s^2$ term to
    correction polynomials $(d_0,d_1)$ under secret $s$.  The result
    $(c_0+d_0,c_1+d_1)$ preserves the phase up to key-switch error.  Level,
    actual scale, and active Q rows remain unchanged; the direct implementation
    returns coefficient-domain standard residues."""

    name = "fhelium_ckks.relinearize"


@irdl_op_definition
class SwitchKeyOp(_UnaryCiphertextOp):
    r"""Change the secret-key relation of a CT2 ciphertext.

    The input is a coefficient-domain standard-Q CT2 ciphertext.  For phase
    $c_0+c_1s_{\mathrm{src}}$, the key named by ``key_symbol`` maps
    the $c_1s_{\mathrm{src}}$ term to corrections $(d_0,d_1)$ satisfying the
    destination relation.  The result $(c_0+d_0,d_1)$ decrypts under
    $s_{\mathrm{dst}}$ to the same approximate message, apart from key-switch
    error.  Level, actual scale, active Q rows, and component count are preserved."""

    name = "fhelium_ckks.switch_key"
    key_symbol = attr_def(StringAttr)

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        key_symbol: str | StringAttr,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs["key_symbol"] = (
            StringAttr(key_symbol)
            if isinstance(key_symbol, str)
            else key_symbol
        )
        super().__init__(value, result_type, attributes=attrs)

    def verify_(self) -> None:
        if not self.key_symbol.data:
            raise VerifyException("CKKS switch-key symbol must be non-empty")


@irdl_op_definition
class ConjugateOp(_UnaryCiphertextOp):
    r"""Apply complex conjugation to every encoded CKKS slot.

    The input is a coefficient-domain standard-Q CT2 ciphertext.  The ring
    automorphism $\sigma_{2N-1}:X\mapsto X^{-1}$ maps the CKKS
    embedding to slotwise complex conjugation.  It also changes the secret
    relation to $\sigma_{2N-1}(s)$, so a conjugation key switches the result back
    to $s$.  CT2 shape, level, actual scale, and active Q rows are preserved."""

    name = "fhelium_ckks.conjugate"


@irdl_op_definition
class RescaleOp(IRDLOperation):
    r"""Divide a ciphertext by its leading active Q prime.

    For dropped prime $q_d$, every component coefficient is quotient-rounded as
    $c'_j=\operatorname{round}(c_j/q_d)$ and represented on the surviving Q
    rows.  The $q_d$ row is removed, level advances by one, and actual scale
    changes from $\Delta$ to $\Delta/q_d$.  Component count and
    coefficient-domain standard representation remain.  ``rounding`` selects the
    supported quotient rule."""

    name = "fhelium_ckks.rescale"

    value = operand_def(CiphertextType)
    result = result_def(CiphertextType)
    rounding = opt_attr_def(StringAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        rounding: str | StringAttr = "nearest",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["rounding"] = (
            StringAttr(rounding) if isinstance(rounding, str) else rounding
        )
        super().__init__(
            operands=[value],
            result_types=[result_type or SSAValue.get(value).type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        """Accept only represented CKKS quotient-rounding rules when present."""

        if self.rounding is not None and self.rounding.data not in {
            "nearest",
            "floor",
        }:
            raise VerifyException(
                "CKKS rescale rounding must be 'nearest' or 'floor'"
            )


@irdl_op_definition
class ModSwitchOp(_UnaryCiphertextOp):
    r"""Restrict a ciphertext to the Q rows at ``target_level``.

    Rows removed before the target level are discarded without dividing or
    rounding any coefficient.  Surviving residue values and the ciphertext actual
    scale are unchanged.  The operation therefore changes the modulus and level,
    not the scale; component count, polynomial domain, and residue representation
    are preserved."""

    name = "fhelium_ckks.mod_switch"
    target_level = attr_def(IntegerAttr)


@irdl_op_definition
class ReinterpretScaleOp(_UnaryCiphertextOp):
    r"""Replace ciphertext scale metadata while preserving all residues.

    The recorded actual scale becomes the supplied $\Delta'$, but ciphertext
    components, prime rows, level, polynomial domain, and Montgomery state do not
    change.  The represented decoded value is consequently interpreted relative
    to $\Delta'$.  This operation performs no multiplication, rescaling, or
    modulus switch."""

    name = "fhelium_ckks.reinterpret_scale"
    scale = attr_def(FloatAttr)


class _PrepareOp(IRDLOperation):
    public = operand_def()
    ciphertext = operand_def(CiphertextType)
    result = result_def(PlaintextType)
    operation = opt_attr_def(StringAttr)
    source_role = opt_attr_def(StringAttr)
    scale_mode = opt_attr_def(StringAttr)


@irdl_op_definition
class PrepareAddMessageOp(_PrepareOp):
    r"""Encode a public message for ciphertext addition.

    The message is encoded at the ciphertext level and actual scale, reduced into
    the ciphertext's Q or QP rows, and converted to coefficient-domain Montgomery
    residues.  The prepared polynomial can then be added to component zero by
    ``AddPlaintextOp`` without changing ciphertext level or scale."""

    name = "fhelium_ckks.prepare.add.message"


@irdl_op_definition
class PrepareAddPlaintextOp(_PrepareOp):
    r"""Convert a caller-owned plaintext to addition-ready RNS state.

    The plaintext already supplies its actual scale and level; both must match the
    ciphertext.  Encoding or residue conversion as needed produces
    coefficient-domain Montgomery rows in the ciphertext's Q or QP basis.  No
    message arithmetic is performed before ``AddPlaintextOp``."""

    name = "fhelium_ckks.prepare.add.plaintext"


@irdl_op_definition
class PrepareAddStaticOp(_PrepareOp):
    r"""Encode a statically known public value for ciphertext addition.

    The value is encoded at the ciphertext actual scale and level, reduced to the
    same Q or QP rows, and converted to coefficient-domain Montgomery form.  The
    result has the state required by ``AddPlaintextOp``."""

    name = "fhelium_ckks.prepare.add.static"


@irdl_op_definition
class PrepareMultiplyMessageOp(_PrepareOp):
    r"""Encode a public message for ciphertext multiplication.

    The message is encoded at the selected default actual scale, placed at the
    ciphertext level, reduced into matching Q or QP rows, and transformed to
    NTT/Montgomery form.  ``MultiplyPlaintextOp`` then produces scale
    $\Delta_c\Delta_p$."""

    name = "fhelium_ckks.prepare.multiply.message"


@irdl_op_definition
class PrepareMultiplyPlaintextOp(_PrepareOp):
    r"""Convert a caller-owned plaintext to multiplication-ready state.

    The plaintext retains its own actual scale and must share the ciphertext level
    and Q or QP rows.  Encoding, residue conversion, and a forward NTT as needed
    produce NTT/Montgomery data for ``MultiplyPlaintextOp``.  Preparation itself
    does not multiply or rescale."""

    name = "fhelium_ckks.prepare.multiply.plaintext"


@irdl_op_definition
class PrepareMultiplyStaticOp(_PrepareOp):
    r"""Encode a statically known public value for ciphertext multiplication.

    The value is encoded at the selected default actual scale and ciphertext
    level, reduced into matching Q or QP rows, and transformed to NTT/Montgomery
    form.  Its scale later multiplies the ciphertext scale in
    ``MultiplyPlaintextOp``."""

    name = "fhelium_ckks.prepare.multiply.static"


def _with_ciphertext_components(
    validator: OperationValidator,
    *,
    operands: tuple[tuple[int, int], ...] = (),
    results: tuple[tuple[int, int], ...] = (),
    every_result: int | None = None,
) -> OperationValidator:
    """Add represented ciphertext-component requirements to a validator."""

    def component_diagnostic(
        value: object,
        *,
        expected: int,
        label: str,
    ) -> str | None:
        value_type = getattr(value, "type", None)
        if not isinstance(value_type, CiphertextType):
            return None
        represented = value_type.state.data.get("components")
        if represented is None:
            return (
                f"{label} requires represented ciphertext components={expected}"
            )
        if not isinstance(represented, IntegerAttr):
            return f"{label} requires integer ciphertext components={expected}"
        actual = int(represented.value.data)
        if actual != expected:
            return (
                f"{label} requires ciphertext components={expected}, "
                f"got {actual}"
            )
        return None

    def validate(operation: Operation) -> tuple[str, ...]:
        diagnostics = list(validator(operation))
        for index, expected in operands:
            if index >= len(operation.operands):
                continue
            diagnostic = component_diagnostic(
                operation.operands[index],
                expected=expected,
                label=f"operand {index}",
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        expected_results = results
        if every_result is not None:
            expected_results = tuple(
                (index, every_result) for index in range(len(operation.results))
            )
        for index, expected in expected_results:
            if index >= len(operation.results):
                continue
            diagnostic = component_diagnostic(
                operation.results[index],
                expected=expected,
                label=f"result {index}",
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        return tuple(diagnostics)

    return validate


def _rotate_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(operation, ())
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    return tuple(diagnostics)


def _hoisted_rotate_many_specification(
    operation: Operation,
) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(operation, ())
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    if not isinstance(operation, RotateManyOp):
        diagnostics.append(
            "requires the registered CKKS hoisted-rotation operation"
        )
    return tuple(diagnostics)


def _prepare_specification(
    action: str,
    source_role: str,
) -> OperationValidator:
    def validate(operation: Operation) -> tuple[str, ...]:
        diagnostics = list(flat_operation(operation))
        unsupported = unsupported_attributes(
            operation,
            ("operation", "source_role", "scale_mode"),
        )
        if unsupported:
            diagnostics.append(f"unsupported attributes {list(unsupported)}")
        if required_string_attribute(operation, "operation") != action:
            diagnostics.append(f"requires operation={action!r}")
        if required_string_attribute(operation, "source_role") != source_role:
            diagnostics.append(f"requires source_role={source_role!r}")
        mode = required_string_attribute(operation, "scale_mode")
        allowed = (
            {"ciphertext_scale"}
            if action == "add"
            else (
                {"runtime_plaintext_scale"}
                if source_role == "plaintext"
                else {"default_scale"}
            )
        )
        if mode not in allowed:
            diagnostics.append(f"scale_mode must be one of {sorted(allowed)}")
        return tuple(diagnostics)

    return validate


def _rescale_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(operation, ("rounding",))
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    rounding = required_string_attribute(operation, "rounding")
    if rounding not in {"nearest", "floor"}:
        diagnostics.append(
            "rescale requires represented rounding='nearest' or 'floor'"
        )
    expected_roles = ("encrypted",)
    actual_roles = tuple(
        (
            "encrypted"
            if isinstance(operand.type, CiphertextType)
            else "plaintext"
            if isinstance(operand.type, PlaintextType)
            else None
        )
        for operand in operation.operands
    )
    if actual_roles != expected_roles:
        diagnostics.append(
            f"rescale requires operand roles {expected_roles}, got {actual_roles}"
        )
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = (
    registered_operation_spec(
        EncodeOp,
        "ckks",
        effect="rng-write",
        validator=flat_with_attributes("level", "scale"),
    ),
    registered_operation_spec(
        DecodeOp,
        "ckks",
        validator=flat_with_attributes("is_real"),
    ),
    registered_operation_spec(
        IntegerCoefficientsToRnsOp,
        "ckks",
        validator=flat_with_attributes("modulus_basis", "level"),
    ),
    registered_operation_spec(
        EncryptOp,
        "ckks",
        effect="rng-write",
        validator=flat_with_attributes("key_symbol"),
    ),
    registered_operation_spec(
        DecryptOp,
        "ckks",
        validator=flat_with_attributes("key_symbol"),
    ),
    *(
        registered_operation_spec(op, "ckks", validator=flat_operation)
        for op in (
            NegateOp,
            ToNttOp,
            FromNttOp,
            ToMontgomeryResiduesOp,
            ToStandardResiduesOp,
            AddOp,
            SubtractOp,
            AddPlaintextOp,
            MultiplyPlaintextOp,
        )
    ),
    registered_operation_spec(
        RotateOp,
        "ckks",
        validator=_with_ciphertext_components(
            _rotate_specification,
            operands=((0, 2),),
            results=((0, 2),),
        ),
    ),
    registered_operation_spec(
        RotateManyOp,
        "ckks",
        validator=_with_ciphertext_components(
            _hoisted_rotate_many_specification,
            operands=((0, 2),),
            every_result=2,
        ),
    ),
    registered_operation_spec(
        MultiplyOp,
        "ckks",
        validator=_with_ciphertext_components(
            flat_operation,
            operands=((0, 2), (1, 2)),
            results=((0, 3),),
        ),
    ),
    registered_operation_spec(
        AddCompressedPlaintextOp,
        "ckks",
        validator=flat_with_attributes("inplace"),
    ),
    registered_operation_spec(
        MultiplyCompressedPlaintextOp,
        "ckks",
        validator=flat_with_attributes("inplace"),
    ),
    registered_operation_spec(
        RelinearizeOp,
        "ckks",
        validator=_with_ciphertext_components(
            flat_operation,
            operands=((0, 3),),
            results=((0, 2),),
        ),
    ),
    registered_operation_spec(
        SwitchKeyOp,
        "ckks",
        validator=_with_ciphertext_components(
            flat_with_attributes("key_symbol"),
            operands=((0, 2),),
            results=((0, 2),),
        ),
    ),
    registered_operation_spec(
        ConjugateOp,
        "ckks",
        validator=_with_ciphertext_components(
            flat_operation,
            operands=((0, 2),),
            results=((0, 2),),
        ),
    ),
    registered_operation_spec(
        RescaleOp,
        "ckks",
        validator=_rescale_specification,
    ),
    registered_operation_spec(
        ModSwitchOp,
        "ckks",
        validator=flat_with_attributes("target_level"),
    ),
    registered_operation_spec(
        ReinterpretScaleOp,
        "ckks",
        validator=flat_with_attributes("scale"),
    ),
    *(
        registered_operation_spec(
            operation_type,
            "ckks",
            effect="rng-write",
            validator=_prepare_specification(operation, role),
        )
        for operation, role, operation_type in (
            ("add", "message", PrepareAddMessageOp),
            ("add", "plaintext", PrepareAddPlaintextOp),
            ("add", "static", PrepareAddStaticOp),
            ("multiply", "message", PrepareMultiplyMessageOp),
            ("multiply", "plaintext", PrepareMultiplyPlaintextOp),
            ("multiply", "static", PrepareMultiplyStaticOp),
        )
    ),
    *(
        registered_operation_spec(
            operation_type,
            "ckks",
            effect="rng-write",
            validator=flat_with_attributes("scalar", "scalar_scale"),
        )
        for operation_type in (AddScalarOp, MultiplyScalarOp)
    ),
    registered_operation_spec(
        MultiplyIntegerScalarOp,
        "ckks",
        validator=flat_with_attributes("scalar"),
    ),
)
"""Semantic specifications owned by the CKKS dialect."""

FHEliumCkks = Dialect(
    "fhelium_ckks",
    [
        EncodeOp,
        DecodeOp,
        IntegerCoefficientsToRnsOp,
        EncryptOp,
        DecryptOp,
        NegateOp,
        RotateOp,
        RotateManyOp,
        ToNttOp,
        FromNttOp,
        ToMontgomeryResiduesOp,
        ToStandardResiduesOp,
        AddOp,
        SubtractOp,
        MultiplyOp,
        AddScalarOp,
        MultiplyScalarOp,
        MultiplyIntegerScalarOp,
        AddPlaintextOp,
        MultiplyPlaintextOp,
        AddCompressedPlaintextOp,
        MultiplyCompressedPlaintextOp,
        RelinearizeOp,
        SwitchKeyOp,
        ConjugateOp,
        RescaleOp,
        ModSwitchOp,
        ReinterpretScaleOp,
        PrepareAddMessageOp,
        PrepareAddPlaintextOp,
        PrepareAddStaticOp,
        PrepareMultiplyMessageOp,
        PrepareMultiplyPlaintextOp,
        PrepareMultiplyStaticOp,
    ],
    [
        CiphertextType,
        PlaintextType,
        CompressedPlaintextType,
        EvaluationKeyType,
    ],
)
"""CKKS operation and representation dialect."""


__all__ = [
    "AddOp",
    "AddScalarOp",
    "AddPlaintextOp",
    "AddCompressedPlaintextOp",
    "CiphertextType",
    "CompressedPlaintextType",
    "ConjugateOp",
    "DecodeOp",
    "DecryptOp",
    "EncodeOp",
    "EncryptOp",
    "EvaluationKeyType",
    "FHEliumCkks",
    "FromNttOp",
    "ModSwitchOp",
    "MultiplyOp",
    "MultiplyIntegerScalarOp",
    "MultiplyScalarOp",
    "MultiplyPlaintextOp",
    "MultiplyCompressedPlaintextOp",
    "NegateOp",
    "PlaintextType",
    "OPERATION_SPECS",
    "PrepareAddMessageOp",
    "PrepareAddPlaintextOp",
    "PrepareAddStaticOp",
    "PrepareMultiplyMessageOp",
    "PrepareMultiplyPlaintextOp",
    "PrepareMultiplyStaticOp",
    "IntegerCoefficientsToRnsOp",
    "RelinearizeOp",
    "ReinterpretScaleOp",
    "RescaleOp",
    "RotateOp",
    "RotateManyOp",
    "SubtractOp",
    "SwitchKeyOp",
    "ToNttOp",
    "ToMontgomeryResiduesOp",
    "ToStandardResiduesOp",
]
