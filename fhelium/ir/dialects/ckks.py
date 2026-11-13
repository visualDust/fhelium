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

from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal, cast

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
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
from ..ckks_state import output_residues, transform_state
from ._common import OpenStateType, ValueRole
from .core import MessageType
from .._dependencies import (
    DependencyKind,
    OperationDependencies,
    ValueDependency,
    operand_dependencies,
    operand_relations,
    string_fact,
)


@irdl_attr_definition
class CiphertextType(OpenStateType):
    """CKKS ciphertext with partial depth, scale, basis, and domain state."""

    name = "fhelium_ckks.ciphertext"
    ROLE: ClassVar[ValueRole] = "encrypted"


@irdl_attr_definition
class PlaintextType(OpenStateType):
    """CKKS plaintext with partial encoding and RNS representation state."""

    name = "fhelium_ckks.plaintext"
    ROLE: ClassVar[ValueRole] = "plaintext"


@irdl_attr_definition
class EvaluationKeyType(OpenStateType):
    """Evaluation-key Tensor with partial digit, row, and polynomial state."""

    name = "fhelium_ckks.evaluation_key"


@irdl_attr_definition
class CompressedPlaintextType(OpenStateType):
    """Represent operation-ready compressed RNS plaintext storage."""

    name = "fhelium_ckks.compressed_plaintext"

    ROLE: ClassVar[ValueRole] = "plaintext"


@irdl_op_definition
class PrepareCompressedPlaintextOp(IRDLOperation):
    r"""Prepare compressed NTT rows from a periodic slot message.

    For U=2r (at least four for generator 3), R=N/U and a period-r message,
    the encoded polynomial has the form p(X)=a(X^R). The operation performs
    the U-point inverse embedding, stochastic rounding at the corresponding
    full-ring random-stream positions, RNS reduction and a U-point NTT with
    roots psi_N^R for NTT output. Coefficient output omits the transform and
    returns strided-sparse compact data plus zero implicit rows; NTT output
    uses contiguous repetition. Both use Montgomery residues and preserve
    the supplied depth, prime rows and actual scale.
    """

    name = "fhelium_ckks.prepare_compressed_plaintext"
    message = operand_def(MessageType)
    parameters = var_operand_def()
    results_ = var_result_def()
    traits = traits_def()

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self, {0: {'slot': 'mixing', 'batch': 'element'}}
        )


@irdl_op_definition
class EncodeOp(IRDLOperation):
    r"""Encode ordered complex slots as a scaled integer polynomial.

    For ring $R=\mathbb{Z}[X]/(X^N+1)$, let $\sigma$ be the configured CKKS
    embedding in the configured generator's slot order.  Given slots $m$ and
    actual scale $\Delta$, the operation returns
    $a=\operatorname{RandRound}(\Delta\,\sigma^{-1}(m))\in R$.
    ``depth`` records the later modulus-chain placement but does not reduce the
    integer coefficients.  The registered ``native-ckks-encode`` implementation
    performs the embedding and stochastic rounding."""

    name = "fhelium_ckks.encode"
    message = operand_def(MessageType)
    parameters = var_operand_def()
    result = result_def(PlaintextType)
    depth = attr_def(IntegerAttr)
    scale = attr_def(FloatAttr)

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self, {0: {'slot': 'mixing', 'batch': 'element'}}
        )


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
    parameters = var_operand_def()
    result = result_def(MessageType)
    is_real = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self, {0: {'coefficient': 'mixing', 'batch': 'element'}}
        )


@irdl_op_definition
class IntegerCoefficientsToRnsOp(IRDLOperation):
    r"""Reduce integer polynomial coefficients into active RNS rows.

    RNS means residue number system.  At ``depth`` $\ell$, every coefficient
    $a_j$ is mapped to $a_j\bmod q_i$ for each active Q prime and, for basis
    ``QP``, to $a_j\bmod p_i$ for each special P prime.  The result is
    coefficient-domain standard-residue data.  Scale and polynomial meaning do
    not change."""

    name = "fhelium_ckks.integer_coefficients_to_rns"
    plaintext = operand_def(PlaintextType)
    parameters = var_operand_def()
    result = result_def(PlaintextType)
    modulus_basis = attr_def(StringAttr)
    depth = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self, {0: {'coefficient': 'element', 'batch': 'element'}}
        )


@irdl_op_definition
class EncryptOp(IRDLOperation):
    r"""Encrypt an integer plaintext polynomial as a CT2 ciphertext.

    Let the public key be $(k_0,k_1)$, satisfying
    $k_0+k_1s\approx0$, let $v$ have independent ZO(0.5) coefficients
    (probabilities 1/4, 1/2, 1/4 for -1, 0, +1), and let
    $e_0,e_1$ be sampled error polynomials.  For plaintext $a$, the result is
    $c_0=k_0v+a+e_0$ and $c_1=k_1v+e_1$ modulo every active prime.  The output
    has two components and keeps the plaintext depth and actual scale.
    ``output_domain`` selects coefficient/standard or NTT/Montgomery Q or QP
    residues. The second operand supplies the public-key Tensor."""

    name = "fhelium_ckks.encrypt"
    plaintext = operand_def(PlaintextType)
    key = operand_def()
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )

    def dependencies(self) -> OperationDependencies:
        """Describe plaintext addition and multiplication by the public key."""
        domain = string_fact(self, "output_domain")
        position: DependencyKind = (
            "element"
            if domain == "coefficient"
            else "mixing"
            if domain == "ntt"
            else "unknown"
        )
        return OperationDependencies(
            (
                ValueDependency(
                    0, 0, {"coefficient": position, "batch": "element"}
                ),
                ValueDependency(
                    0,
                    1,
                    {
                        "coefficient": "mixing",
                        "limb": "reindexed",
                        "component": "element",
                        "batch": "reindexed",
                    },
                ),
            )
        )


@irdl_op_definition
class DecryptOp(IRDLOperation):
    r"""Evaluate a ciphertext phase and reconstruct coefficient data.

    For CT2 $(c_0,c_1)$, the phase is $c_0+c_1s$; for CT3 it is
    $c_0+c_1s+c_2s^2$, computed modulo each active prime.  The registered
    decryption implementation converts products to coefficient-domain standard
    RNS, then reconstructs the centered integer class from all active Q
    rows.  The result is an approximate coefficient plaintext at the ciphertext's
    depth and actual scale. ``input_domain`` records whether the ciphertext
    payload arrives in coefficient/standard or NTT/Montgomery form, and the second operand supplies the secret-key Tensor."""

    name = "fhelium_ckks.decrypt"
    ciphertext = operand_def(CiphertextType)
    key = operand_def()
    parameters = var_operand_def()
    result = result_def(PlaintextType)
    input_domain = attr_def(StringAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'mixing',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'reindexed',
                    'batch': 'reindexed',
                },
            },
        )


class _UnaryCiphertextOp(IRDLOperation):
    value = operand_def(CiphertextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        parameters: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type or SSAValue.get(value).type],
            attributes=attributes,
        )


class _BinaryCiphertextOp(IRDLOperation):
    lhs = operand_def(CiphertextType)
    rhs = operand_def(CiphertextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        parameters: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type or SSAValue.get(lhs).type],
            attributes=attributes,
        )


@irdl_op_definition
class NegateOp(_UnaryCiphertextOp):
    r"""Negate every component of a CKKS ciphertext.

    For $c=(c_0,\ldots,c_{k-1})$, the result is
    $(-c_0,\ldots,-c_{k-1})$ modulo each active prime and therefore decrypts to
    the negated message.  Component count, depth, actual scale, prime rows,
    polynomial domain, and Montgomery state are unchanged."""

    name = "fhelium_ckks.negate"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class RotateOp(IRDLOperation):
    r"""Rotate CKKS slots through a Galois automorphism and key switch.

    The input is a CT2 ciphertext in coefficient/standard or NTT/Montgomery
    Q state. A rotation key records a signed slot displacement $r$ and its odd
    Galois element $g$.
    Applying $\sigma_g:X\mapsto X^g$ rotates the encoded slots and changes the
    secret relation from $s$ to $\sigma_g(s)$; the key-switch stage returns
    that relation to $s$.  The result is a CT2 ciphertext
    with depth, actual scale, and active Q rows preserved. ``output_domain``
    selects coefficient/standard or NTT/Montgomery output; the NTT choice
    equals a coefficient rotation followed by ``ToNttOp``.

    ``moddown_rounding`` selects $\lfloor(x+(P-1)/2)/P\rfloor$
    (nearest, default) or $\lfloor x/P\rfloor$ for each key-switch
    accumulator coefficient."""

    name = "fhelium_ckks.rotate"
    value = operand_def(CiphertextType)
    key = operand_def(EvaluationKeyType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )
    input_domain = attr_def(StringAttr, default_value=StringAttr("coefficient"))
    moddown_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    modup_method = opt_attr_def(StringAttr, default_value=StringAttr("exact"))
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'mixing',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'mixing',
                    'batch': 'reindexed',
                },
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        key: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        moddown_rounding: str = "nearest",
        modup_method: str = "exact",
        parameters: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.setdefault("moddown_rounding", StringAttr(moddown_rounding))
        attrs.setdefault("modup_method", StringAttr(modup_method))
        if "input_domain" not in attrs:
            source_type = cast(CiphertextType, SSAValue.get(value).type)
            represented = source_type.state.data.get("polynomial_domain")
            attrs["input_domain"] = (
                represented
                if isinstance(represented, StringAttr)
                else StringAttr("coefficient")
            )
        if result_type is None:
            source_type = cast(CiphertextType, SSAValue.get(value).type)
            domain = attrs.get("output_domain", StringAttr("coefficient"))
            result_type = source_type.with_state(
                {
                    "polynomial_domain": domain,
                    "residue_representation": StringAttr(
                        "montgomery"
                        if domain == StringAttr("ntt")
                        else "standard"
                    ),
                }
            )
        super().__init__(
            operands=[value, key, parameters],
            result_types=[result_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        if self.modup_method.data not in {"exact", "crt", "normalized_crt"}:
            raise VerifyException(
                "modup_method must be exact, crt, or normalized_crt"
            )
        if self.moddown_rounding.data not in {"floor", "nearest"}:
            raise VerifyException(
                "moddown_rounding must be 'floor' or 'nearest'"
            )


@irdl_op_definition
class RotateManyOp(IRDLOperation):
    r"""Produce several slot rotations from one shared key-switch preparation.

    The input is a coefficient-domain standard-Q CT2 ciphertext.  Each key operand
    defines one Galois automorphism and signed slot displacement.  For every key
    the mathematical result equals an independent ``RotateOp`` of the same input.
    Hoisting shares hybrid digit decomposition and Q-to-QP basis-extension work
    across outputs but does not change their order or values.  Every result
    preserves the input depth, actual scale, and active Q rows.
    ``output_domain`` selects coefficient/standard (the default) or
    NTT/Montgomery output. The latter equals applying ``ToNttOp`` to each
    coefficient result and permits ModDown to retain Q evaluations.

    ``moddown_rounding`` selects $\lfloor(x+(P-1)/2)/P\rfloor$
    (nearest, default) or $\lfloor x/P\rfloor$ for each key-switch
    accumulator coefficient."""

    name = "fhelium_ckks.hoisted_rotate_many"
    value = operand_def(CiphertextType)
    keys = var_operand_def(EvaluationKeyType)
    parameters = var_operand_def()
    outputs = var_result_def(CiphertextType)
    irdl_options = (AttrSizedOperandSegments(as_property=True),)
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )
    moddown_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    modup_method = opt_attr_def(StringAttr, default_value=StringAttr("exact"))
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe each rotation's reads from the ciphertext and its key."""
        return OperationDependencies(
            tuple(
                ValueDependency(
                    result,
                    operand,
                    {
                        "coefficient": "mixing",
                        "limb": "mixing",
                        "component": "mixing",
                        "batch": "element" if operand == 0 else "reindexed",
                    },
                )
                for result in range(len(self.results))
                for operand in (0, 1 + result)
            )
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        keys: tuple[SSAValue | Operation, ...],
        result_types: tuple[CiphertextType, ...],
        *,
        output_domain: Literal["coefficient", "ntt"] = "coefficient",
        moddown_rounding: str = "nearest",
        modup_method: str = "exact",
        parameters: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=(value, keys, parameters),
            result_types=(result_types,),
            attributes={
                "modup_method": StringAttr(modup_method),
                "moddown_rounding": StringAttr(moddown_rounding),
                **(attributes or {}),
                "output_domain": StringAttr(output_domain),
            },
        )

    def verify_(self) -> None:
        if self.modup_method.data not in {"exact", "crt", "normalized_crt"}:
            raise VerifyException(
                "modup_method must be exact, crt, or normalized_crt"
            )
        if self.moddown_rounding.data not in {"floor", "nearest"}:
            raise VerifyException(
                "moddown_rounding must be 'floor' or 'nearest'"
            )
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


@irdl_op_definition
class ModulusRaiseOp(IRDLOperation):
    r"""Centered-extend each CT2 component from its Q basis to an earlier depth.

    Input and output are coefficient/standard Q residues. Component count,
    batch shape and actual scale are preserved; output prime rows and depth
    belong to target_depth. The lifted decryption phase contains the source
    modulus times an integer polynomial, consumed by Bootstrap's periodic map.
    """

    name = "fhelium_ckks.modulus_raise"
    value = operand_def(CiphertextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    target_depth = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value,
        result_type,
        *,
        target_depth=0,
        parameters=(),
        attributes=None,
    ):
        super().__init__(
            operands=(value, parameters),
            result_types=(result_type,),
            attributes={
                **(attributes or {}),
                "target_depth": IntegerAttr(target_depth, 64),
            },
        )

    def verify_(self) -> None:
        diagnostics = _modulus_raise_specification(self)
        if diagnostics:
            raise VerifyException("; ".join(diagnostics))


@irdl_op_definition
class EncapsulatedModulusRaiseOp(IRDLOperation):
    r"""Switch to a low secret relation, centered-raise and return to the source key.

    The low hybrid key encrypts P_low*E_j*s under t. The one-digit return key
    encrypts guard*P_return*t under s on [guard,Q,P_return]. Execution applies
    the low key switch, centered extension, then one return key product and
    quotient by guard*P_return. Output is CT2 coefficient/standard Q at
    target_depth; actual scale and batch shape are preserved. Keys and all
    numerical parameters are ordinary live operands.
    """

    name = "fhelium_ckks.encapsulated_modulus_raise"
    value = operand_def(CiphertextType)
    low_key = operand_def(EvaluationKeyType)
    return_key = operand_def(EvaluationKeyType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    target_depth = attr_def(IntegerAttr)
    low_p_moduli = attr_def(ArrayAttr[IntegerAttr])
    return_prime = attr_def(IntegerAttr)
    return_p_moduli = attr_def(ArrayAttr[IntegerAttr])
    moddown_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    traits = traits_def(Pure())

    def __init__(
        self,
        value,
        low_key,
        return_key,
        result_type,
        *,
        target_depth=0,
        low_p_moduli,
        return_prime,
        return_p_moduli,
        rounding="nearest",
        parameters=(),
        attributes=None,
    ):
        super().__init__(
            operands=(value, low_key, return_key, parameters),
            result_types=(result_type,),
            attributes={
                **(attributes or {}),
                "target_depth": IntegerAttr(target_depth, 64),
                "low_p_moduli": ArrayAttr(
                    IntegerAttr(q, 64) for q in low_p_moduli
                ),
                "return_prime": IntegerAttr(return_prime, 64),
                "return_p_moduli": ArrayAttr(
                    IntegerAttr(q, 64) for q in return_p_moduli
                ),
                "moddown_rounding": StringAttr(rounding),
            },
        )

    def verify_(self) -> None:
        diagnostics = _modulus_raise_specification(self)
        if diagnostics:
            raise VerifyException("; ".join(diagnostics))


@irdl_op_definition
class RotationSumOp(IRDLOperation):
    r"""Sum rotated ciphertext relations with one QP quotient.

    Each source is CT2 over one common Q basis and actual scale. Nonzero
    rotations contribute hybrid key products under the common destination
    relation. Ordinary components contribute P times their Q residues and
    zero P residues. The shared extended-basis sum is divided once by P,
    or by P times the leading complete Q group when ``rescale`` is one.
    This changes rounding relative to independent RotateOp results and is
    selected through a caller-requested numerical rewrite.

    Output is CT2 in coefficient/standard Q form. Without rescaling, depth,
    prime rows and actual scale are preserved. With rescaling, depth advances
    by one, the complete leading Q group is removed, and actual scale becomes
    $\Delta_{out}=\Delta_{in}/\prod_{q\in G_d}q$.
    """

    name = "fhelium_ckks.rotation_sum"
    values = var_operand_def(CiphertextType)
    keys = var_operand_def(EvaluationKeyType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    irdl_options = (AttrSizedOperandSegments(as_property=True),)
    rotation_steps = attr_def(ArrayAttr[IntegerAttr])
    source_count = attr_def(IntegerAttr)
    rescale = attr_def(IntegerAttr)
    input_domain = attr_def(StringAttr)
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )
    moddown_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    modup_method = opt_attr_def(StringAttr, default_value=StringAttr("exact"))
    traits = traits_def(Pure())

    def __init__(
        self,
        values,
        keys,
        steps,
        result_type,
        *,
        rescale=False,
        input_domain="coefficient",
        rounding="nearest",
        modup_method="exact",
        parameters=(),
        attributes=None,
    ):
        super().__init__(
            operands=(values, keys, parameters),
            result_types=(result_type,),
            attributes={
                "modup_method": StringAttr(modup_method),
                **(attributes or {}),
                "rotation_steps": ArrayAttr(
                    IntegerAttr(int(step), 64) for step in steps
                ),
                "source_count": IntegerAttr(len(values), 64),
                "rescale": IntegerAttr(int(rescale), 64),
                "input_domain": StringAttr(input_domain),
                "output_domain": StringAttr("coefficient"),
                "moddown_rounding": StringAttr(rounding),
            },
        )

    def dependencies(self) -> OperationDependencies:
        return operand_relations(
            self,
            {
                i: {
                    "coefficient": "mixing",
                    "limb": "mixing",
                    "component": "mixing",
                    "batch": "element" if i < len(self.values) else "reindexed",
                }
                for i in range(len(self.values) + len(self.keys))
            },
        )

    def verify_(self) -> None:
        if self.modup_method.data not in {"exact", "crt", "normalized_crt"}:
            raise VerifyException(
                "modup_method must be exact, crt, or normalized_crt"
            )
        steps = tuple(int(x.value.data) for x in self.rotation_steps)
        if self.source_count.value.data != len(self.values):
            raise VerifyException(
                "Rotation-sum source_count must match its operands"
            )
        if self.rescale.value.data not in {0, 1}:
            raise VerifyException("Rotation-sum rescale must be zero or one")
        if not self.values or len(steps) != len(self.values):
            raise VerifyException(
                "Rotation-sum sources and steps must have equal nonzero lengths"
            )
        if len(self.keys) != sum(step != 0 for step in steps):
            raise VerifyException(
                "Rotation sum requires one key for each nonzero step"
            )
        if self.moddown_rounding.data not in {"floor", "nearest"}:
            raise VerifyException(
                "Rotation-sum rounding must be floor or nearest"
            )


@irdl_op_definition
class GroupedRotationWeightedSumOp(IRDLOperation):
    r"""Apply plaintext rows to one caller-supplied direct rotation group.

    For baby steps $b_t$, ciphertext $c$, and a group-major matrix of prepared
    plaintexts $p_{g,t}$, the result batch is

    $$
    y_g=\sum_{t=0}^{T-1}p_{g,t}\operatorname{Rot}_{b_t}(c).
    $$

    ``baby_steps`` lists $T$ distinct steps including at most one zero. Each
    nonzero step has one matching key operand; zero denotes the unrotated
    ciphertext. ``group_count`` and ``term_count`` describe the plaintext
    matrix. Every rotation completes its own hybrid ModDown before plaintext
    multiplication, preserving ordinary key-switch rounding. The output adds
    the group as the first batch axis and is NTT/Montgomery Q at the input
    depth with scale $\Delta_c\Delta_p$.


    ``moddown_rounding`` selects $\lfloor(x+(P-1)/2)/P\rfloor$
    (nearest, default) or $\lfloor x/P\rfloor$ for each key-switch
    accumulator coefficient."""

    name = "fhelium_ckks.grouped_rotation_weighted_sum"
    value = operand_def(CiphertextType)
    keys = var_operand_def(EvaluationKeyType)
    plaintexts = var_operand_def(PlaintextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    baby_steps = attr_def(ArrayAttr[IntegerAttr])
    term_count = attr_def(IntegerAttr)
    group_count = attr_def(IntegerAttr)
    moddown_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    modup_method = opt_attr_def(StringAttr, default_value=StringAttr("exact"))
    traits = traits_def(Pure())
    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def dependencies(self) -> OperationDependencies:
        """Describe the rotated input, evaluation keys and plaintext weights."""
        count = 1 + len(self.keys) + len(self.plaintexts)
        return OperationDependencies(
            tuple(
                ValueDependency(
                    0,
                    index,
                    {
                        "coefficient": "mixing"
                        if index <= len(self.keys)
                        else "element",
                        "limb": "mixing"
                        if index <= len(self.keys)
                        else "element",
                        "component": "mixing"
                        if index <= len(self.keys)
                        else "reindexed",
                        "batch": "reindexed",
                    },
                )
                for index in range(count)
            )
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        keys: tuple[SSAValue | Operation, ...],
        plaintexts: tuple[SSAValue | Operation, ...],
        result_type: CiphertextType,
        *,
        baby_steps: tuple[int, ...],
        group_count: int,
        moddown_rounding: str = "nearest",
        modup_method: str = "exact",
        parameters: Sequence[SSAValue | Operation] = (),
    ) -> None:
        super().__init__(
            operands=(value, keys, plaintexts, parameters),
            result_types=[result_type],
            attributes={
                "modup_method": StringAttr(modup_method),
                "moddown_rounding": StringAttr(moddown_rounding),
                "baby_steps": ArrayAttr(
                    IntegerAttr(step, 64) for step in baby_steps
                ),
                "term_count": IntegerAttr(len(baby_steps), 64),
                "group_count": IntegerAttr(group_count, 64),
            },
        )

    def verify_(self) -> None:
        if self.modup_method.data not in {"exact", "crt", "normalized_crt"}:
            raise VerifyException(
                "modup_method must be exact, crt, or normalized_crt"
            )
        if self.moddown_rounding.data not in {"floor", "nearest"}:
            raise VerifyException(
                "moddown_rounding must be 'floor' or 'nearest'"
            )
        term_count = int(self.term_count.value.data)
        group_count = int(self.group_count.value.data)
        steps = tuple(int(step.value.data) for step in self.baby_steps)
        if term_count <= 0 or group_count <= 0 or len(steps) != term_count:
            raise VerifyException(
                "CKKS grouped rotation weighted sum requires positive counts "
                "and one baby step per term"
            )
        if len(set(steps)) != len(steps) or steps.count(0) > 1:
            raise VerifyException(
                "CKKS grouped rotation weighted sum requires distinct baby steps"
            )
        if len(self.keys) != term_count - steps.count(0):
            raise VerifyException(
                "CKKS grouped rotation weighted sum requires one key per "
                "nonzero baby step"
            )
        if len(self.plaintexts) != term_count * group_count:
            raise VerifyException(
                "CKKS grouped rotation weighted sum plaintext matrix size differs"
            )


@irdl_op_definition
class SumBatchOp(IRDLOperation):
    r"""Reduce one ciphertext batch axis by modular addition.

    For components $c_j^{(k)}$ over the selected axis of length $n$, the
    result is $c'_j=\sum_{k<n}c_j^{(k)}\bmod Q$ with every partial sum
    evaluated modulo the active primes.  The operation removes the selected
    logical batch axis from the value shape; depth, actual scale, prime rows,
    component count, polynomial domain, and residue representation are
    unchanged."""

    name = "fhelium_ckks.sum_batch"
    value = operand_def(CiphertextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    axis = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'mixing',
                },
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute,
        *,
        axis: int | IntegerAttr,
        parameters: Sequence[SSAValue | Operation] = (),
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes={
                "axis": (
                    IntegerAttr(axis, 64) if isinstance(axis, int) else axis
                )
            },
        )


class _RepresentationOp(IRDLOperation):
    value = operand_def()
    parameters = var_operand_def()
    result = result_def()
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        parameters: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        source_type = SSAValue.get(value).type
        if result_type is None and isinstance(source_type, OpenStateType):
            updates: dict[str, Attribute] = {}
            if self.name in {"fhelium_ckks.to_ntt", "fhelium_ckks.from_ntt"}:
                domain, residues = transform_state(
                    "forward"
                    if self.name == "fhelium_ckks.to_ntt"
                    else "inverse",
                    ciphertext=isinstance(source_type, CiphertextType),
                )
                updates = {
                    "polynomial_domain": StringAttr(domain),
                    "residue_representation": StringAttr(residues),
                }
            elif self.name == "fhelium_ckks.to_montgomery_residues":
                updates = {"residue_representation": StringAttr("montgomery")}
            elif self.name == "fhelium_ckks.to_standard_residues":
                updates = {"residue_representation": StringAttr("standard")}
            if updates:
                result_type = source_type.with_state(updates)
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type or source_type],
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
    element, component count, prime rows, depth, basis, and actual scale remain
    unchanged.  Lowering selects the corresponding operation from the NTT
    dialect according to the input residue representation."""

    name = "fhelium_ckks.to_ntt"

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
class FromNttOp(_RepresentationOp):
    r"""Transform CKKS NTT/Montgomery polynomials to coefficient form.

    An inverse negacyclic transform is applied independently to every component
    and prime row.  Ciphertexts end in standard residues; plaintext lowering may
    retain Montgomery residues.  The represented ring element, component count,
    prime rows, depth, basis, and actual scale are unchanged."""

    name = "fhelium_ckks.from_ntt"

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
class ToMontgomeryResiduesOp(_RepresentationOp):
    r"""Multiply coefficient-domain plaintext residues by the Montgomery radix.

    For each active prime $q_i$, the stored row changes from $x_i$ to
    $x_iR_i\bmod q_i$.  This representation transition preserves the
    plaintext polynomial, Q or QP rows, depth, and actual scale."""

    name = "fhelium_ckks.to_montgomery_residues"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class ToStandardResiduesOp(_RepresentationOp):
    r"""Remove the Montgomery factor from coefficient-domain plaintext rows.

    For each active prime $q_i$, Montgomery reduction maps $x_iR_i$ to
    $x_i\bmod q_i$.  The plaintext polynomial, Q or QP rows, depth, and actual
    scale are preserved."""

    name = "fhelium_ckks.to_standard_residues"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class AddOp(_BinaryCiphertextOp):
    r"""Add two state-compatible ciphertexts component by component.

    For each component $j$, active prime $q_i$, and coefficient, the operation
    computes $c'_j=c^{(a)}_j+c^{(b)}_j\pmod {q_i}$.  Inputs have the same
    component count, depth, prime rows, polynomial domain, residue representation,
    and actual scale.  The result retains that state and represents the sum of the
    decoded slot values."""

    name = "fhelium_ckks.add"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
            },
        )


@irdl_op_definition
class SubtractOp(_BinaryCiphertextOp):
    r"""Subtract ciphertexts component by component.

    For each component $j$ and active prime $q_i$, the operation computes
    $c'_j=c^{(a)}_j-c^{(b)}_j\pmod {q_i}$.  Compatible depth, prime rows,
    representation, component count, and actual scale are preserved, and the
    result represents the slotwise difference."""

    name = "fhelium_ckks.subtract"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
            },
        )


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

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'mixing',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'mixing',
                    'batch': 'reindexed',
                },
            },
        )


class _RealScalarCiphertextOp(IRDLOperation):
    ciphertext = operand_def(CiphertextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    scalar = attr_def(FloatAttr)
    scalar_scale = attr_def(FloatAttr)

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        parameters: Sequence[SSAValue | Operation] = (),
        scalar: FloatAttr,
        scalar_scale: FloatAttr,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext, parameters],
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

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class MultiplyScalarOp(_RealScalarCiphertextOp):
    r"""Multiply all ciphertext components by a quantized real constant.

    For scalar $u$ at scalar scale $\delta$, the implementation samples
    $k=\operatorname{RandRound}(u\delta)$ and computes $c'_j=kc_j$ modulo
    every active prime.  The output actual scale is $\Delta\delta$ for input
    scale $\Delta$, so it represents multiplication by approximately $u$.
    Depth, component count, prime rows, and polynomial representation are
    preserved; rescaling is separate."""

    name = "fhelium_ckks.multiply_scalar"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class MultiplyIntegerScalarOp(IRDLOperation):
    r"""Multiply every ciphertext component by an integer scalar.

    For integer $k$, each polynomial becomes $c'_j=kc_j$ modulo every active
    prime.  Because $k$ is not a newly scaled encoding, the ciphertext actual
    scale, depth, component count, prime rows, polynomial domain, and Montgomery
    state remain unchanged."""

    name = "fhelium_ckks.multiply_integer_scalar"
    ciphertext = operand_def(CiphertextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    scalar = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        parameters: Sequence[SSAValue | Operation] = (),
        scalar: IntegerAttr,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext, parameters],
            result_types=[result_type or SSAValue.get(ciphertext).type],
            attributes={
                **({} if attributes is None else attributes),
                "scalar": scalar,
            },
        )


class _CiphertextPlaintextOp(IRDLOperation):
    ciphertext = operand_def(CiphertextType)
    plaintext = operand_def(PlaintextType)
    parameters = var_operand_def()
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    traits = traits_def(Pure())

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        plaintext: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        parameters: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext, plaintext, parameters],
            result_types=[result_type or SSAValue.get(ciphertext).type],
            attributes=attributes,
        )


@irdl_op_definition
class AddPlaintextOp(_CiphertextPlaintextOp):
    r"""Add a prepared plaintext polynomial to ciphertext component zero.

    For ciphertext $(c_0,\ldots,c_{k-1})$ and plaintext $p$, the result is
    $(c_0+p,c_1,\ldots,c_{k-1})$ modulo each active prime.  Matching depth,
    prime rows, and actual scale make this a slotwise message addition.  Component
    count and ciphertext representation remain unchanged."""

    name = "fhelium_ckks.add_plaintext"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'reindexed',
                },
            },
        )


@irdl_op_definition
class MultiplyPlaintextOp(_CiphertextPlaintextOp):
    r"""Multiply each ciphertext component by a prepared plaintext polynomial.

    For ciphertext components $c_j$ and plaintext $p$, the result components
    are $c'_j=c_jp$ in every active-prime negacyclic ring.  Operands use
    NTT/Montgomery form so execution is pointwise.  The depth and component count
    remain unchanged, while output actual scale is $\Delta_c\Delta_p$; no
    rescale is implicit."""

    name = "fhelium_ckks.multiply_plaintext"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'reindexed',
                },
            },
        )


class _CiphertextCompressedPlaintextOp(IRDLOperation):
    ciphertext = operand_def(CiphertextType)
    plaintext = operand_def(CompressedPlaintextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    inplace = opt_attr_def(IntegerAttr)


@irdl_op_definition
class AddCompressedPlaintextOp(_CiphertextCompressedPlaintextOp):
    r"""Add a compact plaintext to component zero without expanding its rows.

    Coefficient/standard ciphertext and coefficient/Montgomery plaintext give
    c'_0=c_0+REDC(pR). NTT/Montgomery operands give c'_0=c_0+pR. Other
    components, depth, scale, prime rows and representation are preserved.
    Sparse layouts supply a separate [*batch, limb, 1] implicit-value Tensor.
    """

    name = "fhelium_ckks.add_compressed_plaintext"

    def dependencies(self) -> OperationDependencies:
        """Describe direct ciphertext reads and compact plaintext expansion."""
        relations = [
            ValueDependency(
                0,
                0,
                {
                    "coefficient": "element",
                    "limb": "element",
                    "component": "element",
                    "batch": "reindexed",
                },
            ),
            ValueDependency(
                0,
                1,
                {
                    "coefficient": "reindexed",
                    "limb": "element",
                    "component": "reindexed",
                    "batch": "reindexed",
                },
            ),
        ]
        if string_fact(self, "compression_layout") == "strided_sparse":
            relations.append(
                ValueDependency(
                    0,
                    2,
                    {
                        "coefficient": "reindexed",
                        "limb": "element",
                        "component": "reindexed",
                        "batch": "reindexed",
                    },
                )
            )
        return OperationDependencies(tuple(relations))


@irdl_op_definition
class MultiplyCompressedPlaintextOp(_CiphertextCompressedPlaintextOp):
    r"""Multiply ciphertext components by compressed NTT plaintext data.

    The compressed layout expands logically to a prepared NTT/Montgomery
    plaintext $p$. Cyclic, contiguous and strided-sparse layouts are supported;
    sparse implicit rows enter as separate Tensor operands. Each result component represents $c_jp$ modulo every
    active prime, computed without materializing that expansion.  Component count
    and depth remain; output actual scale is $\Delta_c\Delta_p$."""

    name = "fhelium_ckks.multiply_compressed_plaintext"

    def dependencies(self) -> OperationDependencies:
        """Describe direct ciphertext reads and compact plaintext expansion."""
        relations = [
            ValueDependency(
                0,
                0,
                {
                    "coefficient": "element",
                    "limb": "element",
                    "component": "element",
                    "batch": "reindexed",
                },
            ),
            ValueDependency(
                0,
                1,
                {
                    "coefficient": "reindexed",
                    "limb": "element",
                    "component": "reindexed",
                    "batch": "reindexed",
                },
            ),
        ]
        if string_fact(self, "compression_layout") == "strided_sparse":
            relations.append(
                ValueDependency(
                    0,
                    2,
                    {
                        "coefficient": "reindexed",
                        "limb": "element",
                        "component": "reindexed",
                        "batch": "reindexed",
                    },
                )
            )
        return OperationDependencies(tuple(relations))


class _OutputDomainCiphertextOp(IRDLOperation):
    value = operand_def(CiphertextType)
    key = operand_def()
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    moddown_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    modup_method = opt_attr_def(StringAttr, default_value=StringAttr("exact"))
    traits = traits_def(Pure())
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )

    def __init__(
        self,
        value: SSAValue | Operation,
        key: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        moddown_rounding: str = "nearest",
        modup_method: str = "exact",
        parameters: Sequence[SSAValue | Operation] = (),
        output_domain: Literal["coefficient", "ntt"] = "coefficient",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.setdefault("moddown_rounding", StringAttr(moddown_rounding))
        attrs.setdefault("modup_method", StringAttr(modup_method))
        domain_attr = attrs.get("output_domain", StringAttr(output_domain))
        if not isinstance(domain_attr, StringAttr):
            raise TypeError("output_domain must be a StringAttr")
        attrs["output_domain"] = domain_attr
        output_domain = cast(Literal["coefficient", "ntt"], domain_attr.data)
        if result_type is None:
            source_type = cast(CiphertextType, SSAValue.get(value).type)
            result_type = source_type.with_state(
                {
                    "polynomial_domain": StringAttr(output_domain),
                    "residue_representation": StringAttr(
                        "standard"
                        if output_domain == "coefficient"
                        else "montgomery"
                    ),
                }
            )
        super().__init__(
            operands=[value, key, parameters],
            result_types=[result_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        if self.modup_method.data not in {"exact", "crt", "normalized_crt"}:
            raise VerifyException(
                "modup_method must be exact, crt, or normalized_crt"
            )
        if self.moddown_rounding.data not in {"floor", "nearest"}:
            raise VerifyException(
                "moddown_rounding must be 'floor' or 'nearest'"
            )
        if self.output_domain.data not in {"coefficient", "ntt"}:
            raise VerifyException(
                "CKKS key-switch output_domain must be 'coefficient' or 'ntt'"
            )


@irdl_op_definition
class RelinearizeOp(_OutputDomainCiphertextOp):
    r"""Relinearize CT3 and optionally consume the next complete Q group.

    The input is an NTT/Montgomery three-component Q ciphertext.

    With products-only accumulator $v$, original components $a=(c_0,c_1)$ and the key's auxiliary product $P$, the two-component correction is $\lfloor(v+H_P)/P\rfloor+a$. ``moddown_rounding`` selects $H_P=0$ or $(P-1)/2$. With ``rescale=True``, let $D$ be the next complete Q-group product and choose $H_D$ independently from ``rescale_rounding``. The result is $\lfloor(v+Pa+H_P+PH_D)/(PD)\rfloor$, removes D rows, advances depth once and divides actual scale by D. Without rescaling, depth, Q rows and scale are preserved. Both coefficient/standard and NTT/Montgomery outputs implement this same integer equation. The key is consumed at the input depth.
    """

    name = "fhelium_ckks.relinearize"
    rescale = opt_attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    rescale_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )

    def __init__(
        self,
        value: SSAValue | Operation,
        key: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        rescale: bool = False,
        rescale_rounding: str = "nearest",
        output_domain: Literal["coefficient", "ntt"] = "coefficient",
        moddown_rounding: str = "nearest",
        modup_method: str = "exact",
        parameters: Sequence[SSAValue | Operation] = (),
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs = {
            "rescale": IntegerAttr(int(rescale), 64),
            "rescale_rounding": StringAttr(rescale_rounding),
            **(attributes or {}),
        }
        domain_attr = attrs.get("output_domain")
        if isinstance(domain_attr, StringAttr):
            output_domain = cast(
                Literal["coefficient", "ntt"], domain_attr.data
            )
        if result_type is None:
            source_type = cast(CiphertextType, SSAValue.get(value).type)
            fields = {
                name: attribute
                for name, attribute in source_type.state.data.items()
                if name not in {"shape", "strides", "requires_grad"}
            }
            fields.update(components=IntegerAttr(2, 64))
            if cast(IntegerAttr, attrs["rescale"]).value.data:
                depth = fields.pop("depth", None)
                rows = fields.pop("prime_ids", None)
                fields.pop("scale", None)
                if isinstance(depth, IntegerAttr):
                    fields["depth"] = IntegerAttr(depth.value.data + 1, 64)
                drop = attrs.get("joint_drop_prime_ids")
                if isinstance(rows, ArrayAttr) and isinstance(drop, ArrayAttr):
                    fields["prime_ids"] = ArrayAttr(
                        i for i in rows if i not in tuple(drop)
                    )
            result_type = CiphertextType().with_state(
                {
                    **fields,
                    "polynomial_domain": StringAttr(output_domain),
                    "residue_representation": StringAttr(
                        output_residues(output_domain)
                    ),
                }
            )
        super().__init__(
            value,
            key,
            result_type,
            output_domain=output_domain,
            moddown_rounding=moddown_rounding,
            modup_method=modup_method,
            parameters=parameters,
            attributes=attrs,
        )

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in the input value and evaluation key."""
        return operand_relations(
            self,
            {
                0: {
                    "coefficient": "mixing",
                    "limb": "mixing",
                    "component": "mixing",
                    "batch": "element",
                },
                1: {
                    "coefficient": "mixing",
                    "limb": "mixing",
                    "component": "mixing",
                    "batch": "reindexed",
                },
            },
        )


@irdl_op_definition
class SwitchKeyOp(_OutputDomainCiphertextOp):
    r"""Change the secret-key relation of a CT2 ciphertext.

    The input is a coefficient-domain standard-Q CT2 ciphertext.  For phase
    $c_0+c_1s_{\mathrm{src}}$, the supplied key Tensor maps
    the $c_1s_{\mathrm{src}}$ term to corrections $(d_0,d_1)$ satisfying the
    destination relation.  The result $(c_0+d_0,d_1)$ decrypts under
    $s_{\mathrm{dst}}$ to the same approximate message, apart from key-switch
    error. ``output_domain`` selects coefficient/standard or NTT/Montgomery
    output. Depth, actual scale, active Q rows, and component count are
    preserved."""

    name = "fhelium_ckks.switch_key"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'mixing',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'mixing',
                    'batch': 'reindexed',
                },
            },
        )


@irdl_op_definition
class ConjugateOp(_OutputDomainCiphertextOp):
    r"""Apply complex conjugation to every encoded CKKS slot.

    The input is a coefficient-domain standard-Q CT2 ciphertext.  The ring
    automorphism $\sigma_{2N-1}:X\mapsto X^{-1}$ maps the CKKS
    embedding to slotwise complex conjugation.  It also changes the secret
    relation to $\sigma_{2N-1}(s)$, so a conjugation key switches the result back
    to $s$. ``output_domain`` selects coefficient/standard or NTT/Montgomery
    output. CT2 shape, depth, actual scale, and active Q rows are preserved."""

    name = "fhelium_ckks.conjugate"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'mixing',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'mixing',
                    'limb': 'mixing',
                    'component': 'mixing',
                    'batch': 'reindexed',
                },
            },
        )


@irdl_op_definition
class RescaleOp(IRDLOperation):
    r"""Divide a ciphertext by its leading Q depth-group product.

    For group $G_d$, every component coefficient is quotient-rounded by
    $M_d=\prod_{q\in G_d}q$ and represented on the surviving Q rows. All rows
    in $G_d$ are removed, depth advances once, and actual scale changes from
    $\Delta$ to $\Delta/M_d$. Component count is unchanged.
    Coefficient/standard input uses coefficient quotient arithmetic;
    NTT/Montgomery input retains surviving evaluations by transforming only the
    rounding correction. ``rounding`` selects the quotient rule."""

    name = "fhelium_ckks.rescale"

    value = operand_def(CiphertextType)
    parameters = var_operand_def()
    result = result_def(CiphertextType)
    rounding = opt_attr_def(StringAttr)
    input_domain = attr_def(StringAttr, default_value=StringAttr("coefficient"))
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe the rounded quotient in its represented polynomial domain."""
        domain = string_fact(
            self, "input_domain", state_name="polynomial_domain"
        )
        position: DependencyKind = (
            "element"
            if domain == "coefficient"
            else "mixing"
            if domain == "ntt"
            else "unknown"
        )
        return operand_dependencies(
            self,
            (0,),
            {
                "coefficient": position,
                "limb": "mixing",
                "component": "element",
                "batch": "element",
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute | None = None,
        *,
        parameters: Sequence[SSAValue | Operation] = (),
        rounding: str | StringAttr = "nearest",
        polynomial_domain: Literal["coefficient", "ntt"] = "coefficient",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["rounding"] = (
            StringAttr(rounding) if isinstance(rounding, str) else rounding
        )
        attrs["input_domain"] = StringAttr(polynomial_domain)
        attrs["output_domain"] = StringAttr(polynomial_domain)
        if result_type is None:
            source_type = cast(CiphertextType, SSAValue.get(value).type)
            result_type = source_type.with_state(
                {
                    "polynomial_domain": StringAttr(polynomial_domain),
                    "residue_representation": StringAttr(
                        "standard"
                        if polynomial_domain == "coefficient"
                        else "montgomery"
                    ),
                }
            )
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
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
        if (
            self.input_domain.data not in {"coefficient", "ntt"}
            or self.output_domain != self.input_domain
        ):
            raise VerifyException(
                "CKKS rescale must preserve coefficient or NTT domain"
            )


@irdl_op_definition
class ModSwitchOp(_UnaryCiphertextOp):
    r"""Restrict a ciphertext to the Q rows at ``target_depth``.

    Rows removed before the target depth are discarded without dividing or
    rounding any coefficient.  Surviving residue values and the ciphertext actual
    scale are unchanged.  The operation therefore changes the modulus and depth,
    not the scale; component count, polynomial domain, and residue representation
    are preserved."""

    name = "fhelium_ckks.mod_switch"
    target_depth = attr_def(IntegerAttr)

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'reindexed',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class ReinterpretScaleOp(_UnaryCiphertextOp):
    r"""Replace ciphertext scale metadata while preserving all residues.

    The recorded actual scale becomes the supplied $\Delta'$, but ciphertext
    components, prime rows, depth, polynomial domain, and Montgomery state do not
    change.  The represented decoded value is consequently interpreted relative
    to $\Delta'$.  This operation performs no multiplication, rescaling, or
    modulus switch."""

    name = "fhelium_ckks.reinterpret_scale"
    scale = attr_def(FloatAttr)

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


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

    The message is encoded at the ciphertext depth and actual scale, reduced into
    the ciphertext's Q or QP rows, and converted to coefficient-domain Montgomery
    residues.  The prepared polynomial can then be added to component zero by
    ``AddPlaintextOp`` without changing ciphertext depth or scale."""

    name = "fhelium_ckks.prepare.add.message"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self, {0: {'slot': 'mixing', 'batch': 'element'}}
        )


@irdl_op_definition
class PrepareAddPlaintextOp(_PrepareOp):
    r"""Convert a caller-owned plaintext to addition-ready RNS state.

    The plaintext already supplies its actual scale and depth; both must match the
    ciphertext.  Encoding or residue conversion as needed produces
    coefficient-domain Montgomery rows in the ciphertext's Q or QP basis.  No
    message arithmetic is performed before ``AddPlaintextOp``."""

    name = "fhelium_ckks.prepare.add.plaintext"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(self, {0: {'batch': 'element'}})


@irdl_op_definition
class PrepareAddStaticOp(_PrepareOp):
    r"""Encode a statically known public value for ciphertext addition.

    The value is encoded at the ciphertext actual scale and depth, reduced to the
    same Q or QP rows, and converted to coefficient-domain Montgomery form.  The
    result has the state required by ``AddPlaintextOp``."""

    name = "fhelium_ckks.prepare.add.static"


@irdl_op_definition
class PrepareMultiplyMessageOp(_PrepareOp):
    r"""Encode a public message for ciphertext multiplication.

    The message is encoded at the selected default actual scale, placed at the
    ciphertext depth, reduced into matching Q or QP rows, and transformed to
    NTT/Montgomery form.  ``MultiplyPlaintextOp`` then produces scale
    $\Delta_c\Delta_p$."""

    name = "fhelium_ckks.prepare.multiply.message"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self, {0: {'slot': 'mixing', 'batch': 'element'}}
        )


@irdl_op_definition
class PrepareMultiplyPlaintextOp(_PrepareOp):
    r"""Convert a caller-owned plaintext to multiplication-ready state.

    The plaintext must share the ciphertext depth and Q or QP rows. Encoding, residue conversion, and a forward NTT as needed produce NTT/Montgomery data for ``MultiplyPlaintextOp``. The result preserves the plaintext's actual scale and the matching depth and prime rows.
    """

    name = "fhelium_ckks.prepare.multiply.plaintext"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(self, {0: {'batch': 'element'}})


@irdl_op_definition
class PrepareMultiplyStaticOp(_PrepareOp):
    r"""Encode a statically known public value for ciphertext multiplication.

    The value is encoded at the selected default actual scale and ciphertext
    depth, reduced into matching Q or QP rows, and transformed to NTT/Montgomery
    form.  Its scale later multiplies the ciphertext scale in
    ``MultiplyPlaintextOp``."""

    name = "fhelium_ckks.prepare.multiply.static"


def _modulus_raise_specification(operation: Operation) -> tuple[str, ...]:
    """Reject known raising-state contradictions while leaving absent facts open."""
    assert isinstance(operation, (ModulusRaiseOp, EncapsulatedModulusRaiseOp))
    diagnostics = list(flat_operation(operation))
    for label, value in (
        ("input", operation.value),
        ("result", operation.result),
    ):
        state = cast(CiphertextType, value.type).state.data
        for name, expected in (
            ("basis", "Q"),
            ("polynomial_domain", "coefficient"),
            ("residue_representation", "standard"),
            ("components", 2),
        ):
            represented = state.get(name)
            if represented is None or represented == StringAttr("unknown"):
                continue
            actual = (
                represented.data
                if isinstance(represented, StringAttr)
                else int(represented.value.data)
                if isinstance(represented, IntegerAttr)
                else represented
            )
            if actual != expected:
                diagnostics.append(
                    f"{label} {name} must be {expected!r}, got {actual!r}"
                )
    target = int(operation.target_depth.value.data)
    source = cast(CiphertextType, operation.value.type).state.data.get("depth")
    result = cast(CiphertextType, operation.result.type).state.data.get("depth")
    if (
        target < 0
        or isinstance(source, IntegerAttr)
        and target > int(source.value.data)
    ):
        diagnostics.append(
            "modulus raise target must precede or equal source depth and be nonnegative"
        )
    if isinstance(result, IntegerAttr) and int(result.value.data) != target:
        diagnostics.append("modulus raise result depth must equal target_depth")
    if isinstance(operation, EncapsulatedModulusRaiseOp):
        diagnostics.extend(_moddown_rounding_diagnostics(operation))
    return tuple(diagnostics)


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
    diagnostics.extend(_moddown_rounding_diagnostics(operation))
    diagnostics.extend(_modup_method_diagnostics(operation))
    unsupported = unsupported_attributes(
        operation,
        ("input_domain", "output_domain", "moddown_rounding", "modup_method"),
    )
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    if isinstance(operation, RotateOp):
        if operation.input_domain.data not in ("coefficient", "ntt"):
            diagnostics.append("input_domain must be 'coefficient' or 'ntt'")
        if operation.output_domain.data not in ("coefficient", "ntt"):
            diagnostics.append("output_domain must be 'coefficient' or 'ntt'")
        diagnostics.extend(_rotation_output_state_diagnostics(operation))
    return tuple(diagnostics)


def _hoisted_rotate_many_specification(
    operation: Operation,
) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    diagnostics.extend(_moddown_rounding_diagnostics(operation))
    diagnostics.extend(_modup_method_diagnostics(operation))
    unsupported = unsupported_attributes(
        operation, ("output_domain", "moddown_rounding", "modup_method")
    )
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    if not isinstance(operation, RotateManyOp):
        diagnostics.append(
            "requires the registered CKKS hoisted-rotation operation"
        )
    elif operation.output_domain.data not in ("coefficient", "ntt"):
        diagnostics.append("output_domain must be 'coefficient' or 'ntt'")
    else:
        diagnostics.extend(_rotation_output_state_diagnostics(operation))
    return tuple(diagnostics)


def _modup_method_diagnostics(operation: Operation) -> tuple[str, ...]:
    value = operation.attributes.get("modup_method", StringAttr("exact"))
    if not isinstance(value, StringAttr) or value.data not in {
        "exact",
        "crt",
        "normalized_crt",
    }:
        return ("modup_method must be exact, crt, or normalized_crt",)
    return ()


def _moddown_rounding_diagnostics(operation: Operation) -> tuple[str, ...]:
    value = operation.attributes.get("moddown_rounding", StringAttr("nearest"))
    if not isinstance(value, StringAttr) or value.data not in {
        "floor",
        "nearest",
    }:
        return ("moddown_rounding must be 'floor' or 'nearest'",)
    return ()


def _rotation_output_state_diagnostics(
    operation: (
        RotateOp
        | RotateManyOp
        | EncryptOp
        | RescaleOp
        | _OutputDomainCiphertextOp
    ),
) -> tuple[str, ...]:
    """Check represented result domains against execution output."""

    domain = operation.output_domain.data
    if domain not in {"coefficient", "ntt"}:
        return ("output_domain must be 'coefficient' or 'ntt'",)
    expected = {
        "polynomial_domain": domain,
        "residue_representation": output_residues(
            cast(Literal["coefficient", "ntt"], domain)
        ),
    }
    diagnostics: list[str] = []
    if isinstance(
        operation, RelinearizeOp
    ) and operation.rescale_rounding.data not in {"nearest", "floor"}:
        diagnostics.append("rescale_rounding must be nearest or floor")
    for index, result in enumerate(operation.results):
        if not isinstance(result.type, CiphertextType):
            continue
        for name, value in expected.items():
            represented = result.type.state.data.get(name)
            if represented is None:
                continue
            if (
                not isinstance(represented, StringAttr)
                or represented.data != value
            ):
                diagnostics.append(
                    f"result {index} {name} must be {value!r} for output_domain={domain!r}"
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
    unsupported = unsupported_attributes(
        operation, ("rounding", "input_domain", "output_domain", "drop_count")
    )
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    rounding = required_string_attribute(operation, "rounding")
    if rounding not in {"nearest", "floor"}:
        diagnostics.append(
            "rescale requires represented rounding='nearest' or 'floor'"
        )
    if isinstance(operation, RescaleOp):
        if operation.input_domain.data not in {"coefficient", "ntt"}:
            diagnostics.append("input_domain must be 'coefficient' or 'ntt'")
        if operation.output_domain != operation.input_domain:
            diagnostics.append("rescale must preserve its polynomial domain")
        diagnostics.extend(_rotation_output_state_diagnostics(operation))
    expected_roles = ("encrypted",)
    actual_roles = tuple(
        (
            "encrypted"
            if isinstance(operand.type, CiphertextType)
            else "plaintext"
            if isinstance(operand.type, PlaintextType)
            else None
        )
        for operand in operation.operands[:1]
    )
    if actual_roles != expected_roles:
        diagnostics.append(
            f"rescale requires operand roles {expected_roles}, got {actual_roles}"
        )
    return tuple(diagnostics)


def _decrypt_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(operation, ("input_domain",))
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    domain = required_string_attribute(operation, "input_domain")
    if domain not in {"coefficient", "ntt"}:
        diagnostics.append("input_domain must be 'coefficient' or 'ntt'")
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = (
    registered_operation_spec(
        PrepareCompressedPlaintextOp,
        "ckks",
        effect="rng-write",
        validator=flat_with_attributes(
            "depth",
            "scale",
            "ring_dimension",
            "min_modulus",
            "polynomial_domain",
        ),
    ),
    registered_operation_spec(
        EncodeOp,
        "ckks",
        effect="rng-write",
        validator=flat_with_attributes("depth", "scale"),
    ),
    registered_operation_spec(
        DecodeOp,
        "ckks",
        validator=flat_with_attributes("is_real"),
    ),
    registered_operation_spec(
        IntegerCoefficientsToRnsOp,
        "ckks",
        validator=flat_with_attributes("modulus_basis", "depth"),
    ),
    registered_operation_spec(
        EncryptOp,
        "ckks",
        effect="rng-write",
        validator=lambda operation: (
            *flat_with_attributes("output_domain")(operation),
            *_rotation_output_state_diagnostics(cast(EncryptOp, operation)),
        ),
    ),
    registered_operation_spec(
        DecryptOp,
        "ckks",
        validator=_decrypt_specification,
    ),
    registered_operation_spec(
        SumBatchOp,
        "ckks",
        validator=flat_with_attributes("axis"),
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
        ModulusRaiseOp, "ckks", validator=_modulus_raise_specification
    ),
    registered_operation_spec(
        EncapsulatedModulusRaiseOp,
        "ckks",
        validator=_modulus_raise_specification,
    ),
    registered_operation_spec(RotationSumOp, "ckks"),
    registered_operation_spec(
        GroupedRotationWeightedSumOp,
        "ckks",
        validator=_with_ciphertext_components(
            lambda operation: (
                *flat_with_attributes(
                    "baby_steps",
                    "term_count",
                    "group_count",
                    "moddown_rounding",
                    "modup_method",
                )(operation),
                *_moddown_rounding_diagnostics(operation),
                *_modup_method_diagnostics(operation),
            ),
            operands=((0, 2),),
            results=((0, 2),),
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
        validator=flat_with_attributes(
            "inplace", "compression_layout", "polynomial_domain"
        ),
    ),
    registered_operation_spec(
        MultiplyCompressedPlaintextOp,
        "ckks",
        validator=flat_with_attributes("inplace", "compression_layout"),
    ),
    registered_operation_spec(
        RelinearizeOp,
        "ckks",
        validator=_with_ciphertext_components(
            lambda operation: (
                *flat_with_attributes(
                    "output_domain",
                    "moddown_rounding",
                    "modup_method",
                    "rescale",
                    "rescale_rounding",
                )(operation),
                *_moddown_rounding_diagnostics(operation),
                *_modup_method_diagnostics(operation),
                *_rotation_output_state_diagnostics(
                    cast(RelinearizeOp, operation)
                ),
            ),
            operands=((0, 3),),
            results=((0, 2),),
        ),
    ),
    registered_operation_spec(
        SwitchKeyOp,
        "ckks",
        validator=_with_ciphertext_components(
            lambda operation: (
                *flat_with_attributes(
                    "output_domain", "moddown_rounding", "modup_method"
                )(operation),
                *_moddown_rounding_diagnostics(operation),
                *_modup_method_diagnostics(operation),
                *_rotation_output_state_diagnostics(
                    cast(SwitchKeyOp, operation)
                ),
            ),
            operands=((0, 2),),
            results=((0, 2),),
        ),
    ),
    registered_operation_spec(
        ConjugateOp,
        "ckks",
        validator=_with_ciphertext_components(
            lambda operation: (
                *flat_with_attributes(
                    "output_domain", "moddown_rounding", "modup_method"
                )(operation),
                *_moddown_rounding_diagnostics(operation),
                *_modup_method_diagnostics(operation),
                *_rotation_output_state_diagnostics(
                    cast(ConjugateOp, operation)
                ),
            ),
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
        validator=flat_with_attributes("target_depth"),
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
        PrepareCompressedPlaintextOp,
        EncodeOp,
        DecodeOp,
        IntegerCoefficientsToRnsOp,
        EncryptOp,
        DecryptOp,
        NegateOp,
        RotateOp,
        RotateManyOp,
        ModulusRaiseOp,
        EncapsulatedModulusRaiseOp,
        RotationSumOp,
        GroupedRotationWeightedSumOp,
        ToNttOp,
        FromNttOp,
        ToMontgomeryResiduesOp,
        ToStandardResiduesOp,
        AddOp,
        SumBatchOp,
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
    "SumBatchOp",
    "AddScalarOp",
    "AddPlaintextOp",
    "AddCompressedPlaintextOp",
    "CiphertextType",
    "CompressedPlaintextType",
    "ConjugateOp",
    "DecodeOp",
    "DecryptOp",
    "PrepareCompressedPlaintextOp",
    "EncodeOp",
    "EncryptOp",
    "EvaluationKeyType",
    "FHEliumCkks",
    "FromNttOp",
    "GroupedRotationWeightedSumOp",
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
    "ModulusRaiseOp",
    "EncapsulatedModulusRaiseOp",
    "RotationSumOp",
    "SubtractOp",
    "SwitchKeyOp",
    "ToNttOp",
    "ToMontgomeryResiduesOp",
    "ToStandardResiduesOp",
]
