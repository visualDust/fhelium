"""CKKS evaluator semantics with representation-bearing value types."""

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
    """Encode one public CKKS slot message as integer coefficients."""

    name = "fhelium_ckks.encode"
    message = operand_def(MessageType)
    result = result_def(PlaintextType)
    level = attr_def(IntegerAttr)
    scale = attr_def(FloatAttr)


@irdl_op_definition
class DecodeOp(IRDLOperation):
    """Decode one coefficient plaintext into ordered CKKS slots."""

    name = "fhelium_ckks.decode"
    plaintext = operand_def(PlaintextType)
    result = result_def(MessageType)
    is_real = attr_def(IntegerAttr)
    traits = traits_def(Pure())


@irdl_op_definition
class IntegerCoefficientsToRnsOp(IRDLOperation):
    """Reduce integer coefficients into standard RNS rows."""

    name = "fhelium_ckks.integer_coefficients_to_rns"
    plaintext = operand_def(PlaintextType)
    result = result_def(PlaintextType)
    modulus_basis = attr_def(StringAttr)
    level = attr_def(IntegerAttr)
    traits = traits_def(Pure())


@irdl_op_definition
class EncryptOp(IRDLOperation):
    """Encrypt slots or integer coefficients with a named public key."""

    name = "fhelium_ckks.encrypt"
    plaintext = operand_def(PlaintextType)
    result = result_def(CiphertextType)
    key_symbol = attr_def(StringAttr)


@irdl_op_definition
class DecryptOp(IRDLOperation):
    """Decrypt a ciphertext with a named secret key."""

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
    """Negate a CKKS ciphertext without changing its represented state."""

    name = "fhelium_ckks.negate"


@irdl_op_definition
class RotateOp(IRDLOperation):
    """Apply the slot rotation carried by one evaluation-key operand."""

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
    """Apply a scheduled group of rotations with shared key-switch preparation."""

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
    """Map coefficient-domain CKKS residues to NTT/Montgomery form."""

    name = "fhelium_ckks.to_ntt"


@irdl_op_definition
class FromNttOp(_RepresentationOp):
    """Map NTT/Montgomery CKKS residues to coefficient-domain form."""

    name = "fhelium_ckks.from_ntt"


@irdl_op_definition
class ToMontgomeryResiduesOp(_RepresentationOp):
    """Convert coefficient-domain plaintext residues to Montgomery form."""

    name = "fhelium_ckks.to_montgomery_residues"


@irdl_op_definition
class ToStandardResiduesOp(_RepresentationOp):
    """Convert coefficient-domain plaintext residues to standard form."""

    name = "fhelium_ckks.to_standard_residues"


@irdl_op_definition
class AddOp(_BinaryCiphertextOp):
    """Add two state-compatible CKKS ciphertexts."""

    name = "fhelium_ckks.add"


@irdl_op_definition
class SubtractOp(_BinaryCiphertextOp):
    """Subtract one state-compatible CKKS ciphertext from another."""

    name = "fhelium_ckks.subtract"


@irdl_op_definition
class MultiplyOp(_BinaryCiphertextOp):
    """Multiply two CT2 NTT ciphertexts and produce a CT3 ciphertext."""

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
    r"""Add a real scalar encoded at a caller-selected scale."""

    name = "fhelium_ckks.add_scalar"


@irdl_op_definition
class MultiplyScalarOp(_RealScalarCiphertextOp):
    r"""Multiply by a real scalar encoded at a caller-selected scale."""

    name = "fhelium_ckks.multiply_scalar"


@irdl_op_definition
class MultiplyIntegerScalarOp(IRDLOperation):
    """Multiply by an integer without changing the ciphertext scale."""

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
    """Add an operation-ready CKKS plaintext to a ciphertext."""

    name = "fhelium_ckks.add_plaintext"


@irdl_op_definition
class MultiplyPlaintextOp(_CiphertextPlaintextOp):
    """Multiply a CKKS ciphertext by an operation-ready plaintext."""

    name = "fhelium_ckks.multiply_plaintext"


class _CiphertextCompressedPlaintextOp(IRDLOperation):
    ciphertext = operand_def(CiphertextType)
    plaintext = operand_def(CompressedPlaintextType)
    result = result_def(CiphertextType)
    inplace = opt_attr_def(IntegerAttr)


@irdl_op_definition
class AddCompressedPlaintextOp(_CiphertextCompressedPlaintextOp):
    """Add an operation-ready compressed plaintext to ciphertext component zero."""

    name = "fhelium_ckks.add_compressed_plaintext"


@irdl_op_definition
class MultiplyCompressedPlaintextOp(_CiphertextCompressedPlaintextOp):
    """Multiply NTT ciphertext components by compressed NTT plaintext data."""

    name = "fhelium_ckks.multiply_compressed_plaintext"


@irdl_op_definition
class RelinearizeOp(_UnaryCiphertextOp):
    """Key-switch a CT3 product back to a CT2 CKKS ciphertext."""

    name = "fhelium_ckks.relinearize"


@irdl_op_definition
class SwitchKeyOp(_UnaryCiphertextOp):
    """Switch a CT2 ciphertext through caller-identified evaluation material."""

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
    """Apply complex conjugation with the context's conjugation key."""

    name = "fhelium_ckks.conjugate"


@irdl_op_definition
class RescaleOp(IRDLOperation):
    """Drop the leading Q prime and advance one public CKKS level."""

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
    """Restrict a ciphertext to a later Q-chain level without scale change."""

    name = "fhelium_ckks.mod_switch"
    target_level = attr_def(IntegerAttr)


@irdl_op_definition
class ReinterpretScaleOp(_UnaryCiphertextOp):
    """Change ciphertext scale metadata without changing RNS residues."""

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
    """Encode and prepare a public message for ciphertext addition."""

    name = "fhelium_ckks.prepare.add.message"


@irdl_op_definition
class PrepareAddPlaintextOp(_PrepareOp):
    """Prepare a caller-owned plaintext for ciphertext addition."""

    name = "fhelium_ckks.prepare.add.plaintext"


@irdl_op_definition
class PrepareAddStaticOp(_PrepareOp):
    """Encode and prepare a specialized scalar for ciphertext addition."""

    name = "fhelium_ckks.prepare.add.static"


@irdl_op_definition
class PrepareMultiplyMessageOp(_PrepareOp):
    """Encode and prepare a public message for ciphertext multiplication."""

    name = "fhelium_ckks.prepare.multiply.message"


@irdl_op_definition
class PrepareMultiplyPlaintextOp(_PrepareOp):
    """Prepare a caller-owned plaintext for ciphertext multiplication."""

    name = "fhelium_ckks.prepare.multiply.plaintext"


@irdl_op_definition
class PrepareMultiplyStaticOp(_PrepareOp):
    """Encode and prepare a specialized scalar for ciphertext multiplication."""

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
