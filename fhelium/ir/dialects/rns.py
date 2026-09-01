"""Logical residue-number-system operations used below CKKS lowering.

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
    """Add equal-layout standard-range RNS bundles modulo each active prime."""

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
    """Subtract equal-layout standard-range RNS bundles modulo active primes."""

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
    """Negate one standard-range RNS bundle modulo each active prime."""

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
    """Add prepared Montgomery plaintext residues to ciphertext component zero."""

    name = "fhelium_rns.add_plaintext"


@irdl_op_definition
class MultiplyPlaintextOp(_CiphertextPlaintextOp):
    """Multiply each ciphertext component by an NTT/Montgomery plaintext."""

    name = "fhelium_rns.multiply_plaintext"


@irdl_op_definition
class RescaleDropLeadingPrimeOp(IRDLOperation):
    """Divide-round by the leading Q prime and remove its residue row."""

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
    """Extract one polynomial from a represented RNS component axis."""

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
    """Pack two equal-layout RNS polynomials into one component bundle."""

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
    """Pack three equal-layout RNS polynomials into one component bundle."""

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
    """Multiply two NTT/Montgomery polynomial bundles componentwise."""

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
    """ModUp one selected hybrid digit from active Q into active QP."""

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
    """Multiply one NTT QP digit by its two evaluation-key components."""

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
    """Add equal-layout Montgomery RNS bundles in their represented lazy range."""

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
    """Sequentially divide-round by P and return active-Q residues."""

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
    """Apply one coefficient-domain CKKS ring automorphism to an RNS bundle."""

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
    """Map standard RNS residues into Montgomery representation."""

    name = "fhelium_rns.standard_to_montgomery"


@irdl_op_definition
class MontgomeryToStandardOp(_ResidueConversionOp):
    """Reduce Montgomery RNS residues into standard representation."""

    name = "fhelium_rns.montgomery_to_standard"


@irdl_op_definition
class RestrictLevelOp(IRDLOperation):
    """Select the residue rows represented by a later modulus-chain level."""

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
    """Change RNS scale metadata while preserving every residue."""

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
