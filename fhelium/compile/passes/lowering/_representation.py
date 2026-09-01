"""Lower CKKS representation and level transitions into logical operations."""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation

from fhelium.config import CkksConfig

from ....ir.dialects import ckks, core, ntt, rns
from ._core import (
    CkksLoweringDefinition,
    LoweredCkksOperation,
)
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _resource_type_state,
    _rns_type,
)


def _lower_ntt(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, (ckks.ToNttOp, ckks.FromNttOp)):
        raise TypeError("NTT lowering received another operation")
    input_cast, value = _cast_to_rns(operation.value)
    plan_type = ntt.NttPlanType.new((_resource_type_state(kind="ntt-plan"),))
    resource = core.ResourceRefOp(
        plan_type,
        symbol="active-ntt-plan",
        kind="ntt-plan",
    )
    value_kind = (
        "ciphertext"
        if isinstance(operation.value.type, ckks.CiphertextType)
        else "plaintext"
    )
    if isinstance(operation, ckks.ToNttOp):
        operation_type: type[Operation] = (
            ntt.CoefficientStandardToNttMontgomeryOp
            if value_kind == "ciphertext"
            else ntt.CoefficientMontgomeryToNttMontgomeryOp
        )
    else:
        operation_type = (
            ntt.NttMontgomeryToCoefficientStandardOp
            if value_kind == "ciphertext"
            else ntt.InverseMontgomeryOp
        )
    logical = operation_type.create(
        operands=(value, resource.value),
        result_types=(_rns_type(operation.result.type),),
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (input_cast, resource, logical, result_cast),
        result,
    )


def _lower_rescale(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.RescaleOp):
        raise TypeError("rescale lowering received another operation")
    input_cast, value = _cast_to_rns(operation.value)
    plan_type = rns.RescalePlanType.new(
        (_resource_type_state(kind="rescale-plan"),)
    )
    resource = core.ResourceRefOp(
        plan_type,
        symbol="active-rescale-plan",
        kind="rescale-plan",
    )
    rounding = operation.rounding or StringAttr("nearest")
    logical = rns.RescaleDropLeadingPrimeOp(
        value,
        resource,
        _rns_type(operation.result.type),
        rounding=rounding,
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (input_cast, resource, logical, result_cast),
        result,
    )


def _lower_residue_conversion(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(
        operation,
        (ckks.ToMontgomeryResiduesOp, ckks.ToStandardResiduesOp),
    ):
        raise TypeError(
            "Residue conversion lowering received another operation"
        )
    input_cast, value = _cast_to_rns(operation.value)
    parameters_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    resource = core.ResourceRefOp(
        parameters_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    operation_type: type[Operation] = (
        rns.StandardToMontgomeryOp
        if isinstance(operation, ckks.ToMontgomeryResiduesOp)
        else rns.MontgomeryToStandardOp
    )
    logical = operation_type.create(
        operands=(value, resource.value),
        result_types=(_rns_type(operation.result.type),),
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (input_cast, resource, logical, result_cast),
        result,
    )


def _lower_mod_switch(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.ModSwitchOp):
        raise TypeError("Modulus-switch lowering received another operation")
    input_cast, value = _cast_to_rns(operation.value)
    parameters_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    resource = core.ResourceRefOp(
        parameters_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    logical = rns.RestrictLevelOp(
        value,
        resource,
        _rns_type(operation.result.type),
        target_level=operation.target_level,
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (input_cast, resource, logical, result_cast),
        result,
    )


def _lower_reinterpret_scale(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    del config
    if not isinstance(operation, ckks.ReinterpretScaleOp):
        raise TypeError(
            "Scale reinterpretation lowering received another operation"
        )
    input_cast, value = _cast_to_rns(operation.value)
    logical = rns.ReinterpretScaleOp(
        value,
        _rns_type(operation.result.type),
        scale=operation.scale,
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation((input_cast, logical, result_cast), result)


REPRESENTATION_LOWERINGS = (
    CkksLoweringDefinition(
        "rns-ntt-transition",
        ckks.ToNttOp,
        _lower_ntt,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-ntt-transition",
        ckks.FromNttOp,
        _lower_ntt,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-drop-leading-prime",
        ckks.RescaleOp,
        _lower_rescale,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-residue-conversion",
        ckks.ToMontgomeryResiduesOp,
        _lower_residue_conversion,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-residue-conversion",
        ckks.ToStandardResiduesOp,
        _lower_residue_conversion,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-restrict-level",
        ckks.ModSwitchOp,
        _lower_mod_switch,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-reinterpret-scale",
        ckks.ReinterpretScaleOp,
        _lower_reinterpret_scale,
        is_default=True,
    ),
)
