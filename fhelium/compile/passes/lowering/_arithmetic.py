"""Lower CKKS arithmetic operations into logical RNS compositions."""

from __future__ import annotations

from xdsl.ir import Operation

from fhelium.config import CkksConfig

from ....ir.dialects import ckks, core, rns
from ._core import (
    CkksLoweringDefinition,
    LoweredCkksOperation,
)
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _polynomial_type,
    _resource_type_state,
    _rns_type,
)


def _lower_add(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.AddOp):
        raise TypeError("add lowering received another operation")
    lhs_cast, lhs = _cast_to_rns(operation.lhs)
    rhs_cast, rhs = _cast_to_rns(operation.rhs)
    parameter_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    resource = core.ResourceRefOp(
        parameter_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    logical = rns.AddStandardOp(
        lhs,
        rhs,
        resource,
        _rns_type(operation.result.type),
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (lhs_cast, rhs_cast, resource, logical, result_cast),
        result,
    )


def _lower_subtract(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.SubtractOp):
        raise TypeError("subtract lowering received another operation")
    lhs_cast, lhs = _cast_to_rns(operation.lhs)
    rhs_cast, rhs = _cast_to_rns(operation.rhs)
    parameter_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    resource = core.ResourceRefOp(
        parameter_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    logical = rns.SubtractStandardOp(
        lhs,
        rhs,
        resource,
        _rns_type(operation.result.type),
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (lhs_cast, rhs_cast, resource, logical, result_cast),
        result,
    )


def _lower_negate(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.NegateOp):
        raise TypeError("negate lowering received another operation")
    input_cast, value = _cast_to_rns(operation.value)
    parameter_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    resource = core.ResourceRefOp(
        parameter_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    logical = rns.NegateStandardOp(
        value,
        resource,
        _rns_type(operation.result.type),
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (input_cast, resource, logical, result_cast),
        result,
    )


def _lower_plaintext_arithmetic(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(
        operation, (ckks.AddPlaintextOp, ckks.MultiplyPlaintextOp)
    ):
        raise TypeError(
            "Plaintext arithmetic lowering received another operation"
        )
    ciphertext_cast, ciphertext = _cast_to_rns(operation.ciphertext)
    plaintext_cast, plaintext = _cast_to_rns(operation.plaintext)
    parameters_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    resource = core.ResourceRefOp(
        parameters_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    operation_type: type[Operation] = (
        rns.AddPlaintextOp
        if isinstance(operation, ckks.AddPlaintextOp)
        else rns.MultiplyPlaintextOp
    )
    logical = operation_type.create(
        operands=(ciphertext, plaintext, resource.value),
        result_types=(_rns_type(operation.result.type),),
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (ciphertext_cast, plaintext_cast, resource, logical, result_cast),
        result,
    )


def _lower_multiply(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.MultiplyOp):
        raise TypeError(
            "Ciphertext multiply lowering received another operation"
        )
    lhs_cast, lhs = _cast_to_rns(operation.lhs)
    rhs_cast, rhs = _cast_to_rns(operation.rhs)
    parameters_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    resource = core.ResourceRefOp(
        parameters_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    lhs_component_type = _polynomial_type(operation.lhs.type)
    rhs_component_type = _polynomial_type(operation.rhs.type)
    product_type = _polynomial_type(operation.result.type)
    lhs0 = rns.ExtractComponentOp(lhs, lhs_component_type, component=0)
    lhs1 = rns.ExtractComponentOp(lhs, lhs_component_type, component=1)
    rhs0 = rns.ExtractComponentOp(rhs, rhs_component_type, component=0)
    rhs1 = rns.ExtractComponentOp(rhs, rhs_component_type, component=1)
    product00 = rns.MontgomeryMultiplyOp(
        lhs0.result,
        rhs0.result,
        resource,
        product_type,
    )
    product01 = rns.MontgomeryMultiplyOp(
        lhs0.result,
        rhs1.result,
        resource,
        product_type,
    )
    product10 = rns.MontgomeryMultiplyOp(
        lhs1.result,
        rhs0.result,
        resource,
        product_type,
    )
    product11 = rns.MontgomeryMultiplyOp(
        lhs1.result,
        rhs1.result,
        resource,
        product_type,
    )
    cross = rns.AddMontgomeryLazyOp(
        product01.result,
        product10.result,
        resource,
        product_type,
    )
    packed = rns.PackThreeComponentsOp(
        product00.result,
        cross.result,
        product11.result,
        _rns_type(operation.result.type),
    )
    result_cast, result = _cast_to_ckks(packed.result, operation.result.type)
    return LoweredCkksOperation(
        (
            lhs_cast,
            rhs_cast,
            resource,
            lhs0,
            lhs1,
            rhs0,
            rhs1,
            product00,
            product01,
            product10,
            product11,
            cross,
            packed,
            result_cast,
        ),
        result,
    )


ARITHMETIC_LOWERINGS = (
    CkksLoweringDefinition(
        "rns-standard-add",
        ckks.AddOp,
        _lower_add,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-standard-subtract",
        ckks.SubtractOp,
        _lower_subtract,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-standard-negate",
        ckks.NegateOp,
        _lower_negate,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-plaintext-add",
        ckks.AddPlaintextOp,
        _lower_plaintext_arithmetic,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-plaintext-multiply",
        ckks.MultiplyPlaintextOp,
        _lower_plaintext_arithmetic,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-ct2-convolution",
        ckks.MultiplyOp,
        _lower_multiply,
        is_default=True,
    ),
)
