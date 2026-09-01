"""Lower CKKS key-switch operations and their logical RNS/NTT composition."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, SSAValue

from fhelium.config import CkksConfig

from ....ir.dialects import ckks, core, ntt, rns
from ._core import (
    CkksLoweringDefinition,
    LoweredCkksOperation,
)
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _component_bundle_type,
    _key_switch_digit_indices,
    _polynomial_type,
    _resource_type_state,
    _rns_type,
    _state_integer,
)


@dataclass(frozen=True)
class _KeySwitchResources:
    operations: tuple[Operation, ...]
    parameters: SSAValue
    ntt_plan: SSAValue
    key_switch_plan: SSAValue
    evaluation_key: SSAValue


def _key_switch_resources(
    config: CkksConfig,
    *,
    key_kind: str,
    key_symbol: str | None = None,
    evaluation_key: SSAValue | None = None,
) -> _KeySwitchResources:
    parameter_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    parameters = core.ResourceRefOp(
        parameter_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    ntt_plan_type = ntt.NttPlanType.new(
        (_resource_type_state(kind="ntt-plan"),)
    )
    ntt_plan = core.ResourceRefOp(
        ntt_plan_type,
        symbol="active-ntt-plan",
        kind="ntt-plan",
    )
    key_switch_plan_type = rns.KeySwitchPlanType.new(
        (_resource_type_state(kind="key-switch-plan"),)
    )
    key_switch_plan = core.ResourceRefOp(
        key_switch_plan_type,
        symbol="active-key-switch-plan",
        kind="key-switch-plan",
    )
    operations: tuple[Operation, ...]
    if evaluation_key is None:
        if key_symbol is None:
            raise ValueError("Key-switch lowering requires a key resource")
        evaluation_key_type = rns.EvaluationKeyResourceType.new(
            (_resource_type_state(kind=key_kind),)
        )
        key_reference = core.ResourceRefOp(
            evaluation_key_type,
            symbol=key_symbol,
            kind=key_kind,
        )
        evaluation_key = key_reference.value
        operations = (parameters, ntt_plan, key_switch_plan, key_reference)
    else:
        operations = (parameters, ntt_plan, key_switch_plan)
    return _KeySwitchResources(
        operations,
        parameters.value,
        ntt_plan.value,
        key_switch_plan.value,
        evaluation_key,
    )


def _extract_component(
    value: SSAValue,
    source_type: object,
    component: int,
    **updates: Attribute | None,
) -> tuple[rns.ExtractComponentOp, SSAValue]:
    operation = rns.ExtractComponentOp(
        value,
        _polynomial_type(source_type, **updates),
        component=component,
    )
    return operation, operation.result


def _key_switch_corrections(
    source: SSAValue,
    source_type: object,
    resources: _KeySwitchResources,
    key_digit_indices: tuple[int, ...],
) -> tuple[tuple[Operation, ...], SSAValue]:
    operations: list[Operation] = []
    accumulator: SSAValue | None = None
    qp_coefficient_type = _polynomial_type(
        source_type,
        basis=StringAttr("QP"),
        prime_ids=None,
        polynomial_domain=StringAttr("coefficient"),
        residue_representation=StringAttr("montgomery"),
    )
    qp_ntt_type = qp_coefficient_type.with_state(
        polynomial_domain=StringAttr("ntt")
    )
    product_type = _component_bundle_type(
        source_type,
        2,
        basis=StringAttr("QP"),
        prime_ids=None,
        polynomial_domain=StringAttr("ntt"),
        residue_representation=StringAttr("montgomery"),
    )
    for digit_index, key_digit_index in enumerate(key_digit_indices):
        modup = rns.HybridModUpDigitOp(
            source,
            resources.parameters,
            resources.key_switch_plan,
            qp_coefficient_type,
            digit_index=digit_index,
        )
        forward = ntt.CoefficientMontgomeryToNttMontgomeryOp(
            modup.result,
            resources.ntt_plan,
            qp_ntt_type,
        )
        product = rns.KeySwitchDigitProductOp(
            forward.result,
            resources.evaluation_key,
            resources.parameters,
            resources.key_switch_plan,
            product_type,
            key_digit_index=key_digit_index,
        )
        operations.extend((modup, forward, product))
        if accumulator is None:
            accumulator = product.result
            continue
        addition = rns.AddMontgomeryLazyOp(
            accumulator,
            product.result,
            resources.parameters,
            product_type,
        )
        operations.append(addition)
        accumulator = addition.result
    if accumulator is None:
        raise ValueError("Key switching requires at least one active RNS digit")

    inverse_type = product_type.with_state(
        polynomial_domain=StringAttr("coefficient"),
        residue_representation=StringAttr("standard"),
    )
    inverse = ntt.NttMontgomeryToCoefficientStandardOp(
        accumulator,
        resources.ntt_plan,
        inverse_type,
    )
    q_prime_ids = None
    if isinstance(source_type, ckks.CiphertextType):
        q_prime_ids = source_type.state.data.get("prime_ids")
    correction_type = _component_bundle_type(
        source_type,
        2,
        basis=StringAttr("Q"),
        prime_ids=q_prime_ids,
        polynomial_domain=StringAttr("coefficient"),
        residue_representation=StringAttr("standard"),
    )
    moddown = rns.ModDownQpToQOp(
        inverse.result,
        resources.parameters,
        resources.key_switch_plan,
        correction_type,
    )
    operations.extend((inverse, moddown))
    return tuple(operations), moddown.result


def _assemble_switched_ciphertext(
    component0: SSAValue,
    source_component1: SSAValue,
    source_type: object,
    result_type: object,
    resources: _KeySwitchResources,
    key_digit_indices: tuple[int, ...],
) -> tuple[tuple[Operation, ...], SSAValue]:
    switch_ops, corrections = _key_switch_corrections(
        source_component1,
        source_type,
        resources,
        key_digit_indices,
    )
    correction0_op, correction0 = _extract_component(
        corrections, source_type, 0
    )
    correction1_op, correction1 = _extract_component(
        corrections, source_type, 1
    )
    polynomial_type = _polynomial_type(
        source_type,
        polynomial_domain=StringAttr("coefficient"),
        residue_representation=StringAttr("standard"),
    )
    add0 = rns.AddStandardOp(
        component0,
        correction0,
        resources.parameters,
        polynomial_type,
    )
    packed = rns.PackTwoComponentsOp(
        add0.result,
        correction1,
        _rns_type(result_type),
    )
    return (
        (*switch_ops, correction0_op, correction1_op, add0, packed),
        packed.result,
    )


def _lower_relinearize(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.RelinearizeOp):
        raise TypeError("relinearize lowering received another operation")
    key_digit_indices = _key_switch_digit_indices(
        operation.value.type,
        config,
        operation="CKKS relinearization",
    )
    input_cast, value = _cast_to_rns(operation.value)
    resources = _key_switch_resources(
        config,
        key_symbol="relinearization-key",
        key_kind="relinearization-key",
    )
    coefficient_type = _rns_type(operation.value.type).with_state(
        {
            "polynomial_domain": StringAttr("coefficient"),
            "residue_representation": StringAttr("standard"),
        }
    )
    inverse = ntt.NttMontgomeryToCoefficientStandardOp(
        value,
        resources.ntt_plan,
        coefficient_type,
    )
    coefficient_updates: dict[str, Attribute] = {
        "polynomial_domain": StringAttr("coefficient"),
        "residue_representation": StringAttr("standard"),
    }
    d0_op, d0 = _extract_component(
        inverse.result, operation.value.type, 0, **coefficient_updates
    )
    d1_op, d1 = _extract_component(
        inverse.result, operation.value.type, 1, **coefficient_updates
    )
    d2_op, d2 = _extract_component(
        inverse.result, operation.value.type, 2, **coefficient_updates
    )
    switch_ops, corrections = _key_switch_corrections(
        d2,
        operation.value.type,
        resources,
        key_digit_indices,
    )
    correction0_op, correction0 = _extract_component(
        corrections, operation.value.type, 0, **coefficient_updates
    )
    correction1_op, correction1 = _extract_component(
        corrections, operation.value.type, 1, **coefficient_updates
    )
    polynomial_type = _polynomial_type(
        operation.value.type,
        polynomial_domain=StringAttr("coefficient"),
        residue_representation=StringAttr("standard"),
    )
    result0 = rns.AddStandardOp(
        d0,
        correction0,
        resources.parameters,
        polynomial_type,
    )
    result1 = rns.AddStandardOp(
        d1,
        correction1,
        resources.parameters,
        polynomial_type,
    )
    packed = rns.PackTwoComponentsOp(
        result0.result,
        result1.result,
        _rns_type(operation.result.type),
    )
    result_cast, result = _cast_to_ckks(packed.result, operation.result.type)
    return LoweredCkksOperation(
        (
            input_cast,
            *resources.operations,
            inverse,
            d0_op,
            d1_op,
            d2_op,
            *switch_ops,
            correction0_op,
            correction1_op,
            result0,
            result1,
            packed,
            result_cast,
        ),
        result,
    )


def _lower_direct_key_switch(
    operation: ckks.SwitchKeyOp | ckks.RotateOp | ckks.ConjugateOp,
    config: CkksConfig,
    *,
    value: SSAValue,
    key_kind: str,
    key_symbol: str | None = None,
    evaluation_key: SSAValue | None = None,
) -> tuple[tuple[Operation, ...], SSAValue]:
    key_digit_indices = _key_switch_digit_indices(
        operation.value.type,
        config,
        operation=operation.name,
    )
    resources = _key_switch_resources(
        config,
        key_symbol=key_symbol,
        key_kind=key_kind,
        evaluation_key=evaluation_key,
    )
    component0_op, component0 = _extract_component(
        value, operation.value.type, 0
    )
    component1_op, component1 = _extract_component(
        value, operation.value.type, 1
    )
    assembly, switched = _assemble_switched_ciphertext(
        component0,
        component1,
        operation.value.type,
        operation.result.type,
        resources,
        key_digit_indices,
    )
    return (
        (*resources.operations, component0_op, component1_op, *assembly),
        switched,
    )


def _lower_switch_key(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.SwitchKeyOp):
        raise TypeError("switch-key lowering received another operation")
    input_cast, value = _cast_to_rns(operation.value)
    operations, switched = _lower_direct_key_switch(
        operation,
        config,
        value=value,
        key_symbol=operation.key_symbol.data,
        key_kind="key-switch-key",
    )
    result_cast, result = _cast_to_ckks(switched, operation.result.type)
    return LoweredCkksOperation(
        (input_cast, *operations, result_cast),
        result,
    )


def _concrete_ring_dimension(value_type: object, *, operation: str) -> int:
    ring_dimension = _state_integer(
        value_type,
        "ring_dimension",
        operation=operation,
    )
    if ring_dimension <= 0:
        raise ValueError(f"{operation} requires a positive ring dimension")
    return ring_dimension


def _lower_rotate(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.RotateOp):
        raise TypeError("rotate lowering received another operation")
    ring_dimension = _concrete_ring_dimension(
        operation.value.type,
        operation="CKKS rotation",
    )
    slot_count = ring_dimension // 2
    key_type = operation.key.type
    if not isinstance(key_type, ckks.EvaluationKeyType):
        raise TypeError("CKKS rotation requires an evaluation-key operand")
    key_state = key_type.state.data
    represented_step = key_state.get("rotation_step")
    if not isinstance(represented_step, IntegerAttr):
        raise ValueError(
            "CKKS rotation lowering requires a represented key rotation step"
        )
    shift = int(represented_step.value.data)
    normalized_shift = (shift + slot_count // 2) % slot_count - slot_count // 2
    if normalized_shift == 0:
        raise ValueError(
            "Zero CKKS rotation requires a represented clone operation"
        )
    represented_galois_element = key_state.get("galois_element")
    if not isinstance(represented_galois_element, IntegerAttr):
        raise ValueError(
            "CKKS rotation lowering requires a represented key Galois element"
        )
    galois_element = int(represented_galois_element.value.data)
    input_cast, value = _cast_to_rns(operation.value)
    key_owner = operation.key.owner
    if (
        not isinstance(key_owner, core.ResourceRefOp)
        or key_owner.symbol is None
        or key_owner.kind is None
    ):
        raise ValueError(
            "CKKS rotation lowering requires a symbolic key resource operand"
        )
    key_symbol = key_owner.symbol.data
    key_kind = key_owner.kind.data
    if key_kind != "rotation-key":
        raise ValueError(
            "CKKS rotation key resource must have kind 'rotation-key'"
        )
    evaluation_key_type = rns.EvaluationKeyResourceType.new(
        (_resource_type_state(kind=key_kind),)
    )
    key_cast, evaluation_key = UnrealizedConversionCastOp.cast_one(
        operation.key,
        evaluation_key_type,
    )
    parameter_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    parameters = core.ResourceRefOp(
        parameter_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    automorphism = rns.CoefficientAutomorphismOp(
        value,
        parameters,
        _rns_type(operation.value.type).with_state({"strides": None}),
        galois_element=galois_element,
    )
    operations, switched = _lower_direct_key_switch(
        operation,
        config,
        value=automorphism.result,
        key_kind=key_kind,
        key_symbol=key_symbol,
        evaluation_key=evaluation_key,
    )
    result_cast, result = _cast_to_ckks(switched, operation.result.type)
    return LoweredCkksOperation(
        (
            input_cast,
            key_cast,
            parameters,
            automorphism,
            *operations,
            result_cast,
        ),
        result,
    )


def _lower_conjugate(
    operation: Operation,
    config: CkksConfig,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.ConjugateOp):
        raise TypeError("conjugate lowering received another operation")
    ring_dimension = _concrete_ring_dimension(
        operation.value.type,
        operation="CKKS conjugation",
    )
    input_cast, value = _cast_to_rns(operation.value)
    parameter_type = rns.RnsParametersType.new(
        (_resource_type_state(kind="rns-parameters"),)
    )
    parameters = core.ResourceRefOp(
        parameter_type,
        symbol="active-rns-parameters",
        kind="rns-parameters",
    )
    automorphism = rns.CoefficientAutomorphismOp(
        value,
        parameters,
        _rns_type(operation.value.type).with_state({"strides": None}),
        galois_element=2 * ring_dimension - 1,
    )
    operations, switched = _lower_direct_key_switch(
        operation,
        config,
        value=automorphism.result,
        key_symbol="active-conjugation-key",
        key_kind="conjugation-key",
    )
    result_cast, result = _cast_to_ckks(switched, operation.result.type)
    return LoweredCkksOperation(
        (input_cast, parameters, automorphism, *operations, result_cast),
        result,
    )


KEY_SWITCH_LOWERINGS = (
    CkksLoweringDefinition(
        "rns-ntt-relinearize",
        ckks.RelinearizeOp,
        _lower_relinearize,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-ntt-switch-key",
        ckks.SwitchKeyOp,
        _lower_switch_key,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-ntt-rotate",
        ckks.RotateOp,
        _lower_rotate,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-ntt-conjugate",
        ckks.ConjugateOp,
        _lower_conjugate,
        is_default=True,
    ),
)
