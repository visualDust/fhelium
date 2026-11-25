"""Lower CKKS arithmetic operations into logical RNS compositions."""

from __future__ import annotations

from typing import cast

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Operation

from fhelium.config import CkksConfig
from ..._materials import (
    declare_material,
    rns_parameter_identity,
    parameter_description,
)

from ....ir.dialects import ckks, rns
from ._core import (
    CkksLoweringDefinition,
    LoweredCkksOperation,
)
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _polynomial_type,
    _rns_type,
)


def _parameters(operation: Operation, config, compilation):
    supplied = tuple(getattr(operation, "parameters", ()))
    if supplied:
        return (), supplied[0]
    state = getattr(
        getattr(operation.operands[0].type, "state", None), "data", {}
    )
    ids = state.get("prime_ids")
    prime_ids = (
        tuple(int(item.value.data) for item in ids)
        if isinstance(ids, ArrayAttr)
        and all(isinstance(item, IntegerAttr) for item in ids)
        else ()
    )
    physical = {
        name: state[name] for name in ("dtype", "device") if name in state
    }
    description = (
        {"kind": "rns_parameters", "prime_ids": list(prime_ids)}
        if config is None
        else parameter_description(
            config, "rns_parameters", prime_ids=list(prime_ids)
        )
    )
    return declare_material(
        compilation,
        operation,
        "rns_parameters",
        rns.RnsParametersType().with_state(physical),
        description,
        identity=rns_parameter_identity(config, prime_ids, physical),
    )


def _parameter_descriptions(parameter_ops, operation, config):
    state = getattr(
        getattr(operation.operands[0].type, "state", None), "data", {}
    )
    ids = state.get("prime_ids")
    fields = (
        {"prime_ids": [int(item.value.data) for item in ids]}
        if isinstance(ids, ArrayAttr)
        and all(isinstance(item, IntegerAttr) for item in ids)
        else {}
    )
    description = (
        {"kind": "rns_parameters", **fields}
        if config is None
        else parameter_description(config, "rns_parameters", **fields)
    )
    return {reference.symbol.data: description for reference in parameter_ops}


def _lower_add(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.AddOp):
        raise TypeError("add lowering received another operation")
    lhs_cast, lhs = _cast_to_rns(operation.lhs)
    rhs_cast, rhs = _cast_to_rns(operation.rhs)
    parameter_ops, parameters = _parameters(operation, config, compilation)
    counts = tuple(
        cast(ckks.CiphertextType, value.type).state.data.get("components")
        for value in (operation.lhs, operation.rhs)
    )
    result_type = operation.result.type
    operations: list[Operation] = [lhs_cast, rhs_cast, *parameter_ops]
    if None in counts or counts[0] == counts[1]:
        logical = rns.AddStandardOp(
            lhs, rhs, parameters, _rns_type(result_type)
        )
        operations.append(logical)
        packed = logical.result
    else:
        # Components present in only one operand pass through unchanged.
        counts = tuple(
            int(cast(IntegerAttr, count).value.data) for count in counts
        )
        polynomial = _polynomial_type(result_type)
        components = []
        for j in range(max(counts)):
            terms = []
            for value, count in zip((lhs, rhs), counts, strict=True):
                if j < count:
                    extract = rns.ExtractComponentOp(
                        value, polynomial, component=j
                    )
                    operations.append(extract)
                    terms.append(extract.result)
            if len(terms) == 2:
                total = rns.AddStandardOp(
                    terms[0], terms[1], parameters, polynomial
                )
                operations.append(total)
                terms = [total.result]
            components.append(terms[0])
        pack = (
            rns.PackThreeComponentsOp(
                components[0],
                components[1],
                components[2],
                _rns_type(result_type),
            )
            if len(components) == 3
            else rns.PackTwoComponentsOp(
                components[0], components[1], _rns_type(result_type)
            )
        )
        operations.append(pack)
        packed = pack.result
    result_cast, result = _cast_to_ckks(packed, result_type)
    return LoweredCkksOperation(
        (*operations, result_cast),
        result,
        _parameter_descriptions(parameter_ops, operation, config),
    )


def _lower_batch_sum(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.SumBatchOp):
        raise TypeError("batch-sum lowering received another operation")
    source_cast, source = _cast_to_rns(operation.value)
    parameter_ops, parameters = _parameters(operation, config, compilation)
    logical = rns.SumStandardBatchOp(
        source,
        parameters,
        _rns_type(operation.result.type),
        dim=int(operation.axis.value.data) + 1,
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (source_cast, *parameter_ops, logical, result_cast),
        result,
        _parameter_descriptions(parameter_ops, operation, config),
    )


def _lower_subtract(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.SubtractOp):
        raise TypeError("subtract lowering received another operation")
    lhs_cast, lhs = _cast_to_rns(operation.lhs)
    rhs_cast, rhs = _cast_to_rns(operation.rhs)
    parameter_ops, parameters = _parameters(operation, config, compilation)
    logical = rns.SubtractStandardOp(
        lhs,
        rhs,
        parameters,
        _rns_type(operation.result.type),
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (lhs_cast, rhs_cast, *parameter_ops, logical, result_cast),
        result,
        _parameter_descriptions(parameter_ops, operation, config),
    )


def _lower_negate(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.NegateOp):
        raise TypeError("negate lowering received another operation")
    input_cast, value = _cast_to_rns(operation.value)
    parameter_ops, parameters = _parameters(operation, config, compilation)
    logical = rns.NegateStandardOp(
        value,
        parameters,
        _rns_type(operation.result.type),
    )
    result_cast, result = _cast_to_ckks(logical.result, operation.result.type)
    return LoweredCkksOperation(
        (input_cast, *parameter_ops, logical, result_cast),
        result,
        _parameter_descriptions(parameter_ops, operation, config),
    )


def _lower_plaintext_arithmetic(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(
        operation, (ckks.AddPlaintextOp, ckks.MultiplyPlaintextOp)
    ):
        raise TypeError(
            "Plaintext arithmetic lowering received another operation"
        )
    ciphertext_cast, ciphertext = _cast_to_rns(operation.ciphertext)
    plaintext_cast, plaintext = _cast_to_rns(operation.plaintext)
    parameter_ops, parameters = _parameters(operation, config, compilation)
    operation_type: type[Operation] = (
        rns.AddPlaintextOp
        if isinstance(operation, ckks.AddPlaintextOp)
        else rns.MultiplyPlaintextOp
    )
    polynomial_domain = cast(
        StringAttr,
        cast(ckks.CiphertextType, operation.ciphertext.type).state.data[
            "polynomial_domain"
        ],
    )
    logical = operation_type.create(
        operands=(ciphertext, plaintext, parameters),
        result_types=(_rns_type(operation.result.type),),
        attributes=(
            {"polynomial_domain": StringAttr(polynomial_domain.data)}
            if isinstance(operation, ckks.AddPlaintextOp)
            else {}
        ),
    )
    result_cast, result = _cast_to_ckks(
        logical.results[0], operation.result.type
    )
    return LoweredCkksOperation(
        (ciphertext_cast, plaintext_cast, *parameter_ops, logical, result_cast),
        result,
        _parameter_descriptions(parameter_ops, operation, config),
    )


def _lower_multiply(
    operation: Operation,
    config: CkksConfig | None,
    compilation,
) -> LoweredCkksOperation:
    if not isinstance(operation, ckks.MultiplyOp):
        raise TypeError(
            "Ciphertext multiply lowering received another operation"
        )
    lhs_cast, lhs = _cast_to_rns(operation.lhs)
    rhs_cast, rhs = _cast_to_rns(operation.rhs)
    parameter_ops, parameters = _parameters(operation, config, compilation)
    product = rns.TensorProductOp(
        lhs, rhs, parameters, _rns_type(operation.result.type)
    )
    result_cast, result = _cast_to_ckks(product.result, operation.result.type)
    return LoweredCkksOperation(
        (lhs_cast, rhs_cast, *parameter_ops, product, result_cast),
        result,
        _parameter_descriptions(parameter_ops, operation, config),
    )


ARITHMETIC_LOWERINGS = (
    CkksLoweringDefinition(
        "rns-standard-add",
        ckks.AddOp,
        _lower_add,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-standard-batch-sum",
        ckks.SumBatchOp,
        _lower_batch_sum,
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
