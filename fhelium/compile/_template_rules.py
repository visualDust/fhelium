"""Evaluate capture provenance using CKKS operation state equations."""

from __future__ import annotations

import json

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp

from fhelium.config import CkksConfig
from fhelium.config._derived import config_from_text
from fhelium.ir.ckks_state import arithmetic_state
from fhelium.ir.dialects import ckks, core, semantic
from fhelium.ir.dialects.torch import TensorCallOp
from fhelium.ir.dialects._common import state_attribute


RULES = {
    ckks.AddOp: "add",
    ckks.SubtractOp: "subtract",
    ckks.NegateOp: "negate",
    ckks.MultiplyOp: "multiply",
    ckks.AddPlaintextOp: "add_plaintext",
    ckks.MultiplyPlaintextOp: "multiply_plaintext",
    ckks.AddCompressedPlaintextOp: "add_compressed_plaintext",
    ckks.MultiplyCompressedPlaintextOp: "multiply_compressed_plaintext",
    ckks.AddScalarOp: "add_scalar",
    ckks.MultiplyScalarOp: "multiply_scalar",
    ckks.MultiplyIntegerScalarOp: "multiply_integer_scalar",
    ckks.ToNttOp: "to_ntt",
    ckks.FromNttOp: "from_ntt",
    ckks.ToMontgomeryResiduesOp: "to_montgomery",
    ckks.ToStandardResiduesOp: "to_standard",
    ckks.RescaleOp: "rescale",
    ckks.ModSwitchOp: "mod_switch",
    ckks.ReinterpretScaleOp: "reinterpret_scale",
    ckks.SumBatchOp: "sum_batch",
    ckks.SelectBatchOp: "select_batch",
    ckks.RotateOp: "rotate",
    ckks.RotateManyOp: "rotate_many",
    ckks.RelinearizeOp: "relinearize",
    ckks.ConjugateOp: "conjugate",
    ckks.SwitchKeyOp: "switch_key",
    ckks.RotationSumOp: "rotation_sum",
    ckks.ModulusRaiseOp: "modulus_raise",
    ckks.EncapsulatedModulusRaiseOp: "encapsulated_modulus_raise",
}


SEMANTIC_RULES = {
    semantic.AddOp: "semantic_add",
    semantic.SubtractOp: "semantic_subtract",
    semantic.MultiplyOp: "semantic_multiply",
    semantic.NegateOp: "semantic_negate",
    semantic.RollOp: "semantic_roll",
}


def rule_name(operation, record):
    if isinstance(
        operation, (ckks.RepeatOp, ckks.RepeatYieldOp, ckks.RepeatIterationOp)
    ):
        return operation.name
    if isinstance(operation, TensorCallOp):
        return "tensor_call"
    return RULES.get(
        type(operation),
        SEMANTIC_RULES.get(type(operation), record["options"].get("cast")),
    )


def attribute_value(attribute):
    if isinstance(attribute, (IntegerAttr, FloatAttr)):
        return attribute.value.data
    if isinstance(attribute, StringAttr):
        return attribute.data
    if isinstance(attribute, ArrayAttr):
        return tuple(attribute_value(item) for item in attribute)
    raise TypeError(f"Unsupported template metadata attribute {attribute}")


def state_values(typ):
    return {
        name: attribute_value(value) for name, value in typ.state.data.items()
    }


def represented_state(state):
    return {
        name: attribute
        for name, value in state.items()
        if (attribute := state_attribute(value)) is not None
    }


def _with_physical_state(type_class, state):
    """Complete storage extents directly on a copy of ordinary state facts."""
    state = dict(state)
    if "shape" not in state:
        batch = tuple(state.get("batch_shape", ()))
        if issubclass(type_class, ckks.CiphertextType):
            state["shape"] = (
                state["components"],
                *batch,
                len(state["prime_ids"]),
                state["ring_dimension"],
            )
        elif issubclass(type_class, ckks.CompressedPlaintextType):
            state["shape"] = (
                *batch,
                len(state["prime_ids"]),
                state["unique_count"],
            )
        elif issubclass(type_class, ckks.PlaintextType):
            state["shape"] = (
                *batch,
                *(
                    (len(state["prime_ids"]),)
                    if state.get("representation") == "rns"
                    else ()
                ),
                state["ring_dimension"],
            )
    if state.get("dtype") in {"torch.int32", "torch.int64"}:
        state.setdefault("requires_grad", False)
    return state


def physical_state(typ):
    """Describe a value's storage extent while retaining unknown output strides."""
    return _with_physical_state(type(typ), state_values(typ))


def _represented_result_state(state, source, source_attributes):
    """Reuse unchanged current-operand attributes, retaining only result fields."""
    attributes = {}
    for name, value in state.items():
        if (
            name in source
            and type(value) is type(source[name])
            and value == source[name]
        ):
            attributes[name] = source_attributes[name]
        elif (attribute := state_attribute(value)) is not None:
            attributes[name] = attribute
    return attributes


def supported(operation, record):
    if isinstance(
        operation, (ckks.RepeatOp, ckks.RepeatYieldOp, ckks.RepeatIterationOp)
    ):
        return True
    if isinstance(operation, (core.MaterialRefOp, core.ConstantOp, ReturnOp)):
        return True
    if isinstance(operation, UnrealizedConversionCastOp):
        return record["options"].get("cast") in {"tensor_data", "semantic_role"}
    if isinstance(operation, TensorCallOp):
        from fhelium.backend.torch import TORCH_FUNCTIONS

        return (
            operation.kind.data == "function"
            and operation.target.data in TORCH_FUNCTIONS
            and (
                record["options"].get("tensor_call") == "registered"
                or record["options"].get("view") == "append_unit_axis"
            )
        )
    if type(operation) in SEMANTIC_RULES:
        return all(not result.type.state.data for result in operation.results)
    return type(operation) in RULES


def operation_config(operation, workspace):
    encoded = operation.attributes.get("ckks_config")
    if isinstance(encoded, StringAttr):
        return config_from_text(encoded.data)
    return workspace.get(CkksConfig)


def result_types(operation, record, workspace):
    """Return current result types and refresh state-derived intrinsic attributes."""
    if isinstance(operation, UnrealizedConversionCastOp):
        if record["options"].get("cast") == "semantic_role":
            return (
                type(operation.results[0].type)().with_state(
                    operation.operands[0].type.state.data
                ),
            )
        state = physical_state(operation.operands[0].type)
        return (core.MessageType().with_state(represented_state(state)),)
    if isinstance(operation, TensorCallOp):
        from .frontend._tensor_capture import tensor_call_result_type

        if record["options"].get("view") == "append_unit_axis":
            shape = physical_state(operation.operands[0].type)["shape"]
            descriptor = json.loads(operation.argument_descriptor.data)
            descriptor["args"]["items"][1] = {
                "kind": "tuple",
                "items": [
                    {"kind": "literal", "value": size} for size in (*shape, 1)
                ],
            }
            operation.attributes["fhelium.call.arguments"] = StringAttr(
                json.dumps(descriptor)
            )
        return (tensor_call_result_type(operation),)
    if type(operation) in SEMANTIC_RULES:
        # Scheme state belongs to the subsequent scheduling passes. These expressions have no CKKS state facts yet.
        return tuple(result.type for result in operation.results)
    if isinstance(operation, (core.MaterialRefOp, core.ConstantOp, ReturnOp)):
        return tuple(result.type for result in operation.results)
    rule = RULES[type(operation)]
    if (
        isinstance(
            operation,
            (ckks.AddCompressedPlaintextOp, ckks.MultiplyCompressedPlaintextOp),
        )
        and operation.attributes.get("compression_layout")
        == StringAttr("strided_sparse")
        and len(operation.parameters) == 1
    ):
        # The variadic operand group contains a semantic implicit-value Tensor before numerical tables.
        operation.attributes["parameter_names"] = ArrayAttr(
            (StringAttr("implicit_plaintext"),)
        )
    source = state_values(operation.operands[0].type)
    options = {
        name: attribute_value(value)
        for name, value in operation.attributes.items()
        if isinstance(value, (ArrayAttr, IntegerAttr, FloatAttr, StringAttr))
    }
    options.update(record["options"])
    rhs = (
        state_values(operation.operands[1].type)
        if rule
        in {
            "add",
            "subtract",
            "multiply",
            "add_plaintext",
            "multiply_plaintext",
            "add_compressed_plaintext",
            "multiply_compressed_plaintext",
        }
        else None
    )
    if isinstance(operation, ckks.RotationSumOp):
        for value in operation.values[1:]:
            if state_values(value.type) != source:
                raise ValueError(
                    "Rotation-sum terms require matching value states"
                )
    state = arithmetic_state(
        rule,
        source,
        rhs=rhs,
        config=operation_config(operation, workspace),
        options=options,
    )
    if rule in {"add_scalar", "multiply_scalar"} and options.get(
        "scalar_scale_from_input"
    ):
        operation.attributes["scalar_scale"] = state_attribute(source["scale"])
    if rule == "mod_switch" and options.get("depth_offset") is not None:
        operation.attributes["target_depth"] = state_attribute(state["depth"])
    source_attributes = operation.operands[0].type.state.data
    return tuple(
        type(result.type)(
            _represented_result_state(
                _with_physical_state(type(result.type), state),
                source,
                source_attributes,
            )
        )
        for result in operation.results
    )


def read_metadata(value, field):
    state = physical_state(value.type)
    field = {"modulus_basis": "basis", "component_count": "components"}.get(
        field, field
    )
    if field == "limb_count":
        return len(state["prime_ids"])
    if field == "batch_size":
        import math

        return math.prod(state["batch_shape"])
    if field == "is_batched":
        return bool(state["batch_shape"])
    if field == "ndim":
        return len(state["shape"])
    if field.startswith("is_"):
        return state.get("representation") == field.removeprefix("is_")
    if field == "layout":
        return "torch.strided"
    return state[field]


# These choices affect capture topology or select a public representation.
# Physical strides and row count remain current instance facts.
STRUCTURAL_FIELDS = frozenset(
    {
        "dtype",
        "device",
        "components",
        "batch_shape",
        "ring_dimension",
        "basis",
        "polynomial_domain",
        "residue_representation",
        "representation",
        "compression_layout",
        "unique_count",
    }
)


def leaf_compatible(sample, current):
    if type(sample) is not type(current):
        return False
    if isinstance(sample, ckks.EvaluationKeyType):
        fields = STRUCTURAL_FIELDS | {
            "prime_ids",
            "digit_prime_ids",
            "shape",
        }
    elif isinstance(
        sample,
        (ckks.CiphertextType, ckks.PlaintextType, ckks.CompressedPlaintextType),
    ):
        fields = STRUCTURAL_FIELDS
    else:
        fields = STRUCTURAL_FIELDS | {"shape"}
    return all(
        sample.state.data.get(name) == current.state.data.get(name)
        for name in fields
    )
