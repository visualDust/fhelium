"""Translate CKKS value state into logical RNS types and casts."""

from __future__ import annotations

from xdsl.dialects.builtin import (
    DictionaryAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, SSAValue

from fhelium.config import CkksConfig
from fhelium.backend.rns.chain import RnsChain
from fhelium.backend.rns.decomposition import HybridRnsDecomposition

from ....ir.dialects import ckks, rns


def _resource_type_state(*, kind: str) -> DictionaryAttr:
    return DictionaryAttr({"kind": StringAttr(kind)})


def _rns_type(value_type: object) -> rns.RnsBundleType:
    if not isinstance(value_type, (ckks.CiphertextType, ckks.PlaintextType)):
        raise TypeError("CKKS lowering requires ciphertext or plaintext state")
    state = dict(value_type.state.data)
    components = state.pop("components", None)
    if components is not None:
        state["component_count"] = components
    state["value_kind"] = StringAttr(
        "ciphertext"
        if isinstance(value_type, ckks.CiphertextType)
        else "plaintext"
    )
    return rns.RnsBundleType.new((DictionaryAttr(state),))


def _cast_to_rns(value: SSAValue) -> tuple[Operation, SSAValue]:
    cast, result = UnrealizedConversionCastOp.cast_one(
        value,
        _rns_type(value.type),
    )
    return cast, result


def _cast_to_ckks(
    value: SSAValue,
    result_type: object,
) -> tuple[Operation, SSAValue]:
    if not isinstance(result_type, (ckks.CiphertextType, ckks.PlaintextType)):
        raise TypeError("CKKS lowering result type is not a CKKS value")
    cast, result = UnrealizedConversionCastOp.cast_one(value, result_type)
    return cast, result


def _state_integer(value_type: object, name: str, *, operation: str) -> int:
    if not isinstance(value_type, ckks.CiphertextType):
        raise TypeError(f"{operation} requires a CKKS ciphertext type")
    attribute = value_type.state.data.get(name)
    if not isinstance(attribute, IntegerAttr):
        raise ValueError(
            f"{operation} requires a concrete ciphertext {name!r} state"
        )
    return int(attribute.value.data)


def _key_switch_digit_indices(
    value_type: object,
    config: CkksConfig,
    *,
    operation: str,
) -> tuple[int, ...]:
    depth = _state_integer(value_type, "depth", operation=operation)
    if not 0 <= depth <= config.max_depth:
        raise ValueError(
            f"{operation} requires a public CKKS depth, got {depth}"
        )
    chain = RnsChain(
        config.num_q_primes,
        config.num_p_primes,
        tuple(len(group) for group in config.q_depth_groups),
    )
    decomposition = HybridRnsDecomposition(
        chain, config.q_moduli, config.p_moduli
    )
    return tuple(
        digit.key_digit_index for digit in decomposition.digits_at_depth(depth)
    )


def _polynomial_type(
    value_type: object,
    **updates: Attribute | None,
) -> rns.RnsBundleType:
    fields: dict[str, Attribute | None] = {
        "value_kind": StringAttr("polynomial"),
        "component_count": None,
        "strides": None,
    }
    fields.update(updates)
    return _rns_type(value_type).with_state(fields)


def _component_bundle_type(
    value_type: object,
    component_count: int,
    **updates: Attribute | None,
) -> rns.RnsBundleType:
    fields: dict[str, Attribute | None] = {
        "value_kind": StringAttr("polynomial"),
        "component_count": IntegerAttr(component_count, 64),
        "strides": None,
    }
    fields.update(updates)
    return _rns_type(value_type).with_state(fields)
