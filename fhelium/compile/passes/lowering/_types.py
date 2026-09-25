"""Translate CKKS value state into logical RNS types and casts."""

from __future__ import annotations

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, SSAValue


from ....ir.dialects import ckks, rns


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


def _bundle_type(
    value_type: object,
    components: int | None,
    updates: dict[str, Attribute | None],
) -> rns.RnsBundleType:
    source = _rns_type(value_type)
    fields: dict[str, Attribute | None] = {
        "value_kind": StringAttr("polynomial"),
        "component_count": None
        if components is None
        else IntegerAttr(components, 64),
        "strides": None,
    }
    shape = source.state.data.get("shape")
    if isinstance(shape, ArrayAttr):
        dims = tuple(shape)
        batch = (
            dims[1:-2]
            if isinstance(value_type, ckks.CiphertextType)
            else dims[:-2]
        )
        ids = updates.get("prime_ids", source.state.data.get("prime_ids"))
        rows = (
            IntegerAttr(len(ids), 64)
            if isinstance(ids, ArrayAttr)
            else dims[-2]
        )
        fields["shape"] = ArrayAttr(
            (
                *(
                    (IntegerAttr(components, 64),)
                    if components is not None
                    else ()
                ),
                *batch,
                rows,
                dims[-1],
            )
        )
    fields.update(updates)
    return source.with_state(fields)


def _polynomial_type(
    value_type: object,
    **updates: Attribute | None,
) -> rns.RnsBundleType:
    return _bundle_type(value_type, None, updates)


def _component_bundle_type(
    value_type: object,
    component_count: int,
    **updates: Attribute | None,
) -> rns.RnsBundleType:
    return _bundle_type(value_type, component_count, updates)
