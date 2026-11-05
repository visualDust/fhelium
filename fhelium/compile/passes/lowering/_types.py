"""Translate CKKS value state into logical RNS types and casts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, SSAValue


from ....ir.dialects import ckks, rns


def _key_switch_qp_basis(
    source_type: Attribute,
    keys: Sequence[SSAValue],
    fields: Mapping[str, Attribute],
) -> tuple[ArrayAttr | None, IntegerAttr | None]:
    """Read supplied Q/key identities and the separately declared physical extent."""
    q_count, p_count = fields.get("q_count"), fields.get("p_count")
    row_count = (
        IntegerAttr(q_count.value.data + p_count.value.data, 64)
        if isinstance(q_count, IntegerAttr) and isinstance(p_count, IntegerAttr)
        else None
    )
    source_state = getattr(getattr(source_type, "state", None), "data", {})
    q_ids = source_state.get("prime_ids")
    if not isinstance(q_ids, ArrayAttr) or not all(
        isinstance(i, IntegerAttr) for i in q_ids
    ):
        return None, row_count
    auxiliary: tuple[int, ...] | None = None
    for key in keys:
        state = getattr(getattr(key.type, "state", None), "data", {})
        stored_ids = state.get("prime_ids")
        supports = state.get("digit_prime_ids")
        if not (
            isinstance(stored_ids, ArrayAttr)
            and isinstance(supports, ArrayAttr)
            and all(isinstance(rows, ArrayAttr) for rows in supports)
        ):
            return None, row_count
        stored_q = tuple(
            item for rows in supports for item in cast(ArrayAttr, rows)
        )
        if not stored_q or not all(
            isinstance(i, IntegerAttr) for i in (*stored_ids, *stored_q)
        ):
            return None, row_count
        stored = tuple(int(cast(IntegerAttr, i).value.data) for i in stored_ids)
        supported = tuple(
            int(cast(IntegerAttr, i).value.data) for i in stored_q
        )
        if stored[: len(supported)] != supported or len(supported) == len(
            stored
        ):
            return None, row_count
        p_ids = stored[len(supported) :]
        if auxiliary is not None and p_ids != auxiliary:
            return None, row_count
        auxiliary = p_ids
    if auxiliary is None:
        return None, row_count
    active_q = tuple(int(cast(IntegerAttr, i).value.data) for i in q_ids)
    return ArrayAttr(
        IntegerAttr(i, 64) for i in (*active_q, *auxiliary)
    ), row_count


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
    *,
    row_count: IntegerAttr | None = None,
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
            else row_count
            if row_count is not None
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
    *,
    row_count: IntegerAttr | None = None,
    **updates: Attribute | None,
) -> rns.RnsBundleType:
    return _bundle_type(value_type, None, updates, row_count=row_count)


def _component_bundle_type(
    value_type: object,
    component_count: int,
    *,
    row_count: IntegerAttr | None = None,
    **updates: Attribute | None,
) -> rns.RnsBundleType:
    return _bundle_type(
        value_type, component_count, updates, row_count=row_count
    )
