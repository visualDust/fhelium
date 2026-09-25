"""Declare operands for complete native CKKS key-switch implementations."""

from __future__ import annotations
from typing import cast
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation
from fhelium.config import CkksConfig
from fhelium.ir.dialects import ckks
from fhelium.backend.ntt._selection import select_operation_policy, table_count
from fhelium.backend.rns._preparation import (
    hybrid_decomposition_facts,
    rns_names,
    table_description,
)


def native_key_switch_requirements(
    operation: Operation, config: CkksConfig
) -> tuple[dict[str, Attribute], dict[str, dict[str, object]]] | None:
    """Select the whole implementation's internal schedule and declare operands.

    Existing numerical operands are retained by Compile. Internal NTT choices
    belong to this native whole implementation; logical NTT operations exposed
    by a lowering use the ordinary NTT implementation selector instead.
    """
    state = getattr(
        getattr(operation.operands[0].type, "state", None), "data", {}
    )
    depth = state.get("depth")
    if not isinstance(depth, IntegerAttr):
        return None

    policy = select_operation_policy(operation, log_n=config.logN)
    if policy is None:
        return None
    facts = hybrid_decomposition_facts(config, int(depth.value.data))
    if isinstance(spans := operation.attributes.get("digits"), ArrayAttr):
        facts["digits"] = tuple(
            tuple(int(x.value.data) for x in cast(ArrayAttr[IntegerAttr], span))
            for span in spans
        )
    facts["ntt_backend"] = policy.name
    names = list(rns_names(facts["digits"]))
    transforms = [
        (basis, direction)
        for basis in ("q", "qp", "p")
        for direction in ("forward", "inverse")
    ]
    for basis, direction in transforms:
        names.extend(
            f"{basis}_{direction}_{i}" for i in range(table_count(policy) + 1)
        )
    attributes: dict[str, Attribute] = {
        "ntt_backend": StringAttr(policy.name),
        "galois_generator": IntegerAttr(config.galois_generator, 64),
    }
    if isinstance(operation, ckks.RotateOp):
        step = cast(ckks.EvaluationKeyType, operation.key.type).state.data.get(
            "rotation_step"
        )
        if not isinstance(step, IntegerAttr):
            return None
        attributes["rotation_step"] = step
    for name in ("q_count", "p_count", "key_row_start"):
        attributes[name] = IntegerAttr(cast(int, facts[name]), 64)
    attributes["digits"] = ArrayAttr(
        ArrayAttr(IntegerAttr(x, 64) for x in span)
        for span in cast(tuple[tuple[int, ...], ...], facts["digits"])
    )
    return attributes, {
        name: table_description(config, int(depth.value.data), name, facts)
        for name in names
    }


def prepare_native_key_switch(implementation, operation: Operation):
    """Require concrete invocation parameters before linking native execution."""
    required = ("parameter_names", "ntt_backend")
    if any(
        name not in operation.attributes
        or operation.attributes[name] == StringAttr("unknown")
        for name in required
    ):
        raise ValueError(
            f"{operation.name} execution operands remain unresolved; run PrepareOperationOperandsPass with the required execution facts"
        )
    return implementation
