"""Declare operands for complete native CKKS key-switch implementations."""

from __future__ import annotations
from typing import cast
from collections.abc import Mapping
from math import prod
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation
from fhelium.backend.rns.decomposition import key_decomposition_facts
from fhelium.config import CkksConfig
from fhelium.config._derived import derived_fact
from fhelium.ir.dialects import ckks
from fhelium.ir.dialects._common import state_attribute
from fhelium.backend.ntt._selection import select_operation_policy, table_count
from fhelium.backend.rns._preparation import (
    hybrid_decomposition_facts,
    joint_rescale_attributes,
    joint_rescale_facts,
    rns_names,
    table_description,
)


def key_switch_runtime_attributes(
    facts: Mapping[str, object],
) -> dict[str, object]:
    """Select the native invocation fields from prepared mathematical facts.

    Catalog row identities and stored key layouts describe numerical operands;
    execution consumes their Tensor data with counts and physical digit spans.
    """
    attributes = {
        name: facts[name]
        for name in (
            "q_count",
            "p_count",
            "key_row_start",
            "key_p_row_start",
            "digits",
        )
    }
    if facts.get("key_q_rows"):
        attributes["key_q_rows"] = facts["key_q_rows"]
    attributes.update(
        (name, facts[name])
        for name in (
            "ntt_backend",
            "galois_generator",
            "basis_conversion",
            "joint_drop_count",
            "joint_drop_is_prefix",
            "joint_drop_prime_ids",
            "joint_target_prime_ids",
        )
        if name in facts
    )
    return attributes


def key_seed_rounds(key) -> int:
    """Return the ChaCha round count a compact key value's state records; 0 when expanded."""
    state = getattr(getattr(key, "type", None), "state", None)
    rounds = None if state is None else state.data.get("seed_rounds")
    return int(rounds.value.data) if isinstance(rounds, IntegerAttr) else 0


def operation_key_rounds(operation: Operation) -> int:
    """Return the generated-key round count of an operation's key operands.

    Operations with ``key_seeds`` operands consume compact keys whose state
    records ``seed_rounds``; others consume expanded keys.
    """
    seeds = getattr(operation, "key_seeds", ())
    if not seeds:
        return 0
    for name in ("keys", "key", "low_key"):
        keys = getattr(operation, name, None)
        if keys is None:
            continue
        first = keys[0] if name == "keys" else keys
        return key_seed_rounds(first)
    return 0


def operation_key_facts(operation: Operation, config: CkksConfig, depth: int):
    """Read mathematical key layout from supplied operands, retaining unknowns."""
    keys = (
        tuple(getattr(operation, "keys"))
        if hasattr(operation, "keys")
        else (getattr(operation, "key"),)
    )
    layouts = []
    for key in keys:
        state = getattr(key.type, "state", None)
        if state is None:
            return None
        rows = state.data.get("prime_ids")
        digits = state.data.get("digit_prime_ids")
        if not isinstance(rows, ArrayAttr) or not isinstance(digits, ArrayAttr):
            return None
        layouts.append(
            (
                tuple(int(i.value.data) for i in rows),
                tuple(
                    tuple(int(i.value.data) for i in digit) for digit in digits
                ),
            )
        )
    if not layouts:
        return hybrid_decomposition_facts(config, depth)
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError(
            "A shared key-switch execution group requires matching key rows and digit supports"
        )
    return key_decomposition_facts(config, depth, *layouts[0])


_UNKEYED_ATTRIBUTES = frozenset(
    {"rotation_step", "parameter_names", "ckks_config"}
)


def native_key_switch_requirements(
    operation: Operation, config: CkksConfig
) -> tuple[dict[str, Attribute], dict[str, dict[str, object]]] | None:
    """Select the whole implementation's internal schedule and declare operands.

    Existing numerical operands are retained by Compile. Internal NTT choices
    belong to this native whole implementation; logical NTT operations exposed
    by a lowering use the ordinary NTT implementation selector instead.

    The requirements depend on the operation name, the input depth and device, the key row and digit layouts, and the operation's other attributes and properties. They are derived once per such key and configuration; the rotation step selects only Galois tables and is copied onto each result.
    """
    state = getattr(
        getattr(operation.operands[0].type, "state", None), "data", {}
    )
    keys = (
        tuple(getattr(operation, "keys"))
        if hasattr(operation, "keys")
        else (getattr(operation, "key"),)
    )
    layouts = []
    for key in keys:
        key_state = getattr(getattr(key.type, "state", None), "data", {})
        layouts.append(
            (key_state.get("prime_ids"), key_state.get("digit_prime_ids"))
        )
    requirements = derived_fact(
        config,
        "native_key_switch_requirements",
        (
            operation.name,
            state.get("depth"),
            state.get("device"),
            tuple(layouts),
            tuple(
                sorted(
                    item
                    for item in operation.attributes.items()
                    if item[0] not in _UNKEYED_ATTRIBUTES
                )
            ),
            tuple(sorted(operation.properties.items())),
        ),
        lambda: _native_key_switch_requirements(operation, config),
    )
    if requirements is None:
        return None
    attributes, descriptions = requirements
    attributes = dict(attributes)
    if isinstance(operation, ckks.RotateOp):
        attributes["rotation_step"] = operation.rotation_step
    if rounds := operation_key_rounds(operation):
        attributes["key_rounds"] = IntegerAttr(rounds, 64)
    return attributes, dict(descriptions)


def _native_key_switch_requirements(
    operation: Operation, config: CkksConfig
) -> tuple[dict[str, Attribute], dict[str, dict[str, object]]] | None:
    rounding = cast(
        StringAttr,
        operation.attributes.get("moddown_rounding", StringAttr("nearest")),
    )
    state = getattr(
        getattr(operation.operands[0].type, "state", None), "data", {}
    )
    depth = state.get("depth")
    if not isinstance(depth, IntegerAttr):
        return None

    policy = select_operation_policy(operation, log_n=config.logN)
    if policy is None:
        return None
    facts = operation_key_facts(operation, config, int(depth.value.data))
    if facts is None:
        return None
    if isinstance(spans := operation.attributes.get("digits"), ArrayAttr):
        facts["digits"] = tuple(
            tuple(int(x.value.data) for x in cast(ArrayAttr[IntegerAttr], span))
            for span in spans
        )
    method = cast(
        StringAttr,
        operation.attributes.get("basis_conversion", StringAttr("exact")),
    ).data
    facts["basis_conversion"] = method
    facts["moddown_rounding"] = rounding.data
    facts["ntt_backend"] = policy.name
    rescale = isinstance(operation, ckks.RelinearizeOp) and bool(
        operation.rescale.value.data
    )
    if rescale:
        facts.update(
            joint_rescale_facts(
                config,
                int(depth.value.data),
                cast(tuple[int, ...], facts["q_prime_ids"]),
            ),
            rescale_rounding=cast(
                ckks.RelinearizeOp, operation
            ).rescale_rounding.data,
            output_domain=cast(
                ckks.RelinearizeOp, operation
            ).output_domain.data,
        )
    names = list(
        rns_names(
            facts["digits"],
            moddown_rounding=rounding.data,
            basis_conversion=method,
            rescale=rescale,
            restored=rescale
            and not set(cast(tuple[int, ...], facts["joint_target_prime_ids"]))
            <= set(cast(tuple[int, ...], facts["q_prime_ids"])),
            rescale_rounding=str(facts.get("rescale_rounding", "nearest")),
        )
    )
    transforms = [
        (basis, direction)
        for basis in ("q", "qp", "p")
        for direction in ("forward", "inverse")
    ]
    if rescale:
        transforms = [
            ("q", "inverse"),
            ("qp", "forward"),
            ("joint_dropped", "inverse"),
            (
                "joint_remaining",
                "forward"
                if cast(ckks.RelinearizeOp, operation).output_domain.data
                == "ntt"
                else "inverse",
            ),
        ]
    for basis, direction in transforms:
        names.extend(
            f"{basis}_{direction}_{i}" for i in range(table_count(policy) + 1)
        )
    attributes: dict[str, Attribute] = {
        "ntt_backend": StringAttr(policy.name),
        "moddown_rounding": rounding,
        "basis_conversion": StringAttr(method),
        "galois_generator": IntegerAttr(config.galois_generator, 64),
    }
    if rescale:
        attributes.update(joint_rescale_attributes(facts))
    for name in ("q_count", "p_count", "key_row_start", "key_p_row_start"):
        attributes[name] = cast(
            Attribute, state_attribute(cast(int, facts[name]))
        )
    attributes["digits"] = cast(
        Attribute,
        state_attribute(
            tuple(
                tuple(span)
                for span in cast(tuple[tuple[int, ...], ...], facts["digits"])
            )
        ),
    )
    if facts.get("key_q_rows"):
        attributes["key_q_rows"] = cast(
            Attribute,
            state_attribute(tuple(cast(tuple[int, ...], facts["key_q_rows"]))),
        )
    return attributes, {
        name: table_description(config, int(depth.value.data), name, facts)
        for name in names
    }


def rotation_sum_requirements(
    operation: ckks.RotationSumOp, config: CkksConfig
) -> tuple[dict[str, Attribute], dict[str, dict[str, object]]] | None:
    requirements = native_key_switch_requirements(operation, config)
    if requirements is None:
        return None
    attributes, descriptions = requirements
    depth = int(
        cast(
            IntegerAttr,
            cast(ckks.CiphertextType, operation.values[0].type).state.data[
                'depth'
            ],
        ).value.data
    )
    rescale = bool(operation.rescale.value.data)
    rounding = operation.moddown_rounding.data
    dropped, restored = (
        config.basis_transition(depth, depth + 1) if rescale else ((), ())
    )
    if restored:
        return None
    drop = len(dropped)
    q_count = len(config.q_prime_ids(depth))
    M = prod(config.moduli[i] for i in dropped)
    facts = operation_key_facts(operation, config, depth)
    if facts is None:
        return None
    p_ids = cast(tuple[int, ...], facts["p_prime_ids"])
    attributes.update(
        {
            'sum_remaining_q_count': IntegerAttr(q_count - drop, 64),
            'sum_dropped_q_count': IntegerAttr(drop, 64),
            'sum_divisor': StringAttr(
                str(prod(config.moduli[i] for i in p_ids) * M)
            ),
            'sum_rescale_divisor': StringAttr(str(M)),
        }
    )
    names = [
        'sum_p_montgomery',
        'sum_quotient_parameters',
        'sum_quotient_inverses',
    ]
    if rescale:
        names.append('sum_quotient_order')
    if rounding == 'nearest':
        names.append('sum_quotient_half')
    descriptions.update(
        {
            name: {
                'kind': 'rotation_sum_table',
                'basis_conversion': operation.basis_conversion.data,
                'depth': depth,
                'name': name,
                'rescale': rescale,
                'rounding': rounding,
                **{
                    field: facts[field]
                    for field in ('key_prime_ids', 'digit_prime_ids')
                    if field in facts
                },
            }
            for name in names
        }
    )
    return attributes, descriptions


def prepare_native_key_switch(implementation, operation: Operation):
    """Require concrete invocation parameters before linking native execution."""
    required = ("parameter_names", "ntt_backend")
    if any(
        name not in operation.attributes
        or operation.attributes[name] == StringAttr("unknown")
        for name in required
    ):
        raise ValueError(
            f"{operation.name} execution operands remain unresolved; check the configuration, layout and implementation requirements reported during compilation"
        )
    return implementation
