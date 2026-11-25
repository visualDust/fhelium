"""Describe hybrid RNS tables and native ModDown operand requirements."""

from __future__ import annotations
from typing import cast
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation
from fhelium.config import CkksConfig
from fhelium.ir.dialects import rns
from fhelium.backend.ntt._selection import select_operation_policy, table_count
from .layout import RnsLayout
from .crt import CRT_OPERANDS
from .tables import (
    MODDOWN_RESCALE_TABLE_NAMES,
    MODDOWN_RESTORE_TABLE_NAMES,
    centered_moddown_rescale_names,
)


def joint_rescale_facts(
    config: CkksConfig, depth: int, q_prime_ids: tuple[int, ...]
) -> dict[str, object]:
    """Describe the rescale from ``depth`` fused into a key-switch ModDown."""

    drop_ids, _ = config.basis_transition(depth, depth + 1)
    return {
        "rescale": True,
        "joint_drop_prime_ids": drop_ids,
        "joint_target_prime_ids": config.q_prime_ids(depth + 1),
        "joint_drop_count": len(drop_ids),
        "joint_drop_is_prefix": drop_ids == q_prime_ids[: len(drop_ids)],
    }


def joint_rescale_attributes(facts: dict[str, object]) -> dict[str, Attribute]:
    """Return the operation attributes of :func:`joint_rescale_facts`."""

    drop_ids = cast(tuple[int, ...], facts["joint_drop_prime_ids"])
    return {
        "joint_drop_count": IntegerAttr(len(drop_ids), 64),
        "joint_drop_is_prefix": IntegerAttr(
            int(cast(bool, facts["joint_drop_is_prefix"])), 64
        ),
        "joint_drop_prime_ids": ArrayAttr(IntegerAttr(i, 64) for i in drop_ids),
        "joint_target_prime_ids": ArrayAttr(
            IntegerAttr(i, 64)
            for i in cast(tuple[int, ...], facts["joint_target_prime_ids"])
        ),
    }


def hybrid_decomposition_facts(
    config: CkksConfig, depth: int
) -> dict[str, object]:
    """Describe the default mathematical hybrid decomposition at one depth."""
    layout = RnsLayout.from_config(config)
    q_ids = layout.prime_ids(depth)
    if q_ids != tuple(range(q_ids[0], q_ids[0] + len(q_ids))):
        raise ValueError(
            f"depth {depth} is not served by the default key layout; "
            "supply a key whose rows cover its Q basis"
        )
    return {
        "digits": tuple(
            (
                s.component_row_ids[0],
                s.component_row_ids[-1] + 1,
                s.key_digit_index,
            )
            for s in layout.digit_specs(depth)
        ),
        "q_count": layout.row_count(depth),
        "p_count": config.num_p_primes,
        "q_prime_ids": layout.prime_ids(depth),
        "p_prime_ids": tuple(
            range(config.num_q_primes, config.total_num_primes)
        ),
        "qp_prime_ids": layout.prime_ids(depth, include_p=True),
        "key_row_start": q_ids[0],
        "key_p_row_start": config.num_q_primes,
        "key_q_rows": (),
    }


def rns_names(
    digits,
    *,
    moddown_rounding: str = "nearest",
    basis_conversion: str = "exact",
    rescale: bool = False,
    restored: bool = False,
    rescale_rounding: str = "nearest",
) -> tuple[str, ...]:
    """List RNS-only preparation tables, independently of an NTT schedule."""
    if basis_conversion not in {"exact", "crt", "normalized_crt"}:
        raise ValueError("basis_conversion must be exact, crt, or normalized_crt")
    names = ["q_parameters", "qp_parameters"]
    names.extend(
        MODDOWN_RESCALE_TABLE_NAMES
        if rescale
        else ("moddown_inverses", "p_inverse")
    )
    if rescale and restored:
        names.extend(MODDOWN_RESTORE_TABLE_NAMES)
    if not rescale and moddown_rounding == "nearest":
        names.append("moddown_half")
    if basis_conversion == "normalized_crt":
        names.extend(
            centered_moddown_rescale_names(rescale_rounding)
            if rescale
            else (f"moddown_crt_{name}" for name in CRT_OPERANDS)
        )
    for i, (start, stop, _) in enumerate(digits):
        if basis_conversion != "exact":
            names.extend(f"digit{i}_crt_{name}" for name in CRT_OPERANDS)
            continue
        names.append(f"digit{i}_parameters")
        if stop - start > 1:
            names.extend(
                (
                    f"digit{i}_normalizers",
                    f"digit{i}_propagation",
                    *(f"digit{i}_reduction{part}" for part in range(4)),
                )
            )
        names.append(f"digit{i}_extension")
    return tuple(names)


def table_description(
    config: CkksConfig, depth: int | None, name: str, facts: dict[str, object]
) -> dict[str, object]:
    """Describe one table by its own mathematical or NTT-layout requirements."""
    q_ids = cast(tuple[int, ...], facts["q_prime_ids"])
    p_ids = cast(tuple[int, ...], facts["p_prime_ids"])
    drop_ids = cast(tuple[int, ...], facts.get("joint_drop_prime_ids", ()))
    target = cast(
        tuple[int, ...],
        facts.get(
            "joint_target_prime_ids",
            tuple(i for i in q_ids if i not in drop_ids),
        ),
    )
    ids = {
        "q": q_ids,
        "p": p_ids,
        "qp": (*q_ids, *p_ids),
        "joint_remaining": target,
        "joint_dropped": (*drop_ids, *p_ids),
        "joint_ordered": (*target, *drop_ids, *p_ids),
        "joint_product": (*q_ids, *p_ids),
    }
    if name.endswith("_parameters") and name.removesuffix("_parameters") in ids:
        return {
            "kind": "rns_parameters",
            "prime_ids": list(ids[name.removesuffix("_parameters")]),
        }
    parts = name.rsplit("_", 2)
    if (
        len(parts) == 3
        and parts[0] in ids
        and parts[1] in {"forward", "inverse"}
    ):
        index = int(parts[2])
        if index == 0:
            return {"kind": "rns_parameters", "prime_ids": list(ids[parts[0]])}
        return {
            "kind": "ntt_table",
            "prime_ids": list(ids[parts[0]]),
            "direction": parts[1],
            "index": index,
            "ntt_backend": facts["ntt_backend"],
        }
    if (
        name in MODDOWN_RESCALE_TABLE_NAMES
        or name in MODDOWN_RESTORE_TABLE_NAMES
        or name.startswith("joint_crt_")
    ):
        return {
            "kind": "joint_quotient_table",
            "name": name,
            "q_prime_ids": list(q_ids),
            "p_prime_ids": list(p_ids),
            "drop_prime_ids": list(drop_ids),
            "target_prime_ids": list(target),
            "moddown_rounding": facts.get("moddown_rounding", "nearest"),
            "rescale_rounding": facts.get("rescale_rounding", "nearest"),
            "basis_conversion": facts.get("basis_conversion", "exact"),
        }
    return {
        "kind": "key_switch_table",
        "moddown_rounding": facts.get("moddown_rounding", "nearest"),
        "basis_conversion": facts.get("basis_conversion", "exact"),
        "rescale": facts.get("rescale", False),
        "rescale_rounding": facts.get("rescale_rounding", "nearest"),
        "output_domain": facts.get("output_domain", "coefficient"),
        "depth": depth,
        "name": name,
        **{
            name: facts[name]
            for name in (
                "key_prime_ids",
                "digit_prime_ids",
                "q_prime_ids",
                "p_prime_ids",
            )
            if name in facts
        },
        "digits": [
            list(span)
            for span in cast(
                tuple[tuple[int, ...], ...], facts.get("digits", ())
            )
        ],
    }


def native_moddown_requirements(operation: Operation, config: CkksConfig):
    """Declare the NTT and RNS operands of native auxiliary-basis removal."""
    if isinstance(operation, rns.ModDownAddRescaleOp):
        return _joint_moddown_requirements(operation, config)
    rounding = cast(
        StringAttr, operation.attributes.get("rounding", StringAttr("nearest"))
    )
    state = getattr(
        getattr(operation.operands[0].type, "state", None), "data", {}
    )
    depth = state.get("depth")
    if not isinstance(depth, IntegerAttr):
        return None
    represented_rows = state.get("prime_ids")
    if not isinstance(represented_rows, ArrayAttr):
        return None
    qp_ids = tuple(int(i.value.data) for i in represented_rows)
    q_ids = config.q_prime_ids(int(depth.value.data))
    if qp_ids[: len(q_ids)] != q_ids or len(qp_ids) <= len(q_ids):
        raise ValueError(
            "ModDown requires its represented Q prefix and auxiliary rows"
        )
    p_ids = qp_ids[len(q_ids) :]
    conversion = operation.attributes.get(
        "basis_conversion", StringAttr("exact")
    )
    facts = {
        "q_prime_ids": q_ids,
        "p_prime_ids": p_ids,
        "qp_prime_ids": qp_ids,
        "q_count": len(q_ids),
        "p_count": len(p_ids),
        "basis_conversion": cast(StringAttr, conversion).data,
    }
    centered = (
        tuple(f"moddown_crt_{name}" for name in CRT_OPERANDS)
        if facts["basis_conversion"] == "normalized_crt"
        else ()
    )
    if isinstance(operation, rns.ModDownQpToQOp):
        names = ["qp_parameters", "moddown_inverses"]
        if rounding.data == "nearest":
            names.append("moddown_half")
        if centered:
            names.extend(("q_parameters", "p_inverse", *centered))
        return {
            "rounding": rounding,
            "p_count": IntegerAttr(len(p_ids), 64),
        }, {
            name: table_description(config, int(depth.value.data), name, facts)
            for name in names
        }
    policy = select_operation_policy(operation, log_n=config.logN)
    if policy is None:
        return None
    facts["ntt_backend"] = policy.name
    names = ["q_parameters", "qp_parameters", "moddown_inverses", "p_inverse"]
    if rounding.data == "nearest":
        names.append("moddown_half")
    names.extend(centered)
    names.extend(
        f"{basis}_{direction}_{i}"
        for basis, direction in (("p", "inverse"), ("q", "forward"))
        for i in range(table_count(policy) + 1)
    )
    attributes: dict[str, Attribute] = {
        "ntt_backend": StringAttr(policy.name),
        "rounding": rounding,
    }
    for name in ("q_count", "p_count"):
        attributes[name] = IntegerAttr(cast(int, facts[name]), 64)
    return attributes, {
        name: table_description(config, int(depth.value.data), name, facts)
        for name in names
    }


def prepare_native_moddown(implementation, operation: Operation):
    """Require native ModDown's invocation fields before execution linking."""
    names = (
        ("parameter_names", "ntt_backend")
        if isinstance(
            operation, (rns.ModDownNttQpToQOp, rns.ModDownAddRescaleOp)
        )
        else ("parameter_names",)
    )
    if any(
        name not in operation.attributes
        or operation.attributes[name] == StringAttr("unknown")
        for name in names
    ):
        raise ValueError(
            f"{operation.name} execution operands remain unresolved; check the configuration, layout and implementation requirements reported during compilation"
        )
    return implementation


def _joint_moddown_requirements(
    operation: rns.ModDownAddRescaleOp, config: CkksConfig
):
    """Prepare a joint quotient from represented row sets without requiring depth."""

    def ids(value):
        attribute = value.type.state.data.get("prime_ids")
        return (
            tuple(int(v.value.data) for v in attribute)
            if isinstance(attribute, ArrayAttr)
            else None
        )

    q_ids, qp_ids, remaining = (
        ids(operation.base),
        ids(operation.value),
        ids(operation.result),
    )
    if q_ids is None or qp_ids is None or qp_ids[: len(q_ids)] != q_ids:
        return None
    represented_drop = operation.attributes.get("joint_drop_prime_ids")
    if remaining is None:
        return None
    if isinstance(represented_drop, ArrayAttr):
        drop_ids = tuple(int(i.value.data) for i in represented_drop)
    else:
        drop_ids = tuple(i for i in q_ids if i not in remaining)
    policy = select_operation_policy(operation, log_n=config.logN)
    if policy is None:
        return None
    facts = {
        "q_prime_ids": q_ids,
        "p_prime_ids": qp_ids[len(q_ids) :],
        "joint_drop_prime_ids": drop_ids,
        "joint_target_prime_ids": remaining,
        "ntt_backend": policy.name,
        "moddown_rounding": operation.moddown_rounding.data,
        "rescale_rounding": operation.rescale_rounding.data,
        "basis_conversion": cast(
            StringAttr,
            operation.attributes.get("basis_conversion", StringAttr("exact")),
        ).data,
    }
    attributes = {
        "ntt_backend": StringAttr(policy.name),
        "q_count": IntegerAttr(len(q_ids), 64),
        "p_count": IntegerAttr(len(qp_ids) - len(q_ids), 64),
        "joint_drop_count": IntegerAttr(len(drop_ids), 64),
        "joint_drop_is_prefix": IntegerAttr(
            int(drop_ids == q_ids[: len(drop_ids)]), 64
        ),
        "joint_drop_prime_ids": ArrayAttr(IntegerAttr(i, 64) for i in drop_ids),
        "joint_target_prime_ids": ArrayAttr(
            IntegerAttr(i, 64) for i in remaining
        ),
    }
    names = ["q_parameters", *MODDOWN_RESCALE_TABLE_NAMES]
    if not set(remaining) <= set(q_ids):
        names.extend(MODDOWN_RESTORE_TABLE_NAMES)
    if facts["basis_conversion"] == "normalized_crt":
        names.extend(
            centered_moddown_rescale_names(operation.rescale_rounding.data)
        )
    for basis, direction in (
        ("joint_dropped", "inverse"),
        (
            "joint_remaining",
            "forward" if operation.output_domain.data == "ntt" else "inverse",
        ),
    ):
        names.extend(
            f"{basis}_{direction}_{i}" for i in range(table_count(policy) + 1)
        )
    return attributes, {
        name: table_description(config, None, name, facts) for name in names
    }
