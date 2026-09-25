"""Describe hybrid RNS tables and native ModDown operand requirements."""

from __future__ import annotations
from typing import cast
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation
from fhelium.config import CkksConfig
from fhelium.ir.dialects import rns
from fhelium.backend.ntt._selection import select_operation_policy, table_count
from .layout import RnsLayout


def hybrid_decomposition_facts(
    config: CkksConfig, depth: int
) -> dict[str, object]:
    """Describe the default mathematical hybrid decomposition at one depth."""
    layout = RnsLayout.from_config(config)
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
        "key_row_start": layout.start_row(depth),
    }


def rns_names(digits) -> tuple[str, ...]:
    """List RNS-only preparation tables, independently of an NTT schedule."""
    names = ["q_parameters", "qp_parameters", "moddown_inverses", "p_inverse"]
    for i, (start, stop, _) in enumerate(digits):
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
    config: CkksConfig, depth: int, name: str, facts: dict[str, object]
) -> dict[str, object]:
    """Describe one table by its own mathematical or NTT-layout requirements."""
    q_ids = tuple(
        range(int(cast(int, facts["key_row_start"])), config.num_q_primes)
    )
    ids = {
        "q": q_ids,
        "qp": (*q_ids, *range(config.num_q_primes, config.total_num_primes)),
        "p": tuple(range(config.num_q_primes, config.total_num_primes)),
    }
    if name in {"q_parameters", "qp_parameters", "p_parameters"}:
        return {
            "kind": "rns_parameters",
            "prime_ids": list(ids[name.removesuffix("_parameters")]),
        }
    parts = name.split("_")
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
    return {
        "kind": "key_switch_table",
        "depth": depth,
        "name": name,
        "digits": [
            list(span)
            for span in cast(tuple[tuple[int, ...], ...], facts["digits"])
        ],
    }


def native_moddown_requirements(operation: Operation, config: CkksConfig):
    """Declare the NTT and RNS operands of native auxiliary-basis removal."""
    if isinstance(operation, rns.ModDownQpToQOp):
        return {}, {}
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
    names = ["q_parameters", "qp_parameters", "moddown_inverses", "p_inverse"]
    names.extend(
        f"{basis}_{direction}_{i}"
        for basis, direction in (("p", "inverse"), ("q", "forward"))
        for i in range(table_count(policy) + 1)
    )
    attributes: dict[str, Attribute] = {"ntt_backend": StringAttr(policy.name)}
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


def prepare_native_moddown(implementation, operation: Operation):
    """Require native ModDown's invocation fields before execution linking."""
    names = (
        ("parameter_names", "ntt_backend")
        if isinstance(operation, rns.ModDownNttQpToQOp)
        else ("parameter_names",)
    )
    if any(
        name not in operation.attributes
        or operation.attributes[name] == StringAttr("unknown")
        for name in names
    ):
        raise ValueError(
            f"{operation.name} execution operands remain unresolved; run PrepareOperationOperandsPass with the required execution facts"
        )
    return implementation
