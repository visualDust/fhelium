"""Centered CKKS basis raising and low-basis key encapsulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fhelium.ir.dialects import ckks
from xdsl.ir import Attribute
from fhelium.backend._native_operands import key_switch_options, named_tables
from fhelium.native.wrapper import ckks_ops, rns_ops
from .tables import encapsulation_configs
from ._key_switch_preparation import (
    key_switch_runtime_attributes,
    operation_key_rounds,
)


def modulus_raise_requirements(operation, config):
    """Declare deterministic numerical operands without constructing their data."""
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
    from fhelium.backend.ntt._selection import (
        select_operation_policy,
        table_count,
    )
    from fhelium.backend.rns._preparation import (
        hybrid_decomposition_facts,
        rns_names,
    )

    state = getattr(operation.value.type, "state", None)
    depth = None if state is None else state.data.get("depth")
    if not isinstance(depth, IntegerAttr):
        return None
    depth = int(depth.value.data)
    target = int(operation.target_depth.value.data)
    basis_names = (
        "source_parameters",
        "target_parameters",
        "source_lo",
        "source_hi",
        "source_neg_lo",
        "source_neg_hi",
        "normalizers",
        "propagation",
        "extension",
        "half_digits",
        "source_product",
        "target_moduli",
    )
    if isinstance(operation, ckks.ModulusRaiseOp):
        return {}, {
            name: {
                "kind": "centered_basis_extension_table",
                "depth": depth,
                "target_depth": target,
                "name": name,
            }
            for name in basis_names
        }
    policy = select_operation_policy(operation, log_n=config.logN)
    if policy is None:
        return None
    low_p = tuple(int(x.value.data) for x in operation.low_p_moduli)
    return_p = tuple(int(x.value.data) for x in operation.return_p_moduli)
    guard = int(operation.return_prime.value.data)
    low, _ = encapsulation_configs(config, depth, low_p, guard, return_p)
    rounding = operation.moddown_rounding.data
    low_facts = hybrid_decomposition_facts(low, 0)
    low_facts["ntt_backend"] = policy.name
    names = [
        "low_" + name
        for name in rns_names(low_facts["digits"], moddown_rounding=rounding)
    ]
    for basis in ("q", "qp", "p"):
        for direction in ("forward", "inverse"):
            names.extend(
                f"low_{basis}_{direction}_{i}"
                for i in range(table_count(policy) + 1)
            )
    names.extend("raise_" + name for name in basis_names)
    names.append("return_parameters")
    for direction in ("forward", "inverse"):
        names.extend(
            f"return_{direction}_{i}" for i in range(table_count(policy) + 1)
        )
    names.extend(
        (
            "quotient_order",
            "quotient_parameters",
            "quotient_inverses",
            "target_parameters",
        )
    )
    if rounding == "nearest":
        names.append("quotient_half")
    facts = {
        "low_" + name: value
        for name, value in key_switch_runtime_attributes(low_facts).items()
    }
    facts.update(
        ntt_backend=policy.name,
        q_count=config.num_q_primes,
        target_row_start=config.q_prime_ids(target)[0],
        target_row_count=len(config.q_prime_ids(target)),
        return_drop_count=len(return_p) + 1,
    )
    attributes: dict[str, Attribute] = {
        name: StringAttr(value)
        if isinstance(value, str)
        else IntegerAttr(cast(int, value), 64)
        for name, value in facts.items()
        if name != "low_digits"
    }
    attributes["low_digits"] = ArrayAttr(
        ArrayAttr(IntegerAttr(i, 64) for i in span)
        for span in cast(tuple[tuple[int, ...], ...], low_facts["digits"])
    )
    if rounds := operation_key_rounds(operation):
        attributes["key_rounds"] = IntegerAttr(rounds, 64)
    descriptions = {
        name: {
            "kind": "encapsulated_modulus_raise_table",
            "name": name,
            "depth": depth,
            "target_depth": target,
            "low_p_moduli": list(low_p),
            "return_prime": guard,
            "return_p_moduli": list(return_p),
            "rounding": rounding,
            "ntt_backend": policy.name,
        }
        for name in names
    }
    return attributes, descriptions


@dataclass(frozen=True)
class NativeModulusRaiseImplementation:
    """Execute centered or encapsulated basis raising from bound Tensor operands."""

    name: str = "native-modulus-raise"
    supports_in_place: bool = False
    operation_types: tuple = (
        ckks.ModulusRaiseOp,
        ckks.EncapsulatedModulusRaiseOp,
    )
    tensor_requirements = staticmethod(modulus_raise_requirements)

    def resource_requirements(self, invocation):
        return ()

    def prepare_operation(self, operation):
        if "parameter_names" not in operation.attributes:
            raise ValueError(
                "Modulus-raise numerical operands remain unresolved"
            )
        return self

    def execute(self, invocation, inputs, resources, *, in_place):
        del resources, in_place
        attributes = invocation.attributes
        if invocation.operation_type is not ckks.EncapsulatedModulusRaiseOp:
            return (
                rns_ops.centered_basis_extend(
                    inputs[0],
                    *named_tables(attributes["parameter_names"], inputs[1:]),
                ),
            )
        low = key_switch_options(attributes, prefix="low_")
        low["key_rounds"] = int(attributes.get("key_rounds", 0))
        end = 3
        if low["key_rounds"]:
            # Seeded low and return keys supply their seeds after the keys.
            low["key_seeds"] = list(inputs[3:5])
            end = 5
        low["moddown_rounding"] = str(
            attributes.get("moddown_rounding", "nearest")
        )
        return (
            ckks_ops.encapsulated_modulus_raise(
                inputs[0],
                inputs[1],
                inputs[2],
                *named_tables(attributes["parameter_names"], inputs[end:]),
                **low,
                return_drop_count=int(attributes["return_drop_count"]),
                target_row_start=int(attributes["target_row_start"]),
                q_count=int(attributes["target_row_start"])
                + int(attributes["target_row_count"]),
            ),
        )
