"""Centered CKKS basis raising and low-basis key encapsulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch

from fhelium.ir.dialects import ckks
from xdsl.ir import Attribute
from fhelium.backend.ntt.operations import execute_named_transform
from fhelium.backend.rns.basis_extension import centered_basis_extend
from fhelium.backend.rns.moddown import divide_trailing_rows
from fhelium.native.wrapper import ckks_ops, rns_ops
from .key_switch import key_switch_corrections
from .tables import encapsulation_configs
from ._key_switch_preparation import key_switch_runtime_attributes


def encapsulated_modulus_raise(
    source: torch.Tensor,
    low_key: torch.Tensor,
    return_key: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
) -> torch.Tensor:
    r"""Switch into a low secret relation, center-raise and return to the main key.

    Source is CT2 coefficient/standard Q at the declared low basis. Low key
    digits encode P_low*s under t. The one-digit return hint encodes
    D*t under s, where D=guard*P_return. The output is
    (c0_tilde,0)+Round(c1_tilde*(H0,H1)/D) on the selected target Q rows.
    All operands remain live and unmodified; output storage is independent.
    """
    low_tensors = {
        name[4:]: value
        for name, value in tensors.items()
        if name.startswith("low_")
    }
    low_facts = {
        name[4:]: value
        for name, value in attributes.items()
        if name.startswith("low_")
    }
    rounding = str(attributes.get("moddown_rounding", "nearest"))
    low_facts["moddown_rounding"] = rounding
    corrections = key_switch_corrections(
        source[1], tensors=low_tensors, attributes=low_facts, key=low_key
    )
    sparse = torch.stack(
        (
            rns_ops.add_standard(
                source[0], corrections[0], low_tensors["q_parameters"]
            ),
            corrections[1],
        )
    )
    del corrections
    raised = centered_basis_extend(
        sparse,
        {
            name[6:]: value
            for name, value in tensors.items()
            if name.startswith("raise_")
        },
    )
    del sparse
    component = raised[1]
    execute_named_transform(
        component, tensors, attributes, "return", "forward_to_montgomery_"
    )
    products = torch.zeros(
        (2, *component.shape), dtype=component.dtype, device=component.device
    )
    ckks_ops.keyswitch_accumulate_digit_products_(
        products[0],
        products[1],
        component,
        return_key[0],
        tensors["return_parameters"],
        0,
    )
    del component
    execute_named_transform(
        products, tensors, attributes, "return", "inverse_to_standard_"
    )
    switched = divide_trailing_rows(
        products,
        tensors["quotient_inverses"],
        tensors["quotient_parameters"],
        cast(int, attributes["return_drop_count"]),
        half=tensors.get("quotient_half"),
        row_order=tensors["quotient_order"],
        rounding=rounding,
    )
    start = cast(int, attributes["target_row_start"])
    count = cast(int, attributes["q_count"])
    output = switched[..., start:, :].contiguous()
    rns_ops.add_standard_(
        output[0],
        raised[0, ..., start + 1 : count + 1, :],
        tensors["target_parameters"],
    )
    return output


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
        target_row_start=config.q_row_start(target),
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
        encapsulated = (
            invocation.operation_type is ckks.EncapsulatedModulusRaiseOp
        )
        start = 3 if encapsulated else 1
        tensors = dict(
            zip(
                invocation.attributes["parameter_names"],
                inputs[start:],
                strict=True,
            )
        )
        if encapsulated:
            return (
                encapsulated_modulus_raise(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    tensors,
                    invocation.attributes,
                ),
            )
        return (centered_basis_extend(inputs[0], tensors),)
