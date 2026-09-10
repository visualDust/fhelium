"""Adapt public Python values at Eager-backed Program interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr

from fhelium.values import Ciphertext, CompressedPlaintext, Plaintext
from fhelium.ir.dialects import ckks, core, semantic


def attribute_value(attribute: object) -> object:
    """Convert one builtin IR attribute to its Python value."""

    if isinstance(attribute, StringAttr):
        return attribute.data
    if isinstance(attribute, IntegerAttr):
        return attribute.value.data
    if isinstance(attribute, FloatAttr):
        return attribute.value.data
    if isinstance(attribute, ArrayAttr):
        return tuple(attribute_value(item) for item in attribute)
    return attribute


def _verify_fields(
    represented_type: object,
    actual_fields: Mapping[str, object],
    *,
    label: str,
) -> None:
    state = getattr(represented_type, "state", None)
    data = getattr(state, "data", None)
    if not isinstance(data, Mapping):
        return
    for name, attribute in data.items():
        if name not in actual_fields:
            continue
        expected = attribute_value(attribute)
        if expected == "unknown":
            continue
        actual = actual_fields[name]
        if actual != expected:
            raise ValueError(
                f"{label} represented {name} is {expected!r}, "
                f"but the runtime value has {actual!r}"
            )


def verify_runtime_value_type(
    value_type: object,
    value: object,
    *,
    label: str,
) -> None:
    """Reject disagreement between an IR type and one public runtime value."""

    if isinstance(value_type, ckks.CiphertextType):
        if not isinstance(value, Ciphertext):
            raise TypeError(f"{label} requires a Ciphertext runtime value")
        _verify_fields(
            value_type,
            {
                "depth": value.depth,
                "scale": value.scale,
                "prime_ids": value.prime_ids,
                "basis": value.modulus_basis,
                "polynomial_domain": value.polynomial_domain,
                "residue_representation": value.residue_representation,
                "batch_shape": tuple(value.batch_shape),
                "ring_dimension": value.ring_dimension,
                "dtype": str(value.data.dtype),
                "device": str(value.data.device),
                "components": value.component_count,
                "component_count": value.component_count,
                "strides": tuple(value.data.stride()),
                "value_kind": "ciphertext",
            },
            label=label,
        )
        return
    if isinstance(value_type, ckks.PlaintextType):
        if not isinstance(value, Plaintext):
            raise TypeError(f"{label} requires a Plaintext runtime value")
        tensor = value.data
        _verify_fields(
            value_type,
            {
                "depth": value.depth,
                "scale": value.scale,
                "representation": value.representation,
                "polynomial_domain": value.polynomial_domain,
                "basis": value.modulus_basis,
                "residue_representation": value.residue_representation,
                "prime_ids": value.prime_ids,
                "batch_shape": tuple(value.batch_shape),
                "ring_dimension": None if tensor is None else tensor.size(-1),
                "device": None if tensor is None else str(tensor.device),
                "dtype": None if tensor is None else str(tensor.dtype),
                "strides": None if tensor is None else tuple(tensor.stride()),
                "value_kind": "plaintext",
            },
            label=label,
        )
        return
    if isinstance(value_type, ckks.CompressedPlaintextType):
        if not isinstance(value, CompressedPlaintext):
            raise TypeError(
                f"{label} requires a CompressedPlaintext runtime value"
            )
        _verify_fields(
            value_type,
            {
                "depth": value.depth,
                "scale": value.scale,
                "prime_ids": value.prime_ids,
                "basis": value.modulus_basis,
                "polynomial_domain": value.polynomial_domain,
                "residue_representation": value.residue_representation,
                "ring_dimension": value.ring_dimension,
                "compression_layout": value.compression_layout,
                "batch_shape": value.batch_shape,
                "device": str(value.device),
                "dtype": str(value.data.dtype),
            },
            label=label,
        )
        return
    if isinstance(value_type, (core.MessageType, semantic.PublicType)):
        if not isinstance(
            value,
            (Sequence, torch.Tensor, complex, float, int),
        ) or isinstance(value, (str, bytes)):
            raise TypeError(f"{label} requires a public message runtime value")
        if isinstance(value, torch.Tensor):
            _verify_fields(
                value_type,
                {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                },
                label=label,
            )


def public_value_from_tensor(
    value_type: object,
    data: torch.Tensor,
    *,
    template: Ciphertext | Plaintext | None = None,
    scale_if_unrepresented: float | None = None,
) -> Ciphertext | Plaintext:
    """Construct one public CKKS value from a represented result and Tensor."""

    if not isinstance(value_type, (ckks.CiphertextType, ckks.PlaintextType)):
        raise TypeError("CKKS tensor result requires ciphertext/plaintext type")
    if isinstance(value_type, ckks.CiphertextType):
        if template is not None and not isinstance(template, Ciphertext):
            raise TypeError("Ciphertext result template must be Ciphertext")
    elif template is not None and not isinstance(template, Plaintext):
        raise TypeError("Plaintext result template must be Plaintext")
    state = value_type.state.data

    def represented_string(name: str, default: str | None) -> str:
        attribute = state.get(name)
        if isinstance(attribute, StringAttr) and attribute.data != "unknown":
            return attribute.data
        if default is None:
            raise ValueError(f"CKKS result lacks represented {name!r}")
        return default

    def represented_integer(name: str, default: int | None) -> int:
        attribute = state.get(name)
        if isinstance(attribute, IntegerAttr):
            return int(attribute.value.data)
        if default is None:
            raise ValueError(f"CKKS result lacks represented {name!r}")
        return default

    def represented_float(name: str, default: float | None) -> float:
        attribute = state.get(name)
        if isinstance(attribute, FloatAttr):
            return float(attribute.value.data)
        if default is None:
            raise ValueError(f"CKKS result lacks represented {name!r}")
        return default

    template_depth = None if template is None else template.depth
    template_scale = None if template is None else template.scale
    template_domain = None if template is None else template.polynomial_domain
    template_basis = None if template is None else template.modulus_basis
    template_residues = (
        None if template is None else template.residue_representation
    )
    depth = represented_integer("depth", template_depth)
    scale = represented_float(
        "scale",
        scale_if_unrepresented
        if scale_if_unrepresented is not None
        else template_scale,
    )
    domain = represented_string("polynomial_domain", template_domain)
    basis = represented_string("basis", template_basis)
    residues = represented_string(
        "residue_representation",
        template_residues,
    )
    prime_attribute = state.get("prime_ids")
    if isinstance(prime_attribute, ArrayAttr):
        prime_ids = tuple(
            int(cast(IntegerAttr, item).value.data) for item in prime_attribute
        )
    elif template is not None:
        row_count = data.size(-2)
        if row_count > len(template.prime_ids):
            raise ValueError(
                "CKKS result has more RNS rows than its public template"
            )
        prime_ids = tuple(template.prime_ids[-row_count:])
    else:
        raise ValueError("CKKS result lacks represented 'prime_ids'")
    if isinstance(value_type, ckks.CiphertextType):
        return Ciphertext(
            data=data,
            depth=depth,
            scale=scale,
            prime_ids=prime_ids,
            polynomial_domain=domain,  # type: ignore[arg-type]
            modulus_basis=basis,  # type: ignore[arg-type]
            residue_representation=residues,  # type: ignore[arg-type]
        )
    return Plaintext(
        message=None,
        depth=depth,
        scale=scale,
        data=data,
        representation="rns",
        polynomial_domain=domain,  # type: ignore[arg-type]
        modulus_basis=basis,  # type: ignore[arg-type]
        residue_representation=residues,  # type: ignore[arg-type]
        prime_ids=prime_ids,
    )


__all__ = [
    "attribute_value",
    "public_value_from_tensor",
    "verify_runtime_value_type",
]
