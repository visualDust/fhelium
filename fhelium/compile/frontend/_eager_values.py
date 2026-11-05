"""Represent live CKKS argument metadata as frontend IR types."""

from __future__ import annotations

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    f64,
)
from xdsl.ir import Attribute

from fhelium.ir.dialects import ckks, core
from fhelium.ir.dialects._common import OpenStateType
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    Plaintext,
    KeySwitchKey,
    RotationKey,
)


def value_state(
    value: Ciphertext | Plaintext | CompressedPlaintext,
) -> dict[str, object]:
    """Read mathematical state without reading or copying Tensor contents."""

    state: dict[str, object] = {
        "depth": value.depth,
        "scale": value.scale,
        "prime_ids": value.prime_ids,
        "polynomial_domain": value.polynomial_domain,
        "residue_representation": value.residue_representation,
        "basis": value.modulus_basis,
        "batch_shape": tuple(value.batch_shape),
    }
    if isinstance(value, Ciphertext):
        state["components"] = value.component_count
        state["ring_dimension"] = value.ring_dimension
    elif isinstance(value, CompressedPlaintext):
        state["representation"] = "rns"
        state["unique_count"] = value.unique_count
        state["compression_layout"] = value.compression_layout
        state["ring_dimension"] = value.ring_dimension
    else:
        state["representation"] = value.representation
        if value.data is not None:
            state["ring_dimension"] = value.data.size(-1)
    return state


def state_attributes(value: object) -> dict[str, Attribute]:
    """Represent argument state and physical Tensor layout in an open IR type."""

    fields: dict[str, object] = {}
    if isinstance(value, KeySwitchKey):
        fields = {
            "prime_ids": value.prime_ids,
            "digit_prime_ids": value.digit_prime_ids,
            "polynomial_domain": value.polynomial_domain,
            "residue_representation": value.residue_representation,
            "basis": value.modulus_basis,
        }
        if isinstance(value, RotationKey):
            fields["rotation_step"] = value.rotation_step
    tensor: torch.Tensor | None = None
    if isinstance(value, (Ciphertext, Plaintext, CompressedPlaintext)):
        fields = value_state(value)
        tensor = (
            value.message
            if isinstance(value, Plaintext) and value.representation == "slots"
            else value.data
        )
    elif isinstance(value, (torch.Tensor, KeySwitchKey)):
        tensor = value.data if isinstance(value, KeySwitchKey) else value
    if tensor is not None:
        fields.update(
            shape=tuple(tensor.shape),
            strides=tuple(tensor.stride()),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            requires_grad=tensor.requires_grad,
        )
    return {
        name: attribute
        for name, value in fields.items()
        if (attribute := state_attribute(value)) is not None
    }


def state_attribute(value: object) -> Attribute | None:
    """Convert one supported metadata field into a builtin IR attribute."""

    if isinstance(value, str):
        return StringAttr(value)
    if isinstance(value, bool):
        return IntegerAttr(int(value), 1)
    if isinstance(value, int):
        return IntegerAttr(value, 64)
    if isinstance(value, float):
        return FloatAttr(value, f64)
    if isinstance(value, tuple):
        items = tuple(state_attribute(item) for item in value)
        if all(item is not None for item in items):
            return ArrayAttr(item for item in items if item is not None)
    return None


def value_type(value: object) -> OpenStateType:
    """Create the input type for an ordinary CKKS value or public Tensor."""

    cls = (
        ckks.CiphertextType
        if isinstance(value, Ciphertext)
        else ckks.CompressedPlaintextType
        if isinstance(value, CompressedPlaintext)
        else ckks.PlaintextType
        if isinstance(value, Plaintext)
        else ckks.EvaluationKeyType
        if isinstance(value, KeySwitchKey)
        else core.MessageType
    )
    return cls().with_state(state_attributes(value))


def replacement_state(source, data, implicit, updates, *, controlled):
    """Derive replacement metadata with the public value's own field semantics.

    Fake tensors provide storage metadata without allocating numerical payloads.
    Public replacement constructors validate arbitrary replacements; controlled
    resident views use the same prepared-field constructor as ordinary views.
    """
    import dataclasses
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .._errors import CaptureError

    cls = source.__class__
    fields = {}
    for field in dataclasses.fields(cls):
        if field.name in {'data', 'implicit_data', 'message'}:
            continue
        if field.name in updates:
            fields[field.name] = updates[field.name]
            continue
        name = 'basis' if field.name == 'modulus_basis' else field.name
        if name in source.value.type.state.data:
            fields[field.name] = getattr(source, field.name)
        elif field.default is not dataclasses.MISSING:
            fields[field.name] = field.default
        else:
            raise CaptureError(f'Replacement requires known {field.name}')
    fields.update(updates)

    def fake(value):
        if value is None:
            return None
        state = value._tensor_state()
        strides = state.get('strides')
        options = {
            'dtype': value.dtype,
            'device': value.device,
            'requires_grad': value.requires_grad,
        }
        if isinstance(strides, ArrayAttr):
            return torch.empty_strided(
                value.shape,
                tuple(item.value.data for item in strides),
                **options,
            )
        return torch.empty(value.shape, **options)

    with FakeTensorMode():
        fields['data'] = fake(data)
        if cls is Plaintext:
            fields.setdefault('message', None)
        if cls is CompressedPlaintext:
            fields['implicit_data'] = fake(implicit)
        value = cls._from_fields(**fields) if controlled else cls(**fields)
        result = state_attributes(value)
    # Fake allocation supplies a layout only to evaluate public metadata. An
    # unknown producer layout is not evidence of contiguous execution storage.
    physical = data._tensor_state()
    for name in ('strides', 'requires_grad'):
        if name not in physical:
            result.pop(name, None)
    return result
