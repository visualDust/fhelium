"""Metadata descriptors and receiver allocation for distributed transfers.

The distributed wire protocol carries an exact serialization value description
plus transport-specific tensor allocation metadata.  The protocol version is
independent of the durable value schema version: changing one does not imply a
change to the other.  Raw ``torch.Tensor`` remains a transport-only special
case and is not part of the exact FHElium value schema.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from fhelium.core import TensorResident
from fhelium.serialization.value import (
    ValueEnvelope,
    validate_value_description,
)

TRANSFER_PROTOCOL_VERSION = 4

TransferDescriptor = dict[str, Any]


def describe_value(value: object) -> TransferDescriptor:
    """Describe a workload value without serializing its tensor payload."""

    if isinstance(value, torch.Tensor):
        return {
            "protocol_version": TRANSFER_PROTOCOL_VERSION,
            "kind": "tensor",
            "tensor": _describe_tensor(value),
        }
    if not isinstance(value, TensorResident):
        raise TypeError(
            "Typed value transfer supports torch.Tensor and exact FHElium "
            f"values; got {type(value).__name__}"
        )
    return _describe_envelope(ValueEnvelope.from_value(value))


def allocate_value(
    descriptor: TransferDescriptor,
    *,
    local_device: torch.device,
) -> torch.Tensor | TensorResident:
    """Allocate the receiver-side value described by ``describe_value``."""

    _check_descriptor(descriptor)
    if descriptor["kind"] == "tensor":
        return _allocate_tensor(descriptor["tensor"], local_device)

    tensor_descriptors = cast(
        dict[str, TransferDescriptor], descriptor["tensors"]
    )
    envelope = ValueEnvelope(
        schema_version=descriptor["value_schema_version"],
        value_type=descriptor["value_type"],
        context_id=descriptor["context_id"],
        metadata=descriptor["metadata"],
        tensors={
            name: _allocate_tensor(tensor_descriptor, local_device)
            for name, tensor_descriptor in tensor_descriptors.items()
        },
    )
    return envelope.to_value()


def describe_key(key: object) -> TransferDescriptor:
    """Describe selected key material for transfer."""

    key_type = type(key)
    if _key_types().get(key_type.__name__) is not key_type:
        raise TypeError(
            "broadcast_key supports dense CKKS key objects; "
            f"got {type(key).__name__}"
        )
    return describe_value(key)


def allocate_key(
    descriptor: TransferDescriptor,
    *,
    local_device: torch.device,
) -> TensorResident:
    """Allocate receiver-side key material from a transmitted descriptor."""

    _check_descriptor(descriptor)
    if descriptor["kind"] != "fhelium_value":
        raise ValueError(
            f"Expected a key transfer descriptor, got {descriptor['kind']!r}"
        )
    key_type_name = descriptor["value_type"]
    if key_type_name not in _key_types():
        raise ValueError(f"Unsupported key transfer type: {key_type_name!r}")
    result = allocate_value(descriptor, local_device=local_device)
    if not isinstance(result, TensorResident):
        raise RuntimeError("Key transfer allocated a non-FHElium value")
    return result


def _transfer_tensors(
    value: torch.Tensor | TensorResident,
) -> tuple[torch.Tensor, ...]:
    """Return payload tensors in the same order used by the descriptor."""

    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(ValueEnvelope.from_value(value).tensors.values())


def _describe_envelope(envelope: ValueEnvelope) -> TransferDescriptor:
    return {
        "protocol_version": TRANSFER_PROTOCOL_VERSION,
        "kind": "fhelium_value",
        "value_schema_version": envelope.schema_version,
        "value_type": envelope.value_type,
        "context_id": envelope.context_id,
        "metadata": envelope.metadata,
        "tensors": {
            name: _describe_tensor(tensor)
            for name, tensor in envelope.tensors.items()
        },
    }


def _describe_tensor(tensor: torch.Tensor) -> TransferDescriptor:
    if tensor.layout != torch.strided:
        raise TypeError(
            "Allocation-aware transfer only supports dense strided tensors; "
            f"got layout={tensor.layout}"
        )
    if tensor.device.type not in {"cpu", "cuda"}:
        raise TypeError(
            "Allocation-aware transfer only supports CPU and CUDA tensors; "
            f"got device={tensor.device}"
        )
    return {
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "device_type": tensor.device.type,
    }


def _allocate_tensor(
    descriptor: TransferDescriptor,
    local_device: torch.device,
) -> torch.Tensor:
    device_type = descriptor["device_type"]
    if device_type == "cuda":
        if local_device.type != "cuda":
            raise RuntimeError(
                "Cannot receive a CUDA tensor when the rank-local "
                f"device is {local_device}"
            )
        device = local_device
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(
            f"Unsupported transfer tensor device type: {device_type!r}"
        )
    return torch.empty(
        tuple(descriptor["shape"]),
        dtype=descriptor["dtype"],
        device=device,
    )


def _check_descriptor(descriptor: TransferDescriptor) -> None:
    if not isinstance(descriptor, dict):
        raise TypeError(
            "Transfer descriptor must be a dict, got "
            f"{type(descriptor).__name__}"
        )
    if (
        type(descriptor.get("protocol_version")) is not int
        or descriptor.get("protocol_version") != TRANSFER_PROTOCOL_VERSION
    ):
        raise ValueError(
            "Unsupported transfer protocol version: "
            f"{descriptor.get('protocol_version')!r}"
        )
    kind = descriptor.get("kind")
    if not isinstance(kind, str):
        raise ValueError("Transfer descriptor kind must be a string")
    if kind == "tensor":
        expected = {"protocol_version", "kind", "tensor"}
        _require_descriptor_fields(descriptor, expected)
        _validate_tensor_descriptor(descriptor["tensor"])
        return
    if kind != "fhelium_value":
        raise ValueError(f"Unsupported transfer descriptor kind: {kind!r}")

    expected = {
        "protocol_version",
        "kind",
        "value_schema_version",
        "value_type",
        "context_id",
        "metadata",
        "tensors",
    }
    _require_descriptor_fields(descriptor, expected)
    tensors = descriptor["tensors"]
    if not isinstance(tensors, dict):
        raise ValueError("Transfer value tensors must be an object")
    typed_tensors = cast(dict[object, object], tensors)
    for name, tensor_descriptor in typed_tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Transfer value tensor names must be non-empty strings"
            )
        _validate_tensor_descriptor(tensor_descriptor)
    validate_value_description(
        schema_version=descriptor["value_schema_version"],
        value_type=descriptor["value_type"],
        context_id=descriptor["context_id"],
        metadata=descriptor["metadata"],
        tensor_names=cast(set[str], set(tensors)),
    )


def _require_descriptor_fields(
    descriptor: TransferDescriptor,
    expected: set[str],
) -> None:
    if set(descriptor) != expected:
        raise ValueError(
            "Transfer descriptor fields do not match its kind: "
            f"expected={sorted(expected)}, actual={sorted(descriptor)}"
        )


def _validate_tensor_descriptor(descriptor: object) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError("Transfer tensor descriptor must be an object")
    expected = {"shape", "dtype", "device_type"}
    if set(descriptor) != expected:
        raise ValueError("Transfer tensor descriptor fields are not exact")
    shape = descriptor["shape"]
    if not isinstance(shape, tuple) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError("Transfer tensor shape must be an integer tuple")
    if not isinstance(descriptor["dtype"], torch.dtype):
        raise ValueError("Transfer tensor dtype must be torch.dtype")
    if descriptor["device_type"] not in ("cpu", "cuda"):
        raise ValueError("Transfer tensor device_type must be 'cpu' or 'cuda'")


def _key_types() -> dict[str, type[TensorResident]]:
    from fhelium.core import (
        ConjugationKey,
        KeySwitchKey,
        PublicKey,
        RelinearizationKey,
        RotationKey,
        SecretKey,
    )

    return {
        key_type.__name__: key_type
        for key_type in (
            SecretKey,
            PublicKey,
            KeySwitchKey,
            RotationKey,
            RelinearizationKey,
            ConjugationKey,
        )
    }
