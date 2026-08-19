from __future__ import annotations

import pytest
import torch

from fhelium.core import (
    Ciphertext,
    CompressedPlaintext,
    RotationKey,
    TensorResident,
)
from fhelium.distributed._ciphertext_reduction import _tree_reduce_phases
from fhelium.distributed._limb_collectives import (
    _concatenate_limb_shards,
    _validate_limb_shards,
)
from fhelium.distributed._transfer import (
    TRANSFER_PROTOCOL_VERSION,
    allocate_key,
    allocate_value,
    describe_key,
    describe_value,
)
from fhelium.serialization import VALUE_SCHEMA_VERSION, ValueEnvelope


def _ciphertext() -> Ciphertext:
    return Ciphertext(
        data=torch.arange(2 * 2 * 3 * 3 * 8, dtype=torch.int64).reshape(
            2,
            2,
            3,
            3,
            8,
        ),
        level=2,
        scale=2.0**40,
        context_id="test-context",
        prime_ids=(2, 3, 4),
        polynomial_domain="coefficient",
        modulus_basis="Q",
        residue_representation="standard",
    )


def _sparse_plaintext() -> CompressedPlaintext:
    return CompressedPlaintext(
        data=torch.arange(2 * 3 * 2, dtype=torch.int64).reshape(2, 3, 2),
        ring_dimension=8,
        compression_layout="strided_sparse",
        level=2,
        scale=2.0**40,
        context_id="test-context",
        polynomial_domain="coefficient",
        modulus_basis="Q",
        residue_representation="montgomery",
        prime_ids=(2, 3, 4),
        implicit_data=torch.arange(2 * 3, dtype=torch.int64).reshape(2, 3),
    )


def _rotation_key() -> RotationKey:
    return RotationKey(
        data=torch.arange(2 * 2 * 3 * 8, dtype=torch.int64).reshape(2, 2, 3, 8),
        context_id="test-context",
        prime_ids=(2, 3, 4),
        rotation_step=1,
    )


@pytest.mark.parametrize(
    "source",
    [_ciphertext(), _sparse_plaintext()],
    ids=("batched-ciphertext", "sparse-compressed-plaintext"),
)
def test_typed_descriptor_allocates_exact_receive_state(source) -> None:
    received = allocate_value(
        describe_value(source),
        local_device=torch.device("cpu"),
    )
    assert isinstance(received, TensorResident)
    source_envelope = ValueEnvelope.from_value(source)
    received_envelope = ValueEnvelope.from_value(received)

    assert type(received) is type(source)
    assert received_envelope.context_id == source_envelope.context_id
    assert received_envelope.metadata == source_envelope.metadata
    assert received_envelope.tensors.keys() == source_envelope.tensors.keys()
    for name, source_tensor in source_envelope.tensors.items():
        received_tensor = received_envelope.tensors[name]
        assert received_tensor.shape == source_tensor.shape
        assert received_tensor.dtype == source_tensor.dtype
        assert received_tensor.device.type == "cpu"


def test_transfer_protocol_reuses_exact_value_schema() -> None:
    source = _ciphertext()
    envelope = ValueEnvelope.from_value(source)
    descriptor = describe_value(source)

    assert TRANSFER_PROTOCOL_VERSION != VALUE_SCHEMA_VERSION
    assert descriptor["protocol_version"] == TRANSFER_PROTOCOL_VERSION
    assert descriptor["kind"] == "fhelium_value"
    assert descriptor["value_schema_version"] == envelope.schema_version
    assert descriptor["value_type"] == envelope.value_type
    assert descriptor["context_id"] == envelope.context_id
    assert descriptor["metadata"] == envelope.metadata
    assert descriptor["tensors"].keys() == envelope.tensors.keys()
    assert "level" not in descriptor

    invalid = dict(descriptor)
    invalid["metadata"] = {
        **descriptor["metadata"],
        "prime_ids": [2, 2, 4],
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        allocate_value(invalid, local_device=torch.device("cpu"))


def test_raw_tensor_remains_a_transport_only_descriptor() -> None:
    source = torch.empty((2, 3), dtype=torch.float64)
    descriptor = describe_value(source)
    received = allocate_value(descriptor, local_device=torch.device("cpu"))

    assert descriptor == {
        "protocol_version": TRANSFER_PROTOCOL_VERSION,
        "kind": "tensor",
        "tensor": {
            "shape": (2, 3),
            "dtype": torch.float64,
            "device_type": "cpu",
        },
    }
    assert isinstance(received, torch.Tensor)
    assert received.shape == source.shape
    assert received.dtype == source.dtype


def test_key_transfer_uses_the_same_exact_value_description() -> None:
    source = _rotation_key()
    descriptor = describe_key(source)
    received = allocate_key(descriptor, local_device=torch.device("cpu"))

    source_envelope = ValueEnvelope.from_value(source)
    received_envelope = ValueEnvelope.from_value(received)
    assert descriptor["kind"] == "fhelium_value"
    assert descriptor["value_type"] == "RotationKey"
    assert received_envelope.metadata == source_envelope.metadata
    assert received_envelope.tensors["data"].shape == source.data.shape


def test_batched_limb_shards_reconstruct_along_the_rns_axis() -> None:
    source = _ciphertext()
    shards = (source.slice_limbs(0, 1), source.slice_limbs(1, 3))

    _validate_limb_shards(shards, check_device=True)
    reconstructed = _concatenate_limb_shards(shards)

    assert reconstructed.batch_shape == source.batch_shape == (2, 3)
    assert reconstructed.prime_ids == source.prime_ids
    torch.testing.assert_close(reconstructed.data, source.data)

    mismatched_batch = shards[1].with_data(shards[1].data[:, :1])
    with pytest.raises(ValueError, match="batch_shape"):
        _validate_limb_shards(
            (shards[0], mismatched_batch),
            check_device=True,
        )


def test_tree_schedule_covers_arbitrary_non_power_of_two_world_sizes() -> None:
    for world_size in (1, 2, 3, 5, 6, 7):
        phases = _tree_reduce_phases(world_size)
        edges = [edge for phase in phases for edge in phase]

        assert len(edges) == world_size - 1
        assert sorted(sender for sender, _ in edges) == list(
            range(1, world_size)
        )
        assert all(sender > receiver for sender, receiver in edges)
