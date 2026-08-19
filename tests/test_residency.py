from __future__ import annotations

import torch

import fhelium as fh
from fhelium.serialization import ValueEnvelope


def _representative_values() -> list[fh.TensorResident]:
    return [
        fh.Ciphertext(
            data=torch.arange(2 * 3 * 8, dtype=torch.int64).reshape(2, 3, 8),
            level=1,
            scale=2.0**40,
            context_id="test-context",
            prime_ids=(1, 2, 3),
        ),
        fh.CompressedPlaintext(
            data=torch.arange(3 * 4, dtype=torch.int64).reshape(3, 4),
            ring_dimension=16,
            compression_layout="strided_sparse",
            level=1,
            scale=2.0**40,
            context_id="test-context",
            polynomial_domain="coefficient",
            modulus_basis="Q",
            residue_representation="montgomery",
            prime_ids=(1, 2, 3),
            implicit_data=torch.tensor([101, 103, 107], dtype=torch.int64),
        ),
    ]


def test_functional_residency_preserves_one_and_two_tensor_exact_values() -> (
    None
):
    for value in _representative_values():
        envelope = ValueEnvelope.from_value(value)
        assert value.device.type == "cpu"
        assert value.is_cpu
        assert not value.is_cuda
        assert value.nbytes == envelope.nbytes
        assert value.to("cpu") is value

        copied = value.to("cpu", copy=True)
        copied_envelope = ValueEnvelope.from_value(copied)
        assert copied is not value
        assert copied_envelope.context_id == envelope.context_id
        assert copied_envelope.metadata == envelope.metadata
        assert copied_envelope.tensors.keys() == envelope.tensors.keys()
        for name, tensor in envelope.tensors.items():
            copied_tensor = copied_envelope.tensors[name]
            assert copied_tensor is not tensor
            torch.testing.assert_close(copied_tensor, tensor)
