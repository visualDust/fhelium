from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import fhelium as fh
from fhelium.core import (
    Ciphertext,
    CompressedPlaintext,
    Plaintext,
    RotationKey,
    SecretKey,
)
from fhelium.serialization import ValueEnvelope


def _ciphertext(device: str = "cpu") -> Ciphertext:
    return Ciphertext(
        data=torch.arange(
            3 * 2 * 3 * 8,
            dtype=torch.int64,
            device=device,
        ).reshape(3, 2, 3, 8),
        level=1,
        scale=2.0**40,
        context_id="test-context",
        prime_ids=(1, 2, 3),
    )


def _prepared_plaintext(device: str = "cpu") -> Plaintext:
    return Plaintext(
        message=None,
        level=1,
        scale=2.0**40,
        data=torch.arange(
            3 * 8,
            dtype=torch.int64,
            device=device,
        ).reshape(3, 8),
        context_id="test-context",
        representation="rns",
        polynomial_domain="ntt",
        modulus_basis="Q",
        residue_representation="montgomery",
        prime_ids=(1, 2, 3),
    )


def _sparse_plaintext(device: str = "cpu") -> CompressedPlaintext:
    return CompressedPlaintext(
        data=torch.arange(
            3 * 4,
            dtype=torch.int64,
            device=device,
        ).reshape(3, 4),
        ring_dimension=16,
        compression_layout="strided_sparse",
        level=1,
        scale=2.0**40,
        context_id="test-context",
        polynomial_domain="coefficient",
        modulus_basis="Q",
        residue_representation="montgomery",
        prime_ids=(1, 2, 3),
        implicit_data=torch.tensor(
            [101, 103, 107],
            dtype=torch.int64,
            device=device,
        ),
    )


def _rotation_key(device: str = "cpu") -> RotationKey:
    return RotationKey(
        data=torch.arange(
            2 * 2 * 3 * 8,
            dtype=torch.int64,
            device=device,
        ).reshape(2, 2, 3, 8),
        context_id="test-context",
        prime_ids=(0, 1, 2),
        rotation_step=1,
    )


def _assert_same_exact_value(actual, expected) -> None:
    assert type(actual) is type(expected)
    actual_envelope = ValueEnvelope.from_value(actual)
    expected_envelope = ValueEnvelope.from_value(expected)
    assert actual_envelope.schema_version == expected_envelope.schema_version
    assert actual_envelope.value_type == expected_envelope.value_type
    assert actual_envelope.context_id == expected_envelope.context_id
    assert actual_envelope.metadata == expected_envelope.metadata
    assert actual_envelope.tensors.keys() == expected_envelope.tensors.keys()
    for name, expected_tensor in expected_envelope.tensors.items():
        torch.testing.assert_close(
            actual_envelope.tensors[name], expected_tensor
        )


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata is not None
    return dict(metadata)


def test_value_envelope_is_a_path_independent_representation() -> None:
    key = _rotation_key()
    envelope = ValueEnvelope.from_value(key)

    assert envelope.tensors["data"] is key.data
    assert envelope.nbytes == key.nbytes
    assert not hasattr(envelope, "path")

    backend_owned_tensors = {
        name: tensor.clone() for name, tensor in envelope.tensors.items()
    }
    restored = replace(envelope, tensors=backend_owned_tensors).to_value()

    _assert_same_exact_value(restored, key)
    assert restored.data is backend_owned_tensors["data"]


@pytest.mark.parametrize(
    "value",
    [
        Plaintext(
            message=torch.tensor(
                [1.0 + 2.0j, 3.0 - 4.0j],
                dtype=torch.complex128,
            ),
            level=2,
            scale=2.0**35,
        ),
        Plaintext(
            message=None,
            level=1,
            scale=2.0**40,
            data=torch.arange(8, dtype=torch.float64),
            context_id="test-context",
            representation="approximate_coefficients",
            polynomial_domain="coefficient",
        ),
        _prepared_plaintext(),
        _sparse_plaintext(),
        _ciphertext(),
        _rotation_key(),
    ],
    ids=(
        "complex-slots",
        "approximate-coefficients",
        "rns-plaintext",
        "sparse-compressed-plaintext",
        "batched-three-component-ciphertext",
        "rotation-key",
    ),
)
def test_value_file_roundtrip_preserves_distinct_storage_families(
    tmp_path: Path,
    value,
) -> None:
    path = tmp_path / "caller-selected-name.bin"
    written = fh.save_value(value, path)

    assert path.is_file()
    assert written.value_type == type(value).__name__
    assert written.context_id == value.context_id
    assert written.nbytes == value.nbytes
    assert fh.inspect_value(path) == written

    restored = fh.load_value(
        path,
        expected_type=type(value),
        expected_context_id=value.context_id,
    )
    _assert_same_exact_value(restored, value)


def test_direct_file_lifecycle_keeps_namespace_and_overwrite_policy_explicit(
    tmp_path: Path,
) -> None:
    value = _ciphertext()
    path = tmp_path / "tenant-a" / "ciphertext.safetensors"

    with pytest.raises(FileNotFoundError, match="parent directory"):
        fh.save_value(value, path)

    path.parent.mkdir()
    fh.save_value(value, path)
    with pytest.raises(FileExistsError):
        fh.save_value(value, path)
    with pytest.raises(ValueError, match="context mismatch"):
        fh.load_value(path, expected_context_id="another-context")
    with pytest.raises(TypeError, match="expected Plaintext"):
        fh.load_value(path, expected_type=Plaintext)

    replacement = value.with_data(torch.ones_like(value.data))
    fh.save_value(replacement, path, overwrite=True)
    _assert_same_exact_value(fh.load_value(path), replacement)


def test_secret_key_file_requires_explicit_unencrypted_opt_in(
    tmp_path: Path,
) -> None:
    secret_key = SecretKey(
        data=torch.arange(3 * 8, dtype=torch.int64).reshape(3, 8),
        context_id="test-context",
        prime_ids=(0, 1, 2),
    )
    path = tmp_path / "secret.safetensors"

    with pytest.raises(PermissionError, match="disabled by default"):
        fh.save_value(secret_key, path)

    metadata = fh.save_value(secret_key, path, allow_secret=True)
    assert metadata.value_type == "SecretKey"
    _assert_same_exact_value(
        fh.load_value(path, expected_type=SecretKey),
        secret_key,
    )


def test_inspect_rejects_unsupported_value_schema(tmp_path: Path) -> None:
    path = tmp_path / "ciphertext.safetensors"
    fh.save_value(_ciphertext(), path)
    tensors = load_file(path)
    metadata = _metadata(path)
    manifest = json.loads(metadata["fhelium.manifest"])
    manifest["value_schema_version"] = True
    metadata["fhelium.manifest"] = json.dumps(manifest)
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="schema version"):
        fh.inspect_value(path)


def test_inspect_compares_manifest_with_payload_header(tmp_path: Path) -> None:
    path = tmp_path / "ciphertext.safetensors"
    fh.save_value(_ciphertext(), path)
    tensors = load_file(path)
    metadata = _metadata(path)
    manifest = json.loads(metadata["fhelium.manifest"])
    manifest["tensor_metadata"]["data"]["shape"][-1] += 1
    metadata["fhelium.manifest"] = json.dumps(manifest)
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="payload header"):
        fh.inspect_value(path)


@pytest.mark.gpu
def test_direct_value_file_can_materialize_on_an_explicit_cuda_device(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    path = tmp_path / "ciphertext.safetensors"
    value = _ciphertext()
    fh.save_value(value, path)
    restored = fh.load_value(path, device="cuda:0", expected_type=Ciphertext)

    assert restored.device == torch.device("cuda:0")
    torch.testing.assert_close(restored.data.cpu(), value.data)
