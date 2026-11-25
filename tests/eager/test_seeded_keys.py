"""Compact and expanded key-switching keys produce the same results."""

from __future__ import annotations

import pytest
import torch

import fhelium as fh
from fhelium import Preset
from fhelium.eager import Engine


@pytest.mark.parametrize(
    "device",
    (
        pytest.param("cpu", id="cpu"),
        pytest.param("cuda:0", marks=pytest.mark.gpu, id="cuda"),
    ),
)
def test_compact_keys_match_expanded_keys(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    with torch.device(device):
        engine = Engine(Preset.slots8192_scale40_depth7, expand_keys=False)
        relin = engine.relinearization_key
        rotation = engine.create_rotation_key(3, engine.secret_key)
        assert not relin.is_expanded and not rotation.is_expanded
        expanded_relin = relin.expand()
        expanded_rotation = rotation.expand()
        assert expanded_relin.is_expanded and not relin.is_expanded
        index = torch.arange(engine.num_slots, dtype=torch.float64)
        message = 0.01 * torch.sin(index * 0.013)
        for depth in (0, 4):
            value = engine.encrypt_message(message, depth=depth)
            ntt = engine.coefficient_domain_to_ntt_domain(value)
            triplet = engine.multiply(ntt, ntt)
            assert torch.equal(
                engine.relinearize(triplet, relin).data,
                engine.relinearize(triplet, expanded_relin).data,
            )
            assert torch.equal(
                engine.rotate_with_key(value, rotation).data,
                engine.rotate_with_key(value, expanded_rotation).data,
            )


def test_generated_keys_are_expanded_and_files_store_the_seed(tmp_path) -> None:
    engine = Engine(Preset.slots8192_scale40_depth7)
    key = engine.create_rotation_key(1, engine.secret_key)
    assert key.is_expanded and key.seed is not None
    expanded = key.data.clone()
    assert torch.equal(key.compact().expand().data, expanded)
    assert key.is_expanded
    path = tmp_path / "rotation.safetensors"
    fh.save_value(key, path)
    loaded = fh.load_value(path)
    assert not loaded.is_expanded and loaded.seed_spec == key.seed_spec
    engine.install_evaluation_key("rotation-key:1", loaded)
    assert torch.equal(engine.rotation_key(1).data, expanded)
    if torch.cuda.is_available():
        assert torch.equal(
            fh.load_value(path, device="cuda:0").expand().data.cpu(), expanded
        )
