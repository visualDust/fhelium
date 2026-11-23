"""Public-key encryption roundtrip checks."""

import pytest
import torch

import fhelium as fh
from fhelium.eager import Engine


_DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("output_domain", ["coefficient", "ntt"])
def test_public_key_encryption_roundtrip(
    device: str, output_domain: str
) -> None:
    preset = fh.Preset.slots8192_scale40_depth7_int64
    engine = Engine(preset, rng_seed=17, rng_nonce=11)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    options = {"output_domain": output_domain, "device": device}
    indices = torch.arange(engine.num_slots, dtype=torch.float64, device=device)
    message = torch.complex(
        0.02 * torch.sin(indices * 0.013),
        0.01 * torch.cos(indices * 0.017),
    )
    for slots in (message, torch.stack((message, -message)).unsqueeze(1)):
        ciphertext = engine.encrypt_message(slots, public, **options)
        decoded = engine.decrypt_message(ciphertext, secret)
        assert ciphertext.data.device == slots.device
        assert ciphertext.data.dtype == torch.int64
        assert ciphertext.data.shape[1:-2] == slots.shape[:-1]
        torch.testing.assert_close(
            decoded, slots.to(decoded.device), rtol=0, atol=1e-5
        )
