"""Sampling-law and public-key roundtrip checks for the encryption mask."""

import pytest
import torch

import fhelium as fh
from fhelium.eager import Engine
from fhelium.legacy.engine import CkksEngine
from fhelium.rng import Csprng
from fhelium.rng.csprng import _sample_encryption_mask


_DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_encryption_mask_maps_four_equiprobable_outcomes(
    monkeypatch: pytest.MonkeyPatch, device: str, dtype: torch.dtype
) -> None:
    rng = Csprng(
        num_coefs=4, devices=[device], torch_dtype=dtype, seed=17, nonce=11
    )
    uniform = torch.arange(4, device=device, dtype=dtype).repeat(2, 1)

    def four_outcomes(*, amax: int, repeats: int) -> list[torch.Tensor]:
        assert amax == 4
        assert repeats == 2
        return [uniform]

    monkeypatch.setattr(rng, "randint", four_outcomes)
    mask = _sample_encryption_mask(rng, repeats=2)[0]
    torch.testing.assert_close(
        mask,
        torch.tensor([0, 1, -1, 0], device=device, dtype=dtype).repeat(2, 1),
        rtol=0,
        atol=0,
    )
    # Enumerating the equally likely inputs proves the law without a
    # finite-sample frequency threshold: counts for -1, 0, +1 are 1, 2, 1.
    assert torch.bincount(mask[0].to(torch.int64) + 1).tolist() == [1, 2, 1]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_encryption_mask_preserves_seeded_stream_advancement(
    device: str, dtype: torch.dtype
) -> None:
    settings = dict(
        num_coefs=64, devices=[device], torch_dtype=dtype, seed=17, nonce=11
    )
    sampled = Csprng(**settings)
    reference = Csprng(**settings)
    mask = _sample_encryption_mask(sampled, repeats=3)[0]
    uniform = reference.randint(amax=4, repeats=3)[0]
    expected = torch.tensor([0, 1, -1, 0], device=device, dtype=dtype)[
        uniform.to(torch.int64)
    ]
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        sampled.randbytes(repeats=1)[0],
        reference.randbytes(repeats=1)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "path", ["backend-coefficient", "backend-ntt", "legacy"]
)
def test_encryption_mask_public_key_roundtrip(device: str, path: str) -> None:
    preset = fh.Preset.slots8192_scale40_depth7_int64
    if path == "legacy":
        engine = CkksEngine(preset, device=device, rng_seed=17, rng_nonce=11)
        secret = engine.secret_key
        public = engine.public_key
        options = {}
    else:
        engine = Engine(preset, rng_seed=17, rng_nonce=11)
        secret = engine.create_secret_key(device=device)
        public = engine.create_public_key(secret, device=device)
        options = {
            "output_domain": path.removeprefix("backend-"),
            "device": device,
        }
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
