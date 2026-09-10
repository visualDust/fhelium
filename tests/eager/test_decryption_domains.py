"""Behavior checks for coefficient- and NTT-domain decryption phases."""

from typing import cast

import pytest
import torch

import fhelium as fh
from fhelium.backend.ckks.crypto._decryption import (
    _decrypt_tensor_to_coefficient_standard_rns,
)
from fhelium.eager import Engine
from fhelium.values import Ciphertext, ModulusBasis


@pytest.mark.parametrize("modulus_basis", ["Q", "QP"])
@pytest.mark.parametrize("component_count", [2, 3])
def test_decryption_phase_matches_across_input_domains(
    modulus_basis: ModulusBasis,
    component_count: int,
) -> None:
    engine = Engine(
        fh.Preset.slots8192_scale40_depth7_int64,
        rng_seed=20260905,
        rng_nonce=component_count,
    )
    secret = engine.create_secret_key(modulus_basis="QP", device="cpu")
    public = engine.create_public_key(
        secret,
        modulus_basis=modulus_basis,
        device="cpu",
    )
    slots = engine.num_slots
    indices = torch.arange(slots, dtype=torch.float64)
    message = torch.stack(
        (
            0.02 * torch.sin(indices * 0.013),
            0.01 * torch.cos(indices * 0.017),
        )
    )
    coefficient = engine.encrypt_message(message, public)
    ntt = cast(Ciphertext, engine.coefficient_domain_to_ntt_domain(coefficient))
    if component_count == 3:
        ntt = engine.multiply(ntt, ntt)
        coefficient = cast(
            Ciphertext,
            engine.ntt_domain_to_coefficient_domain(ntt),
        )

    coefficient_before = coefficient.data.clone()
    ntt_before = ntt.data.clone()
    secret_before = secret.data.clone()
    rns_context = engine._rns_context_for(torch.device("cpu"))
    ntt_context = engine._ntt_context_for(torch.device("cpu"))
    includes_p = modulus_basis == "QP"

    coefficient_phase = _decrypt_tensor_to_coefficient_standard_rns(
        coefficient.data,
        secret.data,
        depth=coefficient.depth,
        includes_p=includes_p,
        secret_key_basis=secret.modulus_basis,
        input_domain="coefficient",
        rns_context=rns_context,
        ntt_context=ntt_context,
    )
    ntt_phase = _decrypt_tensor_to_coefficient_standard_rns(
        ntt.data,
        secret.data,
        depth=ntt.depth,
        includes_p=includes_p,
        secret_key_basis=secret.modulus_basis,
        input_domain="ntt",
        rns_context=rns_context,
        ntt_context=ntt_context,
    )

    assert torch.equal(ntt_phase, coefficient_phase)
    assert torch.equal(coefficient.data, coefficient_before)
    assert torch.equal(ntt.data, ntt_before)
    assert torch.equal(secret.data, secret_before)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_delayed_rescale_decrypts_actual_scale_independently_of_default(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    config = fh.CkksConfig.parse(
        fh.Preset.slots16384_scale25_depth29_int32,
        default_scale=2.0**16,
    )
    engine = Engine(config, rng_seed=91, rng_nonce=3)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    relin = engine.create_relinearization_key(secret, device=device)
    message = torch.linspace(0.5, 0.8, engine.num_slots, dtype=torch.float64, device=device)
    source = engine.encrypt_message(message, public, scale=2.0**50, output_domain="ntt", device=device)
    squared = engine.relinearize(engine.multiply(source, source), relin, output_domain="ntt")
    cubed = engine.multiply(squared, source)
    assert cubed.depth == 0
    assert cubed.scale == 2.0**150
    decoded = engine.decrypt_message(cubed, secret, is_real=True)
    torch.testing.assert_close(decoded, message**3, rtol=0.0, atol=1e-5)
    rescaled = engine.rescale_to_next_depth(cubed)
    assert rescaled.depth == 1
    assert rescaled.scale == cubed.scale / config.rescale_divisor(0)
    torch.testing.assert_close(
        engine.decrypt_message(rescaled, secret, is_real=True),
        message**3,
        rtol=0.0,
        atol=1e-5,
    )
