"""Behavior checks for eager hoisted rotation resource lifetimes."""

from __future__ import annotations

from fhelium.legacy.engine import CkksEngine

import pytest
import torch

from fhelium import CkksConfig, Preset
from fhelium.eager import Engine


def _assert_hoisted_rotation_lifecycle(device: str) -> None:
    config = Preset.slots8192_scale40_levels7_int64
    reference = CkksEngine(
        config,
        device=device,
        rng_seed=941,
        rng_nonce=17,
    )
    engine = Engine(
        config,
        rng_seed=941,
        rng_nonce=17,
    )
    message = torch.linspace(
        -0.01,
        0.01,
        reference.num_slots,
        dtype=torch.float64,
    )
    secret_key = reference.create_secret_key()
    public_key = reference.create_public_key(secret_key)
    source = reference.encrypt(reference.encode(message), public_key)
    source = reference.mod_switch_to_level(source, 2)
    keys = [
        reference.create_rotation_key(step, secret_key) for step in (1, -3, 1)
    ]

    for _ in range(2):
        actual = engine.rotate_many_with_keys(
            source,
            keys,
            use_hoisting=True,
        )
        expected = reference.rotate_many_with_keys(
            source,
            keys,
            use_hoisting=True,
        )
        assert len(actual) == len(expected)
        assert all(
            torch.equal(actual_value.data, expected_value.data)
            for actual_value, expected_value in zip(
                actual,
                expected,
                strict=True,
            )
        )
        ntt_results = engine.rotate_many_with_keys(
            source, keys, output_domain="ntt"
        )
        for ntt_value, coefficient_value in zip(
            ntt_results, expected, strict=True
        ):
            assert ntt_value.polynomial_domain == "ntt"
            assert ntt_value.residue_representation == "montgomery"
            assert torch.equal(
                engine.ntt_domain_to_coefficient_domain(ntt_value).data,
                coefficient_value.data,
            )
        independent_ntt = engine.rotate_many_with_keys(
            source, keys, use_hoisting=False, output_domain="ntt"
        )
        independent_coefficient = [
            engine.rotate_with_key(source, key) for key in keys
        ]
        for ntt_value, coefficient_value in zip(
            independent_ntt, independent_coefficient, strict=True
        ):
            assert torch.equal(
                engine.ntt_domain_to_coefficient_domain(ntt_value).data,
                coefficient_value.data,
            )

    keys[0].data[0, 0, 0, 0].add_(1)
    actual_after_mutation = engine.rotate_many_with_keys(
        source,
        keys,
        use_hoisting=True,
    )
    expected_after_mutation = reference.rotate_many_with_keys(
        source,
        keys,
        use_hoisting=True,
    )
    assert all(
        torch.equal(actual_value.data, expected_value.data)
        for actual_value, expected_value in zip(
            actual_after_mutation,
            expected_after_mutation,
            strict=True,
        )
    )


def test_cpu_hoisted_rotation_reuses_executor_across_calls() -> None:
    _assert_hoisted_rotation_lifecycle("cpu")


@pytest.mark.gpu
def test_cuda_hoisted_rotation_reuses_executor_across_calls() -> None:
    _assert_hoisted_rotation_lifecycle("cuda:0")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_hybrid_key_operations_across_levels(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CkksConfig(
        logN=12,
        scale_bits=40,
        num_scale_primes=7,
        num_p_primes=4,
        buffer_bit_length=62,
        enforce_security_budget=False,
    )
    engine = Engine(config, allow_automatic_key_generation=False)
    with torch.device(device):
        source_secret = engine.create_secret_key()
        destination_secret = engine.create_secret_key()
        public = engine.create_public_key(source_secret)
        switch_key = engine.create_key_switch_key(
            source_secret, destination_secret
        )
        rotation_keys = [
            engine.create_rotation_key(step, source_secret) for step in (1, -3)
        ]
        message = torch.linspace(
            -0.01, 0.01, engine.num_slots, dtype=torch.float64
        ) * (1 + 0.3j)
        encrypted = engine.encrypt_message(message, public)
        for level in (0, 3, 6):
            source = (
                engine.mod_switch_to_level(encrypted, level)
                if level
                else encrypted
            )
            switched = engine.switch_key(source, switch_key)
            assert (
                engine.decrypt_message(switched, destination_secret) - message
            ).abs().max() < 1e-5
            rotations = engine.rotate_many_with_keys(
                source, rotation_keys, output_domain="ntt"
            )
            for step, rotated in zip((1, -3), rotations, strict=True):
                coefficient = engine.ntt_domain_to_coefficient_domain(rotated)
                decoded = engine.decrypt_message(coefficient, source_secret)
                assert (decoded - torch.roll(message, step)).abs().max() < 1e-5
