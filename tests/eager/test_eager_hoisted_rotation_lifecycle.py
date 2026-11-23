"""Behavior checks for eager hoisted rotation resource lifetimes."""

from __future__ import annotations

import pytest
import torch

from fhelium import CkksConfig, Preset
from fhelium.eager import Engine


def _assert_hoisted_rotation_lifecycle(device: str) -> None:
    engine = Engine(
        Preset.slots8192_scale40_depth7_int64, rng_seed=941, rng_nonce=17
    )
    with torch.device(device):
        message = torch.linspace(
            -0.01, 0.01, engine.num_slots, dtype=torch.float64
        )
        secret_key = engine.create_secret_key()
        public_key = engine.create_public_key(secret_key)
        source = engine.encrypt(engine.encode(message), public_key)
        source = engine.mod_switch_to_depth(source, 2)
        keys = [
            engine.create_rotation_key(step, secret_key) for step in (1, -3, 1)
        ]

    def fresh_hoisted(rotation_keys):
        """Return hoisted rotations from an Engine without cached executors."""

        return Engine(engine.config).rotate_many_with_keys(
            source, rotation_keys, use_hoisting=True, moddown_rounding="floor"
        )

    for _ in range(2):
        actual = engine.rotate_many_with_keys(
            source, keys, use_hoisting=True, moddown_rounding="floor"
        )
        expected = fresh_hoisted(keys)
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
            source, keys, output_domain="ntt", moddown_rounding="floor"
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
            source,
            keys,
            use_hoisting=False,
            output_domain="ntt",
            moddown_rounding="floor",
        )
        independent_coefficient = [
            engine.rotate_with_key(source, key, moddown_rounding="floor")
            for key in keys
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
        source, keys, use_hoisting=True, moddown_rounding="floor"
    )
    expected_after_mutation = fresh_hoisted(keys)
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
def test_hybrid_key_operations_across_depths(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source_config = CkksConfig.parse(Preset.slots32768_scale40_depth34_int64)
    config = CkksConfig(
        default_scale=source_config.default_scale,
        q_depth_groups=source_config.q_depth_groups[:8],
        p_moduli=source_config.p_moduli,
        logN=12,
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
        for depth in (0, 3, 6):
            source = (
                engine.mod_switch_to_depth(encrypted, depth)
                if depth
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
