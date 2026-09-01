"""Behavior checks for eager hoisted rotation resource lifetimes."""

from __future__ import annotations

from fhelium.legacy.engine import CkksEngine

import pytest
import torch

from fhelium import Preset
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
