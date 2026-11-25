from __future__ import annotations

import math
from itertools import product

import pytest
import torch

import fhelium as fh
from fhelium import CkksConfig, Preset
from fhelium.eager import Engine
from fhelium.native.wrapper import ckks_ops


def test_prime_catalog_covers_scaling_and_special_widths() -> None:
    degrees = [1 << log_degree for log_degree in range(12, 18)]
    for bits, degree in product(range(20, 55, 5), degrees):
        if (bits, degree) == (20, 1 << 17):
            with pytest.raises(ValueError, match="no scaling primes"):
                ckks_ops.scaling_primes(bits, degree)
            continue
        primes = ckks_ops.scaling_primes(bits, degree)
        assert primes and all((p - 1) % (2 * degree) == 0 for p in primes)
    for bits, degree in product((28, 60), degrees):
        primes = ckks_ops.special_primes(bits, degree)
        assert primes and all((p - 1) % (2 * degree) == 0 for p in primes)
    assert ckks_ops.preset_names() == [preset.value for preset in Preset]


@pytest.mark.parametrize("preset", list(Preset))
def test_preset_name_records_slots_scale_and_max_depth(preset: Preset) -> None:
    config = CkksConfig.parse(preset)
    parts = preset.value.split("-")
    slots, scale_bits, max_depth = (
        int(part.removeprefix(prefix))
        for part, prefix in zip(parts[:3], ("slots", "scale", "depth"), strict=True)
    )
    # A "-boot{n}" preset reserves the depths before its last n for bootstrapping.
    application_depths = (
        int(parts[3].removeprefix("boot")) if len(parts) == 4 else max_depth - 1
    )

    assert config.num_slots == slots
    assert config.max_depth == max_depth
    assert len(config.q_bases) == len(config.scales) == max_depth + 1
    assert config.q_bases[0] == tuple(range(config.num_q_primes))
    assert Engine(config).dtype == torch.int32
    for depth in range(max_depth - application_depths, max_depth + 1):
        assert abs(math.log2(config.scale_at(depth)) - scale_bits) < 1.5
    for depth in range(max_depth):
        ratio = config.transition_ratio(depth, depth + 1)
        assert ratio > 1
    final_bits = sum(math.log2(q) for q in config.active_q_moduli(max_depth))
    assert final_bits >= scale_bits + 9
    config.validate_security_budget()


def test_all_presets_resolve_to_exact_distinct_ntt_primes() -> None:
    for preset in Preset:
        config = CkksConfig.parse(preset)
        assert len(set(config.moduli)) == len(config.moduli)
        assert all((prime - 1) % (2 * config.N) == 0 for prime in config.moduli)
        assert config.depth_remaining(0) == config.max_depth
        assert config.depth_remaining(config.max_depth) == 0


def test_dump_round_trip_records_bases_and_scales() -> None:
    config = CkksConfig.parse(
        Preset.slots8192_scale40_depth7,
        galois_generator=5,
    )
    dumped = config.dumps()

    assert dumped["fhelium_version"] == fh.__version__
    assert dumped["q_moduli"] == list(config.q_moduli)
    assert dumped["q_bases"] == [list(b) for b in config.q_bases]
    assert dumped["p_moduli"] == list(config.p_moduli)
    assert CkksConfig.parse(dumped).dumps() == dumped


def test_config_rejects_invalid_primes_and_removed_fields() -> None:
    baseline = CkksConfig.parse(Preset.slots8192_scale40_depth7)
    q = list(baseline.q_moduli)
    q[0] += 2

    with pytest.raises(ValueError, match="negacyclic NTT"):
        CkksConfig(
            q_moduli=tuple(q),
            q_bases=baseline.q_bases,
            scales=baseline.scales,
            p_moduli=baseline.p_moduli,
            logN=baseline.logN,
        )
    dumped = baseline.dumps()
    dumped["q_depth_groups"] = [[q] for q in baseline.q_moduli]
    with pytest.raises(TypeError, match="unexpected keyword"):
        CkksConfig.parse(dumped)


def test_serialized_config_rejects_another_package_version() -> None:
    dumped = CkksConfig.parse(Preset.slots8192_scale40_depth7).dumps()
    dumped["fhelium_version"] = "incompatible-test-version"

    with pytest.raises(ValueError, match="requires FHElium"):
        CkksConfig.parse(dumped)


def test_config_and_exact_modulus_sequences_are_immutable() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7)
    with pytest.raises(AttributeError, match="CkksConfig is immutable"):
        config.sigma = 3.2
    with pytest.raises(AttributeError):
        config.q_moduli.append(1)  # type: ignore[attr-defined]
