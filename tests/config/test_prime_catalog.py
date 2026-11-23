from __future__ import annotations

from itertools import product

import pytest

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


@pytest.mark.parametrize(
    ("preset", "slots", "max_depth", "total_bits", "dtype"),
    [
        (Preset.slots8192_scale25_depth14_int32, 8192, 14, 404, "int32"),
        (Preset.slots16384_scale40_depth16_int64, 16384, 16, 800, "int64"),
        (Preset.slots32768_scale50_depth27_int64, 32768, 27, 1640, "int64"),
        (Preset.slots32768_scale50_depth29_int64, 32768, 29, 1740, "int64"),
        (Preset.slots65536_scale30_depth95_int64, 65536, 95, 3278, "int64"),
    ],
)
def test_preset_name_records_max_depth_and_exact_parameter_capacity(
    preset: Preset,
    slots: int,
    max_depth: int,
    total_bits: int,
    dtype: str,
) -> None:
    config = CkksConfig.parse(preset)
    engine = Engine(config)

    assert config.num_slots == slots
    assert config.max_depth == max_depth
    assert len(config.q_depth_groups) == max_depth + 1
    assert config.total_modulus_bits == total_bits
    assert str(engine.dtype).removeprefix("torch.") == dtype
    assert f"_depth{max_depth}_" in preset.name
    assert f"-depth{max_depth}-" in preset.value
    config.validate_security_budget()


def test_all_presets_resolve_to_exact_distinct_ntt_primes() -> None:
    for preset in Preset:
        config = CkksConfig.parse(preset)
        assert len(set(config.moduli)) == len(config.moduli)
        assert all((prime - 1) % (2 * config.N) == 0 for prime in config.moduli)
        assert config.depth_remaining(0) == config.max_depth
        assert config.depth_remaining(config.max_depth) == 0


def test_dump_round_trip_records_exact_depth_groups() -> None:
    config = CkksConfig.parse(
        Preset.slots8192_scale40_depth7_int64,
        galois_generator=5,
    )
    dumped = config.dumps()

    assert dumped["fhelium_version"] == fh.__version__
    assert dumped["q_depth_groups"] == [
        list(group) for group in config.q_depth_groups
    ]
    assert dumped["p_moduli"] == list(config.p_moduli)
    assert CkksConfig.parse(dumped).dumps() == dumped


def test_exact_config_rejects_invalid_prime_groups() -> None:
    baseline = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    groups = list(baseline.q_depth_groups)
    groups[0] = (groups[0][0] + 2,)

    with pytest.raises(ValueError, match="negacyclic NTT"):
        CkksConfig(
            default_scale=baseline.default_scale,
            q_depth_groups=tuple(groups),
            p_moduli=baseline.p_moduli,
            logN=baseline.logN,
        )


def test_serialized_config_rejects_another_package_version() -> None:
    dumped = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64).dumps()
    dumped["fhelium_version"] = "incompatible-test-version"

    with pytest.raises(ValueError, match="requires FHElium"):
        CkksConfig.parse(dumped)


def test_config_and_exact_modulus_sequences_are_immutable() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    with pytest.raises(AttributeError, match="CkksConfig is immutable"):
        config.sigma = 3.2
    with pytest.raises(AttributeError):
        config.q_depth_groups.append((1,))  # type: ignore[attr-defined]
