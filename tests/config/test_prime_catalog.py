from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import pytest
from safetensors import safe_open

import fhelium as fh
from fhelium import CkksConfig, Preset
from fhelium.config import _prime_catalog
from fhelium.config._prime_catalog import get_prime_catalog
from fhelium.eager import Engine


@contextmanager
def _resource_path(name: str) -> Iterator[Path]:
    resource = _prime_catalog.resources.files(
        "fhelium.config.resources"
    ).joinpath(name)
    with _prime_catalog.resources.as_file(resource) as path:
        yield Path(path)


def test_packaged_prime_catalog_has_expected_version_and_coverage() -> None:
    expected_degrees = {1 << log_degree for log_degree in range(12, 18)}
    expected_scaling_keys = set(product(range(20, 55, 5), expected_degrees)) - {
        (20, 1 << 17)
    }
    expected_special_keys = set(product((28, 60), expected_degrees))

    catalog = get_prime_catalog()
    assert set(catalog.scaling_keys) == expected_scaling_keys
    assert set(catalog.special_keys) == expected_special_keys

    for resource_name, expected_format in (
        ("scaling_primes_v1.safetensors", "ckks-scaling-primes"),
        ("special_primes_v1.safetensors", "ckks-special-primes"),
    ):
        with (
            _resource_path(resource_name) as path,
            safe_open(str(path), framework="pt", device="cpu") as handle,
        ):
            assert handle.metadata() == {
                "format": expected_format,
                "version": "1",
            }


def test_packaged_catalog_validation_uses_public_error_hierarchy() -> None:
    with pytest.raises(fh.errors.PrimeCatalogResourceError) as captured:
        _prime_catalog._decode_key(
            "invalid-catalog-key",
            resource_name="invalid.safetensors",
        )

    assert isinstance(captured.value, fh.errors.PrimeCatalogError)
    assert captured.value.resource_name == "invalid.safetensors"


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
