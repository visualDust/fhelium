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
from fhelium.errors import ScalePrimeCatalogEntryNotFoundError


@contextmanager
def _resource_path(name: str) -> Iterator[Path]:
    resource = _prime_catalog.resources.files(
        "fhelium.config.resources"
    ).joinpath(name)
    with _prime_catalog.resources.as_file(resource) as path:
        yield Path(path)


def test_packaged_prime_catalog_has_expected_version_and_coverage() -> None:
    expected_degrees = {1 << log_degree for log_degree in range(12, 18)}
    expected_scale_keys = set(product(range(20, 55, 5), expected_degrees)) - {
        (20, 1 << 17)
    }
    expected_message_keys = set(product((28, 60), expected_degrees))

    catalog = get_prime_catalog()
    assert set(catalog.scale_keys) == expected_scale_keys
    assert set(catalog.message_keys) == expected_message_keys

    resources = (
        ("scale_primes_v1.safetensors", "ckks-scale-primes"),
        ("message_primes_v1.safetensors", "ckks-message-primes"),
    )
    for resource_name, expected_format in resources:
        with (
            _resource_path(resource_name) as path,
            safe_open(
                str(path),
                framework="pt",
                device="cpu",
            ) as handle,
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
    assert "invalid parameter key" in captured.value.detail


def test_all_presets_construct_from_the_immutable_catalog() -> None:
    for preset in Preset:
        config = CkksConfig.parse(preset)
        config.validate_security_budget()
        assert len(config.q_moduli) == config.num_q_primes
        assert len(config.p_moduli) == config.num_p_primes
        assert config.message_bits == config.buffer_bit_length - 2
        assert config.default_scale == 2**config.scale_bits
        assert all(
            4 * modulus < 2**config.buffer_bit_length
            for modulus in config.moduli
        )


@pytest.mark.parametrize(
    (
        "preset",
        "slots",
        "scale_bits",
        "public_levels",
        "num_p_primes",
        "total_modulus_bits",
        "maximum_modulus_bits",
        "dtype_name",
    ),
    [
        (
            Preset.slots8192_scale25_levels14_int32,
            8192,
            25,
            14,
            1,
            407,
            430,
            "int32",
        ),
        (
            Preset.slots16384_scale25_levels29_int32,
            16384,
            25,
            29,
            2,
            816,
            868,
            "int32",
        ),
        (
            Preset.slots32768_scale25_levels24_int32,
            32768,
            25,
            24,
            4,
            740,
            1747,
            "int32",
        ),
        (
            Preset.slots65536_scale25_levels14_int32,
            65536,
            25,
            14,
            6,
            543,
            3523,
            "int32",
        ),
        (
            Preset.slots8192_scale30_levels9_int64,
            8192,
            30,
            9,
            1,
            391,
            430,
            "int64",
        ),
        (
            Preset.slots8192_scale40_levels7_int64,
            8192,
            40,
            7,
            1,
            400,
            430,
            "int64",
        ),
        (
            Preset.slots8192_scale50_levels5_int64,
            8192,
            50,
            5,
            1,
            371,
            430,
            "int64",
        ),
        (
            Preset.slots16384_scale30_levels21_int64,
            16384,
            30,
            21,
            2,
            810,
            868,
            "int64",
        ),
        (
            Preset.slots16384_scale40_levels16_int64,
            16384,
            40,
            16,
            2,
            821,
            868,
            "int64",
        ),
        (
            Preset.slots16384_scale50_levels12_int64,
            16384,
            50,
            12,
            2,
            781,
            868,
            "int64",
        ),
        (
            Preset.slots32768_scale30_levels45_int64,
            32768,
            30,
            45,
            4,
            1650,
            1747,
            "int64",
        ),
        (
            Preset.slots32768_scale40_levels34_int64,
            32768,
            40,
            34,
            4,
            1660,
            1747,
            "int64",
        ),
        (
            Preset.slots32768_scale50_levels27_int64,
            32768,
            50,
            27,
            4,
            1650,
            1747,
            "int64",
        ),
        (
            Preset.slots65536_scale30_levels95_int64,
            65536,
            30,
            95,
            6,
            3311,
            3523,
            "int64",
        ),
        (
            Preset.slots65536_scale40_levels72_int64,
            65536,
            40,
            72,
            6,
            3300,
            3523,
            "int64",
        ),
        (
            Preset.slots65536_scale50_levels58_int64,
            65536,
            50,
            58,
            6,
            3320,
            3523,
            "int64",
        ),
    ],
)
def test_preset_name_records_its_baseline_capacity(
    preset: Preset,
    slots: int,
    scale_bits: int,
    public_levels: int,
    num_p_primes: int,
    total_modulus_bits: int,
    maximum_modulus_bits: int,
    dtype_name: str,
) -> None:
    config = CkksConfig.parse(preset)

    assert config.N // 2 == slots
    assert config.scale_bits == scale_bits
    assert config.num_scale_primes == public_levels
    assert config.num_p_primes == num_p_primes
    assert config.total_modulus_bits == total_modulus_bits
    assert config.maximum_modulus_bits == maximum_modulus_bits
    assert str(config.torch_dtype).removeprefix("torch.") == dtype_name
    assert preset.name == (
        f"slots{slots}_scale{scale_bits}_levels{public_levels}_{dtype_name}"
    )
    assert preset.value == (
        f"slots{slots}-scale{scale_bits}-levels{public_levels}-{dtype_name}"
    )


def test_old_logn_preset_members_are_absent() -> None:
    assert fh.Preset is Preset
    assert not hasattr(Preset, "logN14")


def test_preset_external_values_round_trip_as_strings() -> None:
    for preset in Preset:
        assert Preset[preset.name] is preset
        assert Preset(preset.value) is preset
        assert isinstance(preset.value, str)


def test_int32_modulus_selection_enforces_native_lazy_range() -> None:
    valid = CkksConfig.parse(Preset.slots32768_scale25_levels24_int32)
    assert valid.num_scale_primes == 24
    assert all(4 * modulus < 2**30 for modulus in valid.moduli)

    unsafe_prefix = CkksConfig.parse(
        Preset.slots32768_scale25_levels24_int32,
        num_scale_primes=25,
    )
    with pytest.raises(ValueError, match=r"4 \* modulus"):
        _ = unsafe_prefix.moduli

    unsafe_scale_family = CkksConfig.parse(
        Preset.slots8192_scale25_levels14_int32,
        scale_bits=30,
        num_scale_primes=1,
    )
    with pytest.raises(ValueError, match=r"4 \* modulus"):
        _ = unsafe_scale_family.moduli


@pytest.mark.parametrize(
    ("preset", "expected_levels"),
    [
        (Preset.slots32768_scale25_levels24_int32, 24),
        (Preset.slots65536_scale25_levels14_int32, 14),
    ],
)
def test_automatic_int32_depth_stops_at_native_or_catalog_limit(
    preset: Preset,
    expected_levels: int,
) -> None:
    config = CkksConfig.parse(preset, num_scale_primes=None)
    assert config.num_scale_primes == expected_levels
    assert len(config.moduli) == expected_levels + 1 + config.num_p_primes


def test_default_config_matches_the_default_baseline() -> None:
    default = CkksConfig()
    preset = CkksConfig.parse(Preset.slots16384_scale40_levels16_int64)

    assert default.dumps() == preset.dumps()


def test_preset_override_does_not_mutate_the_baseline() -> None:
    baseline = Preset.slots8192_scale40_levels7_int64

    derived = CkksConfig.parse(baseline, scale_bits=30)
    reparsed = CkksConfig.parse(baseline)

    assert derived.scale_bits == 30
    assert derived.num_scale_primes == 7
    assert reparsed.scale_bits == 40
    assert reparsed.num_scale_primes == 7


def test_preset_depth_changes_only_through_an_explicit_override() -> None:
    config = CkksConfig.parse(
        Preset.slots8192_scale40_levels7_int64,
        scale_bits=30,
        num_scale_primes=9,
    )

    assert config.scale_bits == 30
    assert config.num_scale_primes == 9


def test_unsupported_scale_configuration_fails_explicitly() -> None:
    config = CkksConfig(scale_bits=21, logN=14)
    with pytest.raises(ScalePrimeCatalogEntryNotFoundError):
        _ = config.moduli
