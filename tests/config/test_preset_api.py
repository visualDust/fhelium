from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from fhelium import Preset
from fhelium._cli.main import cli
from fhelium.benchmarks.registry import load_builtin_benchmarks


def test_preset_names_and_external_values_are_unique() -> None:
    presets = tuple(Preset)

    assert len(Preset.__members__) == len(presets)
    assert len({preset.value for preset in presets}) == len(presets)


@pytest.mark.parametrize(
    "legacy_value", ["logN14", "logN15", "logN16", "logN17"]
)
def test_removed_logn_values_are_rejected(legacy_value: str) -> None:
    with pytest.raises(ValueError):
        Preset(legacy_value)


def test_unsuffixed_int64_values_are_rejected() -> None:
    for preset in Preset:
        if not preset.value.endswith("-int64"):
            continue
        with pytest.raises(ValueError):
            Preset(preset.value.removesuffix("-int64"))


def test_ntt_recommendation_cli_lists_registry_values() -> None:
    result = CliRunner().invoke(
        cli,
        ["benchmark", "recommend", "ntt", "--help"],
    )

    assert result.exit_code == 0, result.output
    for preset in Preset:
        assert preset.value in result.output
    assert "[default: slots32768-scale40-levels34-int64]" in result.output
    assert "logN16" not in result.output


def test_builtin_benchmark_preset_values_resolve() -> None:
    registry = load_builtin_benchmarks()
    values = []
    for definition in registry:
        for profile in definition.profiles:
            raw = profile.parameters.get("preset")
            if raw is None:
                continue
            assert type(raw) is str
            value = str(raw)
            values.append(value)
            assert Preset(value).value == value
            json.dumps(dict(profile.parameters))

    assert values
