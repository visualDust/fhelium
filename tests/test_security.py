from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import fhelium.engine.ckks_engine as ckks_engine_module
from fhelium import CkksConfig, CkksEngine, Preset, config
from fhelium.config import (
    SecurityAssessment,
    assess_config_security,
    assess_security,
)
from fhelium.errors import (
    SecurityBudgetExceededError,
    SecurityParametersUnsupportedError,
)

_TABLE_ROWS = {
    "ternary": {
        128: {
            1024: 26,
            2048: 53,
            4096: 106,
            8192: 214,
            16384: 430,
            32768: 868,
            65536: 1747,
            131072: 3523,
        },
        192: {
            2048: 36,
            4096: 73,
            8192: 147,
            16384: 297,
            32768: 597,
            65536: 1199,
            131072: 2411,
        },
        256: {
            2048: 27,
            4096: 56,
            8192: 114,
            16384: 230,
            32768: 462,
            65536: 929,
            131072: 1866,
        },
    },
    "gaussian": {
        128: {
            1024: 28,
            2048: 55,
            4096: 108,
            8192: 216,
            16384: 432,
            32768: 870,
            65536: 1749,
            131072: 3525,
        },
        192: {
            2048: 38,
            4096: 75,
            8192: 149,
            16384: 299,
            32768: 599,
            65536: 1201,
            131072: 2413,
        },
        256: {
            2048: 30,
            4096: 58,
            8192: 116,
            16384: 232,
            32768: 464,
            65536: 931,
            131072: 1868,
        },
    },
}
_TABLE_CASES = [
    (secret, target, ring_dimension, maximum)
    for secret, by_target in _TABLE_ROWS.items()
    for target, by_ring in by_target.items()
    for ring_dimension, maximum in by_ring.items()
]


@pytest.mark.parametrize(
    ("secret_distribution", "target_bits", "ring_dimension", "maximum_bits"),
    _TABLE_CASES,
)
def test_every_exact_table_row_is_exposed_without_interpolation(
    secret_distribution: str,
    target_bits: int,
    ring_dimension: int,
    maximum_bits: int,
) -> None:
    assessment = assess_security(
        ring_dimension,
        modulus=2**maximum_bits,
        target_bits=target_bits,
        secret_distribution=secret_distribution,
        error_stddev=3.19,
    )

    assert assessment.status == "meets"
    assert assessment.modulus_bits == maximum_bits
    assert assessment.maximum_modulus_bits == maximum_bits
    assert assessment.modulus_margin_bits == 0


@pytest.mark.parametrize("maximum_bits", [26, 430, 3523])
def test_table_budget_limit_is_exact(maximum_bits: int) -> None:
    ring_dimension = {26: 1024, 430: 16384, 3523: 131072}[maximum_bits]

    at_budget = assess_security(ring_dimension, modulus=2**maximum_bits)
    over_budget = assess_security(ring_dimension, modulus=2**maximum_bits + 1)

    assert at_budget.status == "meets"
    assert at_budget.modulus_margin_bits == 0
    assert over_budget.status == "exceeds"
    assert over_budget.modulus_bits == maximum_bits + 1
    assert over_budget.modulus_margin_bits == -1
    assert over_budget.reason is None


@pytest.mark.parametrize(
    ("modulus", "expected_bits"),
    [(2, 1), (3, 2), (4, 2), (5, 3), (2**4096, 4096), (2**4096 + 1, 4097)],
)
def test_requested_modulus_bit_width_uses_exact_integer_semantics(
    modulus: int, expected_bits: int
) -> None:
    assessment = assess_security(1024, modulus=modulus)

    assert assessment.modulus_bits == expected_bits


def test_exact_modulus_factors_are_multiplied_before_bit_width() -> None:
    assessment = assess_security(1024, moduli=(3, 5, 17))

    assert assessment.modulus_bits == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ring_dimension": 512},
        {"ring_dimension": 1024, "target_bits": 192},
        {"ring_dimension": 1024, "target_bits": 129},
        {"ring_dimension": 1024, "error_stddev": 3.2},
        {"ring_dimension": 1024, "secret_distribution": "sparse_ternary"},
    ],
)
def test_table_outside_parameters_are_explicitly_unsupported(
    kwargs: dict[str, object],
) -> None:
    assessment = assess_security(modulus=2, **kwargs)  # type: ignore[arg-type]

    assert assessment.status == "unsupported"
    assert assessment.maximum_modulus_bits is None
    assert assessment.modulus_margin_bits is None
    assert assessment.reason is not None
    assert "external security assessment" in assessment.reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"modulus": 17, "moduli": (17,)},
        {"modulus": 0},
        {"moduli": ()},
        {"moduli": (17, 1)},
        {"modulus": True},
    ],
)
def test_invalid_raw_modulus_inputs_raise_clear_errors(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        assess_security(1024, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("inf"), float("nan")])
def test_config_rejects_nonpositive_or_nonfinite_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        CkksConfig(sigma=sigma)


def test_disabled_enforcement_keeps_unsupported_config_printable() -> None:
    config = CkksConfig(
        sigma=3.2,
        security_bits=129,
        enforce_security_budget=False,
    )

    assert "maximum_modulus_bits=unsupported" in str(config)
    assert config.security_assessment.status == "unsupported"


def test_config_security_state_and_modulus_sequences_are_immutable() -> None:
    config = CkksConfig()
    assessment = config.security_assessment

    with pytest.raises(AttributeError, match="CkksConfig is immutable"):
        config.sigma = 3.2
    with pytest.raises(AttributeError):
        config.moduli.append((1 << 61) - 1)  # type: ignore[attr-defined]

    assert config.security_assessment is assessment
    assert assessment.modulus_bits == config.total_modulus_bits


@pytest.mark.parametrize(
    "removed_field",
    ["uniform_ternary_secret", "quantum", "distribution", "force_secured"],
)
def test_removed_false_security_choices_are_rejected(
    removed_field: str,
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        CkksConfig(**{removed_field: True})  # type: ignore[arg-type]


def test_assessment_is_immutable() -> None:
    assessment = assess_security(16384, modulus=2**400)

    assert isinstance(assessment, SecurityAssessment)
    assert assessment.status == "meets"
    with pytest.raises(FrozenInstanceError):
        assessment.status = "exceeds"  # type: ignore[misc]


def test_config_convenience_assesses_the_exact_complete_qp_product() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)

    assessment = assess_config_security(config)

    assert assessment == config.security_assessment
    assert assessment.modulus_bits == config.total_modulus_bits
    assert assessment.maximum_modulus_bits == config.maximum_modulus_bits
    assert assessment.status == "meets"


def test_engine_enforces_over_budget_before_rns_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedRnsRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("RNS initialization must not run")

    monkeypatch.setattr(ckks_engine_module, "RnsRuntime", UnexpectedRnsRuntime)
    config = CkksConfig(
        logN=14,
        scale_bits=50,
        num_scale_primes=7,
        num_p_primes=1,
    )
    assert config.security_assessment.status == "exceeds"

    with pytest.raises(SecurityBudgetExceededError):
        CkksEngine(config, device="cpu")


def test_engine_rejects_unsupported_sigma_before_rns_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedRnsRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("RNS initialization must not run")

    monkeypatch.setattr(ckks_engine_module, "RnsRuntime", UnexpectedRnsRuntime)
    config = CkksConfig(sigma=3.2)

    with pytest.raises(SecurityParametersUnsupportedError):
        CkksEngine(config, device="cpu")


def test_public_exports() -> None:
    assert config.SecurityAssessment is SecurityAssessment
    assert config.assess_security is assess_security
    assert config.assess_config_security is assess_config_security
    assert {
        "SecurityAssessment",
        "assess_security",
        "assess_config_security",
    } <= set(config.__all__)
