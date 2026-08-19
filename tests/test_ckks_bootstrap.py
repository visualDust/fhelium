from __future__ import annotations

import gc
import os
from collections.abc import Iterator

import numpy as np
import pytest
import torch

import fhelium as fh
from fhelium import core
from fhelium.experimental import bootstrap as bs
from fhelium.experimental.bootstrap._modraise import (
    reference_centered_basis_extend,
)
from fhelium.experimental.bootstrap.presets import (
    cosine_depth_refresh_logn16_v1,
)


def test_legacy_evaluation_key_aggregate_remains_removed() -> None:
    assert "CkksEvaluationKeys" not in fh.__all__
    assert "CkksEvaluationKeys" not in core.__all__
    assert not hasattr(fh, "CkksEvaluationKeys")
    assert not hasattr(core, "CkksEvaluationKeys")
    assert "EvaluationKeySet" in core.__all__
    assert "EvaluationKeyRequirements" in core.__all__


def test_evaluation_key_set_revalidates_mutable_capability_roles() -> None:
    keys = fh.EvaluationKeySet()
    keys.relinearization = object()  # type: ignore[assignment]

    with pytest.raises(TypeError, match="RelinearizationKey or None"):
        keys.validate()

    keys.relinearization = None
    keys.rotations.table[1] = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="rotation values must be RotationKey"):
        keys.validate()


def test_approximation_and_evaluation_choices_are_separate() -> None:
    polynomial = bs.ChebyshevInterpolator(degree=12).approximate(
        np.sin,
        name="sine",
    )
    evaluator = bs.BinaryDecompositionChebyshevEvaluator(skip_near_zero=1e-15)

    assert polynomial.basis == "chebyshev"
    assert polynomial.max_error is not None and polynomial.max_error < 1e-10
    assert evaluator.required_levels(polynomial) > 0


def test_power_evaluator_depth_and_operation_inventories_are_exact() -> None:
    degree_four = bs.PolynomialApproximation(
        basis="power",
        coefficients=(0.01, 0.2, -0.3, 0.125, -0.0625),
    )
    degree_fifteen = bs.PolynomialApproximation(
        basis="power",
        coefficients=tuple(
            (-1.0) ** degree / 2 ** (degree + 1) for degree in range(16)
        ),
    )

    cases = (
        (
            bs.BalancedPowerEvaluator(),
            degree_four,
            3,
            (3, 4, 3, 10),
        ),
        (bs.HornerPowerEvaluator(), degree_four, 4, (3, 1, 3, 7)),
        (
            bs.PatersonStockmeyerPowerEvaluator(baby_step=2),
            degree_four,
            3,
            (2, 3, 4, 9),
        ),
        (
            bs.BalancedPowerEvaluator(),
            degree_fifteen,
            5,
            (14, 15, 7, 36),
        ),
        (
            bs.HornerPowerEvaluator(),
            degree_fifteen,
            15,
            (14, 1, 14, 29),
        ),
        (
            bs.PatersonStockmeyerPowerEvaluator(baby_step=4),
            degree_fifteen,
            6,
            (6, 12, 12, 30),
        ),
    )
    for evaluator, polynomial, levels, expected in cases:
        inventory = evaluator.operation_inventory(polynomial)
        assert evaluator.required_levels(polynomial) == levels
        assert (
            inventory["ciphertext_multiplications"],
            inventory["coefficient_multiplications"],
            inventory["alignment_multiplications"],
            inventory["total_multiplications"],
        ) == expected
        assert inventory["relinearizations"] == expected[0]
        assert inventory["rescale_operations"] == expected[-1]


def test_dense_chebyshev_inventory_includes_recurrence_alignment() -> None:
    polynomial = bs.PolynomialApproximation(
        basis="chebyshev",
        coefficients=tuple(
            (-1.0) ** degree / 2 ** (degree + 1) for degree in range(16)
        ),
    )
    evaluator = bs.BinaryDecompositionChebyshevEvaluator()

    assert evaluator.required_levels(polynomial) == 5
    assert evaluator.operation_inventory(polynomial) == {
        "ciphertext_multiplications": 14,
        "coefficient_multiplications": 15,
        "alignment_multiplications": 31,
        "relinearizations": 14,
        "rescale_operations": 60,
        "total_multiplications": 60,
    }


def test_fixed_baby_step_is_validated_and_part_of_depth() -> None:
    with pytest.raises(TypeError, match="baby_step must be an integer"):
        bs.PatersonStockmeyerPowerEvaluator(baby_step=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        bs.PatersonStockmeyerPowerEvaluator(baby_step=1)

    polynomial = bs.PolynomialApproximation(
        basis="power", coefficients=tuple(range(16))
    )
    assert bs.PatersonStockmeyerPowerEvaluator(baby_step=3).required_levels(
        polynomial
    ) != bs.PatersonStockmeyerPowerEvaluator(baby_step=4).required_levels(
        polynomial
    )


@pytest.mark.parametrize(
    "evaluator",
    [
        bs.HornerPowerEvaluator(),
        bs.PatersonStockmeyerPowerEvaluator(baby_step=4),
    ],
)
def test_power_only_evaluators_reject_chebyshev_basis(
    evaluator: object,
) -> None:
    polynomial = bs.PolynomialApproximation(
        basis="chebyshev", coefficients=(1.0, 2.0)
    )
    with pytest.raises(ValueError, match="requires power basis"):
        evaluator.required_levels(polynomial)  # type: ignore[attr-defined]


def test_periodic_reference_uses_normalized_coordinate_and_input_bound() -> (
    None
):
    reduction = bs.CosineDoubleAngleReduction(
        input_bound=16,
        double_angle_iterations=3,
        approximator=bs.ChebyshevInterpolator(degree=16),
        evaluator=bs.BinaryDecompositionChebyshevEvaluator(
            skip_near_zero=1e-15
        ),
        fuse_input_normalization=True,
    )
    normalized = np.linspace(-1.0, 1.0, 8193)[::1024]
    raw = reduction.input_bound * normalized
    expected = np.sin(np.pi * raw) / np.pi

    assert reduction.fused_input_divisor == reduction.input_bound
    assert reduction.requires_relinearization is True
    assert np.max(np.abs(reduction.reference(normalized) - expected)) <= (
        reduction.approximation_error
    )
    with pytest.raises(ValueError, match="requires relinearization key"):
        reduction.evaluate(  # type: ignore[arg-type]
            None,
            None,
            relinearization_key=None,
        )


def test_reference_radix2_transform_roundtrip() -> None:
    slots = 32
    compiler = bs.Radix2FourierTransformCompiler(stage_count=2)
    forward = compiler.compile(
        slots=slots,
        direction="coeffs_to_slots",
        generator=5,
        scale=1.0,
    )
    inverse = compiler.compile(
        slots=slots,
        direction="slots_to_coeffs",
        generator=5,
        scale=1.0 / slots,
    )
    rng = np.random.default_rng(20260725)
    values = rng.normal(size=slots) + 1j * rng.normal(size=slots)

    transformed = values
    for stage in (*forward, *inverse):
        transformed = stage.reference(transformed)

    np.testing.assert_allclose(transformed, values, atol=2e-14, rtol=0)


def test_reference_centered_basis_extension_handles_batches() -> None:
    residues = torch.tensor(
        [
            [[1, 10, 16], [1, 10, 3]],
            [[4, 7, 12], [4, 7, 12]],
        ],
        dtype=torch.int64,
    )
    result = reference_centered_basis_extend(
        residues,
        source_moduli=(17, 13),
        target_moduli=(19, 23),
    )
    roundtrip = reference_centered_basis_extend(
        result,
        source_moduli=(19, 23),
        target_moduli=(17, 13),
    )

    torch.testing.assert_close(roundtrip, residues)


@pytest.fixture(scope="module")
def evaluation_fixture() -> Iterator[
    tuple[
        fh.CkksEngine,
        fh.SecretKey,
        fh.PublicKey,
        fh.RelinearizationKey,
    ]
]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    engine = fh.CkksEngine(
        fh.Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        galois_generator=5,
        allow_sk_gen=False,
    )
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    relinearization_key = engine.create_relinearization_key(secret_key)
    yield engine, secret_key, public_key, relinearization_key
    del relinearization_key, public_key, secret_key, engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_polynomial_component_evaluates_numerically(
    evaluation_fixture: tuple[
        fh.CkksEngine,
        fh.SecretKey,
        fh.PublicKey,
        fh.RelinearizationKey,
    ],
) -> None:
    engine, secret_key, public_key, relinearization_key = evaluation_fixture
    values = torch.linspace(
        -0.025,
        0.025,
        engine.num_slots,
        dtype=torch.float64,
    )
    ciphertext = engine.encrypt_message(values, public_key)
    polynomial = bs.PolynomialApproximation(
        basis="power",
        coefficients=(0.01, 0.2, -0.3),
    )

    result = bs.BalancedPowerEvaluator().evaluate(
        engine,
        ciphertext,
        polynomial,
        relinearization_key=relinearization_key,
    )
    decoded = engine.decrypt_message(result, secret_key, is_real=True)
    expected = 0.01 + 0.2 * values - 0.3 * values**2

    torch.testing.assert_close(decoded, expected, atol=2e-5, rtol=0)


@pytest.mark.gpu
def test_linear_polynomial_needs_no_relinearization_key(
    evaluation_fixture: tuple[
        fh.CkksEngine,
        fh.SecretKey,
        fh.PublicKey,
        fh.RelinearizationKey,
    ],
) -> None:
    engine, secret_key, public_key, _ = evaluation_fixture
    values = torch.linspace(
        -0.025, 0.025, engine.num_slots, dtype=torch.float64
    )
    ciphertext = engine.encrypt_message(values, public_key)
    polynomial = bs.PolynomialApproximation(
        basis="power",
        coefficients=(0.01, 0.2),
    )

    result = bs.BalancedPowerEvaluator().evaluate(
        engine,
        ciphertext,
        polynomial,
    )
    decoded = engine.decrypt_message(result, secret_key, is_real=True)

    torch.testing.assert_close(decoded, 0.01 + 0.2 * values, atol=2e-5, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "evaluator",
    [
        bs.HornerPowerEvaluator(),
        bs.PatersonStockmeyerPowerEvaluator(baby_step=2),
    ],
)
def test_new_power_evaluators_validate_relinearization_before_execution(
    evaluation_fixture: tuple[
        fh.CkksEngine,
        fh.SecretKey,
        fh.PublicKey,
        fh.RelinearizationKey,
    ],
    evaluator: object,
) -> None:
    engine, _, public_key, _ = evaluation_fixture
    ciphertext = engine.encrypt_message(
        torch.zeros(engine.num_slots, dtype=torch.float64), public_key
    )
    polynomial = bs.PolynomialApproximation(
        basis="power", coefficients=(1.0, 2.0, 3.0)
    )

    with pytest.raises(ValueError, match="requires a relinearization key"):
        evaluator.evaluate(  # type: ignore[attr-defined]
            engine,
            ciphertext,
            polynomial,
            relinearization_key=None,
        )


@pytest.mark.gpu
def test_horner_rejects_insufficient_public_depth_before_execution(
    evaluation_fixture: tuple[
        fh.CkksEngine,
        fh.SecretKey,
        fh.PublicKey,
        fh.RelinearizationKey,
    ],
) -> None:
    engine, _, public_key, relinearization_key = evaluation_fixture
    ciphertext = engine.encrypt_message(
        torch.zeros(engine.num_slots, dtype=torch.float64),
        public_key,
        level=4,
    )
    polynomial = bs.PolynomialApproximation(
        basis="power", coefficients=(1.0, 2.0, 3.0, 4.0)
    )

    with pytest.raises(ValueError, match="final public level"):
        bs.HornerPowerEvaluator().evaluate(
            engine,
            ciphertext,
            polynomial,
            relinearization_key=relinearization_key,
        )


@pytest.mark.gpu
def test_new_power_evaluators_validate_entry_scale_and_arithmetic_state(
    evaluation_fixture: tuple[
        fh.CkksEngine,
        fh.SecretKey,
        fh.PublicKey,
        fh.RelinearizationKey,
    ],
) -> None:
    engine, _, public_key, _ = evaluation_fixture
    ciphertext = engine.encrypt_message(
        torch.zeros(engine.num_slots, dtype=torch.float64), public_key
    )
    polynomial = bs.PolynomialApproximation(
        basis="power", coefficients=(1.0, 2.0)
    )
    wrong_scale = ciphertext.clone()
    wrong_scale.scale *= 1.01
    with pytest.raises(ValueError, match="requires input scale"):
        bs.HornerPowerEvaluator().evaluate(
            engine,
            wrong_scale,
            polynomial,
        )

    ntt_ciphertext = engine.coefficient_domain_to_ntt_domain(ciphertext)
    with pytest.raises(ValueError, match="polynomial_domain"):
        bs.PatersonStockmeyerPowerEvaluator(baby_step=2).evaluate(
            engine,
            ntt_ciphertext,
            polynomial,
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FHELIUM_RUN_BOOTSTRAP_SLOW") != "1",
    reason="set FHELIUM_RUN_BOOTSTRAP_SLOW=1 for the N=2^16 test",
)
def test_logn16_noninteractive_bootstrap_end_to_end() -> None:
    engine = fh.CkksEngine(
        fh.CkksConfig.parse(
            fh.Preset.slots32768_scale50_levels27_int64,
            base_prime_bits=50,
        ),
        device="cuda:0",
        galois_generator=5,
        allow_sk_gen=False,
    )
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    bootstrap = cosine_depth_refresh_logn16_v1(engine)
    rotation_keys = bootstrap.create_rotation_keys(secret_key)
    relinearization_key = engine.create_relinearization_key(secret_key)
    conjugation_key = engine.create_conjugation_key(secret_key)
    evaluation_keys = core.EvaluationKeySet(
        rotations=rotation_keys,
        relinearization=relinearization_key,
        conjugation=conjugation_key,
    )

    values = torch.linspace(-0.1, 0.1, engine.num_slots, dtype=torch.float64)
    source = engine.encrypt_message(values, public_key, level=0)
    ones = torch.ones(engine.num_slots, dtype=torch.float64)
    while source.level < engine.final_public_level:
        identity = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                ones,
                level=source.level,
                scale=engine.config.default_scale,
            )
        )
        source = engine.rescale_to_next_level(
            engine.ntt_domain_to_coefficient_domain(
                engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(source), identity
                )
            )
        )

    refreshed = bootstrap(
        source,
        evaluation_keys=evaluation_keys,
    )
    decoded = engine.decrypt_message(refreshed, secret_key, is_real=True)
    error = (decoded - values).abs()

    assert refreshed.level == bootstrap.output_level == 19
    assert error.max().item() < 7e-3
    assert error.mean().item() < 1.5e-3
    assert torch.corrcoef(torch.stack((decoded, values)))[0, 1] > 0.999
