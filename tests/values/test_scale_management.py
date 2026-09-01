import math
from typing import Literal

import pytest
import torch

from fhelium import (
    CkksConfig,
    Preset,
)
from fhelium.eager import Engine
from fhelium.errors import (
    InvalidScaleError,
    MaximumLevelError,
    ScaleMismatchError,
)


def test_config_names_the_encoding_scale_as_a_default() -> None:
    config = CkksConfig(logN=14, num_scale_primes=2)

    assert config.default_scale == float(1 << config.scale_bits)
    assert not hasattr(config, "scale")


@pytest.mark.parametrize("count", [0, -1])
def test_config_requires_at_least_one_public_level(count: int) -> None:
    with pytest.raises(ValueError, match="num_scale_primes must be at least 1"):
        CkksConfig(logN=14, num_scale_primes=count)


@pytest.mark.parametrize("count", [True, 2.0])
def test_config_requires_an_integer_public_level_count(count: object) -> None:
    with pytest.raises(TypeError, match="num_scale_primes must be an integer"):
        CkksConfig(logN=14, num_scale_primes=count)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def engine() -> Engine:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return Engine(Preset.slots8192_scale40_levels7_int64)


def _deterministic_engine(seed: int) -> Engine:
    """Create an engine whose cryptographic test vectors are reproducible."""

    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)
    return Engine(
        config,
        rng_seed=seed,
        rng_nonce=0,
    )


@pytest.mark.gpu
def test_public_level_limit_and_rescale_to_next_level_queries_are_explicit(
    engine: Engine,
) -> None:
    scale = 2.0**78
    dropped_prime = engine.rescale_to_next_drop_prime(level=0)

    assert engine.public_level_count == engine.config.num_scale_primes
    assert engine.final_public_level == engine.public_level_count - 1
    assert not hasattr(engine, "ckks_level_count")
    assert not hasattr(engine, "rescale")
    assert not hasattr(engine, "rescale_")
    assert not hasattr(engine, "rescale_prime")
    assert not hasattr(engine, "rescale_drop_prime")
    assert not hasattr(engine, "scale_after_rescale")
    assert not hasattr(engine, "reinterpret_scale")
    assert not hasattr(engine, "reinterpret_scale_")
    assert dropped_prime == engine.config.q_moduli[0]
    assert (
        engine.rescale_to_next_drop_prime(level=2) == engine.config.q_moduli[2]
    )
    assert (
        engine.rescale_to_next_output_scale(scale, level=0)
        == scale / dropped_prime
    )

    with pytest.raises(TypeError, match="level must be an integer"):
        engine.rescale_to_next_drop_prime(level=True)
    with pytest.raises(ValueError, match="non-negative"):
        engine.rescale_to_next_drop_prime(level=-1)
    assert (
        engine.rescale_to_next_drop_prime(level=engine.final_public_level)
        == engine.config.q_moduli[engine.final_public_level]
    )
    with pytest.raises(MaximumLevelError):
        engine.rescale_to_next_drop_prime(level=engine.final_public_level + 1)
    with pytest.raises(InvalidScaleError):
        engine.rescale_to_next_output_scale(math.nan, level=0)
    with pytest.raises(InvalidScaleError):
        engine.rescale_to_next_output_scale(
            float.fromhex("0x0.0000000000001p-1022"),
            level=0,
        )


def test_rescale_executes_the_selected_drop_at_the_final_public_level() -> None:
    runtime = Engine(
        Preset.slots8192_scale40_levels7_int64,
        rng_seed=19,
        rng_nonce=0,
    )
    value = runtime.encrypt_message(
        torch.zeros(8, dtype=torch.float64),
        level=runtime.final_public_level,
    )

    result = runtime.rescale_to_next_level(value)
    assert result.level == runtime.public_level_count
    assert result.prime_ids == value.prime_ids[1:]
    assert result.limb_count == value.limb_count - 1


@pytest.mark.gpu
@pytest.mark.parametrize("rounding", ["nearest", "floor"])
def test_arbitrary_scales_multiply_and_rescale_by_the_actual_prime(
    engine: Engine,
    rounding: Literal["nearest", "floor"],
) -> None:
    message = torch.linspace(-0.01, 0.01, 32, dtype=torch.float64)
    factor_message = torch.full((32,), 1.25, dtype=torch.float64)
    source = engine.encrypt_message(message, scale=2.0**39)
    factor = engine.prepare_plaintext_for_multiplication(
        engine.encode(factor_message, level=source.level, scale=2.0**39)
    )

    product = engine.multiply_plaintext(
        engine.coefficient_domain_to_ntt_domain(source), factor
    )
    assert product.scale == 2.0**78
    expected_scale = engine.rescale_to_next_output_scale(
        product.scale,
        level=product.level,
    )

    unchecked = engine.rescale_to_next_level(product, rounding=rounding)
    assert unchecked.level == product.level + 1
    assert unchecked.scale == expected_scale

    product_coefficient = engine.ntt_domain_to_coefficient_domain(product)
    rescaled = engine.rescale_to_next_level(
        product_coefficient, rounding=rounding
    )
    assert rescaled.level == product.level + 1
    assert rescaled.scale == expected_scale
    torch.testing.assert_close(
        engine.decrypt_message(rescaled, is_real=True, device="cpu")[:32],
        message * factor_message,
        atol=1e-5 if rounding == "nearest" else 3e-5,
        rtol=0.0,
    )

    inplace = product_coefficient.clone()
    assert engine.rescale_to_next_level_(inplace, rounding=rounding) is inplace
    assert inplace.scale == expected_scale
    assert torch.equal(inplace.data, rescaled.data)

    if rounding == "nearest":
        with pytest.raises(ValueError, match="rounding must be"):
            engine.rescale_to_next_level(
                product_coefficient,
                rounding="truncate",  # type: ignore[arg-type]
            )


@pytest.mark.gpu
def test_three_component_ciphertext_can_be_rescaled_before_relinearization() -> (
    None
):
    """Check this noisier operation order against one reproducible vector.

    Rescaling three components before relinearization drops the scale before
    key-switch noise is added. Consequently a tolerance test over fresh random
    keys is probabilistic rather than a stable numerical guarantee. The fixed
    CSPRNG vector makes this a deterministic implementation-regression test;
    it is not claimed as a worst-case CKKS noise bound.
    """

    engine = _deterministic_engine(seed=0)
    left_message = torch.linspace(-0.005, 0.005, 32, dtype=torch.float64)
    right_message = torch.full((32,), 0.5, dtype=torch.float64)
    left = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(left_message, scale=2.0**40)
    )
    right = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(right_message, scale=2.0**40)
    )

    triplet = engine.multiply(left, right)
    assert triplet.scale == 2.0**80
    rescaled_triplet = engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(triplet)
    )
    assert rescaled_triplet.component_count == 3
    assert rescaled_triplet.scale == engine.rescale_to_next_output_scale(
        triplet.scale,
        level=triplet.level,
    )

    result = engine.relinearize(
        engine.coefficient_domain_to_ntt_domain(rescaled_triplet)
    )
    torch.testing.assert_close(
        engine.decrypt_message(result, is_real=True, device="cpu")[:32],
        left_message * right_message,
        atol=2e-6,
        rtol=0.0,
    )


@pytest.mark.gpu
def test_qp_ciphertext_rescale_preserves_p_rows_and_message() -> None:
    """Check QP row retention against one reproducible numerical vector.

    Fresh encryption/key noise makes a numerical tolerance over an unseeded
    key a probabilistic test. The fixed vector keeps this focused on QP row
    selection and rescale regression rather than claiming a worst-case CKKS
    noise bound.
    """

    engine = _deterministic_engine(seed=0)
    message = torch.linspace(-0.005, 0.005, 32, dtype=torch.float64)
    qp_public_key = engine.create_public_key(
        engine.secret_key, modulus_basis="QP"
    )
    source = engine.encrypt_message(message, qp_public_key)
    identity = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.ones(32, dtype=torch.float64),
            level=source.level,
            scale=engine.config.default_scale,
        ),
        modulus_basis='QP',
    )
    pending = engine.ntt_domain_to_coefficient_domain(
        engine.multiply_plaintext(
            engine.coefficient_domain_to_ntt_domain(source), identity
        )
    )

    rescaled = engine.rescale_to_next_level(pending)
    assert rescaled.modulus_basis == "QP"
    assert rescaled.prime_ids == pending.prime_ids[1:]
    assert rescaled.limb_count == pending.limb_count - 1
    assert rescaled.scale == engine.rescale_to_next_output_scale(
        pending.scale,
        level=pending.level,
    )
    torch.testing.assert_close(
        engine.decrypt_message(rescaled, is_real=True, device="cpu")[:32],
        message,
        atol=1e-6,
        rtol=0.0,
    )

    inplace = pending.clone()
    assert engine.rescale_to_next_level_(inplace) is inplace
    assert inplace.prime_ids == rescaled.prime_ids
    assert torch.equal(inplace.data, rescaled.data)


@pytest.mark.gpu
def test_mod_switch_drops_q_rows_without_changing_scale(
    engine: Engine,
) -> None:
    message = torch.linspace(-0.01, 0.01, 32, dtype=torch.float64)
    source = engine.encrypt_message(
        message,
        scale=engine.config.default_scale,
    )

    switched = engine.mod_switch_to_level(source, 3)
    assert source.level == 0
    assert switched.level == 3
    assert switched.scale == source.scale
    assert switched.prime_ids == source.prime_ids[3:]
    assert torch.equal(switched.data, source.data[..., 3:, :])
    torch.testing.assert_close(
        engine.decrypt_message(switched, is_real=True, device="cpu")[:32],
        message,
        atol=1e-5,
        rtol=0.0,
    )

    inplace = source.clone()
    storage = inplace.data.untyped_storage().data_ptr()
    assert engine.mod_switch_to_next_level_(inplace) is inplace
    assert inplace.level == 1
    assert inplace.scale == source.scale
    assert inplace.data.untyped_storage().data_ptr() == storage
    assert torch.equal(inplace.data, source.data[..., 1:, :])

    ntt_source = engine.coefficient_domain_to_ntt_domain(source)
    ntt_switched = engine.mod_switch_to_next_level(ntt_source)
    assert (
        ntt_switched.is_ntt_domain
        and ntt_switched.residue_representation == "montgomery"
    )
    assert ntt_switched.scale == ntt_source.scale
    assert torch.equal(ntt_switched.data, ntt_source.data[..., 1:, :])

    final = engine.mod_switch_to_level(
        source,
        engine.final_public_level,
    )
    structural = engine.mod_switch_to_next_level(final)
    assert structural.level == engine.config.num_q_primes - 1
    assert structural.prime_ids == (engine.config.num_q_primes - 1,)
    with pytest.raises(ValueError, match="level must be"):
        engine.mod_switch_to_next_level(structural)


@pytest.mark.gpu
def test_add_executes_at_caller_metadata_and_reinterpretation_can_be_guarded(
    engine: Engine,
) -> None:
    message = torch.linspace(-0.01, 0.01, 32, dtype=torch.float64)
    left = engine.encrypt_message(message, scale=2.0**35)
    right = engine.encrypt_message(message, scale=2.0**35)
    target_scale = math.nextafter(right.scale, math.inf)

    reinterpreted = engine.reinterpret_at_scale(right, target_scale)
    assert right.scale == 2.0**35
    assert reinterpreted.scale == target_scale
    assert torch.equal(reinterpreted.data, right.data)
    added = engine.add(left, reinterpreted)
    assert added.scale == left.scale
    assert torch.equal(added.data, engine.add(left, right).data)

    addend = engine.prepare_plaintext_for_addition(
        engine.encode(
            torch.ones(32, dtype=torch.float64),
            level=left.level,
            scale=target_scale,
        )
    )
    plaintext_added = engine.add_plaintext(left, addend)
    assert plaintext_added.scale == left.scale

    guarded = engine.reinterpret_at_scale(
        right,
        right.scale * (1.0 + 1e-6),
        max_relative_change=2e-6,
    )
    assert guarded.scale == right.scale * (1.0 + 1e-6)
    with pytest.raises(ScaleMismatchError):
        engine.reinterpret_at_scale(
            right,
            right.scale * 1.01,
            max_relative_change=1e-3,
        )
