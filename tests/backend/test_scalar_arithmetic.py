"""Behavior of lightweight CKKS arithmetic with real and integer scalars."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import pytest
import torch
from fhelium import Preset
from fhelium.eager import Engine
from fhelium.values import Ciphertext, PublicKey, SecretKey


@dataclass(frozen=True)
class _ScalarFixture:
    engine: Engine
    secret_key: SecretKey
    public_key: PublicKey
    ciphertext: Ciphertext
    decoded_input: torch.Tensor


@pytest.fixture(scope="module")
def scalar_fixture() -> _ScalarFixture:
    engine = Engine(
        Preset.slots8192_scale40_depth7_int64,
        rng_seed=611,
        rng_nonce=29,
    )
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    message = torch.linspace(
        -0.03125,
        0.046875,
        engine.num_slots,
        dtype=torch.float64,
    )
    plaintext = engine.encode(message)
    ciphertext = engine.encrypt(plaintext, public_key)
    decoded_input = engine.decode(
        engine.decrypt(ciphertext, secret_key), is_real=True
    )
    return _ScalarFixture(
        engine,
        secret_key,
        public_key,
        ciphertext,
        decoded_input,
    )


def _decode(fixture: _ScalarFixture, value: Ciphertext) -> torch.Tensor:
    return fixture.engine.decode(
        fixture.engine.decrypt(value, fixture.secret_key), is_real=True
    )


def test_scalar_arithmetic_supports_int32_residues() -> None:
    engine = Engine(
        Preset.slots8192_scale25_depth14_int32,
        rng_seed=431,
    )
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    message = torch.full(
        (engine.num_slots,),
        0.03125,
        dtype=torch.float64,
    )
    source = engine.encrypt(engine.encode(message), public_key)

    def decode(value: Ciphertext) -> torch.Tensor:
        return engine.decode(engine.decrypt(value, secret_key), is_real=True)

    decoded_source = decode(source)
    added = engine.add_scalar(source, -0.125)
    real_product = engine.multiply_scalar(source, -0.75)
    integer_product = engine.multiply_integer_scalar(source, -7)
    assert added.data.dtype == torch.int32
    assert real_product.data.dtype == torch.int32
    assert integer_product.data.dtype == torch.int32
    torch.testing.assert_close(
        decode(added),
        decoded_source - 0.125,
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        decode(real_product),
        decoded_source * -0.75,
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        decode(integer_product),
        decoded_source * -7,
        atol=2e-6,
        rtol=0.0,
    )


def test_add_scalar_uses_the_caller_selected_scale(
    scalar_fixture: _ScalarFixture,
) -> None:
    fixture = scalar_fixture
    source = fixture.ciphertext
    scalar = -0.125
    scalar_scale = source.scale / 4.0
    source_data = source.data.clone()

    result = fixture.engine.add_scalar(
        source,
        scalar,
        scalar_scale=scalar_scale,
    )

    expected_addend = scalar * scalar_scale / source.scale
    torch.testing.assert_close(
        _decode(fixture, result),
        fixture.decoded_input + expected_addend,
        atol=2e-6,
        rtol=0.0,
    )
    assert result.scale == source.scale
    assert result.depth == source.depth
    assert result.polynomial_domain == source.polynomial_domain
    assert result.residue_representation == source.residue_representation
    assert result.data.data_ptr() != source.data.data_ptr()
    assert torch.equal(source.data, source_data)
    assert torch.equal(result.data[1:], source.data[1:])
    assert torch.equal(result.data[0, ..., 1:], source.data[0, ..., 1:])

    default_scale_result = fixture.engine.add_scalar(source, scalar)
    torch.testing.assert_close(
        _decode(fixture, default_scale_result),
        fixture.decoded_input + scalar,
        atol=2e-6,
        rtol=0.0,
    )


def test_multiply_scalar_records_product_scale_without_rescaling(
    scalar_fixture: _ScalarFixture,
) -> None:
    fixture = scalar_fixture
    source = fixture.ciphertext
    scalar = -0.75
    scalar_scale = source.scale

    result = fixture.engine.multiply_scalar(source, scalar)

    assert result.scale == source.scale * scalar_scale
    assert result.depth == source.depth
    assert result.prime_ids == source.prime_ids
    torch.testing.assert_close(
        _decode(fixture, result),
        fixture.decoded_input * scalar,
        atol=2e-6,
        rtol=0.0,
    )

    dropped_prime = fixture.engine.config.moduli[source.prime_ids[0]]
    rescaled = fixture.engine.rescale_to_next_depth(result)
    assert rescaled.depth == source.depth + 1
    assert rescaled.scale == result.scale / dropped_prime
    torch.testing.assert_close(
        _decode(fixture, rescaled),
        fixture.decoded_input * scalar,
        atol=2e-6,
        rtol=0.0,
    )


def test_integer_scalar_multiplication_is_exact_and_preserves_scale(
    scalar_fixture: _ScalarFixture,
) -> None:
    fixture = scalar_fixture
    source = fixture.ciphertext
    scalar = -7

    result = fixture.engine.multiply_integer_scalar(source, scalar)

    expected = torch.empty_like(source.data)
    for row, prime_id in enumerate(source.prime_ids):
        modulus = fixture.engine.config.moduli[prime_id]
        expected[..., row, :] = torch.remainder(
            source.data[..., row, :].to(torch.int64) * scalar,
            modulus,
        ).to(source.data.dtype)
    assert torch.equal(result.data, expected)
    assert result.scale == source.scale
    assert result.depth == source.depth
    assert result.prime_ids == source.prime_ids
    assert result.data.data_ptr() != source.data.data_ptr()

    active_modulus_product = prod(
        fixture.engine.config.moduli[index] for index in source.prime_ids
    )
    congruent = fixture.engine.multiply_integer_scalar(
        source,
        scalar + active_modulus_product,
    )
    assert torch.equal(congruent.data, result.data)

    later = fixture.engine.mod_switch_to_depth(source, 2)
    later_result = fixture.engine.multiply_integer_scalar(later, scalar)
    later_expected = torch.empty_like(later.data)
    for row, prime_id in enumerate(later.prime_ids):
        modulus = fixture.engine.config.moduli[prime_id]
        later_expected[..., row, :] = torch.remainder(
            later.data[..., row, :].to(torch.int64) * scalar,
            modulus,
        ).to(later.data.dtype)
    assert torch.equal(later_result.data, later_expected)
    assert later_result.depth == 2
    assert later_result.prime_ids == later.prime_ids


def test_scalar_multiplication_preserves_ntt_montgomery_representation(
    scalar_fixture: _ScalarFixture,
) -> None:
    fixture = scalar_fixture
    source_ntt = fixture.engine.coefficient_domain_to_ntt_domain(
        fixture.ciphertext
    )

    with pytest.raises(
        ValueError,
        match="coefficient-domain standard residues",
    ):
        fixture.engine.add_scalar(source_ntt, 0.25)

    integer = fixture.engine.multiply_integer_scalar(source_ntt, -3)
    real = fixture.engine.multiply_scalar(source_ntt, 0.5)
    assert integer.polynomial_domain == "ntt"
    assert integer.residue_representation == "montgomery"
    assert integer.scale == source_ntt.scale
    assert real.polynomial_domain == "ntt"
    assert real.residue_representation == "montgomery"
    assert real.scale == source_ntt.scale * source_ntt.scale

    integer_coefficients = fixture.engine.ntt_domain_to_coefficient_domain(
        integer
    )
    real_coefficients = fixture.engine.ntt_domain_to_coefficient_domain(real)
    torch.testing.assert_close(
        _decode(fixture, integer_coefficients),
        fixture.decoded_input * -3,
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _decode(fixture, real_coefficients),
        fixture.decoded_input * 0.5,
        atol=2e-6,
        rtol=0.0,
    )
