from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import fhelium as fh
from fhelium.core import (
    Ciphertext,
    CompressedPlaintext,
    Plaintext,
    PublicKey,
    SecretKey,
)
from fhelium.errors import InvalidScaleError
from fhelium.serialization import ValueEnvelope


@pytest.fixture(scope="module")
def engine() -> fh.CkksEngine:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return fh.CkksEngine(
        fh.Preset.slots8192_scale40_levels7_int64, device="cuda:0"
    )


def _deterministic_engine(seed: int) -> fh.CkksEngine:
    """Create an engine whose cryptographic regression vector is reproducible."""

    config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_levels7_int64)
    return fh.CkksEngine(
        config,
        device="cuda:0",
        rng_seed=seed,
        rng_nonce=0,
    )


def test_core_values_reject_invalid_scale_and_level_metadata() -> None:
    invalid_scales = (
        True,
        "1099511627776",
        0,
        -1,
        float("nan"),
        float("inf"),
    )
    for invalid_scale in invalid_scales:
        constructors = (
            lambda: Ciphertext(
                data=torch.zeros((2, 1, 2), dtype=torch.int64),
                level=0,
                scale=invalid_scale,
                context_id="context",
                prime_ids=(0,),
            ),
            lambda: Plaintext(
                message=torch.zeros(1),
                level=0,
                scale=invalid_scale,
            ),
            lambda: CompressedPlaintext(
                data=torch.zeros((1, 1), dtype=torch.int64),
                ring_dimension=2,
                compression_layout="cyclic",
                level=0,
                scale=invalid_scale,
                context_id="context",
                polynomial_domain="coefficient",
                modulus_basis="Q",
                residue_representation="montgomery",
                prime_ids=(0,),
            ),
            lambda: fh.CkksContextSpec(
                logN=1,
                default_scale=invalid_scale,
                q_moduli=(17,),
            ),
        )
        for constructor in constructors:
            with pytest.raises(InvalidScaleError):
                constructor()

    for invalid_level in (True, 1.5):
        for constructor in (
            lambda: Plaintext(
                message=torch.zeros(2),
                level=invalid_level,
                scale=2.0**40,
            ),
            lambda: Ciphertext(
                data=torch.zeros((2, 1, 8), dtype=torch.int64),
                level=invalid_level,
                scale=2.0**40,
                context_id="context",
                prime_ids=(0,),
            ),
        ):
            with pytest.raises(TypeError):
                constructor()


@pytest.mark.parametrize(
    "invalid_scale", [True, "1099511627776", 0, -1, float("nan"), float("inf")]
)
@pytest.mark.gpu
def test_public_scale_entry_points_validate_raw_values(
    engine: fh.CkksEngine, invalid_scale: object
) -> None:
    message = torch.zeros(4, dtype=torch.float64)
    for operation in (
        lambda: engine.plaintext(message, scale=invalid_scale),
        lambda: engine.encode(message, scale=invalid_scale),
        lambda: engine.encrypt_message(message, scale=invalid_scale),
    ):
        with pytest.raises(InvalidScaleError):
            operation()


@pytest.mark.parametrize("invalid_level", [True, 1.5, -1, 7])
@pytest.mark.gpu
def test_public_value_creation_requires_a_strict_public_level(
    engine: fh.CkksEngine, invalid_level: object
) -> None:
    message = torch.zeros(4, dtype=torch.float64)
    for operation in (
        lambda: engine.plaintext(message, level=invalid_level),
        lambda: engine.encode(message, level=invalid_level),
        lambda: engine.encrypt_message(message, level=invalid_level),
    ):
        with pytest.raises((TypeError, ValueError)):
            operation()


@pytest.mark.gpu
def test_message_encryption_validates_input_before_lazy_key_generation() -> (
    None
):
    engine = fh.CkksEngine(
        fh.Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    message = torch.zeros(4, dtype=torch.float64)

    with pytest.raises(ValueError, match="0 <= level"):
        engine.encrypt_message(message, level=-1)
    with pytest.raises(InvalidScaleError):
        engine.encrypt_message(message, scale=0)


@pytest.mark.gpu
def test_imported_engine_values_cannot_use_private_structural_levels(
    engine: fh.CkksEngine,
) -> None:
    encoded = engine.encode(torch.zeros(4))
    imported_plaintext = replace(encoded, level=engine.public_level_count)
    with pytest.raises(ValueError, match="0 <= level"):
        engine.integer_coefficients_to_rns(imported_plaintext)
    with pytest.raises(ValueError, match="0 <= level"):
        engine.encrypt(imported_plaintext)

    ciphertext = engine.encrypt_message(torch.zeros(4))
    imported_ciphertext = replace(ciphertext, level=engine.public_level_count)
    with pytest.raises(ValueError, match="0 <= level"):
        engine.coefficient_domain_to_ntt_domain(imported_ciphertext)


@pytest.mark.parametrize("invalid_basis", [True, "q"])
@pytest.mark.gpu
def test_integer_coefficients_to_rns_rejects_invalid_basis_before_native_lifting(
    engine: fh.CkksEngine, invalid_basis: object
) -> None:
    encoded = engine.encode(torch.zeros(4))
    with pytest.raises((TypeError, ValueError), match="modulus_basis"):
        engine.integer_coefficients_to_rns(
            encoded,
            modulus_basis=invalid_basis,  # type: ignore[arg-type]
        )


def test_integer_coefficients_require_integral_tensor_storage() -> None:
    with pytest.raises(TypeError, match="integral"):
        Plaintext(
            message=None,
            level=0,
            scale=2.0**40,
            data=torch.arange(8, dtype=torch.float64),
            context_id="context",
            representation="integer_coefficients",
            polynomial_domain="coefficient",
        )


@pytest.mark.gpu
def test_decrypt_coefficients_are_explicitly_approximate_and_decode_only() -> (
    None
):
    """Check the representation invariants against one reproducible noise vector."""

    engine = _deterministic_engine(seed=0)
    message = torch.linspace(-0.01, 0.01, 16, dtype=torch.float64)
    ciphertext = engine.encrypt_message(message)
    decrypted = engine.decrypt(ciphertext)

    assert decrypted.representation == "approximate_coefficients"
    assert decrypted.data is not None and decrypted.data.dtype == torch.float64
    torch.testing.assert_close(
        engine.decode(decrypted, is_real=True)[: message.numel()],
        message,
        atol=2e-6,
        rtol=0,
    )
    assert ValueEnvelope.from_value(decrypted).metadata["representation"] == (
        "approximate_coefficients"
    )

    for operation in (
        lambda: engine.encrypt(decrypted),
        lambda: engine.integer_coefficients_to_rns(decrypted),
        lambda: engine.prepare_plaintext_for_addition(decrypted),
        lambda: engine.prepare_plaintext_for_multiplication(decrypted),
    ):
        with pytest.raises(ValueError, match="integer_coefficients"):
            operation()


@pytest.mark.parametrize("malformed_dtype", [torch.float64, torch.int32])
@pytest.mark.gpu
def test_integer_coefficient_consumers_reject_nonconfigured_dtype_before_native(
    engine: fh.CkksEngine,
    malformed_dtype: torch.dtype,
) -> None:
    malformed = engine.encode(torch.zeros(4))
    assert malformed.data is not None
    malformed.data = malformed.data.to(malformed_dtype)
    for operation in (engine.integer_coefficients_to_rns, engine.encrypt):
        with pytest.raises(TypeError, match="dtype does not match engine"):
            operation(malformed)


@pytest.mark.parametrize(
    ("representation", "dtype"),
    [
        ("integer_coefficients", torch.int64),
        ("approximate_coefficients", torch.float64),
    ],
)
@pytest.mark.gpu
def test_decode_rejects_wrong_coefficient_ring_dimension(
    engine: fh.CkksEngine,
    representation: str,
    dtype: torch.dtype,
) -> None:
    malformed = Plaintext(
        message=None,
        level=0,
        scale=2.0**40,
        data=torch.zeros(
            engine.config.N - 1, dtype=dtype, device=engine.device
        ),
        context_id=engine.context.context_id,
        representation=representation,  # type: ignore[arg-type]
        polynomial_domain="coefficient",
    )
    with pytest.raises(ValueError, match="ring dimension"):
        engine.decode(malformed)


def test_approximate_coefficients_require_dense_finite_float64_tensor() -> None:
    common = dict(
        message=None,
        level=0,
        scale=2.0**40,
        context_id="context",
        representation="approximate_coefficients",
        polynomial_domain="coefficient",
    )
    with pytest.raises(TypeError, match="torch.Tensor"):
        Plaintext(data=[0.0], **common)  # type: ignore[arg-type]
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([1.0]),
        (8,),
        check_invariants=False,
    )
    with pytest.raises(TypeError, match="dense strided"):
        Plaintext(data=sparse, **common)
    with pytest.raises(ValueError, match="finite"):
        Plaintext(
            data=torch.tensor([float("nan")], dtype=torch.float64), **common
        )


@pytest.mark.gpu
def test_plaintext_raw_transition_chain_is_composable_and_axis_exact(
    engine: fh.CkksEngine,
) -> None:
    encoded = engine.encode(torch.linspace(-0.01, 0.01, 16), level=2)
    standard = engine.integer_coefficients_to_rns(encoded, modulus_basis="QP")
    montgomery = engine.standard_residues_to_montgomery_residues(standard)
    ntt = engine.coefficient_domain_to_ntt_domain(montgomery)
    coefficient_montgomery = engine.ntt_domain_to_coefficient_domain(ntt)
    roundtrip = engine.montgomery_residues_to_standard_residues(
        coefficient_montgomery
    )

    assert encoded.representation == "integer_coefficients"
    assert (standard.polynomial_domain, standard.residue_representation) == (
        "coefficient",
        "standard",
    )
    assert (
        montgomery.polynomial_domain,
        montgomery.residue_representation,
    ) == (
        "coefficient",
        "montgomery",
    )
    assert (ntt.polynomial_domain, ntt.residue_representation) == (
        "ntt",
        "montgomery",
    )
    assert (
        coefficient_montgomery.polynomial_domain,
        coefficient_montgomery.residue_representation,
    ) == ("coefficient", "montgomery")
    assert roundtrip.residue_representation == "standard"
    assert standard.data is not None and roundtrip.data is not None
    expected_standard = standard.data.clone()
    actual_standard = roundtrip.data.clone()
    engine.rns_runtime.canonicalize_residues_(expected_standard, include_p=True)
    engine.rns_runtime.canonicalize_residues_(actual_standard, include_p=True)
    torch.testing.assert_close(
        actual_standard, expected_standard, rtol=0, atol=0
    )

    identity = (
        standard.representation,
        standard.modulus_basis,
        standard.prime_ids,
        standard.level,
        standard.scale,
    )
    for value in (montgomery, ntt, coefficient_montgomery, roundtrip):
        assert (
            value.representation,
            value.modulus_basis,
            value.prime_ids,
            value.level,
            value.scale,
        ) == identity

    addition = engine.prepare_plaintext_for_addition(
        encoded, modulus_basis="QP"
    )
    multiplication = engine.prepare_plaintext_for_multiplication(
        encoded, modulus_basis="QP"
    )
    assert addition.residue_representation == "montgomery"
    assert addition.polynomial_domain == "coefficient"
    assert multiplication.residue_representation == "montgomery"
    assert multiplication.polynomial_domain == "ntt"

    inplace = standard.clone()
    assert inplace.data is not None
    pointer = inplace.data.data_ptr()
    returned = engine.standard_residues_to_montgomery_residues_(inplace)
    assert returned is inplace
    assert inplace.data is not None and inplace.data.data_ptr() == pointer
    assert standard.data.data_ptr() != inplace.data.data_ptr()

    strict_source_cases = (
        (
            engine.standard_residues_to_montgomery_residues,
            montgomery,
            "requires standard residues",
        ),
        (
            engine.montgomery_residues_to_standard_residues,
            standard,
            "requires Montgomery residues",
        ),
        (
            engine.coefficient_domain_to_ntt_domain,
            ntt,
            "requires a coefficient-domain plaintext",
        ),
        (
            engine.ntt_domain_to_coefficient_domain,
            coefficient_montgomery,
            "requires an NTT-domain plaintext",
        ),
    )
    for transition, source, message in strict_source_cases:
        with pytest.raises(ValueError, match=message):
            transition(source)

    with pytest.raises(ValueError, match="requires Montgomery"):
        engine.coefficient_domain_to_ntt_domain(standard)
    with pytest.raises(ValueError, match="coefficient-domain"):
        engine.montgomery_residues_to_standard_residues(ntt)
    malformed_basis = standard.clone()
    malformed_basis.modulus_basis = "bad"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="modulus_basis"):
        engine.coefficient_domain_to_ntt_domain_(malformed_basis)
    with pytest.raises(ValueError, match="RNS reconstruction is not implicit"):
        engine.decode(addition)


def test_core_rejects_unsupported_ntt_standard_plaintext() -> None:
    with pytest.raises(ValueError, match="must use Montgomery"):
        Plaintext(
            message=None,
            level=0,
            scale=2.0**40,
            data=torch.zeros((1, 8), dtype=torch.int64),
            context_id="context",
            representation="rns",
            polynomial_domain="ntt",
            modulus_basis="Q",
            residue_representation="standard",
            prime_ids=(0,),
        )


@pytest.mark.parametrize("prime_ids", [(), (True,), (0.0,), (2, 1)])
def test_core_prime_ids_are_not_coerced(prime_ids: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        Ciphertext(
            data=torch.zeros((2, len(prime_ids), 8), dtype=torch.int64),
            level=0,
            scale=2.0**40,
            context_id="context",
            prime_ids=prime_ids,  # type: ignore[arg-type]
        )


def test_context_bound_values_and_native_payloads_are_structural() -> None:
    with pytest.raises(ValueError, match="non-empty context_id"):
        SecretKey(
            data=torch.zeros((1, 8), dtype=torch.int64),
            context_id="",
            prime_ids=(0,),
        )
    with pytest.raises(TypeError, match="integral"):
        PublicKey(
            data=torch.zeros((2, 1, 8), dtype=torch.float64),
            context_id="context",
            prime_ids=(0,),
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        Plaintext(
            message=None,
            level=0,
            scale=2.0**40,
            data=torch.zeros((1, 0), dtype=torch.int64),
            context_id="context",
            representation="rns",
            polynomial_domain="coefficient",
            modulus_basis="Q",
            residue_representation="standard",
            prime_ids=(0,),
        )


@pytest.mark.gpu
def test_exact_key_consumers_reject_malformed_or_wrong_concrete_states(
    engine: fh.CkksEngine,
) -> None:
    secret = engine.create_secret_key()
    coefficient_secret = replace(
        secret,
        polynomial_domain="coefficient",
        residue_representation="standard",
    )
    with pytest.raises(ValueError, match="NTT polynomial domain"):
        engine.set_secret_key(coefficient_secret)

    wrong_dtype = replace(secret, data=secret.data.to(torch.int32))
    with pytest.raises(TypeError, match="dtype"):
        engine.set_secret_key(wrong_dtype)

    relinearization = engine.create_relinearization_key(secret)
    malformed_digits = replace(
        relinearization, data=relinearization.data[:-1].clone()
    )
    with pytest.raises(ValueError, match="digit count"):
        engine.set_relinearization_key(malformed_digits)

    generic_key = fh.KeySwitchKey(
        data=relinearization.data.clone(),
        context_id=relinearization.context_id,
        prime_ids=relinearization.prime_ids,
    )
    with pytest.raises(TypeError, match="RelinearizationKey"):
        engine.set_relinearization_key(generic_key)  # type: ignore[arg-type]


@pytest.mark.gpu
def test_qp_ciphertext_rejects_q_only_key_switch_operations(
    engine: fh.CkksEngine,
) -> None:
    qp_secret = engine.create_secret_key(modulus_basis="QP")
    qp_public = engine.create_public_key(qp_secret, modulus_basis="QP")
    q_secret = engine.create_secret_key(modulus_basis="Q")
    ciphertext = engine.encrypt_message(torch.zeros(4), qp_public)
    assert ciphertext.modulus_basis == "QP"
    with pytest.raises(ValueError, match="requires a QP SecretKey"):
        engine.decrypt(ciphertext, q_secret)

    relinearization_key = engine.create_relinearization_key(engine.secret_key)
    rotation_key = engine.create_rotation_key(1, engine.secret_key)
    conjugation_key = engine.create_conjugation_key(engine.secret_key)
    product = engine.multiply(
        engine.coefficient_domain_to_ntt_domain(ciphertext),
        engine.coefficient_domain_to_ntt_domain(ciphertext),
    )
    operations = (
        lambda: engine.relinearize(product, relinearization_key),
        lambda: engine.switch_key(ciphertext, relinearization_key),
        lambda: engine.rotate_with_key(ciphertext, rotation_key),
        lambda: engine.rotate_by_step(ciphertext, 1),
        lambda: engine.rotate_many_by_steps(ciphertext, (1, 2)),
        lambda: engine.rotate_many_with_keys(ciphertext, (rotation_key,)),
        lambda: engine.conjugate(ciphertext, conjugation_key),
    )
    for operation in operations:
        with pytest.raises(ValueError, match="modulus_basis.*expected 'Q'"):
            operation()


@pytest.mark.gpu
def test_rotate_many_by_steps_rejects_three_components_before_hoisting(
    engine: fh.CkksEngine,
) -> None:
    ciphertext = engine.encrypt_message(torch.zeros(4))
    three_components = replace(
        ciphertext,
        data=torch.cat(
            (ciphertext.data, torch.zeros_like(ciphertext.data[:1]))
        ),
    )
    with pytest.raises(ValueError, match="expected 2"):
        engine.rotate_many_by_steps(
            three_components,
            (1, 2),
            use_hoisting=True,
        )
