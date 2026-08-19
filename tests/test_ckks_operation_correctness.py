"""End-to-end correctness criteria for the public CKKS operation surface.

These tests intentionally compare decrypted results with cleartext references.
Shape-only and state-only assertions are useful, but they cannot detect a
native RNS/NTT kernel that returns a structurally valid wrong answer.

The suite is the correctness baseline for refactoring the local execution ABI.
It covers the CPU and CUDA device defaults here; backend-family equivalence
remains covered by ``test_ntt_backend.py`` and distributed communication
remains covered by the multi-rank SPMD example/tests.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator

import pytest
import torch

from fhelium import (
    Ciphertext,
    CkksConfig,
    CkksEngine,
    Plaintext,
    Preset,
    RotationKeySet,
)
from fhelium.errors import MaximumLevelError


def test_default_device_falls_back_to_included_cpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fhelium import native as native_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        native_module,
        "native_backend_available",
        lambda backend: backend == "cpu",
    )
    engine = CkksEngine(Preset.slots8192_scale40_levels7_int64)
    assert engine.device == torch.device("cpu")


_CODEC_ATOL = 1e-5
_ARITHMETIC_ATOL = 2e-5
_KEYSWITCH_ATOL = 2e-5
_MULTIPLICATION_ATOL = 3e-5


@pytest.fixture(
    scope="module",
    params=(
        pytest.param("cpu", id="cpu"),
        pytest.param("cuda:0", marks=pytest.mark.gpu, id="cuda"),
    ),
)
def engine(request: pytest.FixtureRequest) -> Iterator[CkksEngine]:
    device = str(request.param)
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    instance = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device=device,
    )
    yield instance

    del instance
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _message(
    engine: CkksEngine, *, complex_values: bool = True
) -> torch.Tensor:
    """Return a deterministic, bounded, full-slot message."""

    index = torch.arange(engine.num_slots, dtype=torch.float64)
    real = 0.012 * torch.sin(index * 0.013) + 0.006 * torch.cos(index * 0.007)
    if not complex_values:
        return real
    imag = 0.009 * torch.cos(index * 0.011) - 0.004 * torch.sin(index * 0.005)
    return torch.complex(real, imag)


def _other_message(engine: CkksEngine) -> torch.Tensor:
    index = torch.arange(engine.num_slots, dtype=torch.float64)
    real = 0.010 * torch.cos(index * 0.017) - 0.003 * torch.sin(index * 0.009)
    imag = 0.007 * torch.sin(index * 0.015) + 0.002 * torch.cos(index * 0.003)
    return torch.complex(real, imag)


def _assert_array_close(
    actual,
    expected: torch.Tensor,
    *,
    atol: float,
    operation: str,
) -> None:
    actual_tensor = torch.as_tensor(actual).resolve_conj().resolve_neg()
    expected_tensor = torch.as_tensor(expected).resolve_conj().resolve_neg()
    expected_tensor = expected_tensor.to(actual_tensor.device)
    if actual_tensor.shape != expected_tensor.shape:
        raise AssertionError(
            f"{operation} returned shape {actual_tensor.shape}; "
            f"expected {expected_tensor.shape}"
        )
    max_error = float(torch.max(torch.abs(actual_tensor - expected_tensor)))
    assert max_error < atol, (
        f"{operation} max absolute error {max_error:.3e} exceeds {atol:.3e}"
    )


def _assert_decrypts_to(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    expected: torch.Tensor,
    *,
    atol: float,
    operation: str,
    secret_key=None,
) -> None:
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    decoded = engine.decrypt_message(ciphertext, secret_key=secret_key)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    _assert_array_close(decoded, expected, atol=atol, operation=operation)


def _assert_decryption_error_distribution(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    expected: torch.Tensor,
    *,
    p99_noise_factor: float,
    operation: str,
) -> None:
    """Check RMS and 99th-percentile CKKS error against ``N / scale``.

    The Gaussian error distribution is unbounded, so a full-slot maximum is
    not a stable numerical specification. Exact CPU/CUDA residue comparison in
    the caller detects isolated native corruption independently.
    """

    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    actual = torch.as_tensor(engine.decrypt_message(ciphertext))
    expected = expected.to(actual.device)
    error = torch.abs(actual - expected)
    assert torch.all(torch.isfinite(error)), (
        f"{operation} returned nonfinite data"
    )

    noise_unit = engine.config.N / engine.config.default_scale
    p99_error = float(torch.quantile(error, 0.99))
    rms_error = float(torch.sqrt(torch.mean(error.square())))
    assert rms_error < 6.0 * noise_unit, (
        f"{operation} RMS absolute error {rms_error:.3e} exceeds "
        f"{6.0 * noise_unit:.3e}"
    )
    assert p99_error < p99_noise_factor * noise_unit, (
        f"{operation} 99th-percentile absolute error {p99_error:.3e} exceeds "
        f"{p99_noise_factor * noise_unit:.3e}"
    )


@pytest.fixture(scope="module")
def cpu_engine() -> CkksEngine:
    return CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cpu",
    )


def test_cpu_complete_ckks_engine_surface(cpu_engine: CkksEngine) -> None:
    """Exercise every Example 1 dependency and the remaining key switches."""

    engine = cpu_engine
    message = _message(engine)
    other = _other_message(engine)
    source = engine.encrypt_message(message)
    rhs = engine.encrypt_message(other)

    _assert_decrypts_to(
        engine,
        engine.add(source, rhs),
        message + other,
        atol=_ARITHMETIC_ATOL,
        operation="CPU add",
    )

    triplet = engine.multiply(
        engine.coefficient_domain_to_ntt_domain(source),
        engine.coefficient_domain_to_ntt_domain(rhs),
    )
    product = engine.rescale_to_next_level(engine.relinearize(triplet))
    _assert_decrypts_to(
        engine,
        product,
        message * other,
        atol=_MULTIPLICATION_ATOL,
        operation="CPU multiply/relinearize/rescale",
    )

    rotated = engine.rotate_with_key(source, engine.rotation_key(1))
    _assert_decrypts_to(
        engine,
        rotated,
        torch.roll(message, shifts=1),
        atol=_KEYSWITCH_ATOL,
        operation="CPU rotate",
    )

    prepared = engine.prepare_plaintext_for_addition(
        engine.encode(other, level=source.level)
    )
    _assert_decrypts_to(
        engine,
        engine.add_plaintext(source, prepared),
        message + other,
        atol=_ARITHMETIC_ATOL,
        operation="CPU plaintext addition",
    )

    source_secret = engine.create_secret_key()
    destination_secret = engine.create_secret_key()
    source_public = engine.create_public_key(source_secret)
    switch_key = engine.create_key_switch_key(source_secret, destination_secret)
    switched = engine.switch_key(
        engine.encrypt_message(message, source_public), switch_key
    )
    _assert_decrypts_to(
        engine,
        switched,
        message,
        atol=_KEYSWITCH_ATOL,
        operation="CPU source-to-destination key switch",
        secret_key=destination_secret,
    )

    conjugation_key = engine.create_conjugation_key(engine.secret_key)
    _assert_decrypts_to(
        engine,
        engine.conjugate(source, conjugation_key),
        torch.conj(message),
        atol=_KEYSWITCH_ATOL,
        operation="CPU conjugation",
    )


def _assert_same_rns_value_modulo_primes(
    cpu_data: torch.Tensor,
    cuda_data: torch.Tensor,
    *,
    prime_ids: tuple[int, ...],
    moduli: tuple[int, ...],
) -> None:
    row_moduli = torch.tensor(
        [moduli[prime_id] for prime_id in prime_ids],
        dtype=cpu_data.dtype,
    )
    shape = [1] * cpu_data.ndim
    shape[-2] = len(prime_ids)
    difference = torch.remainder(
        cpu_data - cuda_data.cpu(), row_moduli.view(shape)
    )
    assert torch.count_nonzero(difference).item() == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("preset", "cuda_ntt_backend"),
    [
        (Preset.slots8192_scale25_levels14_int32, "radix2_indexed"),
        (Preset.slots16384_scale25_levels29_int32, "radix2_indexed"),
        (Preset.slots32768_scale25_levels24_int32, None),
        (Preset.slots65536_scale25_levels14_int32, "radix2_indexed"),
        (Preset.slots8192_scale40_levels7_int64, "radix2_indexed"),
        (Preset.slots16384_scale40_levels16_int64, None),
        (Preset.slots32768_scale40_levels34_int64, "radix2_indexed"),
    ],
    ids=[
        "indexed-int32-p1",
        "indexed-int32-p2",
        "default-compact-int32-p4",
        "indexed-int32-p6",
        "indexed-int64-p1",
        "default-compact-int64-p2",
        "indexed-int64-p4",
    ],
)
def test_seeded_cpu_cuda_operation_matrix_is_correct_at_multiple_levels(
    preset: Preset,
    cuda_ntt_backend: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    config = CkksConfig.parse(preset)

    cpu = CkksEngine(
        config,
        device="cpu",
        rng_seed=0x12345678,
        rng_nonce=7,
    )
    cuda = CkksEngine(
        config,
        device="cuda:0",
        ntt_backend=cuda_ntt_backend,
        rng_seed=0x12345678,
        rng_nonce=7,
    )

    def assert_value(cpu_value, cuda_value) -> None:
        assert cpu_value.prime_ids == cuda_value.prime_ids
        _assert_same_rns_value_modulo_primes(
            cpu_value.data,
            cuda_value.data,
            prime_ids=cpu_value.prime_ids,
            moduli=tuple(config.moduli),
        )

    message = _message(cpu)
    cpu_rotation_key = cpu.rotation_key(1)
    cuda_rotation_key = cuda.rotation_key(1)
    assert_value(cpu_rotation_key, cuda_rotation_key)

    levels = sorted(
        {
            0,
            (config.num_scale_primes - 1) // 2,
            config.num_scale_primes - 1,
        }
    )
    with pytest.raises(ValueError, match="level must satisfy"):
        cpu.encrypt_message(message, level=config.num_scale_primes)
    with pytest.raises(ValueError, match="level must satisfy"):
        cuda.encrypt_message(message, level=config.num_scale_primes)
    for level in levels:
        cpu_ciphertext = cpu.encrypt_message(message, level=level)
        cuda_ciphertext = cuda.encrypt_message(message, level=level)
        assert_value(cpu_ciphertext, cuda_ciphertext)
        _assert_decryption_error_distribution(
            cuda,
            cuda_ciphertext,
            message,
            p99_noise_factor=12.0,
            operation=f"{preset.value} encrypt at level {level}",
        )

        cpu_ntt = cpu.coefficient_domain_to_ntt_domain(cpu_ciphertext)
        cuda_ntt = cuda.coefficient_domain_to_ntt_domain(cuda_ciphertext)
        assert_value(cpu_ntt, cuda_ntt)

        cpu_triplet = cpu.multiply(cpu_ntt, cpu_ntt)
        cuda_triplet = cuda.multiply(cuda_ntt, cuda_ntt)
        assert_value(cpu_triplet, cuda_triplet)

        cpu_relinearized = cpu.relinearize(cpu_triplet)
        cuda_relinearized = cuda.relinearize(cuda_triplet)
        assert_value(cpu_relinearized, cuda_relinearized)
        _assert_decryption_error_distribution(
            cuda,
            cuda_relinearized,
            message * message,
            p99_noise_factor=0.5,
            operation=f"{preset.value} relinearize at level {level}",
        )

        cpu_rotated = cpu.rotate_with_key(cpu_ciphertext, cpu_rotation_key)
        cuda_rotated = cuda.rotate_with_key(cuda_ciphertext, cuda_rotation_key)
        assert_value(cpu_rotated, cuda_rotated)
        _assert_decryption_error_distribution(
            cuda,
            cuda_rotated,
            torch.roll(message, shifts=1),
            p99_noise_factor=12.0,
            operation=f"{preset.value} rotate at level {level}",
        )

        if level + 1 < config.num_scale_primes:
            cpu_rescaled = cpu.rescale_to_next_level(cpu_relinearized)
            cuda_rescaled = cuda.rescale_to_next_level(cuda_relinearized)
            assert_value(cpu_rescaled, cuda_rescaled)
            _assert_decryption_error_distribution(
                cuda,
                cuda_rescaled,
                message * message,
                p99_noise_factor=1.25,
                operation=f"{preset.value} rescale at level {level}",
            )

    del cpu, cuda
    gc.collect()
    torch.cuda.empty_cache()


def _mixed_radix_integer_reference(
    source: torch.Tensor,
    moduli: tuple[int, ...],
) -> torch.Tensor:
    source_cpu = source.cpu()
    result = torch.empty_like(source_cpu)
    for coefficient in range(source_cpu.size(-1)):
        prefix = 1
        partial = 0
        for row, modulus in enumerate(moduli):
            residue = int(source_cpu[row, coefficient])
            digit = (residue - partial) * pow(prefix, -1, modulus) % modulus
            result[row, coefficient] = digit
            partial += digit * prefix
            prefix *= modulus
    return result


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("preset", "digit_rows"),
    [
        (Preset.slots8192_scale25_levels14_int32, 2),
        (Preset.slots8192_scale25_levels14_int32, 4),
        (Preset.slots8192_scale25_levels14_int32, 6),
        (Preset.slots8192_scale40_levels7_int64, 2),
        (Preset.slots8192_scale40_levels7_int64, 4),
        (Preset.slots8192_scale40_levels7_int64, 6),
    ],
    ids=[
        "int32-rows2",
        "int32-rows4",
        "int32-rows6",
        "int64-rows2",
        "int64-rows4",
        "int64-rows6",
    ],
)
def test_cuda_mixed_radix_decomposition_matches_integer_reference(
    preset: Preset,
    digit_rows: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    config = CkksConfig.parse(
        preset,
        num_scale_primes=max(digit_rows, 2),
        num_p_primes=digit_rows,
        enforce_security_budget=False,
    )
    cpu = CkksEngine(config, device="cpu", allow_sk_gen=False)
    cuda = CkksEngine(config, device="cuda:0", allow_sk_gen=False)
    cpu_switcher = cpu._hybrid_key_switcher
    cuda_switcher = cuda._hybrid_key_switcher
    cpu_spec = cpu_switcher.rns_layout.digit_specs(0)[0]
    cuda_spec = cuda_switcher.rns_layout.digit_specs(0)[0]
    assert cpu_spec.prime_ids == cuda_spec.prime_ids
    assert len(cpu_spec.prime_ids) == digit_rows

    moduli = tuple(config.moduli[index] for index in cpu_spec.prime_ids)
    coefficient_count = 19
    source = torch.empty(
        digit_rows,
        coefficient_count,
        dtype=config.torch_dtype,
    )
    for row, modulus in enumerate(moduli):
        for coefficient in range(coefficient_count):
            value = (
                (coefficient + 3) * (row + 5) * 1_000_003
                + (digit_rows - row) * 97
            ) % modulus
            source[row, coefficient] = (
                modulus - 1 - value if row % 2 == 0 else value
            )
            if coefficient % 5 == 0:
                source[row, coefficient] += modulus
    source_before = source.clone()

    cpu_actual = cpu_switcher._decompose_digit_mixed_radix(source, cpu_spec)
    cuda_source = source.to("cuda:0")
    cuda_before = cuda_source.clone()
    cuda_actual = cuda_switcher._decompose_digit_mixed_radix(
        cuda_source, cuda_spec
    )
    expected = _mixed_radix_integer_reference(source, moduli)

    torch.testing.assert_close(cpu_actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cuda_actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source, source_before, rtol=0, atol=0)
    torch.testing.assert_close(cuda_source, cuda_before, rtol=0, atol=0)


@pytest.mark.gpu
def test_cuda_rns_parameters_are_isolated_between_engines(
    engine: CkksEngine,
) -> None:
    """Contexts with different rings and RNS bases remain independent."""

    other_engine = CkksEngine(
        Preset.slots16384_scale40_levels16_int64,
        device="cuda:0",
    )
    try:
        message = _message(engine)
        ciphertext = engine.encrypt_message(message)
        _assert_decrypts_to(
            engine,
            ciphertext,
            message,
            atol=_ARITHMETIC_ATOL,
            operation="first engine after second-context initialization",
        )
    finally:
        del other_engine
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("level", [0, 3, 6])
def test_codec_encrypt_decrypt_and_convenience_paths_are_correct(
    engine: CkksEngine,
    level: int,
) -> None:
    """Cover encode/decode, lazy plaintext, encrypt/decrypt and encrypt_message."""

    message = _message(engine)

    eager = engine.encode(message, level=level)
    assert eager.is_integer_coefficients and eager.level == level
    _assert_array_close(
        engine.decode(eager),
        message,
        atol=_CODEC_ATOL,
        operation=f"encode/decode at level {level}",
    )

    lazy = engine.plaintext(message, level=level)
    assert lazy.is_slots
    encrypted = engine.encrypt(lazy, engine.public_key)
    # Consuming operations materialize the required encoded state without
    # changing the caller-owned slots Plaintext.
    assert lazy.is_slots
    _assert_decrypts_to(
        engine,
        encrypted,
        message,
        atol=_CODEC_ATOL,
        operation=f"plaintext/encrypt/decrypt_message at level {level}",
    )

    decrypted_plaintext = engine.decrypt(encrypted, engine.secret_key)
    assert isinstance(decrypted_plaintext, Plaintext)
    assert decrypted_plaintext.is_approximate_coefficients
    assert decrypted_plaintext.data is not None
    assert decrypted_plaintext.data.dtype == torch.float64
    _assert_array_close(
        engine.decode(decrypted_plaintext),
        message,
        atol=_CODEC_ATOL,
        operation=f"decrypt/decode at level {level}",
    )

    convenience = engine.encrypt_message(
        message, engine.public_key, level=level
    )
    _assert_decrypts_to(
        engine,
        convenience,
        message,
        atol=_CODEC_ATOL,
        operation=f"encrypt_message/decrypt_message at level {level}",
    )


@pytest.mark.gpu
def test_int64_coefficients_wider_than_scale_prime_use_exact_rns_lift() -> None:
    """An int64 coefficient may still exceed a 40-bit active modulus."""

    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
    )
    message = torch.full((32,), 2.0e8, dtype=torch.float64)

    convenience = engine.decrypt_message(
        engine.encrypt_message(message),
        is_real=True,
    )[: message.numel()]
    plaintext = engine.encode(message)
    composed = engine.decode(
        engine.decrypt(engine.encrypt(plaintext)),
        is_real=True,
    )[: message.numel()]
    operation_ready = engine.prepare_plaintext_for_addition(
        engine.encode(message)
    )
    operation_ready_data = operation_ready.data
    plaintext_data = plaintext.data
    assert operation_ready_data is not None
    assert plaintext_data is not None
    standard_rns = operation_ready_data.clone()
    engine.rns_runtime.from_montgomery_(standard_rns)
    active_moduli = torch.tensor(
        engine.config.q_moduli,
        dtype=plaintext_data.dtype,
        device=plaintext_data.device,
    ).view(-1, 1)
    expected_rns = torch.remainder(
        plaintext_data.unsqueeze(-2),
        active_moduli,
    )
    # The two independent stochastic-rounding calls can differ by one integer;
    # a broken one-prime lift differs by an entire active modulus instead.
    torch.testing.assert_close(standard_rns, expected_rns, rtol=0, atol=1)

    _assert_array_close(
        convenience,
        message,
        atol=5e-6,
        operation="40-bit exact-RNS convenience roundtrip",
    )
    _assert_array_close(
        composed,
        message,
        atol=5e-6,
        operation="40-bit exact-RNS composed roundtrip",
    )


@pytest.mark.gpu
def test_message_encryption_rejects_decoder_range_aliasing() -> None:
    engine = CkksEngine(Preset.slots8192_scale40_levels7_int64, device="cuda:0")
    message = torch.full((32,), 1.0e20, dtype=torch.float64)

    with pytest.raises(OverflowError, match="direct decoder range"):
        engine.encrypt_message(message)


def test_ntt_and_coefficient_state_round_trips_are_correct(
    engine: CkksEngine,
) -> None:
    """Cover functional and in-place coefficient/NTT state transitions."""

    message = _message(engine)
    source = engine.encrypt_message(message)
    malformed_basis = source.clone()
    malformed_basis.modulus_basis = "bad"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="modulus_basis"):
        engine.coefficient_domain_to_ntt_domain_(malformed_basis)

    ntt = engine.coefficient_domain_to_ntt_domain(source)
    assert ntt is not source
    assert (
        source.is_coefficient_domain
        and source.residue_representation == "standard"
    )
    assert ntt.is_ntt_domain and ntt.residue_representation == "montgomery"
    malformed_ntt = ntt.clone()
    malformed_ntt.residue_representation = "standard"
    with pytest.raises(ValueError, match="Montgomery"):
        engine.ntt_domain_to_coefficient_domain_(malformed_ntt)
    round_trip = engine.ntt_domain_to_coefficient_domain(ntt)
    assert round_trip.limb_count == source.limb_count
    assert round_trip.prime_ids == source.prime_ids
    assert round_trip.modulus_basis == source.modulus_basis
    assert round_trip.level == source.level
    assert round_trip.scale == source.scale
    _assert_decrypts_to(
        engine,
        round_trip,
        message,
        atol=_ARITHMETIC_ATOL,
        operation="coefficient_domain_to_ntt_domain/ntt_domain_to_coefficient_domain",
    )

    inplace = source.clone()
    assert engine.coefficient_domain_to_ntt_domain_(inplace) is inplace
    assert (
        inplace.is_ntt_domain and inplace.residue_representation == "montgomery"
    )
    assert engine.ntt_domain_to_coefficient_domain_(inplace) is inplace
    assert (
        inplace.is_coefficient_domain
        and inplace.residue_representation == "standard"
    )
    _assert_decrypts_to(
        engine,
        inplace,
        message,
        atol=_ARITHMETIC_ATOL,
        operation="coefficient_domain_to_ntt_domain_/ntt_domain_to_coefficient_domain_",
    )


def test_ciphertext_linear_arithmetic_and_encrypted_zero_are_correct(
    engine: CkksEngine,
) -> None:
    """Cover functional/in-place linear arithmetic and encrypted zero."""

    left_message = _message(engine)
    right_message = _other_message(engine)
    left = engine.encrypt_message(left_message)
    right = engine.encrypt_message(right_message)

    _assert_decrypts_to(
        engine,
        engine.add(left, right),
        left_message + right_message,
        atol=_ARITHMETIC_ATOL,
        operation="add",
    )
    _assert_decrypts_to(
        engine,
        engine.subtract(left, right),
        left_message - right_message,
        atol=_ARITHMETIC_ATOL,
        operation="subtract",
    )

    batched_left = engine.encrypt_message(torch.stack((left_message,) * 3))
    batched_right = engine.encrypt_message(torch.stack((right_message,) * 3))
    strided_right = batched_right.with_data(
        batched_right.data.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    )
    assert not strided_right.data.is_contiguous()
    _assert_decrypts_to(
        engine,
        engine.add(batched_left, strided_right),
        torch.stack((left_message + right_message,) * 3),
        atol=_ARITHMETIC_ATOL,
        operation="add with a component-collapsible strided view",
    )

    added_inplace = left.clone()
    assert engine.add_(added_inplace, right) is added_inplace
    _assert_decrypts_to(
        engine,
        added_inplace,
        left_message + right_message,
        atol=_ARITHMETIC_ATOL,
        operation="add_",
    )

    subtracted_inplace = left.clone()
    assert engine.subtract_(subtracted_inplace, right) is subtracted_inplace
    _assert_decrypts_to(
        engine,
        subtracted_inplace,
        left_message - right_message,
        atol=_ARITHMETIC_ATOL,
        operation="subtract_",
    )
    _assert_decrypts_to(
        engine,
        engine.sum_ciphertexts([left, right, left]),
        2 * left_message + right_message,
        atol=_ARITHMETIC_ATOL,
        operation="sum_ciphertexts",
    )
    negated = engine.negate(left)
    _assert_decrypts_to(
        engine,
        negated,
        -left_message,
        atol=_ARITHMETIC_ATOL,
        operation="negate",
    )

    inplace = left.clone()
    assert engine.negate_(inplace) is inplace
    for candidate in (negated, inplace):
        for row, prime_id in enumerate(candidate.prime_ids):
            modulus = engine.montgomery_parameters.moduli[prime_id]
            assert torch.all(candidate.data[..., row, :] >= 0)
            assert torch.all(candidate.data[..., row, :] < modulus)
    _assert_decrypts_to(
        engine,
        inplace,
        -left_message,
        atol=_ARITHMETIC_ATOL,
        operation="negate_",
    )

    encrypted_zero = engine.encrypt_zero_like(left, engine.public_key)
    _assert_decrypts_to(
        engine,
        encrypted_zero,
        torch.zeros_like(left_message),
        atol=_CODEC_ATOL,
        operation="encrypt_zero_like",
    )


def test_plaintext_arithmetic_and_plaintext_zero_are_correct(
    engine: CkksEngine,
) -> None:
    """Cover add_plaintext/multiply_plaintext, their in-place forms, and constructed plaintext zero."""

    message = _message(engine)
    addend_message = torch.full((engine.num_slots,), 0.125, dtype=torch.float64)
    factor_message = torch.full((engine.num_slots,), 1.5, dtype=torch.float64)
    source = engine.encrypt_message(message)
    addend = engine.prepare_plaintext_for_addition(
        engine.encode(addend_message, level=source.level)
    )
    factor = engine.prepare_plaintext_for_multiplication(
        engine.encode(factor_message, level=source.level)
    )

    added = engine.add_plaintext(source, addend)
    _assert_decrypts_to(
        engine,
        added,
        message + addend_message,
        atol=_ARITHMETIC_ATOL,
        operation="add_plaintext",
    )

    added_inplace = source.clone()
    component_storage = (
        added_inplace.c0.data_ptr(),
        added_inplace.c1.data_ptr(),
    )
    assert engine.add_plaintext_(added_inplace, addend) is added_inplace
    assert (
        added_inplace.c0.data_ptr(),
        added_inplace.c1.data_ptr(),
    ) == component_storage
    _assert_decrypts_to(
        engine,
        added_inplace,
        message + addend_message,
        atol=_ARITHMETIC_ATOL,
        operation="add_plaintext_",
    )

    source_ntt = engine.coefficient_domain_to_ntt_domain(source)
    multiplied = engine.multiply_plaintext(source_ntt, factor)
    assert multiplied.level == source.level
    assert multiplied.polynomial_domain == "ntt"
    assert multiplied.residue_representation == "montgomery"
    multiplied = engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(multiplied)
    )
    _assert_decrypts_to(
        engine,
        multiplied,
        message * factor_message,
        atol=_MULTIPLICATION_ATOL,
        operation="multiply_plaintext followed by explicit rescale",
    )

    multiplied_inplace = source_ntt.clone()
    assert (
        engine.multiply_plaintext_(multiplied_inplace, factor)
        is multiplied_inplace
    )
    assert multiplied_inplace.level == source.level
    engine.ntt_domain_to_coefficient_domain_(multiplied_inplace)
    engine.rescale_to_next_level_(multiplied_inplace)
    _assert_decrypts_to(
        engine,
        multiplied_inplace,
        message * factor_message,
        atol=_MULTIPLICATION_ATOL,
        operation="multiply_plaintext_ followed by explicit rescale_to_next_level_",
    )

    with pytest.raises(ValueError, match="expected 'ntt'"):
        engine.multiply_plaintext(source, factor)

    encoded_zero = engine.zero_plaintext_like(addend)
    assert encoded_zero.representation == addend.representation
    assert encoded_zero.polynomial_domain == addend.polynomial_domain
    assert encoded_zero.modulus_basis == addend.modulus_basis
    _assert_decrypts_to(
        engine,
        engine.add_plaintext(source, encoded_zero),
        message,
        atol=_ARITHMETIC_ATOL,
        operation="zero_plaintext_like(RNS coefficient state)",
    )
    lazy_zero = engine.zero_plaintext_like(
        engine.plaintext(addend_message, level=source.level)
    )
    assert lazy_zero.is_slots
    _assert_array_close(
        engine.decode(lazy_zero),
        torch.zeros_like(addend_message),
        atol=_CODEC_ATOL,
        operation="zero_plaintext_like(lazy)",
    )


def test_multiply_and_relinearize_pipeline_is_correct(
    engine: CkksEngine,
) -> None:
    """Check the direct-CKKS multiply, relinearize, rescale pipeline.

    Fresh operands enter multiplication at the ordinary CKKS scale.  Their
    product carries the pending square scale and is rescaled only after
    relinearization.
    """

    left_message = _message(engine)
    right_message = _other_message(engine)
    expected = left_message * right_message

    left = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(left_message)
    )
    right = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(right_message)
    )
    triplet = engine.multiply(left, right)
    assert triplet.component_count == 3
    assert (
        triplet.is_ntt_domain and triplet.residue_representation == "montgomery"
    )

    relinearized = engine.relinearize(triplet, engine.relinearization_key)
    assert relinearized.component_count == 2
    assert (
        relinearized.is_coefficient_domain
        and relinearized.residue_representation == "standard"
    )
    rescaled = engine.rescale_to_next_level(relinearized)
    _assert_decrypts_to(
        engine,
        rescaled,
        expected,
        atol=_MULTIPLICATION_ATOL,
        operation="multiply/relinearize/rescale",
    )


@pytest.mark.gpu
def test_final_legal_level_remains_decryptable() -> None:
    """A short chain must support decryption at its last legal level."""

    engine = CkksEngine(
        CkksConfig.parse(
            Preset.slots8192_scale40_levels7_int64, num_scale_primes=3
        ),
        device="cuda:0",
    )
    try:
        message = torch.linspace(
            -0.01,
            0.01,
            engine.num_slots,
            dtype=torch.float64,
        )
        first = engine.coefficient_domain_to_ntt_domain(
            engine.encrypt_message(message, level=0)
        )
        second = engine.coefficient_domain_to_ntt_domain(
            engine.encrypt_message(message, level=0)
        )
        product = engine.rescale_to_next_level(
            engine.relinearize(engine.multiply(first, second))
        )
        product = engine.coefficient_domain_to_ntt_domain(product)
        third = engine.coefficient_domain_to_ntt_domain(
            engine.encrypt_message(message, level=1)
        )
        final_product = engine.rescale_to_next_level(
            engine.relinearize(engine.multiply(product, third))
        )

        assert final_product.level == engine.final_public_level
        _assert_decrypts_to(
            engine,
            final_product,
            message**3,
            atol=_MULTIPLICATION_ATOL,
            operation="decryption at final legal level",
        )
        with pytest.raises(MaximumLevelError):
            engine.rescale_to_next_level(final_product)
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def test_explicit_key_switch_changes_key_without_changing_message(
    engine: CkksEngine,
) -> None:
    """Exercise decomposition, ModUp, key-digit accumulation, and ModDown."""

    message = _message(engine)
    source_secret_key = engine.create_secret_key()
    destination_secret_key = engine.create_secret_key()
    source_public_key = engine.create_public_key(source_secret_key)
    switching_key = engine.create_key_switch_key(
        source_secret_key,
        destination_secret_key,
    )
    key_data_ref = weakref.ref(switching_key.data)

    for level in (0, 3):
        source = engine.encrypt_message(
            message,
            source_public_key,
            level=level,
        )
        switched = engine.switch_key(source, switching_key)
        _assert_decrypts_to(
            engine,
            switched,
            message,
            atol=_KEYSWITCH_ATOL,
            operation=f"switch_key at level {level}",
            secret_key=destination_secret_key,
        )

        with pytest.raises(
            ValueError, match="Invalid Ciphertext polynomial_domain"
        ):
            engine.switch_key(
                engine.coefficient_domain_to_ntt_domain(source), switching_key
            )

    # Applying a provided key must not leave hidden engine-owned views that
    # retain its GPU storage after the caller releases the key.
    del switching_key
    gc.collect()
    assert key_data_ref() is None


@pytest.mark.gpu
def test_explicit_key_installation_drives_correct_operations() -> None:
    """Installed public, relinearization, and rotation keys are operational."""

    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    try:
        with pytest.raises(RuntimeError, match="generation is disabled"):
            _ = engine.secret_key

        secret_key = engine.create_secret_key()
        public_key = engine.create_public_key(secret_key)
        relinearization_key = engine.create_relinearization_key(secret_key)
        rotation_key = engine.create_rotation_key(1, secret_key)

        engine.set_secret_key(secret_key)
        engine.set_public_key(public_key)
        engine.set_relinearization_key(relinearization_key)
        engine.set_rotation_key(rotation_key)

        assert engine.secret_key is secret_key
        assert engine.public_key is public_key
        assert engine.relinearization_key is relinearization_key
        assert engine.rotation_keys[1] is rotation_key
        assert engine.rotation_key(1) is rotation_key

        message = _message(engine, complex_values=False)
        source = engine.encrypt_message(message)
        _assert_decrypts_to(
            engine,
            source,
            message,
            atol=_CODEC_ATOL,
            operation="installed public/secret keys",
        )

        prepared = engine.coefficient_domain_to_ntt_domain(source)
        squared = engine.rescale_to_next_level(
            engine.relinearize(engine.multiply(prepared, prepared))
        )
        _assert_decrypts_to(
            engine,
            squared,
            message.square(),
            atol=_MULTIPLICATION_ATOL,
            operation="installed relinearization key",
        )

        rotated = engine.rotate_by_step(source, 1)
        assert isinstance(rotated, Ciphertext)
        _assert_decrypts_to(
            engine,
            rotated,
            torch.roll(message, shifts=1),
            atol=_KEYSWITCH_ATOL,
            operation="installed rotation key",
        )
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def test_rotation_variants_and_hoisting_are_correct(
    engine: CkksEngine,
) -> None:
    """Cover step-owned, key-owned, grouped, and in-place rotation APIs."""

    message = _message(engine)
    source = engine.encrypt_message(message)
    rotation_steps = [1, -1]
    for rotation_step in rotation_steps:
        engine.rotation_key(rotation_step)

    single = engine.rotate_with_key(source, engine.rotation_key(1))
    _assert_decrypts_to(
        engine,
        single,
        torch.roll(message, shifts=1),
        atol=_KEYSWITCH_ATOL,
        operation="rotate_with_key",
    )

    wrapped = engine.rotate_by_step(source, -1)
    assert isinstance(wrapped, Ciphertext)
    _assert_decrypts_to(
        engine,
        wrapped,
        torch.roll(message, shifts=-1),
        atol=_KEYSWITCH_ATOL,
        operation="rotate_by_step(-1)",
    )

    rotation_step_result = engine.rotate_by_step(source, 1)
    assert isinstance(rotation_step_result, Ciphertext)
    _assert_decrypts_to(
        engine,
        rotation_step_result,
        torch.roll(message, shifts=1),
        atol=_KEYSWITCH_ATOL,
        operation="rotate_by_step(+1)",
    )

    requested = [0, 1, -1, 1]
    expected = [
        torch.roll(message, shifts=rotation_step) for rotation_step in requested
    ]
    for use_hoisting in (False, True):
        grouped = engine.rotate_many_by_steps(
            source,
            requested,
            use_hoisting=use_hoisting,
        )
        for index, (ciphertext, reference) in enumerate(
            zip(grouped, expected, strict=True)
        ):
            _assert_decrypts_to(
                engine,
                ciphertext,
                reference,
                atol=_KEYSWITCH_ATOL,
                operation=(
                    "rotate_many_by_steps"
                    f"[{index}] (use_hoisting={use_hoisting})"
                ),
            )

    direct_keys = [engine.rotation_key(step) for step in (1, -1, 1)]
    keyed_result = engine.rotate_many_with_keys(source, direct_keys)
    for index, (ciphertext, reference) in enumerate(
        zip(keyed_result, (expected[1], expected[2], expected[3]), strict=True)
    ):
        _assert_decrypts_to(
            engine,
            ciphertext,
            reference,
            atol=_KEYSWITCH_ATOL,
            operation=f"rotate_many_with_keys[{index}]",
        )
    assert keyed_result[0] is not keyed_result[2]
    assert keyed_result[0].data.data_ptr() != keyed_result[2].data.data_ptr()

    step_three_key = engine.rotation_key(3)
    previous_rotation_keys = engine._rotation_keys
    previous_allow_sk_gen = engine.allow_sk_gen
    engine._rotation_keys = RotationKeySet({3: step_three_key})
    engine.allow_sk_gen = False
    try:
        mixed = engine.rotate_many_by_steps(source, [3, 6, 3, 0])
    finally:
        engine.allow_sk_gen = previous_allow_sk_gen
        engine._rotation_keys = previous_rotation_keys
    for index, (ciphertext, rotation_step) in enumerate(
        zip(mixed, (3, 6, 3, 0), strict=True)
    ):
        _assert_decrypts_to(
            engine,
            ciphertext,
            torch.roll(message, shifts=rotation_step),
            atol=_KEYSWITCH_ATOL,
            operation=f"mixed direct/composed rotation[{index}]",
        )

    inplace = source.clone()
    assert engine.rotate_by_step_(inplace, 1) is inplace
    _assert_decrypts_to(
        engine,
        inplace,
        torch.roll(message, shifts=1),
        atol=_KEYSWITCH_ATOL,
        operation="rotate_by_step_",
    )


def test_conjugation_is_correct(engine: CkksEngine) -> None:
    message = _message(engine)
    source = engine.encrypt_message(message)
    key = engine.create_conjugation_key(engine.secret_key)

    conjugated = engine.conjugate(source, key)
    _assert_decrypts_to(
        engine,
        conjugated,
        torch.conj(message),
        atol=_KEYSWITCH_ATOL,
        operation="conjugate",
    )
