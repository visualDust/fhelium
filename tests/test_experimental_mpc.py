"""CPU/CUDA correctness tests for experimental multiparty CKKS arithmetic."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch

import fhelium as fh
from fhelium.core import PublicKey, SecretKey
from fhelium.experimental import mpc
from fhelium.experimental.mpc import _ops as mpc_ops

# These match the established end-to-end correctness tolerances in
# tests/test_ckks_operation_correctness.py; they are not MPC security bounds.
_CODEC_ATOL = 1e-5
_KEYSWITCH_ATOL = 2e-5
_MULTIPLICATION_ATOL = 3e-5


@dataclass(frozen=True)
class _Collective:
    secret_shares: tuple[SecretKey, ...]
    aggregate_secret: SecretKey
    public_key: PublicKey


@pytest.fixture(
    scope="module",
    params=(
        pytest.param("cpu", id="cpu"),
        pytest.param("cuda:0", marks=pytest.mark.gpu, id="cuda"),
    ),
)
def engine(request: pytest.FixtureRequest) -> Iterator[fh.CkksEngine]:
    device = str(request.param)
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    instance = fh.CkksEngine(
        fh.Preset.slots8192_scale40_levels7_int64, device=device
    )
    yield instance
    del instance
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def collective(engine: fh.CkksEngine) -> _Collective:
    secret_shares = tuple(mpc.sample_secret_share(engine) for _ in range(2))
    aggregate_data = engine.rns_runtime.add_lazy(
        secret_shares[0].data,
        secret_shares[1].data,
        include_p=True,
    )
    aggregate_secret = SecretKey(
        data=aggregate_data,
        context_id=engine.context.context_id,
        prime_ids=engine.rns_layout.prime_ids(0, include_p=True),
        polynomial_domain="ntt",
        modulus_basis="QP",
        residue_representation="montgomery",
    )
    common_a = mpc.sample_common_uniform(engine, basis="Q")
    public_key = mpc.aggregate_ckg(
        engine,
        [mpc.ckg_share(engine, share, common_a) for share in secret_shares],
        common_a,
    )
    return _Collective(secret_shares, aggregate_secret, public_key)


def _real_message(engine: fh.CkksEngine) -> torch.Tensor:
    index = torch.arange(engine.num_slots, dtype=torch.float64)
    return 0.012 * torch.sin(index * 0.013) + 0.006 * torch.cos(index * 0.007)


def _complex_message(engine: fh.CkksEngine) -> torch.Tensor:
    index = torch.arange(engine.num_slots, dtype=torch.float64)
    real = 0.010 * torch.cos(index * 0.017) - 0.003 * torch.sin(index * 0.009)
    imag = 0.007 * torch.sin(index * 0.015) + 0.002 * torch.cos(index * 0.003)
    return torch.complex(real, imag)


def _assert_close(
    actual,
    expected: torch.Tensor,
    *,
    atol: float,
    operation: str,
) -> None:
    actual_tensor = torch.as_tensor(actual).resolve_conj().resolve_neg()
    expected_tensor = (
        expected.resolve_conj().resolve_neg().to(actual_tensor.device)
    )
    assert actual_tensor.shape == expected_tensor.shape
    max_error = float(torch.max(torch.abs(actual_tensor - expected_tensor)))
    assert max_error < atol, (
        f"{operation} max absolute error {max_error:.3e} exceeds {atol:.3e}"
    )


def _compact_zeros(
    engine: fh.CkksEngine,
    *batch_shape: int,
) -> torch.Tensor:
    return torch.zeros(
        (*batch_shape, engine.config.N),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )


def _compact_ternary_pattern(
    engine: fh.CkksEngine,
    *batch_shape: int,
) -> torch.Tensor:
    base = (
        torch.arange(
            engine.config.N,
            dtype=engine.config.torch_dtype,
            device=engine.device,
        )
        % 3
    ) - 1
    if not batch_shape:
        return base
    count = torch.Size(batch_shape).numel()
    rows = [torch.roll(base, shifts=index) for index in range(count)]
    return torch.stack(rows).reshape(*batch_shape, engine.config.N).contiguous()


@pytest.mark.gpu
def test_common_sampling_ckg_and_core_value_invariants(
    engine: fh.CkksEngine,
    collective: _Collective,
) -> None:
    q_rows = engine.rns_layout.row_count(0)
    qp_rows = engine.rns_layout.row_count(0, include_p=True)
    digit_count = engine.rns_layout.key_digit_count

    common_q = mpc.sample_common_uniform(engine, basis="Q")
    explicit_single_qp = mpc.sample_common_uniform(
        engine,
        basis="QP",
        count=1,
    )
    common_qp = mpc.sample_common_uniform(
        engine,
        basis="QP",
        count=digit_count,
    )
    assert common_q.shape == (q_rows, engine.config.N)
    assert explicit_single_qp.shape == (1, qp_rows, engine.config.N)
    assert common_qp.shape == (digit_count, qp_rows, engine.config.N)
    assert (
        common_q.dtype
        == explicit_single_qp.dtype
        == common_qp.dtype
        == engine.config.torch_dtype
    )
    assert (
        common_q.device
        == explicit_single_qp.device
        == common_qp.device
        == engine.device
    )
    assert (
        common_q.is_contiguous()
        and explicit_single_qp.is_contiguous()
        and common_qp.is_contiguous()
    )

    assert collective.public_key.data.shape == (2, q_rows, engine.config.N)
    assert collective.public_key.prime_ids == engine.rns_layout.prime_ids(0)
    assert collective.public_key.polynomial_domain == "ntt"
    assert collective.public_key.residue_representation == "montgomery"
    assert collective.public_key.modulus_basis == "Q"

    message = _real_message(engine)
    ciphertext = engine.encrypt_message(message, collective.public_key)
    decoded = engine.decrypt_message(
        ciphertext,
        collective.aggregate_secret,
        is_real=True,
    )
    _assert_close(
        decoded,
        message,
        atol=_CODEC_ATOL,
        operation="collective public-key encryption",
    )


@pytest.mark.gpu
def test_two_round_rkg_preserves_separate_families_and_relinearizes(
    engine: fh.CkksEngine,
    collective: _Collective,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digit_count = engine.rns_layout.key_digit_count
    qp_rows = engine.rns_layout.row_count(0, include_p=True)
    common_a = mpc.sample_common_uniform(
        engine,
        basis="QP",
        count=digit_count,
    )
    ephemeral_shares = tuple(
        mpc.sample_secret_share(engine) for _ in collective.secret_shares
    )
    round1_shares = [
        mpc.rkg_round1_share(engine, secret, ephemeral, common_a)
        for secret, ephemeral in zip(
            collective.secret_shares,
            ephemeral_shares,
            strict=True,
        )
    ]
    for family0, family1 in round1_shares:
        assert (
            family0.shape
            == family1.shape
            == (
                digit_count,
                qp_rows,
                engine.config.N,
            )
        )
        assert family0.data_ptr() != family1.data_ptr()

    round1 = mpc.aggregate_rkg_round1(engine, round1_shares)
    expected_round1_0 = engine.rns_runtime.add_lazy(
        round1_shares[0][0],
        round1_shares[1][0],
        include_p=True,
    )
    expected_round1_1 = engine.rns_runtime.add_lazy(
        round1_shares[0][1],
        round1_shares[1][1],
        include_p=True,
    )
    assert torch.equal(round1[0], expected_round1_0)
    assert torch.equal(round1[1], expected_round1_1)

    round2_shares = []
    for party_index, (secret, ephemeral) in enumerate(
        zip(
            collective.secret_shares,
            ephemeral_shares,
            strict=True,
        )
    ):
        error2_coefficients = torch.zeros(
            (digit_count, engine.config.N),
            dtype=engine.config.torch_dtype,
            device=engine.device,
        )
        error3_coefficients = torch.zeros_like(error2_coefficients)
        magnitudes = torch.arange(
            1,
            digit_count + 1,
            dtype=engine.config.torch_dtype,
            device=engine.device,
        )
        error2_coefficients[:, 0] = (party_index + 1) * magnitudes
        error3_coefficients[:, 1] = -(party_index + 2) * magnitudes
        error2 = mpc_ops._lift_coefficients(
            engine,
            error2_coefficients,
            level=0,
            basis="QP",
            to_ntt=True,
        )
        error3 = mpc_ops._lift_coefficients(
            engine,
            error3_coefficients,
            level=0,
            basis="QP",
            to_ntt=True,
        )
        sampled_errors = iter((error2, error3))

        def controlled_error(*_args, **_kwargs) -> torch.Tensor:
            return next(sampled_errors).clone()

        monkeypatch.setattr(
            mpc_ops,
            "_sample_gaussian_rns",
            controlled_error,
        )
        actual = mpc.rkg_round2_share(engine, secret, ephemeral, round1)

        expected0 = engine.rns_runtime.montgomery_mul(
            round1[0],
            secret.data,
            include_p=True,
        )
        expected0 = engine.rns_runtime.add_lazy(
            expected0,
            error2,
            include_p=True,
        )
        ephemeral_minus_secret = engine.rns_runtime.sub_lazy(
            ephemeral.data,
            secret.data,
            include_p=True,
        )
        expected1 = engine.rns_runtime.montgomery_mul(
            round1[1],
            ephemeral_minus_secret,
            include_p=True,
        )
        expected1 = engine.rns_runtime.add_lazy(
            expected1,
            error3,
            include_p=True,
        )
        assert torch.equal(actual[0], expected0)
        assert torch.equal(actual[1], expected1)

        disguised = (
            engine.rns_runtime.add_lazy(
                expected0,
                expected1,
                include_p=True,
            ),
            torch.zeros_like(expected1),
        )
        assert not (
            torch.equal(actual[0], disguised[0])
            and torch.equal(actual[1], disguised[1])
        )
        round2_shares.append(actual)
    key = mpc.aggregate_rkg_round2(engine, round2_shares, round1)
    assert key.data.shape == (digit_count, 2, qp_rows, engine.config.N)
    assert torch.equal(key.data[:, 1], round1[1])

    message = _real_message(engine)
    ciphertext = engine.encrypt_message(message, collective.public_key)
    prepared = engine.coefficient_domain_to_ntt_domain(ciphertext)
    squared = engine.rescale_to_next_level(
        engine.relinearize(engine.multiply(prepared, prepared), key)
    )
    decoded = engine.decrypt_message(
        squared,
        collective.aggregate_secret,
        is_real=True,
    )
    _assert_close(
        decoded,
        message.square(),
        atol=_MULTIPLICATION_ATOL,
        operation="distributed relinearization",
    )


@pytest.mark.gpu
def test_distributed_rotation_and_conjugation_keys_are_consumable(
    engine: fh.CkksEngine,
    collective: _Collective,
) -> None:
    message = _complex_message(engine)
    ciphertext = engine.encrypt_message(message, collective.public_key)
    digit_count = engine.rns_layout.key_digit_count

    for rotation_step in (1, -1):
        common_a = mpc.sample_common_uniform(
            engine,
            basis="QP",
            count=digit_count,
        )
        shares = [
            mpc.rotation_key_share(
                engine,
                secret,
                common_a,
                rotation_step,
            )
            for secret in collective.secret_shares
        ]
        key = mpc.aggregate_rotation_key(
            engine,
            shares,
            common_a,
            rotation_step,
        )
        rotated = engine.rotate_with_key(ciphertext, key)
        decoded = engine.decrypt_message(rotated, collective.aggregate_secret)
        _assert_close(
            decoded,
            torch.roll(message, shifts=rotation_step),
            atol=_KEYSWITCH_ATOL,
            operation=f"distributed rotation {rotation_step}",
        )

    common_a = mpc.sample_common_uniform(
        engine,
        basis="QP",
        count=digit_count,
    )
    shares = [
        mpc.conjugation_key_share(engine, secret, common_a)
        for secret in collective.secret_shares
    ]
    key = mpc.aggregate_conjugation_key(engine, shares, common_a)
    conjugated = engine.conjugate(ciphertext, key)
    decoded = engine.decrypt_message(conjugated, collective.aggregate_secret)
    _assert_close(
        decoded,
        torch.conj(message),
        atol=_KEYSWITCH_ATOL,
        operation="distributed conjugation",
    )


@pytest.mark.gpu
def test_unsafe_collective_decryption_matches_central_phase_at_levels(
    engine: fh.CkksEngine,
    collective: _Collective,
) -> None:
    message = torch.stack(
        (_real_message(engine), -0.75 * _real_message(engine))
    )
    for level in (0, 3, engine.final_public_level):
        ciphertext = engine.encrypt_message(
            message,
            collective.public_key,
            level=level,
        )
        error0 = _compact_zeros(engine, *ciphertext.batch_shape)
        error0[..., 0] = 1
        error1 = -error0
        zero = _compact_zeros(engine, *ciphertext.batch_shape)
        zero_error_share = mpc.unsafe_collective_decryption_share(
            engine,
            ciphertext,
            collective.secret_shares[0],
            smudging_error_coefficients=zero,
        )
        nonzero_error_share = mpc.unsafe_collective_decryption_share(
            engine,
            ciphertext,
            collective.secret_shares[0],
            smudging_error_coefficients=error0,
        )
        observed_error = engine.rns_runtime.sub_canonical(
            nonzero_error_share,
            zero_error_share,
        )
        expected_error = mpc_ops._lift_coefficients(
            engine,
            error0,
            level=level,
            basis="Q",
            to_ntt=False,
        )
        assert torch.equal(observed_error, expected_error)
        shares = [
            nonzero_error_share,
            mpc.unsafe_collective_decryption_share(
                engine,
                ciphertext,
                collective.secret_shares[1],
                smudging_error_coefficients=error1,
            ),
        ]
        plaintext = mpc.unsafe_fuse_collective_decryption(
            engine,
            ciphertext,
            shares,
        )
        centralized = engine.decrypt(ciphertext, collective.aggregate_secret)
        assert plaintext.representation == "approximate_coefficients"
        assert torch.equal(plaintext.data, centralized.data)
        decoded = engine.decode(plaintext, is_real=True)
        _assert_close(
            decoded,
            message,
            atol=_CODEC_ATOL,
            operation=f"unsafe collective decryption at level {level}",
        )


@pytest.mark.gpu
def test_unsafe_public_key_switch_is_consumable_at_levels(
    engine: fh.CkksEngine,
    collective: _Collective,
) -> None:
    destination_secret = engine.create_secret_key(modulus_basis="QP")
    destination_public = engine.create_public_key(destination_secret)
    message = torch.stack((_real_message(engine), -0.5 * _real_message(engine)))

    for level in (0, 3, engine.final_public_level):
        ciphertext = engine.encrypt_message(
            message,
            collective.public_key,
            level=level,
        )
        zero = _compact_zeros(engine, *ciphertext.batch_shape)
        error0 = zero.clone()
        error0[..., 0] = 1
        error1 = zero.clone()
        error1[..., 1] = -1
        fixed_ephemeral = _compact_ternary_pattern(
            engine,
            *ciphertext.batch_shape,
        )
        baseline_share = mpc.unsafe_public_key_switch_share(
            engine,
            ciphertext,
            collective.secret_shares[0],
            destination_public,
            ephemeral_coefficients=fixed_ephemeral,
            smudging_error0_coefficients=zero,
            error1_coefficients=zero,
        )
        component0_error_share = mpc.unsafe_public_key_switch_share(
            engine,
            ciphertext,
            collective.secret_shares[0],
            destination_public,
            ephemeral_coefficients=fixed_ephemeral,
            smudging_error0_coefficients=error0,
            error1_coefficients=zero,
        )
        component1_error_share = mpc.unsafe_public_key_switch_share(
            engine,
            ciphertext,
            collective.secret_shares[0],
            destination_public,
            ephemeral_coefficients=fixed_ephemeral,
            smudging_error0_coefficients=zero,
            error1_coefficients=error1,
        )
        observed_error0 = engine.rns_runtime.sub_canonical(
            component0_error_share[0],
            baseline_share[0],
        )
        observed_error1 = engine.rns_runtime.sub_canonical(
            component1_error_share[1],
            baseline_share[1],
        )
        expected_error0 = mpc_ops._lift_coefficients(
            engine,
            error0,
            level=level,
            basis="Q",
            to_ntt=False,
        )
        expected_error1 = mpc_ops._lift_coefficients(
            engine,
            error1,
            level=level,
            basis="Q",
            to_ntt=False,
        )
        assert torch.equal(observed_error0, expected_error0)
        assert torch.equal(component0_error_share[1], baseline_share[1])
        assert torch.equal(observed_error1, expected_error1)
        assert torch.equal(component1_error_share[0], baseline_share[0])

        party_errors = ((error0, error1), (-error0, -error1))
        shares = []
        for party_index, (secret, errors) in enumerate(
            zip(collective.secret_shares, party_errors, strict=True)
        ):
            ephemeral = _compact_ternary_pattern(
                engine,
                *ciphertext.batch_shape,
            )
            ephemeral = torch.roll(ephemeral, shifts=party_index + 1, dims=-1)
            shares.append(
                mpc.unsafe_public_key_switch_share(
                    engine,
                    ciphertext,
                    secret,
                    destination_public,
                    ephemeral_coefficients=ephemeral.contiguous(),
                    smudging_error0_coefficients=errors[0].contiguous(),
                    error1_coefficients=errors[1].contiguous(),
                )
            )
        switched = mpc.unsafe_fuse_public_key_switch(
            engine,
            ciphertext,
            destination_public,
            shares,
        )
        assert switched.level == ciphertext.level
        assert switched.scale == ciphertext.scale
        assert switched.prime_ids == ciphertext.prime_ids
        assert switched.batch_shape == ciphertext.batch_shape
        assert switched.polynomial_domain == "coefficient"
        assert switched.residue_representation == "standard"
        decoded = engine.decrypt_message(
            switched,
            destination_secret,
            is_real=True,
        )
        _assert_close(
            decoded,
            message,
            atol=_KEYSWITCH_ATOL,
            operation=f"unsafe public-key switch at level {level}",
        )


@pytest.mark.gpu
def test_experimental_mpc_rejects_malformed_boundaries(
    engine: fh.CkksEngine,
    collective: _Collective,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        mpc.sample_common_uniform(engine, basis="Q", count=0)
    with pytest.raises(ValueError, match="basis must"):
        mpc.sample_common_uniform(engine, basis="q")  # type: ignore[arg-type]

    common_q = mpc.sample_common_uniform(engine, basis="Q")
    with pytest.raises(ValueError, match="at least one"):
        mpc.aggregate_ckg(engine, [], common_q)
    with pytest.raises(TypeError, match="dtype"):
        mpc.ckg_share(
            engine,
            collective.secret_shares[0],
            common_q.to(torch.int32),
        )
    noncontiguous = common_q[:, :1].expand_as(common_q)
    with pytest.raises(ValueError, match="contiguous"):
        mpc.ckg_share(
            engine,
            collective.secret_shares[0],
            noncontiguous,
        )
    q_only_secret = engine.create_secret_key(modulus_basis="Q")
    with pytest.raises(ValueError, match="basis QP"):
        mpc.ckg_share(engine, q_only_secret, common_q)

    message = _real_message(engine)
    ciphertext = engine.encrypt_message(message, collective.public_key)
    zero = _compact_zeros(engine)
    with pytest.raises(ValueError, match="polynomial_domain"):
        mpc.unsafe_collective_decryption_share(
            engine,
            engine.coefficient_domain_to_ntt_domain(ciphertext),
            collective.secret_shares[0],
            smudging_error_coefficients=zero,
        )
    wrong_error_shape = _compact_zeros(engine, 1)
    with pytest.raises(ValueError, match="shape differs"):
        mpc.unsafe_collective_decryption_share(
            engine,
            ciphertext,
            collective.secret_shares[0],
            smudging_error_coefficients=wrong_error_shape,
        )

    destination_secret = engine.create_secret_key(modulus_basis="QP")
    destination_qp = engine.create_public_key(
        destination_secret,
        modulus_basis="QP",
    )
    with pytest.raises(ValueError, match="basis Q"):
        mpc.unsafe_public_key_switch_share(
            engine,
            ciphertext,
            collective.secret_shares[0],
            destination_qp,
            ephemeral_coefficients=zero,
            smudging_error0_coefficients=zero,
            error1_coefficients=zero,
        )
    with pytest.raises(ValueError, match="at least one"):
        mpc.unsafe_fuse_public_key_switch(
            engine,
            ciphertext,
            engine.create_public_key(destination_secret),
            [],
        )
