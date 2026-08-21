r"""Process-local tensor operations for collective CKKS key and output protocols.

The functions form share-generation and aggregation steps for collective key
generation, evaluation-key generation, collective decryption arithmetic, and
public-key switching arithmetic. Their supported scope is arithmetic
correctness for compatible FHElium values and engine tensors. The implementation
accepts a CPU or CUDA ``CkksEngine``. The module provides no
authentication, transcript binding, transport, replay protection,
malicious-party security, secure aggregation, lineage or persistence policy,
reviewed output-error sampler, supported smudging/useful-precision parameter
profile, or privacy guarantee.  Functions whose names
begin with
``unsafe_`` implement secret-dependent output arithmetic with caller-provided
randomness and errors.  Zero or small errors are correctness fixtures with no
privacy property.

Each call operates against one :class:`~fhelium.CkksEngine`.  Secret shares
and ephemeral Protocol-2 secrets are ordinary process-local
:class:`~fhelium.SecretKey` values in the complete level-zero QP basis.
Common randomness and protocol messages are raw integral tensors on the engine
device.  The caller owns party membership, all-party participation, freshness,
delivery, and pairing each aggregate key with the correct additive shares.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch

from fhelium.core import (
    Ciphertext,
    ConjugationKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from fhelium.engine import CkksEngine
from fhelium.engine.galois import (
    apply_coefficient_galois_automorphism,
    rotation_galois_element,
)

__all__ = [
    "aggregate_ckg",
    "aggregate_conjugation_key",
    "aggregate_rkg_round1",
    "aggregate_rkg_round2",
    "aggregate_rotation_key",
    "ckg_share",
    "conjugation_key_share",
    "rkg_round1_share",
    "rkg_round2_share",
    "rotation_key_share",
    "sample_common_uniform",
    "sample_secret_share",
    "unsafe_collective_decryption_share",
    "unsafe_fuse_collective_decryption",
    "unsafe_fuse_public_key_switch",
    "unsafe_public_key_switch_share",
]

Basis = Literal["Q", "QP"]
RkgMessage = tuple[torch.Tensor, torch.Tensor]


def _require_secret_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    *,
    value_name: str,
) -> None:
    try:
        engine._assert_engine_key(
            secret_share,
            expected_type=SecretKey,
            modulus_basis="QP",
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"{value_name}: {error}") from error


def _require_public_key(
    engine: CkksEngine,
    public_key: PublicKey,
    *,
    value_name: str,
) -> None:
    try:
        engine._assert_engine_key(
            public_key,
            expected_type=PublicKey,
            modulus_basis="Q",
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"{value_name}: {error}") from error


def _require_source_ciphertext(
    engine: CkksEngine,
    ciphertext: Ciphertext,
) -> None:
    if not isinstance(ciphertext, Ciphertext):
        raise TypeError(
            f"ciphertext must be a Ciphertext, got {type(ciphertext).__name__}"
        )
    ciphertext.assert_state(
        polynomial_domain="coefficient",
        modulus_basis="Q",
        residue_representation="standard",
        components=2,
    )
    engine._assert_engine_ciphertext(ciphertext)


def _expected_rns_shape(
    engine: CkksEngine,
    *,
    basis: Basis,
    count: int | None = None,
) -> tuple[int, ...]:
    limb_count = engine.rns_layout.row_count(0, include_p=basis == "QP")
    tail = (limb_count, engine.config.N)
    return tail if count is None else (count, *tail)


def _require_raw_rns(
    engine: CkksEngine,
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    value_name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{value_name} must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.layout != torch.strided:
        raise TypeError(f"{value_name} must use dense strided storage")
    if value.dtype != engine.config.torch_dtype:
        raise TypeError(
            f"{value_name} dtype differs from engine: "
            f"{value.dtype} != {engine.config.torch_dtype}"
        )
    if value.device != engine.device:
        raise ValueError(
            f"{value_name} device differs from engine: "
            f"{value.device} != {engine.device}"
        )
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{value_name} shape differs from the required layout: "
            f"{tuple(value.shape)} != {expected_shape}"
        )
    if not value.is_contiguous():
        raise ValueError(f"{value_name} must be contiguous")


def _require_common_uniform(
    engine: CkksEngine,
    common_a: torch.Tensor,
    *,
    basis: Basis,
    count: int | None = None,
) -> None:
    _require_raw_rns(
        engine,
        common_a,
        expected_shape=_expected_rns_shape(
            engine,
            basis=basis,
            count=count,
        ),
        value_name="common_a",
    )


def _require_compact_coefficients(
    engine: CkksEngine,
    coefficients: torch.Tensor,
    *,
    batch_shape: torch.Size | tuple[int, ...],
    value_name: str,
) -> None:
    _require_raw_rns(
        engine,
        coefficients,
        expected_shape=(*tuple(batch_shape), engine.config.N),
        value_name=value_name,
    )


def _sample_gaussian_coefficients(
    engine: CkksEngine,
    *,
    count: int,
) -> torch.Tensor:
    if type(count) is not int or count <= 0:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    samples = [
        engine.rng.discrete_gaussian(repeats=1)[0][0] for _ in range(count)
    ]
    return torch.stack(samples, dim=0).contiguous()


def _lift_coefficients(
    engine: CkksEngine,
    coefficients: torch.Tensor,
    *,
    level: int,
    basis: Basis,
    to_ntt: bool,
) -> torch.Tensor:
    include_p = basis == "QP"
    contiguous = coefficients.contiguous()
    max_abs = int(torch.max(torch.abs(contiguous)).item())
    result = engine.rns_runtime.lift_integer_coefficients_exact(
        contiguous,
        level,
        include_p=include_p,
        max_abs=max_abs,
    )
    if to_ntt:
        engine.rns_runtime.forward_to_montgomery_(
            result,
            include_p=include_p,
        )
    else:
        engine.rns_runtime.canonicalize_residues_(
            result,
            include_p=include_p,
        )
    return result


def _sample_gaussian_rns(
    engine: CkksEngine,
    *,
    count: int,
    basis: Basis,
) -> torch.Tensor:
    return _lift_coefficients(
        engine,
        _sample_gaussian_coefficients(engine, count=count),
        level=0,
        basis=basis,
        to_ntt=True,
    )


def _sum_rns_lazy(
    engine: CkksEngine,
    values: Sequence[torch.Tensor],
    *,
    expected_shape: tuple[int, ...],
    value_name: str,
    include_p: bool,
) -> torch.Tensor:
    if not values:
        raise ValueError(f"{value_name} must contain at least one tensor")
    for index, value in enumerate(values):
        _require_raw_rns(
            engine,
            value,
            expected_shape=expected_shape,
            value_name=f"{value_name}[{index}]",
        )
    result = values[0].clone()
    for value in values[1:]:
        result = engine.rns_runtime.add_lazy(
            result,
            value,
            include_p=include_p,
        )
    return result


def _sum_rns_canonical(
    engine: CkksEngine,
    values: Sequence[torch.Tensor],
    *,
    expected_shape: tuple[int, ...],
    value_name: str,
) -> torch.Tensor:
    if not values:
        raise ValueError(f"{value_name} must contain at least one tensor")
    for index, value in enumerate(values):
        _require_raw_rns(
            engine,
            value,
            expected_shape=expected_shape,
            value_name=f"{value_name}[{index}]",
        )
    result = values[0].clone()
    for value in values[1:]:
        result = engine.rns_runtime.add_canonical(result, value)
    return result


def _require_rkg_message(
    engine: CkksEngine,
    message: RkgMessage,
    *,
    value_name: str,
) -> None:
    if not isinstance(message, tuple) or len(message) != 2:
        raise TypeError(f"{value_name} must be a two-tensor tuple")
    expected_shape = _expected_rns_shape(
        engine,
        basis="QP",
        count=engine.rns_layout.key_digit_count,
    )
    for family_index, family in enumerate(message):
        _require_raw_rns(
            engine,
            family,
            expected_shape=expected_shape,
            value_name=f"{value_name}[{family_index}]",
        )


def _embed_p_times_secret_by_digit(
    engine: CkksEngine,
    secret_share: SecretKey,
) -> torch.Tensor:
    q_rows = secret_share.data[: engine.rns_runtime.q_row_stop].clone()
    engine.rns_runtime.montgomery_mul_row_scalars_(
        q_rows,
        engine.p_product_montgomery_q,
    )
    digit_count = engine.rns_layout.key_digit_count
    embedded = torch.zeros(
        _expected_rns_shape(engine, basis="QP", count=digit_count),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    for digit_spec in engine.rns_layout.digit_specs(0):
        rows = cast(tuple[int, ...], digit_spec.prime_ids)
        row_start = rows[0]
        row_stop = rows[-1] + 1
        embedded[
            digit_spec.key_digit_index,
            row_start:row_stop,
        ].copy_(q_rows[row_start:row_stop])
    return embedded


def _automorph_secret_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    *,
    galois_element: int,
) -> SecretKey:
    transformed = secret_share.data.clone()
    engine.rns_runtime.inverse_montgomery_(transformed, include_p=True)
    transformed = apply_coefficient_galois_automorphism(
        transformed,
        galois_element,
        engine.rns_runtime.moduli,
    )
    engine.rns_runtime.forward_montgomery_(transformed, include_p=True)
    return SecretKey(
        data=transformed,
        context_id=secret_share.context_id,
        prime_ids=secret_share.prime_ids,
        polynomial_domain="ntt",
        modulus_basis="QP",
        residue_representation="montgomery",
    )


def _galois_key_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    common_a_by_digit: torch.Tensor,
    *,
    galois_element: int,
) -> torch.Tensor:
    _require_secret_share(engine, secret_share, value_name="secret_share")
    digit_count = engine.rns_layout.key_digit_count
    _require_common_uniform(
        engine,
        common_a_by_digit,
        basis="QP",
        count=digit_count,
    )
    transformed = _automorph_secret_share(
        engine,
        secret_share,
        galois_element=galois_element,
    )
    embedded = _embed_p_times_secret_by_digit(engine, transformed)
    destination_product = engine.rns_runtime.montgomery_mul(
        common_a_by_digit,
        secret_share.data,
        include_p=True,
    )
    error = _sample_gaussian_rns(
        engine,
        count=digit_count,
        basis="QP",
    )
    share = engine.rns_runtime.sub_lazy(
        embedded,
        destination_product,
        include_p=True,
    )
    return engine.rns_runtime.add_lazy(share, error, include_p=True)


def _aggregate_galois_key_data(
    engine: CkksEngine,
    shares: Sequence[torch.Tensor],
    common_a_by_digit: torch.Tensor,
) -> torch.Tensor:
    digit_count = engine.rns_layout.key_digit_count
    expected_shape = _expected_rns_shape(
        engine,
        basis="QP",
        count=digit_count,
    )
    _require_common_uniform(
        engine,
        common_a_by_digit,
        basis="QP",
        count=digit_count,
    )
    component0 = _sum_rns_lazy(
        engine,
        shares,
        expected_shape=expected_shape,
        value_name="shares",
        include_p=True,
    )
    return torch.stack((component0, common_a_by_digit), dim=1)


def _active_q_secret_rows(
    engine: CkksEngine,
    secret_share: SecretKey,
    *,
    level: int,
) -> torch.Tensor:
    return secret_share.data[
        engine.rns_runtime.level_row_starts[
            level
        ] : engine.rns_runtime.q_row_stop
    ]


def _ciphertext_secret_product(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    secret_share: SecretKey,
) -> torch.Tensor:
    c1_ntt = engine.rns_runtime.forward_to_montgomery(ciphertext.c1)
    product = engine.rns_runtime.montgomery_mul(
        c1_ntt,
        _active_q_secret_rows(
            engine,
            secret_share,
            level=ciphertext.level,
        ),
    )
    engine.rns_runtime.inverse_to_standard_(product)
    return product


def sample_secret_share(engine: CkksEngine) -> SecretKey:
    r"""Sample one additive secret share in complete level-zero QP form."""

    return engine.create_secret_key(modulus_basis="QP")


def sample_common_uniform(
    engine: CkksEngine,
    *,
    basis: Basis,
    count: int | None = None,
) -> torch.Tensor:
    r"""Sample raw common uniform NTT/Montgomery tensors.

    The direct row-wise residues are interpreted as uniform NTT/Montgomery
    values, matching ordinary FHElium key generation.  Omitting ``count``
    returns one unbatched ``[limb, N]`` tensor.  Every explicit positive
    ``count``, including ``count=1``, returns
    ``[count, limb, N]`` with a leading item/digit axis.  The caller must
    distribute the returned values to every participant.
    """

    if basis not in ("Q", "QP"):
        raise ValueError(f"basis must be 'Q' or 'QP', got {basis!r}")
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError(f"count must be a positive integer, got {count!r}")
    sample_count = 1 if count is None else count
    include_p = basis == "QP"
    moduli = engine.rns_runtime.moduli_for_basis(0, include_p=include_p)
    repeats = engine.config.num_p_primes if include_p else 0
    values = [
        engine.rng.randint([moduli], repeats=repeats)[0]
        for _ in range(sample_count)
    ]
    if count is None:
        return values[0].contiguous()
    return torch.stack(values, dim=0).contiguous()


def ckg_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    common_a: torch.Tensor,
) -> torch.Tensor:
    r"""Return one Protocol-1 share $b_i=e_i-a s_i$ on Q."""

    _require_secret_share(engine, secret_share, value_name="secret_share")
    _require_common_uniform(engine, common_a, basis="Q")
    error = _sample_gaussian_rns(engine, count=1, basis="Q")[0]
    secret_q = secret_share.data[: engine.rns_runtime.q_row_stop]
    product = engine.rns_runtime.montgomery_mul(common_a, secret_q)
    return engine.rns_runtime.sub_lazy(error, product)


def aggregate_ckg(
    engine: CkksEngine,
    shares: Sequence[torch.Tensor],
    common_a: torch.Tensor,
) -> PublicKey:
    r"""Aggregate Protocol-1 shares into an ordinary FHElium public key."""

    _require_common_uniform(engine, common_a, basis="Q")
    component0 = _sum_rns_lazy(
        engine,
        shares,
        expected_shape=_expected_rns_shape(engine, basis="Q"),
        value_name="shares",
        include_p=False,
    )
    return PublicKey(
        data=torch.stack((component0, common_a), dim=0),
        context_id=engine.context.context_id,
        prime_ids=engine.rns_layout.prime_ids(0),
        polynomial_domain="ntt",
        modulus_basis="Q",
        residue_representation="montgomery",
    )


def rkg_round1_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    ephemeral_share: SecretKey,
    common_a_by_digit: torch.Tensor,
) -> RkgMessage:
    r"""Return the two separate Protocol-2 round-one message families."""

    _require_secret_share(engine, secret_share, value_name="secret_share")
    _require_secret_share(
        engine,
        ephemeral_share,
        value_name="ephemeral_share",
    )
    digit_count = engine.rns_layout.key_digit_count
    _require_common_uniform(
        engine,
        common_a_by_digit,
        basis="QP",
        count=digit_count,
    )
    error0 = _sample_gaussian_rns(
        engine,
        count=digit_count,
        basis="QP",
    )
    error1 = _sample_gaussian_rns(
        engine,
        count=digit_count,
        basis="QP",
    )
    ephemeral_product = engine.rns_runtime.montgomery_mul(
        common_a_by_digit,
        ephemeral_share.data,
        include_p=True,
    )
    family0 = engine.rns_runtime.sub_lazy(
        _embed_p_times_secret_by_digit(engine, secret_share),
        ephemeral_product,
        include_p=True,
    )
    family0 = engine.rns_runtime.add_lazy(
        family0,
        error0,
        include_p=True,
    )
    family1 = engine.rns_runtime.montgomery_mul(
        common_a_by_digit,
        secret_share.data,
        include_p=True,
    )
    family1 = engine.rns_runtime.add_lazy(
        family1,
        error1,
        include_p=True,
    )
    return family0, family1


def aggregate_rkg_round1(
    engine: CkksEngine,
    shares: Sequence[RkgMessage],
) -> RkgMessage:
    r"""Aggregate each Protocol-2 round-one message family independently."""

    if not shares:
        raise ValueError("shares must contain at least one RKG message")
    for index, share in enumerate(shares):
        _require_rkg_message(engine, share, value_name=f"shares[{index}]")
    expected_shape = _expected_rns_shape(
        engine,
        basis="QP",
        count=engine.rns_layout.key_digit_count,
    )
    family0 = _sum_rns_lazy(
        engine,
        [share[0] for share in shares],
        expected_shape=expected_shape,
        value_name="round1_family0",
        include_p=True,
    )
    family1 = _sum_rns_lazy(
        engine,
        [share[1] for share in shares],
        expected_shape=expected_shape,
        value_name="round1_family1",
        include_p=True,
    )
    return family0, family1


def rkg_round2_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    ephemeral_share: SecretKey,
    aggregate_round1: RkgMessage,
) -> RkgMessage:
    r"""Return the two separate Protocol-2 round-two message families."""

    _require_secret_share(engine, secret_share, value_name="secret_share")
    _require_secret_share(
        engine,
        ephemeral_share,
        value_name="ephemeral_share",
    )
    _require_rkg_message(
        engine,
        aggregate_round1,
        value_name="aggregate_round1",
    )
    digit_count = engine.rns_layout.key_digit_count
    error2 = _sample_gaussian_rns(
        engine,
        count=digit_count,
        basis="QP",
    )
    error3 = _sample_gaussian_rns(
        engine,
        count=digit_count,
        basis="QP",
    )
    family0 = engine.rns_runtime.montgomery_mul(
        aggregate_round1[0],
        secret_share.data,
        include_p=True,
    )
    family0 = engine.rns_runtime.add_lazy(
        family0,
        error2,
        include_p=True,
    )
    ephemeral_minus_secret = engine.rns_runtime.sub_lazy(
        ephemeral_share.data,
        secret_share.data,
        include_p=True,
    )
    family1 = engine.rns_runtime.montgomery_mul(
        aggregate_round1[1],
        ephemeral_minus_secret,
        include_p=True,
    )
    family1 = engine.rns_runtime.add_lazy(
        family1,
        error3,
        include_p=True,
    )
    return family0, family1


def aggregate_rkg_round2(
    engine: CkksEngine,
    shares: Sequence[RkgMessage],
    aggregate_round1: RkgMessage,
) -> RelinearizationKey:
    r"""Aggregate Protocol-2 round two into a relinearization key."""

    _require_rkg_message(
        engine,
        aggregate_round1,
        value_name="aggregate_round1",
    )
    if not shares:
        raise ValueError("shares must contain at least one RKG message")
    for index, share in enumerate(shares):
        _require_rkg_message(engine, share, value_name=f"shares[{index}]")
    expected_shape = _expected_rns_shape(
        engine,
        basis="QP",
        count=engine.rns_layout.key_digit_count,
    )
    family0 = _sum_rns_lazy(
        engine,
        [share[0] for share in shares],
        expected_shape=expected_shape,
        value_name="round2_family0",
        include_p=True,
    )
    family1 = _sum_rns_lazy(
        engine,
        [share[1] for share in shares],
        expected_shape=expected_shape,
        value_name="round2_family1",
        include_p=True,
    )
    component0 = engine.rns_runtime.add_lazy(
        family0,
        family1,
        include_p=True,
    )
    return RelinearizationKey(
        data=torch.stack((component0, aggregate_round1[1]), dim=1),
        context_id=engine.context.context_id,
        prime_ids=engine.rns_layout.prime_ids(0, include_p=True),
        polynomial_domain="ntt",
        modulus_basis="QP",
        residue_representation="montgomery",
    )


def rotation_key_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    common_a_by_digit: torch.Tensor,
    rotation_step: int,
) -> torch.Tensor:
    r"""Return one distributed rotation-key share for a signed slot step."""

    canonical_step = RotationKey.canonical_step(
        rotation_step,
        ring_dimension=engine.config.N,
    )
    return _galois_key_share(
        engine,
        secret_share,
        common_a_by_digit,
        galois_element=rotation_galois_element(
            engine.config.N,
            canonical_step,
            engine.galois_generator,
        ),
    )


def aggregate_rotation_key(
    engine: CkksEngine,
    shares: Sequence[torch.Tensor],
    common_a_by_digit: torch.Tensor,
    rotation_step: int,
) -> RotationKey:
    r"""Aggregate distributed shares into one ordinary rotation key."""

    canonical_step = RotationKey.canonical_step(
        rotation_step,
        ring_dimension=engine.config.N,
    )
    return RotationKey(
        data=_aggregate_galois_key_data(
            engine,
            shares,
            common_a_by_digit,
        ),
        context_id=engine.context.context_id,
        prime_ids=engine.rns_layout.prime_ids(0, include_p=True),
        polynomial_domain="ntt",
        modulus_basis="QP",
        residue_representation="montgomery",
        rotation_step=canonical_step,
    )


def conjugation_key_share(
    engine: CkksEngine,
    secret_share: SecretKey,
    common_a_by_digit: torch.Tensor,
) -> torch.Tensor:
    r"""Return one distributed conjugation-key share."""

    return _galois_key_share(
        engine,
        secret_share,
        common_a_by_digit,
        galois_element=2 * engine.config.N - 1,
    )


def aggregate_conjugation_key(
    engine: CkksEngine,
    shares: Sequence[torch.Tensor],
    common_a_by_digit: torch.Tensor,
) -> ConjugationKey:
    r"""Aggregate distributed shares into one ordinary conjugation key."""

    return ConjugationKey(
        data=_aggregate_galois_key_data(
            engine,
            shares,
            common_a_by_digit,
        ),
        context_id=engine.context.context_id,
        prime_ids=engine.rns_layout.prime_ids(0, include_p=True),
        polynomial_domain="ntt",
        modulus_basis="QP",
        residue_representation="montgomery",
    )


def unsafe_collective_decryption_share(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    secret_share: SecretKey,
    *,
    smudging_error_coefficients: torch.Tensor,
) -> torch.Tensor:
    r"""Return arithmetic-only $c_1s_i+e_i$ for collective decryption.

    ``smudging_error_coefficients`` must have
    ``[*ciphertext.batch_shape, N]`` contiguous engine-integral layout on
    ``engine.device``.  Distribution selection and privacy analysis belong to
    the caller.
    """

    _require_source_ciphertext(engine, ciphertext)
    _require_secret_share(engine, secret_share, value_name="secret_share")
    _require_compact_coefficients(
        engine,
        smudging_error_coefficients,
        batch_shape=ciphertext.batch_shape,
        value_name="smudging_error_coefficients",
    )
    product = _ciphertext_secret_product(engine, ciphertext, secret_share)
    error = _lift_coefficients(
        engine,
        smudging_error_coefficients,
        level=ciphertext.level,
        basis="Q",
        to_ntt=False,
    )
    return engine.rns_runtime.add_canonical(product, error)


def unsafe_fuse_collective_decryption(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    shares: Sequence[torch.Tensor],
) -> Plaintext:
    r"""Fuse arithmetic-only shares and apply the bounded tail-Q decoder."""

    _require_source_ciphertext(engine, ciphertext)
    expected_shape = (
        *ciphertext.batch_shape,
        ciphertext.limb_count,
        engine.config.N,
    )
    share_sum = _sum_rns_canonical(
        engine,
        shares,
        expected_shape=expected_shape,
        value_name="shares",
    )
    phase = engine.rns_runtime.add_canonical(ciphertext.c0, share_sum)
    coefficients = engine._decryptor._reconstruct_tail_q_coefficients_float64(
        phase,
        ciphertext,
    )
    return Plaintext(
        message=None,
        level=ciphertext.level,
        scale=ciphertext.scale,
        data=coefficients,
        context_id=ciphertext.context_id,
        representation="approximate_coefficients",
        polynomial_domain="coefficient",
    )


def unsafe_public_key_switch_share(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    secret_share: SecretKey,
    destination_public_key: PublicKey,
    *,
    ephemeral_coefficients: torch.Tensor,
    smudging_error0_coefficients: torch.Tensor,
    error1_coefficients: torch.Tensor,
) -> RkgMessage:
    r"""Return arithmetic-only Protocol-4 share components.

    All caller-provided coefficient tensors must have
    ``[*ciphertext.batch_shape, N]`` contiguous engine-integral layout on
    ``engine.device``.  Freshness, smallness, smudging adequacy, and
    destination-key provenance are caller responsibilities.
    """

    _require_source_ciphertext(engine, ciphertext)
    _require_secret_share(engine, secret_share, value_name="secret_share")
    _require_public_key(
        engine,
        destination_public_key,
        value_name="destination_public_key",
    )
    for value_name, coefficients in (
        ("ephemeral_coefficients", ephemeral_coefficients),
        ("smudging_error0_coefficients", smudging_error0_coefficients),
        ("error1_coefficients", error1_coefficients),
    ):
        _require_compact_coefficients(
            engine,
            coefficients,
            batch_shape=ciphertext.batch_shape,
            value_name=value_name,
        )

    source_product = _ciphertext_secret_product(
        engine,
        ciphertext,
        secret_share,
    )
    ephemeral_rns = _lift_coefficients(
        engine,
        ephemeral_coefficients,
        level=ciphertext.level,
        basis="Q",
        to_ntt=True,
    )
    start = engine.rns_runtime.level_row_starts[ciphertext.level]
    destination0 = destination_public_key.k0[start:]
    destination1 = destination_public_key.k1[start:]
    encrypted0 = engine.rns_runtime.montgomery_mul(
        ephemeral_rns,
        destination0,
    )
    encrypted1 = engine.rns_runtime.montgomery_mul(
        ephemeral_rns,
        destination1,
    )
    engine.rns_runtime.inverse_to_standard_(encrypted0)
    engine.rns_runtime.inverse_to_standard_(encrypted1)
    error0 = _lift_coefficients(
        engine,
        smudging_error0_coefficients,
        level=ciphertext.level,
        basis="Q",
        to_ntt=False,
    )
    error1 = _lift_coefficients(
        engine,
        error1_coefficients,
        level=ciphertext.level,
        basis="Q",
        to_ntt=False,
    )
    component0 = engine.rns_runtime.add_canonical(
        source_product,
        encrypted0,
    )
    component0 = engine.rns_runtime.add_canonical(component0, error0)
    component1 = engine.rns_runtime.add_canonical(encrypted1, error1)
    return component0, component1


def unsafe_fuse_public_key_switch(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    destination_public_key: PublicKey,
    shares: Sequence[RkgMessage],
) -> Ciphertext:
    r"""Fuse arithmetic-only Protocol-4 shares into a Q ciphertext."""

    _require_source_ciphertext(engine, ciphertext)
    _require_public_key(
        engine,
        destination_public_key,
        value_name="destination_public_key",
    )
    if not shares:
        raise ValueError("shares must contain at least one Protocol-4 message")
    expected_shape = (
        *ciphertext.batch_shape,
        ciphertext.limb_count,
        engine.config.N,
    )
    for index, share in enumerate(shares):
        if not isinstance(share, tuple) or len(share) != 2:
            raise TypeError(f"shares[{index}] must be a two-tensor tuple")
        for component_index, component in enumerate(share):
            _require_raw_rns(
                engine,
                component,
                expected_shape=expected_shape,
                value_name=f"shares[{index}][{component_index}]",
            )
    component0_sum = _sum_rns_canonical(
        engine,
        [share[0] for share in shares],
        expected_shape=expected_shape,
        value_name="share_component0",
    )
    component1 = _sum_rns_canonical(
        engine,
        [share[1] for share in shares],
        expected_shape=expected_shape,
        value_name="share_component1",
    )
    component0 = engine.rns_runtime.add_canonical(
        ciphertext.c0,
        component0_sum,
    )
    return Ciphertext(
        data=torch.stack((component0, component1), dim=0),
        level=ciphertext.level,
        scale=ciphertext.scale,
        context_id=ciphertext.context_id,
        prime_ids=ciphertext.prime_ids,
        polynomial_domain="coefficient",
        modulus_basis="Q",
        residue_representation="standard",
    )
