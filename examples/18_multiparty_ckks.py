#!/usr/bin/env python3

"""Multiparty CKKS with application-owned protocol state.

This runnable example emulates two logical party slots in one process on one
local CPU or CUDA engine. Each slot keeps its local additive secret share while
the application passes raw messages through collective key generation,
evaluation-key generation, and two secret-dependent output workflows.  The
example validates arithmetic correctness with synthetic data, throwaway keys,
fixed output-error fixtures, and fixed PCKS ephemeral coefficients.

The single process provides no trust-domain or process isolation between the
two logical slots.

Authentication, transport, transcript binding, malicious-party security, a
reviewed output-error sampler, a supported smudging/useful-precision parameter
profile, a privacy guarantee, and representative precision analysis are not
provided by this example.  The application never reconstructs the aggregate
secret.
"""

from __future__ import annotations

import argparse

import torch
from common import add_engine_args, error_stats, make_engine, print_table

from fhelium.experimental import mpc


def announce(state: str, detail: str) -> None:
    """Print one application-owned workflow state."""

    print(f"[{state}] {detail}")


def fixed_impulse(
    engine,
    *,
    coefficient: int,
    value: int,
) -> torch.Tensor:
    """Return one fixed compact coefficient fixture on the engine device."""

    result = torch.zeros(
        engine.config.N,
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    result[coefficient] = value
    return result


def fixed_ternary(engine, *, shift: int) -> torch.Tensor:
    """Return one fixed compact ternary fixture on the engine device."""

    values = (
        torch.arange(
            engine.config.N,
            dtype=engine.config.torch_dtype,
            device=engine.device,
        )
        % 3
    ) - 1
    return torch.roll(values, shifts=shift).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-levels7-int64")
    args = parser.parse_args()

    engine = make_engine(args)

    print(
        "Scope: arithmetic correctness with synthetic data, throwaway keys, "
        "and fixed output fixtures."
    )
    print(
        "Authentication, transcript security, reviewed output-error sampling, "
        "a supported smudging/useful-precision parameter profile, and a privacy "
        "guarantee are outside this example."
    )
    print(
        "The application owns every state transition; FHElium MPC functions "
        "remain stateless."
    )

    announce("CREATED", "the epoch descriptor and two-party roster are frozen")

    # Application-owned transition: CREATED -> CKG_COLLECTING.
    party_secret_shares = tuple(
        mpc.sample_secret_share(engine) for _ in range(2)
    )
    announce(
        "CKG_COLLECTING",
        "two compatible QP additive shares exist; neither leaves its party slot",
    )
    print_table(
        ["party", "local secret shape", "context"],
        [
            [
                party_index,
                tuple(secret_share.data.shape),
                f"{secret_share.context_id[:12]}...",
            ]
            for party_index, secret_share in enumerate(party_secret_shares)
        ],
    )

    # Application-owned transition: CKG_COLLECTING -> PUBLIC_READY -> ACTIVE.
    # Protocol 1 is one round: every party receives the same common `a`, emits
    # one `b_i`, and the application aggregates only those public shares.
    ckg_common_a = mpc.sample_common_uniform(engine, basis="Q")
    ckg_shares = tuple(
        mpc.ckg_share(engine, secret_share, ckg_common_a)
        for secret_share in party_secret_shares
    )
    collective_public_key = mpc.aggregate_ckg(
        engine,
        ckg_shares,
        ckg_common_a,
    )
    announce(
        "PUBLIC_READY",
        "one-round CKG produced a public Q encryption key",
    )
    announce("ACTIVE", "material and evaluation requests may now run")

    # Application-owned RKG child state machine under ACTIVE.
    # Protocol 2 retains one local ephemeral QP secret per party. The
    # same objects are passed to that party's round-one and round-two calls.
    digit_count = engine.rns_layout.key_digit_count
    rkg_common_a = mpc.sample_common_uniform(
        engine,
        basis="QP",
        count=digit_count,
    )
    rkg_ephemeral_by_party = tuple(
        mpc.sample_secret_share(engine) for _ in party_secret_shares
    )
    rkg_round1_by_party = tuple(
        mpc.rkg_round1_share(
            engine,
            secret_share,
            rkg_ephemeral_by_party[party_index],
            rkg_common_a,
        )
        for party_index, secret_share in enumerate(party_secret_shares)
    )
    announce("RKG:R1_LOCAL_CACHED", "every party cached its round-one tuple")
    aggregate_round1 = mpc.aggregate_rkg_round1(
        engine,
        rkg_round1_by_party,
    )
    announce("RKG:R1_AGGREGATED", "both round-one families were aggregated")
    rkg_round2_by_party = tuple(
        mpc.rkg_round2_share(
            engine,
            secret_share,
            rkg_ephemeral_by_party[party_index],
            aggregate_round1,
        )
        for party_index, secret_share in enumerate(party_secret_shares)
    )
    announce("RKG:R2_LOCAL_CACHED", "every party cached its round-two tuple")
    relinearization_key = mpc.aggregate_rkg_round2(
        engine,
        rkg_round2_by_party,
        aggregate_round1,
    )
    announce("RKG:COMPLETE", "the relinearization key is an ordinary core key")

    rotation_step = 1
    rotation_common_a = mpc.sample_common_uniform(
        engine,
        basis="QP",
        count=digit_count,
    )
    rotation_shares = tuple(
        mpc.rotation_key_share(
            engine,
            secret_share,
            rotation_common_a,
            rotation_step,
        )
        for secret_share in party_secret_shares
    )
    rotation_key = mpc.aggregate_rotation_key(
        engine,
        rotation_shares,
        rotation_common_a,
        rotation_step,
    )
    announce("ROTATION:COMPLETE", "one rotation-key request completed")

    conjugation_common_a = mpc.sample_common_uniform(
        engine,
        basis="QP",
        count=digit_count,
    )
    conjugation_shares = tuple(
        mpc.conjugation_key_share(
            engine,
            secret_share,
            conjugation_common_a,
        )
        for secret_share in party_secret_shares
    )
    conjugation_key = mpc.aggregate_conjugation_key(
        engine,
        conjugation_shares,
        conjugation_common_a,
    )
    announce("CONJUGATION:COMPLETE", "one conjugation-key request completed")

    # Public evaluation while the collective epoch remains ACTIVE.
    positions = torch.arange(engine.num_slots, dtype=torch.float64)
    real = 0.008 * torch.sin(positions * 0.013) + 0.004 * torch.cos(
        positions * 0.007
    )
    imag = 0.003 * torch.sin(positions * 0.011) - 0.002 * torch.cos(
        positions * 0.005
    )
    message = torch.complex(real, imag)
    source = engine.encrypt_message(message, collective_public_key)
    announce(
        "ACTIVE:ENCRYPTED_INPUT",
        "an ordinary CKKS ciphertext was encrypted with the collective public key",
    )

    # Public evaluation consumes ordinary core keys and values.
    rotated = engine.rotate_with_key(source, rotation_key)
    conjugated = engine.conjugate(source, conjugation_key)
    transformed = engine.add(rotated, conjugated)
    source_ntt_left = engine.coefficient_domain_to_ntt_domain(source)
    source_ntt_right = engine.coefficient_domain_to_ntt_domain(source)
    product = engine.multiply(source_ntt_left, source_ntt_right)
    squared = engine.rescale_to_next_level(
        engine.relinearize(product, relinearization_key)
    )
    announce(
        "ACTIVE:EVALUATED",
        "rotation, conjugation, and relinearization keys were consumed by ordinary engine operations",
    )

    # Application-owned Protocol-3 child request under ACTIVE.
    # These opposite fixed impulses cancel in the fused arithmetic.  They are
    # small deterministic correctness fixtures.  They supply no smudging
    # distribution or privacy property.
    p3_error0 = fixed_impulse(engine, coefficient=0, value=1)
    p3_error_by_party = (p3_error0, -p3_error0)
    p3_shares = tuple(
        mpc.unsafe_collective_decryption_share(
            engine,
            transformed,
            secret_share,
            smudging_error_coefficients=p3_error_by_party[party_index],
        )
        for party_index, secret_share in enumerate(party_secret_shares)
    )
    transformed_plaintext = mpc.unsafe_fuse_collective_decryption(
        engine,
        transformed,
        p3_shares,
    )
    transformed_decoded = engine.decode(transformed_plaintext, is_real=False)
    transformed_expected = torch.roll(
        message, shifts=rotation_step
    ) + torch.conj(message)
    transformed_error = error_stats(transformed_decoded, transformed_expected)
    if transformed_error["max_abs"] >= 2e-5:
        raise AssertionError(
            "collective transform output exceeded the established key-switch "
            f"correctness tolerance: {transformed_error['max_abs']:.3e}"
        )
    announce(
        "P3:COMPLETE",
        "unsafe collective shares fused to Plaintext and decoded without reconstructing the aggregate secret",
    )

    # Application-owned Protocol-4 child request under ACTIVE.
    # The destination key pair is ordinary throwaway FHElium material.  The
    # fixed ternary ephemerals and opposite component errors below exercise PCKS
    # arithmetic only.  They supply neither fresh randomness nor a privacy
    # property.
    destination_secret = engine.create_secret_key(modulus_basis="QP")
    destination_public = engine.create_public_key(destination_secret)
    p4_error0 = fixed_impulse(engine, coefficient=0, value=1)
    p4_error1 = fixed_impulse(engine, coefficient=1, value=-1)
    p4_errors_by_party = (
        (p4_error0, p4_error1),
        (-p4_error0, -p4_error1),
    )
    p4_ephemeral_by_party = (
        fixed_ternary(engine, shift=1),
        fixed_ternary(engine, shift=2),
    )
    p4_shares = tuple(
        mpc.unsafe_public_key_switch_share(
            engine,
            squared,
            secret_share,
            destination_public,
            ephemeral_coefficients=p4_ephemeral_by_party[party_index],
            smudging_error0_coefficients=p4_errors_by_party[party_index][0],
            error1_coefficients=p4_errors_by_party[party_index][1],
        )
        for party_index, secret_share in enumerate(party_secret_shares)
    )
    recipient_ciphertext = mpc.unsafe_fuse_public_key_switch(
        engine,
        squared,
        destination_public,
        p4_shares,
    )
    recipient_decoded = engine.decrypt_message(
        recipient_ciphertext,
        destination_secret,
        is_real=False,
    )
    recipient_expected = message.square()
    recipient_error = error_stats(recipient_decoded, recipient_expected)
    if recipient_error["max_abs"] >= 3e-5:
        raise AssertionError(
            "PCKS squared output exceeded the established multiplication "
            f"correctness tolerance: {recipient_error['max_abs']:.3e}"
        )
    announce(
        "P4:COMPLETE",
        "unsafe PCKS produced a ciphertext decrypted only by the throwaway destination key",
    )

    print_table(
        ["output", "expected", "max_abs", "rms"],
        [
            [
                "collective transform",
                "roll(message, +1) + conj(message)",
                f"{transformed_error['max_abs']:.3e}",
                f"{transformed_error['rms']:.3e}",
            ],
            [
                "recipient PCKS",
                "message**2",
                f"{recipient_error['max_abs']:.3e}",
                f"{recipient_error['rms']:.3e}",
            ],
        ],
    )
    print(
        "Completed without constructing, installing, decrypting with, or "
        "checking an aggregate secret."
    )
    announce("CLOSED", "the local synthetic epoch accepts no further requests")


if __name__ == "__main__":
    main()
