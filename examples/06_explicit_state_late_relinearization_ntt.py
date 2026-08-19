#!/usr/bin/env python3

"""Explicit late-relinearization and NTT-reuse workflows."""

from __future__ import annotations

import argparse
from typing import cast

import torch
from common import (
    add_engine_args,
    error_stats,
    make_engine,
    print_table,
    small_complex_vector,
)

import fhelium as fh


def _state(ct: fh.Ciphertext) -> str:
    return (
        f"components={ct.component_count},polynomial_domain={ct.polynomial_domain},"
        f"residue_representation={ct.residue_representation},scale={ct.scale:.3e}"
    )


def late_relinearization(
    engine: fh.CkksEngine, pair_count: int
) -> tuple[fh.Ciphertext, torch.Tensor]:
    accumulator: fh.Ciphertext | None = None
    reference: torch.Tensor | None = None
    for index in range(pair_count):
        multiplicand = small_complex_vector(
            engine.num_slots, seed=100 + index, scale=0.005
        )
        multiplier = small_complex_vector(
            engine.num_slots, seed=200 + index, scale=0.005
        )
        multiplicand_ntt = cast(
            fh.Ciphertext,
            engine.coefficient_domain_to_ntt_domain(
                engine.encrypt_message(multiplicand)
            ),
        )
        multiplier_ntt = cast(
            fh.Ciphertext,
            engine.coefficient_domain_to_ntt_domain(
                engine.encrypt_message(multiplier)
            ),
        )
        product = engine.multiply(multiplicand_ntt, multiplier_ntt)
        accumulator = (
            product if accumulator is None else engine.add(accumulator, product)
        )
        reference = (
            multiplicand * multiplier
            if reference is None
            else reference + multiplicand * multiplier
        )

    if accumulator is None or reference is None:
        raise ValueError("pair_count must be positive")
    output = engine.rescale_to_next_level(engine.relinearize(accumulator))
    return output, reference


def ntt_reuse(
    engine: fh.CkksEngine,
) -> tuple[fh.Ciphertext, torch.Tensor]:
    fixed_values = small_complex_vector(engine.num_slots, seed=300, scale=0.005)
    source_values = small_complex_vector(
        engine.num_slots, seed=301, scale=0.005
    )
    fixed = cast(
        fh.Ciphertext,
        engine.coefficient_domain_to_ntt_domain(
            engine.encrypt_message(fixed_values)
        ),
    )
    source = cast(
        fh.Ciphertext,
        engine.coefficient_domain_to_ntt_domain(
            engine.encrypt_message(source_values)
        ),
    )

    product = engine.multiply(source, fixed)
    output = engine.rescale_to_next_level(engine.relinearize(product))
    return output, source_values * fixed_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument("--pair-count", type=int, default=3)
    args = parser.parse_args()
    engine = make_engine(args)

    late, late_reference = late_relinearization(engine, args.pair_count)
    reuse, reuse_reference = ntt_reuse(engine)
    rows = []
    for name, value, reference in [
        ("late relinearization", late, late_reference),
        ("NTT operand reuse", reuse, reuse_reference),
    ]:
        error = error_stats(
            engine.decrypt_message(value), reference, engine.num_slots
        )
        rows.append(
            [name, value.level, _state(value), f"{error['max_abs']:.3e}"]
        )
    print_table(["demo", "level", "state", "max abs error"], rows)


if __name__ == "__main__":
    main()
