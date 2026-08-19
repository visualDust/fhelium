#!/usr/bin/env python3

"""Basic dense-tensor CKKS encrypt/decrypt and arithmetic flow."""

from __future__ import annotations

import argparse

import torch
from common import (
    add_engine_args,
    error_stats,
    make_engine,
    print_table,
    small_complex_vector,
    sync_if_cuda,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    engine = make_engine(args)
    slots = engine.num_slots
    x = small_complex_vector(slots, seed=1)
    y = small_complex_vector(slots, seed=2)
    ct_x = engine.encrypt_message(x, level=args.level)
    ct_y = engine.encrypt_message(y, level=args.level)

    # These are three independent branches. add is out-of-place and does
    # not feed the multiplication or rotation below.
    ct_sum = engine.add(ct_x, ct_y)

    # Direct CKKS operands enter multiplication at the ordinary scale. The
    # pending square scale is consumed after relinearization.
    mul_x = engine.coefficient_domain_to_ntt_domain(ct_x)
    mul_y = engine.coefficient_domain_to_ntt_domain(ct_y)
    product_triplet = engine.multiply(mul_x, mul_y)
    ct_product = engine.rescale_to_next_level(
        engine.relinearize(product_triplet)
    )

    ct_rotated = engine.rotate_with_key(ct_x, engine.rotation_key(1))
    sync_if_cuda(engine.device)

    rows = []
    for name, ct, reference in [
        ("add", ct_sum, x + y),
        (
            "NTT + multiplication + relinearization + rescale",
            ct_product,
            x * y,
        ),
        ("rotate(+1)", ct_rotated, torch.roll(x, shifts=1, dims=0)),
    ]:
        error = error_stats(engine.decrypt_message(ct), reference, slots)
        rows.append(
            [name, ct.level, f"{error['max_abs']:.3e}", f"{error['rms']:.3e}"]
        )

    print(engine)
    print_table(
        ["operation", "output level", "max abs error", "rms error"], rows
    )
    print(
        "Unbatched ciphertext layout: "
        f"[component, limb, coeff]={tuple(ct_x.data.shape)}, "
        f"batch_shape={tuple(ct_x.batch_shape)}, "
        f"prime_ids={ct_x.prime_ids}, bytes={ct_x.data.nbytes}"
    )


if __name__ == "__main__":
    main()
