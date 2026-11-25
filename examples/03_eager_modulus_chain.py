#!/usr/bin/env python3

"""Inspect the per-depth Q bases, default scales, and rescale ratios of a preset.

Run:
    python examples/03_eager_modulus_chain.py \
        --preset slots32768-scale40-depth34
"""

from __future__ import annotations

import argparse
import math

import torch

from common import add_engine_args, format_bytes, parse_preset, print_table

from fhelium.config import CkksConfig
from fhelium.eager import Engine


def selected_depths(config: CkksConfig) -> list[int]:
    """Return representative depths of one configuration."""

    candidates = [
        0,
        1,
        config.max_depth // 2,
        config.max_depth - 1,
        config.max_depth,
    ]
    return sorted({depth for depth in candidates if depth >= 0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots32768-scale40-depth34")
    parser.add_argument(
        "--depths",
        default=None,
        help="Comma-separated depths. Default: first, middle, and last depths.",
    )
    args = parser.parse_args()
    torch.set_default_device(args.device)

    config = CkksConfig.parse(parse_preset(args.preset))
    engine = Engine(config, ntt_backend=args.ntt_backend)
    depths = (
        [int(item) for item in args.depths.split(",")]
        if args.depths
        else selected_depths(config)
    )

    rows = []
    for depth in depths:
        dropped, restored = (
            config.basis_transition(depth, depth + 1)
            if depth < config.max_depth
            else ((), ())
        )
        ratio = (
            f"{math.log2(config.rescale_divisor(depth)):.2f}"
            if depth < config.max_depth
            else "-"
        )
        ciphertext = engine.encrypt_message([1, 2, 3, 4], depth=depth)
        rows.append(
            [
                depth,
                config.depth_remaining(depth),
                len(config.q_prime_ids(depth)),
                f"{math.log2(math.prod(config.active_q_moduli(depth))):.1f}",
                f"{math.log2(config.scale_at(depth)):.2f}",
                f"{len(dropped)}/{len(restored)}",
                ratio,
                format_bytes(ciphertext.data.nbytes),
            ]
        )

    print(
        f"Preset {args.preset}: max depth {config.max_depth}, "
        f"{config.num_q_primes} Q and {config.num_p_primes} P primes, "
        f"log2(QP)={config.total_modulus_bits:.1f} of "
        f"{config.maximum_modulus_bits} security budget bits, "
        f"RNS dtype {str(engine.dtype).removeprefix('torch.')}."
    )
    print_table(
        [
            "depth",
            "remaining",
            "Q rows",
            "log2 B_d",
            "log2 scale",
            "|D|/|R|",
            "log2 rho_d",
            "CT size",
        ],
        rows,
    )
    print(
        "\nA rescale from depth d divides by rho_d = D/R, the product of the "
        "primes leaving the basis over the product of the primes entering it. "
        "Ciphertext size and native work follow the active row count of each basis."
    )


if __name__ == "__main__":
    main()
