#!/usr/bin/env python3

"""Compare exact Q chains with different maximum public depths.

Run:
    python examples/04_modulus_chain_depth.py \
        --preset slots32768-scale40-depth34-int64
"""

from __future__ import annotations

import argparse
import math

import torch

from common import add_engine_args, format_bytes, parse_preset, print_table

from fhelium.config import CkksConfig
from fhelium.eager import Engine


def selected_depths(config: CkksConfig) -> list[int]:
    """Return representative maximum depths for one source parameter set."""

    candidates = [
        max(0, config.max_depth // 2),
        max(0, (config.max_depth * 3) // 4),
        config.max_depth,
    ]
    return sorted(set(candidates))


def prefix_config(source: CkksConfig, max_depth: int) -> CkksConfig:
    """Retain the requested public Q-group prefix and terminal group."""

    if not 0 <= max_depth <= source.max_depth:
        raise ValueError(f"max_depth must be in [0, {source.max_depth}]")
    return CkksConfig(
        default_scale=source.default_scale,
        q_depth_groups=(
            *source.q_depth_groups[:max_depth],
            source.q_depth_groups[-1],
        ),
        p_moduli=source.p_moduli,
        logN=source.logN,
        sigma=source.sigma,
        security_bits=source.security_bits,
        enforce_security_budget=source.enforce_security_budget,
        galois_generator=source.galois_generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots32768-scale40-depth34-int64")
    parser.add_argument(
        "--depths",
        default=None,
        help="Comma-separated max_depth values. Default: middle and full variants.",
    )
    args = parser.parse_args()
    torch.set_default_device(args.device)

    source = CkksConfig.parse(parse_preset(args.preset))
    depths = (
        [int(item) for item in args.depths.split(",")]
        if args.depths
        else selected_depths(source)
    )

    rows = []
    for max_depth in depths:
        config = prefix_config(source, max_depth)
        engine = Engine(config, ntt_backend=args.ntt_backend)
        ciphertext = engine.encrypt_message([1, 2, 3, 4], depth=0)
        rows.append(
            [
                max_depth,
                config.num_q_primes,
                config.num_p_primes,
                config.total_modulus_bits,
                config.maximum_modulus_bits,
                str(engine.dtype).removeprefix("torch."),
                format_bytes(ciphertext.data.nbytes),
            ]
        )

    print(
        f"Source preset {args.preset}: "
        f"log2(default_scale)={math.log2(source.default_scale):.3f}, "
        f"terminal Q rows={len(source.q_depth_groups[-1])}."
    )
    print_table(
        [
            "max depth",
            "Q rows",
            "P rows",
            "QP bits",
            "security budget bits",
            "RNS dtype",
            "depth-0 CT size",
        ],
        rows,
    )
    print(
        "\nOne public rescale consumes one Q depth group. Ciphertext size and "
        "native work follow the active prime-row count inside those groups."
    )


if __name__ == "__main__":
    main()
