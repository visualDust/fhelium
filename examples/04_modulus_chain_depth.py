#!/usr/bin/env python3

"""Explore modulus-chain depth, modulus bits, and level-dependent sizes.

Run:
    python examples/04_modulus_chain_depth.py --preset slots32768-scale40-levels34-int64
"""

from __future__ import annotations

import argparse

from common import add_engine_args, format_bytes, parse_preset, print_table

from fhelium import CkksEngine
from fhelium.config import CkksConfig


def variant_depths(preset_name: str) -> list[int]:
    default_cfg = CkksConfig.parse(parse_preset(preset_name))
    full = default_cfg.num_scale_primes
    # Low/mid/full defaults, clipped and de-duplicated.
    candidates = [max(1, full // 2), max(1, (full * 3) // 4), full]
    return sorted(set(candidates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots32768-scale40-levels34-int64")
    parser.add_argument(
        "--depths",
        default=None,
        help=(
            "Comma-separated num_scale_primes values. "
            "Default: low/mid/full for the preset."
        ),
    )
    args = parser.parse_args()

    preset = parse_preset(args.preset)
    depths = (
        [int(x) for x in args.depths.split(",")]
        if args.depths
        else variant_depths(args.preset)
    )

    rows = []
    for depth in depths:
        cfg = CkksConfig.parse(preset, num_scale_primes=depth)
        engine = CkksEngine(
            cfg,
            device=args.device,
            ntt_backend=args.ntt_backend,
        )
        ct0 = engine.encrypt_message([1, 2, 3, 4], level=0)
        # A ciphertext at level l stores the active Q_l scale rows plus the
        # base Q row.  Level 0 is therefore the largest ciphertext.
        rows.append(
            [
                depth,
                cfg.total_modulus_bits,
                cfg.maximum_modulus_bits,
                cfg.num_q_primes,
                cfg.num_p_primes,
                cfg.total_num_primes,
                format_bytes(ct0.data.nbytes),
                f"{ct0.data.nbytes / 1e6:.3f}",
            ]
        )

    default_cfg = CkksConfig.parse(preset)
    print(
        f"Preset {args.preset}: scale_bits={default_cfg.scale_bits} by "
        "default; num_scale_primes is the public-level count."
    )
    print_table(
        [
            "scale primes/public levels",
            "QP modulus bits",
            "security budget bits",
            "Q primes",
            "P primes",
            "total primes",
            "level-0 ct size",
            "level-0 ct MB",
        ],
        rows,
    )
    print(f"\nRule of thumb for the selected {default_cfg.torch_dtype} preset:")
    base_bits = default_cfg.base_prime_bits or default_cfg.message_bits
    print(
        "  total_modulus_bits ~= num_scale_primes * "
        f"{default_cfg.scale_bits} + {base_bits}(base) + "
        f"{default_cfg.message_bits}*num_p_primes"
    )
    print(
        "  current ciphertext size follows the active Q_l rows at the current level, not just the initial chain length."
    )


if __name__ == "__main__":
    main()
