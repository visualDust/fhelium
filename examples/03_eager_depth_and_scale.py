#!/usr/bin/env python3

"""Inspect the modulus chain of a preset, then plan and track per-value CKKS scales.

The first table lists the active Q basis, default scale, and rescale ratio of
selected depths. The second part allocates the scales of two plaintext factors
so that their product, after one rescale, reaches a target scale; aligns a
second branch by depth with a modulus switch; and compares the result with a
power-of-two allocation followed by a bounded scale reinterpretation.

Run:
    python examples/03_eager_depth_and_scale.py --preset slots8192-scale40-depth7
"""

from __future__ import annotations

import argparse
import math

import torch
from common import (
    add_engine_args,
    error_stats,
    format_bytes,
    make_engine,
    print_table,
    sync_if_cuda,
)

import fhelium as fh
from fhelium.eager import Engine


def chain_rows(engine: Engine, depths: list[int]) -> list[list[object]]:
    """Return one display row per depth of the engine's modulus chain."""

    config = engine.config
    rows = []
    for depth in depths:
        if depth < config.max_depth:
            dropped, introduced = config.basis_transition(depth, depth + 1)
            ratio = f"{math.log2(config.rescale_divisor(depth)):.2f}"
        else:
            dropped, introduced, ratio = (), (), "-"
        ciphertext = engine.encrypt_message([1, 2, 3, 4], depth=depth)
        rows.append(
            [
                depth,
                config.depth_remaining(depth),
                len(config.q_prime_ids(depth)),
                f"{math.log2(math.prod(config.active_q_moduli(depth))):.1f}",
                f"{math.log2(config.scale_at(depth)):.2f}",
                f"{len(dropped)}/{len(introduced)}",
                ratio,
                format_bytes(ciphertext.data.nbytes),
            ]
        )
    return rows


def multiply_twice_then_rescale(
    engine: Engine,
    source: fh.Ciphertext,
    first_message: torch.Tensor,
    second_message: torch.Tensor,
    *,
    first_scale: float,
    second_scale: float,
) -> tuple[fh.Ciphertext, fh.Ciphertext, fh.Ciphertext]:
    r"""Apply two plaintext products at the given scales, then one rescale.

    The actual-scale path is

    $$
    \Delta(c_1)=\Delta(c_0)\Delta(p_1),\qquad
    \Delta(c_2)=\Delta(c_1)\Delta(p_2),\qquad
    \Delta(c_3)=\frac{\Delta(c_2)}{\rho_d},
    \qquad \rho_d=\frac{\prod_{q\in D_d}q}{\prod_{q\in R_d}q}.
    $$

    Decoded slots satisfy $m_3\approx m_0m_1m_2$ up to CKKS approximation
    error. The two products keep depth $d$; the rescale moves to $d+1$.

    Returns:
        Ciphertexts after the first product, second product, and rescale.
    """

    first = engine.prepare_plaintext_for_multiplication(
        engine.encode(first_message, depth=source.depth, scale=first_scale)
    )
    second = engine.prepare_plaintext_for_multiplication(
        engine.encode(second_message, depth=source.depth, scale=second_scale)
    )
    after_first = engine.multiply_plaintext(
        engine.coefficient_domain_to_ntt_domain(source), first
    )
    before_rescale = engine.multiply_plaintext(after_first, second)
    return (
        after_first,
        before_rescale,
        engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(before_rescale)
        ),
    )


def scale_row(label: str, value: fh.Ciphertext) -> list[object]:
    """Return one display row for a ciphertext's depth and recorded scale."""

    return [
        label,
        value.depth,
        f"{value.scale:.17g}",
        f"{math.log2(value.scale):.9f}",
    ]


def error_row(
    label: str, engine: Engine, value: fh.Ciphertext, expected: torch.Tensor
) -> list[object]:
    """Decrypt one result and return its approximation error."""

    error = error_stats(
        engine.decrypt_message(value).cpu(), expected, engine.num_slots
    )
    return [label, f"{error['max_abs']:.3e}", f"{error['rms']:.3e}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-depth7")
    parser.add_argument(
        "--depths",
        default=None,
        help="Comma-separated depths for the chain table. Default: first, middle, and last depths.",
    )
    parser.add_argument(
        "--reinterpret-bound",
        type=float,
        default=5e-2,
        help="Maximum symmetric relative scale change accepted by the reinterpretation.",
    )
    args = parser.parse_args()

    engine = make_engine(args)
    config = engine.config

    # Part 1: the modulus chain. Each depth selects a set of Q primes; a
    # rescale from depth d drops D_d, introduces R_d, and divides by rho_d.
    depths = (
        [int(item) for item in args.depths.split(",")]
        if args.depths
        else sorted(
            {
                0,
                1,
                config.max_depth // 2,
                config.max_depth - 1,
                config.max_depth,
            }
        )
    )
    print(
        f"Preset {args.preset}: max depth {config.max_depth}, "
        f"{config.num_q_primes} Q and {config.num_p_primes} P primes, "
        f"log2(QP)={config.total_modulus_bits} of "
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
        chain_rows(engine, depths),
    )

    # Part 2: per-value scales.
    slots = engine.num_slots
    message = 0.01 * torch.sin(torch.arange(slots, dtype=torch.float64) * 0.013)
    first_message = torch.linspace(0.8, 1.2, slots, dtype=torch.float64)
    second_message = torch.linspace(1.1, 0.9, slots, dtype=torch.float64)
    expected_product = message * first_message * second_message

    source = engine.encrypt_message(message)
    rescale_ratio = engine.rescale_divisor(depth=source.depth)

    # Depth alignment is independent of scale alignment: the second branch
    # reaches the next depth by a modulus switch, and the planned product
    # targets the scale that switch produces.
    depth_aligned_source = engine.mod_switch_to_depth(source, source.depth + 1)
    target_scale = depth_aligned_source.scale

    # The first factor receives a precision allocation; the second completes
    # the product that reaches the target after division by rho_0.
    first_scale = float(1 << (math.floor(math.log2(target_scale)) // 2))
    second_scale = (
        target_scale * float(rescale_ratio) / (source.scale * first_scale)
    )
    planned_first, planned_pre, planned = multiply_twice_then_rescale(
        engine,
        source,
        first_message,
        second_message,
        first_scale=first_scale,
        second_scale=second_scale,
    )
    predicted_scale = engine.rescale_output_scale(
        planned_pre.scale, depth=planned_pre.depth
    )
    planned_sum = engine.add(planned, depth_aligned_source)

    # The comparison allocation replaces rho_0 by the nearest power of two.
    # Rescale divides by the actual rho_0, so this result's scale differs
    # from the target and the branches cannot be added directly.
    nominal_ratio = 2.0 ** round(math.log2(rescale_ratio))
    approximate_second_scale = (
        target_scale * nominal_ratio / (source.scale * first_scale)
    )
    _, approximate_pre, approximate = multiply_twice_then_rescale(
        engine,
        source,
        first_message,
        second_message,
        first_scale=first_scale,
        second_scale=approximate_second_scale,
    )

    # Reinterpretation records the target scale and keeps the residues; the
    # bound limits the accepted change of the decoded message.
    reinterpreted = engine.reinterpret_at_scale(
        approximate, target_scale, max_relative_change=args.reinterpret_bound
    )
    reinterpreted_sum = engine.add(reinterpreted, depth_aligned_source)
    sync_if_cuda(torch.get_default_device())

    print(
        f"\ndepth-{source.depth} rescale ratio log2: {math.log2(rescale_ratio):.6f}"
    )
    print(f"target scale at depth {planned.depth}: {target_scale:.17g}")
    print(f"predicted scale after rescale: {predicted_scale:.17g}")
    print(f"plaintext scales: p1={first_scale:.17g}, p2={second_scale:.17g}")
    print("\nScale states")
    print_table(
        ["value", "depth", "scale", "log2(scale)"],
        [
            scale_row("source", source),
            scale_row("planned after product 1", planned_first),
            scale_row("planned before rescale", planned_pre),
            scale_row("planned after rescale", planned),
            scale_row("depth-switched source", depth_aligned_source),
            scale_row("power-of-two before rescale", approximate_pre),
            scale_row("power-of-two after rescale", approximate),
            scale_row("reinterpreted", reinterpreted),
        ],
    )
    print(
        "\nRelative scale difference of the power-of-two result: "
        f"{approximate.scale / target_scale - 1.0:.9e}"
    )

    # Reinterpretation multiplies the decoded product by old_scale/new_scale.
    reinterpret_ratio = approximate.scale / reinterpreted.scale
    print("\nDecoded error")
    print_table(
        ["result", "max abs error", "rms error"],
        [
            error_row("planned product", engine, planned, expected_product),
            error_row(
                "power-of-two product", engine, approximate, expected_product
            ),
            error_row(
                "planned product + depth-switched source",
                engine,
                planned_sum,
                expected_product + message,
            ),
            error_row(
                "reinterpreted product + source",
                engine,
                reinterpreted_sum,
                expected_product * reinterpret_ratio + message,
            ),
        ],
    )


if __name__ == "__main__":
    main()
