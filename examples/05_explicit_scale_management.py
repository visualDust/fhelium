#!/usr/bin/env python3

"""Plan and track actual per-value CKKS scales across two plaintext products.

Run:
    python examples/05_explicit_scale_management.py --preset slots8192-scale40-levels7-int64
"""

from __future__ import annotations

import argparse
import math

import torch
from common import (
    add_engine_args,
    error_stats,
    make_engine,
    print_table,
    sync_if_cuda,
)

import fhelium as fh


def multiply_twice_then_rescale_to_next_level(
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    first_message: torch.Tensor,
    second_message: torch.Tensor,
    *,
    first_scale: float,
    second_scale: float,
) -> tuple[fh.Ciphertext, fh.Ciphertext, fh.Ciphertext]:
    r"""Apply two planned plaintext products followed by rescale-to-next.

    The actual-scale path is

    $$
    \Delta(c_1)=\Delta(c_0)\Delta(p_1),\qquad
    \Delta(c_2)=\Delta(c_1)\Delta(p_2),\qquad
    \Delta(c_3)=\frac{\Delta(c_2)}{q_{\mathrm{drop}}}.
    $$

    Decoded slots satisfy $m_3\mathrel{\approx}m_0m_1m_2$ up to CKKS
    approximation error. Inputs are unchanged; all three returned ciphertexts
    own independent storage.

    Args:
        engine: Rank-local CKKS engine.
        source: Two-component coefficient-domain ciphertext.
        first_message: First cleartext slot-wise factor.
        second_message: Second cleartext slot-wise factor.
        first_scale: Encoding scale allocated to the first factor.
        second_scale: Encoding scale allocated to the second factor.

    Returns:
        Ciphertexts after the first product, second product, and rescale.
    """

    first = engine.prepare_plaintext_for_multiplication(
        engine.encode(first_message, level=source.level, scale=first_scale)
    )
    second = engine.prepare_plaintext_for_multiplication(
        engine.encode(second_message, level=source.level, scale=second_scale)
    )
    after_first = engine.multiply_plaintext(
        engine.coefficient_domain_to_ntt_domain(source), first
    )
    before_rescale = engine.multiply_plaintext(after_first, second)
    return (
        after_first,
        before_rescale,
        engine.rescale_to_next_level(
            engine.ntt_domain_to_coefficient_domain(before_rescale)
        ),
    )


def scale_row(label: str, value: fh.Ciphertext) -> list[object]:
    """Return one display row for a ciphertext's recorded scale state."""

    return [
        label,
        value.level,
        f"{value.scale:.17g}",
        f"{math.log2(value.scale):.9f}",
    ]


def error_row(
    label: str,
    engine: fh.CkksEngine,
    value: fh.Ciphertext,
    expected: torch.Tensor,
) -> list[object]:
    """Decrypt one result and return compact approximation-error statistics."""

    error = error_stats(
        engine.decrypt_message(value),
        expected,
        engine.num_slots,
    )
    return [label, f"{error['max_abs']:.3e}", f"{error['rms']:.3e}"]


def main() -> None:
    """Run the scale-planning and guarded-reinterpretation example."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-levels7-int64")
    parser.add_argument(
        "--reinterpret-bound",
        type=float,
        default=1e-2,
        help=(
            "Maximum symmetric relative scale change accepted by the guarded "
            "reinterpretation."
        ),
    )
    args = parser.parse_args()

    engine = make_engine(args)
    slots = engine.num_slots
    default_scale = engine.config.default_scale

    message = 0.01 * torch.sin(torch.arange(slots, dtype=torch.float64) * 0.013)
    first_message = torch.linspace(0.8, 1.2, slots, dtype=torch.float64)
    second_message = torch.linspace(1.1, 0.9, slots, dtype=torch.float64)
    expected_product = message * first_message * second_message

    source = engine.encrypt_message(message, scale=default_scale)
    dropped_prime = engine.rescale_to_next_drop_prime(level=source.level)

    # The first factor receives a precision allocation. The second
    # scale completes the product required to reach default_scale after
    # division by the actual q_0.
    first_scale = float(1 << (engine.config.scale_bits // 2))
    second_scale = default_scale * dropped_prime / (source.scale * first_scale)
    planned_first, planned_pre, planned = (
        multiply_twice_then_rescale_to_next_level(
            engine,
            source,
            first_message,
            second_message,
            first_scale=first_scale,
            second_scale=second_scale,
        )
    )
    predicted_scale = engine.rescale_to_next_output_scale(
        planned_pre.scale,
        level=planned_pre.level,
    )
    assert planned.scale == predicted_scale == default_scale

    # Level alignment is independent of scale alignment. Because the planned
    # branch reaches default_scale exactly, a level-only modulus switch makes
    # the original branch directly add-compatible.
    level_aligned_source = engine.mod_switch_to_level(source, planned.level)
    planned_sum = engine.add(planned, level_aligned_source)

    # The comparison allocation uses a plaintext-scale product of Delta.
    # Rescale records Delta^2/q_0, so the following addition has unequal scales.
    approximate_second_scale = default_scale / first_scale
    approximate_first, approximate_pre, approximate = (
        multiply_twice_then_rescale_to_next_level(
            engine,
            source,
            first_message,
            second_message,
            first_scale=first_scale,
            second_scale=approximate_second_scale,
        )
    )
    try:
        engine.add(approximate, level_aligned_source)
    except fh.errors.ScaleMismatchError as error:
        strict_add_diagnostic = str(error)
    else:
        raise RuntimeError("strict addition accepted unequal scales")

    # Guarded reinterpretation records the configured target scale while
    # preserving residues. The bound limits the accepted message bias.
    reinterpreted = engine.reinterpret_at_scale(
        approximate,
        default_scale,
        max_relative_change=args.reinterpret_bound,
    )
    reinterpreted_sum = engine.add(reinterpreted, level_aligned_source)
    sync_if_cuda(engine.device)

    print(engine)
    print(f"default scale Delta: {default_scale:.17g}")
    print(f"level-0 rescale prime q0: {dropped_prime}")
    print(
        f"allocated plaintext scales: p1={first_scale:.17g}, p2={second_scale:.17g}"
    )
    print("\nExplicit scale states")
    print_table(
        ["value", "level", "scale", "log2(scale)"],
        [
            scale_row("source", source),
            scale_row("planned after pmult 1", planned_first),
            scale_row("planned before rescale", planned_pre),
            scale_row("planned after rescale", planned),
            scale_row("level-only switched source", level_aligned_source),
            scale_row("Delta-product before rescale", approximate_pre),
            scale_row("actual Delta^2/q0 result", approximate),
            scale_row("explicitly reinterpreted", reinterpreted),
        ],
    )

    print("\nStrict-add diagnostic before explicit reinterpretation")
    print(strict_add_diagnostic)

    # Reinterpretation changes the decoded product by old_scale/new_scale.
    reinterpret_ratio = approximate.scale / reinterpreted.scale
    print("\nCleartext error")
    print_table(
        ["result", "max abs error", "rms error"],
        [
            error_row("planned product", engine, planned, expected_product),
            error_row(
                "actual-scale Delta-product",
                engine,
                approximate,
                expected_product,
            ),
            error_row(
                "planned product + level-switched source",
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
