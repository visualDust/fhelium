#!/usr/bin/env python3

"""Evaluate a linear layer and a Chebyshev polynomial activation, Eagerly and compiled.

The layer computes ``y = W x`` on a packed vector with
``fhelium.algorithms.linear`` and then ``z = p(y)`` slot-wise with
``fhelium.algorithms.polynomial``, where ``p`` is a Chebyshev interpolant of
the GELU activation on ``[-1, 1]``. The same Python function runs directly on
the Engine and under ``@fc.compile``.

Run:
    python examples/08_algorithms_linear_polynomial.py
    python examples/08_algorithms_linear_polynomial.py --device cuda:0
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from common import (
    add_engine_args,
    error_stats,
    format_bytes,
    make_engine,
    print_table,
    time_ms,
)

from fhelium import compile as fc
from fhelium.algorithms import linear, polynomial
from fhelium.values import EvaluationKeySet


def gelu(t: np.ndarray) -> np.ndarray:
    """Return the tanh form of GELU, the activation approximated by ``p``."""

    return (
        0.5
        * t
        * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (t + 0.044715 * t**3)))
    )


def weight_diagonals(weight: np.ndarray, slots: int) -> dict[int, np.ndarray]:
    r"""Return the cyclic diagonals of ``weight`` for a periodically packed vector.

    With slots holding $x_{j \bmod n}$, rotation by $k$ (``torch.roll`` with
    ``shifts=k``) places $x_{(i-k) \bmod n}$ in slot $i$, so
    $d_k[i]=W_{i \bmod n,\,(i-k) \bmod n}$ gives
    $\sum_k d_k\odot\operatorname{Rot}_k(x)=Wx$ in every period.
    """

    n = weight.shape[0]
    rows = np.arange(slots) % n
    return {
        step: weight[rows, (rows - step) % n].astype(np.complex128)
        for step in range(n)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-depth7")
    parser.add_argument(
        "--features", type=int, default=16, help="Width n of the n x n weight."
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=15,
        help="Chebyshev degree of the activation.",
    )
    args = parser.parse_args()

    engine = make_engine(args)
    slots = engine.num_slots
    n = args.features
    generator = np.random.default_rng(8)

    # Rows of W have unit L1 norm, so |W x| <= 1 for |x| <= 1: the layer
    # output stays inside the activation's interpolation interval.
    weight = generator.uniform(-1.0, 1.0, (n, n))
    weight /= np.abs(weight).sum(axis=1, keepdims=True)
    features = generator.uniform(-1.0, 1.0, n)
    expected = gelu(weight @ features)

    transform = linear.LinearTransform(
        weight_diagonals(weight, slots), slots, name="dense"
    )
    linear_plan = linear.plan(transform)
    activation = polynomial.chebyshev_fit(gelu, args.degree, name="gelu")
    activation_plan = polynomial.plan(activation)

    # Key provisioning: the linear plan reports its rotation steps and the
    # polynomial needs one relinearization key.
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    steps = linear_plan.required_rotation_offsets(transform)
    keys = EvaluationKeySet(
        [engine.create_rotation_key(step, secret_key) for step in steps]
        + [engine.create_relinearization_key(secret_key)]
    )

    x = engine.encrypt_message(
        torch.as_tensor(np.resize(features, slots), dtype=torch.float64),
        public_key,
    )

    # Encode the diagonals and select keys once for inputs with x's state.
    prepared = linear.prepare(engine, x, transform, keys=keys)

    def layer(value):
        hidden = linear.apply(engine, value, prepared=prepared)
        return polynomial.evaluate(
            engine, hidden, activation, activation_plan, keys=keys
        )

    compiled_layer = fc.compile(layer)

    hidden = linear.apply(engine, x, prepared=prepared)
    eager = polynomial.evaluate(
        engine, hidden, activation, activation_plan, keys=keys
    )
    compiled = compiled_layer(x)

    print(
        f"Preset {args.preset}, {n} x {n} weight, {len(steps)} rotation keys "
        f"({format_bytes(sum(key.data.nbytes for key in keys))} of evaluation keys)"
    )
    print(
        f"BSGS baby step {linear_plan.baby_step}; activation degree {activation.degree}, "
        f"sampled interpolation error {activation.max_error:.2e}"
    )
    print_table(
        ["value", "depth", "log2 scale"],
        [
            ["input", x.depth, f"{math.log2(x.scale):.3f}"],
            ["W x", hidden.depth, f"{math.log2(hidden.scale):.3f}"],
            ["p(W x), Eager", eager.depth, f"{math.log2(eager.scale):.3f}"],
            [
                "p(W x), compiled",
                compiled.depth,
                f"{math.log2(compiled.scale):.3f}",
            ],
        ],
    )
    print(
        f"depths consumed: linear {linear_plan.required_depths(transform)}, "
        f"activation {activation_plan.required_depths(activation)}"
    )

    rows = []
    for label, value in (("Eager", eager), ("compiled", compiled)):
        decoded = engine.decrypt_message(value, secret_key, is_real=True).cpu()
        error = error_stats(decoded[:n], torch.as_tensor(expected))
        rows.append([label, f"{error['max_abs']:.3e}", f"{error['rms']:.3e}"])
    print_table(["path", "max abs error", "rms error"], rows)

    device = torch.get_default_device()
    eager_time, _ = time_ms(lambda: layer(x), device=device)
    compiled_time, _ = time_ms(lambda: compiled_layer(x), device=device)
    print_table(
        ["path", "mean ms", "min ms"],
        [
            [
                "Eager",
                f"{eager_time['mean_ms']:.2f}",
                f"{eager_time['min_ms']:.2f}",
            ],
            [
                "compiled",
                f"{compiled_time['mean_ms']:.2f}",
                f"{compiled_time['min_ms']:.2f}",
            ],
        ],
    )


if __name__ == "__main__":
    main()
