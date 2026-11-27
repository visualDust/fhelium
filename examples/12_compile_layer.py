#!/usr/bin/env python3

"""Compile an encrypted dense layer with a square activation.

The layer is a class whose ``forward`` method is compiled per instance. It applies a prepared cyclic-diagonal linear transform from ``fhelium.algorithms.linear``, calls the compiled ``activate`` method, which is inlined into the caller, and multiplies by a Python ``float`` gain, which selects one specialization per value. The example then reads the callable's preparation, cache, execution, and CUDA Graph statistics.
"""

from __future__ import annotations

import argparse

import torch
from common import print_table, time_ms

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.algorithms import linear
from fhelium.eager import Engine
from fhelium.values import Ciphertext, EvaluationKeySet, RelinearizationKey


def dense_transform(
    weights: torch.Tensor, slots: int
) -> linear.LinearTransform:
    """Return ``x -> W x`` for ``x`` tiled with period ``W.size(0)`` across ``slots``."""

    size = weights.size(0)
    rows = torch.arange(slots) % size
    # Diagonal -k pairs output slot j with input slot j + k, which
    # torch.roll(x, shifts=-k) moves to slot j.
    return linear.LinearTransform(
        {-k: weights[rows, (rows + k) % size] for k in range(size)},
        slots,
        name="dense",
    )


class EncryptedLayer:
    """``y = gain * (W x)^2`` for ``x`` tiled with period ``W.size(0)`` across all CKKS slots."""

    def __init__(
        self,
        engine: Engine,
        transform: linear.LinearTransform,
        like: Ciphertext,
        keys: EvaluationKeySet,
        relinearization_key: RelinearizationKey,
    ) -> None:
        self.engine = engine
        self.relinearization_key = relinearization_key
        # Diagonals are encoded once and read by every compiled call.
        self.prepared = linear.prepare(engine, like, transform, keys=keys)

    @fc.compile
    def activate(self, value: Ciphertext) -> Ciphertext:
        engine = self.engine
        ntt = engine.coefficient_domain_to_ntt_domain(value)
        return engine.relinearize(
            engine.multiply(ntt, ntt), self.relinearization_key, rescale=True
        )

    @fc.compile
    def forward(self, x: Ciphertext, gain: float = 1.0) -> Ciphertext:
        hidden = linear.apply(self.engine, x, prepared=self.prepared)
        return self.engine.multiply_constant(self.activate(hidden), gain)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)

    config = CkksConfig.parse(Preset.slots8192_scale40_depth7)
    engine = Engine(config, rng_seed=24)
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    relinearization_key = engine.create_relinearization_key(
        secret_key, device=device
    )

    size = args.size
    generator = torch.Generator().manual_seed(12)
    clear_x = 0.5 * torch.rand(size, generator=generator, dtype=torch.float64)
    tiled_x = clear_x.repeat(config.num_slots // size).to(device)
    x = engine.encrypt_message(tiled_x, public_key, device=device)

    weights = [
        0.2 * torch.randn(size, size, generator=generator, dtype=torch.float64)
        for _ in range(2)
    ]
    transforms = [dense_transform(w, config.num_slots) for w in weights]
    steps = {
        step
        for transform in transforms
        for step in linear.plan(transform).required_rotation_offsets(transform)
    }
    keys = EvaluationKeySet(
        engine.create_rotation_key(step, secret_key, device=device)
        for step in sorted(steps)
    )
    layers = [
        EncryptedLayer(engine, transform, x, keys, relinearization_key)
        for transform in transforms
    ]

    rows = []
    for index, (layer, w) in enumerate(zip(layers, weights, strict=True)):
        for gain in (1.0, 0.5):
            result = layer.forward(x, gain)
            decoded = engine.decrypt_message(result, secret_key, is_real=True)
            expected = gain * (w @ clear_x) ** 2
            error = float((decoded[:size].cpu() - expected).abs().max())
            rows.append(
                [
                    index,
                    gain,
                    result.depth,
                    f"{result.scale:.6e}",
                    f"{error:.3e}",
                ]
            )
    print_table(["layer", "gain", "depth", "scale", "max abs error"], rows)

    layer = layers[0]
    steady, _ = time_ms(
        lambda: layer.forward(x, 1.0), warmup=3, runs=args.runs, device=device
    )
    stats = layer.forward.stats()
    print()
    print_table(
        ["statistic", "value"],
        [
            ["specializations of layer 0", len(layer.forward.specializations)],
            [
                "specializations of layer 1",
                len(layers[1].forward.specializations),
            ],
            ["preparation seconds", f"{stats['prepare_seconds']:.3f}"],
            ["steady call mean ms", f"{steady['mean_ms']:.3f}"],
            ["resident cache hits", stats["resident_cache_hits"]],
            ["executions", stats["executions"]],
            ["source cache", stats["cache"] or "eligible"],
        ],
    )
    for record in stats["graphs"]:
        segments = record["segments"]
        print(
            "Graph route:",
            record["state"],
            "replays:",
            record["replays"],
            "segments:",
            [segment["choice"] for segment in segments],
            "fallback:",
            record["fallback_reason"],
        )


if __name__ == "__main__":
    main()
