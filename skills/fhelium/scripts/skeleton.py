#!/usr/bin/env python3

"""Minimal FHElium program: verify a CKKS computation in Eager, then compile it.

Computes ``y = (x + roll(x, 1)) * w`` squared, decrypts it, and compares it with
the same computation on clear values. Replace ``reference`` and ``layer`` with the
target computation; keep the comparison.
"""

from __future__ import annotations

import argparse
import time

import torch

import fhelium as fh
from fhelium import compile as fc
from fhelium.eager import Engine


def reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ((x + torch.roll(x, 1)) * w) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="slots8192-scale40-depth7")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # The Engine owns the CKKS context; values live on the device they are created on.
    engine = Engine(fh.CkksConfig.parse(fh.Preset(args.preset)))
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    # Every key the computation uses is created here and passed in.
    rotation_key = engine.create_rotation_key(1, secret_key, device=device)
    relinearization_key = engine.create_relinearization_key(
        secret_key, device=device
    )

    slots = engine.num_slots
    x_clear = torch.linspace(-0.5, 0.5, slots, dtype=torch.float64)
    # A weight with a short period is stored compressed and tiled across all slots.
    w_period = torch.linspace(0.1, 0.9, 16, dtype=torch.float64)
    x = engine.encrypt_message(x_clear, public_key, device=device)

    def layer(value):
        # depth d -> d + 2: one rescale after the plaintext product, one after the square.
        plain = engine.prepare_compressed_plaintext(
            w_period, depth=value.depth, device=value.device
        )
        rotated = engine.add(value, engine.rotate_with_key(value, rotation_key))
        product = engine.multiply_plaintext(
            engine.coefficient_domain_to_ntt_domain(rotated), plain
        )
        hidden = engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(product)
        )
        ntt = engine.coefficient_domain_to_ntt_domain(hidden)
        return engine.relinearize(
            engine.multiply(ntt, ntt), relinearization_key, rescale=True
        )

    expected = reference(x_clear, w_period.repeat(slots // w_period.numel()))
    eager = layer(x)
    compiled = fc.compile(layer)

    start = time.perf_counter()
    first = compiled(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    first_ms = 1e3 * (time.perf_counter() - start)
    del first
    for _ in range(3):
        compiled(x)
    runs = 10
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(runs):
        result = compiled(x)
        del result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    steady_ms = 1e3 * (time.perf_counter() - start) / runs
    result = compiled(x)

    print(f"preset {args.preset}, device {device}, slots {slots}")
    print(
        f"depth {x.depth} -> {result.depth}, "
        f"remaining rescales {engine.depth_remaining(result)}, scale {result.scale:.6e}"
    )
    for name, value in (("eager", eager), ("compiled", result)):
        decoded = engine.decrypt_message(value, secret_key, is_real=True).cpu()
        error = (decoded - expected).abs()
        print(
            f"{name:9s} max abs error {error.max().item():.3e}, mean {error.mean().item():.3e}"
        )
    print(f"compiled first call {first_ms:.1f} ms, steady {steady_ms:.3f} ms")
    if device.type == "cuda":
        print(
            f"peak allocated {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB"
        )
    compiled.clear()


if __name__ == "__main__":
    main()
