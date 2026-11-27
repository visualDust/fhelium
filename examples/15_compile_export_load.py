#!/usr/bin/env python3

"""Export a compiled function to one file and call it in another process.

The client process holds the secret key. It compiles an encrypted layer, exports it with ``fhelium.compile.export``, and writes the evaluation keys and an encrypted input as value files. A separate server process loads the artifact with ``fhelium.compile.load``, binds the evaluation keys from their files, calls the layer, and writes the encrypted result, which the client decrypts. The same artifact, key files, and input file are accepted by ``examples/cpp/load_artifact``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from common import print_table

import fhelium as fh
from fhelium import compile as fc
from fhelium.eager import Engine
from fhelium.values import Ciphertext

_SIZE = 16


def serve(directory: Path, device: str) -> None:
    """Load the artifact and evaluate the stored input; no secret key is present."""

    started = time.perf_counter()
    keys = [
        fh.load_value(path, device=device)
        for path in sorted(directory.glob("key-*.fhelium"))
    ]
    layer = fc.load(directory / "layer.fhelium", keys=keys, device=device)
    loaded = time.perf_counter() - started
    x = fh.load_value(
        directory / "input.fhelium", device=device, expected_type=Ciphertext
    )
    result = layer(x)
    fh.save_value(result, directory / "output.fhelium", overwrite=True)
    print(f"server: loaded the artifact in {loaded:.3f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep the artifact and value files here; otherwise use a temporary directory.",
    )
    parser.add_argument("--serve", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve is not None:
        serve(args.serve, args.device)
        return

    device = torch.device(args.device)
    config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7)
    engine = Engine(config, rng_seed=15)
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    rotation_key = engine.create_rotation_key(1, secret_key, device=device)
    weight = torch.linspace(0.1, 0.2, _SIZE, dtype=torch.float64)

    @fc.compile
    def layer(value: Ciphertext) -> Ciphertext:
        plain = engine.prepare_compressed_plaintext(
            weight, depth=value.depth, device=value.device
        )
        rotated = engine.add(value, engine.rotate_with_key(value, rotation_key))
        product = engine.multiply_plaintext(
            engine.coefficient_domain_to_ntt_domain(rotated), plain
        )
        return engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(product)
        )

    clear = torch.linspace(-0.2, 0.3, _SIZE, dtype=torch.float64)
    x = engine.encrypt_message(
        clear.repeat(config.num_slots // _SIZE), public_key, device=device
    )
    context = (
        TemporaryDirectory(prefix="fhelium-export-")
        if args.output_dir is None
        else nullcontext(args.output_dir)
    )
    with context as root:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "layer.fhelium"
        fc.export(layer, artifact, signatures=[(x,)], overwrite=True)
        fh.save_value(
            rotation_key, directory / "key-rotation-1.fhelium", overwrite=True
        )
        fh.save_value(x, directory / "input.fhelium", overwrite=True)

        subprocess.run(
            [
                sys.executable,
                __file__,
                "--serve",
                str(directory),
                "--device",
                args.device,
            ],
            check=True,
        )

        result = fh.load_value(
            directory / "output.fhelium",
            device=device,
            expected_type=Ciphertext,
        )
        decoded = engine.decrypt_message(result, secret_key, is_real=True)
        expected = (clear + torch.roll(clear, shifts=1)) * weight
        error = float((decoded[:_SIZE].cpu() - expected).abs().max())
        print_table(
            ["file", "bytes"],
            [
                [path.name, path.stat().st_size]
                for path in sorted(directory.iterdir())
            ],
        )
        print_table(
            ["result", "value"],
            [
                ["depth", result.depth],
                ["scale", f"{result.scale:.6e}"],
                ["max abs error", f"{error:.3e}"],
            ],
        )
        if args.output_dir is not None:
            print(
                "C++: load_artifact",
                artifact,
                directory / "output-cpp.fhelium",
                directory / "input.fhelium",
                directory / "key-rotation-1.fhelium",
            )


if __name__ == "__main__":
    main()
