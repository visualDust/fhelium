#!/usr/bin/env python3

"""Move CKKS values and restore typed values from caller-owned files."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from common import (
    add_engine_args,
    format_bytes,
    make_engine,
    print_table,
    small_complex_vector,
)

import fhelium as fh
from fhelium.eager import Engine


def _persistence_demo(
    root: str | Path,
    *,
    engine: Engine,
    ciphertext: fh.Ciphertext,
    message: torch.Tensor,
) -> None:
    factor_message = torch.full_like(message, 1.25)
    encoded_factor = engine.encode(
        factor_message,
        depth=ciphertext.depth,
    )
    factor = engine.prepare_plaintext_for_multiplication(
        engine.encode(factor_message, depth=ciphertext.depth)
    )
    encoded_bytes = encoded_factor.nbytes

    ciphertext_cpu = ciphertext.to("cpu")
    factor_cpu = factor.to("cpu")
    assert ciphertext_cpu.is_cpu and factor_cpu.is_cpu

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    activation_path = root / "activation.safetensors"
    factor_path = root / "factor.safetensors"
    fh.save_value(
        ciphertext_cpu,
        activation_path,
        overwrite=True,
    )
    fh.save_value(
        factor_cpu,
        factor_path,
        overwrite=True,
    )

    restored_ciphertext = fh.load_value(
        activation_path,
        device=torch.get_default_device(),
        expected_type=fh.Ciphertext,
    )
    restored_factor = fh.load_value(
        factor_path,
        device=torch.get_default_device(),
        expected_type=fh.Plaintext,
    )

    result = engine.rescale_to_next_depth(
        engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(restored_ciphertext),
                restored_factor,
            )
        )
    )
    decoded = engine.decrypt_message(result).cpu()[: message.numel()]
    expected = (message * 1.25).to(decoded.dtype)
    torch.testing.assert_close(decoded, expected, atol=3e-5, rtol=0)

    print("\nValue movement and file roundtrip:")
    print(f"  original ciphertext:       {ciphertext.device}")
    print(f"  offloaded ciphertext: {ciphertext_cpu.device}")
    print(f"  encoded factor:       {format_bytes(encoded_bytes)}")
    print(f"  prepared factor:      {format_bytes(factor.nbytes)}")
    print(f"  activation file:      {activation_path.name}")
    print(f"  plaintext file:       {factor_path.name}")
    print(f"  persistence root:     {root.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep value files under this root; otherwise use a temporary one.",
    )
    args = parser.parse_args()

    engine = make_engine(args)
    message = small_complex_vector(engine.num_slots, seed=42)
    plaintext = engine.encode(message, depth=args.depth)
    ciphertext = engine.encrypt(plaintext)
    print_table(
        ["value", "device", "logical bytes"],
        [
            ["plaintext", plaintext.device, format_bytes(plaintext.nbytes)],
            ["ciphertext", ciphertext.device, format_bytes(ciphertext.nbytes)],
        ],
    )
    context = (
        TemporaryDirectory(prefix="fhelium-value-files-")
        if args.output_dir is None
        else nullcontext(args.output_dir)
    )
    with context as root:
        _persistence_demo(
            root,
            engine=engine,
            ciphertext=ciphertext,
            message=message,
        )


if __name__ == "__main__":
    main()
