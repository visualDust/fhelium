#!/usr/bin/env python3

"""Inspect residency and round-trip values through direct files."""

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
from fhelium.artifacts import ArtifactStore


def _persistence_demo(
    root: str | Path,
    *,
    engine: fh.CkksEngine,
    ciphertext: fh.Ciphertext,
    message: torch.Tensor,
) -> None:
    factor_message = torch.full_like(message, 1.25)
    canonical_factor = engine.encode(
        factor_message,
        level=ciphertext.level,
    )
    factor = engine.prepare_plaintext_for_multiplication(
        engine.encode(factor_message, level=ciphertext.level)
    )
    canonical_bytes = canonical_factor.nbytes

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
        device=engine.device,
        expected_type=fh.Ciphertext,
    )
    restored_factor = fh.load_value(
        factor_path,
        device=engine.device,
        expected_type=fh.Plaintext,
    )

    # ArtifactStore is a first-party repository layered on the same
    # typed value-file primitives. It adds names, references, collections,
    # checksums, and local durability policy without changing core values.
    artifact_store = ArtifactStore(root / "artifact-store")
    activation_ref = artifact_store.put(
        "requests/example/activation",
        ciphertext_cpu,
        overwrite=True,
    )
    factor_ref = artifact_store.put(
        "model/example/factor",
        factor_cpu,
        overwrite=True,
    )
    artifact_ciphertext = artifact_store.get(
        activation_ref,
        device=engine.device,
        expected_type=fh.Ciphertext,
    )
    artifact_factor = artifact_store.get(
        factor_ref,
        device=engine.device,
        expected_type=fh.Plaintext,
    )
    torch.testing.assert_close(
        artifact_ciphertext.data,
        restored_ciphertext.data,
    )
    torch.testing.assert_close(artifact_factor.data, restored_factor.data)

    result = engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(restored_ciphertext),
                restored_factor,
            )
        )
    )
    decoded = engine.decrypt_message(result)[: message.numel()]
    expected = (message * 1.25).to(decoded.dtype)
    torch.testing.assert_close(decoded, expected, atol=3e-5, rtol=0)

    print("\nResidency and value-file roundtrip:")
    print(f"  CUDA ciphertext:       {ciphertext.device}")
    print(f"  offloaded ciphertext: {ciphertext_cpu.device}")
    print(f"  canonical factor:     {format_bytes(canonical_bytes)}")
    print(f"  prepared factor:      {format_bytes(factor.nbytes)}")
    print(f"  activation file:      {activation_path.name}")
    print(f"  plaintext file:       {factor_path.name}")
    print(f"  activation artifact:  {activation_ref.name}")
    print(f"  plaintext artifact:   {factor_ref.name}")
    print(f"  persistence root:     {root.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep value files under this root; otherwise use a temporary one.",
    )
    args = parser.parse_args()

    engine = make_engine(args)
    requested = [int(item) for item in args.levels.split(",") if item.strip()]
    levels = [
        level for level in requested if 0 <= level < engine.public_level_count
    ]
    if not levels:
        raise ValueError("--levels did not select a valid CKKS level")
    message = small_complex_vector(engine.num_slots, seed=42)

    rows = []
    sample_ciphertext = None
    for level in levels:
        plaintext = engine.encode(message, level=level)
        assert plaintext.data is not None
        ciphertext = engine.encrypt(plaintext)
        if sample_ciphertext is None:
            sample_ciphertext = ciphertext
        rows.append(
            [
                level,
                ciphertext.limb_count,
                tuple(ciphertext.data.shape),
                format_bytes(ciphertext.nbytes),
                format_bytes(plaintext.nbytes),
            ]
        )
    print_table(
        ["level", "Q limbs", "ciphertext shape", "ciphertext", "plaintext"],
        rows,
    )

    assert sample_ciphertext is not None
    context = (
        TemporaryDirectory(prefix="fhelium-value-files-")
        if args.output_dir is None
        else nullcontext(args.output_dir)
    )
    with context as root:
        _persistence_demo(
            root,
            engine=engine,
            ciphertext=sample_ciphertext,
            message=message,
        )


if __name__ == "__main__":
    main()
