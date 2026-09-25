#!/usr/bin/env python3

"""Save selected Program materials, load, complete bindings, and execute.

Material names identify external Tensor operands. Descriptions help callers
supply missing data and do not constrain deliberate replacements. The selected
keys are prepared before capture; loading and linking never generate keys.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from common import print_table

from fhelium import compile as fc
from fhelium.backend import OperationBackend
from fhelium.backend.ckks import CkksDeviceResources
from fhelium.config import CkksConfig, Preset
from fhelium.eager import Engine
from fhelium.serialization import load_compilation, save_compilation
from fhelium.values import Ciphertext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-materials",
        choices=("none", "partial", "all"),
        default="partial",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep the file here; otherwise use a temporary directory.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    engine = Engine(config, rng_seed=16)
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    keys = {
        step: engine.create_rotation_key(step, secret_key, device=device)
        for step in (1, 2)
    }

    def rotations(value: Ciphertext) -> tuple[Ciphertext, Ciphertext]:
        return (
            engine.rotate_with_key(value, keys[1]),
            engine.rotate_with_key(value, keys[2]),
        )

    clear = torch.linspace(
        -0.1, 0.1, config.num_slots, dtype=torch.float64, device=device
    )
    source = engine.encrypt_message(clear, public_key, device=device)
    captured = fc.capture_eager(
        rotations,
        arguments={"value": source},
        material_names={f"rotation_{step}": key for step, key in keys.items()},
    )
    captured.program.set_material_description(
        "rotation_1",
        {
            **captured.program.material_descriptions["rotation_1"],
            "note": "First cyclic shift",
        },
    )
    included = {"none": False, "partial": {"rotation_1"}, "all": True}[
        args.include_materials
    ]
    context = (
        TemporaryDirectory(prefix="fhelium-program-materials-")
        if args.output_dir is None
        else nullcontext(args.output_dir)
    )
    with context as root:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "rotations.safetensors"
        save_compilation(
            captured, path, include_materials=included, overwrite=True
        )
        restored = load_compilation(path, device=device)
        saved_symbols = set(restored.material_bindings)
        unresolved = fc.prepare_material_bindings(
            restored,
            resources=CkksDeviceResources(config=config, device=device),
            keys=keys,
        )
        if unresolved:
            raise RuntimeError(f"Unbound materials: {unresolved}")
        # Direct assignment remains available, including intentional replacements.
        restored.material_bindings["rotation_2"] = keys[2].data
        executable = OperationBackend().link(restored)
        result = executable.run(source)
        assert isinstance(result, tuple)
        for actual, expected in zip(result, rotations(source), strict=True):
            assert isinstance(actual, Ciphertext)
            torch.testing.assert_close(
                actual.data, expected.data, rtol=0, atol=0
            )
        print_table(
            ["symbol", "kind", "data source"],
            [
                [
                    symbol,
                    restored.program.material_descriptions[symbol].get("kind"),
                    "file"
                    if symbol in saved_symbols
                    else "supplied after load",
                ]
                for symbol in restored.material_bindings
            ],
        )
        print(f"Restored execution matches Eager; file: {path}")


if __name__ == "__main__":
    main()
