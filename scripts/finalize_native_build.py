#!/usr/bin/env python3
"""Write the minimal Python/Torch ABI manifest after native installation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import tomllib
from pathlib import Path
from types import ModuleType

import torch


def load_helpers(source_root: Path) -> ModuleType:
    path = source_root / "fhelium" / "native" / "_abi.py"
    spec = importlib.util.spec_from_file_location("_fhelium_native_abi", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load native ABI helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument(
        "--backends",
        required=True,
        help="Comma-separated native execution backends included in _ops.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    package_root = args.package_root.resolve()
    helpers = load_helpers(source_root)
    backends = tuple(
        part.strip().lower()
        for part in args.backends.split(",")
        if part.strip()
    )
    if not backends or any(
        backend not in {"cpu", "cuda"} for backend in backends
    ):
        raise RuntimeError("--backends must select one or both of: cpu, cuda")
    project = tomllib.loads(
        (source_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    identity = helpers.runtime_identity(torch)
    soabi = identity["python"]["soabi"]
    if not soabi or not identity["python"]["ext_suffix"]:
        raise RuntimeError("Python did not report SOABI/EXT_SUFFIX")

    destination = helpers.manifest_path(
        package_root / "native" / "torchops", soabi
    )
    atomic_write_json(
        destination,
        {
            "manifest_schema_version": helpers.MANIFEST_SCHEMA_VERSION,
            "project_version": project["version"],
            "native_backends": list(backends),
            **(
                {
                    "release_configuration_id": release_configuration_id,
                    "build_toolkit_version": os.environ.get(
                        "FHELIUM_RELEASE_TOOLKIT_VERSION"
                    )
                    or None,
                    "build_cuda_architectures": [
                        value
                        for value in os.environ.get(
                            "FHELIUM_RELEASE_CUDA_ARCHITECTURES", ""
                        ).split(";")
                        if value
                    ],
                }
                if (
                    release_configuration_id := os.environ.get(
                        "FHELIUM_RELEASE_CONFIGURATION_ID"
                    )
                )
                else {}
            ),
            **identity,
        },
    )
    print(f"Wrote FHElium native ABI manifest: {destination}")


if __name__ == "__main__":
    main()
