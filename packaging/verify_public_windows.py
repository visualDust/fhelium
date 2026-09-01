#!/usr/bin/env python3
"""Verify every published FHElium Windows wheel in an isolated environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PACKAGING_ROOT = Path(__file__).parent
sys.path.insert(0, str(PACKAGING_ROOT))

from matrix import WINDOWS, Configuration, load_matrix
from windows_wheel import clean_environment, python_identity, run

ROOT = Path(os.path.abspath(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-312", type=Path, required=True)
    parser.add_argument("--python-313", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=24)
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args()


def verification_code(configuration: Configuration, version: str) -> str:
    """Return the installed-wheel verification program for one configuration."""

    return f"""
import json

import fhelium
import torch
from fhelium.native import native_status

assert fhelium.__version__ == {version!r}
assert str(torch.__version__) == {configuration.torch_runtime_version!r}
assert torch.version.cuda == {configuration.torch_cuda_version!r}
status = native_status()
assert status.available, status
assert set(status.backends) == {set(configuration.native_backends)!r}, status
parameters = torch.zeros((8, 1), dtype=torch.int64)
parameters[0, 0] = 34
value = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int64)
expected = torch.tensor([[[2, 4, 6, 8]]], dtype=torch.int64)
actual = torch.ops.fhelium_rns_ops.add_standard(value, value, parameters)
assert torch.equal(actual, expected)
if {configuration.has_cuda!r}:
    value = value.cuda()
    parameters = parameters.cuda()
    actual = torch.ops.fhelium_rns_ops.add_standard(value, value, parameters)
    assert torch.equal(actual.cpu(), expected)
print(json.dumps({{
    "configuration": {configuration.id!r},
    "fhelium": fhelium.__version__,
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "native": str(status),
}}))
"""


def verify_cell(
    *,
    base_python: Path,
    configuration: Configuration,
    python_abi: str,
    version: str,
    work_root: Path,
    simple_index_base_url: str,
    attempts: int,
    interval: float,
) -> None:
    """Install and execute one public Windows wheel cell."""

    root = work_root / f"{configuration.id}-{python_abi}"
    shutil.rmtree(root, ignore_errors=True)
    environment = clean_environment()
    run(
        str(base_python),
        "-I",
        "-m",
        "venv",
        "--copies",
        str(root),
        cwd=ROOT,
        env=environment,
    )
    python = root / "Scripts" / "python.exe"
    system_root = Path(environment.get("SystemRoot", "C:/Windows"))
    environment["PATH"] = os.pathsep.join(
        (str(python.parent), str(system_root / "System32"), str(system_root))
    )
    run(
        str(python),
        "-I",
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--index-url",
        configuration.torch_index_url,
        configuration.torch_requirement,
        cwd=root,
        env=environment,
    )
    for attempt in range(1, attempts + 1):
        try:
            run(
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--only-binary=fhelium",
                "--extra-index-url",
                f"{simple_index_base_url}/{configuration.id}/simple/",
                f"fhelium=={version}",
                cwd=root,
                env=environment,
            )
            break
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(interval)
    run(
        str(python),
        "-I",
        "-c",
        verification_code(configuration, version),
        cwd=root,
        env=environment,
    )
    shutil.rmtree(root)


def main() -> None:
    arguments = parse_args()
    matrix = load_matrix()
    work_root = Path(os.path.abspath(arguments.work_root))
    work_root.mkdir(parents=True, exist_ok=True)
    pythons = {
        "cp312-cp312": Path(os.path.abspath(arguments.python_312)),
        "cp313-cp313": Path(os.path.abspath(arguments.python_313)),
    }
    for python_abi, python in pythons.items():
        python_identity(python, python_abi, cwd=ROOT)
    for configuration in matrix.configurations:
        if not configuration.supports(WINDOWS):
            continue
        for python_abi in matrix.python_abis:
            verify_cell(
                base_python=pythons[python_abi],
                configuration=configuration,
                python_abi=python_abi,
                version=arguments.version,
                work_root=work_root,
                simple_index_base_url=matrix.simple_index_base_url,
                attempts=arguments.attempts,
                interval=arguments.interval,
            )


if __name__ == "__main__":
    main()
