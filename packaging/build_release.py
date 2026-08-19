#!/usr/bin/env python3
"""Build one release-matrix cell in its declared manylinux environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_PROJECT = "/tmp/fhelium-source"


def load_matrix_module() -> ModuleType:
    path = Path(__file__).with_name("release_matrix.py")
    spec = importlib.util.spec_from_file_location(
        "_fhelium_release_matrix", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument(
        "--python-abi",
        required=True,
        choices=("cp312-cp312", "cp313-cp313"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--build-image", action="store_true")
    return parser.parse_args()


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    module = load_matrix_module()
    matrix = module.load_release_matrix()
    configuration = matrix.configuration(args.configuration)
    if args.python_abi not in matrix.python_abis:
        raise ValueError(f"unsupported Python ABI: {args.python_abi}")

    dockerfile = matrix.builder_dockerfile_by_environment[
        configuration.builder_environment
    ]
    image = f"fhelium-manylinux-{configuration.builder_environment}"
    if args.build_image:
        run(
            "docker",
            "build",
            "--file",
            dockerfile,
            "--tag",
            image,
            ".",
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = f"{CONTAINER_PROJECT}/packaging/release_matrix.json"
    build_command = [
        f"/opt/python/{args.python_abi}/bin/python",
        f"{CONTAINER_PROJECT}/packaging/build_manylinux_wheel.py",
        "--configuration",
        configuration.id,
        "--python",
        f"/opt/python/{args.python_abi}/bin/python",
        "--source",
        CONTAINER_PROJECT,
        "--output",
        f"/output/{configuration.id}/{args.python_abi}",
    ]
    if args.smoke:
        build_command.append("--smoke")
    container_command = (
        f"cp -a /project {CONTAINER_PROJECT} && "
        f"/opt/python/{args.python_abi}/bin/python -m pip install "
        f"jsonschema=={matrix.jsonschema_version} && "
        f"/opt/python/{args.python_abi}/bin/python "
        f"{CONTAINER_PROJECT}/packaging/release_matrix.py "
        f"--matrix {matrix_path} validate && "
        + shlex.join(build_command)
        + f" && chown -R {os.getuid()}:{os.getgid()} /output"
    )
    run(
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{ROOT}:/project:ro",
        "--volume",
        f"{output}:/output",
        image,
        "bash",
        "-lc",
        container_command,
    )


if __name__ == "__main__":
    main()
