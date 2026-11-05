#!/usr/bin/env python3
"""Build one declared Linux or Windows release-matrix wheel cell."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Protocol


ROOT = Path(os.path.abspath(Path(__file__).parent.parent))
CONTAINER_PROJECT = "/tmp/fhelium-source"


class LinuxAcceleratorConfiguration(Protocol):
    id: str
    torch_requirement: str
    torch_index_url: str
    torch_runtime_version: str
    torch_cuda_version: str | None
    torch_hip_version: str | None
    torch_cxx11_abi: bool
    runtime_requirements: tuple[str, ...]
    native_backends: tuple[str, ...]
    has_cuda: bool
    has_hip: bool
    has_gpu: bool


def load_module(filename: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(
        "_fhelium_packaging_" + path.stem, path
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
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("release_matrix.json"),
    )
    parser.add_argument(
        "--image", help="prebuilt builder image, preferably pinned by digest"
    )
    parser.add_argument(
        "--container-python",
        help="Python executable inside the selected builder image",
    )
    parser.add_argument("--use-installed-torch", action="store_true")
    parser.add_argument(
        "--platform-tag",
        choices=("manylinux_2_28_x86_64", "linux_x86_64"),
        default="manylinux_2_28_x86_64",
    )
    parser.add_argument(
        "--platform",
        choices=("manylinux_2_28_x86_64", "win_amd64"),
        default="manylinux_2_28_x86_64",
    )
    parser.add_argument(
        "--python-abi",
        required=True,
        choices=("cp312-cp312", "cp313-cp313"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--cuda-toolkit-root", type=Path)
    parser.add_argument(
        "--work-root",
        type=Path,
        help="caller-owned build and installed-wheel verification workspace",
    )
    parser.add_argument("--verify-install", action="store_true")
    parser.add_argument("--build-image", action="store_true")
    return parser.parse_args()


def run(
    *command: str,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def verify_linux_accelerator_wheel(
    wheel: Path,
    *,
    configuration: LinuxAcceleratorConfiguration,
    target,
    python_abi: str,
    work_root: Path,
    project_version: str,
) -> None:
    """Install one accelerator wheel in an isolated environment on its GPU host."""
    if not configuration.has_gpu:
        raise ValueError(
            "host GPU verification requires an accelerator configuration"
        )
    python_name = {"cp312-cp312": "python3.12", "cp313-cp313": "python3.13"}[
        python_abi
    ]
    base_python = shutil.which(python_name)
    if base_python is None:
        raise RuntimeError(f"host verification requires {python_name}")
    root = work_root / f"{configuration.id}-{python_abi}-linux-runtime"
    shutil.rmtree(root, ignore_errors=True)
    verifier = load_module("verify_public_wheels.py")
    environment = verifier.clean_environment()
    run(
        base_python,
        "-I",
        "-m",
        "venv",
        "--copies",
        str(root),
        cwd=Path("/"),
        env=environment,
    )
    python = root / "bin" / "python"
    environment["PATH"] = verifier.runtime_path(python, target)
    run(
        str(python),
        "-I",
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        configuration.torch_requirement,
        *configuration.runtime_requirements,
        "--index-url",
        configuration.torch_index_url,
        cwd=root,
        env=environment,
    )
    run(
        str(python),
        "-I",
        "-m",
        "pip",
        "install",
        str(wheel),
        "--extra-index-url",
        configuration.torch_index_url,
        cwd=root,
        env=environment,
    )
    run(
        str(python),
        "-I",
        "-c",
        verifier.verification_code(configuration, target, project_version),
        cwd=root,
        env=environment,
    )
    shutil.rmtree(root)


def main() -> None:
    args = parse_args()
    if args.platform == "win_amd64":
        if (
            args.matrix.resolve()
            != Path(__file__).with_name("release_matrix.json").resolve()
        ):
            raise ValueError("Windows builds use the source release matrix")
        if (
            args.image
            or args.container_python
            or args.use_installed_torch
            or args.platform_tag != "manylinux_2_28_x86_64"
        ):
            raise ValueError("Linux builder options do not apply to Windows")
        if args.python is None:
            raise ValueError("Windows builds require --python")
        if args.work_root is None:
            raise ValueError("Windows builds require --work-root")

        packaging_root = str(ROOT / "packaging")
        sys.path.insert(0, packaging_root)
        try:
            from windows_wheel import build
        finally:
            sys.path.remove(packaging_root)

        args.source = ROOT
        build(args)
        return
    if args.python is not None or args.cuda_toolkit_root is not None:
        raise ValueError("Linux container builds do not accept Windows options")
    if args.verify_install != (args.work_root is not None):
        raise ValueError(
            "Linux installed-wheel verification requires --verify-install "
            "and --work-root together"
        )
    module = load_module("matrix.py")
    matrix = module.load_matrix(args.matrix)
    configuration = matrix.configuration(args.configuration)
    if not configuration.supports(args.platform):
        raise ValueError(f"{configuration.id} does not support {args.platform}")
    if args.python_abi not in matrix.python_abis:
        raise ValueError(f"unsupported Python ABI: {args.python_abi}")

    image = (
        args.image or f"fhelium-manylinux-{configuration.builder_environment}"
    )
    if args.build_image:
        if args.image:
            raise ValueError(
                "--build-image cannot replace an explicitly selected --image"
            )
        dockerfile = matrix.builder_dockerfile_by_environment[
            configuration.builder_environment
        ]
        run("docker", "build", "--file", dockerfile, "--tag", image, ".")

    output = Path(os.path.abspath(args.output))
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = "/fhelium-matrix.json"
    container_python = (
        args.container_python or f"/opt/python/{args.python_abi}/bin/python"
    )
    build_command = [
        container_python,
        f"{CONTAINER_PROJECT}/packaging/linux_wheel.py",
        "--configuration",
        configuration.id,
        "--matrix",
        matrix_path,
        "--platform-tag",
        args.platform_tag,
        "--python",
        container_python,
        "--source",
        CONTAINER_PROJECT,
        "--output",
        f"/output/{configuration.id}/{args.python_abi}",
    ]
    if args.use_installed_torch:
        build_command.append("--use-installed-torch")
    if args.verify_install:
        build_command.append("--verify-install")
    python_minor = {"cp312-cp312": 12, "cp313-cp313": 13}[args.python_abi]
    identity_command = shlex.join(
        (
            container_python,
            "-c",
            f"import sys; assert sys.version_info[:2] == (3, {python_minor})",
        )
    )
    container_command = (
        identity_command + " && "
        f"cp -a /project {CONTAINER_PROJECT} && "
        f"{shlex.quote(container_python)} -m pip install "
        f"jsonschema=={matrix.jsonschema_version} && "
        f"{shlex.quote(container_python)} "
        f"{CONTAINER_PROJECT}/packaging/matrix.py "
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
        f"{args.matrix.resolve()}:{matrix_path}:ro",
        "--volume",
        f"{output}:/output",
        image,
        "bash",
        "-lc",
        container_command,
    )
    if args.verify_install and configuration.has_gpu:
        wheelhouse = output / configuration.id / args.python_abi / "wheelhouse"
        wheels = list(wheelhouse.glob("fhelium-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one repaired wheel, found {wheels!r}")
        verify_linux_accelerator_wheel(
            wheels[0],
            configuration=configuration,
            target=matrix.platform(args.platform),
            python_abi=args.python_abi,
            work_root=Path(os.path.abspath(args.work_root)),
            project_version=tomllib.loads(
                (ROOT / "pyproject.toml").read_text(encoding="utf-8")
            )["project"]["version"],
        )


if __name__ == "__main__":
    main()
