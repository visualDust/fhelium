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


class LinuxCudaConfiguration(Protocol):
    id: str
    torch_requirement: str
    torch_index_url: str
    torch_runtime_version: str
    torch_cuda_version: str | None
    native_backends: tuple[str, ...]
    has_cuda: bool


def load_matrix_module() -> ModuleType:
    path = Path(__file__).with_name("matrix.py")
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


def verify_linux_cuda_wheel(
    wheel: Path,
    *,
    configuration: LinuxCudaConfiguration,
    python_abi: str,
    work_root: Path,
    project_version: str,
) -> None:
    """Install one Linux CUDA wheel and execute it on the host GPU."""

    if not configuration.has_cuda or configuration.torch_cuda_version is None:
        raise ValueError("host CUDA verification requires a CUDA configuration")
    python_name = {
        "cp312-cp312": "python3.12",
        "cp313-cp313": "python3.13",
    }[python_abi]
    base_python = shutil.which(python_name)
    if base_python is None:
        raise RuntimeError(f"host verification requires {python_name}")
    root = work_root / f"{configuration.id}-{python_abi}-linux-cuda"
    shutil.rmtree(root, ignore_errors=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_CACHE_DIR="1",
        PYTHONNOUSERSITE="1",
    )
    run(base_python, "-m", "venv", str(root), cwd=Path("/"), env=environment)
    python = root / "bin" / "python"
    run(
        str(python),
        "-m",
        "pip",
        "install",
        configuration.torch_requirement,
        "--index-url",
        configuration.torch_index_url,
        cwd=Path("/"),
        env=environment,
    )
    run(
        str(python),
        "-m",
        "pip",
        "install",
        str(wheel),
        "--extra-index-url",
        configuration.torch_index_url,
        cwd=Path("/"),
        env=environment,
    )
    code = f"""
import json

import torch
import fhelium
from fhelium.native import native_status
from fhelium.native.cuda import get_cuda_device_properties

assert fhelium.__version__ == {project_version!r}
assert str(torch.__version__) == {configuration.torch_runtime_version!r}
assert torch.version.cuda == {configuration.torch_cuda_version!r}
status = native_status()
assert status.available, status
assert set(status.backends) == {set(configuration.native_backends)!r}, status
device = torch.device("cuda:0")
lhs = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int64, device=device)
parameters = torch.zeros((8, 1), dtype=torch.int64, device=device)
parameters[0, 0] = 34
actual = torch.ops.fhelium_rns_ops.add_standard(lhs, lhs, parameters)
expected = torch.tensor([[[2, 4, 6, 8]]], dtype=torch.int64)
assert torch.equal(actual.cpu(), expected)
print(json.dumps({{
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "devices": get_cuda_device_properties(),
    "native": str(status),
}}, default=str))
"""
    run(str(python), "-I", "-c", code, cwd=Path("/"), env=environment)
    shutil.rmtree(root)


def main() -> None:
    args = parse_args()
    if args.platform == "win_amd64":
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
    module = load_matrix_module()
    matrix = module.load_matrix()
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

    output = Path(os.path.abspath(args.output))
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = f"{CONTAINER_PROJECT}/packaging/release_matrix.json"
    build_command = [
        f"/opt/python/{args.python_abi}/bin/python",
        f"{CONTAINER_PROJECT}/packaging/linux_wheel.py",
        "--configuration",
        configuration.id,
        "--python",
        f"/opt/python/{args.python_abi}/bin/python",
        "--source",
        CONTAINER_PROJECT,
        "--output",
        f"/output/{configuration.id}/{args.python_abi}",
    ]
    if args.verify_install:
        build_command.append("--verify-install")
    container_command = (
        f"cp -a /project {CONTAINER_PROJECT} && "
        f"/opt/python/{args.python_abi}/bin/python -m pip install "
        f"jsonschema=={matrix.jsonschema_version} && "
        f"/opt/python/{args.python_abi}/bin/python "
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
        f"{output}:/output",
        image,
        "bash",
        "-lc",
        container_command,
    )
    if args.verify_install and configuration.has_cuda:
        wheelhouse = output / configuration.id / args.python_abi / "wheelhouse"
        wheels = list(wheelhouse.glob("fhelium-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one repaired wheel, found {wheels!r}")
        verify_linux_cuda_wheel(
            wheels[0],
            configuration=configuration,
            python_abi=args.python_abi,
            work_root=Path(os.path.abspath(args.work_root)),
            project_version=tomllib.loads(
                (ROOT / "pyproject.toml").read_text(encoding="utf-8")
            )["project"]["version"],
        )


if __name__ == "__main__":
    main()
