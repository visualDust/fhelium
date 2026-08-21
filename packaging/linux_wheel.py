#!/usr/bin/env python3
"""Build and validate one configured FHElium manylinux wheel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Protocol


class ReleaseConfiguration(Protocol):
    id: str
    torch_requirement: str
    torch_index_url: str
    torch_runtime_version: str
    torch_cuda_version: str | None
    torch_cxx11_abi: bool
    native_backends: tuple[str, ...]
    toolkit_version: str | None
    cuda_architectures: tuple[str, ...]

    @property
    def has_cuda(self) -> bool: ...


class ReleaseMatrix(Protocol):
    scikit_build_core_version: str

    def configuration(self, configuration_id: str) -> ReleaseConfiguration: ...


def load_release_matrix_module(source: Path) -> Any:
    path = source / "packaging" / "matrix.py"
    spec = importlib.util.spec_from_file_location(
        "_fhelium_release_matrix", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load release-matrix implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def load_release_matrix(source: Path) -> ReleaseMatrix:
    module = load_release_matrix_module(source)
    return module.load_matrix(source / "packaging" / "release_matrix.json")


def assert_torch_identity(
    python: Path, configuration: ReleaseConfiguration
) -> None:
    code = """
import json
import torch
print(json.dumps({
    'version': str(torch.__version__),
    'cuda': torch.version.cuda,
    'cxx11_abi': bool(torch.compiled_with_cxx11_abi()),
}))
"""
    actual = json.loads(
        subprocess.run(
            [str(python), "-c", code],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    expected = {
        "version": configuration.torch_runtime_version,
        "cuda": configuration.torch_cuda_version,
        "cxx11_abi": configuration.torch_cxx11_abi,
    }
    if actual != expected:
        raise RuntimeError(
            f"Torch identity does not match {configuration.id}: "
            f"expected={expected!r}, actual={actual!r}"
        )


def torch_cuda_architecture_list(
    cuda_architectures: tuple[str, ...],
) -> str:
    """Translate CMake architecture numbers to Torch's dotted syntax."""

    values = []
    for architecture in cuda_architectures:
        if not architecture.isdecimal() or len(architecture) < 2:
            raise ValueError(
                f"invalid numeric CUDA architecture: {architecture!r}"
            )
        number = int(architecture)
        values.append(f"{number // 10}.{number % 10}")
    return ";".join(values)


def build_environment(
    configuration: ReleaseConfiguration,
) -> dict[str, str]:
    environment = os.environ.copy()
    cmake_arguments = [
        "-DFHELIUM_NATIVE_BACKENDS="
        + "+".join(backend.upper() for backend in configuration.native_backends)
    ]
    if configuration.has_cuda:
        if configuration.torch_cuda_version is None:
            raise RuntimeError(
                "CUDA configuration is missing torch_cuda_version"
            )
        toolkit_root = "/usr/local/cuda-" + configuration.torch_cuda_version
        environment.update(
            {
                "CUDA_HOME": toolkit_root,
                "CUDACXX": f"{toolkit_root}/bin/nvcc",
                "TORCH_CUDA_ARCH_LIST": torch_cuda_architecture_list(
                    configuration.cuda_architectures
                ),
            }
        )
        cmake_arguments.extend(
            (
                "-DCMAKE_CUDA_ARCHITECTURES="
                + ";".join(configuration.cuda_architectures),
                f"-DCUDAToolkit_ROOT={toolkit_root}",
                f"-DCUDA_TOOLKIT_ROOT_DIR={toolkit_root}",
            )
        )
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = environment.get(
        "CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1)
    )
    environment["CMAKE_ARGS"] = " ".join(cmake_arguments)
    environment["FHELIUM_RELEASE_CONFIGURATION_ID"] = configuration.id
    environment["FHELIUM_RELEASE_TOOLKIT_VERSION"] = (
        configuration.toolkit_version or ""
    )
    environment["FHELIUM_RELEASE_CUDA_ARCHITECTURES"] = ";".join(
        configuration.cuda_architectures
    )
    environment["FHELIUM_RELEASE_TORCH_REQUIREMENT"] = (
        configuration.torch_requirement
    )
    return environment


def external_libraries(
    configuration: ReleaseConfiguration,
) -> tuple[str, ...]:
    libraries = ["libtorch*.so", "libc10*.so"]
    if configuration.has_cuda:
        libraries.append("libcudart.so.*")
    return tuple(libraries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="install the repaired wheel and run an isolated CPU smoke test",
    )
    return parser.parse_args()


def run(*command: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def torch_external_library_paths(python: Path) -> str:
    code = """
from pathlib import Path
import torch
root = Path(torch.__file__).resolve().parent.parent
paths = [root / 'torch' / 'lib', *root.glob('nvidia/*/lib')]
print(':'.join(str(path) for path in paths if path.is_dir()))
"""
    return subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    python = args.python.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    release_matrix = load_release_matrix(source)
    configuration = release_matrix.configuration(args.configuration)
    project = tomllib.loads(
        (source / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    project_version = project["version"]
    if output.exists():
        shutil.rmtree(output)
    raw = output / "raw"
    wheelhouse = output / "wheelhouse"
    raw.mkdir(parents=True)
    wheelhouse.mkdir()

    run(
        str(python),
        "-m",
        "pip",
        "install",
        f"scikit-build-core=={release_matrix.scikit_build_core_version}",
    )
    run(
        str(python),
        "-m",
        "pip",
        "install",
        configuration.torch_requirement,
        "--index-url",
        configuration.torch_index_url,
    )

    assert_torch_identity(python, configuration)
    environment = build_environment(configuration)
    run(
        str(python),
        "-m",
        "pip",
        "wheel",
        str(source),
        "--no-build-isolation",
        "--no-deps",
        "--no-cache-dir",
        "--wheel-dir",
        str(raw),
        env=environment,
    )
    wheels = list(raw.glob("fhelium-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one raw wheel, found {wheels!r}")

    audit_environment = os.environ.copy()
    audit_environment["AUDITWHEEL_LD_LIBRARY_PATH"] = (
        torch_external_library_paths(python)
    )
    repair_command = [
        "auditwheel",
        "repair",
        "--plat",
        "manylinux_2_28_x86_64",
    ]
    for library in external_libraries(configuration):
        repair_command.extend(("--exclude", library))
    repair_command.extend(("--wheel-dir", str(wheelhouse), str(wheels[0])))
    run(*repair_command, env=audit_environment)

    repaired_wheels = list(wheelhouse.glob("fhelium-*.whl"))
    if len(repaired_wheels) != 1:
        raise RuntimeError(
            f"expected one repaired wheel, found {repaired_wheels!r}"
        )
    run(
        str(python),
        str(source / "packaging" / "linux_wheel_check.py"),
        str(repaired_wheels[0]),
        "--matrix",
        str(source / "packaging" / "release_matrix.json"),
        "--configuration",
        configuration.id,
        "--project-version",
        project_version,
    )
    if args.smoke:
        run(
            str(python),
            "-m",
            "pip",
            "install",
            "safetensors==0.8.0",
            "--force-reinstall",
        )
        run(
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(repaired_wheels[0]),
        )
        smoke = """
import importlib
import torch
import fhelium
from fhelium.native import native_status

status = native_status()
assert status.available, status
assert set(status.backends) == set(expected_backends)
if 'cuda' in expected_backends:
    importlib.import_module('fhelium.native.cuda.cuda_info')
lhs = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int64)
parameters = torch.zeros((8, 1), dtype=torch.int64)
parameters[0, 0] = 34
actual = torch.ops.fhelium_rns_ops.add_canonical(lhs, lhs, parameters)
expected = torch.tensor([[[2, 4, 6, 8]]], dtype=torch.int64)
assert torch.equal(actual, expected)
print(fhelium.__file__, status)
"""
        smoke_environment = os.environ.copy()
        smoke_environment.pop("LD_LIBRARY_PATH", None)
        smoke_environment.pop("PYTHONPATH", None)
        run(
            str(python),
            "-I",
            "-c",
            f"expected_backends = {list(configuration.native_backends)!r}\n"
            + smoke,
            env=smoke_environment,
        )
    print(repaired_wheels[0])


if __name__ == "__main__":
    main()
