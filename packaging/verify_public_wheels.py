#!/usr/bin/env python3
"""Install and execute every published FHElium wheel for one platform."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

PACKAGING_ROOT = Path(__file__).parent
sys.path.insert(0, str(PACKAGING_ROOT))

from matrix import (
    LINUX,
    WINDOWS,
    MATRIX_PATH,
    Configuration,
    Platform,
    load_matrix,
)

ROOT = Path(os.path.abspath(PACKAGING_ROOT.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument(
        "--configuration",
        action="append",
        help="verify selected configuration identities only",
    )
    parser.add_argument("--platform", choices=(LINUX, WINDOWS), required=True)
    parser.add_argument("--python-312", type=Path, required=True)
    parser.add_argument("--python-313", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=24)
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args()


def run(
    *command: str,
    cwd: Path,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=dict(environment),
        check=True,
        text=True,
    )


def clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    prefixes = (
        "CMAKE_",
        "CONDA_",
        "CUDA_",
        "NVCC_",
        "HIP_",
        "ROCM_",
        "VC",
        "VS",
        "WINDOWSSDK",
    )
    names = {
        "CC",
        "CL",
        "CXX",
        "CUDACXX",
        "HIPCXX",
        "HIP_PATH",
        "ROCM_PATH",
        "ROCM_HOME",
        "PYTORCH_ROCM_ARCH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDAARCHS",
        "CUDA_HOME",
        "CUDA_PATH",
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "LINK",
        "PYTHONHOME",
        "PYTHONPATH",
        "TORCH_CUDA_ARCH_LIST",
        "VIRTUAL_ENV",
        "_CL_",
        "_LINK_",
    }
    for name in list(environment):
        upper = name.upper()
        if upper in {
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        }:
            continue
        if upper in names or upper.startswith(prefixes):
            environment.pop(name, None)
    environment.update(
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_CACHE_DIR="1",
        PYTHONNOUSERSITE="1",
    )
    return environment


def environment_python(root: Path, target: Platform) -> Path:
    if target.operating_system == "windows":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def runtime_path(python: Path, target: Platform) -> str:
    if target.operating_system == "windows":
        system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
        return os.pathsep.join(
            (
                str(python.parent),
                str(system_root / "System32"),
                str(system_root),
            )
        )
    return os.pathsep.join(
        (str(python.parent), "/usr/local/bin", "/usr/bin", "/bin")
    )


def verification_code(
    configuration: Configuration,
    target: Platform,
    version: str,
) -> str:
    """Check an installed wheel using its declared Torch and device identity."""
    product_source = (
        "def product(x):\n"
        "    ntt = engine.coefficient_domain_to_ntt_domain(x)\n"
        "    return engine.ntt_domain_to_coefficient_domain(engine.multiply(ntt, ntt))\n"
    )
    return f"""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import torch
import fhelium
from fhelium.native import native_status

assert Path(fhelium.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), fhelium.__file__
assert fhelium.__version__ == {version!r}
assert str(torch.__version__) == {configuration.torch_runtime_version!r}
assert torch.version.cuda == {configuration.torch_cuda_version!r}
assert torch.version.hip == {configuration.torch_hip_version!r}
assert bool(torch.compiled_with_cxx11_abi()) == {configuration.torch_cxx11_abi!r}
status = native_status()
assert status.available, status
assert set(status.backends) == {set(configuration.native_backends)!r}, status
parameters = torch.zeros((8, 1), dtype=torch.int64)
parameters[0, 0] = 34
value = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int64)
expected = torch.tensor([[[2, 4, 6, 8]]], dtype=torch.int64)
actual = torch.ops.fhelium_rns_ops.add_standard(value, value, parameters)
assert torch.equal(actual, expected)
if {configuration.has_gpu!r}:
    assert torch.cuda.is_available()
    value = value.cuda()
    parameters = parameters.cuda()
    actual = torch.ops.fhelium_rns_ops.add_standard(value, value, parameters)
    assert torch.equal(actual.cpu(), expected)
    from fhelium.native.cuda import get_cuda_device_properties
    assert get_cuda_device_properties()
    from fhelium.eager import Engine
    from fhelium import compile as fc
    engine = Engine(fhelium.Preset.slots8192_scale40_depth7_int64, rng_seed=24, rng_nonce=7)
    secret = engine.create_secret_key(device="cuda:0")
    public = engine.create_public_key(secret)
    message = torch.linspace(-0.02, 0.03, engine.num_slots, dtype=torch.float64, device="cuda:0")
    ciphertext = engine.encrypt_message(message, public, device="cuda:0")
    torch.testing.assert_close(engine.decrypt_message(ciphertext, secret, is_real=True), message, rtol=0, atol=5e-6)
    # Keep source inspection available when verification runs with Python -c.
    namespace = {{"engine": engine}}
    source = {product_source!r}
    import linecache
    filename = "<fhelium-wheel-product>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    reference = namespace["product"]
    compiled = fc.compile(reference)
    for operand in (ciphertext, engine.negate(ciphertext)):
        actual = compiled(operand)
        torch.testing.assert_close(actual.data, reference(operand).data, rtol=0, atol=0)
    names = {{dispatch.implementation.name for dispatch in compiled.specializations[0].executable.dispatch_table.operations.values()}}
    assert any(name.startswith("triton-") for name in names), names
    torch.cuda.synchronize()
if {configuration.has_hip!r}:
    installed = {{d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()}}
    assert "rocm-sdk-devel" not in installed, "installed-wheel verification must exclude the HIP development SDK"
    assert shutil.which("hipcc") is None, "installed-wheel verification must hide the HIP compiler"
    mappings = Path("/proc/self/maps").read_text()
    assert "_rocm_sdk_devel/" not in mappings, "runtime loaded a development SDK library"
print(json.dumps({{
    "platform": {target.id!r},
    "configuration": {configuration.id!r},
    "fhelium": fhelium.__version__,
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "torch_hip": torch.version.hip,
    "native": str(status),
}}))
"""


def install_fhelium(
    python: Path,
    *,
    configuration: Configuration,
    version: str,
    simple_index_base_url: str,
    root: Path,
    environment: Mapping[str, str],
    attempts: int,
    interval: float,
) -> None:
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
                environment=environment,
            )
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(interval)


def verify_cell(
    *,
    base_python: Path,
    configuration: Configuration,
    python_abi: str,
    target: Platform,
    version: str,
    work_root: Path,
    simple_index_base_url: str,
    attempts: int,
    interval: float,
) -> None:
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
        environment=environment,
    )
    python = environment_python(root, target)
    environment["PATH"] = runtime_path(python, target)
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
        *configuration.runtime_requirements,
        cwd=root,
        environment=environment,
    )
    install_fhelium(
        python,
        configuration=configuration,
        version=version,
        simple_index_base_url=simple_index_base_url,
        root=root,
        environment=environment,
        attempts=attempts,
        interval=interval,
    )
    run(
        str(python),
        "-I",
        "-c",
        verification_code(configuration, target, version),
        cwd=root,
        environment=environment,
    )
    shutil.rmtree(root)


def main() -> None:
    arguments = parse_args()
    matrix = load_matrix(arguments.matrix)
    target = matrix.platform(arguments.platform)
    work_root = Path(os.path.abspath(arguments.work_root))
    work_root.mkdir(parents=True, exist_ok=True)
    pythons = {
        "cp312-cp312": Path(os.path.abspath(arguments.python_312)),
        "cp313-cp313": Path(os.path.abspath(arguments.python_313)),
    }
    configurations = (
        tuple(matrix.configuration(name) for name in arguments.configuration)
        if arguments.configuration
        else matrix.configurations
    )
    for configuration in configurations:
        if not configuration.supports(target.id):
            continue
        for python_abi in matrix.python_abis:
            verify_cell(
                base_python=pythons[python_abi],
                configuration=configuration,
                python_abi=python_abi,
                target=target,
                version=arguments.version,
                work_root=work_root,
                simple_index_base_url=matrix.simple_index_base_url,
                attempts=arguments.attempts,
                interval=arguments.interval,
            )


if __name__ == "__main__":
    main()
