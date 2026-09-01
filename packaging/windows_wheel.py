#!/usr/bin/env python3
"""Build or inspect one FHElium Windows wheel."""

from __future__ import annotations

import argparse
import ctypes
import email.parser
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from email.message import Message
from typing import Any, Mapping
from zipfile import ZipFile

from matrix import WINDOWS, Configuration, load_matrix

ROOT = Path(os.path.abspath(Path(__file__).parent.parent))


def absolute_path(path: Path) -> Path:
    """Make a path absolute without dereferencing a caller-selected alias."""

    return Path(os.path.abspath(path))


def run(
    *command: str,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=None if env is None else dict(env),
        check=True,
        capture_output=capture,
        text=True,
    )


def clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    prefixes = ("CMAKE_", "CONDA_", "CUDA_", "NVCC_", "VC", "VS", "WINDOWSSDK")
    names = {
        "CC",
        "CL",
        "CXX",
        "CUDACXX",
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
        if upper in names or upper.startswith(prefixes):
            environment.pop(name, None)
    environment.update(
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_CACHE_DIR="1",
        PYTHONNOUSERSITE="1",
        VSCMD_SKIP_SENDTELEMETRY="1",
    )
    return environment


def python_identity(python: Path, abi: str, *, cwd: Path) -> None:
    code = """
import json, platform, struct, sys, sysconfig
print(json.dumps({
 'implementation': sys.implementation.name,
 'version': [sys.version_info.major, sys.version_info.minor],
 'machine': platform.machine().upper(),
 'bits': struct.calcsize('P') * 8,
 'suffix': sysconfig.get_config_var('EXT_SUFFIX'),
}))
"""
    value = json.loads(
        run(str(python), "-I", "-c", code, cwd=cwd, capture=True).stdout
    )
    minor = int(abi.removeprefix("cp3").split("-", 1)[0])
    expected_suffix = f".cp3{minor}-win_amd64.pyd"
    expected = {
        "implementation": "cpython",
        "version": [3, minor],
        "machine": "AMD64",
        "bits": 64,
        "suffix": expected_suffix,
    }
    if value != expected:
        raise RuntimeError(f"Python does not match {abi}/win_amd64: {value!r}")


def vs_environment(
    builder: dict[str, Any], *, temporary_root: Path
) -> dict[str, str]:
    program_files = Path(
        os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
    )
    vswhere = (
        program_files / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    )
    result = run(
        str(vswhere),
        "-latest",
        "-products",
        "*",
        "-version",
        "[17.0,18.0)",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
        cwd=ROOT,
        env=clean_environment(),
        capture=True,
    )
    visual_studio = Path(result.stdout.strip())
    command = visual_studio / "Common7" / "Tools" / "VsDevCmd.bat"
    with tempfile.TemporaryDirectory(
        prefix="fhelium-vs-", dir=temporary_root
    ) as temporary:
        script = Path(temporary) / "environment.cmd"
        script.write_text(
            "@echo off\n"
            f'call "{command}" -no_logo -host_arch=x64 -arch=x64 '
            f'-winsdk={builder["windows_sdk_version"]} '
            f'-vcvars_ver={builder["msvc_toolset_version"]}\n'
            "if errorlevel 1 exit /b %errorlevel%\nset\n",
            encoding="utf-8",
        )
        result = run(
            os.environ.get("ComSpec", "C:/Windows/System32/cmd.exe"),
            "/d",
            "/c",
            str(script),
            cwd=ROOT,
            env=clean_environment(),
            capture=True,
        )
    environment = clean_environment()
    for line in result.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            name, value = line.split("=", 1)
            environment[name] = value
    if (
        environment.get("VCToolsVersion", "").rstrip("\\/")
        != builder["msvc_toolset_version"]
    ):
        raise RuntimeError(
            "selected MSVC toolset differs from the release matrix"
        )
    if (
        environment.get("WindowsSDKVersion", "").rstrip("\\/")
        != builder["windows_sdk_version"]
    ):
        raise RuntimeError(
            "selected Windows SDK differs from the release matrix"
        )
    return environment


def toolkit_root(
    configuration: Configuration, requested: Path | None
) -> Path | None:
    toolkit = configuration.toolkit(WINDOWS)
    if toolkit is None:
        if requested is not None:
            raise RuntimeError("CPU wheel cannot select a CUDA Toolkit")
        return None
    if requested is None:
        requested = (
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "NVIDIA GPU Computing Toolkit"
            / "CUDA"
            / f"v{'.'.join(toolkit.version.split('.')[:2])}"
        )
    root = requested.resolve()
    if (
        not (root / "bin" / "nvcc.exe").is_file()
        or not (root / "bin" / "cuobjdump.exe").is_file()
    ):
        raise RuntimeError(f"incomplete CUDA Toolkit: {root}")
    return root


def cuda_compiler_root(root: Path) -> Path:
    """Return a no-space Toolkit spelling accepted by nvcc host options."""

    if " " not in str(root):
        return root
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(  # type: ignore[attr-defined]
        str(root), buffer, len(buffer)
    )
    if length == 0 or length >= len(buffer) or " " in buffer.value:
        raise RuntimeError(
            "CUDA Toolkit has no no-space Windows path for native compilation: "
            f"{root}"
        )
    return Path(buffer.value)


def build(args: argparse.Namespace) -> Path:
    if os.name != "nt" or platform.machine().upper() != "AMD64":
        raise RuntimeError("Windows wheel builds require Windows AMD64")
    source = absolute_path(args.source)
    matrix_path = source / "packaging" / "release_matrix.json"
    matrix = load_matrix(matrix_path, validate_schema=False)
    configuration = matrix.configuration(args.configuration)
    if (
        not configuration.supports(WINDOWS)
        or args.python_abi not in matrix.python_abis
    ):
        raise RuntimeError("cell is not declared by the release matrix")
    base_python = absolute_path(args.python)
    python_identity(base_python, args.python_abi, cwd=source)

    work_root = absolute_path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    work = work_root / f"{configuration.id}-{args.python_abi}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    build_environment = work / "environment"
    run(
        str(base_python),
        "-I",
        "-m",
        "venv",
        "--copies",
        str(build_environment),
        cwd=source,
        env=clean_environment(),
    )
    python = build_environment / "Scripts" / "python.exe"
    builder = matrix.platform(WINDOWS).builder
    pins = (
        f"pip=={builder['pip_version']}",
        f"cmake=={builder['cmake_version']}",
        f"ninja=={builder['ninja_version']}",
        f"packaging=={builder['packaging_version']}",
        f"jsonschema=={matrix.jsonschema_version}",
        f"scikit-build-core=={matrix.scikit_build_core_version}",
    )
    run(
        str(python),
        "-I",
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        *pins,
        cwd=source,
        env=clean_environment(),
    )
    run(
        str(python),
        "-I",
        str(source / "packaging" / "matrix.py"),
        "--matrix",
        str(matrix_path),
        "validate",
        cwd=source,
        env=clean_environment(),
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
        cwd=source,
        env=clean_environment(),
    )
    environment = vs_environment(builder, temporary_root=work)
    environment["PATH"] = str(python.parent) + os.pathsep + environment["PATH"]
    root = toolkit_root(configuration, args.cuda_toolkit_root)
    compiler_root = None if root is None else cuda_compiler_root(root)
    backends = "+".join(item.upper() for item in configuration.native_backends)
    cmake_args = [
        f"-DFHELIUM_NATIVE_BACKENDS={backends}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if compiler_root is not None:
        cmake_arch = ";".join(configuration.cmake_cuda_architectures)
        torch_arch = ";".join(configuration.torch_cuda_architectures)
        environment.update(
            CUDA_HOME=str(compiler_root),
            CUDA_PATH=str(compiler_root),
            CUDACXX=str(compiler_root / "bin" / "nvcc.exe"),
            CUDAARCHS=cmake_arch,
            TORCH_CUDA_ARCH_LIST=torch_arch,
        )
        cmake_args.extend(
            (
                f"-DCUDAToolkit_ROOT={compiler_root}",
                f"-DCMAKE_CUDA_ARCHITECTURES={cmake_arch}",
            )
        )
    environment.update(
        CMAKE_ARGS=" ".join(
            f'"{item}"' if " " in item else item for item in cmake_args
        ),
        CMAKE_GENERATOR="Ninja",
        CMAKE_BUILD_PARALLEL_LEVEL=str(os.cpu_count() or 1),
        FHELIUM_RELEASE_CONFIGURATION_ID=configuration.id,
        FHELIUM_RELEASE_TOOLKIT_VERSION=(
            "" if root is None else configuration.toolkit(WINDOWS).version
        ),
        FHELIUM_RELEASE_CUDA_ARCHITECTURES=";".join(
            configuration.cuda_architectures
        ),
        FHELIUM_RELEASE_CUDA_PTX_ARCHITECTURES=";".join(
            configuration.cuda_ptx_architectures
        ),
        FHELIUM_RELEASE_TORCH_REQUIREMENT=configuration.torch_requirement,
    )
    output = absolute_path(args.output)
    wheelhouse = output / configuration.id / args.python_abi / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    for previous in wheelhouse.glob("fhelium-*-win_amd64.whl"):
        previous.unlink()
    run(
        str(python),
        "-I",
        "-m",
        "pip",
        "wheel",
        str(source),
        "--no-build-isolation",
        "--no-deps",
        "--no-cache-dir",
        "--wheel-dir",
        str(wheelhouse),
        "--config-settings",
        f"build-dir={work / 'build'}",
        cwd=source,
        env=environment,
    )
    wheels = list(wheelhouse.glob("fhelium-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {wheels!r}")
    dumpbin = shutil.which("dumpbin.exe", path=environment["PATH"])
    if dumpbin is None:
        raise RuntimeError("selected MSVC environment has no dumpbin.exe")
    check_wheel(
        wheels[0],
        configuration=configuration,
        python_abi=args.python_abi,
        source=source,
        dumpbin=Path(dumpbin),
        toolkit=root,
        work_root=work,
    )
    if args.verify_install:
        verify_installed_wheel(
            wheels[0],
            base_python=base_python,
            configuration=configuration,
            source=source,
            work_root=work,
        )
    shutil.rmtree(work)
    print(wheels[0])
    return wheels[0]


def _metadata(archive: ZipFile) -> Message:
    names = [
        name
        for name in archive.namelist()
        if name.endswith(".dist-info/METADATA")
    ]
    if len(names) != 1:
        raise RuntimeError("wheel must contain one METADATA file")
    return email.parser.Parser().parsestr(archive.read(names[0]).decode())


def check_wheel(
    wheel: Path,
    *,
    configuration: Configuration,
    python_abi: str,
    source: Path,
    dumpbin: Path,
    toolkit: Path | None,
    work_root: Path,
) -> None:
    version = tomllib.loads(
        (source / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    expected_suffix = f".cp3{python_abi[3:5]}-win_amd64.pyd"
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        ops_binaries = [
            name
            for name in names
            if name.startswith("fhelium/native/torchops/_ops")
            and name.endswith(".pyd")
        ]
        cuda_info_binaries = [
            name
            for name in names
            if name.startswith("fhelium/native/cuda/cuda_info")
            and name.endswith(".pyd")
        ]
        manifests = [
            name
            for name in names
            if Path(name).name.startswith("_build_manifest.")
            and name.endswith(".json")
        ]
        if (
            len(ops_binaries) != 1
            or len(cuda_info_binaries) != int(configuration.has_cuda)
            or len(manifests) != 1
            or not all(
                name.endswith(expected_suffix)
                for name in (*ops_binaries, *cuda_info_binaries)
            )
        ):
            raise RuntimeError("wheel native-extension layout is invalid")
        metadata = _metadata(archive)
        if (
            metadata["Version"] != version
            or metadata.get_all("Requires-Dist", []).count(
                configuration.torch_requirement
            )
            != 1
        ):
            raise RuntimeError("wheel project or Torch metadata is invalid")
        manifest = json.loads(archive.read(manifests[0]))
        selected_toolkit = configuration.toolkit(WINDOWS)
        if toolkit is not None and selected_toolkit is None:
            raise RuntimeError("CUDA wheel has no matrix Toolkit declaration")
        expected = {
            "project_version": version,
            "native_backends": list(configuration.native_backends),
            "release_configuration_id": configuration.id,
            "build_toolkit_version": (
                None if selected_toolkit is None else selected_toolkit.version
            ),
            "build_cuda_architectures": list(configuration.cuda_architectures),
            "build_cuda_ptx_architectures": list(
                configuration.cuda_ptx_architectures
            ),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"native manifest {key} differs: {manifest.get(key)!r}"
                )
        with tempfile.TemporaryDirectory(
            prefix="fhelium-wheel-", dir=work_root
        ) as temporary:
            binary = Path(temporary) / Path(ops_binaries[0]).name
            binary.write_bytes(archive.read(ops_binaries[0]))
            text = run(
                str(dumpbin),
                "/dependents",
                str(binary),
                cwd=source,
                capture=True,
            ).stdout.casefold()
            if "libiomp5md.dll" not in text or "vcomp" in text:
                raise RuntimeError(
                    "wheel must use Torch's OpenMP runtime without VCOMP"
                )
            raw = binary.read_bytes()
            forbidden = list(
                dict.fromkeys(
                    (source, source.resolve(), work_root, work_root.resolve())
                )
            )
            if toolkit is not None:
                forbidden.extend((toolkit, cuda_compiler_root(toolkit)))
            for path in forbidden:
                for encoded in (
                    str(path).encode().lower(),
                    str(path).encode("utf-16le").lower(),
                ):
                    if encoded in raw.lower():
                        raise RuntimeError(f"wheel contains local path: {path}")
            if toolkit is not None:
                cuobjdump = toolkit / "bin" / "cuobjdump.exe"
                sass = run(
                    str(cuobjdump),
                    "--list-elf",
                    str(binary),
                    cwd=source,
                    capture=True,
                ).stdout
                ptx = run(
                    str(cuobjdump),
                    "--list-ptx",
                    str(binary),
                    cwd=source,
                    capture=True,
                ).stdout
                sass_architectures = set(re.findall(r"sm_([0-9]+)", sass))
                ptx_architectures = set(
                    re.findall(r"(?:sm|compute)_([0-9]+)", ptx)
                )
                expected_sass = set(configuration.cuda_architectures)
                expected_ptx = set(configuration.cuda_ptx_architectures)
                if sass_architectures != expected_sass:
                    raise RuntimeError(
                        "wheel CUDA cubin targets differ: "
                        f"expected={sorted(expected_sass)!r}, "
                        f"actual={sorted(sass_architectures)!r}"
                    )
                if ptx_architectures != expected_ptx:
                    raise RuntimeError(
                        "wheel CUDA PTX targets differ: "
                        f"expected={sorted(expected_ptx)!r}, "
                        f"actual={sorted(ptx_architectures)!r}"
                    )


def verify_installed_wheel(
    wheel: Path,
    *,
    base_python: Path,
    configuration: Configuration,
    source: Path,
    work_root: Path,
) -> None:
    """Install one wheel in a clean environment and run native operators."""

    root = work_root / "installed-wheel-verification"
    shutil.rmtree(root, ignore_errors=True)
    environment = clean_environment()
    run(
        str(base_python),
        "-I",
        "-m",
        "venv",
        "--copies",
        str(root),
        cwd=source,
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
    run(
        str(python),
        "-I",
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--extra-index-url",
        configuration.torch_index_url,
        str(wheel),
        cwd=root,
        env=environment,
    )
    project_version = tomllib.loads(
        (source / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    code = f"""
import fhelium
import torch
from fhelium.native import native_status
assert fhelium.__version__ == {project_version!r}
status = native_status()
assert status.available and set(status.backends) == {set(configuration.native_backends)!r}
x = torch.tensor([[[1, 2]]], dtype=torch.int64)
p = torch.zeros((8, 1), dtype=torch.int64)
p[0, 0] = 34
assert torch.equal(torch.ops.fhelium_rns_ops.add_standard(x, x, p), x + x)
if {configuration.has_cuda!r}:
    x = x.cuda()
    p = p.cuda()
    assert torch.equal(torch.ops.fhelium_rns_ops.add_standard(x, x, p).cpu(), x.cpu() + x.cpu())
"""
    run(str(python), "-I", "-c", code, cwd=root, env=environment)
    shutil.rmtree(root)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument(
        "--python-abi", required=True, choices=("cp312-cp312", "cp313-cp313")
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--cuda-toolkit-root", type=Path)
    parser.add_argument("--verify-install", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(args())
