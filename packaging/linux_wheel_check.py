#!/usr/bin/env python3
"""Validate Linux native-wheel tags, dependencies, and runtime paths."""

from __future__ import annotations

import argparse
from email.parser import Parser
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile


EXPECTED_OPS_RUNPATH = "$ORIGIN/../../../torch/lib"
FORBIDDEN_PATH_PREFIXES = ("/home/", "/project/", "/tmp/", "/usr/local/cuda")


def load_release_matrix_module(matrix_path: Path) -> ModuleType:
    implementation = matrix_path.with_name("matrix.py")
    spec = importlib.util.spec_from_file_location(
        "_fhelium_release_matrix", implementation
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {implementation}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--project-version", required=True)
    parser.add_argument(
        "--platform-tag",
        default="manylinux_2_28_x86_64",
        help="required tag in the wheel filename and WHEEL metadata",
    )
    return parser.parse_args()


def dynamic_section(path: Path) -> str:
    return subprocess.run(
        ["readelf", "--dynamic", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def dynamic_values(section: str, field: str) -> list[str]:
    return re.findall(rf"\({field}\).*?\[(.*?)\]", section)


def assert_no_build_paths(text: str, *, label: str) -> None:
    leaked = sorted(
        prefix for prefix in FORBIDDEN_PATH_PREFIXES if prefix in text
    )
    if leaked:
        raise RuntimeError(f"{label} contains build paths: {', '.join(leaked)}")


def main() -> None:
    args = parse_args()
    wheel = args.wheel.resolve()
    module = load_release_matrix_module(args.matrix.resolve())
    matrix = module.load_matrix(args.matrix.resolve())
    configuration = matrix.configuration(args.configuration)
    expected_external_libraries = {
        "libc10.so",
        "libtorch_cpu.so",
    }
    if configuration.has_cuda:
        expected_external_libraries.update(
            {
                "libc10_cuda.so",
                "libtorch.so",
                "libtorch_cuda.so",
                configuration.expected_cuda_runtime_soname,
            }
        )
    if args.platform_tag not in wheel.name:
        raise RuntimeError(
            f"wheel filename does not contain {args.platform_tag!r}: {wheel.name}"
        )

    with tempfile.TemporaryDirectory(
        prefix="fhelium-wheel-check-"
    ) as temporary:
        root = Path(temporary)
        with ZipFile(wheel) as archive:
            archive.extractall(root)
            wheel_metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/WHEEL")
            ]
            package_metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
        if len(wheel_metadata_names) != 1:
            raise RuntimeError(
                "wheel must contain exactly one WHEEL metadata file"
            )
        metadata = (root / wheel_metadata_names[0]).read_text(encoding="utf-8")
        if f"-{args.platform_tag}" not in metadata:
            raise RuntimeError(
                f"WHEEL metadata does not contain platform tag {args.platform_tag!r}"
            )
        if len(package_metadata_names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA file")
        package_metadata = Parser().parsestr(
            (root / package_metadata_names[0]).read_text(encoding="utf-8")
        )
        torch_requirements = [
            value
            for value in package_metadata.get_all("Requires-Dist", [])
            if value.lower().startswith("torch")
        ]
        if torch_requirements != [configuration.torch_requirement]:
            raise RuntimeError(
                "wheel Torch requirement does not match release configuration: "
                f"expected={configuration.torch_requirement!r}, "
                f"actual={torch_requirements!r}"
            )

        ops_files = list(root.glob("fhelium/native/torchops/_ops*.so"))
        cuda_info_files = list(root.glob("fhelium/native/cuda/cuda_info*.so"))
        if len(ops_files) != 1:
            raise RuntimeError(
                "wheel must contain exactly one native _ops file"
            )
        expected_cuda_info_count = 1 if configuration.has_cuda else 0
        if len(cuda_info_files) != expected_cuda_info_count:
            raise RuntimeError(
                f"{configuration.id} must contain {expected_cuda_info_count} "
                f"cuda_info module, found {len(cuda_info_files)}"
            )

        manifests = list(
            root.glob("fhelium/native/torchops/_build_manifest.*.json")
        )
        if len(manifests) != 1:
            raise RuntimeError("wheel must contain exactly one native manifest")
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        if manifest.get("native_backends") != list(
            configuration.native_backends
        ):
            raise RuntimeError(
                "native manifest backends do not match release configuration"
            )
        if manifest.get("project_version") != args.project_version:
            raise RuntimeError("native manifest project version does not match")
        if manifest.get("release_configuration_id") != configuration.id:
            raise RuntimeError(
                "native manifest release configuration does not match"
            )
        if (
            manifest.get("build_toolkit_version")
            != configuration.toolkit_version
        ):
            raise RuntimeError("native manifest toolkit version does not match")
        if manifest.get("build_cuda_architectures") != list(
            configuration.cuda_architectures
        ):
            raise RuntimeError(
                "native manifest CUDA architectures do not match"
            )
        torch_identity = manifest.get("torch", {})
        if torch_identity.get("version") != configuration.torch_runtime_version:
            raise RuntimeError("native manifest Torch version does not match")
        if (
            torch_identity.get("cuda_version")
            != configuration.torch_cuda_version
        ):
            raise RuntimeError(
                "native manifest Torch CUDA version does not match"
            )
        if torch_identity.get("cxx11_abi") != configuration.torch_cxx11_abi:
            raise RuntimeError("native manifest Torch C++ ABI does not match")

        ops_section = dynamic_section(ops_files[0])
        ops_paths = dynamic_values(ops_section, "(?:RPATH|RUNPATH)")
        if ops_paths != [EXPECTED_OPS_RUNPATH]:
            raise RuntimeError(
                f"_ops runtime path must be {EXPECTED_OPS_RUNPATH!r}, got {ops_paths!r}"
            )
        assert_no_build_paths(ops_section, label="_ops dynamic section")

        for cuda_info in cuda_info_files:
            section = dynamic_section(cuda_info)
            paths = dynamic_values(section, "(?:RPATH|RUNPATH)")
            if paths:
                raise RuntimeError(
                    f"cuda_info must not contain RPATH/RUNPATH, got {paths!r}"
                )
            assert_no_build_paths(section, label="cuda_info dynamic section")

        external = {
            library
            for native_file in (*ops_files, *cuda_info_files)
            for library in dynamic_values(
                dynamic_section(native_file), "NEEDED"
            )
            if library.startswith(
                ("libtorch", "libc10", "libcuda", "libcudart", "libnvrtc")
            )
        }
        unexpected = external - expected_external_libraries
        if unexpected:
            raise RuntimeError(
                "unexpected Torch/CUDA external libraries: "
                + ", ".join(sorted(unexpected))
            )
        missing = expected_external_libraries - external
        if missing:
            raise RuntimeError(
                "missing Torch/CUDA external libraries: "
                + ", ".join(sorted(missing))
            )

    print(f"Validated Linux native wheel: {wheel}")


if __name__ == "__main__":
    main()
