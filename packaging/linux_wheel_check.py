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

from packaging.requirements import Requirement


EXPECTED_OPS_RUNPATH = "$ORIGIN/../../../torch/lib"
FORBIDDEN_PATH_PREFIXES = ("/home/", "/project/", "/tmp/", "/usr/local/cuda")


def load_release_matrix_module(matrix_path: Path) -> ModuleType:
    implementation = Path(__file__).with_name("matrix.py")
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
        "--cuobjdump",
        type=Path,
        help="cuobjdump executable required for a CUDA wheel",
    )
    parser.add_argument(
        "--llvm-objdump",
        type=Path,
        help="LLVM objdump with HIP fat-binary support",
    )
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


def cuda_image_architectures(
    cuobjdump: Path, binary: Path, option: str
) -> set[str]:
    listing = subprocess.run(
        [str(cuobjdump), option, str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(re.findall(r"(?:sm|compute)_([0-9]+)", listing))


def hip_image_architectures(llvm_objdump: Path, binary: Path) -> set[str]:
    """Inspect HIP code objects inside the temporary extracted wheel tree."""
    listing = subprocess.run(
        [str(llvm_objdump), "--offloading", "--offload-fatbin", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    architectures = set(
        re.findall(
            r"hip-amdgcn-amd-amdhsa--(gfx[0-9a-f]+(?::(?:sramecc|xnack)[+-])*)",
            listing,
        )
    )
    objects = tuple(binary.parent.glob(binary.name + ".*.hip-amdgcn-*"))
    if not objects or not architectures:
        raise RuntimeError("HIP wheel has no inspectable AMD GPU code objects")
    for code_object in objects:
        header = subprocess.run(
            ["readelf", "--file-header", str(code_object)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "AMD GPU" not in header:
            raise RuntimeError(
                f"HIP code object is not an AMD GPU ELF: {code_object.name}"
            )
    return architectures


def main() -> None:
    args = parse_args()
    wheel = args.wheel.resolve()
    module = load_release_matrix_module(args.matrix.resolve())
    matrix = module.load_matrix(args.matrix.resolve())
    configuration = matrix.configuration(args.configuration)
    if configuration.has_cuda:
        if args.cuobjdump is None or not args.cuobjdump.is_file():
            raise RuntimeError("CUDA wheel validation requires --cuobjdump")
    elif args.cuobjdump is not None:
        raise RuntimeError(
            "non-CUDA wheel validation does not accept --cuobjdump"
        )
    if configuration.has_hip and (
        args.llvm_objdump is None or not args.llvm_objdump.is_file()
    ):
        raise RuntimeError("HIP wheel validation requires --llvm-objdump")
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
                configuration.expected_runtime_soname,
            }
        )
    if configuration.has_hip:
        expected_external_libraries.update(
            {
                "libc10_hip.so",
                "libtorch.so",
                "libtorch_hip.so",
                configuration.expected_runtime_soname,
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
        requirements = package_metadata.get_all("Requires-Dist", [])
        torch_requirements = [
            value
            for value in requirements
            if Requirement(value).name.lower() == "torch"
            and not Requirement(value).extras
        ]
        for requirement in configuration.runtime_requirements:
            matches = [
                value
                for value in requirements
                if Requirement(value) == Requirement(requirement)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"wheel runtime requirement differs from configuration: {requirement}"
                )
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
        expected_cuda_info_count = 1 if configuration.has_gpu else 0
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
        if manifest.get("build_cuda_ptx_architectures") != list(
            configuration.cuda_ptx_architectures
        ):
            raise RuntimeError(
                "native manifest CUDA PTX architectures do not match"
            )
        if manifest.get("build_hip_architectures", []) != list(
            configuration.hip_architectures
        ):
            raise RuntimeError("native manifest HIP architectures do not match")
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
        if torch_identity.get("hip_version") != configuration.torch_hip_version:
            raise RuntimeError(
                "native manifest Torch HIP version does not match"
            )
        if torch_identity.get("cxx11_abi") != configuration.torch_cxx11_abi:
            raise RuntimeError("native manifest Torch C++ ABI does not match")

        abi = re.search(r"-(cp[0-9]+)-\1-", wheel.name)
        python_identity = manifest.get("python", {})
        if abi is None or not python_identity.get("soabi", "").startswith(
            "cpython-" + abi.group(1).removeprefix("cp") + "-"
        ):
            raise RuntimeError(
                "native manifest Python ABI differs from wheel tag"
            )
        suffix = python_identity.get("ext_suffix")
        if ops_files[0].name != "_ops" + str(suffix):
            raise RuntimeError(
                "native extension suffix differs from manifest Python ABI"
            )

        ops_section = dynamic_section(ops_files[0])
        ops_paths = dynamic_values(ops_section, "(?:RPATH|RUNPATH)")
        if ops_paths != [EXPECTED_OPS_RUNPATH]:
            raise RuntimeError(
                f"_ops runtime path must be {EXPECTED_OPS_RUNPATH!r}, got {ops_paths!r}"
            )
        assert_no_build_paths(ops_section, label="_ops dynamic section")

        if args.cuobjdump is not None:
            actual_cubins = cuda_image_architectures(
                args.cuobjdump, ops_files[0], "--list-elf"
            )
            actual_ptx = cuda_image_architectures(
                args.cuobjdump, ops_files[0], "--list-ptx"
            )
            expected_cubins = set(configuration.cuda_architectures)
            expected_ptx = set(configuration.cuda_ptx_architectures)
            if actual_cubins != expected_cubins:
                raise RuntimeError(
                    "wheel CUDA cubin targets differ: "
                    f"expected={sorted(expected_cubins)!r}, "
                    f"actual={sorted(actual_cubins)!r}"
                )
            if actual_ptx != expected_ptx:
                raise RuntimeError(
                    "wheel CUDA PTX targets differ: "
                    f"expected={sorted(expected_ptx)!r}, "
                    f"actual={sorted(actual_ptx)!r}"
                )

        if configuration.has_hip:
            actual_hip = hip_image_architectures(
                args.llvm_objdump, ops_files[0]
            )
            if actual_hip != set(configuration.hip_architectures):
                raise RuntimeError(
                    f"wheel HIP code-object targets differ: expected={configuration.hip_architectures!r}, actual={sorted(actual_hip)!r}"
                )

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
                (
                    "libtorch",
                    "libc10",
                    "libcuda",
                    "libcudart",
                    "libnvrtc",
                    "libamdhip",
                    "libhsa-runtime",
                    "libhiprtc",
                )
            )
        }
        unexpected = external - expected_external_libraries
        if unexpected:
            raise RuntimeError(
                "unexpected Torch/accelerator external libraries: "
                + ", ".join(sorted(unexpected))
            )
        missing = expected_external_libraries - external
        if missing:
            raise RuntimeError(
                "missing Torch/accelerator external libraries: "
                + ", ".join(sorted(missing))
            )

    print(f"Validated Linux native wheel: {wheel}")


if __name__ == "__main__":
    main()
