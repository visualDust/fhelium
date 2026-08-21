#!/usr/bin/env python3
"""Validate and query FHElium release configurations."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MATRIX_PATH = Path(__file__).with_name("release_matrix.json")
SCHEMA_PATH = Path(__file__).with_name("release_matrix.schema.json")
LINUX = "manylinux_2_28_x86_64"
WINDOWS = "win_amd64"


@dataclass(frozen=True)
class Toolkit:
    """Platform-specific CUDA Toolkit and runtime identity."""

    version: str
    packages: tuple[str, ...]
    runtime_library: str

    @classmethod
    def parse(cls, value: dict[str, Any]) -> Toolkit:
        return cls(
            version=value["version"],
            packages=tuple(value["packages"]),
            runtime_library=value["runtime_library"],
        )


@dataclass(frozen=True)
class Platform:
    """One wheel platform and its builder declaration."""

    id: str
    operating_system: str
    architecture: str
    catalog_os: str
    builder: dict[str, Any]

    @classmethod
    def parse(cls, value: dict[str, Any]) -> Platform:
        return cls(
            id=value["id"],
            operating_system=value["operating_system"],
            architecture=value["architecture"],
            catalog_os=value["catalog_os"],
            builder=dict(value["builder"]),
        )


@dataclass(frozen=True)
class Configuration:
    """One Torch/native identity and the platforms that publish it."""

    id: str
    role: str
    platform_targets: tuple[str, ...]
    torch_requirement: str
    torch_index_url: str
    torch_runtime_version: str
    torch_cuda_version: str | None
    torch_cxx11_abi: bool
    native_backends: tuple[str, ...]
    toolkits: dict[str, Toolkit | None]
    cuda_architectures: tuple[str, ...]
    cuda_ptx_architectures: tuple[str, ...]

    @classmethod
    def parse(cls, value: dict[str, Any]) -> Configuration:
        return cls(
            id=value["id"],
            role=value["role"],
            platform_targets=tuple(value["platform_targets"]),
            torch_requirement=value["torch_requirement"],
            torch_index_url=value["torch_index_url"],
            torch_runtime_version=value["torch_runtime_version"],
            torch_cuda_version=value["torch_cuda_version"],
            torch_cxx11_abi=value["torch_cxx11_abi"],
            native_backends=tuple(value["native_backends"]),
            toolkits={
                platform: None if toolkit is None else Toolkit.parse(toolkit)
                for platform, toolkit in value["toolkits"].items()
            },
            cuda_architectures=tuple(value["cuda_architectures"]),
            cuda_ptx_architectures=tuple(value["cuda_ptx_architectures"]),
        )

    @property
    def has_cuda(self) -> bool:
        return "cuda" in self.native_backends

    def supports(self, platform: str) -> bool:
        return platform in self.platform_targets

    def toolkit(self, platform: str) -> Toolkit | None:
        if platform not in self.toolkits:
            raise ValueError(
                f"{self.id} has no toolkit declaration for {platform}"
            )
        return self.toolkits[platform]

    @property
    def builder_environment(self) -> str:
        toolkit = self.toolkit(LINUX)
        return "cpu" if toolkit is None else f"cuda-{self.torch_cuda_version}"

    @property
    def toolkit_version(self) -> str | None:
        toolkit = self.toolkit(LINUX)
        return None if toolkit is None else toolkit.version

    @property
    def toolkit_rpms(self) -> tuple[str, ...]:
        toolkit = self.toolkit(LINUX)
        return () if toolkit is None else toolkit.packages

    @property
    def expected_cuda_runtime_soname(self) -> str | None:
        toolkit = self.toolkit(LINUX)
        return None if toolkit is None else toolkit.runtime_library


@dataclass(frozen=True)
class Matrix:
    """Release platforms, configurations, and shared publication values."""

    project: str
    requires_python: str
    python_abis: tuple[str, ...]
    platforms: tuple[Platform, ...]
    configurations: tuple[Configuration, ...]
    scikit_build_core_version: str
    jsonschema_version: str
    artifact_base_url: str
    simple_index_base_url: str
    publication: str

    def platform(self, platform_id: str) -> Platform:
        matches = [item for item in self.platforms if item.id == platform_id]
        if len(matches) != 1:
            raise ValueError(f"unknown release platform: {platform_id}")
        return matches[0]

    def configuration(self, configuration_id: str) -> Configuration:
        matches = [
            item for item in self.configurations if item.id == configuration_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"unknown release configuration: {configuration_id}"
            )
        return matches[0]

    def configurations_for(self, platform: str) -> tuple[Configuration, ...]:
        self.platform(platform)
        return tuple(
            item for item in self.configurations if item.supports(platform)
        )

    @property
    def builder_image(self) -> str:
        return str(self.platform(LINUX).builder["image"])

    @property
    def builder_dockerfile_by_environment(self) -> dict[str, str]:
        return dict(self.platform(LINUX).builder["dockerfile_by_environment"])


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _torch_variant(requirement: str) -> str:
    match = re.fullmatch(
        r"torch==[0-9]+(?:\.[0-9]+){2}\+([a-z0-9]+)", requirement
    )
    if match is None:
        raise ValueError(f"Torch requirement is not pinned: {requirement}")
    return match.group(1)


def _validate(matrix: Matrix) -> None:
    for label, values in (
        ("platform", [item.id for item in matrix.platforms]),
        ("configuration", [item.id for item in matrix.configurations]),
        ("role", [item.role for item in matrix.configurations]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} declaration")
    if {item.id for item in matrix.platforms} != {LINUX, WINDOWS}:
        raise ValueError("release platforms must be Linux and Windows x64")
    if matrix.python_abis != ("cp312-cp312", "cp313-cp313"):
        raise ValueError("release Python ABIs must be CPython 3.12 and 3.13")
    if matrix.publication != "static-simple-repository":
        raise ValueError("unsupported publication model")
    if matrix.artifact_base_url != f"{matrix.simple_index_base_url}/artifacts":
        raise ValueError(
            "artifact URL must be below the Simple Repository root"
        )

    for configuration in matrix.configurations:
        variant = _torch_variant(configuration.torch_requirement)
        if (
            configuration.torch_runtime_version
            != configuration.torch_requirement.removeprefix("torch==")
        ):
            raise ValueError(f"{configuration.id}: Torch identities differ")
        if not configuration.id.endswith(f"-{variant}"):
            raise ValueError(
                f"{configuration.id}: id does not identify {variant}"
            )
        if set(configuration.platform_targets) != set(configuration.toolkits):
            raise ValueError(
                f"{configuration.id}: platform/toolkit keys differ"
            )
        for platform in configuration.platform_targets:
            matrix.platform(platform)
        if configuration.has_cuda:
            if configuration.native_backends != ("cpu", "cuda"):
                raise ValueError(
                    f"{configuration.id}: invalid CUDA backend set"
                )
            if not configuration.cuda_architectures:
                raise ValueError(
                    f"{configuration.id}: missing CUDA architectures"
                )
            if not set(configuration.cuda_ptx_architectures) <= set(
                configuration.cuda_architectures
            ):
                raise ValueError(
                    f"{configuration.id}: PTX target lacks SASS target"
                )
            if any(
                configuration.toolkit(platform) is None
                for platform in configuration.platform_targets
            ):
                raise ValueError(f"{configuration.id}: missing CUDA Toolkit")
        elif (
            configuration.native_backends != ("cpu",)
            or any(
                configuration.toolkit(platform) is not None
                for platform in configuration.platform_targets
            )
            or configuration.cuda_architectures
            or configuration.cuda_ptx_architectures
        ):
            raise ValueError(
                f"{configuration.id}: CPU configuration contains CUDA data"
            )

    cells = sum(
        len(matrix.python_abis) * len(configuration.platform_targets)
        for configuration in matrix.configurations
    )
    if cells != 16:
        raise ValueError(f"release matrix must declare 16 cells, got {cells}")
    linux_environments = {
        item.builder_environment for item in matrix.configurations_for(LINUX)
    }
    if linux_environments != set(matrix.builder_dockerfile_by_environment):
        raise ValueError("Linux Dockerfiles do not match matrix environments")


def load_matrix(path: Path = MATRIX_PATH) -> Matrix:
    raw = _json(path)
    try:
        import jsonschema
    except ImportError as error:
        raise RuntimeError(
            "jsonschema is required to validate the release matrix"
        ) from error
    jsonschema.Draft202012Validator(_json(SCHEMA_PATH)).validate(raw)
    matrix = Matrix(
        project=raw["project"],
        requires_python=raw["requires_python"],
        python_abis=tuple(raw["python_abis"]),
        platforms=tuple(
            Platform.parse(item) for item in raw["platform_targets"]
        ),
        configurations=tuple(
            Configuration.parse(item) for item in raw["configurations"]
        ),
        scikit_build_core_version=raw["scikit_build_core_version"],
        jsonschema_version=raw["jsonschema_version"],
        artifact_base_url=raw["artifact_base_url"],
        simple_index_base_url=raw["simple_index_base_url"],
        publication=raw["publication"],
    )
    _validate(matrix)
    return matrix


def _series(version: str) -> str:
    return ".".join(version.split(".")[:2])


def selector_catalog(matrix: Matrix, *, fhelium_version: str) -> dict[str, Any]:
    distributions: dict[tuple[str, str, str], dict[str, str]] = {}
    recipes = []
    for configuration in matrix.configurations:
        series = _series(configuration.torch_runtime_version)
        compute = (
            "cpu"
            if configuration.torch_cuda_version is None
            else f"cuda-{configuration.torch_cuda_version.replace('.', '')}"
        )
        for platform_id in configuration.platform_targets:
            platform = matrix.platform(platform_id)
            key = (platform.catalog_os, series, compute)
            distributions[key] = {
                "os": key[0],
                "torch": series,
                "compute": compute,
                "requirement": configuration.torch_requirement,
                "index_url": configuration.torch_index_url,
            }
            recipes.append(
                {
                    "os": key[0],
                    "method": "prebuilt-pip",
                    "torch": series,
                    "compute": compute,
                    "configuration": configuration.id,
                    "fhelium_version": fhelium_version,
                    "simple_index_url": f"{matrix.simple_index_base_url}/{configuration.id}/simple/",
                    "published": False,
                }
            )
        if not configuration.has_cuda:
            distributions[("macos-arm64", series, "cpu")] = {
                "os": "macos-arm64",
                "torch": series,
                "compute": "cpu",
                "requirement": f"torch=={configuration.torch_runtime_version.removesuffix('+cpu')}",
                "index_url": configuration.torch_index_url,
            }
    source_profiles = [
        {
            "os": os_name,
            "method": method,
            "native_backend_by_compute": (
                {"cpu": "CPU"}
                if os_name == "macos-arm64"
                else {"cpu": "CPU", "cuda": "CPU+CUDA"}
            ),
        }
        for os_name in ("linux-x86_64", "windows-x86_64", "macos-arm64")
        for method in ("source-pip", "source-github")
    ]
    return {
        "schema_version": 1,
        "fhelium_version": fhelium_version,
        "torch_distributions": list(distributions.values()),
        "binary_recipes": recipes,
        "source_profiles": source_profiles,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    cells = commands.add_parser("cells")
    cells.add_argument("--platform", choices=(LINUX, WINDOWS))
    cells.add_argument("--json", action="store_true")
    show = commands.add_parser("show")
    show.add_argument("configuration")
    catalog = commands.add_parser("docs-catalog")
    catalog.add_argument("--project-version", required=True)
    catalog.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _args()
    matrix = load_matrix(args.matrix)
    if args.command == "validate":
        count = sum(
            len(matrix.python_abis) * len(item.platform_targets)
            for item in matrix.configurations
        )
        print(
            f"Validated {len(matrix.configurations)} configurations and {count} cells"
        )
        return
    if args.command == "show":
        print(
            json.dumps(
                asdict(matrix.configuration(args.configuration)), indent=2
            )
        )
        return
    if args.command == "cells":
        cells = [
            {
                "configuration": configuration.id,
                "platform": platform,
                "python_abi": abi,
            }
            for configuration in matrix.configurations
            for platform in configuration.platform_targets
            for abi in matrix.python_abis
            if args.platform is None or platform == args.platform
        ]
        if args.json:
            print(json.dumps(cells, separators=(",", ":")))
        else:
            for cell in cells:
                print(
                    cell["configuration"], cell["platform"], cell["python_abi"]
                )
        return
    if args.command == "docs-catalog":
        text = (
            json.dumps(
                selector_catalog(matrix, fhelium_version=args.project_version),
                indent=2,
            )
            + "\n"
        )
        if args.output is None:
            print(text, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
