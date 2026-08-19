#!/usr/bin/env python3
"""Validate and query the FHElium binary release matrix."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATRIX_PATH = Path(__file__).with_name("release_matrix.json")
SCHEMA_PATH = Path(__file__).with_name("release_matrix.schema.json")


@dataclass(frozen=True)
class ReleaseConfiguration:
    """One exact Torch and native-backend build configuration."""

    id: str
    role: str
    torch_requirement: str
    torch_index_url: str
    torch_runtime_version: str
    torch_cuda_version: str | None
    torch_cxx11_abi: bool
    native_backends: tuple[str, ...]
    toolkit_version: str | None
    toolkit_rpms: tuple[str, ...]
    cuda_architectures: tuple[str, ...]
    expected_cuda_runtime_soname: str | None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ReleaseConfiguration:
        return cls(
            id=value["id"],
            role=value["role"],
            torch_requirement=value["torch_requirement"],
            torch_index_url=value["torch_index_url"],
            torch_runtime_version=value["torch_runtime_version"],
            torch_cuda_version=value["torch_cuda_version"],
            torch_cxx11_abi=value["torch_cxx11_abi"],
            native_backends=tuple(value["native_backends"]),
            toolkit_version=value["toolkit_version"],
            toolkit_rpms=tuple(value["toolkit_rpms"]),
            cuda_architectures=tuple(value["cuda_architectures"]),
            expected_cuda_runtime_soname=value["expected_cuda_runtime_soname"],
        )

    @property
    def has_cuda(self) -> bool:
        return "cuda" in self.native_backends

    @property
    def builder_environment(self) -> str:
        if self.torch_cuda_version is None:
            return "cpu"
        return f"cuda-{self.torch_cuda_version}"


@dataclass(frozen=True)
class ReleaseMatrix:
    """Validated build configurations and shared Linux build inputs."""

    schema_version: int
    project: str
    platform: str
    python_abis: tuple[str, ...]
    scikit_build_core_version: str
    jsonschema_version: str
    builder_image: str
    artifact_base_url: str
    simple_index_base_url: str
    publication: str
    builder_dockerfile_by_environment: dict[str, str]
    configurations: tuple[ReleaseConfiguration, ...]

    def configuration(self, configuration_id: str) -> ReleaseConfiguration:
        matches = [
            item for item in self.configurations if item.id == configuration_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"release configuration {configuration_id!r} does not exist"
            )
        return matches[0]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_json_schema(value: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as error:
        raise RuntimeError(
            "jsonschema is required to validate the release matrix"
        ) from error
    jsonschema.Draft202012Validator(read_json(SCHEMA_PATH)).validate(value)


def version_series(runtime_version: str) -> str:
    """Return the Torch major.minor selector identity."""

    match = re.fullmatch(
        r"([0-9]+\.[0-9]+)\.[0-9]+\+[a-z0-9]+", runtime_version
    )
    if match is None:
        raise ValueError(f"invalid Torch runtime version: {runtime_version}")
    return match.group(1)


def selector_catalog(
    matrix: ReleaseMatrix, *, fhelium_version: str
) -> dict[str, Any]:
    """Project the release matrix into the documentation selector catalog."""

    binary_recipes = []
    torch_distributions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in matrix.configurations:
        torch_series = version_series(item.torch_runtime_version)
        if item.torch_cuda_version is None:
            compute = "cpu"
        else:
            compute = f"cuda-{item.torch_cuda_version.replace('.', '')}"
        key = ("linux-x86_64", torch_series, compute)
        torch_distributions[key] = {
            "os": key[0],
            "torch": torch_series,
            "compute": compute,
            "requirement": item.torch_requirement,
            "index_url": item.torch_index_url,
        }
        binary_recipes.append(
            {
                "os": key[0],
                "method": "prebuilt-pip",
                "torch": torch_series,
                "compute": compute,
                "configuration": item.id,
                "fhelium_version": fhelium_version,
                "simple_index_url": (
                    f"{matrix.simple_index_base_url}/{item.id}/simple/"
                ),
                "published": False,
            }
        )

        if not item.has_cuda:
            torch_distributions[("macos-arm64", torch_series, "cpu")] = {
                "os": "macos-arm64",
                "torch": torch_series,
                "compute": "cpu",
                "requirement": f"torch=={item.torch_runtime_version.removesuffix('+cpu')}",
                "index_url": item.torch_index_url,
            }

    return {
        "schema_version": 1,
        "fhelium_version": fhelium_version,
        "torch_distributions": list(torch_distributions.values()),
        "binary_recipes": binary_recipes,
        "source_profiles": [
            {
                "os": "linux-x86_64",
                "method": "source-pip",
                "native_backend_by_compute": {
                    "cpu": "CPU",
                    "cuda": "CPU+CUDA",
                },
            },
            {
                "os": "macos-arm64",
                "method": "source-pip",
                "native_backend_by_compute": {"cpu": "CPU"},
            },
            {
                "os": "linux-x86_64",
                "method": "source-github",
                "native_backend_by_compute": {
                    "cpu": "CPU",
                    "cuda": "CPU+CUDA",
                },
            },
            {
                "os": "macos-arm64",
                "method": "source-github",
                "native_backend_by_compute": {"cpu": "CPU"},
            },
        ],
    }


def split_torch_identity(requirement: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"torch==([0-9]+(?:\.[0-9]+){2})\+([a-z0-9]+)", requirement
    )
    if match is None:
        raise ValueError(
            f"Torch requirement must use an exact local version: {requirement}"
        )
    return match.group(1), match.group(2)


def validate_semantics(matrix: ReleaseMatrix) -> None:
    configurations = matrix.configurations
    for label, values in (
        ("configuration id", [item.id for item in configurations]),
        ("configuration role", [item.role for item in configurations]),
    ):
        duplicates = sorted(
            {value for value in values if values.count(value) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")

    required_roles = {
        "current-cuda",
        "previous-cuda",
        "current-cpu",
        "previous-cpu",
    }
    actual_roles = {item.role for item in configurations}
    if actual_roles != required_roles:
        raise ValueError(
            f"configuration roles must be {sorted(required_roles)!r}, got {sorted(actual_roles)!r}"
        )
    if matrix.publication != "static-simple-repository":
        raise ValueError("publication must be static-simple-repository")
    for label, url in (
        ("artifact_base_url", matrix.artifact_base_url),
        ("simple_index_base_url", matrix.simple_index_base_url),
    ):
        if not url.startswith("https://"):
            raise ValueError(f"{label} must use HTTPS")
    expected_artifact_base = f"{matrix.simple_index_base_url}/artifacts"
    if matrix.artifact_base_url != expected_artifact_base:
        raise ValueError(
            "artifact_base_url must be the artifact path below "
            "simple_index_base_url"
        )

    expected_environments = {
        item.builder_environment for item in configurations
    }
    actual_environments = set(matrix.builder_dockerfile_by_environment)
    if actual_environments != expected_environments:
        raise ValueError(
            "builder environments must be "
            f"{sorted(expected_environments)!r}, got {sorted(actual_environments)!r}"
        )

    for item in configurations:
        _, torch_variant = split_torch_identity(item.torch_requirement)
        if (
            item.torch_requirement.removeprefix("torch==")
            != item.torch_runtime_version
        ):
            raise ValueError(
                f"{item.id}: Torch requirement and runtime version differ"
            )
        if not item.id.endswith(f"-{torch_variant}"):
            raise ValueError(
                f"{item.id}: id does not identify Torch variant {torch_variant}"
            )
        if not item.torch_index_url.endswith(f"/{torch_variant}"):
            raise ValueError(
                f"{item.id}: Torch index does not match {torch_variant}"
            )

        if item.has_cuda:
            if item.native_backends != ("cpu", "cuda"):
                raise ValueError(
                    f"{item.id}: CUDA wheel must contain CPU and CUDA backends"
                )
            if not all(
                value is not None
                for value in (
                    item.torch_cuda_version,
                    item.toolkit_version,
                    item.expected_cuda_runtime_soname,
                )
            ):
                raise ValueError(f"{item.id}: CUDA build fields are incomplete")
            if not item.cuda_architectures:
                raise ValueError(f"{item.id}: CUDA architectures are empty")
            if not item.toolkit_rpms:
                raise ValueError(f"{item.id}: CUDA toolkit RPMs are empty")
            assert item.torch_cuda_version is not None
            assert item.toolkit_version is not None
            if (
                ".".join(item.toolkit_version.split(".")[:2])
                != item.torch_cuda_version
            ):
                raise ValueError(
                    f"{item.id}: Torch CUDA and toolkit major.minor differ"
                )
        else:
            if item.native_backends != ("cpu",):
                raise ValueError(
                    f"{item.id}: CPU wheel must contain only the CPU backend"
                )
            if (
                any(
                    value is not None
                    for value in (
                        item.torch_cuda_version,
                        item.toolkit_version,
                        item.expected_cuda_runtime_soname,
                    )
                )
                or item.cuda_architectures
                or item.toolkit_rpms
            ):
                raise ValueError(f"{item.id}: CPU build contains CUDA fields")
            if torch_variant != "cpu":
                raise ValueError(
                    f"{item.id}: CPU build must use the CPU Torch index"
                )


def load_release_matrix(path: Path = MATRIX_PATH) -> ReleaseMatrix:
    raw = read_json(path)
    validate_json_schema(raw)
    matrix = ReleaseMatrix(
        schema_version=raw["schema_version"],
        project=raw["project"],
        platform=raw["platform"],
        python_abis=tuple(raw["python_abis"]),
        scikit_build_core_version=raw["scikit_build_core_version"],
        jsonschema_version=raw["jsonschema_version"],
        builder_image=raw["builder_image"],
        artifact_base_url=raw["artifact_base_url"],
        simple_index_base_url=raw["simple_index_base_url"],
        publication=raw["publication"],
        builder_dockerfile_by_environment=raw[
            "builder_dockerfile_by_environment"
        ],
        configurations=tuple(
            ReleaseConfiguration.from_mapping(item)
            for item in raw["configurations"]
        ),
    )
    validate_semantics(matrix)
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--json", action="store_true")
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("configuration")
    catalog_parser = subparsers.add_parser("docs-catalog")
    catalog_parser.add_argument("--project-version", required=True)
    catalog_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_release_matrix(args.matrix)
    if args.command == "validate":
        print(f"Validated {len(matrix.configurations)} release configurations")
    elif args.command == "list":
        if args.json:
            print(
                json.dumps(
                    [item.__dict__ for item in matrix.configurations],
                    indent=2,
                    default=list,
                )
            )
        else:
            for item in matrix.configurations:
                print(item.id)
    elif args.command == "show":
        print(
            json.dumps(
                matrix.configuration(args.configuration).__dict__,
                indent=2,
                default=list,
            )
        )
    elif args.command == "docs-catalog":
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
            print(args.output)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
