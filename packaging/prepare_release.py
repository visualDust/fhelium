#!/usr/bin/env python3
"""Prepare static PEP 503/691 indexes and a release manifest locally."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from urllib.parse import quote
from zipfile import ZipFile

from packaging.utils import parse_wheel_filename


@dataclass(frozen=True)
class Artifact:
    configuration: str
    platform: str
    python_abi: str
    filename: str
    sha256: str
    size: int
    requires_python: str
    source: Path
    relative_path: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-version")
    parser.add_argument(
        "--prepared-at",
        help="ISO-8601 timestamp; defaults to SOURCE_DATE_EPOCH",
    )
    return parser.parse_args()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_requires_python(path: Path) -> str:
    with ZipFile(path) as archive:
        metadata_names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"{path} must contain one METADATA file")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    values = [
        line.removeprefix("Requires-Python:").strip()
        for line in metadata.splitlines()
        if line.startswith("Requires-Python:")
    ]
    if len(values) != 1:
        raise RuntimeError(f"{path} must declare exactly one Requires-Python")
    return values[0]


def wheel_abi(filename: str, supported: tuple[str, ...]) -> str:
    matches = [abi for abi in supported if f"-{abi}-" in filename]
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot identify one supported Python ABI from {filename!r}"
        )
    return matches[0]


def wheel_platform(filename: str) -> str:
    if filename.endswith("-win_amd64.whl"):
        return "win_amd64"
    if "manylinux" in filename and filename.endswith("_x86_64.whl"):
        return "manylinux_2_28_x86_64"
    raise RuntimeError(f"unsupported release wheel platform: {filename}")


def html_project_page(artifacts: list[Artifact]) -> str:
    anchors = [
        "<a "
        f'href="{html.escape(artifact.url)}#sha256={artifact.sha256}" '
        f'data-requires-python="{html.escape(artifact.requires_python)}">'
        f"{html.escape(artifact.filename)}</a><br>"
        for artifact in artifacts
    ]
    return "\n".join(
        (
            "<!doctype html>",
            '<html><head><meta name="pypi:repository-version" content="1.4">',
            "<title>Links for fhelium</title></head><body>",
            "<h1>Links for fhelium</h1>",
            *anchors,
            "</body></html>",
            "",
        )
    )


def json_project_page(artifacts: list[Artifact]) -> dict[str, object]:
    return {
        "meta": {"api-version": "1.4"},
        "name": "fhelium",
        "files": [
            {
                "filename": artifact.filename,
                "url": artifact.url,
                "hashes": {"sha256": artifact.sha256},
                "requires-python": artifact.requires_python,
            }
            for artifact in artifacts
        ],
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix_module().load_matrix()
    project = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    version = args.project_version or project["version"]
    if version != project["version"]:
        raise ValueError(
            "--project-version must equal the checked-out project version"
        )

    input_root = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    artifacts: list[Artifact] = []
    for configuration in matrix.configurations:
        wheel_dir = input_root / configuration.id
        wheels = sorted(
            {
                *wheel_dir.glob("*.whl"),
                *wheel_dir.glob("*/wheelhouse/*.whl"),
            }
        )
        expected_wheels = len(configuration.platform_targets) * len(
            matrix.python_abis
        )
        if len(wheels) != expected_wheels:
            raise RuntimeError(
                f"{configuration.id} must provide {expected_wheels} "
                f"wheels, found {len(wheels)}"
            )
        seen_cells: set[tuple[str, str]] = set()
        for wheel in wheels:
            abi = wheel_abi(wheel.name, matrix.python_abis)
            platform = wheel_platform(wheel.name)
            if not configuration.supports(platform):
                raise RuntimeError(
                    f"{configuration.id} does not publish {platform}"
                )
            distribution, parsed_version, _, _ = parse_wheel_filename(
                wheel.name
            )
            if str(distribution) != "fhelium" or str(parsed_version) != version:
                raise RuntimeError(
                    f"unexpected wheel project/version: {wheel.name}"
                )
            cell = (platform, abi)
            if cell in seen_cells:
                raise RuntimeError(
                    f"duplicate {configuration.id}/{platform}/{abi}"
                )
            seen_cells.add(cell)
            relative = f"artifacts/{version}/{configuration.id}/{wheel.name}"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheel, destination)
            artifacts.append(
                Artifact(
                    configuration=configuration.id,
                    platform=platform,
                    python_abi=abi,
                    filename=wheel.name,
                    sha256=sha256(destination),
                    size=destination.stat().st_size,
                    requires_python=wheel_requires_python(destination),
                    source=wheel,
                    relative_path=relative,
                    url=(
                        f"{matrix.artifact_base_url}/"
                        f"{quote(relative.removeprefix('artifacts/'), safe='/')}"
                    ),
                )
            )

    by_configuration = {
        configuration.id: sorted(
            [
                artifact
                for artifact in artifacts
                if artifact.configuration == configuration.id
            ],
            key=lambda item: item.filename,
        )
        for configuration in matrix.configurations
    }
    for configuration, files in by_configuration.items():
        project_dir = output / configuration / "simple" / "fhelium"
        project_dir.mkdir(parents=True)
        (project_dir / "index.html").write_text(
            html_project_page(files), encoding="utf-8"
        )
        write_json(project_dir / "index.json", json_project_page(files))
        root_dir = output / configuration / "simple"
        (root_dir / "index.html").write_text(
            "<!doctype html>\n<html><body>\n"
            '<a href="fhelium/">fhelium</a>\n'
            "</body></html>\n",
            encoding="utf-8",
        )
        write_json(
            root_dir / "index.json",
            {
                "meta": {"api-version": "1.4"},
                "projects": [{"name": "fhelium"}],
            },
        )

    if args.prepared_at is not None:
        prepared_at = args.prepared_at
    else:
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if source_date_epoch is None:
            raise RuntimeError(
                "SOURCE_DATE_EPOCH is required when --prepared-at is omitted"
            )
        prepared_at = datetime.fromtimestamp(
            int(source_date_epoch), UTC
        ).isoformat()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_value = {
        "schema_version": 1,
        "project": "fhelium",
        "project_version": version,
        "source_commit": source_commit,
        "prepared_at": prepared_at,
        "published": True,
        "artifacts": [
            {
                "configuration": artifact.configuration,
                "platform": artifact.platform,
                "python_abi": artifact.python_abi,
                "filename": artifact.filename,
                "relative_path": artifact.relative_path,
                "url": artifact.url,
                "size": artifact.size,
                "sha256": artifact.sha256,
                "requires_python": artifact.requires_python,
            }
            for artifact in sorted(
                artifacts,
                key=lambda item: (
                    item.configuration,
                    item.platform,
                    item.python_abi,
                ),
            )
        ],
    }
    write_json(output / "artifacts" / version / "manifest.json", manifest_value)
    write_json(output / "release-manifest.json", manifest_value)
    catalog = load_matrix_module().selector_catalog(
        matrix, fhelium_version=version
    )
    for recipe in catalog["binary_recipes"]:
        recipe["published"] = True
    (output / "install-catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
