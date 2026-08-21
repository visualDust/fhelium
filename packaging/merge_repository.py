#!/usr/bin/env python3
"""Merge published FHElium project indexes into a prepared release tree."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT = "fhelium"
USER_AGENT = "fhelium-release-index-merge/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument(
        "--allow-missing-repository",
        action="store_true",
        help="accept HTTP 404 indexes when creating the repository",
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


def fetch_json(
    url: str, *, allow_missing_repository: bool
) -> dict[str, object] | None:
    error: Exception | None = None
    for attempt in range(6):
        try:
            request = Request(
                url,
                headers={
                    "Accept": "application/vnd.pypi.simple.v1+json",
                    "User-Agent": USER_AGENT,
                },
            )
            with urlopen(request, timeout=30) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise RuntimeError(f"published index is not an object: {url}")
            return value
        except HTTPError as current:
            if current.code == 404:
                if allow_missing_repository:
                    return None
                raise RuntimeError(
                    f"published index does not exist: {url}"
                ) from current
            error = current
        except Exception as current:
            error = current
        if attempt < 5:
            time.sleep(2)
    assert error is not None
    raise RuntimeError(f"cannot retrieve published index: {url}") from error


def validate_file(
    value: object,
    *,
    source: str,
    artifact_base_url: str,
    configuration: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"index file is not an object: {source}")
    filename = value.get("filename")
    url = value.get("url")
    hashes = value.get("hashes")
    requires_python = value.get("requires-python")
    if (
        not isinstance(filename, str)
        or not filename.startswith("fhelium-")
        or not filename.endswith(".whl")
        or not isinstance(url, str)
        or not url.startswith(f"{artifact_base_url}/")
        or f"/{configuration}/" not in urlparse(url).path
        or not isinstance(hashes, dict)
        or set(hashes) != {"sha256"}
        or not isinstance(hashes.get("sha256"), str)
        or not isinstance(requires_python, str)
    ):
        raise RuntimeError(f"invalid index file entry: {source}")
    digest = hashes["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"invalid index file hash: {source}")
    return value


def html_page(files: list[dict[str, object]]) -> str:
    anchors = []
    for value in files:
        hashes = value["hashes"]
        assert isinstance(hashes, dict)
        anchors.append(
            "<a "
            f'href="{html.escape(str(value["url"]))}'
            f'#sha256={html.escape(str(hashes["sha256"]))}" '
            f'data-requires-python="{html.escape(str(value["requires-python"]))}">'
            f'{html.escape(str(value["filename"]))}</a><br>'
        )
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


def main() -> None:
    args = parse_args()
    tree = args.tree.resolve()
    matrix = load_matrix_module().load_matrix()
    for configuration in matrix.configurations:
        project_dir = tree / configuration.id / "simple" / PROJECT
        local_path = project_dir / "index.json"
        local = json.loads(local_path.read_text(encoding="utf-8"))
        if not isinstance(local, dict) or not isinstance(
            local.get("files"), list
        ):
            raise RuntimeError(f"invalid prepared index: {local_path}")
        remote_url = (
            f"{matrix.simple_index_base_url}/{configuration.id}/"
            f"simple/{PROJECT}/index.json"
        )
        remote = fetch_json(
            remote_url,
            allow_missing_repository=args.allow_missing_repository,
        )
        values = []
        if remote is not None:
            if remote.get("meta") != {"api-version": "1.4"}:
                raise RuntimeError(f"unsupported published index: {remote_url}")
            remote_files = remote.get("files")
            if not isinstance(remote_files, list):
                raise RuntimeError(
                    f"published index has no files: {remote_url}"
                )
            values.extend(
                validate_file(
                    value,
                    source=remote_url,
                    artifact_base_url=matrix.artifact_base_url,
                    configuration=configuration.id,
                )
                for value in remote_files
            )
        values.extend(
            validate_file(
                value,
                source=str(local_path),
                artifact_base_url=matrix.artifact_base_url,
                configuration=configuration.id,
            )
            for value in local["files"]
        )

        by_filename: dict[str, dict[str, object]] = {}
        for value in values:
            filename = str(value["filename"])
            existing = by_filename.get(filename)
            if existing is not None and existing != value:
                raise RuntimeError(
                    f"published wheel identity differs for {configuration.id}/{filename}"
                )
            by_filename[filename] = value
        merged = [by_filename[name] for name in sorted(by_filename)]
        local_path.write_text(
            json.dumps(
                {
                    "meta": {"api-version": "1.4"},
                    "name": PROJECT,
                    "files": merged,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (project_dir / "index.html").write_text(
            html_page(merged), encoding="utf-8"
        )
        print(f"{configuration.id}: {len(merged)} published wheel links")


if __name__ == "__main__":
    main()
