#!/usr/bin/env python3
"""Validate a locally prepared FHElium static Simple Repository tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from matrix import load_matrix


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "a":
            self.links.append({key: value or "" for key, value in attrs})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def main() -> None:
    args = parse_args()
    tree = args.tree.resolve()
    matrix = load_matrix()
    manifests = list(tree.glob("artifacts/*/manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one release manifest, found {manifests!r}"
        )
    manifest = read_json(manifests[0])
    root_manifest_path = tree / "release-manifest.json"
    if not root_manifest_path.is_file():
        raise RuntimeError("release-manifest.json is missing")
    if root_manifest_path.read_bytes() != manifests[0].read_bytes():
        raise RuntimeError("root and version release manifests differ")
    if manifest.get("published") is not True:
        raise RuntimeError("release manifest must record published=true")
    source_commit = manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise RuntimeError("release manifest source commit is invalid")
    if args.source_commit is not None and source_commit != args.source_commit:
        raise RuntimeError(
            "release manifest source commit differs from the release ref"
        )
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise RuntimeError("release manifest contains no artifacts")
    catalog = read_json(tree / "install-catalog.json")
    if catalog.get("fhelium_version") != manifest.get("project_version"):
        raise RuntimeError("install catalog version differs from manifest")
    recipes = catalog.get("binary_recipes")
    expected_recipes = sum(
        len(configuration.platform_targets)
        for configuration in matrix.configurations
    )
    if not isinstance(recipes, list) or len(recipes) != expected_recipes:
        raise RuntimeError(
            f"install catalog must contain {expected_recipes} binary recipes"
        )
    if any(
        not isinstance(recipe, dict) or recipe.get("published") is not True
        for recipe in recipes
    ):
        raise RuntimeError("install catalog binary recipes must be published")

    expected: dict[str, dict[str, object]] = {}
    for value in raw_artifacts:
        if not isinstance(value, dict):
            raise RuntimeError("release manifest artifact is not an object")
        relative = value.get("relative_path")
        digest = value.get("sha256")
        size = value.get("size")
        url = value.get("url")
        if not all(
            isinstance(item, expected_type)
            for item, expected_type in (
                (relative, str),
                (digest, str),
                (size, int),
                (url, str),
            )
        ):
            raise RuntimeError("release manifest artifact fields are invalid")
        assert isinstance(relative, str)
        path = tree / relative
        if not path.is_file():
            raise RuntimeError(f"manifest artifact is missing: {path}")
        if path.stat().st_size != size or sha256(path) != digest:
            raise RuntimeError(f"manifest artifact identity differs: {path}")
        expected[str(url)] = value

    html_pages = list(tree.glob("*/simple/fhelium/index.html"))
    json_pages = list(tree.glob("*/simple/fhelium/index.json"))
    root_count = len(matrix.configurations)
    if len(html_pages) != root_count or len(json_pages) != root_count:
        raise RuntimeError(
            f"static tree must contain {root_count} HTML and JSON project pages"
        )

    seen_html: set[str] = set()
    for page in html_pages:
        parser = LinkParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for link in parser.links:
            href = link.get("href", "")
            parsed = urlparse(href)
            base_url = parsed._replace(fragment="").geturl()
            if base_url not in expected:
                continue
            fragment = parsed.fragment
            digest = expected[base_url]["sha256"]
            if fragment != f"sha256={digest}":
                raise RuntimeError(
                    f"HTML index has wrong hash fragment: {href}"
                )
            if link.get("data-requires-python") != expected[base_url].get(
                "requires_python"
            ):
                raise RuntimeError(
                    f"HTML index has wrong Requires-Python: {page}"
                )
            seen_html.add(base_url)

    seen_json: set[str] = set()
    for page in json_pages:
        value = read_json(page)
        if value.get("meta") != {"api-version": "1.4"}:
            raise RuntimeError(f"JSON index API version is invalid: {page}")
        files = value.get("files")
        if not isinstance(files, list):
            raise RuntimeError(f"JSON index has no file list: {page}")
        for item in files:
            if not isinstance(item, dict) or not isinstance(
                item.get("url"), str
            ):
                raise RuntimeError(f"JSON index file is invalid: {page}")
            url = item["url"]
            if url not in expected:
                continue
            if item.get("hashes") != {"sha256": expected[url]["sha256"]}:
                raise RuntimeError(f"JSON index has wrong hash: {url}")
            if item.get("requires-python") != expected[url].get(
                "requires_python"
            ):
                raise RuntimeError(
                    f"JSON index has wrong Requires-Python: {url}"
                )
            seen_json.add(url)

    if not set(expected).issubset(seen_html) or not set(expected).issubset(
        seen_json
    ):
        raise RuntimeError(
            "HTML/JSON indexes do not expose every manifest artifact"
        )
    print(f"Validated static Simple Repository: {tree}")


if __name__ == "__main__":
    main()
