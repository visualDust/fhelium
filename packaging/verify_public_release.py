#!/usr/bin/env python3
"""Verify immutable FHElium release objects through their public URLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


PUBLIC_MANIFEST_BASE = "https://download.fhelium.550w.host/releases"
USER_AGENT = "fhelium-public-release-verifier/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    error: Exception | None = None
    for attempt in range(12):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with (
                urlopen(request, timeout=120) as response,
                path.open("wb") as stream,
            ):
                if response.status != 200:
                    raise RuntimeError(
                        f"GET {url} returned HTTP {response.status}"
                    )
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            return
        except Exception as current:
            error = current
            if attempt == 11:
                break
            time.sleep(5)
    assert error is not None
    raise RuntimeError(
        f"cannot retrieve public release object: {url}"
    ) from error


def main() -> None:
    manifest_path = parse_args().manifest.resolve()
    expected_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = expected_manifest.get("project_version")
    if not isinstance(version, str):
        raise RuntimeError("release manifest has no project version")
    with tempfile.TemporaryDirectory(
        prefix="fhelium-public-release-"
    ) as temporary:
        root = Path(temporary)
        public_manifest_path = root / "manifest.json"
        download(
            f"{PUBLIC_MANIFEST_BASE}/{version}/manifest.json",
            public_manifest_path,
        )
        actual_manifest = json.loads(
            public_manifest_path.read_text(encoding="utf-8")
        )
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                "public release manifest differs from release bundle"
            )
        for index, artifact in enumerate(expected_manifest["artifacts"]):
            path = root / f"artifact-{index}.whl"
            download(artifact["url"], path)
            if path.stat().st_size != artifact["size"]:
                raise RuntimeError(
                    f"public artifact size differs: {artifact['url']}"
                )
            if sha256(path) != artifact["sha256"]:
                raise RuntimeError(
                    f"public artifact hash differs: {artifact['url']}"
                )
    print(f"Verified {len(expected_manifest['artifacts'])} public artifacts")


if __name__ == "__main__":
    main()
