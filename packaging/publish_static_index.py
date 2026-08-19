#!/usr/bin/env python3
"""Publish FHElium release objects and indexes to Cloudflare R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import subprocess
from pathlib import Path


INDEX_SUFFIXES = (
    "/simple/index.html",
    "/simple/index.json",
    "/simple/fhelium/index.html",
    "/simple/fhelium/index.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument(
        "--phase", required=True, choices=("artifacts", "indexes")
    )
    return parser.parse_args()


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def aws_base() -> tuple[str, ...]:
    required = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "FHELIUM_R2_ENDPOINT_URL",
        "FHELIUM_R2_BUCKET",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "missing R2 environment variables: " + ", ".join(missing)
        )
    return (
        "aws",
        "--endpoint-url",
        os.environ["FHELIUM_R2_ENDPOINT_URL"],
        "s3api",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_metadata(
    base: tuple[str, ...], bucket: str, key: str
) -> dict[str, object] | None:
    result = subprocess.run(
        (*base, "head-object", "--bucket", bucket, "--key", key),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise RuntimeError(
                f"invalid object metadata for s3://{bucket}/{key}"
            )
        return value
    combined = result.stdout + result.stderr
    if "Not Found" in combined or "404" in combined or "NoSuchKey" in combined:
        return None
    raise RuntimeError(
        f"cannot inspect s3://{bucket}/{key}: {combined.strip()}"
    )


def upload(
    base: tuple[str, ...],
    bucket: str,
    path: Path,
    key: str,
    *,
    cache_control: str,
) -> None:
    content_type = (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )
    run(
        *base,
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        str(path),
        "--content-type",
        content_type,
        "--cache-control",
        cache_control,
        "--metadata",
        f"sha256={sha256(path)}",
    )


def upload_immutable(
    base: tuple[str, ...], bucket: str, path: Path, key: str
) -> None:
    digest = sha256(path)
    existing = object_metadata(base, bucket, key)
    if existing is not None:
        metadata = existing.get("Metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("sha256") != digest
            or existing.get("ContentLength") != path.stat().st_size
        ):
            raise RuntimeError(
                f"immutable R2 object differs from release bundle: {key}"
            )
        return
    upload(
        base,
        bucket,
        path,
        key,
        cache_control="public, max-age=31536000, immutable",
    )


def main() -> None:
    args = parse_args()
    tree = args.tree.resolve()
    manifest_path = tree / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("published") is not True:
        raise RuntimeError("release bundle must record published=true")
    if not (tree / "install-catalog.json").is_file():
        raise RuntimeError("release bundle has no published install catalog")
    version = manifest.get("project_version")
    if not isinstance(version, str):
        raise RuntimeError("release manifest has no project version")

    base = aws_base()
    bucket = os.environ["FHELIUM_R2_BUCKET"]
    files = sorted(path for path in tree.rglob("*") if path.is_file())

    if args.phase == "artifacts":
        immutable = [
            path
            for path in files
            if path.relative_to(tree).as_posix().startswith("artifacts/")
        ]
        for path in immutable:
            upload_immutable(
                base, bucket, path, path.relative_to(tree).as_posix()
            )
        upload_immutable(
            base,
            bucket,
            manifest_path,
            f"releases/{version}/manifest.json",
        )
        print(f"Published {len(immutable) + 1} immutable release objects")
        return

    indexes = [
        path
        for path in files
        if path.relative_to(tree).as_posix().endswith(INDEX_SUFFIXES)
    ]
    if len(indexes) != 16:
        raise RuntimeError(
            f"expected 16 static index files, found {len(indexes)}"
        )
    for path in indexes:
        upload(
            base,
            bucket,
            path,
            path.relative_to(tree).as_posix(),
            cache_control="public, max-age=60, must-revalidate",
        )
    print(f"Published {len(indexes)} index pages")


if __name__ == "__main__":
    main()
