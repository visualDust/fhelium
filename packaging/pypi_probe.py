#!/usr/bin/env python3
"""Verify that a PyPI source distribution has the expected byte identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT = "fhelium"
USER_AGENT = "fhelium-pypi-release-verifier/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(version: str) -> dict[str, object] | None:
    request = Request(
        f"https://pypi.org/pypi/{PROJECT}/{version}/json",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=30) as response:
            value = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    if not isinstance(value, dict):
        raise RuntimeError("PyPI release response is not a JSON object")
    return value


def validate(value: dict[str, object], sdist: Path) -> bool:
    urls = value.get("urls")
    if not isinstance(urls, list):
        raise RuntimeError("PyPI release response has no file list")
    matches = [
        item
        for item in urls
        if isinstance(item, dict)
        and item.get("filename") == sdist.name
        and item.get("packagetype") == "sdist"
    ]
    if not matches:
        return False
    if len(matches) != 1:
        raise RuntimeError(
            f"PyPI must contain exactly one source distribution named {sdist.name}"
        )
    item = matches[0]
    digests = item.get("digests")
    if (
        not isinstance(digests, dict)
        or digests.get("sha256") != sha256(sdist)
        or item.get("size") != sdist.stat().st_size
    ):
        raise RuntimeError(
            "PyPI source distribution differs from the release bundle"
        )
    return True


def main() -> None:
    args = parse_args()
    sdist = args.sdist.resolve()
    if not sdist.is_file():
        raise FileNotFoundError(sdist)
    match = re.fullmatch(r"fhelium-(\d+(?:\.\d+)*)\.tar\.gz", sdist.name)
    if match is None:
        raise ValueError(
            f"unexpected source distribution filename: {sdist.name}"
        )
    if args.attempts < 1 or args.interval < 0:
        raise ValueError(
            "--attempts must be positive and --interval nonnegative"
        )

    version = match.group(1)
    for attempt in range(args.attempts):
        value = fetch(version)
        if value is not None:
            if validate(value, sdist):
                print(f"Verified PyPI source distribution: {sdist.name}")
                return
        if attempt + 1 < args.attempts:
            time.sleep(args.interval)
    if args.allow_missing:
        print(f"PyPI source distribution is not published yet: {sdist.name}")
        return
    raise RuntimeError(
        f"PyPI source distribution is not published: {sdist.name}"
    )


if __name__ == "__main__":
    main()
