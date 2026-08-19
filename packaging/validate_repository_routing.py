#!/usr/bin/env python3
"""Validate canonical Simple Repository routing and content negotiation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from release_matrix import load_release_matrix


HTML_MEDIA_TYPE = "text/html"
SIMPLE_HTML_MEDIA_TYPE = "application/vnd.pypi.simple.v1+html"
JSON_MEDIA_TYPE = "application/vnd.pypi.simple.v1+json"
USER_AGENT = "fhelium-release-routing-validator/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    state = parser.add_mutually_exclusive_group()
    state.add_argument(
        "--expect-empty",
        action="store_true",
        help="require every canonical HTML and JSON route to return HTTP 404",
    )
    state.add_argument(
        "--expect-empty-or-matching",
        type=Path,
        metavar="TREE",
        help=(
            "accept HTTP 404 routes or routes whose public bytes match the "
            "prepared repository tree"
        ),
    )
    return parser.parse_args()


def get(url: str, accept: str) -> tuple[int, str, bytes]:
    request = Request(
        url,
        headers={"Accept": accept, "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return (
                response.status,
                response.headers.get_content_type(),
                response.read(),
            )
    except HTTPError as error:
        return (
            error.code,
            error.headers.get_content_type(),
            error.read(),
        )


def main() -> None:
    args = parse_args()
    matrix = load_release_matrix()
    prepared = (
        args.expect_empty_or_matching.resolve()
        if args.expect_empty_or_matching is not None
        else None
    )
    for configuration in matrix.configurations:
        root = f"{matrix.simple_index_base_url}/{configuration.id}/simple/"
        project = root + "fhelium/"
        for url, expected_name, relative in (
            (root, None, Path(configuration.id) / "simple"),
            (
                project,
                "fhelium",
                Path(configuration.id) / "simple" / "fhelium",
            ),
        ):
            if args.expect_empty:
                status, _, _ = get(url, HTML_MEDIA_TYPE)
                if status != 404:
                    raise RuntimeError(
                        f"empty repository route returned HTTP {status}: {url}"
                    )
                status, _, _ = get(url, JSON_MEDIA_TYPE)
                if status != 404:
                    raise RuntimeError(
                        "empty repository JSON route returned "
                        f"HTTP {status}: {url}"
                    )
                continue
            if prepared is not None:
                for accept, filename, media_types in (
                    (
                        HTML_MEDIA_TYPE,
                        "index.html",
                        (HTML_MEDIA_TYPE, SIMPLE_HTML_MEDIA_TYPE),
                    ),
                    (JSON_MEDIA_TYPE, "index.json", (JSON_MEDIA_TYPE,)),
                ):
                    status, media_type, payload = get(url, accept)
                    if status == 404:
                        continue
                    expected = prepared / relative / filename
                    if (
                        status != 200
                        or media_type not in media_types
                        or not expected.is_file()
                        or payload != expected.read_bytes()
                    ):
                        raise RuntimeError(
                            "repository initialization route differs from "
                            f"the prepared release: {url} ({accept})"
                        )
                continue
            status, media_type, payload = get(url, HTML_MEDIA_TYPE)
            if status != 200:
                raise RuntimeError(f"GET {url} returned HTTP {status}")
            if (
                media_type not in (HTML_MEDIA_TYPE, SIMPLE_HTML_MEDIA_TYPE)
                or b"<html" not in payload.lower()
            ):
                raise RuntimeError(f"canonical HTML route is invalid: {url}")
            status, media_type, payload = get(url, JSON_MEDIA_TYPE)
            if status != 200 or media_type != JSON_MEDIA_TYPE:
                raise RuntimeError(
                    f"canonical JSON media type is invalid: {url}"
                )
            value = json.loads(payload)
            if value.get("meta") != {"api-version": "1.4"}:
                raise RuntimeError(
                    f"canonical JSON API version is invalid: {url}"
                )
            if expected_name is not None and value.get("name") != expected_name:
                raise RuntimeError(
                    f"canonical JSON project name is invalid: {url}"
                )
        if prepared is not None:
            state = "empty-or-matching repository"
        elif args.expect_empty:
            state = "empty repository"
        else:
            state = "repository"
        print(f"Validated {state} routing: {configuration.id}")


if __name__ == "__main__":
    main()
