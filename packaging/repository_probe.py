#!/usr/bin/env python3
"""Validate declared Simple Repository routing and cumulative content."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from matrix import load_matrix


HTML_MEDIA_TYPE = "text/html"
SIMPLE_HTML_MEDIA_TYPE = "application/vnd.pypi.simple.v1+html"
JSON_MEDIA_TYPE = "application/vnd.pypi.simple.v1+json"
USER_AGENT = "fhelium-release-routing-validator/1"


@dataclass(frozen=True)
class Route:
    url: str
    expected_name: str | None
    relative: Path
    accept: str
    filename: str
    media_types: tuple[str, ...]


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: set[tuple[tuple[tuple[str, str], ...], str]] = set()
        self._attributes: tuple[tuple[str, str], ...] | None = None
        self._text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "a":
            return
        self._attributes = tuple(
            sorted((name.lower(), value or "") for name, value in attrs)
        )
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._attributes is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._attributes is None:
            return
        self.anchors.add((self._attributes, "".join(self._text).strip()))
        self._attributes = None
        self._text = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    state = parser.add_mutually_exclusive_group()
    state.add_argument(
        "--expect-empty",
        action="store_true",
        help="require every declared HTML and JSON route to return HTTP 404",
    )
    state.add_argument(
        "--expect-subset",
        type=Path,
        metavar="TREE",
        help=(
            "require existing public entries to be an unchanged subset of "
            "the prepared cumulative repository"
        ),
    )
    parser.add_argument(
        "--allow-missing-configuration",
        action="append",
        default=[],
        metavar="CONFIGURATION",
        help=(
            "accept all four routes as HTTP 404 for this newly introduced "
            "configuration; repeat for multiple roots"
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


def _json_entries(
    value: dict[str, object], *, source: str
) -> dict[str, object]:
    if value.get("meta") != {"api-version": "1.4"}:
        raise RuntimeError(f"JSON API version is invalid: {source}")
    if "files" in value:
        entries = value["files"]
        identity = "filename"
    elif "projects" in value:
        entries = value["projects"]
        identity = "name"
    else:
        raise RuntimeError(f"JSON index has no files or projects: {source}")
    if not isinstance(entries, list):
        raise RuntimeError(f"JSON index entries are invalid: {source}")
    result: dict[str, object] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(
            entry.get(identity), str
        ):
            raise RuntimeError(f"JSON index entry is invalid: {source}")
        key = str(entry[identity])
        if key in result:
            raise RuntimeError(f"duplicate JSON index entry {key!r}: {source}")
        result[key] = entry
    return result


def validate_json_subset(
    public_payload: bytes,
    prepared_payload: bytes,
    *,
    source: str,
) -> None:
    """Require every public JSON entry to occur unchanged in the candidate."""

    try:
        public_value = json.loads(public_payload)
        prepared_value = json.loads(prepared_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON index: {source}") from error
    if not isinstance(public_value, dict) or not isinstance(
        prepared_value, dict
    ):
        raise RuntimeError(f"JSON index is not an object: {source}")
    if public_value.get("name") != prepared_value.get("name"):
        raise RuntimeError(f"JSON project identity differs: {source}")
    public_entries = _json_entries(public_value, source=source)
    prepared_entries = _json_entries(prepared_value, source=source)
    for identity, entry in public_entries.items():
        if prepared_entries.get(identity) != entry:
            raise RuntimeError(
                f"published JSON entry differs or is missing: {source} "
                f"({identity})"
            )


def validate_html_subset(
    public_payload: bytes,
    prepared_payload: bytes,
    *,
    source: str,
) -> None:
    """Require every published HTML link to remain in the candidate."""

    try:
        public_text = public_payload.decode("utf-8")
        prepared_text = prepared_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"HTML index is not UTF-8: {source}") from error
    public_parser = _AnchorParser()
    prepared_parser = _AnchorParser()
    public_parser.feed(public_text)
    prepared_parser.feed(prepared_text)
    if not public_parser.anchors.issubset(prepared_parser.anchors):
        raise RuntimeError(
            f"published HTML link differs or is missing: {source}"
        )


def routes_are_present(
    statuses: Sequence[int],
    *,
    configuration: str,
    allow_missing: bool,
) -> bool:
    """Validate all-or-none routing and report whether the root exists."""

    if all(status == 404 for status in statuses):
        if not allow_missing:
            raise RuntimeError(
                "repository configuration is missing without authorization: "
                f"{configuration}"
            )
        return False
    if any(status == 404 for status in statuses):
        raise RuntimeError(
            "repository configuration is only partially routed: "
            f"{configuration} ({list(statuses)})"
        )
    return True


def main() -> None:
    args = parse_args()
    matrix = load_matrix()
    configuration_ids = {
        configuration.id for configuration in matrix.configurations
    }
    allowed_missing = set(args.allow_missing_configuration)
    unknown = allowed_missing - configuration_ids
    if unknown:
        raise RuntimeError(
            "unknown missing repository configurations: "
            + ", ".join(sorted(unknown))
        )
    if allowed_missing and args.expect_subset is None:
        raise RuntimeError(
            "--allow-missing-configuration requires --expect-subset"
        )
    prepared = (
        args.expect_subset.resolve() if args.expect_subset is not None else None
    )

    for configuration in matrix.configurations:
        root = f"{matrix.simple_index_base_url}/{configuration.id}/simple/"
        project = root + "fhelium/"
        routes = (
            Route(
                root,
                None,
                Path(configuration.id) / "simple",
                HTML_MEDIA_TYPE,
                "index.html",
                (HTML_MEDIA_TYPE, SIMPLE_HTML_MEDIA_TYPE),
            ),
            Route(
                root,
                None,
                Path(configuration.id) / "simple",
                JSON_MEDIA_TYPE,
                "index.json",
                (JSON_MEDIA_TYPE,),
            ),
            Route(
                project,
                "fhelium",
                Path(configuration.id) / "simple" / "fhelium",
                HTML_MEDIA_TYPE,
                "index.html",
                (HTML_MEDIA_TYPE, SIMPLE_HTML_MEDIA_TYPE),
            ),
            Route(
                project,
                "fhelium",
                Path(configuration.id) / "simple" / "fhelium",
                JSON_MEDIA_TYPE,
                "index.json",
                (JSON_MEDIA_TYPE,),
            ),
        )
        responses = [(route, *get(route.url, route.accept)) for route in routes]
        statuses = [status for _, status, _, _ in responses]

        if args.expect_empty:
            if any(status != 404 for status in statuses):
                raise RuntimeError(
                    "empty repository configuration has a published route: "
                    f"{configuration.id} ({statuses})"
                )
            print(f"Validated empty repository routing: {configuration.id}")
            continue

        if not routes_are_present(
            statuses,
            configuration=configuration.id,
            allow_missing=configuration.id in allowed_missing,
        ):
            print(
                "Validated missing repository initialization: "
                f"{configuration.id}"
            )
            continue

        for route, status, media_type, payload in responses:
            if status != 200 or media_type not in route.media_types:
                raise RuntimeError(
                    f"repository route is invalid: {route.url} "
                    f"({route.accept}, "
                    f"HTTP {status}, {media_type})"
                )
            if prepared is not None:
                expected = prepared / route.relative / route.filename
                if not expected.is_file():
                    raise RuntimeError(
                        f"prepared repository route is missing: {expected}"
                    )
                source = f"{route.url} ({route.accept})"
                if route.filename.endswith(".json"):
                    validate_json_subset(
                        payload, expected.read_bytes(), source=source
                    )
                else:
                    validate_html_subset(
                        payload, expected.read_bytes(), source=source
                    )
                continue
            if route.filename.endswith(".html"):
                if b"<html" not in payload.lower():
                    raise RuntimeError(f"HTML route is invalid: {route.url}")
                continue
            value = json.loads(payload)
            if value.get("meta") != {"api-version": "1.4"}:
                raise RuntimeError(f"JSON API version is invalid: {route.url}")
            if (
                route.expected_name is not None
                and value.get("name") != route.expected_name
            ):
                raise RuntimeError(f"JSON project name is invalid: {route.url}")

        state = "published subset" if prepared is not None else "repository"
        print(f"Validated {state} routing: {configuration.id}")


if __name__ == "__main__":
    main()
