from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


PACKAGING = Path(__file__).parent
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

merge_repository = importlib.import_module("merge_repository")
publish_release = importlib.import_module("publish_release")
repository_probe = importlib.import_module("repository_probe")
matrix_module = importlib.import_module("matrix")


def _entry(filename: str, digest: str) -> dict[str, object]:
    return {
        "filename": filename,
        "url": f"https://download.fhelium.550w.host/artifacts/test/{filename}",
        "hashes": {"sha256": digest},
        "requires-python": ">=3.12,<3.14",
    }


def _project_index(*entries: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "meta": {"api-version": "1.4"},
            "name": "fhelium",
            "files": list(entries),
        }
    ).encode()


def _html_index(*links: str) -> bytes:
    return (
        "<!doctype html><html><body>"
        + "".join(f'<a href="{link}">{link}</a>' for link in links)
        + "</body></html>"
    ).encode()


def test_existing_subset_and_missing_new_root_merge_cumulatively() -> None:
    published = _entry("fhelium-0.10.0-old.whl", "a" * 64)
    current = _entry("fhelium-0.20.0-new.whl", "b" * 64)

    merged = merge_repository.merge_file_entries(
        [published], [current], configuration="test"
    )
    repository_probe.validate_json_subset(
        _project_index(published),
        _project_index(*merged),
        source="test",
    )
    assert merge_repository.merge_file_entries(
        [], [current], configuration="new-root"
    ) == [current]
    assert not repository_probe.routes_are_present(
        (404, 404, 404, 404),
        configuration="new-root",
        allow_missing=True,
    )


def test_conflicting_published_wheel_identity_is_rejected() -> None:
    published = _entry("fhelium-0.10.0-old.whl", "a" * 64)
    changed = _entry("fhelium-0.10.0-old.whl", "b" * 64)

    with pytest.raises(RuntimeError, match="published wheel identity differs"):
        merge_repository.merge_file_entries(
            [published], [changed], configuration="test"
        )

    with pytest.raises(RuntimeError, match="differs or is missing"):
        repository_probe.validate_json_subset(
            _project_index(published),
            _project_index(changed),
            source="test",
        )

    with pytest.raises(RuntimeError, match="HTML link differs or is missing"):
        repository_probe.validate_html_subset(
            _html_index("fhelium-0.10.0.whl"),
            _html_index("fhelium-0.20.0.whl"),
            source="test",
        )


def test_expected_index_paths_are_derived_from_release_matrix() -> None:
    matrix = matrix_module.load_matrix()
    expected = publish_release.expected_index_paths()

    assert len(expected) == len(matrix.configurations) * len(
        publish_release.INDEX_SUFFIXES
    )
    assert {path.parts[0] for path in expected} == {
        configuration.id for configuration in matrix.configurations
    }
