"""Catalog-row and filesystem helpers for the local artifact store."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

from fhelium.artifacts.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_SENSITIVITIES,
    ArtifactMetadata,
    ArtifactRef,
    ArtifactSensitivity,
)
from fhelium.errors import StaleArtifactReferenceError
from fhelium.serialization import VALUE_SCHEMA_VERSION, supported_value_types

CATALOG_NAME = "catalog.sqlite3"
OBJECTS_DIRECTORY_NAME = "objects"
TEMPORARY_DIRECTORY_NAME = "tmp"
STORE_FORMAT = "fhelium-artifact-store"
STORE_SCHEMA_VERSION = 1
SUPPORTED_STORE_SCHEMA_VERSIONS = (STORE_SCHEMA_VERSION,)


def _normalize_name(name: str) -> str:
    """Return one canonical store-relative logical name."""

    if not isinstance(name, str):
        raise TypeError(f"Artifact name must be str, got {type(name).__name__}")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or str(path) != name:
        raise ValueError(
            "Artifact names must be normalized non-empty relative POSIX paths: "
            f"{name!r}"
        )
    if any(
        part in ("", ".", "..") or part.startswith(".") for part in path.parts
    ):
        raise ValueError(
            "Artifact names cannot contain empty, dot, parent, or hidden "
            f"segments: {name!r}"
        )
    return name


def _validate_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Artifact catalog {field} must be a UUID string")
    try:
        canonical = str(UUID(value))
    except ValueError as error:
        raise ValueError(
            f"Artifact catalog {field} must be a UUID string"
        ) from error
    if value != canonical:
        raise ValueError(
            f"Artifact catalog {field} must use canonical UUID text"
        )
    return value


def _json_object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"Artifact catalog {field} must be JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Artifact catalog {field} contains invalid JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError(f"Artifact catalog {field} must encode an object")
    return cast(dict[str, Any], decoded)


def _validate_checksum(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "Artifact catalog payload_sha256 must be a lowercase SHA-256 digest"
        )
    return value


def _validate_created_at(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Artifact catalog created_at must be a string")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "Artifact catalog created_at must be an ISO-8601 timestamp"
        ) from error
    offset = timestamp.utcoffset()
    if (
        timestamp.tzinfo is None
        or offset is None
        or offset.total_seconds() != 0
    ):
        raise ValueError("Artifact catalog created_at must use a UTC offset")
    return value


def _validate_payload_relpath(value: object, *, artifact_id: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Artifact catalog payload_relpath must be a string")
    path = PurePosixPath(value)
    expected = PurePosixPath(
        OBJECTS_DIRECTORY_NAME,
        artifact_id[:2],
        f"{artifact_id}.safetensors",
    )
    if path.is_absolute() or path != expected:
        raise ValueError(
            "Artifact catalog payload path does not match its artifact ID: "
            f"expected={expected.as_posix()!r}, actual={value!r}"
        )
    return value


def _metadata_from_catalog_row(
    row: Any,
    *,
    store_id: str,
) -> tuple[ArtifactMetadata, str]:
    """Validate one SQLite result row and reconstruct its public metadata."""

    store_id = _validate_uuid(store_id, field="store_id")
    name = _normalize_name(row["name"])
    artifact_id = _validate_uuid(row["artifact_id"], field="artifact_id")
    artifact_schema_version = row["artifact_schema_version"]
    if (
        type(artifact_schema_version) is not int
        or artifact_schema_version != ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported artifact metadata schema version: "
            f"{artifact_schema_version!r}"
        )
    value_type = row["value_type"]
    if (
        not isinstance(value_type, str)
        or value_type not in supported_value_types()
    ):
        raise ValueError(f"Unsupported artifact value_type: {value_type!r}")
    value_schema_version = row["value_schema_version"]
    if (
        type(value_schema_version) is not int
        or value_schema_version != VALUE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported nested value schema version: {value_schema_version!r}"
        )
    context_id = row["context_id"]
    if context_id is not None and not isinstance(context_id, str):
        raise ValueError("Artifact catalog context_id must be a string or null")
    nbytes = row["nbytes"]
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError(
            "Artifact catalog nbytes must be a non-negative integer"
        )
    payload_sha256 = _validate_checksum(row["payload_sha256"])
    sensitivity = row["sensitivity"]
    if sensitivity not in SUPPORTED_ARTIFACT_SENSITIVITIES:
        raise ValueError(f"Unsupported artifact sensitivity: {sensitivity!r}")
    if value_type == "SecretKey" and sensitivity != "secret":
        raise ValueError("SecretKey artifacts must use sensitivity='secret'")
    created_at = _validate_created_at(row["created_at"])
    tensor_metadata = _json_object(
        row["tensor_metadata_json"], field="tensor_metadata_json"
    )
    value_metadata = _json_object(
        row["value_metadata_json"], field="value_metadata_json"
    )
    payload_relpath = _validate_payload_relpath(
        row["payload_relpath"], artifact_id=artifact_id
    )
    ref: ArtifactRef[Any] = ArtifactRef(
        store_id=store_id,
        name=name,
        artifact_id=artifact_id,
        value_type=value_type,
        artifact_schema_version=artifact_schema_version,
        context_id=context_id,
        nbytes=nbytes,
        payload_sha256=payload_sha256,
    )
    return (
        ArtifactMetadata(
            ref=ref,
            sensitivity=cast(ArtifactSensitivity, sensitivity),
            created_at=created_at,
            value_schema_version=value_schema_version,
            tensor_metadata=cast(dict[str, dict[str, Any]], tensor_metadata),
            value_metadata=value_metadata,
        ),
        payload_relpath,
    )


def _validate_reference(
    expected: ArtifactRef[Any],
    actual: ArtifactRef[Any] | None,
) -> None:
    """Require a reference to match the store's current name generation."""

    if actual is not None and expected == actual:
        return
    expected_fields = asdict(expected)
    actual_fields = {} if actual is None else asdict(actual)
    differences: dict[str, tuple[object, object]] = {
        field: (expected_fields[field], actual_fields.get(field))
        for field in expected_fields
        if expected_fields[field] != actual_fields.get(field)
    }
    raise StaleArtifactReferenceError(
        name=expected.name,
        expected_artifact_id=expected.artifact_id,
        current_artifact_id=(None if actual is None else actual.artifact_id),
        differences=differences,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
