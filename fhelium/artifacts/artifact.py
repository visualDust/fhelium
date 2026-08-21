"""Tensor-free references and metadata for the local artifact repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T", covariant=True)
ArtifactSensitivity = Literal["public", "confidential", "secret"]
SUPPORTED_ARTIFACT_SENSITIVITIES: tuple[ArtifactSensitivity, ...] = (
    "public",
    "confidential",
    "secret",
)


@dataclass(frozen=True)
class ArtifactRef(Generic[T]):
    """A tensor-free reference to one active persisted value.

    The generic parameter ``T`` is for static type checking and is erased at
    runtime. A reference identifies one generation of a logical name in one
    store. Replacing or deleting that name makes the old reference stale; the
    store does not retain historical generations.

    Args:
        store_id: Unique identity of the local store that issued the reference.
        name: Stable store-relative logical name.
        artifact_id: Unique identity of the active generation. Overwriting a
            name creates a new identity and invalidates the old reference.
        value_type: Serialized-value class name recorded by the catalog.
        artifact_schema_version: Artifact metadata schema version.
        context_id: Cryptographic context identity, or ``None`` for values that
            do not belong to a CKKS context.
        nbytes: Logical tensor payload bytes of one materialization.
        payload_sha256: SHA-256 checksum of the serialized payload file. The
            checksum detects corruption but does not authenticate an artifact.
    """

    store_id: str
    name: str
    artifact_id: str
    value_type: str
    artifact_schema_version: int
    context_id: str | None
    nbytes: int
    payload_sha256: str


@dataclass(frozen=True)
class ArtifactMetadata:
    """Validated public metadata read from the artifact catalog.

    Args:
        ref: Tensor-free identity and integrity fields for the artifact.
        sensitivity: Caller-declared public, confidential, or secret label.
            This label does not itself provide encryption at rest.
        created_at: UTC creation timestamp recorded by the catalog.
        value_schema_version: Nested value-envelope schema version.
        tensor_metadata: Per-tensor serialized shape, dtype, and related data.
        value_metadata: Non-tensor value state required for reconstruction.
    """

    ref: ArtifactRef[Any]
    sensitivity: ArtifactSensitivity
    created_at: str
    value_schema_version: int
    tensor_metadata: dict[str, dict[str, Any]]
    value_metadata: dict[str, Any]


ARTIFACT_SCHEMA_VERSION = 1
