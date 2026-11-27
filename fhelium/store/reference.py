"""Tensor-free references and metadata for values in a local value store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T", covariant=True)
ValueSensitivity = Literal["public", "confidential", "secret"]
SUPPORTED_SENSITIVITIES: tuple[ValueSensitivity, ...] = (
    "public",
    "confidential",
    "secret",
)


@dataclass(frozen=True)
class StoredValueRef(Generic[T]):
    """A tensor-free reference to one active persisted value.

    The generic parameter ``T`` is for static type checking and is erased at
    runtime. A reference identifies one generation of a logical name in one
    store. Replacing or deleting that name makes the old reference stale; the
    store does not retain historical generations.

    Args:
        store_id: Unique identity of the local store that issued the reference.
        name: Stable store-relative logical name.
        generation_id: Unique identity of the active generation. Overwriting a
            name creates a new identity and invalidates the old reference.
        value_type: Serialized-value class name recorded by the catalog.
        metadata_schema_version: Stored-value metadata schema version.
        nbytes: Logical tensor payload bytes of one materialization.
        payload_sha256: SHA-256 checksum of the serialized payload file. The
            checksum detects corruption but does not authenticate a stored value.
    """

    store_id: str
    name: str
    generation_id: str
    value_type: str
    metadata_schema_version: int
    nbytes: int
    payload_sha256: str


@dataclass(frozen=True)
class StoredValueMetadata:
    """Validated public metadata read from the value-store catalog.

    Args:
        ref: Tensor-free identity and integrity fields for the stored value.
        sensitivity: Caller-declared public, confidential, or secret label.
            This label does not itself provide encryption at rest.
        created_at: UTC creation timestamp recorded by the catalog.
        value_schema_version: Nested value-envelope schema version.
        tensor_metadata: Per-tensor serialized shape, dtype, and related data.
        value_metadata: Non-tensor value state required for reconstruction.
    """

    ref: StoredValueRef[Any]
    sensitivity: ValueSensitivity
    created_at: str
    value_schema_version: int
    tensor_metadata: dict[str, dict[str, Any]]
    value_metadata: dict[str, Any]


METADATA_SCHEMA_VERSION = 2
