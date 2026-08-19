"""Transactional local artifact catalog built on exact value serialization."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar, overload
from uuid import uuid4
from warnings import warn

import torch

from fhelium.artifacts._catalog import (
    CATALOG_NAME,
    OBJECTS_DIRECTORY_NAME,
    STORE_FORMAT,
    STORE_SCHEMA_VERSION,
    SUPPORTED_STORE_SCHEMA_VERSIONS,
    TEMPORARY_DIRECTORY_NAME,
    _fsync_directory,
    _fsync_file,
    _metadata_from_catalog_row,
    _normalize_name,
    _sha256_file,
    _validate_reference,
    _validate_uuid,
)
from fhelium.artifacts.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_SENSITIVITIES,
    ArtifactMetadata,
    ArtifactRef,
    ArtifactSensitivity,
)
from fhelium.core import SecretKey, TensorResident
from fhelium.errors import ArtifactError, UnsupportedArtifactStoreVersionError
from fhelium.serialization import (
    ValueFileMetadata,
    inspect_value,
    load_value,
    save_value,
)

T = TypeVar("T", bound=TensorResident)
U = TypeVar("U", bound=TensorResident)
_BUSY_TIMEOUT_MILLISECONDS = 30_000
_BOOTSTRAP_LOCK_NAME = ".store.lock"
_MINIMUM_SQLITE_VERSION = (3, 37, 0)

_ARTIFACT_COLUMNS = """
    name,
    artifact_id,
    artifact_schema_version,
    value_type,
    value_schema_version,
    context_id,
    nbytes,
    payload_sha256,
    payload_relpath,
    sensitivity,
    created_at,
    tensor_metadata_json,
    value_metadata_json
"""

_CREATE_SCHEMA = """
CREATE TABLE store_metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE artifacts (
    name TEXT PRIMARY KEY NOT NULL,
    artifact_id TEXT NOT NULL UNIQUE,
    artifact_schema_version INTEGER NOT NULL,
    value_type TEXT NOT NULL,
    value_schema_version INTEGER NOT NULL,
    context_id TEXT,
    nbytes INTEGER NOT NULL CHECK (nbytes >= 0),
    payload_sha256 TEXT NOT NULL,
    payload_relpath TEXT NOT NULL UNIQUE,
    sensitivity TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tensor_metadata_json TEXT NOT NULL,
    value_metadata_json TEXT NOT NULL
) STRICT;
"""


def _canonical_schema_sql(statement: str) -> str:
    return " ".join(statement.rstrip(";").split()).casefold()


_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in _CREATE_SCHEMA.split(";")
    if statement.strip()
)
_EXPECTED_TABLE_SQL = {
    statement.split()[2]: _canonical_schema_sql(statement)
    for statement in _SCHEMA_STATEMENTS
}

_EXPECTED_TABLE_COLUMNS = {
    "store_metadata": ("key", "value"),
    "artifacts": (
        "name",
        "artifact_id",
        "artifact_schema_version",
        "value_type",
        "value_schema_version",
        "context_id",
        "nbytes",
        "payload_sha256",
        "payload_relpath",
        "sensitivity",
        "created_at",
        "tensor_metadata_json",
        "value_metadata_json",
    ),
}


class ArtifactStore:
    """Store one active exact-value generation per local logical name.

    SQLite owns the namespace, metadata transaction, stale-generation checks,
    and process concurrency. Immutable safetensors files under ``objects/`` own
    the large tensor payloads. A writer makes a new payload durable before its
    catalog row commits; a crash can therefore leave only unreachable temporary
    or orphan files, never a committed row pointing to a partially written
    payload. Store opening removes such unreachable files under an exclusive
    catalog transaction.

    Readers retain one rollback-journal read transaction through payload
    validation and reconstruction. Writer commit consequently waits for active
    readers before an overwritten or deleted payload is removed. Writers are
    serialized by SQLite. ``overwrite=True`` publishes the last successfully
    committed generation and makes every older :class:`ArtifactRef` stale.

    Version 1 requires SQLite 3.37 or later and supports one trusted host on a
    local POSIX filesystem with ordinary SQLite locking, same-filesystem
    publication, and file/directory ``fsync``. NFS, SMB, FUSE/object-store
    mounts, multi-host access, hostile writers that bypass this API, encryption
    at rest, and authenticated integrity are not supported. The SHA-256 digest
    detects accidental payload corruption but an attacker able to modify both
    catalog and payload can replace both.

    Args:
        root: Local directory that owns the catalog and immutable payloads. A
            missing or empty directory is initialized. A non-empty directory
            without a recognized catalog is rejected rather than migrated.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        if sqlite3.sqlite_version_info < _MINIMUM_SQLITE_VERSION:
            required = ".".join(map(str, _MINIMUM_SQLITE_VERSION))
            raise RuntimeError(
                "ArtifactStore requires SQLite "
                f">={required} for strict catalog schemas; found "
                f"{sqlite3.sqlite_version}"
            )
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise ValueError("Artifact store root cannot be a symlink")
        self.root = requested.resolve()
        root_existed = self.root.exists()
        if self.root.exists() and not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not root_existed:
            os.chmod(self.root, 0o700)
        self._catalog_path = self.root / CATALOG_NAME
        self._objects_path = self.root / OBJECTS_DIRECTORY_NAME
        self._temporary_path = self.root / TEMPORARY_DIRECTORY_NAME
        self._bootstrap_lock_path = self.root / _BOOTSTRAP_LOCK_NAME
        self.store_id = ""

        with self._bootstrap_lock():
            if self._catalog_path.is_symlink():
                raise ValueError("Artifact catalog cannot be a symlink")
            if not self._catalog_path.exists():
                unexpected = self._unexpected_root_entries()
                if unexpected:
                    raise UnsupportedArtifactStoreVersionError(
                        found_version=0,
                        supported_versions=SUPPORTED_STORE_SCHEMA_VERSIONS,
                        migration_available=False,
                    )
                self._initialize_catalog()
            else:
                self._require_private_regular_catalog()
                if self._catalog_is_uninitialized():
                    # A crash may leave an empty SQLite file before the first
                    # schema transaction commits. This is initialization
                    # recovery, not a migration from another store format.
                    unexpected = self._unexpected_root_entries()
                    if unexpected:
                        raise UnsupportedArtifactStoreVersionError(
                            found_version=0,
                            supported_versions=SUPPORTED_STORE_SCHEMA_VERSIONS,
                            migration_available=False,
                        )
                    self._discard_uninitialized_catalog()
                    self._initialize_catalog()
            self._require_private_regular_catalog()
            self._validate_catalog_and_recover()

    def _unexpected_root_entries(self) -> tuple[str, ...]:
        allowed = {_BOOTSTRAP_LOCK_NAME, CATALOG_NAME}
        return tuple(
            sorted(
                entry.name
                for entry in self.root.iterdir()
                if entry.name not in allowed
            )
        )

    def _require_private_regular_catalog(self) -> None:
        status = self._catalog_path.lstat()
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError("Artifact catalog must be a private regular file")

    @contextmanager
    def _bootstrap_lock(self) -> Iterator[None]:
        if self._bootstrap_lock_path.is_symlink():
            raise ValueError("Artifact store lock cannot be a symlink")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(
            self._bootstrap_lock_path,
            flags,
            0o600,
        )
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise ValueError(
                    "Artifact store lock must be a private regular file"
                )
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _configure_connection(connection: sqlite3.Connection) -> None:
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MILLISECONDS}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA trusted_schema=OFF")

    def _raw_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._catalog_path,
            timeout=_BUSY_TIMEOUT_MILLISECONDS / 1000,
            isolation_level=None,
        )
        self._configure_connection(connection)
        return connection

    def _connection(self) -> sqlite3.Connection:
        connection = self._raw_connection()
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "delete":
            connection.close()
            raise ValueError(
                "Artifact catalog must use SQLite rollback-journal DELETE "
                f"mode, got {mode!r}"
            )
        return connection

    def _initialize_catalog(self) -> None:
        connection = self._raw_connection()
        try:
            mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[
                0
            ]
            if str(mode).lower() != "delete":
                raise RuntimeError(
                    "Could not initialize artifact catalog in SQLite DELETE "
                    f"journal mode: {mode!r}"
                )
            connection.execute("BEGIN EXCLUSIVE")
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            store_id = str(uuid4())
            connection.executemany(
                "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
                (("format", STORE_FORMAT), ("store_id", store_id)),
            )
            connection.execute(f"PRAGMA user_version={STORE_SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        os.chmod(self._catalog_path, 0o600)
        _fsync_file(self._catalog_path)
        _fsync_directory(self.root)

    def _catalog_is_uninitialized(self) -> bool:
        connection = self._raw_connection()
        try:
            version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            tables = tuple(
                connection.execute(
                    "SELECT name FROM sqlite_schema "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            )
            return version == 0 and not tables
        finally:
            connection.close()

    def _discard_uninitialized_catalog(self) -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{self._catalog_path}{suffix}").unlink(missing_ok=True)
        _fsync_directory(self.root)

    def _validate_catalog_and_recover(self) -> None:
        connection = self._raw_connection()
        try:
            version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if version not in SUPPORTED_STORE_SCHEMA_VERSIONS:
                raise UnsupportedArtifactStoreVersionError(
                    found_version=version,
                    supported_versions=SUPPORTED_STORE_SCHEMA_VERSIONS,
                    migration_available=False,
                )
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower() != "delete":
                raise ValueError(
                    "Artifact catalog must use SQLite rollback-journal DELETE "
                    f"mode, got {mode!r}"
                )
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            if quick_check != "ok":
                raise ValueError(
                    f"Artifact SQLite catalog integrity check failed: {quick_check}"
                )
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables != set(_EXPECTED_TABLE_COLUMNS):
                raise ValueError(
                    "Artifact catalog tables do not match format v1: "
                    f"{sorted(tables)}"
                )
            unexpected_schema_objects = tuple(
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    "SELECT type, name FROM sqlite_schema "
                    "WHERE type != 'table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY type, name"
                )
            )
            if unexpected_schema_objects:
                raise ValueError(
                    "Artifact catalog contains unexpected schema objects: "
                    f"{unexpected_schema_objects!r}"
                )
            actual_table_sql = {
                str(row[0]): _canonical_schema_sql(str(row[1]))
                for row in connection.execute(
                    "SELECT name, sql FROM sqlite_schema "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if actual_table_sql != _EXPECTED_TABLE_SQL:
                raise ValueError(
                    "Artifact catalog table definitions do not match format v1"
                )
            for table, expected_columns in _EXPECTED_TABLE_COLUMNS.items():
                actual_columns = tuple(
                    row[1]
                    for row in connection.execute(f"PRAGMA table_info({table})")
                )
                if actual_columns != expected_columns:
                    raise ValueError(
                        f"Artifact catalog table {table!r} has unexpected "
                        f"columns: {actual_columns!r}"
                    )
            metadata = dict(
                connection.execute(
                    "SELECT key, value FROM store_metadata"
                ).fetchall()
            )
            if set(metadata) != {"format", "store_id"}:
                raise ValueError(
                    "Artifact catalog metadata keys are invalid: "
                    f"{sorted(metadata)}"
                )
            if metadata["format"] != STORE_FORMAT:
                raise ValueError(
                    "Artifact catalog format identity mismatch: "
                    f"{metadata['format']!r}"
                )
            self.store_id = _validate_uuid(
                metadata["store_id"], field="store_id"
            )
            self._ensure_storage_directories()

            connection.execute("BEGIN EXCLUSIVE")
            rows = connection.execute(
                f"SELECT {_ARTIFACT_COLUMNS} FROM artifacts"
            ).fetchall()
            referenced: set[str] = set()
            for row in rows:
                _, payload_relpath = _metadata_from_catalog_row(
                    row, store_id=self.store_id
                )
                payload_path = self._payload_path(payload_relpath)
                if payload_path.is_symlink() or not payload_path.is_file():
                    raise FileNotFoundError(
                        "Artifact catalog points to a missing or non-regular "
                        f"payload: {payload_path}"
                    )
                referenced.add(payload_relpath)
            self._remove_unreachable_files(referenced)
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_storage_directories(self) -> None:
        for directory in (self._objects_path, self._temporary_path):
            if directory.is_symlink():
                raise ValueError(
                    f"Artifact store directory cannot be a symlink: {directory}"
                )
            directory.mkdir(mode=0o700, exist_ok=True)
            if not directory.is_dir():
                raise NotADirectoryError(directory)
            os.chmod(directory, 0o700)
        _fsync_directory(self.root)

    def _remove_unreachable_files(self, referenced: set[str]) -> None:
        modified_object_directories: set[Path] = set()
        for entry in tuple(self._temporary_path.iterdir()):
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        for entry in tuple(self._objects_path.rglob("*")):
            if entry.is_dir() and not entry.is_symlink():
                continue
            relative = entry.relative_to(self.root).as_posix()
            if relative not in referenced:
                entry.unlink(missing_ok=True)
                modified_object_directories.add(entry.parent)
        for directory in sorted(
            (
                path
                for path in self._objects_path.rglob("*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
            else:
                modified_object_directories.discard(directory)
                modified_object_directories.add(directory.parent)
        for directory in sorted(
            (
                directory
                for directory in modified_object_directories
                if directory.is_dir()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(self._temporary_path)
        _fsync_directory(self._objects_path)

    def collection(self, name: str) -> ArtifactCollection:
        """Return a logical namespace view rooted at ``name``."""

        return ArtifactCollection(self, _normalize_name(name))

    def put(
        self,
        name: str,
        value: T,
        *,
        sensitivity: ArtifactSensitivity | None = None,
        allow_secret: bool = False,
        overwrite: bool = False,
    ) -> ArtifactRef[T]:
        """Persist and atomically publish a new active generation.

        A writer transaction spans staging and catalog publication. Existing
        readers may continue loading the previous immutable payload; commit
        waits for them before that payload becomes eligible for removal.
        Publication snapshots the supported exact value state but does not
        move, mutate, offload, or release the caller's live ``value``.

        Args:
            name: Normalized store-relative logical name.
            value: Exact tensor-resident FHElium value.
            sensitivity: Descriptive public/confidential/secret label. It does
                not provide encryption or access control.
            allow_secret: Explicitly permit unencrypted SecretKey persistence.
            overwrite: Replace the name's active generation. The old reference
                becomes stale and no history is retained.

        Returns:
            A tensor-free :class:`ArtifactRef` identifying the newly published
            generation. The reference is not a materialized copy of ``value``;
            pass it to :meth:`get` to reconstruct that exact generation.
        """

        name = _normalize_name(name)
        sensitivity = sensitivity or (
            "secret" if isinstance(value, SecretKey) else "public"
        )
        if sensitivity not in SUPPORTED_ARTIFACT_SENSITIVITIES:
            raise ValueError(
                f"Unsupported artifact sensitivity: {sensitivity!r}"
            )
        if isinstance(value, SecretKey) and not allow_secret:
            raise PermissionError(
                "SecretKey persistence is disabled by default; pass "
                "allow_secret=True and provide an appropriate at-rest "
                "security policy."
            )
        if isinstance(value, SecretKey) and sensitivity != "secret":
            raise ValueError(
                "SecretKey artifacts must use sensitivity='secret'"
            )

        artifact_id = str(uuid4())
        payload_relpath = PurePosixPath(
            OBJECTS_DIRECTORY_NAME,
            artifact_id[:2],
            f"{artifact_id}.safetensors",
        ).as_posix()
        payload_path = self._payload_path(payload_relpath)
        temporary_path = self._temporary_path / f"{artifact_id}.safetensors"
        old_payload_relpath: str | None = None
        metadata: ArtifactMetadata | None = None
        published = False
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._fetch_current(connection, name)
            if existing is not None and not overwrite:
                raise FileExistsError(f"Artifact {name!r} already exists")
            if existing is not None:
                _, old_payload_relpath = existing
            conflicting_id = connection.execute(
                "SELECT name FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if conflicting_id is not None or payload_path.exists():
                raise ArtifactError(
                    "Generated artifact ID collides with existing store state: "
                    f"artifact_id={artifact_id!r}"
                )

            value_file = save_value(
                value,
                temporary_path,
                allow_secret=allow_secret,
            )
            os.chmod(temporary_path, 0o600)
            _fsync_file(temporary_path)
            payload_sha256 = _sha256_file(temporary_path)
            created_at = datetime.now(UTC).isoformat()
            inspected = inspect_value(temporary_path)
            metadata = self._metadata_from_value_file(
                name=name,
                artifact_id=artifact_id,
                payload_sha256=payload_sha256,
                sensitivity=sensitivity,
                created_at=created_at,
                value_file=value_file,
            )
            self._validate_value_file_metadata(
                metadata,
                inspected,
                artifact_name=name,
            )

            payload_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(payload_path.parent, 0o700)
            os.link(temporary_path, payload_path, follow_symlinks=False)
            temporary_path.unlink()
            _fsync_directory(self._temporary_path)
            _fsync_directory(payload_path.parent)
            _fsync_directory(self._objects_path)

            self._upsert_current(
                connection,
                metadata,
                payload_relpath=payload_relpath,
            )
            connection.commit()
            published = True
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
            temporary_path.unlink(missing_ok=True)
            if temporary_path.parent.exists():
                _fsync_directory(temporary_path.parent)

        assert metadata is not None and published
        if (
            old_payload_relpath is not None
            and old_payload_relpath != payload_relpath
        ):
            self._remove_committed_orphan(old_payload_relpath)
        return metadata.ref

    @overload
    def get(
        self,
        ref_or_name: ArtifactRef[Any],
        *,
        device: torch.device | str = "cpu",
        expected_type: type[U],
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> U: ...

    @overload
    def get(
        self,
        ref_or_name: ArtifactRef[T],
        *,
        device: torch.device | str = "cpu",
        expected_type: None = None,
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> T: ...

    @overload
    def get(
        self,
        ref_or_name: str,
        *,
        device: torch.device | str = "cpu",
        expected_type: type[U],
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> U | None: ...

    @overload
    def get(
        self,
        ref_or_name: str,
        *,
        device: torch.device | str = "cpu",
        expected_type: None = None,
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> TensorResident | None: ...

    def get(
        self,
        ref_or_name: ArtifactRef[Any] | str,
        *,
        device: torch.device | str = "cpu",
        expected_type: type[TensorResident] | None = None,
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> TensorResident | None:
        """Get a repository value while holding a catalog read snapshot.

        A logical name that has no current generation returns ``None``. An
        :class:`ArtifactRef` is generation-specific, so a missing, replaced,
        deleted, or cross-store reference raises
        :class:`~fhelium.errors.StaleArtifactReferenceError` instead. Catalog,
        checksum, type, context, and payload failures are never converted to
        ``None``.

        This is a repository lookup, not a file-codec operation.
        :func:`fhelium.load_value` reads one caller-selected value-file path;
        ``get`` resolves a catalog name or checked generation, verifies store
        policy, and then reconstructs the exact value.

        Args:
            ref_or_name: Logical name for the optional current generation, or
                a generation-specific checked reference.
            device: Device on which to reconstruct the exact value. Defaults
                to CPU and is not inherited from the saved value.
            expected_type: Optional concrete value type required both in the
                file metadata and after reconstruction.
            expected_context_id: Optional context identity required before
                payload materialization.
            verify_checksum: Whether to verify the repository payload digest
                before reconstruction.

        Returns:
            The reconstructed exact value. Returns ``None`` only when a string
            logical name has no current generation.
        """

        requested_ref, name = self._request(ref_or_name)
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            current = self._fetch_current(connection, name)
            if current is None:
                if requested_ref is not None:
                    _validate_reference(requested_ref, None)
                connection.commit()
                return None
            metadata, payload_relpath = current
            if requested_ref is not None:
                _validate_reference(requested_ref, metadata.ref)
            if (
                expected_context_id is not None
                and metadata.ref.context_id != expected_context_id
            ):
                raise ValueError(
                    "Artifact context mismatch: expected "
                    f"{expected_context_id!r}, got "
                    f"{metadata.ref.context_id!r}"
                )
            payload_path = self._require_payload(
                payload_relpath, artifact_name=name
            )
            if verify_checksum:
                checksum = _sha256_file(payload_path)
                if checksum != metadata.ref.payload_sha256:
                    raise ValueError(
                        f"Artifact {name!r} checksum mismatch: expected "
                        f"{metadata.ref.payload_sha256}, got {checksum}"
                    )
            value_file = inspect_value(payload_path)
            self._validate_value_file_metadata(
                metadata, value_file, artifact_name=name
            )
            value = load_value(
                payload_path,
                device=device,
                expected_type=expected_type,
                expected_context_id=metadata.ref.context_id,
            )
            if type(value).__name__ != metadata.ref.value_type:
                raise TypeError(
                    f"Artifact {name!r} has type {type(value).__name__}, "
                    f"expected catalog type {metadata.ref.value_type}"
                )
            connection.commit()
            return value
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def inspect(self, ref_or_name: ArtifactRef[Any] | str) -> ArtifactMetadata:
        """Validate current catalog metadata and payload headers."""

        requested_ref, name = self._request(ref_or_name)
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            current = self._fetch_current(connection, name)
            if current is None:
                if requested_ref is not None:
                    _validate_reference(requested_ref, None)
                raise FileNotFoundError(f"Artifact {name!r} does not exist")
            metadata, payload_relpath = current
            if requested_ref is not None:
                _validate_reference(requested_ref, metadata.ref)
            payload_path = self._require_payload(
                payload_relpath, artifact_name=name
            )
            self._validate_value_file_metadata(
                metadata,
                inspect_value(payload_path),
                artifact_name=name,
            )
            connection.commit()
            return metadata
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def exists(self, name: str) -> bool:
        """Return whether ``name`` has a catalog row and present payload file.

        This is a lightweight availability probe. It does not checksum or
        inspect the payload; use :meth:`inspect` for structural validation.
        """

        name = _normalize_name(name)
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            current = self._fetch_current(connection, name)
            exists = False
            if current is not None:
                _, payload_relpath = current
                payload = self._payload_path(payload_relpath)
                exists = payload.is_file() and not payload.is_symlink()
            connection.commit()
            return exists
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def list(self, *, prefix: str | None = None) -> list[ArtifactRef[Any]]:
        """Return structurally validated current references sorted by name."""

        normalized_prefix = None if prefix is None else _normalize_name(prefix)
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                f"SELECT {_ARTIFACT_COLUMNS} FROM artifacts ORDER BY name"
            ).fetchall()
            references = []
            for row in rows:
                metadata, payload_relpath = _metadata_from_catalog_row(
                    row, store_id=self.store_id
                )
                if normalized_prefix is not None and not (
                    metadata.ref.name == normalized_prefix
                    or metadata.ref.name.startswith(normalized_prefix + "/")
                ):
                    continue
                payload = self._require_payload(
                    payload_relpath, artifact_name=metadata.ref.name
                )
                self._validate_value_file_metadata(
                    metadata,
                    inspect_value(payload),
                    artifact_name=metadata.ref.name,
                )
                references.append(metadata.ref)
            connection.commit()
            return references
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def delete(self, ref_or_name: ArtifactRef[Any] | str) -> None:
        """Delete the one active generation, optionally compare-and-delete."""

        requested_ref, name = self._request(ref_or_name)
        payload_relpath: str | None = None
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = self._fetch_current(connection, name)
            if current is None:
                if requested_ref is not None:
                    _validate_reference(requested_ref, None)
                raise FileNotFoundError(f"Artifact {name!r} does not exist")
            metadata, payload_relpath = current
            if requested_ref is not None:
                _validate_reference(requested_ref, metadata.ref)
            connection.execute("DELETE FROM artifacts WHERE name = ?", (name,))
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        assert payload_relpath is not None
        self._remove_committed_orphan(payload_relpath)

    @staticmethod
    def _request(
        ref_or_name: ArtifactRef[T] | ArtifactRef[Any] | str,
    ) -> tuple[ArtifactRef[Any] | None, str]:
        if isinstance(ref_or_name, ArtifactRef):
            return ref_or_name, _normalize_name(ref_or_name.name)
        return None, _normalize_name(ref_or_name)

    def _fetch_current(
        self,
        connection: sqlite3.Connection,
        name: str,
    ) -> tuple[ArtifactMetadata, str] | None:
        row = connection.execute(
            f"SELECT {_ARTIFACT_COLUMNS} FROM artifacts WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return _metadata_from_catalog_row(row, store_id=self.store_id)

    def _metadata_from_value_file(
        self,
        *,
        name: str,
        artifact_id: str,
        payload_sha256: str,
        sensitivity: ArtifactSensitivity,
        created_at: str,
        value_file: ValueFileMetadata,
    ) -> ArtifactMetadata:
        return ArtifactMetadata(
            ref=ArtifactRef(
                store_id=self.store_id,
                name=name,
                artifact_id=artifact_id,
                value_type=value_file.value_type,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                context_id=value_file.context_id,
                nbytes=value_file.nbytes,
                payload_sha256=payload_sha256,
            ),
            sensitivity=sensitivity,
            created_at=created_at,
            value_schema_version=value_file.value_schema_version,
            tensor_metadata=value_file.tensor_metadata,
            value_metadata=value_file.value_metadata,
        )

    def _upsert_current(
        self,
        connection: sqlite3.Connection,
        metadata: ArtifactMetadata,
        *,
        payload_relpath: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO artifacts (
                name,
                artifact_id,
                artifact_schema_version,
                value_type,
                value_schema_version,
                context_id,
                nbytes,
                payload_sha256,
                payload_relpath,
                sensitivity,
                created_at,
                tensor_metadata_json,
                value_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                artifact_id=excluded.artifact_id,
                artifact_schema_version=excluded.artifact_schema_version,
                value_type=excluded.value_type,
                value_schema_version=excluded.value_schema_version,
                context_id=excluded.context_id,
                nbytes=excluded.nbytes,
                payload_sha256=excluded.payload_sha256,
                payload_relpath=excluded.payload_relpath,
                sensitivity=excluded.sensitivity,
                created_at=excluded.created_at,
                tensor_metadata_json=excluded.tensor_metadata_json,
                value_metadata_json=excluded.value_metadata_json
            """,
            (
                metadata.ref.name,
                metadata.ref.artifact_id,
                metadata.ref.artifact_schema_version,
                metadata.ref.value_type,
                metadata.value_schema_version,
                metadata.ref.context_id,
                metadata.ref.nbytes,
                metadata.ref.payload_sha256,
                payload_relpath,
                metadata.sensitivity,
                metadata.created_at,
                json.dumps(
                    metadata.tensor_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                json.dumps(
                    metadata.value_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )

    def _payload_path(self, payload_relpath: str) -> Path:
        path = PurePosixPath(payload_relpath)
        if path.is_absolute() or not path.parts:
            raise ValueError(
                f"Invalid artifact payload path: {payload_relpath!r}"
            )
        result = self.root.joinpath(*path.parts)
        if result.parent.is_symlink() or result.is_symlink():
            raise ValueError(
                f"Artifact payload path cannot be a symlink: {result}"
            )
        return result

    def _require_payload(
        self, payload_relpath: str, *, artifact_name: str
    ) -> Path:
        path = self._payload_path(payload_relpath)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"Artifact {artifact_name!r} is missing payload {path}"
            )
        return path

    def _remove_committed_orphan(self, payload_relpath: str) -> None:
        path = self._payload_path(payload_relpath)
        try:
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
            try:
                path.parent.rmdir()
            except OSError:
                pass
            _fsync_directory(self._objects_path)
        except OSError as error:
            warn(
                "Committed artifact catalog change left an unreachable payload "
                f"for recovery at {path}: {error}",
                ResourceWarning,
                stacklevel=2,
            )

    @staticmethod
    def _validate_value_file_metadata(
        artifact: ArtifactMetadata,
        value_file: ValueFileMetadata,
        *,
        artifact_name: str,
    ) -> None:
        mismatches = {}
        pairs = {
            "value_schema_version": (
                artifact.value_schema_version,
                value_file.value_schema_version,
            ),
            "value_type": (artifact.ref.value_type, value_file.value_type),
            "context_id": (artifact.ref.context_id, value_file.context_id),
            "nbytes": (artifact.ref.nbytes, value_file.nbytes),
            "tensor_metadata": (
                artifact.tensor_metadata,
                value_file.tensor_metadata,
            ),
            "value_metadata": (
                artifact.value_metadata,
                value_file.value_metadata,
            ),
        }
        for field, (expected, actual) in pairs.items():
            if expected != actual:
                mismatches[field] = (expected, actual)
        if mismatches:
            raise ValueError(
                f"Artifact {artifact_name!r} catalog does not match its "
                f"value file: {mismatches}"
            )


class ArtifactCollection:
    """A logical namespace for independently materializable artifacts."""

    def __init__(self, store: ArtifactStore, prefix: str) -> None:
        self.store = store
        self.prefix = prefix

    def put(self, name: str, value: T, **kwargs: Any) -> ArtifactRef[T]:
        """Persist ``value`` under this collection prefix."""

        return self.store.put(self._name(name), value, **kwargs)

    @overload
    def get(
        self,
        ref_or_name: ArtifactRef[Any],
        *,
        device: torch.device | str = "cpu",
        expected_type: type[U],
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> U: ...

    @overload
    def get(
        self,
        ref_or_name: ArtifactRef[T],
        *,
        device: torch.device | str = "cpu",
        expected_type: None = None,
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> T: ...

    @overload
    def get(
        self,
        ref_or_name: str,
        *,
        device: torch.device | str = "cpu",
        expected_type: type[U],
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> U | None: ...

    @overload
    def get(
        self,
        ref_or_name: str,
        *,
        device: torch.device | str = "cpu",
        expected_type: None = None,
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> TensorResident | None: ...

    def get(
        self,
        ref_or_name: ArtifactRef[Any] | str,
        *,
        device: torch.device | str = "cpu",
        expected_type: type[TensorResident] | None = None,
        expected_context_id: str | None = None,
        verify_checksum: bool = True,
    ) -> TensorResident | None:
        """Get a checked ref or optional collection-relative current value.

        A missing string name returns ``None``. A missing or replaced
        :class:`ArtifactRef` remains a stale-reference error.
        """

        if isinstance(ref_or_name, ArtifactRef):
            self._require_member(ref_or_name.name)
            return self.store.get(
                ref_or_name,
                device=device,
                expected_type=expected_type,
                expected_context_id=expected_context_id,
                verify_checksum=verify_checksum,
            )
        return self.store.get(
            self._name(ref_or_name),
            device=device,
            expected_type=expected_type,
            expected_context_id=expected_context_id,
            verify_checksum=verify_checksum,
        )

    def inspect(self, name: str) -> ArtifactMetadata:
        """Inspect a collection-relative artifact without loading tensors."""

        return self.store.inspect(self._name(name))

    def exists(self, name: str) -> bool:
        """Return whether a collection-relative current artifact exists."""

        return self.store.exists(self._name(name))

    def list(self) -> list[ArtifactRef[Any]]:
        """List current artifacts nested under this collection prefix."""

        return self.store.list(prefix=self.prefix)

    def delete(self, ref_or_name: ArtifactRef[Any] | str) -> None:
        """Delete an artifact constrained to this collection."""

        if isinstance(ref_or_name, ArtifactRef):
            self._require_member(ref_or_name.name)
            self.store.delete(ref_or_name)
            return
        self.store.delete(self._name(ref_or_name))

    def __iter__(self) -> Iterator[ArtifactRef[Any]]:
        return iter(self.list())

    def _name(self, name: str) -> str:
        item = _normalize_name(name)
        return f"{self.prefix}/{item}"

    def _require_member(self, name: str) -> None:
        if not name.startswith(self.prefix + "/"):
            raise ValueError(
                f"Artifact {name!r} is outside collection {self.prefix!r}"
            )
