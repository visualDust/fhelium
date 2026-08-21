from __future__ import annotations

import errno
import multiprocessing as mp
import os
import shutil
import sqlite3
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any
from uuid import uuid4

import pytest
import torch

import fhelium.artifacts.store as artifact_store_module
from fhelium.artifacts import ArtifactRef, ArtifactStore
from fhelium.core import Ciphertext, Plaintext, SecretKey, TensorResident
from fhelium.errors import (
    ArtifactError,
    StaleArtifactReferenceError,
    UnsupportedArtifactStoreVersionError,
)
from fhelium.serialization import ValueEnvelope

CATALOG_NAME = "catalog.sqlite3"


def _ciphertext(fill: int | None = None) -> Ciphertext:
    data = torch.arange(2 * 3 * 8, dtype=torch.int64).reshape(2, 3, 8)
    if fill is not None:
        data = torch.full_like(data, fill)
    return Ciphertext(
        data=data,
        level=1,
        scale=2.0**40,
        context_id="test-context",
        prime_ids=(1, 2, 3),
    )


def _assert_same_exact_value(
    actual: TensorResident,
    expected: TensorResident,
) -> None:
    assert type(actual) is type(expected)
    actual_envelope = ValueEnvelope.from_value(actual)
    expected_envelope = ValueEnvelope.from_value(expected)
    assert actual_envelope.context_id == expected_envelope.context_id
    assert actual_envelope.metadata == expected_envelope.metadata
    assert actual_envelope.tensors.keys() == expected_envelope.tensors.keys()
    for name, expected_tensor in expected_envelope.tensors.items():
        torch.testing.assert_close(
            actual_envelope.tensors[name], expected_tensor
        )


def _catalog_path(root: Path) -> Path:
    return root / CATALOG_NAME


def _payload_path(root: Path, name: str) -> Path:
    with sqlite3.connect(_catalog_path(root)) as connection:
        row = connection.execute(
            "SELECT payload_relpath FROM artifacts WHERE name = ?",
            (name,),
        ).fetchone()
    assert row is not None
    return root / str(row[0])


def _concurrent_put(
    root: str,
    start: Any,
    results: Any,
    fill: int,
) -> None:
    """Attempt one same-name put in a fresh spawned process."""

    try:
        start.wait(timeout=20)
        ref = ArtifactStore(root).put("race/item", _ciphertext(fill))
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error), fill))
    else:
        results.put(("ok", ref.artifact_id, "", fill))


def _concurrent_open(root: str, start: Any, results: Any) -> None:
    """Open one not-yet-initialized store in a fresh spawned process."""

    try:
        start.wait(timeout=20)
        store = ArtifactStore(root)
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)))
    else:
        results.put(("ok", store.store_id, ""))


def _abrupt_put_before_catalog_commit(root: str) -> None:
    """Terminate after catalog mutation but before transaction commit."""

    store = ArtifactStore(root)
    original_upsert = ArtifactStore._upsert_current

    def upsert_then_exit(
        self: ArtifactStore,
        connection: sqlite3.Connection,
        metadata: Any,
        *,
        payload_relpath: str,
    ) -> None:
        original_upsert(
            self,
            connection,
            metadata,
            payload_relpath=payload_relpath,
        )
        os._exit(73)

    ArtifactStore._upsert_current = upsert_then_exit
    store.put("crash/item", _ciphertext(fill=73))


def test_artifact_lifecycle_preserves_one_current_generation(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    collection = store.collection("pir/client-key")
    assert store.get("missing") is None
    assert collection.get("missing") is None
    original_value = _ciphertext()
    original = collection.put("block-000", original_value)

    assert isinstance(original, ArtifactRef)
    assert original.name == "pir/client-key/block-000"
    assert original.artifact_schema_version == 1
    assert len(original.store_id) == 36
    assert len(original.payload_sha256) == 64
    assert store.exists(original.name)
    assert store.inspect(original).ref == original
    assert list(collection) == [original]
    assert store.list(prefix="pir/client-key") == [original]
    loaded_from_collection = collection.get("block-000")
    assert loaded_from_collection is not None
    _assert_same_exact_value(loaded_from_collection, original_value)

    reopened = ArtifactStore(tmp_path)
    assert reopened.inspect(original).ref.store_id == original.store_id
    _assert_same_exact_value(reopened.get(original), original_value)

    with pytest.raises(ValueError, match="context mismatch"):
        store.get(original, expected_context_id="another-context")
    with pytest.raises(TypeError, match="expected Plaintext"):
        store.get(original, expected_type=Plaintext)

    replacement_value = _ciphertext(fill=1)
    replacement = store.put(
        original.name,
        replacement_value,
        overwrite=True,
    )
    assert replacement.artifact_id != original.artifact_id
    assert replacement.store_id == original.store_id
    with pytest.raises(StaleArtifactReferenceError) as stale:
        store.get(original)
    with pytest.raises(StaleArtifactReferenceError):
        store.delete(original)
    assert stale.value.expected_artifact_id == original.artifact_id
    assert stale.value.current_artifact_id == replacement.artifact_id
    _assert_same_exact_value(store.get(replacement), replacement_value)

    store.delete(replacement)
    assert not store.exists(replacement.name)
    with pytest.raises(StaleArtifactReferenceError):
        store.get(replacement)


def test_reference_from_another_store_is_rejected(tmp_path: Path) -> None:
    first = ArtifactStore(tmp_path / "first")
    second = ArtifactStore(tmp_path / "second")
    first_ref = first.put("values/item", _ciphertext())
    second_ref = second.put("values/item", _ciphertext())

    assert first_ref.store_id != second_ref.store_id
    with pytest.raises(StaleArtifactReferenceError) as stale:
        second.get(first_ref)
    with pytest.raises(StaleArtifactReferenceError):
        second.delete(first_ref)
    assert "store_id" in stale.value.differences
    _assert_same_exact_value(second.get(second_ref), _ciphertext())


def test_integrity_catalog_consistency_and_names_are_enforced(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put("values/item", _ciphertext())
    payload = _payload_path(tmp_path, ref.name)
    with payload.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="checksum mismatch"):
        store.get(ref)

    for invalid in ("", "/absolute", "../escape", "a/../b", ".hidden"):
        with pytest.raises(ValueError):
            store.put(invalid, _ciphertext())

    other_root = tmp_path / "metadata-mismatch"
    other = ArtifactStore(other_root)
    other_ref = other.put("values/item", _ciphertext())
    with sqlite3.connect(_catalog_path(other_root)) as connection:
        connection.execute(
            "UPDATE artifacts SET nbytes = nbytes + 1 WHERE name = ?",
            (other_ref.name,),
        )
    with pytest.raises(ValueError, match="does not match"):
        other.inspect(other_ref.name)


def test_secret_key_requires_explicit_unencrypted_persistence_opt_in(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    secret_key = SecretKey(
        data=torch.arange(3 * 8, dtype=torch.int64).reshape(3, 8),
        context_id="test-context",
        prime_ids=(0, 1, 2),
    )

    with pytest.raises(PermissionError, match="disabled by default"):
        store.put("keys/secret", secret_key)

    ref = store.put("keys/secret", secret_key, allow_secret=True)
    assert store.inspect(ref).sensitivity == "secret"
    _assert_same_exact_value(
        store.get(ref, expected_type=SecretKey),
        secret_key,
    )


def test_store_version_and_unrecognized_nonempty_roots_fail_closed(
    tmp_path: Path,
) -> None:
    versioned_root = tmp_path / "versioned"
    ArtifactStore(versioned_root)
    with sqlite3.connect(_catalog_path(versioned_root)) as connection:
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(UnsupportedArtifactStoreVersionError) as unsupported:
        ArtifactStore(versioned_root)
    assert unsupported.value.found_version == 2
    assert unsupported.value.supported_versions == (1,)
    assert not unsupported.value.migration_available

    interrupted = tmp_path / "interrupted-initialization"
    interrupted.mkdir()
    sqlite3.connect(_catalog_path(interrupted)).close()
    recovered = ArtifactStore(interrupted)
    assert recovered.list() == []
    with sqlite3.connect(_catalog_path(interrupted)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)

    unrecognized = tmp_path / "unrecognized"
    unrecognized.mkdir()
    (unrecognized / "legacy-artifact").mkdir()
    (unrecognized / "legacy-artifact" / "manifest.json").write_text("{}")
    sqlite3.connect(_catalog_path(unrecognized)).close()
    with pytest.raises(
        ArtifactError,
        match="non-empty|unrecognized|catalog|artifact-store format",
    ):
        ArtifactStore(unrecognized)


def test_catalog_identity_and_schema_fail_closed(tmp_path: Path) -> None:
    identity_root = tmp_path / "identity"
    ArtifactStore(identity_root)
    with sqlite3.connect(_catalog_path(identity_root)) as connection:
        connection.execute(
            "UPDATE store_metadata SET value = ? WHERE key = 'store_id'",
            ("00000000-0000-0000-0000-00000000000A",),
        )
    with pytest.raises(ValueError, match="canonical UUID"):
        ArtifactStore(identity_root)

    schema_root = tmp_path / "schema"
    ArtifactStore(schema_root)
    with sqlite3.connect(_catalog_path(schema_root)) as connection:
        metadata = connection.execute(
            "SELECT key, value FROM store_metadata"
        ).fetchall()
        connection.execute("DROP TABLE store_metadata")
        connection.execute(
            "CREATE TABLE store_metadata ("
            "key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
            metadata,
        )
    with pytest.raises(ValueError, match="table definitions"):
        ArtifactStore(schema_root)

    trigger_root = tmp_path / "trigger"
    ArtifactStore(trigger_root)
    with sqlite3.connect(_catalog_path(trigger_root)) as connection:
        connection.execute(
            "CREATE TRIGGER alter_artifact_writes "
            "BEFORE INSERT ON artifacts BEGIN SELECT 1; END"
        )
    with pytest.raises(ValueError, match="unexpected schema objects"):
        ArtifactStore(trigger_root)


def test_store_owned_paths_reject_symbolic_and_hard_links(
    tmp_path: Path,
) -> None:
    target = tmp_path / "caller-owned"
    target.write_text("unchanged")
    if os.name != "nt":
        target.chmod(0o644)
    linked_root = tmp_path / "linked-lock"
    linked_root.mkdir()
    try:
        (linked_root / ".store.lock").symlink_to(target)
    except OSError as error:
        # Creating symlinks can require an explicit Windows privilege. This
        # says nothing about the root's ACL and does not waive hard-link checks.
        if os.name != "nt" or getattr(error, "winerror", None) != 1314:
            raise
    else:
        with pytest.raises(ValueError, match="lock.*symlink"):
            ArtifactStore(linked_root)
        assert target.read_text() == "unchanged"
        if os.name != "nt":
            assert target.stat().st_mode & 0o777 == 0o644

    hardlink_target = tmp_path / "hardlink-target"
    hardlink_target.write_text("unchanged")
    if os.name != "nt":
        hardlink_target.chmod(0o644)
    hardlinked_root = tmp_path / "hardlinked-lock"
    hardlinked_root.mkdir()
    os.link(hardlink_target, hardlinked_root / ".store.lock")
    with pytest.raises(ValueError, match="exactly one hard link"):
        ArtifactStore(hardlinked_root)
    assert hardlink_target.read_text() == "unchanged"
    if os.name != "nt":
        assert hardlink_target.stat().st_mode & 0o777 == 0o644

    source_root = tmp_path / "catalog-source"
    source = ArtifactStore(source_root)
    source_catalog_bytes = _catalog_path(source_root).read_bytes()
    hardlinked_catalog_root = tmp_path / "hardlinked-catalog"
    hardlinked_catalog_root.mkdir()
    os.link(
        _catalog_path(source_root),
        _catalog_path(hardlinked_catalog_root),
    )
    with pytest.raises(ValueError, match="catalog.*exactly one hard link"):
        ArtifactStore(hardlinked_catalog_root)
    assert _catalog_path(source_root).read_bytes() == source_catalog_bytes
    _catalog_path(hardlinked_catalog_root).unlink()
    reopened_source = ArtifactStore(source_root)
    assert reopened_source.store_id == source.store_id
    assert reopened_source.list() == []


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX mode bits neither configure nor verify Windows ACL privacy",
)
def test_store_owned_posix_paths_apply_private_modes(tmp_path: Path) -> None:
    root = tmp_path / "permissions"
    root.mkdir(mode=0o750)
    root.chmod(0o750)
    store = ArtifactStore(root)
    ref = store.put("values/item", _ciphertext())
    assert root.stat().st_mode & 0o777 == 0o750
    assert _catalog_path(root).stat().st_mode & 0o777 == 0o600
    assert (root / ".store.lock").stat().st_mode & 0o777 == 0o600
    assert (root / "objects").stat().st_mode & 0o777 == 0o700
    assert (root / "tmp").stat().st_mode & 0o777 == 0o700
    assert _payload_path(root, ref.name).stat().st_mode & 0o777 == 0o600


def test_payload_publication_is_write_once_and_no_replace(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published.safetensors"
    destination.write_bytes(b"committed")
    colliding_source = tmp_path / "collision.safetensors"
    colliding_source.write_bytes(b"replacement")

    with pytest.raises(FileExistsError) as collision:
        artifact_store_module._publish_file(colliding_source, destination)

    assert collision.value.errno == errno.EEXIST
    if os.name == "nt":
        assert getattr(collision.value, "winerror", None) in {80, 183}
    assert destination.read_bytes() == b"committed"
    assert colliding_source.read_bytes() == b"replacement"

    source = tmp_path / "new.safetensors"
    published = tmp_path / "new-published.safetensors"
    source.write_bytes(b"new payload")
    artifact_store_module._publish_file(source, published)
    assert not source.exists()
    assert published.read_bytes() == b"new payload"


def test_artifact_id_collision_cannot_replace_a_committed_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    fixed_id = uuid4()
    monkeypatch.setattr(artifact_store_module, "uuid4", lambda: fixed_id)
    original_value = _ciphertext(fill=41)
    original = store.put("values/original", original_value)

    with pytest.raises(ArtifactError, match="collides"):
        store.put("values/collision", _ciphertext(fill=42))

    assert store.list() == [original]
    _assert_same_exact_value(store.get(original), original_value)


def test_reopen_recovers_tmp_and_unreferenced_objects_only(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put("values/current", _ciphertext())
    referenced = _payload_path(tmp_path, ref.name)

    orphan_id = str(uuid4())
    orphan = tmp_path / "objects" / orphan_id[:2] / f"{orphan_id}.safetensors"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(referenced, orphan)
    temporary = tmp_path / "tmp" / f"{uuid4()}.safetensors"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(referenced, temporary)

    reopened = ArtifactStore(tmp_path)
    assert referenced.is_file()
    assert not orphan.exists()
    assert not temporary.exists()
    _assert_same_exact_value(reopened.get(ref), _ciphertext())


def test_abrupt_process_death_before_commit_recovers_published_object(
    tmp_path: Path,
) -> None:
    ArtifactStore(tmp_path)
    context = mp.get_context("spawn")
    process = context.Process(
        target=_abrupt_put_before_catalog_commit,
        args=(str(tmp_path),),
    )
    process.start()
    process.join(timeout=30)
    assert not process.is_alive()
    assert process.exitcode == 73

    reopened = ArtifactStore(tmp_path)
    assert reopened.list() == []
    assert tuple((tmp_path / "objects").rglob("*.safetensors")) == ()
    assert tuple((tmp_path / "tmp").iterdir()) == ()


def test_two_processes_cannot_both_create_one_absent_name(
    tmp_path: Path,
) -> None:
    ArtifactStore(tmp_path)
    context = mp.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_put,
            args=(str(tmp_path), start, results, fill),
        )
        for fill in (11, 22)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0

    try:
        outcomes = [results.get(timeout=5) for _ in processes]
    except Empty as error:
        raise AssertionError(
            "Artifact writer did not report an outcome"
        ) from error
    assert [outcome[0] for outcome in outcomes].count("ok") == 1
    errors = [outcome for outcome in outcomes if outcome[0] == "error"]
    assert len(errors) == 1
    assert errors[0][1] == "FileExistsError"

    winning_fill = next(
        outcome[3] for outcome in outcomes if outcome[0] == "ok"
    )
    winning_value = ArtifactStore(tmp_path).get("race/item")
    assert winning_value is not None
    _assert_same_exact_value(winning_value, _ciphertext(winning_fill))


def test_two_processes_safely_initialize_one_new_store(tmp_path: Path) -> None:
    root = tmp_path / "concurrent-initialization"
    context = mp.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_open,
            args=(str(root), start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0

    try:
        outcomes = [results.get(timeout=5) for _ in processes]
    except Empty as error:
        raise AssertionError(
            "Artifact store opener did not report an outcome"
        ) from error
    assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
    assert len({outcome[1] for outcome in outcomes}) == 1
    assert ArtifactStore(root).store_id == outcomes[0][1]


def test_failed_catalog_commit_preserves_previous_generation(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    original_value = _ciphertext(fill=7)
    original = store.put("values/item", original_value)
    original_payload = _payload_path(tmp_path, original.name)

    with sqlite3.connect(_catalog_path(tmp_path)) as connection:
        connection.executescript(
            """
            CREATE TRIGGER fail_artifact_update
            BEFORE UPDATE ON artifacts
            BEGIN
                SELECT RAISE(ABORT, 'injected catalog failure');
            END;
            """
        )

    with pytest.raises(
        sqlite3.IntegrityError, match="injected catalog failure"
    ):
        store.put("values/item", _ciphertext(fill=8), overwrite=True)

    assert original_payload.is_file()
    _assert_same_exact_value(store.get(original), original_value)

    with sqlite3.connect(_catalog_path(tmp_path)) as connection:
        connection.execute("DROP TRIGGER fail_artifact_update")
    reopened = ArtifactStore(tmp_path)
    payload_files = tuple((tmp_path / "objects").rglob("*.safetensors"))
    assert payload_files == (original_payload,)
    _assert_same_exact_value(reopened.get(original), original_value)


def test_overwrite_waits_for_active_reader_before_retiring_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    original_value = _ciphertext(fill=31)
    original = store.put("values/item", original_value)
    original_payload = _payload_path(tmp_path, original.name)

    reader_started = Event()
    release_reader = Event()
    writer_reached_commit = Event()
    writer_finished = Event()
    reader_values: list[TensorResident] = []
    writer_refs: list[ArtifactRef[Ciphertext]] = []
    failures: list[BaseException] = []

    original_load_value = artifact_store_module.load_value

    def blocking_load_value(*args: Any, **kwargs: Any) -> TensorResident:
        reader_started.set()
        if not release_reader.wait(timeout=10):
            raise TimeoutError("reader release was not signaled")
        return original_load_value(*args, **kwargs)

    original_upsert = ArtifactStore._upsert_current

    def signaling_upsert(
        self: ArtifactStore,
        connection: sqlite3.Connection,
        metadata: Any,
        *,
        payload_relpath: str,
    ) -> None:
        original_upsert(
            self,
            connection,
            metadata,
            payload_relpath=payload_relpath,
        )
        writer_reached_commit.set()

    monkeypatch.setattr(
        artifact_store_module,
        "load_value",
        blocking_load_value,
    )
    monkeypatch.setattr(ArtifactStore, "_upsert_current", signaling_upsert)

    def read_current() -> None:
        try:
            reader_values.append(store.get(original))
        except BaseException as error:
            failures.append(error)

    def overwrite_current() -> None:
        try:
            writer_refs.append(
                store.put(
                    original.name,
                    _ciphertext(fill=32),
                    overwrite=True,
                )
            )
        except BaseException as error:
            failures.append(error)
        finally:
            writer_finished.set()

    reader = Thread(target=read_current)
    writer = Thread(target=overwrite_current)
    reader.start()
    assert reader_started.wait(timeout=10)
    writer.start()
    assert writer_reached_commit.wait(timeout=10)
    assert not writer_finished.wait(timeout=0.1)
    assert original_payload.is_file()

    release_reader.set()
    reader.join(timeout=10)
    writer.join(timeout=10)
    assert not reader.is_alive()
    assert not writer.is_alive()
    assert failures == []
    _assert_same_exact_value(reader_values[0], original_value)
    assert not original_payload.exists()
    _assert_same_exact_value(store.get(writer_refs[0]), _ciphertext(fill=32))


def test_exists_requires_current_catalog_binding_and_payload(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put("values/item", _ciphertext())
    assert store.exists(ref.name)

    _payload_path(tmp_path, ref.name).unlink()
    assert not store.exists(ref.name)
    with pytest.raises(FileNotFoundError, match="missing payload"):
        store.get(ref.name)
    with pytest.raises(FileNotFoundError, match="missing|non-regular"):
        ArtifactStore(tmp_path)
