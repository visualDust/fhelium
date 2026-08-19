# ArtifactStore internals

`ArtifactStore` implements a transactional repository for exact FHElium values
on one trusted host. Its on-disk format separates exact value identity from
logical artifact identity through a SQLite catalog, immutable payloads, and
defined recovery behavior.

## Storage stack

```mermaid
graph TB
    APP[Python application]
    API[ArtifactStore put / get / delete]
    TX[SQLite transaction<br/>logical name and current generation]
    SER[FHElium exact serialization<br/>ValueEnvelope + safetensors]
    STAGE[tmp staging file<br/>0600 + file fsync]
    OBJ[Immutable object file<br/>UUID path + SHA-256]
    CAT[catalog.sqlite3<br/>STRICT tables]
    FS[Local POSIX filesystem<br/>locks, hard links, directory fsync]

    APP --> API
    API --> TX --> CAT
    API --> SER --> STAGE --> OBJ
    CAT --> FS
    OBJ --> FS
```

The SQLite catalog stores repository identity, logical names, current
generations, exact-value metadata, object paths, and checksums. Tensor bytes
are written by `fhelium.serialization` into immutable safetensors payloads.
Catalog transactions order publication; POSIX no-clobber links and `fsync`
make the corresponding object-file transition durable.

The implementation uses the Python `sqlite3` module and local filesystem
operations. It does not place tensor BLOBs in SQLite or reinterpret exact value
metadata independently of the serialization layer.

## Repository layout

A store root owns four internal paths:

```text
ROOT/
├── .store.lock
├── catalog.sqlite3
├── objects/
│   └── <uuid-prefix>/<artifact-uuid>.safetensors
└── tmp/
```

- `.store.lock` serializes initialization and open-time recovery.
- `catalog.sqlite3` owns logical names, current generations, store identity,
  and transaction ordering.
- `objects/` contains immutable exact-value files addressed by artifact UUID.
- `tmp/` contains unpublished staging files only.

The catalog does not contain tensor BLOBs. Tensor payloads remain ordinary
versioned FHElium safetensors files so the serialization layer retains sole
ownership of exact value encoding and reconstruction.

## Catalog identity and schema

Format v1 requires SQLite 3.37 or later, `STRICT` tables,
rollback-journal `DELETE` mode, and `synchronous=FULL`. The store metadata table
contains exactly:

- `format = "fhelium-artifact-store"`;
- one canonical UUID `store_id`.

The artifact table has one row per normalized logical name. A row records the
current artifact UUID, value and artifact schema versions, concrete value type,
context identity, logical tensor bytes, payload SHA-256, immutable object path,
sensitivity label, creation time, and the tensor/value metadata copied from the
exact-value file header.

Opening a store validates the schema version, exact table definitions,
unexpected schema objects, catalog identity, canonical UUIDs, row structure,
and referenced object presence. Unsupported versions fail closed; v1 provides
no migration API.

The catalog path and bootstrap lock must each be private regular files with one
hard link. Symlinks and shared hard-linked inodes are rejected before SQLite or
permission-changing operations can mutate another pathname's file.

## Publishing a generation

`put(name, value)` uses the following ordering:

1. Validate the logical name and value policy, then allocate a candidate
   artifact UUID and its object/staging paths.
2. Start `BEGIN IMMEDIATE`, serializing participating writers.
3. Resolve the existing name, enforce `overwrite`, and reject candidate UUID
   collision with either catalog or object state.
4. Snapshot the supported exact value state into `tmp/` through `save_value`.
5. Set private permissions and `fsync` the staging file.
6. Validate the staged file header and compute its complete payload SHA-256.
7. Publish the object with no-clobber hard-link semantics, remove the staging
   name, and `fsync` the modified directories.
8. Insert or replace the one current catalog row and commit SQLite.
9. After commit, retire the previous unreachable object.

A committed row therefore never points to a partially written object. Process
death before catalog commit may leave a durable but unreachable object; it does
not publish a generation. Process death after commit may leave the previous
object temporarily present; it is no longer reachable by name.

`put` snapshots persistence state without changing the caller's live value.
Device movement and memory release remain ordinary value/residency operations.

## Reading and retirement

`get(name_or_ref)` starts a rollback-journal read transaction before resolving
the current row. The transaction remains open through:

- optional `ArtifactRef` generation validation;
- context and expected-type checks;
- payload presence and optional SHA-256 verification;
- catalog/file-header cross-validation;
- exact value reconstruction through `load_value`.

In SQLite `DELETE` mode, a writer may prepare a replacement concurrently but
cannot commit while that read transaction remains active. The old object is
removed only after the writer commits, so a selected generation cannot be
retired during materialization.

A string name with no current row is an ordinary repository miss and returns
`None`. A generation-specific reference that is missing, replaced, deleted, or
belongs to another store raises `StaleArtifactReferenceError`. Missing payloads,
checksum failures, malformed metadata, and type/context mismatches remain
errors rather than cache misses.

## Recovery

Store construction holds `.store.lock` while validating catalog identity and
schema, then begins an exclusive SQLite transaction for row/object recovery:

1. Validate catalog version, identity, schema, and row/object bindings.
2. Reject any catalog row whose referenced object is absent or non-regular.
3. Remove all staging entries under `tmp/`.
4. Remove object files not referenced by the current catalog rows.
5. `fsync` every modified surviving object directory.

Recovery never promotes an orphan into the catalog and never treats an old
object as retained version history. Exactly one generation per logical name is
reachable after recovery.

## Filesystem requirements and security model

The supported deployment is one trusted host on a local POSIX filesystem with
working advisory locks, same-filesystem hard links, atomic local filesystem
operations, and file/directory `fsync`. NFS, SMB, FUSE/object-store mounts,
multi-host writers, and processes that modify internal files outside the API
are unsupported.

New internal directories use mode `0700`; the catalog, lock, and payload files
use `0600`. An existing store-root mode remains application-owned.

SHA-256 detects accidental payload corruption but is not authenticated
integrity. A writer that can modify both catalog and payload can replace both.
Sensitivity labels do not encrypt data or enforce access control. Secret-key
persistence requires explicit `allow_secret=True` and still requires an
application-owned at-rest security policy.
