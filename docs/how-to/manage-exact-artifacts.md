# Manage exact artifacts by logical name

Use `ArtifactStore` when an application needs stable local names and
transactional replacement for exact FHElium values. Each successful write
creates an immutable stored version, called a **generation**; exactly one
generation is current for each logical name. Use direct serialization when the
application already owns the complete file path and does not need a repository
namespace.

| Task | API |
| --- | --- |
| Save or load one caller-selected file | `fhelium.save_value` / `fhelium.load_value` |
| Inspect one caller-selected file | `fhelium.inspect_value` |
| Load the current value stored under a logical name | `ArtifactStore.get(name)` |
| Retain a checked identity for one exact stored version | `ArtifactRef` |
| Group names under one namespace | `ArtifactStore.collection(prefix)` |

An artifact store persists exact supported values. It does not persist an
engine, application object graph, arbitrary tensor container, or live Residency
state.

## 1. Create or open a supported local store

```python
from pathlib import Path

import fhelium as fh
from fhelium.artifacts import ArtifactStore

store = ArtifactStore(Path("state") / "artifact-store")
```

Use one trusted host and a local POSIX filesystem. The store relies on ordinary
SQLite locking, same-filesystem atomic replacement, and file/directory `fsync`. NFS,
SMB, FUSE or object-store mounts, multi-host writers, and processes that modify
the catalog or object files outside `ArtifactStore` are unsupported.

Opening a missing or empty root initializes the catalog. Opening an existing
store validates its format, schema, identity, catalog rows, and that every
referenced payload is a present regular file, then removes unreachable staging
and orphan files. It does not inspect every value header or checksum during
open; use `inspect` or `get` for those validations. It does not migrate an
unknown non-empty directory or unsupported schema.

## 2. Save a value under a logical name

```python
activation_ref = store.put(
    "requests/example/activation",
    ciphertext_cpu,
    sensitivity="confidential",
)
```

`put` snapshots the supported exact value into an independent durable payload
and returns a tensor-free `ArtifactRef`. It does not move, mutate, offload, or
release `ciphertext_cpu`.

Without `overwrite=True`, writing an existing logical name raises
`FileExistsError`. This defines create-if-absent behavior:

```python
try:
    factor_ref = store.put("model/v1/factor", prepared_factor)
except FileExistsError:
    factor_ref = store.inspect("model/v1/factor").ref
```

The returned reference records the store ID, normalized logical name, artifact
ID, value type, context ID, logical bytes, and payload checksum. It contains no
tensor payload.

## 3. Load the current value or one exact stored version

Use a string name when the application wants whichever generation is current:

```python
current = store.get(
    "requests/example/activation",
    expected_type=fh.Ciphertext,
    expected_context_id=engine.context.context_id,
    device=engine.device,
)
if current is None:
    ...  # no current generation for this name
```

A string lookup returns `None` only for a missing name. Corruption, checksum,
type, context, and schema failures remain errors.

Use an `ArtifactRef` when the caller requires the exact checked generation:

```python
restored = store.get(
    activation_ref,
    expected_type=fh.Ciphertext,
    expected_context_id=engine.context.context_id,
    device="cpu",
)
```

A reference lookup never converts a missing or replaced generation into a cache
miss. A cross-store, replaced, or deleted reference raises
`StaleArtifactReferenceError`.

`get` verifies the payload checksum by default. Keep
`verify_checksum=True` for durable reads unless a separately justified trusted
pipeline owns equivalent integrity validation.

## 4. Replace the value under a name

```python
replacement_ref = store.put(
    "requests/example/activation",
    replacement,
    sensitivity="confidential",
    overwrite=True,
)
```

Replacement writes a new artifact ID and makes every older reference for the
name stale. The store retains one active generation per logical name; it is not
a version-history repository.

Use the reference when deleting must be compare-and-delete rather than
name-based deletion:

```python
store.delete(replacement_ref)
```

If another generation replaced `replacement_ref` first, deletion fails stale
instead of deleting the newer value. `store.delete(name)` instead deletes the
current generation named at execution time. Neither operation destroys live
values that were already reconstructed.

## 5. Organize and inspect logical names

```python
keys = store.collection("model/v1/rotation-keys")
step_one = keys.put("step-1", rotation_key)
step_two = keys.put("step-2", second_rotation_key)

for ref in keys.list():
    print(ref.name, ref.value_type, ref.nbytes)
```

Collection methods resolve names relative to one normalized prefix. The store
also supports `store.list(prefix="model/v1")` for a broader inventory.

Use `exists(name)` only as a lightweight availability probe. It does not verify
the checksum or fully inspect the payload. Use `inspect(name_or_ref)` to validate
catalog metadata and the exact-value header without reconstructing tensors, and
use `get(...)` for the default complete checksum and reconstruction path.

## 6. Save secret keys only with explicit authorization

Secret-key persistence is disabled by default:

```python
secret_ref = store.put(
    "private/client-a/secret-key",
    secret_key,
    sensitivity="secret",
    allow_secret=True,
)
```

`allow_secret=True` authorizes an unencrypted write; it does not provide
encryption, access control, audit logging, or key management. Secret keys must
use `sensitivity="secret"`. All sensitivity values are descriptive metadata,
not enforcement.

The payload SHA-256 detects accidental corruption but is not authenticated
integrity against an actor that can modify both catalog and payload. The
application remains responsible for filesystem permissions, encrypted storage,
credential/KMS policy, backups, audit, and deletion policy.

## 7. Handle concurrent writes

Writers are serialized by SQLite, and readers retain a catalog snapshot through
payload validation and reconstruction. For a create race, at most one
`put(name, value)` succeeds. A losing process should handle `FileExistsError`
and load or inspect the winner:

```python
try:
    ref = store.put(name, computed_value)
except FileExistsError:
    value = store.get(name, expected_type=type(computed_value))
    if value is None:
        raise RuntimeError("artifact disappeared after create race")
```

With `overwrite=True`, the last successfully committed write becomes current;
the store does not merge values. Coordinate higher-level application policy
when multiple writers must not replace each other.

## 8. Verify a complete operational round trip

For every persisted value category, test the repository's operational semantics
and one real consumer operation:

1. Save under a temporary logical name and retain the returned reference.
2. Assert `store.inspect(ref).ref == ref`.
3. Reconstruct with `expected_type`, `expected_context_id`, and target
   `device`.
4. Use the reconstructed value in a representative FHElium operation and check
   its mathematical result.
5. Save a replacement with `overwrite=True` and assert the old reference
   raises `StaleArtifactReferenceError`.
6. Delete the replacement by reference and assert the logical name is absent.
7. Reopen the store from the same root and verify the remaining inventory.

A tensor byte comparison alone does not prove that level, scale, prime IDs,
polynomial domain, modulus basis, residue representation, key relation, or
context identity was reconstructed correctly.

## Related documentation

- [Serialization and artifacts](../concepts/execution/serialization-and-artifacts.md)
- [Values, memory, and persistence](../tutorial/value-memory-and-persistence.md)
- [ArtifactStore v1 internals](../developer/artifact-store-v1.md)
- [Artifact store API](../api/fhelium/artifacts/store.md)
- [Artifact reference API](../api/fhelium/artifacts/artifact.md)
- [Serialization API](../api/fhelium/serialization/value.md)
