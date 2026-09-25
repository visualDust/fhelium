# Named artifacts and generations

**Example source:** [`examples/10_artifact_store.py`](https://github.com/VisualDust/fhelium/blob/main/examples/10_artifact_store.py)

ArtifactStore manages the current durable generation under each logical name. Example 10 demonstrates a collection, a generation-specific reference, replacement, and loading the current generation by name. [Example 09](value-memory-and-persistence.md) uses individual files without a catalog.

```bash
python examples/10_artifact_store.py
python examples/10_artifact_store.py --device cuda:0
```

The default uses a temporary directory. `--store PATH` retains the catalog and payloads at a caller-selected location; repeated runs replace the demonstration's `requests/activation` entry.

## Names and references

`store.collection("requests")` creates a namespace view. `collection.put(...)` publishes a generation and returns an `ArtifactRef`; `store.get(ref)` requires that generation. `collection.get("activation")` instead asks for the current value under that name.

After `put(..., overwrite=True)`, the previous reference raises `StaleArtifactReferenceError`. It never silently loads the replacement. The store does not retain historical generations. A missing logical name returns None; corruption and stale references remain errors.

The example verifies Tensor equality before and after replacement and prints the current catalog. Publishing does not move or release the caller's live value.

## Security and other payloads

Stored data is unencrypted. Typed SecretKey persistence requires `allow_secret=True`; sensitivity labels do not provide access control. The store also accepts a Compilation with `include_materials` selecting its saved Tensor bindings. That uses the same Serialization codec as [Example 16](compile-material-persistence.md). Arbitrary Tensor contents are not automatically classified as secret or public.

::: details Source
<<< @/../examples/10_artifact_store.py
:::

See [Manage artifacts by logical name](../how-to/manage-artifacts.md) for application cache patterns and storage lifecycle.
