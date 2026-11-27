# Named values and generations

**Example source:** [`examples/11_value_store.py`](https://github.com/VisualDust/fhelium/blob/main/examples/11_value_store.py)

`ValueStore` (`fhelium.store`) manages the current durable generation under each logical name. Example 11 demonstrates a collection, a generation-specific reference, replacement, and loading the current generation by name. [Example 10](value-memory-and-persistence.md) uses individual files without a catalog.

```bash
python examples/11_value_store.py
python examples/11_value_store.py --device cuda:0
```

The default uses a temporary directory. `--store PATH` retains the catalog and payloads at a caller-selected location; repeated runs replace the demonstration's `requests/activation` entry.

## Names and references

`store.collection("requests")` creates a namespace view. `collection.put(...)` publishes a generation and returns a `StoredValueRef`; `store.get(ref)` requires that generation. `collection.get("activation")` instead asks for the current value under that name.

After `put(..., overwrite=True)`, the previous reference raises `StaleValueReferenceError`. It never silently loads the replacement. The store does not retain historical generations. A missing logical name returns None; corruption and stale references remain errors.

The example verifies Tensor equality before and after replacement and prints the current catalog. Publishing does not move or release the caller's live value.

## Security and other payloads

Stored data is unencrypted. Typed SecretKey persistence requires `allow_secret=True`; sensitivity labels do not provide access control. The store also accepts a Compilation with `include_materials` selecting its saved Tensor bindings. That uses the same Program file codec as `fhelium.serialization.save_compilation`; see [Save and relink a Program](../how-to/save-and-relink-program.md). Arbitrary Tensor contents are not automatically classified as secret or public.

::: details Source
<<< @/../examples/11_value_store.py
:::

See [Store named values](../how-to/manage-value-store.md) for application cache patterns and storage lifecycle.
