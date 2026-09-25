# Compilation persistence and rebinding

Compilation persistence stores textual Program IR and optionally selected Tensor bindings in one versioned safetensors file. `fhelium.serialization.save_compilation`, `inspect_compilation`, and `load_compilation` implement the format in `serialization/compilation.py`. Restored Programs can reenter manual pipelines or callable preparation with new execution bindings.

## What is included?

The file contains Program text and a manifest describing selected Tensor objects, their byte storages, and the symbol-to-object mapping. `include_materials=False` is the default and saves no numerical bindings. `True` includes all current bindings; a collection of symbols selects a subset. Unselected or originally absent symbols remain external.

Workspaces, pass reports, callable caches, generated host closures, compiled GPU binaries, process groups, and live sampler objects are not serialized. These objects have process-specific execution identity. A loaded Compilation begins with its restored Program and binding dictionary; callers supply configuration and execution preparation needed for the next stage.

This helper saves only Program IR and reloads it on CPU:

```python
from fhelium.serialization import (
    inspect_compilation,
    load_compilation,
    save_compilation,
)

def persist_program(compilation, path):
    save_compilation(compilation, path)
    metadata = inspect_compilation(path)
    restored = load_compilation(path, device="cpu")
    return metadata, restored
```

Saving rejects an existing destination unless `overwrite=True`. Loading does not run passes, prepare resources, compile kernels, or execute the Program. `inspect_compilation` reads and validates metadata without loading numerical payloads.

## How are aliases and strided views represented?

The writer groups selected Tensors by underlying storage and distinguishes Tensor object identity from storage identity. A manifest record carries dtype, shape, stride, storage-relative offset, conjugate/negative view bits, and `requires_grad`. Multiple symbols pointing to one Tensor remain references to the same restored Tensor object. Distinct views of one storage retain their shared storage relationship.

For a nonempty view with shape $n_i$, nonnegative strides $s_i$, and element width $w$, the address span is

$$
\operatorname{span}=w\left(1+\sum_i(n_i-1)s_i\right).
$$

The serialized storage spans the selected views with an aligned origin. Only bytes covered by those views are copied; holes are zero-filled. This avoids copying unrelated bytes merely because a selected view shares a larger allocation. Sparse, quantized, and other unsupported Tensor layouts are rejected; the supported payload is strided, non-quantized Tensor storage.

Loading copies each stored byte storage once into independent writable memory on the requested device and reconstructs the views. Original addresses, original devices, and autograd history are not retained. The `requires_grad` flag is retained, but it does not restore a computation graph.

## What must be rebound?

Inspect the restored Program's material references and supply unresolved entries through its `material_bindings` dictionary or the optional [preparation utility](materials-and-preparation.md). Then run any required transformations and link with an `OperationBackend` containing compatible implementations and non-Tensor resources.

A saved implementation assignment remains part of the Program. It must be supported at the execution site. Loading numerical storage onto another device does not retune the Program's transform schedule or erase represented layout constraints. Reevaluate the relevant preparation facts before linking for a different execution environment.

A persisted Program is distinct from a persisted public value. `serialization/value.py` and `serialization/safetensors.py` serialize registered value metadata and payloads, while [ArtifactStore](artifact-store-v1.md) adds named generations, catalog operations, and crash recovery around persistent artifacts.

## Security and synchronization

Material inclusion is opt-in because arbitrary Tensor data can contain secret keys, plaintexts, RNG state, or other sensitive information. The file is unencrypted, and serialization does not detect sensitive contents. Stream-state snapshots require particular care: restoring the same random state into independent generators can repeat randomness.

The caller must synchronize with live writers before saving. Shared storage preservation is not an atomic snapshot of concurrent Tensor mutation. Safetensors storage and manifest validation establish format and extent constraints; they do not establish the provenance or safety of the Program's executable extensions. Treat imported IR, Backend implementations, and generated code according to their trust requirements.
