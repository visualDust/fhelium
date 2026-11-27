# Program files and the compile cache

Compilation persistence stores textual Program IR and optionally selected Tensor bindings in one versioned safetensors file. `fhelium.serialization.save_compilation`, `inspect_compilation`, and `load_compilation` implement the format in `serialization/compilation.py`. Restored Programs can reenter manual pipelines or callable preparation with new execution bindings.

## What is included?

The file contains Program text and a manifest describing selected Tensor objects, their byte storages, and the symbol-to-object mapping. `include_materials=False` is the default and saves no numerical bindings. `True` includes all current bindings; a collection of symbols selects a subset. Unselected or originally absent symbols remain external.

This serialization format excludes workspaces, pass reports, resident callable specializations, linked schedules, compiled GPU binaries, process groups, and live sampler objects. These objects have process-specific execution identity. A loaded Compilation begins with its restored Program and binding dictionary; callers supply configuration and execution preparation needed for the next stage.

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

A persisted Program is distinct from a persisted public value. `serialization/value.py` and `serialization/safetensors.py` serialize registered value metadata and payloads, while [ValueStore](value-store-v1.md) adds named generations, catalog operations, and crash recovery around stored values. A [compiled artifact](compiled-artifact-format.md) stores linked schedules instead of Program text.

## Callable reuse and persistent caches

Callable preparation checks two levels in order: a resident specialization for matching `CallSignature` conditions, then a persistent cache entry for that input specialization. A miss at both levels captures the source, runs the pipeline, and links. `compile/_cache.py` implements source identity, the cache key, material relocation, and `CompileCache`.

`cache=True` selects `FHELIUM_CACHE_DIR`, or `$XDG_CACHE_HOME/fhelium/compile` with `~/.cache` as the default base. A path selects another directory, and `cache=False` disables disk reads and writes while preserving resident specializations. Each entry is one [compiled artifact](compiled-artifact-format.md) at `<directory>/<key[:2]>/<key>.fhelium`. Its manifest records the specialization under `cache`: the relocation manifest of its materials, the output structure, the input names, the preparation descriptor, and, when one was stored, the linked executable under `linked`, whose entries, kernels, constants, and material records are the artifact's. The prepared Program is stored as the byte tensor `cache.program`.

Two keys address entries. A source key combines the identity of a Python source within the statically checked subset, which covers the functions it calls and the instance attributes a method reads, with the call signature and the layouts of supplied keys; a new process computes it without capture. A trace key combines the captured Program's digest with the bound material layouts, the call signature, the supplied key layouts, and the execution environment; a process that captures the same Program skips the pipeline. Statistics record the key used and, for a source outside the subset, the `CacheIneligible` reason.

A hit with a linked executable binds current materials by symbol, resolves resource requirements against the Backend workspace, and rebuilds the Runners from the stored entries without decoding or linking the Program. Every relocated material, including keys, tables, and closure plaintexts, is recorded as `caller` with the path by which the current source provides it, so a hit binds the current Tensors of the source; material contents never enter the cache. The Program is decoded when it is requested or when the linked executable does not apply, for example when the registry contains implementations defined outside FHElium. Malformed or mismatching files are misses.

Persistent caching applies to Python functions, methods, callable objects, and Programs compiled with the default pipeline, Backend, and workspace and without input-specification or resource overrides. Materials whose origins cannot be relocated prevent the entry. Clear a resident callable after mutating captured Python constants; `clear()` releases its execution state while disk files remain. Cache entries contain kernel images and schedules that the Runner executes, so use cache directories whose contents are as trusted as the installed package. [Linking, exec schedules, and the Runner](linking-and-runner.md) describes specialization preparation.

## Security and synchronization

Material inclusion is opt-in because arbitrary Tensor data can contain secret keys, plaintexts, RNG state, or other sensitive information. The file is unencrypted, and serialization does not detect sensitive contents. Stream-state snapshots require particular care: restoring the same random state into independent generators can repeat randomness.

The caller must synchronize with live writers before saving. Shared storage preservation is not an atomic snapshot of concurrent Tensor mutation. Safetensors storage and manifest validation establish format and extent constraints; they do not establish the provenance or safety of the Program's executable extensions. Treat imported IR, Backend implementations, and generated code according to their trust requirements.
