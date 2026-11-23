# Compilation persistence and rebinding

Compilation persistence stores textual Program IR and optionally selected Tensor bindings in one versioned safetensors file. `fhelium.serialization.save_compilation`, `inspect_compilation`, and `load_compilation` implement the format in `serialization/compilation.py`. Restored Programs can reenter manual pipelines or callable preparation with new execution bindings.

## What is included?

The file contains Program text and a manifest describing selected Tensor objects, their byte storages, and the symbol-to-object mapping. `include_materials=False` is the default and saves no numerical bindings. `True` includes all current bindings; a collection of symbols selects a subset. Unselected or originally absent symbols remain external.

This serialization format excludes workspaces, pass reports, resident callable specializations, generated host closures, compiled GPU binaries, process groups, and live sampler objects. These objects have process-specific execution identity. A loaded Compilation begins with its restored Program and binding dictionary; callers supply configuration and execution preparation needed for the next stage.

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

## Callable reuse and persistent caches

Callable preparation checks three levels in order: a resident executable for matching `CallSignature` conditions, a persistent optimized Program for that input specialization, and a source-family structural template. `compile/_cache.py` implements source identity, instance persistence and material relocation; `compile/_structural_cache.py` stores guarded templates and their selection plans in process memory and optionally on disk.

`cache=True` selects `FHELIUM_CACHE_DIR`, or `$XDG_CACHE_HOME/fhelium/compile` with `~/.cache` as the default base. Structural entries live under that directory’s `structures` subdirectory. A path selects another Program cache directory. `cache=False` disables disk reads and writes for these Program caches while preserving resident specializations and process-memory template reuse.

An optimized-instance entry stores transformed Program text, callable interface information and a material relocation manifest. Identity combines supported Python source and configuration, the complete call signature, material layouts and execution-environment facts. A hit resolves current materials, checks represented layouts and aliases, supplies available tables, and links a fresh executable.

A source without a supported identity, such as a bound method or a function closing over application objects, is captured in each process. Its optimized Program entry is keyed by a digest of the captured Program together with bound material layouts, the call signature and the execution environment, so a process that captures the same Program skips the transformation pipeline.

Each optimized Program entry can have a linked-executable record, keyed by the entry key and the composition of registered implementations. The record contains the prepared host source, prepared implementations and invocations, the material symbols and resource requirements that linking bound, and the provider selections of generated materials. A hit binds current materials by symbol, resolves the requirements against the Backend workspace and restores the executable without decoding or linking the Program. The Program is decoded when it is requested; a restored fusion that meets an operand layout without a prepared plan links the Program again to obtain its region. Records are omitted when the registry contains implementations defined outside FHElium, when the callable supplies keys, or when a prepared implementation retains an object the record cannot represent. Plaintext-preparation executables have records keyed by a digest of the recorded preparation calls; the preparation Program is built from these calls only when it is linked or requested.

A structural entry stores high-level Program text, state provenance, metadata-read guards, material relocation descriptions and selection plans. Its source-family identity permits supported changes in CKKS depth, actual scale and prime-row extent while retaining fixed source, input-kind, layout and execution-context conditions. Instantiation clones the template and derives concrete types, attributes and material requirements from current input and material state. This reuses captured structure without rerunning Python source or Engine capture adapters. Template guards that fail lead to ordinary capture and another eligible variant.

Descriptive state facts share definitions through references, preserving their expanded `Known`, `InputField`, `Derived` and `Unknown` meanings while reducing repeated serialized metadata. No entry stores material Tensor contents or Graphs. Current CKKS material owners provide their current state; a raw key payload is rebound under its fixed declared key layout. Compatible fresh key data can therefore reuse the computation, with ciphertext/key relations remaining the application’s responsibility. Numerical tables are provisioned for the new instance’s requirements.

Source-identity entries require supported ordinary Python functions and capture-time state, using the default pipeline, Backend and workspace without input-specification, resource or key overrides. Bound methods, custom callable objects and unsupported source patterns use captured-Program entries; custom configurations continue through ordinary compilation. A template also needs state rules for its captured operations; uncovered operations can compile without structural reuse. Materials whose origins cannot be relocated prevent the corresponding cache entry. Statistics distinguish instance-cache results, structural hit or guard miss, and the memory or disk layer that supplied a template.

Clear a resident callable after mutating captured Python constants so preparation observes the new configuration. `clear()` releases that callable’s execution state while shared process templates and disk files retain independent lifetimes. Use cache directories whose contents are trusted as executable Program input. Linked-executable records are Python pickles that execute code when loaded, so the directory requires the same trust as the installed package. [Prepared host execution](prepared-host-execution.md) describes instance preparation and guarded selection plans.

## Generated Triton loop source

The Triton repeat-source cache stores the generated text, parameter declarations and loop counts produced by folding eligible repeated endpoint groups. An in-process source-signature cache and persistent entries reuse this result, so preparing the same body can skip source parsing, structural comparison and loop rewriting. Triton’s binary cache separately supplies the compiled GPU implementation.

Persistent entries use `triton-repeat/` under the existing environment-selected cache directory: `$FHELIUM_CACHE_DIR/triton-repeat/`, or `$XDG_CACHE_HOME/fhelium/compile/triton-repeat/` with `~/.cache` as the default base. Setting `FHELIUM_CACHE_DIR` to `off`, `0` or an empty string disables this disk cache while retaining in-process reuse. The callable’s `cache=` option controls Program persistence; repeat-source persistence uses this environment setting.

The source signature includes expanded source lines, parameter declarations and endpoint source ranges that have passed the NTT phase checks. Persistent identity also includes the projection implementation digest and Python version. Entries contain generated text and structural counts; Tensor data, material addresses and runtime row/stride values stay in current bindings. Changed source or projection code selects a new entry, and an unusable entry is regenerated. [Compilation state and pass composition](compilation-and-passes.md#runtime-loops-for-repeated-triton-groups) describes admission and the `triton_repeat` workspace control.

## Security and synchronization

Material inclusion is opt-in because arbitrary Tensor data can contain secret keys, plaintexts, RNG state, or other sensitive information. The file is unencrypted, and serialization does not detect sensitive contents. Stream-state snapshots require particular care: restoring the same random state into independent generators can repeat randomness.

The caller must synchronize with live writers before saving. Shared storage preservation is not an atomic snapshot of concurrent Tensor mutation. Safetensors storage and manifest validation establish format and extent constraints; they do not establish the provenance or safety of the Program's executable extensions. Treat imported IR, Backend implementations, and generated code according to their trust requirements.
