# Diagnose Compile failures

A compiled callable passes through capture, the pass pipeline, material binding, linking, and execution; an exported callable adds export and load. Each stage raises its own error type or records its outcome in `compiled.stats()`. Identify the stage before changing the computation, its CKKS state, or the supplied keys.

## Prerequisites

Keep the full exception, the source of the compiled function, and representative input metadata (shape, device, depth, scale, domain). Record material symbols and descriptions without printing secret payloads. For a manual Program, keep the last successful Compilation.

## 1. Locate the stage

| Stage | Evidence | First action |
| --- | --- | --- |
| Capture | `fhelium.compile.CaptureError` | Rewrite the reported construct (step 2). |
| Pass pipeline | Pass name and diagnostics in the traceback or in `compilation.reports` | Inspect the facts the pass requires (step 3). |
| Material binding and linking | Unresolved material symbols, unsupported operation or implementation, missing resource | Supply keys or resources, or compare the operations with the Backend registry (step 4). |
| Call | `SpecializationMiss` | Prepare the admitted conditions (step 5). |
| Export or load | `ExportError`, key-matching, identity, or rank errors | Adjust the exported callable or the loading environment (step 6). |
| Persistent cache | `stats()["cache"]` reason | The call still compiles; adjust the source to restore cross-process reuse (step 7). |
| Graph execution | `fallback_reason` in `stats()["graphs"]` | The call runs directly; remove the reported condition when Graphs are required. |

Preparation errors propagate from the call or from `compiled.prepare(*args)`, so `prepare` reproduces a failure without executing.

## 2. Resolve capture errors

Capture executes the Python body with symbolic values and accepts FHElium Engine operations, supported Tensor operations, and Python control flow that does not depend on encrypted or Tensor data. The most common errors are:

| Message | Cause and repair |
| --- | --- |
| `Captured computation cannot branch on Tensor or ciphertext data` | An `if`, `while`, or comparison reads a value computed from an argument. Compute the decision outside the compiled function and pass it as a Python scalar, which selects a specialization, or express the selection arithmetically. |
| `Captured computation cannot mutate bound object state` / `bound mapping state` | The function assigns an attribute or a dictionary entry of an object it reads. Return the value instead, or set it before the call. |
| `<Type>.<method> cannot run inside a compiled computation; call it between compiled calls and pass the resulting values as arguments` | A Residency manager, scope, controller, or lease is used inside the function. Make the Residency decision before the call and pass the leased values as arguments. |
| `Operands disagree on <field>; capture does not insert alignment` | Two operands differ in depth, scale, domain, or another state field. Align them with Eager operations in the source, as Eager requires. |
| `<role> requires an evaluation key` | A rotation, relinearization, or conjugation has no key in the captured computation. Pass the key, read it from a captured object, or supply it through `keys=`. |
| `Prepare slot plaintexts outside the compiled computation` / `Encode slot plaintexts before ciphertext consumption` | A plaintext encoding form that capture does not split into a preparation Program. Encode the plaintext before the call, or use `prepare_compressed_plaintext` inside the function. |
| `Unsupported Engine method <name>` / `Engine.<name> is outside the callable frontend` | The method has no captured form. Express it with supported operations or call it between compiled calls. |
| `Captured functions cannot declare *args or **kwargs` | Declare named parameters. |

## 3. Inspect the Program and reports

```python
compilation = compiled.prepare(x).compilation
print(compilation.program.to_text())
for report in compilation.reports:
    print(report.name, report.seconds, report.stats, report.diagnostics)
print(compilation.program.material_descriptions)
```

For a manual pipeline, run the passes up to the failing one and inspect the Compilation at the last successful stage; [Inspect Program transformations](visualize-mixed-level-ir.md) renders the producer of a mismatched value. Unknown shape, depth, scale, device, or representation can be valid in an intermediate Program; determine which consumer requires the missing fact, then supply a known entry state, a representative input, or the assignment pass.

## 4. Resolve materials and implementations

For a manual Compilation, `fc.prepare_material_bindings(compilation, resources=resources, keys=keys)` fills missing bindings from the supplied providers and keys and returns the symbols it could not resolve. It never overwrites an existing binding and never generates a key. When several keys match a description, bind the intended key under the Program symbol. A compiled callable binds the keys its source reads and those passed with `keys=`.

For an unsupported operation, inventory `backend.registry.implementations` and match the operation class and requested implementation name. `backend.diagnostics(operation)` returns diagnostic strings for implementation or resource resolution failures on one operation. Device-specific failures usually come from the installed native backend or from an operand or key on another device; selecting a Backend does not move them.

## 5. Handle specialization misses

A call prepares a new specialization when an argument's shape, dtype, device, CKKS state, or a Python scalar value differs from every prepared specialization. With `on_miss="error"`, such a call raises `SpecializationMiss`; prepare each admitted condition during setup. A loaded artifact raises `SpecializationMiss("No specialization stored in the artifact matches these arguments")` for a call outside the exported signatures; export another example call for it. Changing Tensor contents with the same conditions never misses. After changing Python constants that the function reads, call `compiled.clear()`.

## 6. Resolve export and load errors

| Error | Cause and repair |
| --- | --- |
| `ExportError: Schedules with Python calls cannot be exported: ...` | A linked schedule runs a Python implementation. Replace the listed operation with one that lowers to native operations, or keep that callable in-process. |
| `ExportError: the callable uses Engines of different CKKS configurations` | Use one configuration per exported callable. |
| `ExportError: the executable is lowered per input layout` | The executable lowers its schedule at each call because its input layouts were not declared at linking; such an executable runs only in-process. |
| `ValueError: No supplied key matches artifact keys <symbol> (<role>)` | The loading process lacks a key with the recorded role and layout. Supply the key with the same prime and digit layout and, for rotations, the same step. |
| `Artifact identity mismatch: <field> ...` | The loading library differs from the exporting library in FHElium version, operator schema hash, Torch ABI, or CUDA runtime. Export again with the deployed build. |
| `Unsupported artifact schema version in <path>` | The file comes from another FHElium version. Export again. |
| `the artifact was captured for process rank <r>` | The source read the process rank during capture; load the file on that rank, or remove the rank-dependent branch from the source. |

## 7. Restore cross-process cache reuse

`stats()["cache"]` reports `{'ineligible': <reason>}` when the function source falls outside the statically checked subset; the callable then uses a key computed from its captured Program, which requires capture in each process. Typical reasons are `Unsupported Python cache syntax: <node>` (for example `ImportFrom`, `With`, `Try`), `Unsupported method call: <name>`, `Unsupported external callable`, `External-state module: <module>`, and `Dynamic call targets are not cached`. The reason refers to the first construct found in the function or in a function it calls; [Choose compile boundaries](choose-compile-boundaries.md) lists the subset.

## Verify the repair

Repeat the failing call or export with the same input conditions, inspect `stats()` and the reports, and compare decoded outputs with the clear-message reference. Use [Diagnose a value-state mismatch](diagnose-value-state-mismatch.md) for CKKS state errors in the source computation.
