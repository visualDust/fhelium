# Compile a callable

`fc.compile` turns a Python function, method, callable object, Program, or Compilation into a `CompiledCallable`. The first call with new input conditions captures the computation, runs the pass pipeline, links the result against the Backend, and executes it; later calls with the same conditions run the linked specialization directly. On CUDA, a specialization executes through CUDA Graphs of its capture-safe segments by default. [Choose compile boundaries](choose-compile-boundaries.md) explains what to put inside a compiled function; this page lists the options and the methods of the compiled callable.

## Prerequisites

Have a calculation that runs in Eager mode and representative inputs. For CKKS, prepare the configuration, ciphertexts, and keys as in the [basic CKKS workflow](../tutorial/basic-ckks-workflow.md). For a calculation written directly as IR, start with [Build a Program pipeline and write passes](build-program-pipeline.md); `fc.compile` accepts that Program as well.

## 1. Wrap the calculation

```python
import torch
from fhelium import compile as fc

@fc.compile
def affine(x: torch.Tensor, gain: float = 2.0) -> torch.Tensor:
    return x * gain + x

y = affine(torch.arange(8, dtype=torch.float64))
```

For encrypted inputs, wrap the Eager operations that express the CKKS transitions:

```python
@fc.compile
def encrypted_sum(left, right):
    return engine.add(left, right)

result = encrypted_sum(cx, cy)
```

Ordinary Tensor arguments keep their public numerical role; a ciphertext argument does not encrypt the other arguments. Keys, weights, and other fixed values read by the function become materials of the specialization. Create keys and run randomized encryption outside the compiled function.

`@fc.compile` uses default options, and `@fc.compile(graph=False, cache=False)` configures the decorator. `fc.compile(function, ...)` wraps an existing function; `compiled.reference` returns the original Python callable.

## 2. Options

| Option | Default | Effect |
| --- | --- | --- |
| `graph` | `"auto"` | CUDA Graph policy for CUDA specializations: `"auto"` keeps each segment's Graph only when its replays are faster than direct execution, `True` keeps every Graph, `False` runs every call directly. |
| `cache` | `True` | Persistent compile cache: `True` uses `FHELIUM_CACHE_DIR` or the user cache directory (`$XDG_CACHE_HOME/fhelium/compile`, else `~/.cache/fhelium/compile`), a path selects a directory, `False` keeps only in-process specializations. |
| `keys` | `()` | Key objects or an `EvaluationKeySet`; key layouts are selected from them and they are bound into each specialization. |
| `on_miss` | `"compile"` | `"error"` raises `SpecializationMiss` for a call whose conditions match no prepared specialization. |
| `pipeline` | `None` | A `Pipeline`, or a function from `CallSignature` to `Pipeline`; `None` selects `default_lower_and_fuse_pipeline`. |
| `backend` | `None` | An `OperationBackend`; `None` selects the standard operation registry. |
| `resources` | `()` | `CkksDeviceResources`, `RnsContext`, or `NttContext` objects that provide numerical tables. |
| `workspace` | `None` | A `CompileWorkspace` with capture and pass inputs. |
| `inputs` | `None` | Input-role declarations that select the FX frontend for captured PyTorch code. |
| `material_names` | `None` | Stable symbols for fixed Tensor or value objects found during capture. |

## 3. Prepare specializations

```python
prepared = affine.prepare(x, gain=2.0)
print(prepared.compilation.program.to_text())
print(len(affine.specializations))
```

`prepare` performs the capture, pass, link, and kernel-warmup work of a missing call without executing it, and returns the `Specialization`. A call with new Tensor contents and the same shape, dtype, device, and CKKS state reuses the specialization. Python scalars such as `gain` are specialized by value, so a new value prepares another specialization.

For a service that must not compile during requests, construct the callable with `on_miss="error"` and prepare the admitted inputs at startup:

```python
strict = fc.compile(affine.reference, on_miss="error")
strict.prepare(x, gain=2.0)
strict(x, gain=2.0)  # another gain raises SpecializationMiss
```

## 4. Reuse across processes

With `cache` enabled, a prepared Program and its linked schedule are stored under a key computed from the function source when the source lies within the statically checked subset of Python, and otherwise from the captured Program. A source key lets a new process skip capture; a trace key requires capture and skips the pass pipeline and linking. Both keys include the input conditions, the CKKS configuration, the material layouts, and the execution environment. Material Tensor contents never enter the cache: a cached entry binds the current process's keys and weights. Persistence applies to Python functions and Programs compiled with the default pipeline, Backend, and workspace and without `inputs` or `resources`; supplied `keys` are matched to each cached entry's recorded key descriptions. `stats()["cache"]` reports which key was used, and the reason when a source falls back to its captured Program.

## 5. CUDA Graphs and results

A CUDA specialization runs its schedule directly on its first execution, captures its Graph segments on the second, and replays them afterwards. `stats()["graphs"]` holds one record per specialization; each record lists its segments with their `choice` (`"graph"`, `"direct"`, or `"measuring"` before the `"auto"` decision) and gives a `fallback_reason` when the specialization runs without Graphs. Host-resident Tensor inputs (`Graph execution requires Tensor inputs on one CUDA device`), inputs that require gradients, and a failure before the first capture fall back to direct execution.

A Graph result is returned in the specialization's output storage without a copy. A result the caller still references when the next execution of the same specialization would overwrite that storage is first moved to newly allocated memory, so its data pointer changes; a result released before the next call costs nothing. A result passed to another compiled callable on the same device is read in place. [Use CUDA Graphs](use-cuda-graphs.md) shows the statistics in use.

## 6. Inspect and clear

```python
stats = affine.stats()
print(stats["prepare_seconds"], stats["executions"], stats["resident_cache_hits"])
for variant in affine.specializations:
    for report in variant.compilation.reports:
        print(report.name, report.seconds)
```

| Key | Content |
| --- | --- |
| `specializations` | One record per call signature with its preparation-stage host durations. |
| `resident_cache_hits` | Calls served by an in-process specialization. |
| `prepare_seconds`, `graph_capture_seconds` | Total preparation and Graph capture host time. |
| `executions`, `execute_seconds`, `timing` | Execution counts and timings; steady-state timing excludes each specialization's first two calls. |
| `graphs` | Graph records and segment choices. |
| `cache` | Persistent cache use and fallback reasons. |
| `overlap` | Summed `ProgramExecutable.overlap_stats()` of the specializations: `moved_transfers`, `h2d_transfers`, `d2h_transfers`, `stream_collectives`, and `planned_tasks` from planning, and the accumulated `tasks_run`, `task_seconds`, `join_wait_seconds`, and `event_waits`. |

`clear()` releases the specializations, their Graphs, and their pools; call it after changing Python constants that the function reads, so the next call captures their new values. Persistent cache entries remain. `with_backend(backend)` returns a callable that relinks the current specializations against another Backend.

## 7. Select a pipeline or Backend

```python
from fhelium.backend import OperationBackend

backend = OperationBackend()
recipe = fc.default_lower_and_fuse_pipeline(backend)
controlled = fc.compile(affine.reference, backend=backend, pipeline=recipe)
```

A supplied `Pipeline` replaces the whole recipe. Inspect `recipe.names` and compose passes with `before`, `after`, `replace`, or `then`; [Build a Program pipeline and write passes](build-program-pipeline.md) describes pass composition. The default recipe preserves the rescale and relinearization transitions written in the source.

## 8. Plaintext preparation

A function that encodes plaintexts from Tensors is split into a preparation Program, which packs and encodes the weights, and the ciphertext Program, which reads the encoded plaintexts as materials. The preparation Program runs once per specialization and runs again when a source Tensor's PyTorch version counter changes, rebinding the materials without repeating lowering, implementation selection, or kernel preparation. `refresh_materials(*args, **kwargs)` forces preparation after changes that version counters do not record.

The workspace options `binding_time_split` (default `True`), `merge_plaintext_encodings` (default `True`), and `plaintext_preparation_mode` (default `"resident"`) control this split. `merge_plaintext_encodings` shares one encoding, including its rounding, among uses with the same source, packing, and encoding parameters. `plaintext_preparation_mode="per_call"` runs preparation on every call and releases the encoded materials afterwards; this mode runs without CUDA Graphs.

## Verify the outcome

Compare public results with the original function and encrypted results with a clear-message oracle. Inspect `stats()` to confirm that repeated calls hit a resident specialization and that the Graph segments have the expected choices. [Diagnose Compile failures](diagnose-compile-failures.md) lists capture, specialization, and export errors.
