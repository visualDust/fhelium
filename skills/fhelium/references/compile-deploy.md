# Compile and deploy

Docs: [Choose compile boundaries](https://fhelium.550w.host/how-to/choose-compile-boundaries), [Compile a callable](https://fhelium.550w.host/how-to/compile-callable), [Serve compiled requests](https://fhelium.550w.host/how-to/serve-compiled-requests), [Deploy a compiled callable](https://fhelium.550w.host/how-to/deploy-compiled-callable), [Compiled execution](https://fhelium.550w.host/concepts/execution/compiled-execution).

## Compile boundaries

- `@fc.compile` on a function or method; calling it with new input conditions runs the Python body once under capture and prepares one *specialization*. Everything reached from the body (helpers, `linear.apply`, `polynomial.evaluate`) is one Program.
- Specialization conditions: Tensor shape/dtype/device; CKKS value depth, scale, domain, modulus basis, component count; Python `bool`/`int`/`float`/`complex`/`str`/`None` by value. A per-request scalar should enter as a Tensor or plaintext.
- Nested compiled functions are inlined with the outer options.
- Python loops unroll; use structured operations (`sum_plaintext_products`, `sum_rotations`, `rotate_many_by_steps`, `algorithms.linear`) for many terms.
- Methods compile per instance; attributes read become materials. Assigning attributes inside raises `CaptureError`.
- Constants reused across calls: `prepared = linear.prepare(...)` / `bootstrap.prepare(...)` before compiling, `prepared=prepared` inside, so encoding runs once. `linear.apply(engine, x, t, keys=keys)` without `prepared=` also captures. `fc.compile(prepared_bootstrap)` compiles a `PreparedBootstrap` directly. Methods such as `Plaintext.stack_batch` work inside but make the source cache-ineligible (`Unsupported method call: stack_batch`).
- Runtime decisions (branches on encrypted data, routing, key selection, Residency, I/O) stay outside; their results enter as arguments.
- Compiled results equal Eager bitwise unless rotation hoisting groups rotations (different rounding order, same error level); compare decrypted values.
- Tensor inputs must be on one CUDA device for Graphs; a host-resident input runs the specialization directly (`fallback_reason`: `Graph execution requires Tensor inputs on one CUDA device`).
- Larger functions: more fusion, fewer Graph launches, longer first call. Compiled functions on one device share one temporary arena; a result passed to the next compiled function is read in place.

## Options

```python
compiled = fc.compile(fn, graph="auto", cache=True, keys=(), on_miss="compile")
```

| Option | Values |
| --- | --- |
| `graph` | `"auto"` (CUDA: capture at the second execution of a specialization, keep each segment's Graph only if the median of 3 replays beats direct execution), `True` (always Graph), `False` (direct Runner) |
| `cache` | `True` (directory `FHELIUM_CACHE_DIR`, else `$XDG_CACHE_HOME/fhelium/compile` or `~/.cache/fhelium/compile`), `False`, or a path. Keys and materials are never cached |
| `keys` | Keys for materials the function does not reference directly (key objects or `fh.EvaluationKeySet`) |
| `on_miss` | `"compile"` or `"error"` (raise `SpecializationMiss` for unprepared conditions) |
| `pipeline`, `backend`, `workspace`, `resources`, `inputs`, `material_names` | Advanced: custom pass pipeline, operation backend, Program inputs |

Methods: `compiled.prepare(*example)` (prepare without running), `compiled.stats()`, `compiled.specializations`, `compiled.clear()` (release specializations), `compiled.refresh_materials()` (rebind changed captured objects), `compiled.with_backend(backend)`.

Cache: a new process reuses an entry without capture when the function source is in the statically checked subset (local assignments, `if`, `for`, `return`, Engine calls, Tensor methods, functions in the subset). `import`, `with`, `try`, attribute/subscript assignment, `**kwargs` expansion, `os`/`sys`/`time`/`random`/`numpy`/`secrets` attributes fall outside; such functions are captured each process and found by Program. `stats()["cache"]` shows `{'ineligible': reason}`.

## Export and load

```python
fc.export(compiled, "layer.fhelium", signatures=[(x,)], weights="raw", targets=(), overwrite=False)
fh.save_value(rotation_key, "key-rotation-1.fhelium")

# loading process
keys = [fh.load_value("key-rotation-1.fhelium", device="cuda")]
loaded = fc.load("layer.fhelium", keys=keys, engine=None, device=None, graph="auto")
y = loaded(x)
```

- Keys are never in the artifact; `keys` are matched by type and layout metadata (or a mapping from artifact key symbols). Compact keys are expanded on load.
- `weights="embedded"` stores encoded plaintexts (faster load, larger file); `targets=("sm_90",)` adds kernel images for other architectures.
- Export refuses schedules with Python calls, executables lowered per runtime input layout, and mixed CKKS configurations (`ExportError`).
- Loading requires the same FHElium native library and Triton version; mismatch raises `Artifact identity mismatch: <field>`.
- For a reproducible load, pass `engine=Engine(config, rng_seed=seed)`; C++ `artifact->seed_rounding(seed)` gives the same plaintexts.

C++ (`#include <fhelium/artifact.h>`, link `FHElium::torch`):

```cpp
auto artifact = c10::make_intrusive<fhelium::Artifact>("layer.fhelium", device);
artifact->bind_key(fhelium::io::load_value(key_path, device));
artifact->prepare();
auto sig = artifact->signatures().at(0);
auto runner = artifact->runner(sig.entry, std::nullopt, "auto", false);
auto out = runner->run({input.tensors.at("data")}, {}, {}).at(0);
```

Full program: `examples/cpp/load_artifact.cpp` (`load_artifact [--seed SEED] ARTIFACT OUTPUT INPUT KEY...`), example 15 writes matching files.

Collectives: `fhelium.distributed.all_reduce_ciphertext` / `reduce_ciphertext` inside a compiled function become one all-reduce (all-gather + P−1 additions in rank order, identical on every rank; `reduce_ciphertext` gives every rank the sum). `broadcast_ciphertext` is not captured: broadcast before the call. Artifacts with collectives: `fc.load(path, keys=..., groups={symbol: group})`; the default group `world` binds automatically. The artifact is rank-specific only if the function reads `torch.distributed.get_rank()`. C++: `artifact->bind_group(symbol, name_or_group)`. Shut groups down with `fhelium.distributed.shutdown(group)` (releases compiled schedules and artifacts bound to it; compiled callables rebind at their next call, artifacts need `fc.load` again); a direct `torch.distributed.destroy_process_group()` can hang on live Graphs. Functions closing over a `ProcessGroup` are cached by its c10d name: create groups in the same order in every process. Docs: [Choose a multi-GPU partition](https://fhelium.550w.host/how-to/choose-multi-gpu-partition).

Program files (IR + compile state without kernels): `fhelium.serialization.save_compilation` / `load_compilation`, [Save and relink a Program](https://fhelium.550w.host/how-to/save-and-relink-program).

## Serving loop

```python
layer = fc.compile(forward, on_miss="error")
for example in admitted_examples:
    layer.prepare(*example)
    layer(*example); layer(*example)        # second call captures CUDA Graphs

from fhelium.runtime import DoubleBuffer
with DoubleBuffer(host_requests[0], device="cuda:0") as buffers:
    for request, value in buffers.feed(host_requests):
        result = layer(value)
        send(request, result)
        del result                          # avoid a copy-out at the next call
```

- `feed` accepts an iterable or a `queue.Queue` ending with `None`; use pinned host memory so copies overlap with evaluation.
- Keys too large to keep resident: pass host keys and call `key.to(device)` inside the compiled function; the copy is a schedule transfer (pinned host storage allows asynchronous copies), but host-resident inputs disable Graphs for that specialization. `.to(device)` on a plaintext argument is plaintext preparation, not a streamed transfer; stream plaintext weights with `DoubleBuffer`.
- Residency (`fhelium.residency.ResidencyManager`): acquire leases between compiled calls and pass leased values as arguments; example 18.
- Device memory = keys + encoded plaintexts + shared arena and Graph pool (`fhelium.backend.runner.compiled_execution_bytes(device)`) + per-specialization outputs + buffers.
