# Performance

Docs: [Measure and optimize a workload](https://fhelium.550w.host/how-to/measure-and-optimize), [Use CUDA Graphs](https://fhelium.550w.host/how-to/use-cuda-graphs), [Choose an NTT implementation](https://fhelium.550w.host/how-to/choose-ntt-implementation), [Manage CUDA memory with buffers and Residency](https://fhelium.550w.host/how-to/manage-cuda-memory), [Cost model](https://fhelium.550w.host/concepts/performance/cost-model).

## Measurement recipe

Start only from a numerically verified evaluator (decrypted output compared with a clear reference).

```python
start = time.perf_counter(); y = compiled(x); torch.cuda.synchronize()
first_ms = 1e3 * (time.perf_counter() - start)          # capture + passes + linking + kernel compile, or cache load
for _ in range(5):
    del y; y = compiled(x)                               # 2nd call captures Graphs
torch.cuda.synchronize(); start = time.perf_counter()
for _ in range(20):
    del y; y = compiled(x)
torch.cuda.synchronize()
steady_ms = 1e3 * (time.perf_counter() - start) / 20
```

Report together: first call, steady latency, output error, `torch.cuda.max_memory_allocated()`/`max_memory_reserved()`, preset, input→output depth, actual scale, residue dtype, Graph on/off, full GPU model name (with power limit when variants exist). First-call time is a compile cost, not execution latency. For a before/after comparison change one variable and keep input, preset, warmup, and run count fixed.

## Reading `compiled.stats()`

| Key | Meaning |
| --- | --- |
| `specializations` | One record per call signature; more than expected means an argument varies |
| `prepare_seconds` | Capture, pipeline, binding, linking, cache load/store, kernel warmup |
| `graph_capture_seconds` | Graph capture and instantiation |
| `executions`, `execute_seconds` | Call counts and time |
| `timing["compiled"]`, `timing["compiled_steady"]` | All calls; calls after each specialization's first two |
| `resident_cache_hits` | In-process specialization reuse |
| `cache` | Persistent cache key in use, or `{'ineligible': reason}` |
| `graphs` | One record per specialization: `state`, `fallback_reason`, `replays`, `segments`, `capture_stats` |
| `overlap` | `moved_transfers`, `h2d_transfers`, `d2h_transfers`, `stream_collectives`, `planned_tasks`, `tasks_run`, `task_seconds`, `join_wait_seconds`, `event_waits` (transfers on copy streams and CPU-lane tasks) |

Segment fields: `choice` (`"graph"`, `"direct"`, `"measuring"`), `nodes` (≈ kernel launches), `decision_direct_seconds`, `decision_replay_seconds`. `capture_stats`: `output_bytes`, `arena_bytes`, `staged_copies`, `detached_outputs` (results kept across calls, one copy each), `variants`.

## Memory

- `fhelium.backend.runner.compiled_execution_bytes(device)`: shared arena of temporaries plus Graph pool for all compiled functions on the device; sized by the largest function, not the sum.
- Keys usually dominate: each rotation key is large (grows with N, Q primes, and digits). Count distinct rotation steps; hoisting and BSGS plans reduce them.
- Residency budgets on a device subtract compiled-execution bytes.
- Transfer overlap: copies are issued early on copy streams and may use up to one extra largest-transfer of arena memory beyond the schedule without moved transfers.

## Weights and keys: bytes, residency, streaming

Encrypted model inference is usually bounded by the bytes of plaintext weights and evaluation keys, not by the ciphertexts.

- Plaintext bytes. A dense prepared plaintext holds `rows × N × itemsize` bytes per diagonal (`rows` = active primes at its depth; itemsize 4 for int32 residues). A `CompressedPlaintext` with slot period `p` holds `rows × 2p × itemsize`, which is `slots / p` times smaller: a 768-wide matvec packed in 32768 slots shrinks about 32× (period 1024), a 256-wide one 128×.
- When diagonals are periodic. A diagonal is periodic when the packing replicates the input vector across the slots, or when token segments packed in one ciphertext share the same weights. Encode such diagonals from one period with `engine.prepare_compressed_plaintext(period_values, depth=d)`. `fhelium.algorithms.linear.prepare` already does this: each diagonal whose smallest power-of-two period is below the slot count is prepared as a `CompressedPlaintext`.
- Depth. `rows` decreases with depth, so encode a plaintext at the depth of the ciphertext that consumes it (`depth=x.depth`), not at depth 0.
- Keys. A rotation key holds `digits × (Q+P rows) × N × itemsize` per component. Expanded keys (the `Engine(expand_keys=True)` default) store both components; compact keys store half the bytes, and the key switch generates the second component on the GPU. Count distinct rotation steps first; hoisting and BSGS plans reduce them.
- Streaming weights. Stream layer weights from pinned host memory with `fhelium.runtime.DoubleBuffer`: one buffer ahead brings a layer close to `max(transfer, compute)`. A deeper window (`count > 2`) adds memory and gains nothing unless transfer-bound and compute-bound layers alternate. `DoubleBuffer` moves values in the form the code wrote them, so compact keys stay compact and expanded keys stay expanded.
- Streaming keys. Moving keys inside the compiled function (`key.to(device)`) is a schedule transfer; such specializations run without CUDA Graphs (host inputs). When transfer-bound, compact keys halve the transferred bytes and are the largest single gain. When keys of a hoisted rotation are needed together at the start of a request, prefetch them during the previous request rather than one key ahead.
- Sizing rule. Transfer hides behind compute when the compute time per weight byte exceeds the transfer time per byte, that is, when the reuse factor of each streamed byte (ciphertexts or requests that read it per transfer) exceeds about `HBM bandwidth / PCIe bandwidth`. Batching more ciphertexts per streamed layer raises the reuse factor.
- Export. With `fc.export(..., weights="raw")` (default) the artifact stores cleartext weights and encodes them at load; `weights="embedded"` stores encoded plaintexts, so the file carries the byte sizes above.

## Mechanisms by dominant cost

| Cost | Try |
| --- | --- |
| First call | Persistent cache (check `stats()["cache"]`), fewer specializations, structured ops instead of loops |
| Key switches | `rotate_many_by_steps`/`sum_rotations` hoisting, `linear.plan` BSGS, fewer distinct steps |
| Relinearization | Late relinearization of summed products (relinearize once after `sum`) |
| NTTs of fixed operands | Prepared NTT-domain plaintexts, compressed plaintexts |
| Launch overhead | `graph="auto"`/`True`, larger compiled functions; keep Tensor inputs on the device (host inputs disable Graphs) |
| NTT kernels | `Engine(ntt_backend=...)` among `fh.compatible_ntt_backends(logN)` |
| Memory | `DoubleBuffer` streaming, Residency, fewer resident keys |
| Independent requests | Homogeneous batch along a leading dimension, multi-GPU data parallelism |

Preset effects: a smaller ring (fewer slots) and fewer Q primes cut every operation's cost roughly linearly in primes and N log N in ring size; pick the smallest preset that fits slots and depth. Built-in presets use int32 residues; custom chains with primes ≥ 2^30 switch to int64 and cost more.
