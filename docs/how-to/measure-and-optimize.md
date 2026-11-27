# Measure and optimize a workload

Optimization starts from a correct evaluator and a small reproducible measurement: the latency of the first call, the latency of warmed calls, the device memory held by compiled execution and keys, and the output error, all measured on the path that will be deployed. This guide gives that measurement for a compiled callable, the statistics that explain it, and the mechanisms to test against each dominant cost.

## Prerequisites

Have an evaluator whose decoded output matches a clear-message reference within its error criterion, representative inputs, and the keys it needs. Record the CKKS preset, the input and output depth, the actual scale, the device with its full model name, power limit, and SM clock, and whether CUDA Graphs are enabled.

## 1. Take a before/after measurement

```python
import time
import torch

compiled = fc.compile(layer)
start = time.perf_counter()
y = compiled(x)
torch.cuda.synchronize()
first_ms = 1e3 * (time.perf_counter() - start)

for _ in range(5):
    del y
    y = compiled(x)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(20):
    del y
    y = compiled(x)
torch.cuda.synchronize()
steady_ms = 1e3 * (time.perf_counter() - start) / 20
```

The first call includes capture, the pass pipeline, linking, and kernel compilation, or a persistent cache load; the second call captures the CUDA Graphs; the warmed loop measures replay. Report the first call and the steady latency separately, together with the output error. Releasing the previous result with `del y` before the next call matches a pipeline that consumes each result; keeping results adds one device copy per kept result (step 3). For a change, repeat the same measurement with only that change applied.

## 2. Read the compile statistics

```python
stats = compiled.stats()
print(stats["prepare_seconds"], stats["graph_capture_seconds"])
print(stats["timing"]["compiled_steady"])
for record in stats["graphs"]:
    print(record["state"], record["fallback_reason"])
    for segment in record["segments"]:
        print(segment["choice"], segment["nodes"],
              segment["decision_direct_seconds"], segment["decision_replay_seconds"])
```

| Quantity | Meaning |
| --- | --- |
| `prepare_seconds` | Host time of capture, pipeline, binding, linking, cache load or store, and kernel warmup. |
| `graph_capture_seconds` | Host time spent capturing and instantiating Graphs. |
| `timing["compiled_steady"]` | Host dispatch and device completion time of calls after each specialization's first two. |
| `specializations` | One record per call signature; more records than expected show that some argument varies (see [Choose compile boundaries](choose-compile-boundaries.md)). |
| Segment `choice` | `"graph"` or `"direct"` after the `"auto"` decision, which compares `decision_direct_seconds` with `decision_replay_seconds`. |
| Segment `nodes` | Graph nodes of the latest capture, roughly the kernel launches the segment replays. |
| `overlap` | Transfers issued ahead of their original position (`moved_transfers`), asynchronous transfers per direction, collectives on communication streams, CPU-lane tasks with `task_seconds`, and the host time blocked joining them (`join_wait_seconds`). |

A specialization that stays on direct execution reports its reason in `fallback_reason`. A pipeline of several compiled functions shows one record per function and specialization.

## 3. Account for device memory

| Component | Measurement |
| --- | --- |
| Compiled execution | `fhelium.backend.runner.compiled_execution_bytes(device)`: the shared arena of temporaries and the CUDA Graph pool on that device. |
| Runner output regions and staged copies | `capture_stats` of each record in `stats()["graphs"]`: `output_bytes`, `arena_bytes`, `staged_copies`, `detached_outputs`, `variants`. |
| Keys and encoded plaintexts | `value.nbytes` of each key and prepared plaintext, which the specialization binds as materials. |
| Process | `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()` after the warmed loop. |

Compiled functions on one device share one arena, so its size follows the largest single function rather than the sum. A nonzero `detached_outputs` count means that the caller kept results across calls, which costs one copy per kept result.

## 4. Match the mechanism to the dominant cost

| Dominant cost | Mechanisms to test |
| --- | --- |
| First call | Persistent cache (`stats()["cache"]` shows whether the source key applies), fewer specializations, structured operations instead of Python loops. |
| Key switches | Rotation hoisting with `rotate_many_by_steps` or `sum_rotations`, baby-step/giant-step plans in `fhelium.algorithms.linear`, fewer distinct rotation steps. |
| Repeated relinearization | Late relinearization of summed products. |
| Repeated NTTs of a fixed operand | Prepared NTT-domain plaintexts; compressed plaintexts. |
| Launch and dispatch overhead | `graph=True` or `"auto"`, larger compiled functions with longer Graph segments. |
| NTT kernels | [Choose an NTT implementation](choose-ntt-implementation.md). |
| Device memory | Streaming with `DoubleBuffer`, Residency placement between compiled calls, fewer resident keys. |
| Independent requests | Homogeneous batching or data parallelism across GPUs. |

Model weights and evaluation keys usually dominate device memory and host-device traffic. A dense prepared plaintext holds $\text{rows}\times N$ residues per diagonal, where rows is the number of active primes at its depth, while a compressed plaintext of slot period $p$ holds $\text{rows}\times 2p$, which is smaller by the factor $\text{slots}/p$; diagonals are periodic when the packing replicates the input vector or token segments share weights, and `fhelium.algorithms.linear` prepares such diagonals compressed. Encode plaintexts at the depth that consumes them, since rows decrease with depth. Compact keys hold half the bytes of expanded keys. Weights streamed with `DoubleBuffer` one layer ahead approach $\max(T, C)$ for transfer time $T$ and compute time $C$; streaming hides the transfer when each streamed byte is reused by enough computation that the compute time per byte exceeds the transfer time per byte, roughly when the reuse factor exceeds the ratio of device-memory bandwidth to host-device bandwidth. [Prepare and evaluate compressed plaintexts](../tutorial/compressed-plaintext.md) and [Manage CUDA memory with buffers and Residency](manage-cuda-memory.md) describe the mechanisms.

Before implementing a mechanism, estimate what it removes, such as the number of key switches, NTTs, launches, or bytes moved, and what it adds, such as larger keys, temporaries, or Graph memory. A substantial latency gain can justify more resident memory; a gain of a few percent rarely justifies a large memory increase.

## 5. Change one variable

Hold the input, packing, preset, depth schedule, key strategy, warmup, run count, and synchronization fixed while changing one mechanism, and measure latency, memory, and error together. Several controls are non-monotonic, including the hoisting group size, the NTT grouping width, tile sizes, and the Residency lookahead; sweep a small range.

## 6. Check the invariants

After each change, check the output error, the depth and scale schedule, the active prime rows, the component count, and the rotation direction. An optimization keeps the declared rescale and relinearization placement or updates the computation to the new placement deliberately.

## Verify the outcome

Keep a change when the error criterion still holds and the steady latency or memory improves on representative inputs. Report the first-call time, steady latency, throughput, and memory as separate quantities with the configuration and device. A result that combines packing, hoisting, Graph replay, and several GPUs is a configuration result; report the component ablations before attributing it to one mechanism. [Benchmark methodology](/benchmarks/methodology) describes the published measurement procedure, and the [cost model](../concepts/performance/cost-model.md) gives the operation costs.
