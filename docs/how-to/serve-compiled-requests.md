# Serve compiled requests

A service that evaluates encrypted requests with compiled callables prepares or loads every admitted specialization before it accepts requests, passes per-request data as arguments, streams request data to the device while the previous request is evaluated, and releases each result before the next call. This guide lists those decisions; [Example 16](../tutorial/reusable-value-buffer.md) runs a request loop with `DoubleBuffer`, and [Example 18](../tutorial/manual-residency.md) places the operands of a compiled stage with Residency.

## Prerequisites

Have a compiled callable that produces the correct decoded result for representative requests, and the evaluation keys it needs on the serving device.

## 1. Fix the admitted signatures at startup

Construct the callable with `on_miss="error"` and call `prepare` for every admitted combination of argument shapes, CKKS states, and Python scalar values:

```python
layer = fc.compile(forward, on_miss="error")
for example in admitted_examples:
    layer.prepare(*example)
```

A request whose conditions match no prepared specialization then raises `SpecializationMiss` instead of compiling during serving. Alternatively, export the callable with those examples as `signatures` and serve the artifact with `fc.load`, which needs no capture, linking, or Triton compilation in the serving process ([Deploy a compiled callable](deploy-compiled-callable.md)). Run one call per specialization after preparation as well: the second call of a CUDA specialization captures its Graphs, so two warmup calls move capture out of the first requests.

## 2. Pass per-request data as arguments

Values that change per request, such as the request ciphertext, per-user plaintext weights, or keys of the requesting party, are arguments of the compiled callable. Values captured from closures or instance attributes become materials of the specialization; changing their contents in place is visible to execution, while binding another object requires a new specialization. Python scalars select specializations by value, so a per-request scalar enters as a Tensor or plaintext.

## 3. Stream request data to the device

`fhelium.runtime.DoubleBuffer` keeps fixed device storage for `count` copies of a value tree and fills them in turn on a transfer stream:

```python
from fhelium.runtime import DoubleBuffer

buffers = DoubleBuffer(host_requests[0], device="cuda:0")
for request, value in buffers.feed(host_requests):
    result = layer(value)
    send(request, result)
    del result
```

`feed` keeps `count - 1` copies in flight ahead of the value being evaluated, and each copy starts after the work already issued on the current stream, including the readers of that buffer's previous value. A loaded value stays valid until its buffer is filled again, `count` calls of `put` later. `feed` also reads a `queue.Queue` until it returns `None`. Pinned host memory lets the copies overlap with evaluation.

A ciphertext or evaluation-key argument (`KeySwitchKey`, `RotationKey`, `RelinearizationKey`) held on the host can instead be moved inside the compiled function with `.to(device)`. The move is a transfer of the schedule, so keys that do not fit on the device at once can be streamed per stage; pinned host storage (`pin_memory()`) lets a schedule that issues its transfers on copy streams overlap them with computation ([Compiled execution](../concepts/execution/compiled-execution.md#transfers-streams-and-the-cpu-lane)). A specialization with host-resident Tensor inputs runs without CUDA Graphs. `.to(device)` on a plaintext argument is part of plaintext preparation, which runs before the schedule whenever the argument changes, so streamed plaintext weights go through `DoubleBuffer`.

## 4. Release results before the next call

A CUDA Graph result is returned in the specialization's output storage. If the caller still references the previous result when the next call of the same specialization would overwrite that storage, the result is moved to fresh memory first, which costs one device copy. Consume or copy each result to the host, then drop the reference before the next call. A rebinding such as `out = layer(x)` inside a loop keeps the previous result referenced while the new call runs; `del out` after use avoids the move. A result passed to another compiled callable on the same device is read in place.

## 5. Keep runtime decisions outside compiled calls

Routing, admission, batching of independent requests, key selection, and Residency placement run between compiled calls. Capture rejects branches on encrypted data and calls to Residency managers, scopes, controllers, and leases inside a compiled function. Acquire a Residency lease for the values of a stage, call the compiled stage with the leased values, and release the lease after the stage's stream has consumed them.

## 6. Account for memory

Device memory of a serving process consists of the resident keys and encoded plaintexts, the shared arena and Graph pool of compiled callables (`fhelium.backend.runner.compiled_execution_bytes(device)`), the output storage of each specialization, and the `DoubleBuffer` storage. A Residency budget on the device includes the compiled-execution bytes. [Manage CUDA memory with buffers and Residency](manage-cuda-memory.md) describes budgets and placement.

## Verify the outcome

Serve a sequence of distinct requests, decrypt each result on the client side, and compare it with the clear-message reference. Check that `stats()["specializations"]` contains only the prepared signatures, that `detached_outputs` in each record's `capture_stats` stays constant during the loop, and that steady latency matches the measurement from [Measure and optimize a workload](measure-and-optimize.md).
