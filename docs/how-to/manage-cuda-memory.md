# Manage CUDA memory with buffers and Residency

Use this procedure when operation-ready plaintexts, evaluation keys, or activations do not fit comfortably in CUDA memory, or when a workload streams resources through a fixed device budget. FHElium provides two mechanisms that compose in application code: fixed-address reusable buffers in `fhelium.runtime`, which hold the current values of a repeated stage, and Residency in `fhelium.residency`, which places, caches, and accounts managed logical values across host and device locations.

## Prerequisites

Have a correct evaluator, Eager or compiled, and measured sizes for its operation-ready weights, evaluation keys, and temporaries.

## 1. Inventory live memory

Separate:

```text
device-local numerical tables
ciphertext activations
operation-ready plaintext weights
evaluation keys
compiled execution storage (arena and CUDA Graph pool)
temporaries
PyTorch allocated and reserved bytes
```

Measure live bytes by category. The largest Python collection is not necessarily the largest RNS materialization.

## 2. Classify resource lifetimes

Label the intended lifetime of every weight and key:

- deployment or model;
- request or user;
- evaluator phase;
- one tile or operation;
- one asynchronous copy;
- one active kernel read.

The lifetime determines whether a value stays resident on the device, is held on the host, or occupies a short device window selected by the application.

## 3. Choose the mechanism

| Need | Use | Application responsibility | FHElium responsibility |
| --- | --- | --- | --- |
| Move one application-owned value | `TensorResident.to(...)` | Own both values and their Python and CUDA lifetimes | Perform one functional tensor movement |
| Stream repeated values of one state through stable device storage | `ReusableValueBuffer`, `DoubleBuffer` | Supply values with matching signatures and consume them in order | Order each copy after the previous readers and expose the loaded value |
| Manage values by opaque handle while choosing every transition | `ResidencyManager` primitives | Choose `ensure`, `move`, `drop`, `discard`, and lease lifetimes | Own materializations, enforce replica and lifetime rules, and account bytes |
| Repeat a fixed stage with a reviewed action order and headroom | Manual `ResidencyPlan` with `manager.scope` | Specify reclaim, reservations, entry, exit, and the stage body | Preflight and execute the ordered plan |
| State a working set and inspect policy choices before admission | `ResidencyController.decide`, then `controller.scope` | Define request endpoints, policy tiers, and acceptance of the decision | Derive a state-versioned plan; the manager validates and executes it |
| Admit and borrow a working set in one context | `ResidencyController.use` | Define the request and supply every stream identity | Decide, check the state version, enter the scope, and acquire leases |

Apply this decision order:

1. Without a managed identity, budget, placement cache, or lease, use `TensorResident.to(...)`.
2. When the tiles of a stage repeat with the same value state and the application controls the order, use `DoubleBuffer` or `ReusableValueBuffer`.
3. When every placement transition is known at the call site, adopt the values into a manager and use direct primitives.
4. When several known transitions and reservations form one repeatable stage, use a manual plan.
5. When the working-set endpoints are known but reclaim choices depend on current state, use a request and controller; choose `decide` and `scope` when the application logs or approves the decision, and `controller.use(...)` when a separate review step is unnecessary.

A reusable buffer and a manager-owned logical value have separate ownership, accounting, and completion lifetimes. Application stage code may select the active window with Residency, lease those values, and copy them into one or two reusable buffers; FHElium does not treat a reusable buffer as a managed materialization.

## 4. Place Residency decisions between compiled calls

Residency reads and changes runtime state on every call, so its decisions belong in the application code that calls a compiled function. A compiled function that calls a `ResidencyManager`, `ResidencyController`, `ResidencyScope`, or `ResidencyLease` method during capture raises `CaptureError`:

```text
ResidencyManager.acquire cannot run inside a compiled computation; call it between compiled calls and pass the resulting values as arguments
```

Prepare the window first, then pass the leased values as arguments:

```python
@fc.compile
def layer(x, weights):
    ...

for stage in stages:
    with manager.acquire(
        stage.handles,
        at=cuda0,
        consumer_stream=torch.cuda.current_stream(),
    ) as resident:
        x = layer(x, [resident[h] for h in stage.handles])
```

Compiled functions register their execution storage, the shared per-device arena and the CUDA Graph pool, with Residency accounting. `ResidencyLocationSnapshot.compiled_execution_bytes` reports it at each CUDA location, and admission subtracts it from the remaining device budget while those functions are alive.

## 5. Stream tiles through reusable buffers

For real host-to-device overlap, prepare pinned host copies with `fhelium.runtime.pin_value_tree`; pageable memory may require staging and does not provide fully asynchronous transfer. Keep value state stable across host and device copies: a tile at a different depth, prime layout, or rotation step needs a different signature or buffer.

The double-buffer schedule is

```text
copy tile 0 -> buffer A
wait for A on the compute stream
compute tile 0 on A
concurrently copy tile 1 -> buffer B
compute tile 1 on B
reuse A after its readers complete
```

`DoubleBuffer(example, count=2, device=..., stream=...)` implements it. `put(value)` copies a value into the next buffer with `ReusableValueBuffer.load`, which runs the copy on the transfer stream after the work already issued on the current stream, including the readers of that buffer's previous value; it returns a `BufferedValue` whose `result()` makes the current stream wait for the copy. `feed(items)` keeps `count - 1` values in flight ahead of the one being consumed:

```python
with DoubleBuffer(host_tiles[0], device="cuda:0") as buffers:
    for tile, weights in buffers.feed(host_tiles):
        outputs.append(layer(x, weights))
```

A compiled function called with the loaded values replays one CUDA Graph for all of them. When using `ReusableValueBuffer` directly, use `CopyHandle.wait_on(compute_stream)` for device-side dependencies and record a consumer-complete event before the next write to the same buffer.

Start with one current tile and one next tile. Increase lookahead only when host-to-device transfer remains exposed and memory permits; a larger window can improve overlap and reduce residency transitions, and it increases pinned and device footprint, retains keys or weights that are not used soon, and raises transient peaks. Measure window size as a controlled variable.

## 6. Manage values with Residency

Direct managed operations:

```python
handle = manager.adopt(value, at=PAGEABLE_HOST)
del value
manager.ensure(handle, cuda0, stream=transfer_stream)
with manager.acquire(
    (handle,),
    at=cuda0,
    consumer_stream=compute_stream,
) as resident:
    run(resident[handle])
```

After `adopt`, retain the handle rather than a concrete alias. `acquire` requires an already-ready materialization and does not invoke placement or policy.

A manual stage uses a `ResidencyPlan` when reclaim order, entry, exit, and reservation headroom are known program structure. Inspect `manager.explain(plan)` before entering `manager.scope(plan, transfer_streams=...)`; the plan describes the placement changes and the manager executes them.

Automatic admission:

```python
decision = controller.decide(request)
log(
    decision.evictions,
    decision.explored_states,
    decision.explanation.predicted_peak_bytes,
)
with controller.scope(
    decision,
    transfer_streams={cuda0: transfer_stream},
):
    with manager.acquire(
        handles,
        at=cuda0,
        consumer_stream=compute_stream,
    ) as resident:
        run(resident)
```

A request specifies the required `(handle, location)` endpoints and reservation headroom. A policy ranks legal reclaim candidates within its configured fallback tiers. A decision records policy evidence, a plan, and the manager state version from which it was derived; `decide` executes no residency action, and scope entry checks `expected_state_version` before the first mutation. `controller.use(...)` combines decision, scope, and leases, and its result is keyed by `ResidencyRequirement`; `use.value(handle, at=...)` looks up one endpoint.

A typical retention plan is:

```text
model weights:       device hold if small and frequent, otherwise host hold
per-user keys:       request-lifetime pageable hold
current phase:       pinned or device preparation plus a short lease
active evaluator:    lease held until device consumers finish
```

A hold keeps a materialization from being reclaimed and exposes no values; an active read needs a lease. `ensure(...)` prepares a location without protecting an idle materialization. Use `drop(handle, location)` to remove one materialization while keeping the logical value, and `discard(handle)` to end a managed identity. Close the manager after leases, holds, reservations, and pending device consumers complete. Successful controller admission keeps its requested endpoints resident; later pressure may reclaim eligible materializations according to policy, and the controller runs no background eviction.

## 7. Assign stream responsibilities

| Mechanism | Transfer stream | Consumer stream |
| --- | --- | --- |
| `TensorResident.to` | Caller supplies `non_blocking` and synchronizes | Not tracked |
| `DoubleBuffer` | `stream=` argument, or a stream created for the device | The current stream when `result()` or `feed` yields the value |
| Direct `ensure` or `move` | One destination `stream=` argument | Supplied to `acquire`; register every additional reader stream |
| Plan or decision scope | `transfer_streams={location: stream}` | Supplied to the lease inside the scope |
| `controller.use` | `transfer_streams` mapping | `consumer_streams` mapping for every requested CUDA location |

A transfer stream governs placement completion; a consumer stream governs how long a ready materialization remains protected after the Python lease is released. Supply every device consumer stream when acquiring, or register additional streams on the lease, so release records completion events instead of synchronizing the device.

## 8. Account beyond the manager

A manager tracks managed storage charges and reservations at every observed location; a per-location budget adds strict admission, and an unbudgeted location reports the same accounting with `budget_bytes=None`. Track:

```text
logical payload and unique managed storage
manager used, reserved, peak-used, and peak-charged bytes
compiled execution bytes
PyTorch allocated and reserved bytes
nvidia-smi process and device usage
activation and temporary peaks
```

Use a scoped `MemoryReservation` for measured unmanaged output or workspace headroom. Offloading can reduce manager used bytes without reducing PyTorch reserved bytes or driver-reported usage; use the process-wide measurements to choose location budgets and leave measured headroom.

## 9. Validate synchronization and compare

Exercise different transfer and compute streams, large copies, alternating buffers, repeated values with different payloads and the same signature, and a signature mismatch before any copy. Incorrect synchronization can produce stale or mixed inputs without an exception. Compare decoded outputs with an all-resident reference.

Report all-resident latency and memory, latency and memory at each window size, host-to-device bytes and exposed transfer time, pinned host footprint, and peak manager charges and process memory separately. Select the window that satisfies the memory cap with acceptable transfer and latency overhead.

[Example 16](../tutorial/reusable-value-buffer.md) streams request tiles through `DoubleBuffer` with a compiled layer, [Example 18](../tutorial/manual-residency.md) places values with a manual plan and leases, and [Example 19](../tutorial/automatic-residency.md) uses automatic admission.

## Related documentation

- [Value signatures and buffers](../concepts/execution/signatures-and-buffers.md)
- [Residency lifetimes](../concepts/execution/residency-lifetimes.md)
- [Diagnose a Residency failure](diagnose-residency-failure.md)
- [`TensorResident` API](../api/fhelium/values/tensor_resident.md#tensorresident)
- [`ResidencyManager` API](../api/fhelium/residency/manager.md#residencymanager)
- [`ResidencyController` API](../api/fhelium/residency/controller.md#residencycontroller)
