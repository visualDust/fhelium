# Stream resources with bounded CUDA memory

Use this procedure when operation-ready plaintexts or evaluation keys do not fit
comfortably in CUDA memory. First choose whether you need fixed physical
buffers, managed value residency, or both.

## 1. Inventory live memory

Separate:

```text
engine plans and tables
ciphertext activations
operation-ready plaintext weights
evaluation keys
graph-private inputs/outputs
temporaries
PyTorch allocated vs reserved bytes
```

Measure live bytes by category where possible. Do not assume the largest Python
collection is the largest encoded/RNS materialization.

## 2. Classify resource lifetimes

For every weight/key, label its intended lifetime:

- deployment/model;
- request/user;
- evaluator phase;
- one tile/operation;
- one asynchronous copy;
- one active kernel read.

This determines whether a value deserves persistent CUDA residency, a host
hold, or a short lease/window selected by the application.

## 3. Choose the mechanism

### Use `ReusableValueBuffer` when

- tile structures and value states repeat;
- you control the transfer/compute schedule;
- a small number of stable CUDA addresses is sufficient;
- values already have a separate logical owner.

### Use `ResidencyManager` when

- managed values need opaque stable handles within one local manager;
- valid pageable, pinned, or indexed CUDA locations need accounting, with
  strict admission budgets on selected locations where appropriate;
- model/request/phase retention matters;
- the application can issue `ensure`, `move`, `drop`, and `discard`
  transitions or inspectable low-level plans.

### Add `ResidencyController` when

- the manager already owns the values and remains the sole state authority;
- the application can state `(handle, location)` working-set endpoints
  and reservation headroom more directly than a transition sequence;
- deterministic policy-selected reclaim should remain visible in an immutable
  `ResidencyDecision`; and
- admission should occur either through reviewed `decide` then `scope` calls or
  the combined `use` context.

### Compose Residency and reusable buffers in application stage code

Application stage/tile code may select the active logical window with a manual
`ResidencyPlan` or automatic `ResidencyRequest`, lease those managed values,
and copy them into one or two fixed CUDA buffers. The manager and
buffer retain separate ownership identities, accounting, and completion
lifetimes. FHElium does not treat a mutable reusable buffer as a managed
materialization and provides no automatic Residency/buffer ownership bridge.

## 4. Prepare host masters

For real host-to-device (H2D) overlap, prepare pinned host materializations where appropriate.
Pageable memory may require staging and should not be assumed to provide fully
asynchronous transfer.

Keep value state stable across host and device materializations. A tile at a
different level, prime layout, or rotation step must use a different signature
or buffer.

## 5. Build a double-buffer schedule

```text
copy tile 0 -> buffer A
wait A ready on compute stream
compute tile 0 on A
concurrently copy tile 1 -> buffer B
record compute completion for A
compute tile 1 on B
reuse A only after its reader-complete event
```

Use `CopyHandle.wait_on(compute_stream)` for device-side dependencies. Record a
consumer-complete event before allowing the next write to the same buffer.

## 6. Add logical holds and leases

A common application plan is:

```text
model weights:
  CUDA hold if small/frequent, otherwise host hold

per-user keys:
  request-lifetime pageable hold

current phase resources:
  pinned/CUDA preparation + short lease

active evaluator:
  lease remains until CUDA consumers finish
```

A hold does not replace an active lease and exposes no concrete values.
`ensure(...)` prepares a location but does not protect an idle materialization
in the same way as an evaluator lease. Supply every CUDA consumer stream when
acquiring or register additional streams on the lease; release then records
completion events rather than requiring a full-device synchronization.

Choose admission separately from the lease:

- use manager primitives or a manual `ResidencyPlan` when the application owns
  the transition order;
- use `controller.decide(request)` when the application owns the endpoint
  requirements but wants deterministic reclaim selection;
- inspect `decision.evictions`, `decision.explored_states`, and predicted peaks
  before entering `controller.scope(decision, ...)`; or
- use `controller.use(...)` when a separate review step is unnecessary.

`decide` executes no residency action, reservation, lease, or reconstruction
source. Scope entry checks the decision's `expected_state_version` before the
first mutation. `acquire` remains already-ready-only in both workflows.

## 7. Choose a lookahead window

Start with one current tile plus one next tile. Increase lookahead only if H2D
transfer is exposed and memory permits it. A larger window can:

- improve overlap;
- reduce residency transitions;
- increase pinned/CUDA footprint;
- retain keys or weights that are not used soon;
- amplify transient peaks.

Benchmark window size as a controlled variable.

## 8. Validate synchronization

Test with:

- different transfer and compute streams;
- slow/large copies;
- alternating buffers;
- repeated values with different payloads but the same signature;
- a forced mismatch before any copy;
- early cleanup only after completion is observed.

Wrong synchronization can produce stale or mixed evaluator inputs without an
immediate exception.

## 9. Account beyond the residency manager

A manager tracks managed backing-storage charges and reservations at
every observed location. A configured per-location budget adds strict
admission to those charges; an unbudgeted location reports the same accounting
with `budget_bytes=None`. Track:

```text
logical payload / unique managed storage
manager used / reserved / peak-used / peak-charged bytes
PyTorch allocated bytes
PyTorch reserved bytes
NVML / nvidia-smi process and device usage
CUDA Graph memory
activation and temporary peaks
```

Use a scoped `MemoryReservation` for measured unmanaged output or workspace
headroom. It consumes remaining budget where one is configured and remains a
visible reservation at an unbudgeted location. Offload can reduce manager used
bytes without reducing PyTorch reserved or NVML usage. Use those process-wide
measurements to select location budgets and additional unmanaged headroom.

Leave measured safety headroom.

## 10. Compare against all-resident and eager baselines

Report:

- all-resident CUDA latency/memory;
- bounded-window latency/memory;
- H2D bytes and exposed transfer time;
- window and tile size;
- pinned host footprint;
- correctness;
- synchronization rule.

The goal is often a memory cap with acceptable overhead, not the absolute
lowest single-request latency.

## Related documentation

- [Value signatures and buffers](../concepts/execution/signatures-and-buffers.md)
- [Residency lifetimes](../concepts/execution/residency-lifetimes.md)
- [Reusable-buffer tutorial](../tutorial/reusable-value-buffer.md)
- [Explicit Residency tutorial](../tutorial/explicit-residency.md)
- [Automatic Residency tutorial](../tutorial/automatic-residency.md)
- [Choose a Residency control level](./choose-residency-control-level.md)
- [Diagnose a Residency failure](./diagnose-residency-failure.md)
