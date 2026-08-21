# Explicit residency plans and CUDA leases

**Example source:** [`examples/13_explicit_residency.py`](https://github.com/VisualDust/fhelium/blob/main/examples/13_explicit_residency.py)

This example places live FHElium plaintext, key, and ciphertext values under a
`ResidencyManager`. It performs residency transitions, dry-runs one named stage
plan with a CUDA reservation, runs real CKKS evaluation on an indexed CUDA
consumer stream, observes unbudgeted pageable accounting beside strict
pinned/CUDA budgets, and ends each managed logical value.

## Run the example

```bash
python examples/13_explicit_residency.py \
  --device cuda:0 \
  --preset slots8192-scale40-levels7-int64
```

CUDA is required. The workload rotates one encrypted message by one slot,
multiplies it by a prepared `0.5` plaintext, rescales, decrypts, and compares
with the cleartext result.

## 1. Start with live values

The example creates:

- an operation-ready `Plaintext` weight;
- a `RotationKey` for step `+1`;
- an encrypted request `Ciphertext`.

The stable `ResidencyManager` owns process-local managed values through
opaque handles.

```python
weight = engine.prepare_plaintext_for_multiplication(
    engine.encode(weight_message, level=0),
    modulus_basis="Q",
).cpu()
rotation_key = engine.create_rotation_key(1, engine.secret_key).cpu()
source = engine.encrypt_message(message)
```

## 2. Configure optional admission budgets

Locations are immutable identities. Host locations are canonical constants;
CUDA locations require a device index.

```python
device_location = cuda_location(engine.device)
residency = ResidencyManager(
    budgets={
        PINNED_HOST: pinned_capacity,
        device_location: cuda_budget,
    },
)
```

The example leaves pageable host unbudgeted while applying strict budgets to
pinned host and CUDA. The manager still accounts every pageable materialization
and lifetime. At the two budgeted locations, current materialization charges
plus active `MemoryReservation` charges must fit the application-supplied byte
limit.

`ResidencyManager()` with no arguments is the fully managed unbudgeted form.
Valid host and indexed CUDA locations become part of its accounting lazily
after their first successful managed use. A `budgets` entry adds a strict
admission limit only to that location; other valid local locations remain
unbudgeted. One
manager can therefore use several indexed CUDA locations in the current
process without predeclaring all of them.

The `budgets` mapping defines strict admission limits. Free-memory readings from
PyTorch or NVML remain observations used by the application when selecting
those values. A direct materialization or reservation that would exceed a
configured budget raises `ResidencyBudgetError`. Plan explanation instead
reports infeasibility, and scope entry raises `ResidencyPlanError` before
executing an infeasible plan. Application code then chooses its next `move`,
`drop`, workload-admission, or manager-budget decision as a named operation.

## 3. Adopt values and receive opaque handles

```python
weight_handle = residency.adopt(
    weight,
    at=PAGEABLE_HOST,
    replica_mode=ReplicaMode.REPLICABLE,
)
source_handle = residency.adopt(
    source,
    at=device_location,
    replica_mode=ReplicaMode.EXCLUSIVE,
)
del weight, source
```

Every `adopt` call returns a fresh unique `ResidencyHandle`. The handle is an
opaque process-local token that application code stores and passes back to the
manager as a complete object. Moving or replicating materializations preserves
that handle.

`adopt` transfers logical ownership under caller-enforced ownership rules.
Python cannot destroy other aliases, so callers must stop using the input value and must not allow raw
values obtained from a later lease to escape that lease.

The weight and rotation key are `REPLICABLE`, so they may have simultaneous
host and CUDA materializations. The request ciphertext is `EXCLUSIVE`, so it
has one steady materialization and changes location with `move`.

Adopted values have `Recoverability.MUST_PRESERVE`: their last materialization
cannot be dropped accidentally. The separate `register_source` API accepts a
synchronous `ResidencySource` only with
`Recoverability.RECONSTRUCTIBLE` and likewise returns a fresh opaque handle;
the returned handle identifies that registration to the manager.

## 4. Issue direct transitions

```python
residency.ensure(weight_handle, PINNED_HOST)
residency.move(
    source_handle,
    PINNED_HOST,
    from_location=device_location,
)
```

`ensure` creates a replica and retains existing materializations. `move`
creates the destination and removes the selected source. `drop` removes one
unprotected replica, while `discard` ends the managed value and removes all of
its unprotected state.

The application names each destination and removal.

## 5. Build and explain a stage plan

```python
plan = ResidencyPlan(
    name="inference/rotate-scale/tile-0",
    enter=(
        EnsureResident(weight_handle, device_location),
        EnsureResident(key_handle, device_location),
        MoveResident(
            source_handle,
            device_location,
            from_location=PINNED_HOST,
        ),
    ),
    exit=(
        MoveResident(
            source_handle,
            PINNED_HOST,
            from_location=device_location,
        ),
        DropResident(key_handle, device_location),
        DropResident(weight_handle, device_location),
    ),
    reservations=(
        MemoryReservation(
            device_location,
            workspace_bytes,
            label="rotate/multiply outputs and workspace",
        ),
    ),
)
explanation = residency.explain(plan)
```

A plan is ordered low-level intermediate representation (IR). `explain` dry-runs
reclaim, reservation, entry, and exit effects against current state, checks
budgets and ownership constraints, resolves sources, and predicts managed
storage peaks without loading a source or executing the plan.

The complete scope order is reclaim actions, reservation admission, entry
actions, the application body, exit actions, and reservation release. This
example has no reclaim prefix because its configured budget already admits the
stage. A reservation is accounted headroom for unmanaged expansion. In this
example the reservation reduces the remaining CUDA budget for the scope
lifetime; it does not allocate the rotation output, multiplication output, or
native workspace.

The string `inference/rotate-scale/tile-0` is a diagnostic plan name. Stages
and tiles are ordinary named, nestable program structure. Plan actions still
refer to the opaque local handles. This is where the application can control
FHE memory expansion: select only the current weights and keys, reserve
measured output/workspace headroom, and release the window at a known
completion point. `scope(..., transfer_streams={device_location:
transfer_stream})` can bind a copy stream per CUDA destination. A
prepared caller may also pass
`expected_state_version=snapshot.state_version`; scope entry rejects a stale
version before mutation.

## 6. Protect a CUDA consumer stream

```python
compute_stream = torch.cuda.Stream(device=engine.device)
scope = residency.scope(plan)
with scope:
    with residency.acquire(
        (source_handle, key_handle, weight_handle),
        at=device_location,
        consumer_stream=compute_stream,
    ) as resident:
        with torch.cuda.stream(compute_stream):
            rotated = engine.rotate_with_key(
                resident[source_handle],
                resident[key_handle],
            )
            output = engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(rotated),
                resident[weight_handle],
            )
            output = engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(output)
            )

    released_snapshot = residency.snapshot()
    compute_stream.synchronize()
```

`acquire` borrows already-resident values; it does not materialize missing
ones. The borrowed mapping checks that the lease remains active and must not be
retained as a source of raw aliases after release.

CUDA acquisition requires the initial `consumer_stream` argument. This is a
lifetime identity, not a performance hint: release may occur on another Python
thread, whose ambient current stream is unrelated to the consumer.

At lease release, the manager records an event on `compute_stream`. If kernels
are still reading the values, the manager retains pending protection until the
event completes. It does not synchronize the full device solely to close the
Python lease. The example synchronizes only the result-producing stream before
host decryption and before plan exit removes or moves the inputs.

A consumer that launches reads on more streams must call
`lease.add_consumer_stream(...)` before release. A `ResidencyHold` is different:
it retains materializations across a longer application lifetime but exposes no
values. Evaluation still requires a lease.

## 7. Choose manual or automatic admission

This example deliberately keeps placement actions and plan order under direct
application control. Use the separate
[automatic residency admission](./automatic-residency.md) workflow when the
application should state working-set endpoints and headroom while a
deterministic policy selects legal reclaim actions. Both workflows execute
through the same manager authority and strict leases; automation does not
change `ResidencyManager.acquire()` or introduce background movement.

## 8. Read the accounting layers correctly

`ResidencySnapshot` contains no tensors. For every materialization it reports:

- logical tensor payload bytes;
- actual unique backing-storage bytes and the fixed conservative charge;
- active use and hold counts;
- pending CUDA consumer-event count.

For each location it reports optional `budget_bytes`, optional
`remaining_budget_bytes`, used and reserved bytes, peak materialization charge
(`peak_used_bytes`), peak total charge including reservations
(`peak_charged_bytes`), and aggregate protection counts. The two budget fields
are `None` for pageable host in this example, while byte accounting remains
present. CUDA location snapshots and transition reports also sample process-wide
PyTorch allocator metrics where applicable. A transition report identifies the
CUDA device represented by its allocator sample.

The example prints manager accounting beside:

```python
torch.cuda.memory_allocated(engine.device)
torch.cuda.memory_reserved(engine.device)
```

These numbers answer different questions:

| Metric | Meaning |
| --- | --- |
| Logical payload | Sum of declared tensor elements; shared/viewed fields still count logically. |
| Managed storage | Conservative backing-storage charge admitted by the manager. |
| Optional residency budget | Application-selected admission limit for one location; `None` means unbudgeted. |
| PyTorch allocated | Live allocations known to the process-local caching allocator. |
| PyTorch reserved | Blocks retained by that allocator, including reusable free blocks. |
| `nvidia-smi` / NVML | Broader device/process usage, including contexts and allocations outside manager accounting. |

Offloading can lower manager `used_bytes` without immediately lowering PyTorch
reserved bytes or NVML usage. Use the process-wide measurements to set
appropriate application budgets and unmanaged headroom.

## 9. End values and close the manager

```python
residency.discard(source_handle)
residency.discard(key_handle)
residency.discard(weight_handle)
residency.close()
```

`discard` ends each managed value. `close` rejects active or
pending lifetimes unless the caller explicitly chooses its force escape hatch.

The evaluator output is not adopted in this example; it remains an ordinary
application-owned ciphertext.

::: details Complete runnable source
<<< @/../examples/13_explicit_residency.py
:::

## Related concepts and guides

- [Residency lifetimes](../concepts/execution/residency-lifetimes.md)
- [Ownership and runtime responsibilities](../concepts/architecture/ownership-and-responsibilities.md)
- [Stream resources with bounded memory](../how-to/stream-bounded-memory.md)
