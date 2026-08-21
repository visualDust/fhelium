# Choose a Residency control level

Use this procedure to select the narrowest placement mechanism that exposes the
control required by the workload.

## 1. Choose from the ownership requirement

| Need | Use | Application responsibility | Runtime responsibility |
| --- | --- | --- | --- |
| Move one application-owned value | `TensorResident.to(...)` | Own both values and all Python/CUDA lifetimes. | Perform one functional tensor movement. |
| Manage values by opaque handle while choosing every transition | `ResidencyManager` primitives | Choose `ensure`, `move`, `drop`, `discard`, and strict lease lifetimes. | Own materializations, enforce replica/lifetime rules, and account bytes. |
| Repeat a fixed stage with reviewed action order and headroom | Manual `ResidencyPlan` + manager scope | Specify reclaim, reservations, entry, exit, and the stage body. | Preflight and execute the ordered plan under manager authority. |
| State a working set but inspect policy choices before admission | `ResidencyController.decide` → inspect → `scope` | Define request endpoints, policy tiers, and acceptance of the decision. | Derive a state-bound plan; the manager validates and executes it. |
| Admit and borrow a working set in one context | `ResidencyController.use` | Define the request and supply all stream identities. | Decide, version-check, enter the scope, and acquire strict leases. |

Apply the following decision tree:

1. If no managed identity, budget, placement cache, or lease is required, use
   `TensorResident.to(...)`.
2. If every placement transition is already known at the call site, adopt the
   value into a manager and use direct primitives.
3. If several known transitions and reservations form one repeatable stage, use
   a manual plan.
4. If the working-set endpoints are known but reclaim choices depend on current
   state, use a request and controller. Choose the inspected `decide`/`scope`
   path when the application must log or approve the decision.
5. Use `controller.use(...)` only when the selected policy and complete request
   make the combined context sufficiently reviewable.

## 2. Keep each layer's input concrete

### Functional movement

```python
moved = value.to("cuda:0")
```

The application retains ordinary ownership of `value` and `moved`. This layer
has no manager handle, budget, reservation, cache policy, or borrowing rules.

### Direct managed operations

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

After `adopt`, retain the handle rather than a concrete alias. The application
chooses every destination and removal. `acquire` is already-ready-only and does
not invoke placement or policy.

### Manual stage

Use a `ResidencyPlan` when reclaim order, entry, exit, and reservation headroom
are known program structure. Inspect `manager.explain(plan)` before entering
`manager.scope(plan, transfer_streams=...)`. The plan says how to change
placement; the manager remains the only transition executor.

### Automatic admission

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

A request specifies required `(handle, location)` endpoints and reservation
headroom. A policy ranks legal reclaim candidates and supplies only configured
fallback tiers. A decision records policy evidence, a concrete plan, and the
manager state version against which it was derived. The controller owns none of
the materializations.

`controller.use(...)` combines the same decision, scope, and strict leases. Its
result is keyed by `ResidencyRequirement`; use `use.value(handle, at=...)` for
an endpoint lookup.

## 3. Assign stream responsibilities

| Mechanism | Transfer stream | Consumer stream |
| --- | --- | --- |
| `TensorResident.to` | Caller supplies `non_blocking` and owns readiness/lifetime synchronization. | Not tracked. |
| Direct `ensure` or `move` | One destination `stream=` argument. | Supplied separately to `acquire`; register every additional reader stream. |
| Plan or decision scope | `transfer_streams={location: stream}` selects per-destination copy streams. | Supplied to the strict lease inside the scope. |
| `controller.use` | `transfer_streams` mapping. | `consumer_streams` mapping for every requested CUDA location. |

A transfer stream governs placement completion. A consumer stream governs how
long an already-ready materialization remains protected after Python lease
release. Do not substitute one identity for the other.

## 4. Choose cache and identity lifetime

Direct operations and manual plan exit actions determine which
materializations remain cached. Successful controller admission retains its
requested endpoints; later pressure may reclaim eligible materializations
according to policy. The controller does not run background eviction and never
emits `DiscardValue`.

Use `drop(handle, location)` to remove one legal materialization while retaining
the logical value. Use `discard(handle)` only when the application intends to
end that managed identity. Close the manager after leases, holds, reservations,
and pending CUDA consumers have completed.

## 5. Keep execution buffers separate

A mutable fixed-address execution buffer and a manager-owned immutable logical
value have different ownership and lifetime rules. CUDA Graph and
execution-buffer bridging is deferred; no Residency control level assigns a
managed handle to such a buffer or transfers graph ownership. Select Residency
for logical-value placement, and treat fixed-address execution resources under
their own documented requirements.

## References

- [Residency lifetimes](../concepts/execution/residency-lifetimes.md)
- [Explicit Residency tutorial](../tutorial/explicit-residency.md)
- [Automatic Residency tutorial](../tutorial/automatic-residency.md)
- [Manual Example 13 source](https://github.com/VisualDust/fhelium/blob/main/examples/13_explicit_residency.py)
- [Automatic Example 14 source](https://github.com/VisualDust/fhelium/blob/main/examples/14_automatic_residency.py)
- [Stream resources with bounded memory](./stream-bounded-memory.md)
- [`TensorResident` API](../api/fhelium/core/tensor_resident.md#tensorresident)
- [`ResidencyManager` API](../api/fhelium/residency/manager.md#residencymanager)
- [`ResidencyController` API](../api/fhelium/residency/controller.md#residencycontroller)
