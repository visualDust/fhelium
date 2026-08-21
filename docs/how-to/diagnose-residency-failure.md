# Diagnose a Residency failure

Use this ordered procedure before changing a budget, fallback tier, or placement
schedule. Preserve the first exception and capture manager evidence before
cleanup changes the state.

## 1. Capture one atomic state view

```python
snapshot = manager.snapshot()
trace = manager.trace()
```

Record `snapshot.state_version`, the manager id, every relevant handle, and the
exception's structured fields. A snapshot is tensor-free and reports, for each
materialization, location, charged bytes, active uses, holds, and pending CUDA
events. Location records report current and peak charges, active reservations,
and optional strict budgets. The bounded trace is supplementary: it may be
disabled or may have overwritten older transitions.

Do not infer manager state from `torch.cuda.memory_reserved()` or NVML. Those
are process/device observations rather than Residency admission evidence.

## 2. Verify the requested endpoint

For each failed operation, record the required pair:

```text
(handle, pageable-host | pinned-host | cuda:N)
```

Then inspect the matching value snapshot:

- confirm the handle belongs to this manager and is not discarded;
- list its current materialization locations;
- check `replica_mode` and `recoverability`;
- check `has_source` and `source_location`; and
- for a request, verify every `ResidencyRequirement` independently.

`ResidencyUnavailableError` means the concrete endpoint or reconstruction path
needed by a direct operation is absent. In particular, `acquire` never places a
missing value. Admit the endpoint first with a direct transition, manual scope,
or controller decision; do not convert acquisition into an implicit copy.

For a repeated handle, remember that several locations are simultaneous
postconditions. They are valid only for `REPLICABLE` values.

## 3. Reconcile budget arithmetic

At each involved location compute:

```text
current charged = used_bytes + reserved_bytes
new peak        = current charged + requested or temporary charge
admissible      = budget_bytes is None or new peak <= budget_bytes
```

For `ResidencyBudgetError`, use its reported `location`, `budget_bytes`,
`used_bytes`, `reserved_bytes`, and `requested_bytes` fields. Determine whether
the request is a new materialization, a `MemoryReservation`, or temporary
reconstruction storage before changing the limit.

For a manual plan, inspect:

```python
explanation = manager.explain(plan)
print(explanation.feasible, explanation.reason)
print(explanation.predicted_peak_bytes)
```

For controller admission, inspect the same evidence in
`decision.explanation`. Include all of the following in the accounting review:

1. reclaim effects before reservation admission;
2. active named `MemoryReservation` charges;
3. destination materialization charge;
4. source and destination overlap during transfer;
5. temporary reconstruction charge at `source_location`; and
6. the fixed conservative value charge rather than only current backing
   storage.

A reconstructible value can still be infeasible when loading it temporarily
exceeds the source-location budget. Increasing a budget is justified only when
the complete measured peak and required unmanaged headroom support it.

## 4. Check protection before changing placement

For the materialization that must be removed, inspect:

```text
use_count
hold_count
pending_event_count
```

`ResidencyInUseError` means an active lease, hold, or pending CUDA consumer event
protects the location. Resolve the owning lifetime:

- exit or release the lease after all reader streams are registered;
- release an obsolete hold;
- wait for the actual consumer stream when the application requires immediate
  removal; or
- defer reclaim until a later decision.

Do not force-close a manager or discard a value to bypass unknown CUDA readers.
A controller does not wait for protected candidates; it searches other legal
choices or reports infeasibility.

## 5. Inspect automatic decision evidence

Derive a fresh decision only from the captured request and its postconditions:

```python
decision = controller.decide(request)
print(decision.expected_state_version)
print(decision.explored_states)
print(decision.evictions)
print(decision.plan.reclaim, decision.plan.enter)
print(decision.explanation.predicted_peak_bytes)
```

Check that every requested endpoint appears in the final simulated state and
that each eviction names the expected released location, byte charge, and
reason. `explored_states` measures bounded deterministic search work; it is not
an eviction count or a performance score.

Distinguish two failures:

- `ResidencyPlanError` means validation or an exhaustive search within the
  configured bound found the plan/request infeasible for current state.
- `ResidencySearchLimitError` means the search was inconclusive because it
  reached `state_limit`. Record `request_name`, `state_limit`,
  `explored_states`, and `detail`. Do not report this as proof that no feasible
  placement exists.

Reduce unnecessary alternatives or raise `search_state_limit` only after
confirming that the request and fallback graph are intentional.

## 6. Treat stale state as a concurrency result

`ResidencyStaleStateError` reports `expected_version` and `actual_version`.
Scope entry detects the mismatch before reclaim, reservation admission, or entry
mutation. Identify the intervening ownership, placement, reservation,
protection, or completed-event reaping operation.

If the request remains valid, call `controller.decide(request)` again and review
the new evidence. Do not silently retry an old decision inside application
execution; the selected victims and peaks may have changed.

## 7. Recover from runtime plan failure without assuming rollback

`ResidencyPlanExecutionError` means preflight succeeded and execution then
failed. Inspect all structured fields before issuing another transition:

```python
except ResidencyPlanExecutionError as error:
    print(error.phase)
    print(error.failed_action, error.failed_action_index)
    print(error.failed_reservation, error.failed_reservation_index)
    print(error.partial_report.transitions)
    print(error.__cause__)
```

`phase` is one of `execute`, `reclaim`, `reserve`, `enter`, or `exit`.
`partial_report.transitions` lists every committed transition; those effects are
not rolled back. Reservation-prefix cleanup occurs on failed scope entry, but
completed reclaim or entry actions remain committed. Capture a new snapshot and
continue from that actual state.

If a scope body also failed, keep the body exception as primary and inspect
`scope.exit_error` for a structured exit failure. A failed single-use scope
cannot be re-entered to complete the plan.

## 8. Classify the result precisely

| Exception | Meaning | First corrective action |
| --- | --- | --- |
| `ResidencyBudgetError` | A direct materialization or reservation exceeds one strict location budget. | Reconcile its five byte fields and the missing headroom. |
| `ResidencyPlanError` | A plan/request is invalid or infeasible before its actions begin. | Inspect explanation, endpoints, ownership, budgets, and protections. |
| `ResidencySearchLimitError` | Automatic search reached its deterministic bound; feasibility is unknown. | Record search evidence, simplify alternatives, or justify a larger bound. |
| `ResidencyStaleStateError` | A decision's expected manager version no longer matches. | Identify the mutation and derive/review a new decision if still required. |
| `ResidencyInUseError` | A lease, hold, or pending CUDA event blocks direct removal. | Resolve the owning lifetime or defer removal. |
| `ResidencyUnavailableError` | A direct operation requires an absent endpoint or source. | Establish the requested endpoint or valid reconstruction path. |
| `ResidencyPlanExecutionError` | Execution failed after preflight and may have committed a prefix. | Inspect `phase`, `partial_report`, failed object/index, cause, and current snapshot. |

`ResidencySearchLimitError` and `ResidencyStaleStateError` are specialized
`ResidencyPlanError` subclasses. Catch them before a general
`ResidencyPlanError` when their structured evidence requires different handling.
`ResidencyPlanExecutionError` is a sibling runtime category and always requires
partial-state analysis.

## References

- [Residency plans and execution internals](../developer/residency-plans-and-execution.md)
- [Residency lifetimes](../concepts/execution/residency-lifetimes.md)
- [Automatic Residency tutorial](../tutorial/automatic-residency.md)
- [Explicit Residency tutorial](../tutorial/explicit-residency.md)
- [Automatic Example 14 source](https://github.com/VisualDust/fhelium/blob/main/examples/14_automatic_residency.py)
- [`ResidencySnapshot` API](../api/fhelium/residency/snapshot.md#residencysnapshot)
- [`ResidencyDecision` API](../api/fhelium/residency/controller.md#residencydecision)
- [Residency exception API](../api/fhelium/errors.md)
