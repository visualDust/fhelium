# Automatic residency admission

**Example source:** [`examples/14_automatic_residency.py`](https://github.com/VisualDust/fhelium/blob/main/examples/14_automatic_residency.py)

This example admits a CKKS working set under a strict managed CUDA budget. A
cold replicable plaintext already occupies CUDA capacity, so deterministic
decision-making must reclaim that cache before placing an encrypted input,
rotation key, and operation-ready plaintext. The application reviews the resulting
decision, enters its state-bound scope, and executes rotation,
plaintext multiplication, and rescaling through a strict CUDA lease.

## Run the example

```bash
python examples/14_automatic_residency.py \
  --device cuda:0 \
  --preset slots8192-scale40-levels7-int64
```

CUDA is required. The output reports selected reclaim evidence, predicted
managed peaks, final cached endpoints, transition count, and decryption error.

## 1. Create a real capacity conflict

The manager budget is sized for the requested inputs and evaluator
headroom, not for those inputs plus an unrelated CUDA cache:

```python
cuda_budget = (
    weight_charge + key_charge + source_charge + workspace_bytes
)
residency = ResidencyManager(
    budgets={
        PINNED_HOST: source_charge,
        cuda: cuda_budget,
    }
)
```

The example adopts a cold plaintext as `REPLICABLE` at pageable host and then
creates a CUDA replica with a direct manager primitive:

```python
cold_handle = residency.adopt(
    cold,
    at=PAGEABLE_HOST,
    replica_mode=ReplicaMode.REPLICABLE,
)
residency.ensure(cold_handle, cuda, stream=transfer_stream)
```

Its host replica preserves the logical value, so the CUDA replica is a legal
reclaim candidate. No synthetic allocation or allocator free-memory estimate
participates in admission.

## 2. Separate intent, choice, evidence, and execution

Automatic Residency adds an inspectable policy layer above the manager:

- `ResidencyRequest` contains required `(handle, location)` postconditions and
  named reservation headroom.
- `ResidencyPolicy` ranks only legal candidates and names configured fallback
  tiers; `DeterministicTieredLRU` is the maintained deterministic policy.
- `ResidencyDecision` records selected reclaim evidence, a concrete plan, dry-run
  explanation, policy identity, and expected manager state version.
- `ResidencyPlan` is ordered low-level command intermediate representation; it
  contains no policy.
- `ResidencyController` derives and admits decisions but owns no concrete
  value. The `ResidencyManager` remains the sole placement, accounting, and
  lifetime authority.
- `ResidencyController.use` is the combined convenience context. This example
  deliberately uses `decide` followed by `scope` so the decision is visible
  before admission.

The request states only the final working set and headroom:

```python
request = ResidencyRequest(
    name="automatic/rotate-scale/tile-0",
    requirements=(
        ResidencyRequirement(source_handle, cuda),
        ResidencyRequirement(key_handle, cuda),
        ResidencyRequirement(weight_handle, cuda),
    ),
    reservations=(
        MemoryReservation(
            cuda,
            workspace_bytes,
            label="rotate/multiply outputs and workspace",
        ),
    ),
)
```

The reservation is managed accounting headroom for outputs and evaluator
workspace. It does not allocate a tensor.

## 3. Inspect a tensor-free, state-bound decision

```python
decision = controller.decide(request)
print(decision.evictions)
print(decision.explored_states)
print(decision.explanation.predicted_peak_bytes)
```

Decision-making reads immutable tensor-free manager snapshots. In this workload,
the only unrelated CUDA materialization is the cold cached replica; the decision
therefore contains its deterministic `DropResident` reclaim action. The
pageable replica remains present.

`decision.expected_state_version` is a precondition, not informational
metadata. An intervening manager mutation makes the decision stale. Entering
`controller.scope(decision, ...)` checks the version atomically before reclaim,
reservation admission, or placement, and raises `ResidencyStaleStateError`
rather than silently replanning.

## 4. Bind copy and consumer lifetimes separately

```python
scope = controller.scope(
    decision,
    transfer_streams={cuda: transfer_stream},
)
with scope:
    with residency.acquire(
        (source_handle, key_handle, weight_handle),
        at=cuda,
        consumer_stream=compute_stream,
    ) as resident:
        with torch.cuda.stream(compute_stream):
            rotated = engine.rotate_with_key(
                resident[source_handle],
                resident[key_handle],
            )
            output = engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(
                    engine.multiply_plaintext(
                        engine.coefficient_domain_to_ntt_domain(rotated),
                        resident[weight_handle],
                    )
                )
            )

    compute_stream.synchronize()
```

The transfer-stream mapping belongs to placement actions selected by the
decision's plan. The consumer stream belongs to the lease protecting concrete
readers. `ResidencyManager.acquire` remains already-ready-only: it neither
places a missing value nor invokes policy. Lease release records CUDA completion
for the managed inputs, while synchronization inside the scope keeps the
reservation active through the result-producing work.

## 5. Use the combined convenience context when review is unnecessary

`controller.use` combines decision-making, version-checked scope entry, and strict
lease acquisition without changing any step's preconditions or effects:

```python
automatic = controller.use(
    request,
    consumer_streams={cuda: compute_stream},
    transfer_streams={cuda: transfer_stream},
)
with automatic as active:
    source_value = active.value(source_handle, at=cuda)
    key_value = active.value(key_handle, at=cuda)
    weight_value = active.value(weight_handle, at=cuda)
```

The exact decision becomes available as `active.decision` after successful
entry, and the completed plan report is available as `automatic.report` after
exit. Use separate `decide` and `scope` calls, as the runnable example does,
when admission must be reviewed before any manager mutation.

## 6. Observe cache and accounting outcomes

The decision's plan has a reclaim prefix and placement entry actions but no exit
removals. After successful scope exit:

- the cold value remains cached at pageable host but no longer on CUDA;
- every requested CUDA endpoint remains cached;
- the workspace reservation is released; and
- manager peak accounting retains the admitted high-water mark.

Keeping successful endpoints cached is intentional. A later request may reuse
them or deterministically reclaim eligible replicas under new pressure.
Automation never ends a managed logical identity. The example therefore calls
`discard` for every handle and then closes the manager after verifying CKKS
correctness and accounting.

::: details Complete runnable source
<<< @/../examples/14_automatic_residency.py
:::

## Continue

- [Explicit residency plans and CUDA leases](./explicit-residency.md)
- [Residency lifetimes](../concepts/execution/residency-lifetimes.md)
- [Residency plans and execution internals](../developer/residency-plans-and-execution.md)
