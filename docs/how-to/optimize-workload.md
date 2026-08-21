# Optimize a workload systematically

Optimize a complete correct evaluator by locating its dominant cost, changing
one mechanism, and retaining the same oracle and state invariants.

## 1. Preserve the baseline

Keep a reproducible single-rank eager evaluator with:

- CKKS state schedule;
- direct keyset;
- cleartext oracle;
- synchronized latency;
- memory profile;
- target amplitude and error tolerance.

Commit or otherwise preserve its benchmark configuration before beginning an
ablation.

## 2. Profile by phase

Measure at least:

```text
input/key provisioning
plaintext preparation
rotations and key switches
NTT transforms
pointwise arithmetic
rescale/relinearization
accumulation
host-to-device (H2D) copies
rank-local launch overhead
distributed gather/reduction
```

Use operator and kernel profiling to explain the workload result, not to replace
it.

## 3. Match the mechanism to the bottleneck

| Dominant cost | First mechanisms to test |
| --- | --- |
| Repeated relinearization | Late relinearization where triplets align |
| Repeated plaintext preparation | Operation-ready level-specific plaintexts |
| Repeated fixed NTT operand | Reuse prepared NTT/Montgomery value |
| Many rotations/key switches | Hoisting, direct keyset, packing/schedule changes |
| NTT table/launch traffic | Indexed/compact and grouping ablation |
| Python/dispatcher launches | Rank-local CUDA Graph |
| CUDA footprint | Prepared-state audit, streaming, bounded residency |
| Independent requests | Data parallelism |
| Additive rotation terms | Additive-term parallelism with typed reduction |
| One huge value with long row-local phase | Limb parallelism, cautiously |

## 4. Estimate the expected effect

Before coding, state what the mechanism should remove:

```text
number of key switches avoided
number of repeated NTTs avoided
number of launches reduced
bytes moved or retained
communication moved from per-term to start/end
```

Also state expected new costs such as larger triplets, hoist temporaries,
graph-private memory, H2D traffic, or key replication.

## 5. Run a one-variable ablation

Hold fixed:

- input and seed;
- packing and mathematical output;
- preset, level, scale, and rows;
- key strategy unless it is the tested variable;
- warmup/runs/statistic;
- synchronization rule and memory measurement point.

Measure latency, memory, and error together.

## 6. Tune bounded parameters

Several mechanisms have a non-monotonic control:

- hoist chunk size;
- NTT grouping width;
- tile size;
- residency lookahead;
- number of graph/program instances;
- distributed term assignment.

Sweep a small justified range. Do not assume the largest value is best.

## 7. Check numerical and semantic invariants

After each change, verify:

- output error and range;
- level and scale schedule;
- active rows and basis;
- component count;
- rotation direction and direct keys;
- in-place/borrowed storage lifetime;
- distributed gather/reduce/reconstruct semantics.

Optimizations must not bypass range guards or silently change the rescale or
relinearization placement without updating the declared operation schedule.

## 8. Validate broader coverage

A successful local result should be checked across:

- fast smoke and target presets;
- early, middle, and late legal levels where relevant;
- multiple seeds and amplitudes;
- source build and installed wheel for native changes;
- target GPU architecture;
- world size one and target rank count for distributed changes.

## 9. Add regression protection

Promote the improvement into:

- a correctness test for its state transition;
- an ABI/wrapper test if native code changed;
- a benchmark profile or recorded baseline;
- documentation describing applicability and trade-offs.

Avoid tests that assert fragile absolute latency on shared hardware. Prefer
structural counters, correctness, and separately monitored performance data.

## 10. Report attribution honestly

If a result combines packing, hoisting, backend changes, graph replay, and
multiple GPUs, call it an **application-level configuration result**. Report
component ablations before attributing the speedup to one kernel.

## Related documentation

- [CKKS cost model](../concepts/performance/cost-model.md)
- [Benchmark a workload](benchmark-a-workload.md)
- [Late-relinearization tutorial](../tutorial/late-relinearization-and-ntt-reuse.md)
- [Rotation-hoisting tutorial](../tutorial/rotation-hoisting.md)
- [Capture a repeated evaluator](capture-repeated-evaluator.md)
- [Stream bounded memory](stream-bounded-memory.md)
