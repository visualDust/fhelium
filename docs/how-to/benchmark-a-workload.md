# Benchmark a workload correctly

A trustworthy FHElium benchmark defines the mathematical workload, CKKS
state, timed work, synchronization, memory accounting, and correctness
criterion. Latency without those fields is not reproducible.

## Run FHElium Benchmark v1

The maintained Benchmark v1 product runs one fixed five-case specification on a selected CPU or indexed CUDA device. Device selection does not resize a workload or alter its CKKS parameters.

```bash
fhelium benchmark v1 run --device cpu \
  --output results/fhelium-benchmark-v1.json
```

Use `--device cuda:0` to run the identical case IDs, parameters, sample counts, and correctness criteria on CUDA. A measured-case failure or interruption produces a nonzero exit after the checkpointed report is written.

Benchmark v1 does not accept case, profile, or parameter overrides. Its all-level operation sweep, fixed indexed NTT, matrix shape, polynomial methods, sampling policy, report format, and portal interpretation form one specification. `execution` records only which backend/device ran it.

Run an independent leaf benchmark when investigating a different question or validating a local harness. It does not produce a publishable Benchmark v1 report. Any maintained specification change belongs to a separate Benchmark version.

## 1. Define the question

Choose one layer:

```text
kernel microbenchmark
CKKS operator
complete packed workload
distributed end-to-end workload
```

Do not use an NTT microbenchmark to claim an equal application speedup, or an
application result to attribute all gains to one kernel.

## 2. Freeze the mathematical definition

Record:

- cleartext operation and packing;
- input amplitude/distribution and seed;
- output region and mask/padding behavior;
- expected maximum absolute/relative error;
- operation schedule and rotation steps.

Run the same oracle for every configuration in an ablation.

## 3. Freeze CKKS state

Record:

- preset and `logN`;
- level and scale at the measured operation;
- active Q/P row counts and relevant `prime_ids`;
- polynomial domain, modulus basis, residue representation, and components;
- NTT backend/grouping;
- hoist chunk and direct keyset.

A later-level operation can be much cheaper than level zero because fewer rows
are active. Compare like with like.

## 4. Define the timed work

State whether timing includes:

- engine construction;
- key generation/loading/movement;
- encoding and operation-ready preparation;
- encryption/decryption;
- input staging;
- CUDA Graph capture or only replay;
- distributed process-group startup;
- final gather/reduction.

If setup is excluded, complete it and synchronize before the measured region.

## 5. Synchronize CUDA correctly

CUDA launches are asynchronous. Use either:

- a benchmark helper that synchronizes the measured CUDA devices; or
- CUDA events on the relevant stream, followed by event synchronization.

For distributed measurements, define whether the reported latency is:

- maximum rank-local elapsed time;
- a barrier-bounded end-to-end interval;
- root completion time;
- local compute excluding/including typed reduction.

Do not add hidden synchronization to one configuration only.

## 6. Warm up appropriately

Warm up until relevant lazy behavior is outside the measured region:

- native extension/operator loading;
- key materialization;
- allocator setup;
- graph capture/warmup;
- process-group initialization;
- reusable buffer setup.

Report warmup and measured-run counts and a robust statistic such as median,
with variance or quantiles when useful.

## 7. Reset and measure memory

Distinguish:

- live value bytes;
- PyTorch peak allocated bytes;
- peak reserved bytes;
- per-rank maximum;
- aggregate retained key/weight bytes;
- transient communication/prefetch peaks.

Reset peak statistics at a documented observation point. Remember that allocator
reserved memory can remain high after live references are released.

## 8. Validate every measured configuration

Decrypt and compare to the oracle. Record at least:

```text
max absolute error
relative error where meaningful
output level and scale
component count/polynomial domain/modulus basis
```

A fast result with wrong residues, modular wrap, or borrowed-output overwrite is
not a successful benchmark.

## 9. Change one variable per ablation

Examples:

```text
indexed vs compact NTT with all else fixed
hoist chunk 4 vs 8 vs 16
CUDA Graph on vs off with same input staging
one rank vs two ranks with same packing/key strategy
all-resident vs bounded window with same evaluator
```

If multiple mechanisms change together, describe the result as a workload-level
configuration comparison rather than attributing it to one mechanism.

For homogeneous batching, keep three reproducible comparisons separate:

```text
singleton batch [1, ...] vs an unbatched value
one B4/B8 evaluation vs a loop over the same B members
complete batched evaluator vs complete loop, including peak memory
```

Run both sides from the same build with the same evaluator. Do not use an
earlier development commit or an implementation unavailable to readers as the
public baseline. Report peak memory with latency and repeat the comparison at
production levels. Active QP rows can move the same batch across a
cache/bandwidth crossover. Follow
[Choose a homogeneous batch size](choose-homogeneous-batch-size.md) for the
full procedure and a worked hardware measurement.

### Maintained packed-matvec reference

The README and documentation-home comparison uses the level-0 packed
`128 x 128` matrix-vector profile. The six FHElium cases were independently
tuned on two NVIDIA RTX PRO 6000 Blackwell GPUs on 2026-07-24; the selected
policies and synchronized ten-run medians are:

| `logN` / public levels (`scale_bits=40`) | Ranks | NTT backend | Hoist | Diagonal batch | CUDA Graph | Median (ms) | Peak/rank (GiB) |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: |
| 14 / 7 | 1 | `radix2_compact_group8_smem8` | 64 | 16 | yes | 14.7698 | 2.387 |
| 14 / 7 | 2 | `radix2_compact_group8_smem8` | 64 | 16 | yes | 7.8192 | 1.205 |
| 15 / 16 | 1 | `radix2_compact_group16_smem8` | 64 | 8 | yes | 46.5788 | 11.280 |
| 15 / 16 | 2 | `radix2_compact_group16_smem8` | 64 | 8 | yes | 22.8967 | 5.694 |
| 16 / 34 | 1 | `radix2_compact_group16_smem8` | 127 | loop | yes | 194.5760 | 50.866 |
| 16 / 34 | 2 | `radix2_compact_group16_smem8` | 64 | loop | yes | 99.4266 | 25.669 |

Every result passed the cleartext oracle with maximum absolute error below
`3.88e-8`. The `logN = 14` and `logN = 15` profile defaults encode their
selected policies. The `logN = 16` profile uses the robust two-rank/default
hoist bound of 64;
the one-rank reference overrides it to 127. This is a workload-level
configuration result: it must not be attributed to batching, hoisting, the NTT
backend, or graph replay alone.

### RTX A6000 portability check

The same commit was independently tuned on two 48-GB NVIDIA RTX A6000 GPUs
connected by NV4 on 2026-07-25. The selected configurations differed from the
RTX PRO 6000 policies:

| `logN` / public levels (`scale_bits=40`) | Ranks | NTT backend | Hoist | Diagonal policy | CUDA Graph | Median (ms) | Peak/rank (GiB) |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: |
| 14 / 7 | 1 | `radix2_compact_group8_smem8` | 127 | batch 2 | yes | 34.3615 | 2.387 |
| 14 / 7 | 2 | `radix2_compact_group8_smem8` | 64 | batch 2 | yes | 18.0921 | 1.205 |
| 15 / 16 | 1 | `radix8_compact` | 127 | loop | yes | 170.9627 | 11.281 |
| 15 / 16 | 2 | `radix8_compact` | 64 | loop | yes | 86.0165 | 5.694 |
| 16 / 34 | 2 | `radix16_compact` | 64 | loop | yes | 377.8936 | 25.669 |

These are synchronized three-block, ten-run median-of-medians results. Every
configuration passed ciphertext residue checks and the cleartext oracle; the
maximum absolute error remained below `3.881e-8`. One-to-two-rank speedup was
1.899x for `logN = 14` and 1.988x for `logN = 15`.

Retuning reduced single-GPU latency relative to the existing profile policy:

| `logN` / workload | Existing latency | A6000-tuned latency | Latency reduction |
| --- | ---: | ---: | ---: |
| 14 / 128 x 128 | 38.6663 ms | 34.3615 ms | 11.1% |
| 15 / 128 x 128 | 206.6773 ms | 170.9627 ms | 17.3% |
| 16 / 64 x 64 | 460.8441 ms | 373.3069 ms | 19.0% |

The single-rank `logN = 16`, 128 x 128 case was excluded by a
memory-capacity check.
The measured 64 x 64 run retained 23.99 GiB of rotation keys and reached
25.33 GiB peak; the 128 x 128 estimate was 51.04 GiB, above the device's
47.40 GiB physical memory. The two-rank 128 x 128 case fit at 25.67 GiB per
rank.

This is a hardware-portability result rather than a direct GPU speed ratio.
Profile values are reproducible starting points; confirm backend, hoist,
diagonal batching, graph replay, and memory capacity on the deployment GPU.

## 10. Record environment and provenance

Include:

- FHElium version and commit;
- clean/source build or installed wheel;
- GPU model/count and topology;
- driver, CUDA, PyTorch, Python;
- launcher and environment variables that affect execution;
- benchmark command/profile;
- whether GPUs were idle and clocks/power policy were controlled.

Useful commands include:

```bash
fhelium cuda info
fhelium cuda topo --bandwidth
fhelium benchmark list
```

Use `fhelium benchmark run ...` with a current listed profile rather than
copying a stale command from an older report.

## Recommend an NTT backend

Use the NTT recommendation command to rank every backend compatible with
one preset. Begin with synchronized raw transforms:

```bash
fhelium benchmark recommend ntt --suite kernel --preset slots32768-scale40-levels34-int64 \
  --device cuda:0 --output results/ntt-kernel-recommendation.json
```

Then confirm the choice with encryption, decryption, multiplication plus
relinearization, scalar rotation, and grouped rotation:

```bash
fhelium benchmark recommend ntt --suite ckks-primitive --preset slots32768-scale40-levels34-int64 \
  --device cuda:0 --output results/ntt-primitives-recommendation.json
```

The command reports a recommendation and confidence but never changes the
library default or caches a hidden device choice. Preserve the JSON evidence
and pass the selected name to `CkksEngine(ntt_backend=...)`. See the
focused [NTT backend screening guide](screen-ntt-backends.md)
for the first-use workflow and example output. Use
[Analyze and choose an NTT backend](choose-ntt-backend.md) when kernel and
primitive rankings need a launch, memory-traffic, occupancy, or CKKS-composition
explanation.

## Built-in NTT backend comparison

Run every canonical backend compatible with one ring dimension using:

```bash
fhelium benchmark run ntt-backend-single-op \
  --profile slots32768-scale40-levels34-int64 \
  --output results/ntt-slots32768-scale40-levels34-int64.json
```

Stable profiles use the four 40-bit preset values from
`slots8192-scale40-levels7-int64` through `slots65536-scale40-levels72-int64`; `quick` is a
short 8,192-slot smoke run. A profile with `backends=null` resolves
`compatible_ntt_backends(logN)` at execution time. Incompatible strict
fixed-radix policies are intentionally omitted. No supported preset is
compatible with all strict radix-4, radix-8, and radix-16 policies, so one run
is an all-compatible comparison at a fixed `N`, not a sweep of every
registered name across different `N` values.

The benchmark uses all level-zero QP rows. Forward input is
coefficient/Montgomery, inverse input is NTT/Montgomery, and input reset copies
are excluded from the synchronized timed interval. Every backend is
roundtrip-validated before measurement. Rows report mean, median, min, max,
population standard deviation, and speedup relative to
`radix2_indexed` when that reference is included. This remains an
NTT microbenchmark and must not be used by itself to claim the same speedup for
a CKKS operator or complete workload.

A subset can be selected with a JSON override:

```bash
fhelium benchmark run ntt-backend-single-op \
  --profile slots32768-scale40-levels34-int64 \
  --set 'backends=["radix2_compact_group8_smem8","radix16_compact"]'
```

Override names are validated against the selected `logN`; they are not
silently skipped.

## Register a specialist custom benchmark

Use `fhelium benchmark --file PATH ...` when a workload should reuse the
structured profiles, result model, CLI overrides, and TUI without becoming a
built-in FHElium workload. The supported extension surface lives in the
unflattened `fhelium.benchmarks` subpackage; CLI, renderer, TUI, worker, timing,
and built-in implementation modules are internal. See the
[benchmark framework API](../api/fhelium/benchmarks/model.md) for the registration hook and
a complete minimal file.

## Minimum result table

| Field | Value |
| --- | --- |
| Workload/config | ... |
| GPU / ranks | ... |
| CKKS state | ... |
| Backend/hoist/graph | ... |
| Timed work | ... |
| Median / variance | ... |
| Peak allocated/reserved | ... |
| Key/weight footprint | ... |
| Max error | ... |
| Version/commit | ... |

## Related documentation

- [CKKS cost model](../concepts/performance/cost-model.md)
- [Choose a preset](choose-preset-and-depth.md)
- [Optimize a workload](optimize-workload.md)
- [Choose a homogeneous batch size](choose-homogeneous-batch-size.md)
- [Rotation-hoisting tutorial](../tutorial/rotation-hoisting.md)
