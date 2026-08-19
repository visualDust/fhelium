---
title: Benchmark v1 methodology and specification
description: Measurement methodology, report format, publication validation, and comparison semantics for FHElium Benchmark v1.
layout: page
pageClass: benchmark-methodology-page
sidebar: false
aside: false
---

<script setup lang="ts">
import BenchmarkPortalHeader from '../../.vitepress/theme/benchmarks/v1/components/BenchmarkPortalHeader.vue'
</script>

<BenchmarkPortalHeader title="Benchmark v1 methodology and specification" />

<div class="benchmark-methodology-content vp-doc">

FHElium Benchmark v1 is one fixed, cross-backend measurement product. The `v1` identity binds five ordered cases, their effective parameters, measurement rules, correctness criteria, report format, publication projection, comparison behavior, and portal interpretation. A formal run selects exactly one report-level execution target—`cpu` or an indexed `cuda:N` device—and executes every case on it. The selected device is provenance, not a workload parameter; therefore CPU and CUDA runs have the same manifest, case IDs, profiles, effective parameters, metrics, and validation criteria.

The website publishes only complete formal v1 reports. An isolated leaf result is never presented as a v1 run. The Explorer, detail view, and Compare use the same run representation and UI for CPU and CUDA. They distinguish the selected execution device from host and visible-hardware inventory, and they define no composite score.

## Fixed case specification

Benchmark v1 contains five cases on the common `slots8192-scale40-levels7-int64` CKKS plan with `radix2_indexed` NTT:

1. **Depth-aware CKKS single operations.** Twelve public operations are measured at all seven ordinary public entry levels with one warmup and three timed samples.
2. **Indexed radix-2 NTT operations.** Q and QP tensors are measured at levels 0 through 6 for forward, inverse, and roundtrip transforms. The backend is fixed; this case does not rank or tune backend implementations.
3. **Plaintext × ciphertext dense matrix multiplication.** One fixed 16 × 16 cyclic-diagonal packed product runs sequentially and unbatched on one selected device.
4. **Ciphertext × ciphertext dense matrix multiplication.** The same 16 × 16 shape and schedule use encrypted diagonals, late relinearization, and one rescale per output column.
5. **Polynomial methods.** One affine polynomial and one dense degree-four polynomial exercise balanced power, corrected Horner, and fixed-`k` Paterson–Stockmeyer methods where applicable.

The fixed matrix and bounded-depth polynomial set are part of v1 for both backends. They are not CPU substitutions: a CUDA run executes the identical shapes, methods, sample counts, and parameters. Multi-GPU scaling and CUDA-only NTT backend studies remain independent benchmarks outside formal v1.

## Execution identity

The raw report records the target once:

```json
{
  "benchmark_version": "v1",
  "execution": {
    "backend": "cpu",
    "device": "cpu"
  }
}
```

CUDA uses `{"backend":"cuda","device":"cuda:0"}` or another nonnegative index. Case parameters contain no device field. The runner passes the selected target to every case while preserving the package-owned resolved manifest.

## Measurement and validation

Every measured case carries a `BenchmarkTimedBoundary` identifying included work, excluded setup, and synchronization. CUDA samples synchronize the selected device; CPU operations complete synchronously. Setup, key construction, input preparation, decryption, and cleartext comparison are excluded unless a case states otherwise.

Metrics remain typed by name, unit, statistic, direction, and dimensions:

| Category | Interpretation |
| --- | --- |
| Latency | Time for the declared interval; lower is better |
| Throughput/rate | Declared work divided by that interval; higher is better |
| Memory | Reported only where one equivalent cross-backend measurement exists |

Absent measurements are blank, never zero. CUDA allocator counters are not used as CPU-versus-CUDA metrics because zero from the CPU path would not represent equivalent memory evidence.

Every measured configuration must pass its exact correctness checks before a report can complete. The raw report retains each oracle, observed value, limit, and supporting details. Numerical limits are never widened merely to make a run pass. The retained affine and degree-four method limits require controlled cross-backend calibration before release; current implementation validation preserves rather than relaxes those criteria.

## Platform and provenance

A report records CPU topology, system memory, operating system, Python, PyTorch, FHElium source/native identity, CUDA build information, and GPU inventory when available. `execution` identifies what ran; inventory only describes the host. Publication rejects credentials, user paths, unrestricted environment dumps, non-finite values, malformed execution targets, and CUDA runs without CUDA-device provenance.

Formal reproducibility evidence requires a clean source checkout. `--allow-dirty` is a developer-evidence exception; the portal labels the resulting run as containing local changes.

## Comparison identity

Compare aligns the same five fixed case IDs across selected v1 runs. CPU and CUDA reports are directly comparable because the manifest, parameters, metric identities, dimensions, and criteria match. Compare does not substitute a profile, resize a workload, drop a case, compute a winner, or infer a composite score.

## Report and catalog formats

`fhelium/benchmarks/v1/specification.json` is the canonical resolved manifest. Runtime definitions, the dependency-free publisher, and the v1 frontend pin the same SHA-256 identity and five-case order.

Published data uses:

```text
docs/public/benchmarks/v1/
  catalog.json
  runs/
    sha256-<64 lowercase hexadecimal digits>.json
```

Each raw filename is the SHA-256 digest of its exact bytes. `catalog.json` is a rendering projection in which one entry is one complete formal run; cases are nested and never published as standalone results. Catalog runs include the exact report-level `execution` object.

The current fixed manifest identity is:

```text
5b9ce22abf59cb5b37dcc59062e2f856df40584470082fd0a4c6f08ee9b81c4b
```

## Running v1

Run every case on CPU:

```bash
fhelium benchmark v1 run \
  --device cpu \
  --output results/fhelium-benchmark-v1-cpu.json
```

Run the identical specification on one CUDA device:

```bash
fhelium benchmark v1 run \
  --device cuda:0 \
  --output results/fhelium-benchmark-v1-cuda.json
```

There are no case, profile, shape, or parameter overrides in formal v1. Use an independent leaf benchmark for another research question; it is not a v1 report.

Publish a completed clean report with:

```bash
python scripts/publish_benchmark_v1.py \
  results/fhelium-benchmark-v1-cpu.json
```

The publisher authenticates the fixed manifest, validates all five case positions and passed checks, sanitizes provenance, writes immutable content-addressed raw bytes, and rebuilds the catalog projection.

</div>
