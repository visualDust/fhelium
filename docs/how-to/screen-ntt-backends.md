# Select an NTT backend from measured evidence

FHElium provides several mathematically equivalent Number Theoretic Transform
(NTT) implementations. Their relative latency depends on the GPU, ring
dimension, number of active modulus primes, and surrounding CKKS operation.
Use the default for a first evaluator. Compare alternatives only after a
correct, representative workload exists on the target GPU.

The current recommendation runner is an internal diagnostic path that uses
the retained `fhelium.legacy.engine.CkksEngine` implementation. It can screen
backend policies, but its measurements are not measurements of the current
`fhelium.eager.Engine`. Confirm a selected backend with the current Eager
evaluator and the production workload before deployment.

Backend choice is an `Engine` execution option. It does not change
CKKS parameters, ciphertext compatibility, or the library default.

The library default is one static backend name for every supported `logN` and
GPU. FHElium does not inspect the device or benchmark during engine
construction. The commands below only produce evidence for an application
choice.

## What the two commands measure

An **NTT backend** is one implementation of the same forward and inverse
Number Theoretic Transform. Changing the backend changes how the GPU executes
the transform; it does not change the CKKS parameters or mathematical result.

The CLI calls each fixed list of measured operations a `suite`:

- `--suite kernel` measures prepared forward NTT, inverse NTT, and an NTT
  round trip. Despite the CLI name, this test does not report one CUDA kernel
  launch. It reports complete NTT method calls while excluding engine and
  input construction. This page calls it the **NTT-only test**.
- `--suite ckks-primitive` measures complete method calls in the retained
  legacy evaluator:
  encryption, decryption, multiplication followed by relinearization (reducing
  a three-component multiplication result to two components), one rotation,
  and four rotations that reuse one ciphertext decomposition (grouped
  hoisting). This page calls it the **complete-CKKS test**.

The first test answers “which transform implementations are clearly slower?”
The second answers “which backend gives the best result after NTT work is
combined with the other work inside common CKKS operations?”

## How the percentage is calculated

Each test contains operations with different latency scales. The command gives
every operation equal weight:

1. For each operation, divide a backend's median latency by the smallest median
   latency observed for that operation.
2. Multiply those ratios and take the root whose degree is the number of
   operations—the geometric mean. This combines the operation latencies into
   one **relative latency** for that test; lower is faster.
3. Report how much that relative latency exceeds the fastest backend's result
   in the same test.

Therefore:

- `0%` means the backend is fastest by that test's equal-weight calculation;
- `4%` means its relative latency is 4% higher than the fastest result; and
- the percentage does not mean that every operation, or the complete
  application, is 4% slower.

FHElium treats results from `0%` through `3%` as too close to justify replacing
the current default backend. When the default falls in that range, the command
keeps it. Otherwise, it recommends the backend with the lowest relative
latency.

## Measure in three steps

1. **Time NTT operations only** and remove clearly slower implementations.
2. **Time composed CKKS operations** in the retained diagnostic evaluator and
   compare the remaining backends after encryption, key-switch, and rotation
   work is included.
3. **Measure the application** before fixing the deployment choice.

### 1. Time NTT operations only

```bash
fhelium benchmark recommend ntt \
  --suite kernel \
  --preset slots32768-scale40-depth34-int64 \
  --device cuda:0 \
  --output results/ntt-kernel.json
```

The command tests every compatible production backend using forward NTT,
inverse NTT, and round-trip latency. Engine construction and input preparation
are outside the timed region.

### 2. Time complete CKKS operations

```bash
fhelium benchmark recommend ntt \
  --suite ckks-primitive \
  --preset slots32768-scale40-depth34-int64 \
  --device cuda:0 \
  --output results/ntt-ckks-primitives.json
```

This command creates keys before timing, validates decrypted results, and
measures:

- `encrypt_message` and `decrypt_message`;
- multiplication followed by relinearization;
- one keyed rotation;
- four rotations through grouped hoisting.

Key generation and correctness checking remain outside the timed region.

### 3. Confirm the application

The complete-CKKS comparison gives every measured operation equal weight. A
production workload does not. Confirm the remaining candidates with
representative depths, batch sizes, numbers of relinearizations and rotations,
and whether the application uses ordinary eager calls or captured CUDA Graph
replay. Preserve the two reports with the application measurement so that the
scope of each decision remains visible.

## Worked example: why group16 is selected

The following historical measurement used GPU 1 on an otherwise idle two-GPU
host. Its FHElium 0.10.0 identity is retained; it is not a measurement of the
current 0.20.0 release or current Eager implementation:

- NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (compute capability
  12.0, also written `sm_120`);
- `slots32768-scale40-depth34-int64`, whose ring dimension is $2^{16}$;
- FHElium 0.10.0, PyTorch 2.13.0+cu130, and CUDA 13.0;
- 3 warmups, 10 timed runs per operation, and 3 repetitions; and
- seed `20260823`; each repetition started with a different backend to reduce
  measurement-order bias.

<NttBackendSelectionChart />

| Backend | NTT-only relative latency | Complete-CKKS relative latency | Meaning for this example |
| --- | ---: | ---: | --- |
| `radix2_compact_group8_smem8` | fastest | +4.004% | Keep after the first test; do not select after the second. |
| `radix2_compact_group16_smem8` | +0.891% | fastest | Keep after the first test; select after the second. |
| `radix16_compact` | +1.445% | +3.456% | Keep after the first test; do not select after the second. |
| `radix2_compact_group4_smem8` | +5.524% | +6.347% | Remove after the first test. |
| `radix4_compact` | +6.991% | +7.144% | Remove after the first test. |

### Result of test 1

Group8 is fastest under the NTT-only calculation. Group16's relative latency
is 0.891% higher and radix16's is 1.445% higher. Because both differences are
at most 3%, the first test keeps all three candidates. Group4 and radix4 have
relative latency more than 5% higher than group8, so they can be removed for
this preset and device. The recorded second test still measured all five
backends so that the table shows the complete results.

### Result of test 2

Group16 is fastest under the complete-CKKS calculation in all three
repetitions. Radix16's relative latency is 3.456% higher and group8's is 4.004%
higher. Both exceed 3%, so the rule that retains the current default for a
close result does not apply. The command selects group16.

For every timed operation, the command divides the sample standard deviation
by the sample mean; this **coefficient of variation** describes how tightly
the repeated timings cluster. The median of those values for group16 is
0.784%. The command labels the recommendation **medium confidence** because
group16 is fastest in every repetition and its lead exceeds 3%, but the lead
does not reach the 5% required for high confidence.

The first test did not select group8 for deployment; it only prevented
group16 and radix16 from being removed too early. The second test provides a
more relevant shortlist for composed operations, but it does not replace
measurement with the current eager `Engine`.

## Apply the measured choice

The CLI prints a constructor expression. Copy the backend name:

```python
import fhelium as fh
import torch
from fhelium.eager import Engine

torch.set_default_device("cuda:0")
engine = Engine(
    fh.Preset.slots32768_scale40_depth34_int64,
    ntt_backend="radix2_compact_group16_smem8",
)
```

The recommendation command does not rewrite `DEFAULT_NTT_BACKEND`, modify
native shared-memory tuning, or cache a machine-global selection. Re-run it
when the GPU model, CUDA/PyTorch stack, FHElium version, preset, or important
workload changes. This example supports group16 only for the measured
environment, and an application benchmark must still confirm it. The result
does not establish a library-wide default.

## Compare a focused candidate set

Repeat `--backend` to restrict a follow-up comparison:

```bash
fhelium benchmark recommend ntt \
  --suite ckks-primitive \
  --preset slots32768-scale40-depth34-int64 \
  --backend radix2_compact_group16_smem8 \
  --backend radix16_compact
```

At least two compatible names are required. The command omits
`radix2_indexed` by default because the diagnostic variant serves controlled
experiments. Add that
backend by name only when a diagnostic comparison needs the indexed
implementation as another result-equality reference.

## Understand why performance changes

The two tests intentionally answer different questions. To analyze launch
count, memory coalescing, shared-memory fusion, register pressure, occupancy,
RNS row count, and key-switch composition, continue with
[Analyze and choose an NTT backend](choose-ntt-backend.md).
For a final production choice, also follow
[Benchmark methodology](/benchmarks/methodology).
