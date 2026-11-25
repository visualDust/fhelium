# Screen NTT backends on the target GPU

FHElium provides several mathematically equivalent Number Theoretic Transform (NTT) implementations. Their relative latency depends on the GPU, ring dimension, number of active modulus primes, and surrounding CKKS operation. Use the default for a first evaluator. Compare alternatives only after a correct, representative workload exists on the target GPU.

The recommendation runner times NTT context transforms and `fhelium.eager.Engine` CKKS methods. Confirm a selected backend with the production workload before deployment.

Apply an NTT policy through the Eager Engine option or Compile implementation-selection passes. It changes transform execution while preserving the mathematical NTT and CKKS parameter set; see [operation implementation selection](select-operation-implementation.md).

The commands below produce a recommendation for an application-selected NTT policy. Apply that policy through the [implementation controls](select-operation-implementation.md) of the execution path being measured.

## What the two commands measure

An **NTT backend** is one implementation of the same forward and inverse Number Theoretic Transform. Changing the backend changes how the GPU executes the transform; it does not change the CKKS parameters or mathematical result.

The CLI calls each fixed list of measured operations a `suite`:

- `--suite kernel` measures prepared forward NTT, inverse NTT, and an NTT round trip. Despite the CLI name, this test does not report one CUDA kernel launch. It reports complete NTT method calls while excluding engine and input construction. This page calls it the **NTT-only test**.
- `--suite ckks-primitive` measures complete `fhelium.eager.Engine` method calls: encryption, decryption, multiplication followed by relinearization (reducing a three-component multiplication result to two components), one rotation, and four rotations that reuse one ciphertext decomposition (grouped hoisting). This page calls it the **complete-CKKS test**.

The first test answers “which transform implementations are clearly slower?” The second answers “which backend gives the best result after NTT work is combined with the other work inside common CKKS operations?”

## Prerequisites and interpretation

Use a CUDA-capable build, a supported preset, and an otherwise stable target device. Keep the software environment, warmup, repetitions, active rows, and measurement scope fixed. Preserve invoking CPU thread settings unless an override is supplied and record effective intra-op/inter-op counts.

The command ranks an equal-weight geometric mean of per-operation median latency ratios and treats a gap of at most 3% as a near tie. This percentage describes the selected suite, not application latency. [Interpret NTT backend performance](choose-ntt-backend.md#understand-the-score-before-interpreting-it) defines the score and confidence conditions used to interpret the reports.

## Measure in three steps

1. **Time NTT operations only** and remove clearly slower implementations.
2. **Time composed CKKS operations** in the retained diagnostic evaluator and compare the remaining backends after encryption, key-switch, and rotation work is included.
3. **Measure the application** before fixing the deployment choice.

### 1. Time NTT operations only

```bash
fhelium benchmark recommend ntt \
  --suite kernel \
  --preset slots32768-scale40-depth34 \
  --device cuda:0 \
  --output results/ntt-kernel.json
```

The command tests every compatible production backend using forward NTT, inverse NTT, and round-trip latency. Engine construction and input preparation are outside the timed region.

### 2. Time complete CKKS operations

```bash
fhelium benchmark recommend ntt \
  --suite ckks-primitive \
  --preset slots32768-scale40-depth34 \
  --device cuda:0 \
  --output results/ntt-ckks-primitives.json
```

This command creates keys before timing, validates decrypted results, and measures:

- `encrypt_message` and `decrypt_message`;
- multiplication followed by relinearization;
- one keyed rotation;
- four rotations through grouped hoisting.

Key generation and correctness checking remain outside the timed region.

### 3. Confirm the application

The complete-CKKS comparison gives every measured operation equal weight. A production workload does not. Confirm the remaining candidates with representative depths, batch sizes, numbers of relinearizations and rotations, and whether the application uses ordinary eager calls or captured CUDA Graph replay. Preserve the two reports with the application measurement so that the scope of each decision remains visible.

## Worked example

This example screens the compatible GPU policies for `slots32768-scale40-depth34`, whose ring dimension is $2^{16}$, on an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition with a 300 W power limit (compute capability 12.0, also written `sm_120`). It uses 3 warmups, 10 timed runs per operation, 3 repetitions, and seed `20260823`; each repetition starts with a different backend to reduce measurement-order bias.

| Backend | NTT-only relative latency | Complete-CKKS relative latency | Meaning for this example |
| --- | ---: | ---: | --- |
| `radix2_compact` | pending | pending | pending |
| `radix2_compact_shoup32` | pending | pending | pending |
| `radix4_compact` | pending | pending | pending |
| `radix16_compact` | pending | pending | pending |

The relative latencies of the current policies are pending re-measurement. The first test keeps every candidate within 3% of the NTT-only winner; the second test selects the complete-CKKS winner and retains the current default when it is within 3% of that winner. The command reports the median within-run **coefficient of variation**, the sample standard deviation divided by the sample mean of each timed operation, alongside the confidence label.

## Apply the measured choice

The CLI prints a constructor expression. Copy the backend name:

```python
import fhelium as fh
import torch
from fhelium.eager import Engine

torch.set_default_device("cuda:0")
engine = Engine(
    fh.Preset.slots32768_scale40_depth34,
    ntt_backend="radix2_compact",
)
```

Re-run the recommendation when the GPU model, CUDA/PyTorch stack, FHElium version, preset, or workload changes. Confirm the recommended policy in the application benchmark before applying it to that workload.

## Compare a focused candidate set

Repeat `--backend` to restrict a follow-up comparison:

```bash
fhelium benchmark recommend ntt \
  --suite ckks-primitive \
  --preset slots32768-scale40-depth34 \
  --backend radix2_compact \
  --backend radix16_compact
```

At least two compatible names are required. The command omits `radix2_indexed` by default because the diagnostic variant serves controlled experiments. Add that backend by name only when a diagnostic comparison needs the indexed implementation as another result-equality reference.

## Understand why performance changes

The NTT-only and complete-CKKS tests measure different execution scopes. To analyze launch count, memory coalescing, shared-memory fusion, register pressure, occupancy, RNS row count, and key-switch composition, continue with [Analyze and choose an NTT backend](choose-ntt-backend.md). For a final production choice, also follow [Benchmark methodology](/benchmarks/methodology).

## Verify the outcome

Retain correctness-qualified candidates from the screening reports and validate the selected policy on the current production execution path. Record both the diagnostic evaluator identity and the application result; historical reports retain their original measurements and scope.
