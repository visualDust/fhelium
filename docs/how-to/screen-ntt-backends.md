# Screen NTT backends on the target GPU

FHElium provides several mathematically equivalent Number Theoretic Transform
(NTT) implementations. Their relative latency depends on the GPU, ring
dimension, active Residue Number System (RNS) rows, and surrounding CKKS
operation. Use the default for a first evaluator. Screen alternatives only
after a correct, representative workload exists on the target GPU.

Backend choice is a `CkksEngine` execution option. It does not change
CKKS parameters, ciphertext compatibility, or the library default.

The library default is one static backend name for every supported `logN` and
GPU. FHElium does not inspect the device or benchmark during engine
construction. The commands below only produce evidence for a
application choice.

## The three-step workflow

1. **Screen kernels** to find obviously slow candidates.
2. **Confirm CKKS primitives** to include encryption, key switching, and
   rotation behavior.
3. **Apply one backend name** only when the evidence supports a choice.

The following real example shows the kind of result the command produces.

![Two NTT recommendation suites on an RTX PRO 6000 Blackwell GPU](/figures/ntt-backend-recommendation-example-sm120.svg)

*Figure 1. Example `logN = 16` measurements recorded on 2026-07-23 with a FHElium
0.10 development build and PyTorch 2.13/CUDA 13 on one NVIDIA RTX PRO 6000
Blackwell (sm_120) GPU. Bars show aggregate latency gap from the numerical
winner; lower is better. Orange marks the stable group8 fallback. These values
illustrate the decision process and are not portable performance claims.*

## 1. Run the kernel screening suite

```bash
fhelium benchmark recommend ntt \
  --suite kernel \
  --preset slots32768-scale40-levels34-int64 \
  --device cuda:0 \
  --output results/ntt-kernel.json
```

The command tests every compatible production backend using forward NTT,
inverse NTT, and roundtrip latency. A shortened view of the example output is:

| Rank | Backend | Pick | Gap from best | Repetition wins |
| ---: | --- | :---: | ---: | ---: |
| 1 | `radix2_compact_group8_smem8` | yes | 0.00% | 2 / 3 |
| 2 | `radix16_compact` | no | 0.47% | 1 / 3 |
| 3 | `radix2_compact_group16_smem8` | no | 1.20% | 0 / 3 |
| 4 | `radix2_compact_group4_smem8` | no | 5.34% | 0 / 3 |
| 5 | `radix4_compact` | no | 9.12% | 0 / 3 |

```text
recommended_backend  radix2_compact_group8_smem8
confidence           low
reason               0.47% runner-up margin; winner changed across repetitions
```

### Judgment from this result

Do **not** claim that group8 is universally faster. The leading two backends
are separated by less than 1%, and the repetition winner changes. The useful
conclusion is narrower: group4 and radix4 are poor candidates for this
kernel case, while group8 and radix16 require higher-level confirmation.

## 2. Confirm with CKKS primitives

```bash
fhelium benchmark recommend ntt \
  --suite ckks-primitive \
  --preset slots32768-scale40-levels34-int64 \
  --device cuda:0 \
  --output results/ntt-ckks-primitives.json
```

This suite creates keys before timing, validates decrypted results, and
measures:

- `encrypt_message` and `decrypt_message`;
- multiplication followed by relinearization;
- one keyed rotation;
- four rotations through grouped hoisting.

In Figure 1, group16 and radix16 have nearly identical aggregate scores, while
group8 remains only 2.01% behind. All three are inside the 3% near-tie band, so
the command selects the stable group8 fallback and reports **low confidence**.
That is an actionable result: keep the current default choice rather
than encoding a noisy machine-specific winner.

A different result should lead to a different action:

| Observation | Action |
| --- | --- |
| Same backend wins all repetitions by at least 5%, with low variation | Select that backend, then validate the application workload. |
| Winner leads by 3–5% | Treat it as medium-confidence evidence; repeat under production conditions. |
| Gap is below 3%, or repetition winners disagree | Keep the stable fallback; record the result as a near tie. |
| Kernel and primitive suites disagree | Prefer primitives for a general CKKS evaluator, then benchmark the real workload. |

## 3. Apply the recommendation

The CLI prints a constructor expression. Copy the backend name:

```python
import fhelium as fh

engine = fh.CkksEngine(
    fh.Preset.slots32768_scale40_levels34_int64,
    device="cuda:0",
    ntt_backend="radix2_compact_group8_smem8",
)
```

The recommendation command does not rewrite `DEFAULT_NTT_BACKEND`, modify
native shared-memory tuning, or cache a machine-global selection. Re-run it
when the GPU model, CUDA/PyTorch stack, FHElium version, preset, or important
workload changes.

## Compare a focused candidate set

Repeat `--backend` to restrict a follow-up comparison:

```bash
fhelium benchmark recommend ntt \
  --suite ckks-primitive \
  --preset slots32768-scale40-levels34-int64 \
  --backend radix2_compact_group8_smem8 \
  --backend radix16_compact
```

At least two compatible names are required. The indexed radix-2
correctness oracle is omitted by default because it is not a production
candidate, but it remains available for diagnostic runs.

## Understand why performance changes

The two suites intentionally answer different questions. To analyze launch
count, memory coalescing, shared-memory fusion, register pressure, occupancy,
RNS row count, and key-switch composition, continue with
[Analyze and choose an NTT backend](choose-ntt-backend.md).
For a final production choice, also follow
[Benchmark a workload correctly](benchmark-a-workload.md).
