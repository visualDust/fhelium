# Choose a preset and chain depth

Choose a CKKS parameter set from the circuit's slot capacity, rescale schedule, precision, range, security budget, and memory requirements. Validate the resulting parameter set against a cleartext oracle before optimizing its execution.

## Prerequisites

Write the clear-message calculation and define output precision and input magnitude requirements before selecting parameters. The same configuration planning applies to Eager operations and to CKKS schedules represented in a Program.

## Read a Preset name

Preset names have the form

```text
slots{capacity}_scale{target}_depth{maximum}
```

For example, `slots16384_scale40_depth16` resolves to a configuration with 16,384 complex slots, terminal default scale $2^{40}$, and public depths `0..16`.

The scale field describes $\log_2$ of `config.scale_at(config.max_depth)`; the default scales of earlier depths are derived from the rescale ratios and stay within about one bit of it, except that depth zero may hold a larger scale when its first rescale ratio exceeds the target. The depth field is `config.max_depth`; it is already the number of public rescale transitions available from depth zero. Every built-in preset uses primes below $2^{30}$, so the Engine selects `torch.int32` residues.

[Scale, depth, and native RNS dispatch](../concepts/ckks/scale-depth-and-execution-format.md) traces how these quantities interact.

## Built-in parameter baselines

Every Preset resolves to prime values in `q_moduli` and `p_moduli` and to the per-depth bases `depth_bases`. The table lists the Q and P row counts, the complete QP product, and the built-in security budget.

| Python member | `logN` | slots | scale bits | max depth | Q / P rows | QP bits / budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Preset.slots8192_scale30_depth9` | 14 | 8,192 | 30 | 9 | 11 / 3 | 410 / 430 |
| `Preset.slots8192_scale40_depth7` | 14 | 8,192 | 40 | 7 | 12 / 3 | 430 / 430 |
| `Preset.slots8192_scale50_depth5` | 14 | 8,192 | 50 | 5 | 11 / 3 | 400 / 430 |
| `Preset.slots16384_scale30_depth22` | 15 | 16,384 | 30 | 22 | 24 / 5 | 860 / 868 |
| `Preset.slots16384_scale40_depth16` | 15 | 16,384 | 40 | 16 | 24 / 5 | 850 / 868 |
| `Preset.slots16384_scale50_depth11` | 15 | 16,384 | 50 | 11 | 21 / 5 | 760 / 868 |
| `Preset.slots32768_scale30_depth46` | 16 | 32,768 | 30 | 46 | 48 / 10 | 1,727 / 1,747 |
| `Preset.slots32768_scale40_depth34` | 16 | 32,768 | 40 | 34 | 48 / 10 | 1,717 / 1,747 |
| `Preset.slots32768_scale50_depth27` | 16 | 32,768 | 50 | 27 | 48 / 10 | 1,708 / 1,747 |
| `Preset.slots65536_scale30_depth100` | 17 | 65,536 | 30 | 100 | 102 / 16 | 3,503 / 3,523 |
| `Preset.slots65536_scale40_depth75` | 17 | 65,536 | 40 | 75 | 103 / 16 | 3,514 / 3,523 |
| `Preset.slots65536_scale50_depth55` | 17 | 65,536 | 50 | 55 | 95 / 16 | 3,269 / 3,523 |

Preset recipes are reviewed fixed baselines. `CkksConfig.parse(preset)` resolves a recipe to prime values. To define another parameter set, construct a `CkksConfig` from `q_moduli`, `depth_bases`, `scales`, and `p_moduli`, or build a suffix chain from ordered prime groups with `CkksConfig.suffix_chain`. `CkksConfig.catalog_suffix_chain(logN=..., scale_bits=..., depth=..., p_count=...)` builds a suffix chain with one catalog prime of `scale_bits` bits per depth; primes of 30 or more bits select `torch.int64` residues.

## 1. Specify the cleartext workload

Record:

- logical input and output shapes;
- slot packing, padding, masks, and replicated regions;
- ciphertext-ciphertext and ciphertext-plaintext multiplications;
- summation structure;
- rotations and conjugations;
- input and intermediate magnitude bounds;
- output error requirements.

Implement a cleartext oracle with the same packing and rotation convention.

## 2. Determine slot capacity

A ring with `logN = k` provides

$$
N=2^k,\qquad S=N/2
$$

complex slots. Count physical packed slots, including padding and intermediate layouts. Select the smallest candidate ring that accommodates them and satisfies the security budget.

## 3. Draw the depth and scale schedule

For each value, annotate:

```text
operation
depth before
actual scale before
Q primes dropped and introduced, if any
depth after
actual scale after
```

Count `rescale_to_next_depth` calls on every execution path. Addition preserves depth. Multiplication preserves depth and multiplies actual scales. One rescale moves from the basis of depth $d$ to the basis of depth $d+1$ and divides the actual scale by the rescale ratio $\rho_d$.

Verify that the largest depth reached is at most `config.max_depth`. Use `config.transition_ratio(depth, depth + 1)` or `engine.rescale_output_scale(input_scale=..., depth=...)` to calculate each scale transition, and `config.scale_at(depth)` for the default scale of each depth, rather than assuming a prime width from the Preset name.

## 4. Check precision and range

Evaluate fractional error and integer headroom together. Exercise:

- representative and maximum input amplitudes;
- positive and negative values;
- wide sums and multiplication chains;
- several random seeds;
- early, middle, and final depths used by the circuit.

Record the observed error distribution under the intended output criterion. Do not alter a tolerance to conceal an unexplained discrepancy.

## 5. Check memory and key requirements

At each important depth, record the active Q-row count and value size. Evaluation-key storage also depends on the QP basis, hybrid digit decomposition, and required rotation inventory. Measure peak temporary storage for the actual operation schedule.

## 6. Preserve parameter identity

Store or report:

- Preset, when one was used;
- `CkksConfig.dumps()` output or its Q primes, depth bases, scales, and P primes;
- default scales of the depths used and actual scales;
- `max_depth` and checkpoint depths;
- active `prime_ids` at a failure;
- Engine RNS dtype and target device;
- error, amplitude, and random-seed evidence.

## 7. Optimize against the same oracle

Keep a clear-message oracle and one correctness-qualified execution path as the comparison point. Introduce hoisting, retained NTT state, CUDA Graph capture, streaming, or distributed execution one mechanism at a time. Preserve the parameter set and cleartext criterion unless the experiment is specifically a parameter tradeoff.

## Verify the outcome

The selected configuration must fit the physical packing, every rescale transition, the full QP security budget, and the measured output error requirement. Keep the resulting configuration and clear-message oracle with the evaluator, then [prepare CKKS data](evaluate-ckks-data.md) or [build its Program](build-program-pipeline.md).

## Related documentation

- [Scale, depth, and native RNS dispatch](../concepts/ckks/scale-depth-and-execution-format.md)
- [Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md)
- [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md)
- [Modulus-chain tutorial](../tutorial/modulus-chain-depth.md)
