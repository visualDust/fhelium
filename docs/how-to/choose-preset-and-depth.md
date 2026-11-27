# Choose a preset and chain depth

Choose a CKKS parameter set from the circuit's slot capacity, rescale schedule, precision, range, security budget, and memory requirements, then validate it against a cleartext oracle before optimizing execution.

## Prerequisites

Write the clear-message calculation and define its output precision and input magnitude requirements. The same planning applies to Eager operations and to CKKS schedules in a compiled Program.

## Read a preset name

Preset values have the form

```text
slots{capacity}-scale{bits}-depth{maximum}
slots{capacity}-scale{bits}-depth{maximum}-boot{k}
```

and the Python members replace hyphens with underscores. `slots16384-scale40-depth16` (`fh.Preset.slots16384_scale40_depth16`) resolves to 16,384 complex slots, maximum-depth default scale $2^{40}$, and depths `0..16`.

- The scale field is $\log_2$ of `config.scale_at(config.max_depth)`. The default scales of earlier depths are derived from the rescale ratios and stay within about one bit of it; depth zero may hold a larger scale when its first rescale ratio exceeds the target.
- The depth field is `config.max_depth`, the number of one-depth transitions available from depth zero.
- A `-boot{k}` suffix marks a chain for full-slot bootstrapping: its first 16 depths form the segment consumed by the default `"3/38"` profile of `fhelium.bootstrap.plan`, leaving `k` application depths after each refresh. These presets use Galois generator 5, which full-slot bootstrapping requires, and Q primes of about 25, 28, and 30 bits; their key layouts take auxiliary primes from P and from Q primes inactive at the source depth.

The residue dtype follows the prime widths: the Engine selects `torch.int32` residues when every configured prime is below $2^{30}$ and `torch.int64` otherwise. Every built-in preset uses primes below $2^{30}$ and therefore runs with `torch.int32`; `engine.dtype` reports the selection.

## Built-in presets

Every preset resolves to prime values in `q_moduli` and `p_moduli` and to the per-depth bases `q_bases`. The table lists the Q and P prime counts, the bit length of the complete QP product, and the security budget for the ring dimension.

| Python member | `logN` | slots | scale bits | max depth | Q / P primes | QP bits / budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Preset.slots8192_scale30_depth9` | 14 | 8,192 | 30 | 9 | 11 / 4 | 430 / 430 |
| `Preset.slots8192_scale40_depth7` | 14 | 8,192 | 40 | 7 | 12 / 3 | 430 / 430 |
| `Preset.slots8192_scale50_depth5` | 14 | 8,192 | 50 | 5 | 11 / 4 | 430 / 430 |
| `Preset.slots16384_scale30_depth22` | 15 | 16,384 | 30 | 22 | 24 / 5 | 860 / 868 |
| `Preset.slots16384_scale40_depth16` | 15 | 16,384 | 40 | 16 | 24 / 6 | 866 / 868 |
| `Preset.slots16384_scale50_depth11` | 15 | 16,384 | 50 | 11 | 21 / 6 | 790 / 868 |
| `Preset.slots32768_scale30_depth46` | 16 | 32,768 | 30 | 46 | 48 / 11 | 1,747 / 1,747 |
| `Preset.slots32768_scale40_depth34` | 16 | 32,768 | 40 | 34 | 48 / 11 | 1,747 / 1,747 |
| `Preset.slots32768_scale50_depth27` | 16 | 32,768 | 50 | 27 | 48 / 11 | 1,738 / 1,747 |
| `Preset.slots65536_scale30_depth100` | 17 | 65,536 | 30 | 100 | 102 / 17 | 3,523 / 3,523 |
| `Preset.slots65536_scale40_depth75` | 17 | 65,536 | 40 | 75 | 103 / 16 | 3,514 / 3,523 |
| `Preset.slots65536_scale50_depth55` | 17 | 65,536 | 50 | 55 | 95 / 17 | 3,298 / 3,523 |
| `Preset.slots32768_scale40_depth20_boot4` | 16 | 32,768 | 40 | 20 | 38 / 22 | 1,747 / 1,747 |
| `Preset.slots32768_scale40_depth24_boot8` | 16 | 32,768 | 40 | 24 | 44 / 16 | 1,731 / 1,747 |
| `Preset.slots32768_scale40_depth29_boot13` | 16 | 32,768 | 40 | 29 | 50 / 10 | 1,747 / 1,747 |
| `Preset.slots32768_scale50_depth24_boot8` | 16 | 32,768 | 50 | 24 | 47 / 13 | 1,731 / 1,747 |
| `Preset.slots32768_scale50_depth28_boot12` | 16 | 32,768 | 50 | 28 | 53 / 7 | 1,747 / 1,747 |

Among the `-boot{k}` presets of one scale, a larger `k` gives more application depths per refresh and leaves fewer primes for P, which changes key-switch decomposition and key size. `CkksConfig.parse(preset)` resolves a preset and accepts keyword overrides for a derived configuration. To define another parameter set, construct a `CkksConfig` from `q_moduli`, `q_bases`, `scales`, and `p_moduli`, build a suffix chain from ordered prime groups with `CkksConfig.suffix_chain`, or use `CkksConfig.catalog_suffix_chain(logN=..., scale_bits=..., depth=..., p_count=...)` for one catalog prime of `scale_bits` bits per depth; primes of 30 or more bits select `torch.int64` residues.

Example 03 prints the per-depth bases, default scales, and rescale ratios of a preset:

```bash
python examples/03_eager_depth_and_scale.py --preset slots16384-scale40-depth16
```

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

A ring with `logN = k` provides $S=2^{k-1}$ complex slots. Count physical packed slots, including padding and intermediate layouts, and select the smallest ring that accommodates them under the security budget.

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

Count `rescale_to_next_depth` calls on every execution path, including the depths consumed by algorithm calls such as `fhelium.algorithms.linear.apply` and `fhelium.algorithms.polynomial.evaluate`, whose plans report `required_depths`. Addition preserves depth; multiplication preserves depth and multiplies actual scales; one rescale moves to the next depth's basis and divides the actual scale by the rescale ratio $\rho_d$.

Verify that the largest depth reached is at most `config.max_depth`; for a bootstrapped circuit, that each segment between refreshes fits the `k` application depths. Calculate scale transitions with `config.rescale_divisor(depth)` or `engine.rescale_output_scale(input_scale, depth=...)` and default scales with `config.scale_at(depth)`, rather than assuming a prime width from the preset name.

## 4. Check precision and range

Evaluate fractional error and integer headroom together. Exercise:

- representative and maximum input amplitudes;
- positive and negative values;
- wide sums and multiplication chains;
- several random seeds;
- early, middle, and final depths used by the circuit.

Record the observed error distribution under the intended output criterion.

## 5. Check memory and key requirements

At each important depth, record the active Q-row count and value size. Evaluation-key storage depends on the QP basis, hybrid digit decomposition, and required rotation inventory; see [Provision evaluation keys](provision-keyset.md). Measure peak temporary storage for the actual operation schedule.

## 6. Preserve parameter identity

Store or report:

- the preset, when one was used;
- `CkksConfig.dumps()` output, or its Q primes, Q bases, scales, and P primes;
- default scales of the depths used and the actual scales;
- `max_depth` and checkpoint depths;
- active `prime_ids` at a failure;
- Engine residue dtype and target device;
- error, amplitude, and random-seed evidence.

## 7. Optimize against the same oracle

Keep the clear-message oracle and one correctness-qualified execution path as the comparison point. Introduce hoisting, retained NTT state, compilation, streaming, or distributed execution one mechanism at a time, and keep the parameter set and cleartext criterion unless the experiment is a parameter tradeoff.

## Verify the outcome

The selected configuration fits the physical packing, every rescale transition, the full QP security budget, and the measured output error requirement. Keep the configuration and clear-message oracle with the evaluator, then [evaluate it eagerly](../tutorial/basic-ckks-workflow.md) or [compile it](compile-callable.md).

## Related documentation

- [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md)
- [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
- [Depth and scale tutorial](../tutorial/depth-and-scale.md)
- [Composable CKKS bootstrapping](../concepts/ckks/composable-bootstrapping.md)
