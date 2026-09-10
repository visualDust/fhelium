# Choose a preset and chain depth

Choose a CKKS parameter set from the circuit's slot capacity, rescale schedule,
precision, range, security budget, and memory requirements. Validate the
resulting exact configuration against a cleartext oracle before optimizing its
execution.

## Read a Preset name

Preset names have the form

```text
slots{capacity}_scale{target}_depth{maximum}_{default_rns_dtype}
```

For example, `slots32768_scale50_depth27_int64` resolves to a configuration
with 32,768 complex slots, default scale $2^{50}$, public depths `0..27`, and
primes that select `torch.int64` as the Engine's default RNS dtype.

The scale field describes `log2(config.default_scale)`. The depth field is
`config.max_depth`; it is already the number of public rescale transitions
available from depth zero. The dtype field reports the default native residue
format implied by the preset's exact primes.

[Scale, depth, and native RNS dispatch](../concepts/ckks/scale-depth-and-execution-format.md)
traces how these quantities interact.

## Built-in parameter baselines

Every Preset resolves to exact `q_depth_groups` and `p_moduli`. The table lists
the current complete QP product and built-in security budget.

| Python member | `logN` | slots | default scale bits | max depth | Q / P rows | QP bits / budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Preset.slots8192_scale25_depth14_int32` | 14 | 8,192 | 25 | 14 | 15 / 1 | 404 / 430 |
| `Preset.slots8192_scale30_depth9_int64` | 14 | 8,192 | 30 | 9 | 10 / 1 | 360 / 430 |
| `Preset.slots8192_scale40_depth7_int64` | 14 | 8,192 | 40 | 7 | 8 / 1 | 381 / 430 |
| `Preset.slots8192_scale50_depth5_int64` | 14 | 8,192 | 50 | 5 | 6 / 1 | 361 / 430 |
| `Preset.slots16384_scale25_depth29_int32` | 15 | 16,384 | 25 | 29 | 30 / 2 | 812 / 868 |
| `Preset.slots16384_scale30_depth21_int64` | 15 | 16,384 | 30 | 21 | 22 / 2 | 781 / 868 |
| `Preset.slots16384_scale40_depth16_int64` | 15 | 16,384 | 40 | 16 | 17 / 2 | 800 / 868 |
| `Preset.slots16384_scale50_depth12_int64` | 15 | 16,384 | 50 | 12 | 13 / 2 | 770 / 868 |
| `Preset.slots32768_scale25_depth24_int32` | 16 | 32,768 | 25 | 24 | 25 / 4 | 740 / 1,747 |
| `Preset.slots32768_scale30_depth45_int64` | 16 | 32,768 | 30 | 45 | 46 / 4 | 1,620 / 1,747 |
| `Preset.slots32768_scale40_depth34_int64` | 16 | 32,768 | 40 | 34 | 35 / 4 | 1,640 / 1,747 |
| `Preset.slots32768_scale50_depth27_int64` | 16 | 32,768 | 50 | 27 | 28 / 4 | 1,640 / 1,747 |
| `Preset.slots32768_scale50_depth29_int64` | 16 | 32,768 | 50 | 29 | 30 / 4 | 1,740 / 1,747 |
| `Preset.slots65536_scale25_depth14_int32` | 17 | 65,536 | 25 | 14 | 15 / 6 | 543 / 3,523 |
| `Preset.slots65536_scale30_depth95_int64` | 17 | 65,536 | 30 | 95 | 96 / 6 | 3,278 / 3,523 |
| `Preset.slots65536_scale40_depth72_int64` | 17 | 65,536 | 40 | 72 | 73 / 6 | 3,280 / 3,523 |
| `Preset.slots65536_scale50_depth58_int64` | 17 | 65,536 | 50 | 58 | 59 / 6 | 3,310 / 3,523 |

Preset recipes are reviewed fixed baselines. `CkksConfig.parse(preset)` resolves
a recipe to exact primes. Expert parameter work constructs a new `CkksConfig`
with exact nested Q groups and exact P primes rather than changing a Preset by
count.

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

complex slots. Count physical packed slots, including padding and intermediate
layouts. Select the smallest candidate ring that accommodates them and satisfies
the security budget.

## 3. Draw the depth and scale schedule

For each value, annotate:

```text
operation
depth before
actual scale before
Q group removed, if any
depth after
actual scale after
```

Count `rescale_to_next_depth` calls on every execution path. Addition preserves
depth. Multiplication preserves depth and multiplies actual scales. One rescale
consumes one complete Q depth group.

Verify that the largest depth reached is at most `config.max_depth`. Use
`config.rescale_divisor(depth)` to calculate each scale transition rather than
assuming a prime width from the Preset name.

## 4. Check precision and range

Evaluate fractional error and integer headroom together. Exercise:

- representative and maximum input amplitudes;
- positive and negative values;
- wide sums and multiplication chains;
- several random seeds;
- early, middle, and final depths used by the circuit.

Record the observed error distribution under the intended output criterion.
Do not alter a tolerance to conceal an unexplained discrepancy.

## 5. Check memory and key requirements

At each important depth, record the active Q-row count and value size.
Evaluation-key storage also depends on the QP basis, hybrid digit decomposition,
and required rotation inventory. Measure peak temporary storage for the actual
operation schedule.

## 6. Preserve the exact parameter identity

Store or report:

- Preset, when one was used;
- `CkksConfig.dumps()` output or its exact Q groups and P primes;
- default and actual scales;
- `max_depth` and checkpoint depths;
- active `prime_ids` at a failure;
- Engine RNS dtype and target device;
- error, amplitude, and random-seed evidence.

## 7. Optimize against the same oracle

Keep the simplest correct Eager execution as the comparison point. Introduce
hoisting, retained NTT state, CUDA Graph capture, streaming, or distributed
execution one mechanism at a time. Preserve the parameter set and cleartext
criterion unless the experiment is specifically a parameter tradeoff.

## Related documentation

- [Scale, depth, and native RNS dispatch](../concepts/ckks/scale-depth-and-execution-format.md)
- [Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md)
- [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md)
- [Modulus-chain tutorial](../tutorial/modulus-chain-depth.md)
