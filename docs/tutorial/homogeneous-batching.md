# Homogeneous batching

**Example source:** [`examples/06_eager_batching.py`](https://github.com/VisualDust/fhelium/blob/main/examples/06_eager_batching.py)

This example compares one homogeneous CKKS batch with a per-message loop over the same packed matrix-vector product, evaluated with `fhelium.algorithms.linear`. The tutorial explains leading batch dimensions and keeps the latency-memory execution choice in application code.

## Run the example

Start with the 8,192-slot, 40-bit-scale baseline:

```bash
python examples/06_eager_batching.py \
  --preset slots8192-scale40-depth7 \
  --depth 0 \
  --batch-sizes 1,4,8 \
  --warmup 2 \
  --runs 10
```

Then compare the two ends of the 32,768-slot, 40-bit-scale chain:

```bash
python examples/06_eager_batching.py \
  --preset slots32768-scale40-depth34 --depth 0 --batch-sizes 1,4,8

python examples/06_eager_batching.py \
  --preset slots32768-scale40-depth34 --depth 30 --batch-sizes 1,4,8
```

The second command measures the smaller active RNS row count at depth 30, which changes both arithmetic work and the size of each NTT/key-switch working set.

## 1. A leading prefix is a value batch

The example constructs `B` independent vectors:

```python
vectors.shape == (B, size)
messages = vectors.repeat(1, engine.num_slots // size)
source = engine.encrypt_message(messages, depth=depth)
assert tuple(source.batch_shape) == (B,)
```

For a ciphertext, the complete data layout is conceptually:

```text
[component, *batch, limb, coefficient]
```

For an RNS plaintext it is:

```text
[*batch, limb, coefficient_or_ntt_index]
```

The leading dimensions index independent messages in the batch. All members of one homogeneous value share its depth, scale, polynomial domain, modulus basis, device, dtype, and component count.

They must also have the same CKKS parameter provenance and effective encryption-key lineage. Runtime values encode neither relation, so the engine cannot infer that independently produced ciphertexts are safe to stack. This example encrypts the complete message batch with one engine/key. When assembling existing ciphertexts, key-switch them when necessary before calling `Ciphertext.stack_batch`.

## 2. The evaluator is batch-polymorphic

The example converts the `size`-by-`size` matrix into cyclic diagonals of a periodically packed vector. With rotation $\operatorname{Rot}_k$ following `torch.roll(..., shifts=k)`, slot $i$ of $\operatorname{Rot}_k(x)$ holds $x_{(i-k) \bmod n}$, so the diagonals $d_k[i]=A_{i \bmod n,\,(i-k) \bmod n}$ give

$$
y=\sum_{k=0}^{n-1} d_k\odot\operatorname{Rot}_k(x)=Ax
$$

in every period of $n$ slots. `fhelium.algorithms.linear` evaluates this sum with a baby-step giant-step schedule, and the example creates the rotation keys that the schedule reports:

```python
transform = dense_transform(matrix, engine.num_slots)
keys = EvaluationKeySet(
    engine.create_rotation_key(step, secret_key)
    for step in linear.plan(transform).required_rotation_offsets(transform)
)
```

`linear.prepare` encodes the diagonals and selects keys once. Diagonal encoding draws stochastic-rounding samples, so the example prepares from one unbatched member and uses the same preparation for both paths:

```python
prepared = linear.prepare(engine, individual_sources[0], transform, keys=keys)
```

`linear.apply(engine, value, prepared=prepared)` contains no batch-specific branch. Rotations, plaintext products, accumulation, and the final rescale broadcast over the leading batch dimensions, so one public matrix is applied to every encrypted vector, and `batch=1` uses the same evaluator.

## Local batch and limb views

RNS ciphertexts and plaintexts share the same trailing axes. A ciphertext adds a component axis; compressed plaintext stores fewer encoded entries:

```text
Ciphertext:          [component, *batch, limb, N]
RNS Plaintext:       [           *batch, limb, N]
CompressedPlaintext: [           *batch, limb, unique]
  implicit_data:    [           *batch, limb]  (when present)
```

All three value types expose the same local batch and limb operations:

| Operation | Data selection | State and storage |
| --- | --- | --- |
| `slice_batch(start, stop, dim=0)` | A nonempty interval on a logical batch axis; the axis is retained | Shared storage; prime IDs and arithmetic state unchanged |
| `select_batch(index, dim=0)` | One batch item; the selected axis is removed | Shared storage; prime IDs and arithmetic state unchanged |
| `unbind_batch(dim=0)` | All individual items along a batch axis | Shared-storage views; prime IDs and arithmetic state unchanged |
| `stack_batch(values)` | Insert a leading batch axis | New storage; input arithmetic states must match |
| `slice_limbs(start, stop)` | An interval of stored RNS rows | Shared storage; select the same interval from `prime_ids`; depth and scale unchanged |

`dim` indexes logical batch axes, excluding component, limb, and encoded-entry axes, and accepts negative dimensions. Slice intervals use nonnegative, nonempty half-open bounds. `limb_count` reports the number of represented RNS rows. Plaintext slots and ordinary coefficient representations have batch axes but no limb axis: their `limb_count` is zero and `slice_limbs` is inapplicable.

```python
ciphertext_chunk = source.slice_batch(start, stop)
plaintext_chunk = batched_plaintext.slice_batch(start, stop)
ciphertext_rows = ciphertext_chunk.slice_limbs(row_start, row_stop)
plaintext_rows = plaintext_chunk.slice_limbs(row_start, row_stop)
```

Each limb interval indexes the source's stored row positions, not global prime IDs. Slicing data and its declared row IDs together preserves that relationship; it does not establish parameter or key provenance. The caller remains responsible for supplying compatible operands and numerical parameters.

Compressed plaintext operations slice both `data` and any `implicit_data` along their corresponding logical axes, preserving the compression format. Batch and limb slices compose in either order and share source storage. A limb slice is not rescale or a depth transition: operations requiring a complete active basis still need all their required rows, whereas row-local arithmetic and NTT can operate on matching intervals.

## 3. Choose batch or loop execution

The example evaluates equivalent logical work in two ways.

Batched submission:

```python
batched_result = linear.apply(engine, source, prepared=prepared)
```

Per-message loop:

```python
individual_sources = tuple(
    value.clone() for value in source.unbind_batch()
)
looped_result = [
    linear.apply(engine, value, prepared=prepared)
    for value in individual_sources
]
```

`unbind_batch` makes the logical members visible. The example clones the views so the loop represents independently owned request values. It stacks the loop results with `Ciphertext.stack_batch` and compares them with the batched result by `torch.equal`; the `batch == loop` column reports whether the two ciphertexts are bit-for-bit identical. Two separate preparations would encode the diagonals with different rounding samples, and the results would then agree only after decryption. `stack_batch` allocates and copies the loop results into one batched value.

Batch-versus-loop selection is a programmer or workload-scheduler decision based on deployment measurements, latency requirements, and memory budget.

## 4. Compare the complete workload

The matrix-vector product composes:

- shared plaintext diagonals;
- baby-step and giant-step rotations with key switching;
- plaintext multiplication;
- ciphertext accumulation;
- rescale.

It is more informative than timing only one elementwise operator. Before timing, the example also decrypts the batched result and compares it with `vectors @ matrix.T`.

## 5. Read both latency and memory columns

For every requested `B`, the example reports:

- the aggregate size of one extended QP digit;
- synchronized batched and loop medians;
- `loop / batch` speedup;
- the faster path at that measured point;
- incremental peak allocated CUDA memory when running on CUDA (`n/a` on CPU);
- bit-for-bit agreement between the batched and loop ciphertexts;
- maximum absolute decryption error.

A ratio above one means the batch won. It does not imply that a larger batch will continue to scale. A batch can reduce launches and expose parallelism while simultaneously enlarging active NTT, automorphism, accumulator, and output tensors beyond the effective cache working set.

The `faster` column describes only the current command. The example does not write an automatic recommendation into engine configuration or cache a hidden device policy.

## 6. Select a policy from the deployed point

Use these rules as a measurement plan, not as hard-coded library behavior:

1. Compare batched ciphertext data bit-for-bit with a per-message loop from the same installed build.
2. Compare `[1, slots]` with `[slots]` to isolate B1 overhead.
3. Compare B4/B8 with a per-message loop over the same members at the same preset and depth.
4. Repeat at the depths used by the real evaluator.
5. Measure the complete workload and peak memory, not only an NTT kernel.
6. Keep the resulting choice in application or scheduler code.

The worked RTX PRO 6000 measurements and the working-set explanation are in [Choose a homogeneous batch size](../how-to/choose-homogeneous-batch-size.md). The stable mechanism is summarized in the [CKKS workload cost model](../concepts/performance/cost-model.md#homogeneous-batching-has-a-working-set-crossover).

::: details Source
<<< @/../examples/06_eager_batching.py
:::

## Related concepts and guides

- [Values and state](../api/fhelium/values/ciphertext.md)
- [Linear layer and polynomial activation](linear-layer-and-polynomial.md)
- [CKKS workload cost model](../concepts/performance/cost-model.md)
- [Choose a homogeneous batch size](../how-to/choose-homogeneous-batch-size.md)
- [Benchmark methodology](/benchmarks/methodology)
- [CUDA Graph matrix-vector](compiled-cuda-graphs.md)
