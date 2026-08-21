# Homogeneous batching

**Example source:** [`examples/15_homogeneous_batching.py`](https://github.com/VisualDust/fhelium/blob/main/examples/15_homogeneous_batching.py)

This example compares one homogeneous CKKS batch with an explicit loop over
the same packed matrix-vector workload. The tutorial explains leading batch
dimensions and keeps the latency-memory execution choice in application code.

## Run the example

Start with the 8,192-slot, 40-bit-scale baseline:

```bash
python examples/15_homogeneous_batching.py \
  --preset slots8192-scale40-levels7-int64 \
  --level 0 \
  --batch-sizes 1,4,8 \
  --warmup 2 \
  --runs 10
```

Then compare the two ends of the 32,768-slot, 40-bit-scale chain:

```bash
python examples/15_homogeneous_batching.py \
  --preset slots32768-scale40-levels34-int64 --level 0 --batch-sizes 1,4,8

python examples/15_homogeneous_batching.py \
  --preset slots32768-scale40-levels34-int64 --level 30 --batch-sizes 1,4,8
```

The second command is not redundant. Level changes the active RNS row count,
which changes both arithmetic work and the size of each NTT/key-switch working
set.

## 1. A leading prefix is a value batch

The example constructs `B` independent vectors:

```python
vectors.shape == (B, size)
messages = vectors.repeat(1, engine.num_slots // size)
source = engine.encrypt_message(messages, level=level)
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

The leading dimensions are semantic message dimensions. They are not RNS
limbs, polynomial components, distributed ranks, or hybrid-decomposition
digits. All members of one homogeneous value share its context, level, scale,
polynomial domain, modulus basis, device, dtype, and component count.

They must also have the same effective encryption-key lineage. A context id
describes parameters, not a particular secret key, so the engine cannot infer
that independently produced ciphertexts are safe to stack. This example
encrypts the complete message batch with one engine/key. When assembling
existing ciphertexts, key-switch them when necessary before calling
`Ciphertext.stack_batch`.

The matrix diagonals are stacked along a new term axis inside the evaluator.
Any pre-existing ciphertext batch axes remain inner dimensions, so one public
matrix is applied to every encrypted vector.

## 2. The evaluator is batch-polymorphic

`matrix_vector` contains no batch-specific branch:

```python
def matrix_vector(source, *, engine, diagonals, rotation_keys):
    rotated_values = []
    for step in range(len(diagonals)):
        rotated = (
            source
            if step == 0
            else engine.rotate_with_key(source, rotation_keys[step])
        )
        rotated_values.append(rotated)
    rotated_ntt = engine.coefficient_domain_to_ntt_domain(
        Ciphertext.stack_batch(rotated_values)
    )
    weighted = engine.multiply_plaintext(
        rotated_ntt, Plaintext.stack_batch(diagonals)
    )
    return engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.sum_ciphertext_batch(weighted)
        )
    )
```

The same function accepts either an unbatched ciphertext or one with a
non-empty batch prefix. `batch=1` therefore uses the same public program rather
than a separate compatibility path. The outer term axis batches the forward
NTT, plaintext products, and binary-tree reduction, avoiding one under-filled
native launch per matrix diagonal.

## 3. Choose batch or loop execution

The example evaluates equivalent logical work in two ways.

Batched submission:

```python
batched_result = matrix_vector(source, ...)
```

Explicit loop:

```python
individual_sources = tuple(
    value.clone() for value in source.unbind_batch()
)
looped_result = [matrix_vector(value, ...) for value in individual_sources]
```

`unbind_batch` makes the logical members visible. The example clones the views
so the loop represents independently owned request values. It then uses
`Ciphertext.stack_batch(looped_result)` only to verify bit-for-bit equivalence:

```python
torch.testing.assert_close(
    batched_result.data,
    Ciphertext.stack_batch(looped_result).data,
    rtol=0,
    atol=0,
)
```

`stack_batch` allocates and copies; it is not a hidden performance shortcut.

Batch-versus-loop selection is a programmer or workload-scheduler decision
based on deployment measurements, latency requirements, and memory budget.

## 4. Compare the complete workload

The example uses the cyclic-diagonal formulation of an 8-by-8 matrix-vector
product:

$$
y=\sum_s p_s\odot\operatorname{Rot}(x,s).
$$

This workload composes:

- shared plaintext broadcasting;
- rotations and direct key switching;
- plaintext multiplication;
- rescale;
- ciphertext accumulation.

It is more informative than timing only one elementwise operator. Before
timing, the example also decrypts the batched result and compares it with
`vectors @ matrix.T`.

## 5. Read both latency and memory columns

For every requested `B`, the example reports:

- the aggregate size of one extended QP digit;
- synchronized batched and loop medians;
- `loop / batch` speedup;
- the faster path at that measured point;
- incremental peak allocated CUDA memory when running on CUDA (`n/a` on CPU);
- maximum absolute decryption error.

A ratio above one means the batch won. It does not imply that a larger batch
will continue to scale. A batch can reduce launches and expose parallelism
while simultaneously enlarging active NTT, automorphism, accumulator, and
output tensors beyond the effective cache working set.

The `faster` column describes only the current command. The example does not
write an automatic recommendation into engine configuration or cache a hidden
device policy.

## 6. Select a policy from the deployed point

Use these rules as a measurement plan, not as hard-coded library behavior:

1. Verify the batched result exactly against an explicit loop from the same
   installed build.
2. Compare `[1, slots]` with `[slots]` to isolate B1 overhead.
3. Compare B4/B8 with an explicit loop over the same members at the same preset
   and level.
4. Repeat at the levels used by the real evaluator.
5. Measure the complete workload and peak memory, not only an NTT kernel.
6. Keep the resulting choice in application or scheduler code.

The worked RTX PRO 6000 measurements and the working-set explanation are in
[Choose a homogeneous batch size](../how-to/choose-homogeneous-batch-size.md).
The stable mechanism is summarized in the
[CKKS workload cost model](../concepts/performance/cost-model.md#homogeneous-batching-has-a-working-set-crossover).

::: details Complete runnable source
<<< @/../examples/15_homogeneous_batching.py
:::

## Related concepts and guides

- [Values and state](../api/fhelium/core/ciphertext.md)
- [CKKS workload cost model](../concepts/performance/cost-model.md)
- [Choose a homogeneous batch size](../how-to/choose-homogeneous-batch-size.md)
- [Benchmark a workload correctly](../how-to/benchmark-a-workload.md)
- [CUDA Graph matrix-vector](cuda-graph-matvec.md)
