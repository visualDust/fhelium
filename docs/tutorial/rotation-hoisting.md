# Rotation hoisting

**Example source:** [`examples/07_rotation_hoisting_benchmark.py`](https://github.com/VisualDust/fhelium/blob/main/examples/07_rotation_hoisting_benchmark.py)

This example benchmarks independent rotations against a grouped request over
the same source and exact keys. The tutorial explains which decomposition and
preparation work can be shared and how to interpret the timings.

## Rotation API

The method name identifies both cardinality and how the operation selects key
material:

| Method | Rotation selector | Key ownership | Result |
| --- | --- | --- | --- |
| `rotate_by_step` | One signed step | Engine inventory | One ciphertext |
| `rotate_with_key` | One self-described key | Caller | One ciphertext |
| `rotate_many_by_steps` | Ordered signed steps | Engine inventory | Ordered ciphertexts |
| `rotate_many_with_keys` | Ordered self-described keys | Caller | Ordered ciphertexts |

The step-based methods may use installed keys, generate direct keys when
allowed, or compose an available engine-owned key path. The key-based methods
use exactly the supplied direct key objects and do not install them.

## Run the benchmark

```bash
python examples/07_rotation_hoisting_benchmark.py \
  --preset slots32768-scale40-levels34-int64 \
  --counts 4,8,16 \
  --warmup 5 \
  --runs 20
```

Start with fewer runs when checking a new environment:

```bash
python examples/07_rotation_hoisting_benchmark.py \
  --preset slots8192-scale40-levels7-int64 \
  --counts 2,4 \
  --warmup 1 \
  --runs 3
```

## 1. Provision every exact key before timing

```python
for rotation_step in rotation_steps_all:
    _ = engine.rotation_key(rotation_step)
```

Lazy key creation must not appear in a rotation timing. The benchmark creates
the public key and all requested rotation keys before warmup.

## 2. Compare equivalent outputs

Independent path:

```python
[
    engine.rotate_by_step(ciphertext, rotation_step)
    for rotation_step in rotation_steps
]
```

Grouped path:

```python
engine.rotate_many_by_steps(ciphertext, rotation_steps)
```

Both request the same set of rotated ciphertexts. The sequence-form API gives
the engine a direct opportunity to hoist input-dependent work shared by
all requested steps.

## 3. What can be shared

A rotation applies a Galois automorphism and a key switch. When many rotations
use the same source ciphertext, decomposition and extension work derived from
that source can be prepared once and reused across exact rotation keys.

Conceptually:

```mermaid
flowchart LR
    subgraph independent["Independent"]
        x1["x"] --> prepare1["prepare(x)"] --> apply1["apply(k1)"]
        x2["x"] --> prepare2["prepare(x)"] --> apply2["apply(k2)"]
        x3["x"] --> prepare3["prepare(x)"] --> apply3["apply(k3)"]
    end
    subgraph hoisted["Hoisted"]
        xh["x"] --> prepared["prepared = prepare(x)"]
        prepared --> applyh1["apply(prepared, k1)"]
        prepared --> applyh2["apply(prepared, k2)"]
        prepared --> applyh3["apply(prepared, k3)"]
    end
```

The result still contains one ciphertext per requested step. Hoisting reduces
repeated preparation; it does not remove the per-key automorphism/key-switch
work or output memory.

## 4. Benchmark without launch-order bias

The example alternates which path runs first on each measured iteration:

```python
if run_idx % 2 == 0:
    independent()
    hoisted()
else:
    hoisted()
    independent()
```

It also synchronizes CUDA around each timing interval, performs warmup, and
reports mean and median. This reduces bias from asynchronous launches,
one-time kernel setup, and temperature drift.

## 5. Interpret the result

The benchmark reports:

- one-by-one mean and median;
- grouped mean and median;
- speedup ratio;
- percentage mean-time saving.

Hoisting tends to become more valuable as the number of rotations from one
source increases. Actual benefit depends on active RNS rows, decomposition
shape, backend, GPU, key residency, and whether the surrounding algorithm can
consume all produced rotations.

::: warning Do not mix unrelated optimizations into a scaling comparison
When comparing devices, ranks, or partition strategies, keep hoisting, NTT
backend, key residency, warmup, and synchronization policy fixed. A faster
result is otherwise not attributable to one variable.
:::

::: details Complete runnable source
<<< @/../examples/07_rotation_hoisting_benchmark.py
:::

## Related concepts and guides

- [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
- [CKKS workload cost model](../concepts/performance/cost-model.md)
- [Benchmark a workload correctly](../how-to/benchmark-a-workload.md)
