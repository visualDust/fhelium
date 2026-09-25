# Rotation hoisting

**Example source:** [`examples/06_eager_rotation_hoisting.py`](https://github.com/VisualDust/fhelium/blob/main/examples/06_eager_rotation_hoisting.py)

This example benchmarks caller-selected independent rotations against one caller-selected hoisted group over the same source and direct keys. Rotation hoisting shares decomposition and preparation derived from one ciphertext; it is a scheduling choice rather than an implementation selected invisibly by a backend.

## Rotation API

The method name identifies both cardinality and how the operation selects key material:

| Method | Rotation selector | Key ownership | Result |
| --- | --- | --- | --- |
| `rotate_by_step` | One signed step | Engine inventory | One ciphertext |
| `rotate_with_key` | One self-described key | Caller | One ciphertext |
| `rotate_many_by_steps` | Ordered signed steps | Engine inventory | Ordered ciphertexts |
| `rotate_many_with_keys` | Ordered self-described keys | Caller | Ordered ciphertexts |

The step-based methods may use installed keys, generate direct keys when allowed, or compose an available engine-owned key path. The key-based methods use exactly the supplied direct key objects and do not install them.

Both sequence methods accept `use_hoisting`. `True` forms one scheduled group from the direct keys in the supplied sequence. `False` executes those direct rotations independently. Eager execution preserves the caller's offsets and order; it does not search surrounding operations or regroup them.

## Run the benchmark

```bash
python examples/06_eager_rotation_hoisting.py \
  --preset slots32768-scale40-depth34-int64 \
  --counts 4,8,16 \
  --warmup 5 \
  --runs 20
```

Start with fewer runs when checking a new environment:

```bash
python examples/06_eager_rotation_hoisting.py \
  --preset slots8192-scale40-depth7-int64 \
  --counts 2,4 \
  --warmup 1 \
  --runs 3
```

## 1. Provision every direct key before timing

```python
for rotation_step in rotation_steps_all:
    _ = engine.rotation_key(rotation_step)
```

Lazy key creation must not appear in a rotation timing. The benchmark creates the public key and all requested rotation keys before warmup.

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
engine.rotate_many_by_steps(
    ciphertext,
    rotation_steps,
    use_hoisting=True,
)
```

Both request the same set of rotated ciphertexts. The sequence-form API gives the caller a way to select one hoisted group containing all direct requested steps. Passing `use_hoisting=False` selects independent execution through the same Eager API.

## Compile-time scheduling

Compile IR represents independent rotations as `fhelium_ckks.rotate` operations. `RotationHoistingPass` identifies rotations that read the same ciphertext and preparation tables, then emits `fhelium_ckks.hoisted_rotate_many` groups. The pattern is a shared preparation in the data-dependency graph, rather than adjacent instructions:

$$
y_i = F(c_0, P(c_1;\theta), k_i;\phi_i)
\quad\longrightarrow\quad
h=P(c_1;\theta),\qquad y_i=F(c_0,h,k_i;\phi_i).
$$

Here $P$ prepares the key-switch digits, $\theta$ identifies its numerical parameters and tables, and $F$ completes each rotation using its key $k_i$ and rotation/output attributes $\phi_i$. Pure consumers and rotations of other inputs may appear between members of a group. The existing multi-result operation requires a common output domain and compatible execution attributes.

The pass chooses a position after all computed operands are available and before any grouped result is consumed. It can move operand-free material references within the same pure interval, but does not move their producer computations. When no common position exists, rotations remain in separate groups. Effects, unknown operations, control flow, and caller-assigned implementations delimit intervals. Known nested regions are visited separately; preparation is not moved across branches or loop iterations.

The shared operands are live Tensor dataflow. A possible write ends an interval, so grouping does not assume that repeated reads of one Tensor observe unchanged contents across mutations. The pass does not compare Tensor contents or generate key material. `max_group_size` bounds the number of outputs produced together. Grouping may extend output lifetimes.

Shared preparation can change key-switch rounding and encryption noise relative to independent rotations. The decoded rotation and value state are preserved; ciphertext residues need not be identical across those schedules. Numerical comparisons between schedules therefore use the decoded-error criterion, while comparisons of the same schedule can additionally check residues modulo Q.

The backend consumes the selected groups without adding offsets, splitting them, or falling back to independent rotation. A caller-selected scheduling pass can instead emit the same operation using a workload-specific memory policy.

## 3. What can be shared

A rotation applies a Galois automorphism and a key switch. When many rotations use the same source ciphertext, decomposition and extension work derived from that source can be prepared once and reused across direct rotation keys.

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

The result still contains one ciphertext per requested step. Hoisting reduces repeated preparation; it does not remove the per-key automorphism/key-switch work or output memory. The currently selected group therefore remains visible in the Program rather than existing only inside a native implementation.

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

It also synchronizes CUDA around each timing interval, performs warmup, and reports mean and median. This reduces bias from asynchronous launches, one-time kernel setup, and temperature drift.

## 5. Interpret the result

The benchmark reports:

- one-by-one mean and median;
- grouped mean and median;
- speedup ratio;
- percentage mean-time saving.

Hoisting tends to become more valuable as the number of rotations from one source increases. Actual benefit depends on active RNS rows, decomposition shape, backend, GPU, key residency, and whether the surrounding algorithm can consume all produced rotations.

::: details Source
<<< @/../examples/06_eager_rotation_hoisting.py
:::

## Related concepts and guides

- [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
- [CKKS workload cost model](../concepts/performance/cost-model.md)
- [Benchmark methodology](/benchmarks/methodology)
