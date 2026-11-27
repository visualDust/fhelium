# Compile a layer

**Example source:** [`examples/12_compile_layer.py`](https://github.com/VisualDust/fhelium/blob/main/examples/12_compile_layer.py)

`@fc.compile` returns a `CompiledCallable` over the same Program, passes, and Backend as manual compilation. The first call with a new input specialization captures the function, transforms and links it, and executes it; later compatible calls reuse the resident specialization. On CUDA, the callable captures CUDA Graphs of its schedule's capture-safe segments at the second execution of a specialization and keeps each Graph only where replay is faster (`graph="auto"`, the default).

Example 12 compiles one encrypted dense layer

$$
y=g\,(Wx)^2,
$$

with a plaintext matrix $W$, an encrypted vector $x$, and a Python `float` gain $g$. It uses `slots8192_scale40_depth7` ($N=16384$, 8192 slots, eight Q rows at depth 0, int64 residues). Key generation, encoding of $W$, encryption, and decryption stay outside the compiled function.

## Run the example

```bash
python examples/12_compile_layer.py
python examples/12_compile_layer.py --device cpu --size 8 --runs 5
```

The default device is `cuda:0` when CUDA is available. The example prints the depth, scale, and clear-message error of each layer and gain, then the specialization counts, preparation seconds, steady call latency, cache status, and the CUDA Graph route of each specialization.

## A layer as a class

```python
from fhelium.algorithms import linear

def dense_transform(weights, slots):
    size = weights.size(0)
    rows = torch.arange(slots) % size
    return linear.LinearTransform(
        {-k: weights[rows, (rows + k) % size] for k in range(size)}, slots, name="dense")

class EncryptedLayer:
    def __init__(self, engine, transform, like, keys, relinearization_key):
        self.engine = engine
        self.relinearization_key = relinearization_key
        self.prepared = linear.prepare(engine, like, transform, keys=keys)

    @fc.compile
    def activate(self, value):
        ntt = self.engine.coefficient_domain_to_ntt_domain(value)
        return self.engine.relinearize(
            self.engine.multiply(ntt, ntt), self.relinearization_key, rescale=True)

    @fc.compile
    def forward(self, x, gain: float = 1.0):
        hidden = linear.apply(self.engine, x, prepared=self.prepared)
        return self.engine.multiply_constant(self.activate(hidden), gain)
```

The vector $x$ of length $n$ is tiled with period $n$ across all slots. The `LinearTransform` stores diagonals $d_{-k}[j]=W[j \bmod n,(j+k)\bmod n]$ for $k=0,\dots,n-1$, keyed by the `torch.roll` shift that aligns them:

$$
Wx=\sum_{k=0}^{n-1} d_{-k}\odot\operatorname{roll}(x,-k).
$$

`linear.plan(transform)` selects a baby-step/giant-step schedule, and `plan.required_rotation_offsets(transform)` lists the rotation keys it needs; the example creates them in one `EvaluationKeySet`. `linear.prepare` encodes the diagonals once for inputs with the depth, scale, and device of `like`; inside the compiled method, `linear.apply(..., prepared=...)` reads those plaintexts as materials and consumes one depth. The square activation relinearizes and rescales, and `multiply_constant` rescales once more, so the output is a two-component ciphertext at depth 3.

## Methods, nested calls, and scalar arguments

A compiled method keeps one `CompiledCallable` per instance, with its own specializations; the two layers in the example each report two specializations. Capture reads instance attributes such as `self.prepared` as live material bindings; assignments to them inside the method are rejected.

`forward` calls the compiled `activate`. A compiled helper called during capture is inlined from its Python source and compiled under the options of the outer callable, so `forward` has one Program and one schedule. Calling `layer.activate(...)` directly uses its own specializations.

Python scalars are static parameters. Each `gain` value selects its own specialization; a quantity that changes from call to call should be a Tensor or value argument. Specialization records value kind, shape, strides, dtype, device, CKKS state, and static scalar arguments; Tensor contents and storage addresses are not part of it, so a new ciphertext with the same state reuses the specialization.

## Preparation, reuse, and statistics

```python
stats = layer.forward.stats()
stats["prepare_seconds"], stats["resident_cache_hits"], stats["executions"]
stats["cache"]      # source-cache eligibility
stats["graphs"]     # one record per specialization and execution context
```

`prepare_seconds` sums capture, pipeline, linking, binding, and kernel warm-up over all specializations. `cache` reports why the source cannot be found in the persistent cache before capture; in this example the source of `fhelium.algorithms.linear` contains a `from ... import` statement, which is outside the statically checked subset, so later processes find the entry by its captured Program instead, which still skips the pass pipeline. Each record in `graphs` has a route `state` (`"captured"` after the capture, `"warming"` before it), its replay count, a fallback reason, and per-segment records whose `choice` is `"graph"`, `"direct"`, or `"measuring"`.

`layer.forward.prepare(x, 1.0)` prepares a specialization without executing it, and `clear()` releases the resident specializations and their Graphs. Steady latency excludes the first two calls of each specialization: the first runs the prepared schedule, the second captures its Graph segments.

## Inspect a specialization

```python
variant = layer.forward.specializations[0]
variant.signature
variant.compilation.program            # transformed Program
for report in variant.compilation.reports:
    print(report.name, report.seconds, report.stats)
```

[Example 13](compose-and-execute-compile-pipeline.md) runs the same compiler stages with a caller-selected pipeline. [Choose compile boundaries](../how-to/choose-compile-boundaries.md) explains how function size, loops, scalars, and runtime decisions affect specializations, and [Compile a callable](../how-to/compile-callable.md) lists the options of `fc.compile`.

::: details Source
<<< @/../examples/12_compile_layer.py
:::
