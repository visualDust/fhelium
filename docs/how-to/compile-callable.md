# Compile and reuse a callable

`fc.compile` creates a callable that captures a Python calculation or accepts an existing Program, prepares an input specialization, and reuses its linked executable for compatible calls. Use this path when the application wants an ordinary function interface and inspectable preparation controls.

## Prerequisites

Have a working calculation and representative inputs. For CKKS, prepare the configuration, ciphertexts, and required keys as in [Evaluate CKKS data](evaluate-ckks-data.md). For a calculation described directly as IR, start with [Build and transform a Program](build-program-pipeline.md); that Program is also accepted by `fc.compile`.

## 1. Wrap the calculation

This self-contained CPU example compiles ordinary public Tensor arithmetic:

```python
import torch
from fhelium import compile as fc

@fc.compile
def affine(x: torch.Tensor, gain: float = 2.0) -> torch.Tensor:
    return x * gain + x

x = torch.arange(8, dtype=torch.float64, device="cpu")
y = affine(x)
torch.testing.assert_close(y, 3 * x)
```

For encrypted inputs, wrap the Eager operations that express their CKKS transitions:

```python
@fc.compile
def encrypted_sum(left, right):
    return engine.add(left, right)

result = encrypted_sum(cx, cy)
```

The second block uses the `engine`, `cx`, and `cy` from the CKKS guide. Ordinary Tensor inputs retain their public numerical role; a ciphertext argument does not encrypt the other arguments. Capture retains the actual fixed Tensor operands, including supplied key data. Prepare keys before capture and keep randomized encryption outside the repeated evaluator.

## 2. Separate preparation from execution

```python
prepared = affine.prepare(x, gain=2.0)
print(len(affine.specializations))
print(prepared.compilation.program.to_text())
y2 = affine(x + 1, gain=2.0)
torch.testing.assert_close(y2, 3 * (x + 1))
```

Changing Tensor contents while preserving the input conditions reuses the specialization. An immutable scalar such as `gain` participates in specialization, so a new value can require preparation. Record first-call latency separately from warmed execution; Backend device-code compilation may still occur during the first execution after `prepare`.

For a serving application that must reject unprepared conditions, construct with `on_miss="error"` and prepare the admitted inputs during setup:

```python
strict_affine = fc.compile(affine.reference, on_miss="error")
strict_affine.prepare(x, gain=2.0)
strict_result = strict_affine(x, gain=2.0)
```

## 3. Select a pipeline or Backend when needed

```python
from fhelium.backend import OperationBackend

backend = OperationBackend()
recipe = fc.default_lower_and_fuse_pipeline(backend)
controlled = fc.compile(affine.reference, backend=backend, pipeline=recipe)
controlled.prepare(x)
```

An omitted pipeline uses `default_lower_and_fuse_pipeline`; a supplied `Pipeline` replaces the whole recipe. A function from `CallSignature` to `Pipeline` can choose a recipe for each signature. Inspect `recipe.names` and compose passes with `before`, `after`, `replace`, or `then`. The default recipe preserves represented CKKS transitions and does not insert rescaling or relinearization; add those scheduling passes when the source calculation requires them.

`compiled.with_backend(other_backend)` creates a callable with the other Backend and fresh linked executables. Clear a callable after changing captured Python constants. Construct a new callable after changing its source Program or pass policy. Binding dictionaries are copied at callable construction, but their Tensor storage remains shared with the caller.

## 4. Verify the observable result

Compare public results with the original function and encrypted results with a clear-message oracle. Inspect the specialization's Program, Backend, and pass reports when a timing or representation changes. `examples/11_compile_jit.py` supplies CKKS weighted-product and matrix workloads with first-call and warmed measurements. [Diagnose Compile preparation](diagnose-compile-preparation.md) separates capture, transformation, binding, and execution failures; [Prepared host execution](../developer/prepared-host-execution.md) explains specialization and linking.
