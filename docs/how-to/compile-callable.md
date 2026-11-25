# Compile and reuse a callable

`fc.compile` creates a reusable function interface over Python capture, Program transformation, linking and execution. The first call with a new input specialization prepares it and then executes it; compatible calls reuse the resident executable. Execution stays on the prepared route by default; CUDA Graph replay is opt-in.

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
print(y)
```

For encrypted inputs, wrap the Eager operations that express their CKKS transitions:

```python
@fc.compile
def encrypted_sum(left, right):
    return engine.add(left, right)

result = encrypted_sum(cx, cy)
```

The second block uses the `engine`, `cx`, and `cy` from the CKKS guide. Ordinary Tensor inputs retain their public numerical role; a ciphertext argument does not encrypt the other arguments. Capture retains the actual fixed Tensor operands, including supplied key data. Prepare keys before capture and keep randomized encryption outside the repeated evaluator.

Python callable instances and bound methods use the same interface. Object configuration is read during capture; Tensor members become live materials when used. Capture rejects assignment to bound object attributes. Clear or recreate the compiled callable after changing captured Python configuration.

```python
class Affine:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return x * self.weight

operation = Affine(torch.ones_like(x))
compiled_operation = fc.compile(operation)
object_result = compiled_operation(x)
```

Prepared library computations follow this model as well: `prepared = bootstrap.prepare(ciphertext, evaluation_keys=keys)` supplies the complete refresh callable, and `fc.compile(prepared)` expands its numerical body through the ordinary pipeline. Prepare encoded constants and keys before capture.

## 2. Separate preparation from execution

```python
prepared = affine.prepare(x, gain=2.0)
print(len(affine.specializations))
print(prepared.compilation.program.to_text())
y2 = affine(x + 1, gain=2.0)
print(y2)
```

`prepare` captures or loads the Program, transforms it when needed, binds materials, links execution and warms supported CUDA kernels without executing it. A call that misses performs the same work before executing, and preparation errors propagate to the caller.

Changing Tensor contents while preserving the input conditions reuses the specialization. An immutable scalar such as `gain` participates in specialization, so a new value can require preparation. Record first-call latency, which includes preparation, separately from warmed execution. When enabled, automatic Graph capture follows the first prepared execution; `.prepare()` alone does not capture a Graph.

For a serving application that must reject unprepared conditions, construct with `on_miss="error"` and prepare the admitted inputs during setup:

```python
strict_affine = fc.compile(affine.reference, on_miss="error")
strict_affine.prepare(x, gain=2.0)
strict_result = strict_affine(x, gain=2.0)
```

## 3. Choose cache and output ownership

Reuse proceeds from a matching resident executable to an eligible optimized Program for the input specialization; another input specialization, such as a new depth, is captured and prepared. The directory is `FHELIUM_CACHE_DIR` when set, otherwise `$XDG_CACHE_HOME/fhelium/compile` or `~/.cache/fhelium/compile`. Pass `cache=False` to disable disk persistence, or a path to choose the directory:

```python
local = fc.compile(affine.reference, cache=".compile-cache")
transient = fc.compile(affine.reference, cache=False)
```

`cache=False` disables disk persistence and retains resident specializations.

Reuse without capture requires a supported ordinary Python function and capture-time configuration, the default pipeline, Backend and workspace, and no supplied `inputs`, `resources` or `keys` overrides. Bound methods, callable objects and unsupported source patterns are captured in each process; an identical captured Program reuses the persisted optimized Program and linked executable. Custom compilation configurations use ordinary compilation. Identity includes source/configuration, execution-environment facts and the input conditions relevant to each reuse level. Material Tensor contents are excluded. Reuse binds the current process’s materials; compatible fresh key data can use a Program when the declared key layout is unchanged. The caller supplies keys with the required cryptographic relation.


`graph=False` is the default and retains prepared execution. Set `graph="auto"` or `graph=True` to enable automatic capture: a CUDA specialization first executes its prepared function, then attempts capture on its next call. Graphs share a device-local pool within the callable. CPU inputs, autograd outputs and capture failures retain prepared execution, with the reason in `stats()["graphs"]`.

```python
owned = fc.compile(affine.reference, graph="auto", copy_output=True)
borrowed = fc.compile(affine.reference, graph="auto", copy_output=False)
```

`copy_output=True` returns owned Graph outputs and includes their copy cost in execution. With `False`, a Graph result borrows retained storage until that Graph replays or the callable is cleared. Consume or clone it before either event. These options affect Graph results; ordinary prepared results retain their normal Tensor ownership.

Call `compiled.clear()` after changing captured Python constants. Clearing closes Graphs before their pools, and releases resident specializations and statistics. Disk entries remain available. Keep captured material owners alive until preparation and execution have finished.

## 4. Select a pipeline or Backend when needed

```python
from fhelium.backend import OperationBackend

backend = OperationBackend()
recipe = fc.default_lower_and_fuse_pipeline(backend)
controlled = fc.compile(affine.reference, backend=backend, pipeline=recipe)
controlled.prepare(x)
```

An omitted pipeline uses the default lowering-and-fusion recipe. A supplied `Pipeline` replaces the whole recipe. A function from `CallSignature` to `Pipeline` can choose a recipe for each signature. Inspect `recipe.names` and compose passes with `before`, `after`, `replace`, or `then`. For CKKS arithmetic that needs additional rescale or relinearization scheduling, insert the corresponding passes before depth and scale assignment; the default recipe preserves the source transitions.

`compiled.with_backend(other_backend)` creates a callable with the other Backend and a snapshot of transformed variants for fresh linking. Clear a callable after changing captured Python constants. Construct a new callable after changing its source Program or pass policy. Binding dictionaries are copied at callable construction, but their Tensor storage remains shared with the caller.

## 5. Inspect preparation and execution

Compare public results with the original function and encrypted results with a clear-message oracle. Inspect the specialization's Program, Backend, and pass reports when a timing or representation changes. `examples/12_compile_jit.py` supplies CKKS weighted-product and matrix workloads with first-call and warmed measurements. [Diagnose Compile preparation](diagnose-compile-preparation.md) separates capture, transformation, binding, and execution failures; [Prepared host execution](../developer/prepared-host-execution.md) explains specialization and linking.


```python
print(affine.stats())
for variant in affine.specializations:
    for report in variant.compilation.reports:
        print(report.name, report.seconds, report.stats)
```

Statistics separate preparation-stage host costs from execution counts and timings. Inspect resident hits, optimized-instance cache status, Graph state and fallback reasons before interpreting latency. `PassReport.seconds` records each pass’s host duration. [Prepared host execution](../developer/prepared-host-execution.md) defines timing scope.

## Prepare plaintext materials separately

Mixed callables evaluate Tensor packing and plaintext encoding in a preparation Program. Ciphertext operations consume its outputs through typed material references. The default resident mode reuses prepared materials while the source Tensor objects and their PyTorch version counters remain unchanged. Compatible replacement weights or ordinary in-place weight updates execute the preparation Program again and rebind materials without repeating ciphertext lowering, implementation selection, or kernel preparation. CUDA Graphs associated with the previous material pointers are released.

Preparation records proven Tensor views as an owner material plus shape, strides, and an element offset. Persistent recipes retain these descriptors; a weight refresh reconstructs the views from the new owner allocation. Triton kernels share the owner pointer and compute view addresses without copying ciphertext banks. `sum_plaintext_products` is captured as one weighted-sum operation whose structure declares the ordered term axis. A batched plaintext remains one operand whose leading axis holds the terms. When the plaintexts share one layout, a Triton region evaluates the terms in a runtime loop in their original accumulation order and advances through the plaintext rows by their stride. Compile lowers and fuses every captured operation, so a BSGS baby-step accumulation written as a Python loop of `multiply_plaintext` and `add` produces a Program proportional to its term count; expressing each group with `sum_plaintext_products` keeps one operation per group and reduces pass and kernel compilation work.

The workspace options `binding_time_split`, `merge_plaintext_encodings`, and `plaintext_preparation_mode` control this behavior. Their defaults are `True`, `True`, and `"resident"`. The encoding merge requires the same source leaves, packing and slice expressions, encoding parameters, and represented plaintext state. It shares one rounding result across equivalent uses. Set `merge_plaintext_encodings=False` to preserve independent encoding events. Select `plaintext_preparation_mode="per_call"` to execute preparation on every invocation and release its material bindings after execution; this mode uses prepared execution without CUDA Graph capture.

`compiled.refresh_materials(*args, **kwargs)` forces preparation after changes that Tensor version counters do not expose. Inference tensors have no version counter and are refreshed conservatively. Preparation statistics report execution and rebinding time, material bytes before and after encoding merging, resident bytes, and the current binding generation. Persistent entries contain both Programs and source locators, without numerical weight or encoded-material payloads. A preparation recipe belongs to one input specialization; a changed input state prepares another specialization.

Preparation supports ordered writes to newly allocated local buffers. Plaintext memoization may store real Tensor or plaintext values in ordinary dictionaries; the preparation Program refreshes those values with its material generation. Captured ciphertext/proxy values cannot escape into bound dictionaries. Recipes retaining such live memoization targets remain in-process and report their persistent-cache restriction.
