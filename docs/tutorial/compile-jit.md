# JIT compilation

The JIT usage model captures and prepares a function when its inputs are first seen, caches the resulting specialization, and reuses it for matching calls. FHElium provides this model through `fhelium.compile`: `@fc.compile` returns a `CompiledCallable` using the same Program, passes, and Backend as manual compilation.

Example 11 compares two encrypted workloads with their original Eager functions:

| Workload | Computation | Default size |
| --- | --- | --- |
| 1. CKKS weighted product | $z_i=2x_i y_i w_i$, with encrypted $x,y$ and plaintext $w$ | 8192 slots, one ciphertext per input |
| 2. CKKS matrix multiplication | $Y=AX$, with plaintext $A$ and encrypted $X$ | $A:128\times128$, $X:128\times8$, BSGS baby step 8 |

Both use `slots8192_scale40_depth7_int64`: $N=16384$, eight active Q rows at entry depth 0, and int64 polynomial storage. The functions contain the encrypted computation; key generation, input encoding, encryption and output decryption remain outside them.

## Set up the weighted product

Create an Engine and prepare the ciphertexts and plaintext weight. Capture records the numerical Tensor operands used by the supported calls. An Engine provides data and the Eager interface; the executable does not retain it as an execution binding.

```python
import torch
from fhelium import Preset, compile as fc
from fhelium.config import CkksConfig
from fhelium.eager import Engine
from fhelium.values import Ciphertext, Plaintext

config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
device = torch.device("cpu")
engine = Engine(config, rng_seed=24)
secret_key = engine.create_secret_key(device=device)
engine.set_secret_key(secret_key)
public_key = engine.create_public_key(secret_key, device=device)
clear_x = torch.linspace(-0.1, 0.1, config.num_slots, dtype=torch.float64, device=device)
clear_y = torch.linspace(0.03, 0.07, config.num_slots, dtype=torch.float64, device=device)
clear_weight = torch.linspace(0.25, 0.5, config.num_slots, dtype=torch.float64, device=device)
x = engine.encrypt_message(clear_x, public_key, device=device)
y = engine.encrypt_message(clear_y, public_key, device=device)
weight = engine.prepare_plaintext_for_multiplication(
    engine.encode(clear_weight, device=device)
)
```

## Decorate and execute

```python
@fc.compile
def weighted_product(
    lhs: Ciphertext,
    rhs: Ciphertext,
    weight: Plaintext,
    negative: bool = False,
) -> Ciphertext:
    a = engine.coefficient_domain_to_ntt_domain(lhs)
    b = engine.coefficient_domain_to_ntt_domain(rhs)
    product = engine.multiply(a, b)
    weighted = engine.multiply_plaintext(product, weight)
    doubled = engine.subtract(weighted, engine.negate(weighted))
    if negative:
        doubled = engine.negate(doubled)
    return engine.ntt_domain_to_coefficient_domain(doubled)

compiled = weighted_product
result = compiled(x, y, weight)
reference = compiled.reference
assert reference is not None
torch.testing.assert_close(
    result.data, reference(x, y, weight).data, rtol=0, atol=0
)
decoded = engine.decrypt_message(result, secret_key)
print("Maximum clear-message error:",
      float((decoded - 2 * clear_x * clear_y * clear_weight).abs().max()))
```

The two input ciphertexts have payload shape `[2, 8, 16384]`; the plaintext weight has shape `[8, 16384]`. Ciphertext multiplication produces three components. This function inserts neither relinearization nor rescale, so its output has shape `[3, 8, 16384]`, remains at depth 0, and carries scale $\Delta_x\Delta_y\Delta_w$.

The decorator constructs a `CompiledCallable` without running the function. Python parameter names, positional/keyword rules and result types are preserved for static tooling. An ordinary call prepares a missing specialization with the default `on_miss="compile"`; `on_miss="error"` instead requires a prior `prepare` call. A Backend and pipeline are optional overrides.

Preparation and device-code compilation are distinct. A Triton binary may still compile during the first real execution. Run an actual startup call before warm timing or caller-owned CUDA Graph capture.

## Understand specialization

Specialization records value kind, shape, strides, dtype, device, CKKS state and static scalar arguments. Tensor contents and storage addresses are not cache keys. Changing ciphertext contents can reuse a variant; changing the static branch value creates another one.

```python
compiled.prepare(x, y, weight, negative=True)
assert len(compiled.specializations) == 2
other_x = engine.negate(x)
result = compiled(other_x, y, weight)
assert len(compiled.specializations) == 2
```

Capture supports pure Python functions with supported out-of-place Eager calls and static control flow. In-place calls, arbitrary callable objects, instance-method binding and async execution are outside this frontend's scope. A supported compiled helper called during capture is inlined from its Python source under the outer pipeline.

## Matrix multiplication with BSGS

The second function computes a plaintext matrix times an encrypted matrix. Each of the eight ciphertext batch items holds one column of $X$, periodically tiled across the 8192 slots. The ciphertext input has shape `[2, 8, 8, 16384]`: component, column batch, Q row and polynomial coefficient.

For $M=128$, define cyclic diagonals

$$
d_k[i]=A[i,(i-k)\bmod M].
$$

With $B=8$, write $k=gB+b$. The example encodes $\widetilde d_{g,b}=\operatorname{Roll}_{-gB}(d_{gB+b})$ and evaluates

$$
Y=\sum_{g=0}^{15}\operatorname{Roll}_{gB}\left(
  \sum_{b=0}^{7}\widetilde d_{g,b}\odot\operatorname{Roll}_b(X)
\right).
$$

Here $\operatorname{Roll}_s(v)[i]=v[(i-s)\bmod M]$ within each periodically packed column. The implementation uses:

- seven baby rotations with shared hoisted preparation;
- eight forward NTTs of the batched baby ciphertexts, reused across groups;
- 128 plaintext–ciphertext products and their group sums;
- sixteen inverse NTTs and fifteen giant rotations;
- one final sum and rescale.

The 22 rotation keys and 128 compressed NTT diagonals are prepared before capture. Each diagonal passes one 128-slot period to `engine.prepare_compressed_plaintext`, which stores 256 values per prime row without first materializing the full-ring plaintext. Compact indexing, plaintext multiplication and group accumulation can participate in the same fusion region as the following inverse NTT. The diagonal list and key dictionary provide live Tensor bindings; they do not require separate variables for each material. The compiled function executes the written BSGS algorithm rather than translating `torch.matmul` into FHE. After decryption, the first 128 slots of each batch item are assembled into the $128\times8$ result and checked against `matrix @ clear_matrix_input`.

The output remains CT2 and advances to depth 1. Both Eager and JIT use the same ciphertexts, keys, diagonal plaintexts and rescale placement, and their output Tensors are compared without a numerical tolerance.

## Inspect the compilation

```python
variant = compiled.specializations[0]
print(variant.signature)
print(variant.source.program)
print(variant.compilation.program)
print(variant.executable.host_source)
for report in variant.compilation.reports:
    print(report.name, report.stats)
```

`source` contains the Program before input specialization and transformation; `compilation` contains the transformed Program and pass reports. `executable` contains the prepared execution, while `reference` retains the original Python behavior for differential checks rather than automatic fallback.

## Edit the lowering-and-fusion recipe

`default_lower_and_fuse_pipeline` is an editable recipe for lowering, local implementation selection, cleanup and compatible-region fusion. It operates on Program facts and supplied materials independently of JIT signatures, and also supports hand-written or loaded Programs.

```python
from fhelium.backend import OperationBackend

backend = OperationBackend()
pipeline = fc.default_lower_and_fuse_pipeline(backend)
print(pipeline.names)
without_fusion = (
    pipeline.replace("fuse-operations")
    if "fuse-operations" in pipeline.names else pipeline
)
manual_choice = fc.compile(reference, backend=backend, pipeline=without_fusion)
```

A supplied pipeline replaces optimization, not Backend linking. Selecting a Backend changes the available implementations without injecting private passes. CPU and CUDA regions can coexist, and unknown placement remains unknown. Neither the recipe nor fusion moves materials to another device.

## Materials and numerical roles

Create evaluation keys before capture or supply missing Tensor placeholders before linking. Capture preserves the actual selected key data and does not generate missing keys or verify ciphertext/key provenance. Each implementation still requires its operand layout and placement conditions.

Ordinary Tensor intermediates retain their numerical role inside a function that also consumes ciphertexts. Supported normalization, encoding and plaintext preparation can be captured when their inputs change on each call. Encoding produces a plaintext representation; it does not encrypt the weight.

Fixed references, including lists and dictionaries, remain live bindings. JIT caches code rather than normalization or encoding results. Prepare fixed encoded weights outside the function when they should be reused; encoding inside the graph consumes its supplied rounding state on each execution. See [Compilation materials and persistence](./compile-material-persistence) for saving and supplying material bindings.

## Run the example

```bash
python examples/11_compile_jit.py --device cpu
python examples/11_compile_jit.py --device cuda:0
python examples/11_compile_jit.py --device cuda:0 --print-ir
python examples/11_compile_jit.py --device cuda:0 --warmup 10 --runs 100
```

The example reports the size, output state, clear-message error, pass decisions and selected implementations for both workloads. It reports first-call latency separately from warmed Eager/JIT medians, their ratio and percentage latency change. Warm samples alternate execution order and use the same inputs. Timing covers Python submission through completion, with CUDA synchronization. Setup, key generation, encoding, encryption, decryption and correctness checks are excluded from warm intervals. No CUDA Graph is used, and inherited PyTorch thread settings are preserved and printed.

First-call latency includes specialization, linking and any required kernel compilation or cache loading. A negative latency change means JIT was faster in that run.

<details>
<summary>Source</summary>

<<< ../../examples/11_compile_jit.py

</details>
