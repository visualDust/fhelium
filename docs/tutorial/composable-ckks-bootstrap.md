# Bootstrap a full-slot CKKS ciphertext

**Example source:** [`examples/25_experimental_bootstrap.py`](https://github.com/VisualDust/fhelium/blob/main/examples/25_experimental_bootstrap.py)

FHElium composes bootstrapping from modulus raising, Fourier transforms and periodic reduction. A configured `FullSlotBootstrap` is callable, exposes its depth and scale schedule, and supports prepared execution. `cosine_bootstrap` assembles common component combinations; its profiles are starting points that can be replaced or customized.

This tutorial constructs both centered raising and **sparse-secret encapsulation**, then uses the same key inventory, execution and verification workflow. The [composition guide](../how-to/compose-bootstrap-circuit) covers component replacement. The [mathematical model](../concepts/ckks/composable-bootstrapping) defines the coordinates and state transitions.

## 1. Choose the construction

The complete refresh follows:

```text
entry coordinates → modulus raising → CoeffsToSlots
                  → periodic reduction → SlotsToCoeffs → output compensation
```

`CenteredModRaise` extends the centered CRT representative of each component under the main secret. `EncapsulatedModRaise` switches to an intermediate secret in a low basis, extends the centered components and returns to the main secret before the Fourier transforms. A sparse intermediate secret changes the integer phase range that the subsequent periodic reduction must handle.

The cosine profiles select the following circuit components:

| Profile | Raw coordinate bound B | Seed degree | Double-angle iterations | CoeffsToSlots stages | SlotsToCoeffs stages | Default polynomial evaluation |
|---|---:|---:|---:|---:|---:|---|
| `7/44` | 1024 | 44 | 7 | 2 | 2 | Blocked |
| `8/28` | 1024 | 28 | 8 | 2 | 2 | Binary |
| `8/28-s2c3` | 1024 | 28 | 8 | 2 | 3 | Binary |
| `3/30` | 32 | 30 | 3 | 4 | 3 | Even decomposition with Blocked evaluation |

A profile does not choose the secret distribution or generate keys. Select the modulus chain, raising relation, coordinate range and scale schedule together. The degree-30 profile uses a narrower range; selecting its name alone does not enable sparse-secret encapsulation.

### Centered raising

The following configuration uses int64 residues, generator five and a default scale of approximately $2^{50}$:

```python
import torch
import fhelium as fh
from fhelium.eager import Engine
from fhelium.experimental import bootstrap as bs
from fhelium.values import EvaluationKeySet

torch.set_default_device("cuda:0")
config = fh.CkksConfig.parse(
    fh.Preset.slots32768_scale50_depth27_int64,
    galois_generator=5,
)
engine = Engine(config, allow_automatic_key_generation=False)
secret_key = engine.create_secret_key()
bootstrap = bs.cosine_bootstrap(engine)
```

The default `7/44` composition uses Blocked Chebyshev evaluation and aggregate giant-step quotients. Its default source basis is the terminal Q group; entry begins at the preceding group and uses integer scale lifting followed by rescale.

### Sparse-secret encapsulation

For an executable example from the repository root:

```bash
python examples/25_experimental_bootstrap.py --encapsulated
```

The example compiles the complete prepared callable by default and provisions all required direct rotations. `--eager` selects Eager execution; `--core-graph` selects the normalized-core Graph example. The cosine composition selects weighted-product implementations with `"auto"` and a chunk size of 16. CUDA contexts use the group-16 compact Montgomery NTT policy unless the caller supplies another policy.

The example defines prime groups and per-depth scale targets in `examples/_bootstrap_encapsulation_parameters.py`. It uses $N=65536$, int32 residues, 51 main Q primes, 11 main P primes, and a two-prime source Q basis. The `profile="3/30"` choice specifies the approximation profile; the example supplies these modulus and scale parameters separately.

From the repository root, the same construction can be assembled as follows. This block replaces the centered configuration above; the remaining tutorial steps apply to either choice.

```python
from examples._bootstrap_encapsulation_parameters import parameters

config, auxiliary, target_scales = parameters()
engine = Engine(
    config,
    rns_dtype=torch.int32,
    allow_automatic_key_generation=False,
)
secret_key = engine.create_secret_key(hamming_weight=config.N // 2)
intermediate = engine.create_secret_key(hamming_weight=32)
low_key, return_key = engine.create_encapsulation_keys(
    secret_key, intermediate, **auxiliary
)
raising = bs.EncapsulatedModRaise(low_key, return_key, **auxiliary)
del intermediate

bootstrap = bs.cosine_bootstrap(
    engine,
    profile="3/30",
    modulus_raising=raising,
    input_depth=engine.max_depth,
    target_scales=target_scales,
    product_implementation="auto",
)
```

The low switch changes the relation from main secret $s$ to intermediate secret $t$. The return hint is encrypted under $s$, so the Fourier transforms, polynomial evaluation and output again use $s$. `EncapsulatedModRaise` retains the two evaluation-key values, not either secret. An intermediate secret need not be regenerated for every refresh; the supplied evaluation material can be reused.

`hamming_weight=h` samples exactly $h$ nonzero coefficients on a uniformly selected support, with independent signs. Omitting it retains independent ternary sampling. The example's weight 32 is one choice: `--intermediate-secret-weight` changes it, and the reducer range must be assessed for the resulting construction. The low Q/P relation, return hint and their joint publication determine the relevant security assumptions. The main QP budget check alone does not assign a security level to that combination; see [Sparse-secret encapsulation](../developer/security#sparse-secret-encapsulation).

The degree-30 seed is even after a coordinate shift. `EvenChebyshevEvaluator` uses $T_{2j}(x)=T_j(T_2(x))$ to evaluate it through a degree-15 inner polynomial. The profile places the Fourier factor and slot-count normalization in the last forward stage, and uses the supplied scale targets for subsequent arithmetic.

## 2. Provision the shared evaluation keys

After selecting either construction, create the encryption key and populate an application-owned inventory for the bootstrap:

```python
public_key = engine.create_public_key(secret_key)
evaluation_keys = EvaluationKeySet()
bootstrap.create_evaluation_keys(
    secret_key,
    into=evaluation_keys,
    device=secret_key.device,
)
```

`EvaluationKeySet` retains ordinary rotation, relinearization and conjugation key values. Its constructor accepts an iterable, and `add` and `update` retain references with object-identity deduplication. Several consumers can share the inventory, and multiple layouts or devices for one rotation step can coexist. The application keeps the members within the intended parameter context and secret relation.

The provisioning helper returns the supplied inventory. It reuses compatible existing groups and creates missing keys, considering additional storage across the composition's consumers. A broader key can serve several stages; generating a separate compact key for every depth can require more total storage. Consumer selection checks actual digit supports, QP rows, ring dimension, dtype and device, including the compatibility of an entire hoisted rotation group.

The default rotation-key strategy is `all_required`. The available strategies are:

| Strategy | Rotation material |
|---|---|
| `signed_power_of_two` | Signed-power keys used through composed rotations |
| `signed_power_of_two_with_baby_steps` | Signed-power coverage plus direct baby steps for the selected BSGS partitions |
| `all_required` | Direct keys for every required logical transform rotation |

Inspect `bootstrap.key_steps(strategy)` and `bootstrap.evaluation_key_requirements(strategy)` before generation. Key coverage and baby-rotation hoisting are separate choices. Encapsulation's low and return hints belong to the raising component; rotation, relinearization and conjugation keys in `evaluation_keys` use the main secret. Generate or load these values before preparation or capture.

## 3. Refresh and verify a ciphertext

An application supplies a ciphertext at `bootstrap.input_depth`. For a standalone numerical check, encrypt directly at that depth and the configured input scale:

```python
values = torch.linspace(-0.05, 0.05, engine.num_slots, dtype=torch.float64)
ciphertext = engine.encrypt_message(
    values,
    public_key,
    depth=bootstrap.input_depth,
    scale=bootstrap.input_scale,
)
refreshed = bootstrap(ciphertext, evaluation_keys=evaluation_keys)
decoded = engine.decrypt_message(refreshed, secret_key, is_real=True)
error = (decoded - values).abs()

print("entry depth:", bootstrap.input_depth)
print("output depth:", refreshed.depth)
print("remaining depth:", bootstrap.remaining_depth)
print("refresh window:", bootstrap.refresh_window)
print("output actual scale:", refreshed.scale)
print("max error:", error.max().item())
print("mean error:", error.mean().item())
```

Input and output are two-component coefficient-domain, standard-residue Q ciphertexts with axes `[component, *batch, limb, coefficient]`. The output records its active `prime_ids` and actual scale. Online evaluation uses ciphertexts, evaluation keys and numerical materials; the secret key is used here only to verify the result.

Let the terminal depth be $D$, the entry depth $d_{in}$ and the output depth $d_{out}$. `remaining_depth` is $D-d_{out}$; `refresh_window` is $d_{in}-d_{out}$. Each transition removes a complete Q group, which may contain several primes. Eighteen remaining Q rows therefore do not imply eighteen multiplication depths.

For preceding-group entry, integer multiplication and rescale produce an actual source scale $\Delta_b$. Direct source-basis entry uses the existing actual scale. The core interprets the source payload at `input_scale` $\Delta_0$, and `finish` compensates the output scale by $\Delta_b/\Delta_0$. Within the circuit, every rescale retains its actual quotient scale.

The raw reduction coordinate $r$ and normalized polynomial coordinate $x=r/B$ are distinct. `reference(values)` consumes normalized coordinates, with target $\rho_B(x)=\sin(\pi Bx)/\pi$. The original slot amplitude alone does not establish the raw-coordinate bound or the complete refresh error. Check the intended message class, approximation range, actual scales and downstream precision requirements together.

## 4. Prepare Eager and compiled execution

Prepare the complete refresh once. Material discovery uses the example's state and device to select key references and encode diagonals and constants, recording the results in ordinary dictionaries. The returned `PreparedBootstrap` includes entry, normalized core and output scale compensation. It retains the selected references and releases the example payload and temporary result. New encodings consume the rounding RNG once; repeated execution and callable capture reuse those samples.

```python
from fhelium import compile as fc
from fhelium.runtime import CudaGraphProgram

prepared = bootstrap.prepare(ciphertext, evaluation_keys=evaluation_keys)
refreshed_eager = prepared(ciphertext)
optimized = fc.compile(prepared)
refreshed_compiled = optimized(ciphertext)

graph = CudaGraphProgram.capture(
    optimized, example_inputs=(ciphertext,), warmup=1
)
try:
    refreshed = graph.replay(ciphertext, copy_output=True, synchronize=True)
finally:
    graph.close()
    prepared.close()
```

Eager and Compile use the same prepared Python computation. Compile represents it as ordinary CKKS, RNS, NTT and Tensor operations with live material operands. Adding or removing inventory entries after preparation leaves these bindings unchanged; prepare a new callable to select other keys. In-place writes to selected key Tensors remain visible under the prepared layout and synchronization contract. Compilation consumes existing keys and encoded samples.

`copy_output=True` returns storage independent of Graph replay. Close a Graph before its prepared owner. A linked compiled executable retains its bound Tensors until that executable is released. Clearing the shared inventory releases only its own references.

The callable can also be source-inlined inside a larger `fc.compile` function. Select a custom Backend or caller-composed pipeline through the ordinary Compile arguments. For applications that manage normalized coordinates directly, `entry`, `prepare_core` and `finish` remain available; the full prepared callable is the common entry.

## 5. Choose component and storage policies

`FullSlotBootstrap` accepts independent Fourier compilers and evaluators, a periodic-reduction component and a modulus-raising component. Use it directly when a named profile does not describe the desired construction. The [composition guide](../how-to/compose-bootstrap-circuit) shows the constructor and replacement points.

| Choice | Control | Effect |
|---|---|---|
| Raising relation | `modulus_raising` and its supplied keys | Centered extension or intermediate-secret encapsulation |
| Entry and scale schedule | `source_depth`, `input_depth`, `input_scale`, `target_scales` | Coordinate conversion, coefficient-product targets and available depth |
| Fourier decomposition | Stage counts and `scale_stage` on transform compilers | Rotation workload, factor placement and group consumption |
| Polynomial evaluation | Binary, Blocked or Even evaluator | Shared basis, multiplication/relinearization schedule and rounding |
| Giant-step quotient | `aggregate` or `independent` | One shared QP quotient or per-giant transitions; different workspace and rounding |
| Weighted products | `product_implementation`, `product_chunk_size` | Native or selected generated implementation and expression extent |
| Key inventory | Shared `EvaluationKeySet` and `create_evaluation_keys` | Reuse existing groups and provision missing material across the composition |
| Key-switch lift | `modup_method` | Default exact lift or CRT alternatives requiring a compatible noise budget |
| Material reuse | `retain_diagonals`, `retain_constants` | Resident storage versus repeated preparation |
| Diagonal storage | `compress_diagonals` | Lossless compression of eligible encoded integers |
| Slot semantics | `message_domain` | Complex branches or generator-five real-slot reconstruction |

Changing a polynomial evaluator preserves the selected coefficients; changing a profile can change the polynomial, Fourier stages and raw bound together. Fixed-weight secrets and auxiliary primes are separate choices from both. No profile automatically searches this design space or chooses a security target for the caller.

## 6. Performance and material footprint

These measurements were collected on October 31, 2026, on an **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**, with $N=65536$, 32768 complex slots and one ciphertext per request. All configurations use the same random unit-modulus complex message and main-secret coefficients, source-basis entry, compressed diagonals and `product_implementation="auto"` with chunk size 16. The measured 3/30 coefficient-to-slot baby steps are 8192, 512, 32 and 4.

The first five rows use Binary evaluation and independent giant-step quotients. The two Blocked rows and both sparse-secret rows use aggregate quotients. The sparse-secret profile evaluates its even seed through a degree-15 Blocked schedule. Each configuration provisions an initially empty `EvaluationKeySet` through `create_evaluation_keys`, allowing full and active-basis keys to share storage across consumers. The final row selects `modup_method="normalized_crt"`; every other row uses the exact lift.

| Construction | Rotation-key strategy | Remaining depths | Ordinary latency (ms) | Graph latency (ms) | Max absolute error | Keys (GiB) | Diagonals (GiB) | Ordinary peak allocated (GiB) | Ordinary peak reserved (GiB) | Graph peak allocated (GiB) | Graph peak reserved (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Centered 7/44 · Binary · int64 | Signed power | 8 | 335.8 | 325.1 | 1.035e-03 | 6.221 | 4.1596 | 11.821 | 12.564 | 11.821 | 12.662 |
| Centered 7/44 · Binary · int64 | Power + babies | 8 | 265.5 | 256.5 | 1.033e-03 | 10.473 | 4.1596 | 16.083 | 16.932 | 16.084 | 16.961 |
| Centered 7/44 · Binary · int64 | All required | 8 | 235.5 | 228.1 | 1.026e-03 | 14.383 | 4.1596 | 20.011 | 20.857 | 20.012 | 20.887 |
| Centered 8/28 · Binary · int64 | Power + babies | 8 | 229.7 | 221.9 | 1.030e-03 | 10.473 | 4.1596 | 16.084 | 16.662 | 16.085 | 16.723 |
| Centered 8/28 · Binary · int64 · three STC stages | Power + babies | 7 | 204.3 | 198.2 | 1.049e-03 | 10.373 | 3.6637 | 15.500 | 16.010 | 15.501 | 16.039 |
| Centered 7/44 · Blocked · int64 | All required | 8 | 161.9 | 156.0 | 1.027e-03 | 15.335 | 4.1596 | 21.010 | 21.355 | 21.005 | 21.922 |
| Centered 7/44 · Blocked · int32 | All required | 8 | 157.0 | 151.2 | 1.013e-03 | 14.427 | 4.1596 | 20.071 | 20.387 | 20.067 | 20.979 |
| Sparse-secret 3/30 · int32 · exact lift | All required | 8 | 67.3 | 64.6 | 8.240e-06 | 6.133 | 0.3905 | 7.410 | 7.699 | 7.410 | 7.740 |
| Sparse-secret 3/30 · int32 · normalized CRT lift | All required | 8 | 62.8 | 60.0 | 8.864e-06 | 6.133 | 0.3905 | 7.411 | 7.701 | 7.411 | 7.742 |

### Parameters and depth

The int64 centered configurations use `slots32768_scale50_depth27_int64` with generator five: 28 Q primes in single-prime depth groups, four P primes and seven key digits. The int32 centered configuration uses 56 Q primes in two-prime groups, eight P primes and seven key digits. Both centered configurations use input scale $2^{50}$. These digit counts describe their configured full-basis keys. Fresh compact keys partition the active Q suffix, and the shared inventory depends on the stage schedule and reuse opportunities.

The sparse-secret configuration uses the example's 51 Q primes, 11 main P primes and five key digits, plus the low and return hints. Its intermediate secret has weight 32; the main secret shared by all cases has weight $N/2$. The source basis contains two primes, and input scale is their product divided by 32, approximately $2^{50.226}$. Its output contains 18 Q rows and eight remaining depth transitions. The int32 centered output also contains 18 rows and eight remaining transitions; the int64 7/44 output contains nine rows and eight transitions. The three-STC 8/28 variant leaves seven transitions.

Every measured callable sets `input_depth=engine.max_depth`, so entry starts directly from its source Q basis. The table's refresh windows therefore equal the remaining-depth counts. The introductory centered example instead uses the default preceding-group entry, which consumes an additional transition before the normalized core.

### Timing and memory scope

Measurements were taken on October 31, 2026, using an AMD Ryzen Threadripper PRO 9965WX, PyTorch 2.13, CUDA 13.0 and driver 580.196.07. PyTorch retained 24 intra-op and 24 inter-op threads. Each configuration ran in three fresh processes, with three warm-ups and twenty timed samples per execution mode. Ordinary execution preceded Graph capture. All cases used `radix2_compact_group8_smem8` with Montgomery twiddle tables. The table reports the median of the three process medians; the complete study contains 27 processes and 1080 timed samples.

Both modes execute the same prepared Eager entry/core/finish callable on a fixed resident input. Ordinary latency measures the call followed by synchronization; Graph latency measures replay followed by synchronization. Key generation, material preparation, input copying and Graph capture are outside the timed interval. The first output was checked against the configured output depth, Q rows, shape, representation and entry/core/finish scale equations. Repeated ordinary output, warmed Graph output and the final output after each timing series matched that result exactly. Inputs were checked unchanged after timing. Within each configuration, input, evaluation-key, encoded-material and qualification-output hashes also matched across the three processes. The six int64 centered cases shared the same input ciphertext.

Keys and diagonals are retained Tensor payloads. The complete application inventory remains resident, including keys shared across consumers; encapsulation includes the low and return hints. Each memory column gives the maximum corresponding peak across the three processes. Ordinary peaks come from a separate post-warm-up complete call; Graph peaks cover capture and replay. Allocated memory includes numerical tables, input/output, workspace and resident client verification keys and contexts. Reserved memory additionally includes allocator-held capacity.

### Numerical interpretation

On this message, the centered configurations have maximum absolute error near $10^{-3}$. Their polynomial references, evaluated from the actual reduction inputs and propagated through SlotsToCoeffs, have errors between $1.023\times10^{-3}$ and $1.037\times10^{-3}$. Execution residuals relative to those references range from $3.557\times10^{-5}$ to $1.042\times10^{-4}$. Because the reference starts at the actual reduction input, it retains preceding input, modulus-raising and Fourier errors. The residual measures subsequent reduction and SlotsToCoeffs differences. Each reported maximum is evaluated separately.

The sparse-secret exact-lift configuration has maximum error $8.24\times10^{-6}$, polynomial-reference error $8.203\times10^{-6}$ and execution residual $1.853\times10^{-6}$. The normalized-CRT configuration gives $8.864\times10^{-6}$, $8.217\times10^{-6}$ and $1.87\times10^{-6}$ respectively. Both passed the $10^{-4}$ absolute-error screen. Centered configurations retained the model-relative screen $4\max(10^{-4},\epsilon_{\rm reference})$.

The centered and sparse-secret constructions change the raising relation, source-modulus/scale ratio, Fourier stages, polynomial and key material together. The table reports each complete configuration's speed, precision and storage; its secret distributions and auxiliary relations remain part of the selected security model.

## Example sources

::: details Complete runner
<<< @/../examples/25_experimental_bootstrap.py
:::

::: details Sparse-secret encapsulation parameters
<<< @/../examples/_bootstrap_encapsulation_parameters.py
:::
