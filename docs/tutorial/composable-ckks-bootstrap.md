# Bootstrap a full-slot CKKS ciphertext

**Example source:** [`examples/09_algorithms_bootstrap.py`](https://github.com/VisualDust/fhelium/blob/main/examples/09_algorithms_bootstrap.py)

`fhelium.bootstrap` refreshes a CKKS ciphertext through modulus raising, CoeffsToSlots, periodic reduction (EvalMod) and SlotsToCoeffs, all executed as ordinary Eager operations. `bootstrap.plan` returns a `BootstrapPlan` that records every component choice and its depth placement; `create_evaluation_keys` provisions the keys the plan requires; `bootstrap` refreshes a ciphertext; and `prepare` returns a `PreparedBootstrap` for repeated execution, `fhelium.compile` capture and CUDA Graph replay.

This tutorial uses a built-in bootstrap preset, inspects its plan, provisions keys, refreshes a ciphertext and prepares compiled execution. The [composition guide](../how-to/compose-bootstrap-circuit) covers component replacement. The [mathematical model](../concepts/ckks/composable-bootstrapping) defines the coordinates and state transitions.

## 1. Choose a chain and a plan

The `-boot` presets reserve a 16-depth bootstrap segment at the start of the chain and place the application depths after it. A ciphertext refreshed at the maximum depth returns at depth 16 with the remaining depths available to the application:

| Preset | N | Scale bits | Maximum depth | Application depths after refresh | Q/P rows |
|---|---:|---:|---:|---:|---|
| `slots32768-scale40-depth20-boot4` | 65536 | 40 | 20 | 4 | 38/22 |
| `slots32768-scale40-depth24-boot8` | 65536 | 40 | 24 | 8 | 44/16 |
| `slots32768-scale40-depth29-boot13` | 65536 | 40 | 29 | 13 | 50/10 |
| `slots32768-scale50-depth24-boot8` | 65536 | 50 | 24 | 8 | 47/13 |
| `slots32768-scale50-depth28-boot12` | 65536 | 50 | 28 | 12 | 53/7 |

All five use int32 residues and Galois generator 5, which full-slot bootstrapping requires. Their prime catalogs use 1731–1747 of the 1747 QP bits that the 128-bit security table allows for $N=2^{16}$; depth bases are ordered subsets of that catalog, so primes of the bootstrap segment reappear in application depths.

```python
import torch
import fhelium as fh
from fhelium import bootstrap
from fhelium.eager import Engine

torch.set_default_device("cuda:0")
config = fh.CkksConfig.parse(fh.Preset.slots32768_scale40_depth29_boot13)
engine = Engine(config, allow_automatic_key_generation=False)
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)

plan = bootstrap.plan(engine, intermediate_weight=48)
for stage in plan.stages():
    print(stage.name, stage.input_depth, "->", stage.output_depth)
print("refresh window:", plan.refresh_window)
```

The default `"3/38"` profile uses raw coordinate bound $B=33$, four CoeffsToSlots stages with the slot-count normalization folded into the last, a degree-38 cosine seed expanded in Chebyshev form, three double-angle iterations evaluated by `RecursiveChebyshevPlan`, and three SlotsToCoeffs stages. It consumes 16 depths: the raise targets depth 0, CoeffsToSlots occupies depths 0→4, EvalMod 4→13 and SlotsToCoeffs 13→16. Key switching uses `basis_conversion="normalized_crt"`.

`intermediate_weight=h` selects `Encapsulated` modulus raising. The raise switches the source-basis ciphertext to an intermediate secret of Hamming weight $h$ in a small auxiliary basis, extends the centered components and returns to the main secret before CoeffsToSlots. The overflow polynomial then has coefficients bounded by $h+1$ rather than by a function of $N$, which is what allows $B=33$. `plan.modulus_raise_bound` reports that worst-case bound. Without `intermediate_weight`, the raise is `Centered` and extends the centered CRT representative under the main secret.

The other named profiles (`"7/44"`, `"8/28"`, `"8/28-s2c3"`, `"3/30"`) select wider coordinate bounds or different stage counts:

| Profile | Raw coordinate bound B | Seed degree | Double-angle iterations | CoeffsToSlots stages | SlotsToCoeffs stages | Polynomial evaluation |
|---|---:|---:|---:|---:|---:|---|
| `3/38` | 33 | 38 | 3 | 4 | 3 | Recursive Chebyshev |
| `3/30` | 32 | 30 | 3 | 4 | 3 | Even decomposition with Blocked evaluation |
| `7/44` | 1024 | 44 | 7 | 2 | 2 | Blocked |
| `8/28` | 1024 | 28 | 8 | 2 | 2 | Binary |
| `8/28-s2c3` | 1024 | 28 | 8 | 2 | 3 | Binary |

A profile selects the approximation and stage structure. The chain, the raising relation and the secret distributions are separate choices; the wide-range profiles suit a centered raise on a chain whose terminal Q group is large relative to the scale, such as `CkksConfig.catalog_suffix_chain(logN=16, scale_bits=50, depth=27, p_count=4, galois_generator=5)` with `"7/44"`.

The encapsulated raise publishes two additional key-switch keys: the low key switches from the main secret $s$ to the intermediate secret $t$ in the auxiliary basis, and the return key is a hint of $t$ encrypted under $s$. The security of publishing them with the remaining evaluation keys depends on $h$, the auxiliary basis and the main parameters; the main QP budget check alone does not assign a security level to that combination. See [Sparse-secret encapsulation](../developer/security#sparse-secret-encapsulation).

## 2. Provision the evaluation keys

```python
keys = bootstrap.create_evaluation_keys(engine, plan, secret_key)
```

The function returns an `EvaluationKeySet`. Pass `into=` to add the keys to an application-owned inventory; compatible keys already present are reused. Each consumed depth receives a key whose prime/digit layout is chosen by `CkksConfig.key_layout(depth)`, which minimizes the key-switch cost over the auxiliary P primes available in the catalog. With `Encapsulated`, the function also samples the intermediate secret, adds the low and return keys and discards the intermediate secret.

The default rotation-key strategy is `all_required`. The available strategies are:

| Strategy | Rotation material |
|---|---|
| `signed_power_of_two` | Signed-power keys used through composed rotations |
| `signed_power_of_two_with_baby_steps` | Signed-power coverage plus direct baby steps for the selected BSGS partitions |
| `all_required` | Direct keys for every required logical transform rotation |

`plan.key_steps(strategy)` and `plan.key_requirements(strategy)` describe the material before generation.

## 3. Refresh and verify a ciphertext

```python
generator = torch.Generator(device="cpu").manual_seed(0)
real, imag = torch.rand(2, engine.num_slots, generator=generator, dtype=torch.float64, device="cpu")
values = (2 * real - 1) + 1j * (2 * imag - 1)
ciphertext = engine.encrypt_message(values, public_key, depth=plan.source_depth)

refreshed = bootstrap.bootstrap(engine, ciphertext, plan, keys=keys)
error = (engine.decrypt_message(refreshed, secret_key).cpu() - values).abs()
print("depth:", ciphertext.depth, "->", refreshed.depth)
print("max error:", error.max().item())
```

Input and output are two-component coefficient-domain, standard-residue Q ciphertexts. An input below `plan.source_depth` first reaches the source basis through `mod_switch_to_depth`. The source-basis value at actual scale $\Delta_b$ is evaluated in coordinates of scale $\Delta_0=$ `plan.input_scale`; an input with $\Delta_b<\Delta_0$ is first multiplied by the integer $\max(1,\operatorname{round}(\Delta_0/\Delta_b))$. The output at `plan.output_depth` records actual scale $s_{out}\Delta_b k/\Delta_0$, so it represents the input message rather than the coordinate.

Depth is the consumed-depth index: depth 0 holds the full Q basis and each transition removes one depth group, which may contain several primes. `plan.refresh_window` is the difference between the source and output depths. The raw reduction coordinate $r$ and the normalized polynomial coordinate $x=r/B$ are distinct; the slot amplitude alone does not establish the raw-coordinate bound required by EvalMod.

## 4. Prepare Eager and compiled execution

```python
from fhelium import compile as fc

prepared = bootstrap.prepare(engine, plan, ciphertext, keys=keys)
graphed = fc.compile(prepared, graph=True)
try:
    refreshed = graphed(ciphertext)
finally:
    graphed.clear()
    prepared.close()
```

`prepare` fixes the input signature of `like` (depth, scale, batch shape and device), encodes the plan's constants and CoeffsToSlots/SlotsToCoeffs diagonals and selects its keys once. Calling the `PreparedBootstrap` refreshes an input without encoding or key selection, so Eager execution, `fhelium.compile` capture and CUDA Graph replay read only the prepared materials. New encodings consume the rounding stream once; repeated execution and capture reuse those samples. Call `prepare` before Graph capture, clear compiled callables before closing the prepared owner, and drop or close the owner to release its materials.

The step functions `modulus_raise`, `coeffs_to_slots`, `split_real_imag`, `eval_mod`, `merge_real_imag` and `slots_to_coeffs` evaluate one stage each for custom compositions.

## 5. Choose component and storage policies

`bootstrap.plan` accepts keyword arguments that replace individual choices, and `dataclasses.replace` on the returned plan changes any field afterwards:

| Choice | Control | Effect |
|---|---|---|
| Raising relation | `modulus_raise`, `intermediate_weight` | Centered extension or intermediate-secret encapsulation |
| Entry and scale schedule | `input_scale`, `target_scales` | Coordinate conversion and per-depth arithmetic targets |
| Fourier decomposition | `coeffs_to_slots_plan`, `slots_to_coeffs_plan` | Rotation workload, factor placement and depth consumption |
| Periodic reduction | `eval_mod`, `polynomial_plan` | Approximation and polynomial-evaluation schedule |
| Giant-step quotient | `giant_step_reduction` | One shared QP quotient or per-giant quotients |
| Weighted products | `product_implementation`, `product_chunk_size` | Native or generated implementation and expression extent |
| Key-switch basis conversion | `basis_conversion` | Exact conversion or CRT alternatives with a larger noise term |
| Diagonal storage | `compress_diagonals` | Lossless compression of eligible encoded integers |
| Slot semantics | `message_domain` | Complex branches or real-slot reconstruction |

No profile searches this design space or chooses a security target for the caller.

## 6. Performance and material footprint

These measurements were collected on November 24, 2026, on an **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** (300 W power limit) at 2272 MHz SM clock, with $N=65536$, 32768 complex slots, int32 residues, a uniform ternary main secret, `intermediate_weight=48`, `basis_conversion="normalized_crt"`, the profile `"3/38"` and one ciphertext per request. Inputs are uniformly random complex messages with $|\Re z|,|\Im z|\le1$ encrypted at the preset's maximum depth. Every configuration refreshes to depth 16, so the application depths after refresh equal the maximum depth minus 16.

| Preset | Key storage | Depth transition | Graph latency (ms) | Eager latency (ms) | Max absolute error | Keys (GiB) | Graph peak allocated (GiB) |
|---|---|---|---:|---:|---:|---:|---:|
| `slots32768-scale40-depth20-boot4` | expanded | 20 → 16 | 19.48 | 21.52 | 2.62e-06 | 2.10 | 3.09 |
| `slots32768-scale40-depth24-boot8` | expanded | 24 → 16 | 24.61 | 26.59 | 2.29e-06 | 2.99 | 4.21 |
| `slots32768-scale40-depth29-boot13` | expanded | 29 → 16 | 32.09 | 35.66 | 2.58e-06 | 5.15 | 6.58 |
| `slots32768-scale40-depth29-boot13` | compact | 29 → 16 | 34.35 | 39.29 | 2.58e-06 | 2.58 | 4.00 |
| `slots32768-scale50-depth24-boot8` | expanded | 24 → 16 | 27.87 | 30.84 | 2.60e-06 | 3.75 | 5.11 |
| `slots32768-scale50-depth28-boot12` | expanded | 28 → 16 | 38.12 | 42.66 | 2.48e-06 | 6.92 | 8.46 |

Graph latency is the median of nine single replays of the captured `PreparedBootstrap`, separated by one second of idle time; key generation, preparation, input copying and capture are outside the timed interval. Eager latency is the median of five calls of the `PreparedBootstrap`. Expanded keys are the default `Engine(expand_keys=True)`; compact keys (`expand_keys=False`) store each key-switching key's uniform component as seed rows and regenerate it inside key products. Under back-to-back replay this GPU reaches its power limit and lowers its clock, and the expanded boot13 replay then takes 34.0–35.2 ms. Keys are the retained payloads of the complete bootstrap inventory, including the two encapsulation keys. Peak allocated memory covers capture and replay and includes keys, encoded diagonals and constants, input/output and workspace. The error is the maximum over all slots of $|\mathrm{decrypt}-z|$ for the Graph output.

On an **NVIDIA RTX A6000** (300 W power limit, 1770–1905 MHz SM clock during replay), `slots32768-scale40-depth29-boot13` with expanded keys refreshes in 101.21 ms (Graph) and 117.34 ms (Eager) with maximum error 2.57e-06 and the same key and peak-memory footprint.

## Example sources

::: details Complete runner
<<< @/../examples/09_algorithms_bootstrap.py
:::
