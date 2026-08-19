# Refresh a full-slot CKKS ciphertext with composable bootstrapping

**Example source:** [`examples/17_ckks_bootstrap_logn16.py`](https://github.com/VisualDust/fhelium/blob/main/examples/17_ckks_bootstrap_logn16.py)

This example depletes a full-slot `logN = 16` ciphertext, constructs an engine-bound
bootstrap, generates the three primitive key inputs required by that callable,
and refreshes the ciphertext. The tutorial explains the mathematical range and
state assumptions that the factory cannot establish from encrypted data.

## 1. Construct the documented bootstrap configuration

```python
import torch
import fhelium as fh
from fhelium.core import EvaluationKeySet
from fhelium.experimental.bootstrap.presets import cosine_depth_refresh_logn16_v1

config = fh.CkksConfig.parse(
    fh.Preset.slots32768_scale50_levels27_int64,
    base_prime_bits=50,
)
engine = fh.CkksEngine(
    config,
    device="cuda:0",
    allow_sk_gen=False,
    galois_generator=5,
)
bootstrap = cosine_depth_refresh_logn16_v1(engine)
```

The global CKKS default remains 40 bits; this measured configuration selects 50-bit scale primes and one 50-bit structural base Q prime. The factory returns a compiled
`FullSlotBootstrap`, not a descriptive circuit awaiting another compilation
step.

The `logn16` name documents the measured configuration. The factory itself checks
transform slot counts, structural-base/default-scale proximity, and depth, but
it does not enforce this preset or certify a numerical range.

## 2. Understand the range precondition

The cosine factory sets

```python
bootstrap.modular_reduction.input_bound == 1024
bootstrap.modular_reduction.fuse_input_normalization is True
```

Let raw branch coordinate be $r$ and $B=1024$. The polynomial coordinate is

$$
x=r/B\in[-1,1],
$$

and the periodic target is

$$
\rho_B(x)=\frac{\sin(\pi Bx)}{\pi}
          =\frac{\sin(\pi r)}{\pi}.
$$

Fusion means `FullSlotBootstrap` folds $1/B$ into CoeffsToSlots; the reducer's
`evaluate()` receives $x$ and must not divide again. In contrast,
`bootstrap.modular_reduction.reference(values)` always expects normalized $x$
regardless of the fusion setting.

For the full pipeline, both raw real and imaginary branch coordinates must lie
within $[-B,B]$. Ciphertext data does not reveal that range to the factory. The
application must establish it from its circuit bounds and validate the observed
error distribution.

## 3. Generate the required primitive keys

```python
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
rotation_keys = bootstrap.create_rotation_keys(
    secret_key,
    rotation_strategy="power_of_two",
)
relinearization_key = engine.create_relinearization_key(secret_key)
conjugation_key = engine.create_conjugation_key(secret_key)
evaluation_keys = EvaluationKeySet(
    rotations=rotation_keys,
    relinearization=relinearization_key,
    conjugation=conjugation_key,
)
```

The compact rotation inventory contains signed powers of two.
`FullSlotBootstrap` composes them when an exact transform rotation is absent.
Use `rotation_strategy="exact"` to trade more key memory for fewer online
rotation compositions.

`create_rotation_keys()` derives only the inventory reported by `key_steps()`.
Built-in polynomial recurrences require the separately generated
`RelinearizationKey`, while full-slot branch splitting requires the
`ConjugationKey`. `EvaluationKeySet` validates the evaluator-only inventory
without mixing in the public or secret key.

## 4. Create a final-public-level input

A real application reaches the entry level after useful operations. The example
below consumes levels with multiplication by encoded ones:

```python
values = torch.linspace(-0.1, 0.1, engine.num_slots, dtype=torch.float64)
ciphertext = engine.encrypt_message(values, public_key)
ones = torch.ones(engine.num_slots, dtype=torch.float64)

while ciphertext.level < engine.final_public_level:
    identity = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            ones,
            level=ciphertext.level,
            scale=engine.config.default_scale,
        )
    )
    ciphertext = engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(ciphertext), identity
            )
        )
    )
```

The entry ciphertext has axes
`[component, *batch, limb, coefficient]`, two components, coefficient domain,
standard residues, Q basis, and exact active `prime_ids`. The built-in topology
uses all $S=N/2$ slots and currently requires final public level with actual
scale near `default_scale` or its square.

Each depletion multiplication records the actual pending scale product, and
each public rescale divides that scale by the exact dropped Q prime. No public
operation silently normalizes to `default_scale`.

## 5. Follow the refresh state transitions

Calling

```python
refreshed = bootstrap(
    ciphertext,
    evaluation_keys=evaluation_keys,
)
```

executes:

1. If entry scale is near $\Delta_0$, multiply by encoded $1$ at $\Delta_0$ to
   obtain pending scale. If already near $\Delta_0^2$, pass through.
2. Divide and round by the final public scale prime, leaving the single active
   structural base Q row `[q_b]`. Record the arithmetic scale first, then explicitly reinterpret the
   unchanged residues at $\Delta_0$ under the private bootstrap policy.
3. Center each component modulo $q_b$ and extend it into the target Q
   `prime_ids`. Centered ModRaise preserves level target, represented centered
   integers, component count, domain, residue representation, and scale; it is
   not a rescale.
4. Apply CoeffsToSlots. Every diagonal stage consumes one leading Q row and
   updates actual scale by

   $$
   \Delta_{j+1}=\Delta_j\Delta_0/q_j.
   $$

5. Multiply by $1/S$, split by conjugation, apply periodic reduction to both
   branches, restore the imaginary branch, and recombine.
6. Apply SlotsToCoeffs with the same per-stage actual-scale recurrence.

The final output is a two-component coefficient-domain standard-RNS Q
ciphertext at `bootstrap.output_level`, with exact
`engine.rns_layout.prime_ids(bootstrap.output_level)`. The final actual scale is
the product of the SlotsToCoeffs recurrences; it is not assumed equal to
`default_scale`.

## 6. Verify with the secret key

```python
decoded = engine.decrypt_message(refreshed, secret_key, is_real=True)
error = (decoded - values).abs()
print("output level:", refreshed.level)
print("output actual scale:", refreshed.scale)
print("max error:", error.max().item())
print("mean error:", error.mean().item())
```

Only client verification uses the secret key. Online bootstrapping uses the
ciphertext and the supplied `RotationKeySet`, `RelinearizationKey`, and
`ConjugationKey`. Evaluate maximum error, mean error, distribution shape, and
workload-specific downstream effects; a factory name is not a tolerance
guarantee.

## 7. Choose another built-in composition

```python
from fhelium.experimental.bootstrap.presets import (
    cosine_depth_refresh_logn16_8_28_v1,
    exponential_depth_refresh_logn16_d16_v1,
)

alternative = cosine_depth_refresh_logn16_8_28_v1(engine)
exponential = exponential_depth_refresh_logn16_d16_v1(engine)
```

The 8/28 cosine composition uses a degree-28 seed and eight double-angle steps. The
exponential composition stores ascending power coefficients for
$\exp(i\pi x)$, squares $\log_2 B$ times, and extracts sine by conjugation.
Both use raw `input_bound=1024` with fused normalization, but their approximation
error, level cost, and CKKS error propagation differ.

## 8. Compose directly

Factories are optional. The baseline can be assembled from public components:

```python
from fhelium.experimental import bootstrap as bs

compiler = bs.Radix2FourierTransformCompiler(stage_count=2)
evaluator = bs.DiagonalBSGSEvaluator(
    baby_step=16,
    hoist_baby_rotations=True,
)
reduction = bs.CosineDoubleAngleReduction(
    input_bound=1024,
    double_angle_iterations=7,
    approximator=bs.ChebyshevInterpolator(degree=44),
    evaluator=bs.BinaryDecompositionChebyshevEvaluator(skip_near_zero=1e-15),
    fuse_input_normalization=True,
)
bootstrap = bs.FullSlotBootstrap(
    engine,
    coeffs_to_slots_compiler=compiler,
    coeffs_to_slots_evaluator=evaluator,
    modular_reduction=reduction,
    slots_to_coeffs_compiler=compiler,
    slots_to_coeffs_evaluator=evaluator,
)
```

`PolynomialApproximation` coefficients are ascending. `basis="power"` means
$\sum_n a_nx^n$; `basis="chebyshev"` means $\sum_n a_nT_n(x)$. A physical
design interval other than $[-1,1]$ requires an affine normalization
before evaluation.

Replace `DiagonalBSGSEvaluator` with `DirectDiagonalEvaluator` to change only
the execution schedule. Both implement the same cyclic-diagonal map and
level/scale transition, although their rotation count and rounding order differ.

For a different full algorithm, write an ordinary function or callable class.
Document who owns raw-to-normalized conversion, the output target, every tensor
axis and state transition, the required key material, and the exact output
actual-scale recurrence.

::: details Complete runnable source
<<< @/../examples/17_ckks_bootstrap_logn16.py
:::
