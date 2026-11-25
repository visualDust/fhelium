# Compose a bootstrap plan

`fhelium.bootstrap.plan` assembles modulus raising, CoeffsToSlots, periodic reduction (EvalMod) and SlotsToCoeffs into a `BootstrapPlan`. A named profile supplies a common circuit; keyword arguments and `dataclasses.replace` replace individual components. Every plan uses the same key provisioning, refresh and preparation functions.

For a complete executable example and measured configurations, start with [Bootstrap a full-slot CKKS ciphertext](../tutorial/composable-ckks-bootstrap). This guide assumes an `Engine` with Galois generator 5, a selected modulus chain, and an input range and precision requirement.

## Choose a profile or replace components

```python
from fhelium import bootstrap

plan = bootstrap.plan(engine, "3/38", intermediate_weight=48)
for stage in plan.stages():
    print(stage.name, stage.input_depth, stage.output_depth, stage.output_scale)
print(plan.output_depth, plan.refresh_window)
```

The plan's source depth is `engine.max_depth`, the raise targets depth 0, and the output depth equals the number of consumed depths. `"3/38"` uses $B=33$, four CoeffsToSlots stages with the slot-count normalization folded into the last, a degree-38 cosine expansion, three double-angle iterations evaluated by `RecursiveChebyshevPlan` and three SlotsToCoeffs stages. `"7/44"` uses $B=1024$, two stages in each transform, a degree-44 interpolated seed, seven double-angle iterations and Blocked evaluation. The degree-28 profiles use Binary evaluation and independent giant-step quotients. `"3/30"` uses $B=32$ with an even degree-30 seed.

To choose the periodic reduction and the linear schedules directly:

```python
from fhelium.algorithms.linear import BsgsPlan
from fhelium.algorithms.polynomial import BlockedChebyshevPlan, ChebyshevInterpolator
from fhelium.bootstrap.periodic import CosineDoubleAngle

eval_mod = CosineDoubleAngle(
    input_bound=1024,
    double_angle_iterations=7,
    approximator=ChebyshevInterpolator(degree=44),
    polynomial_plan=BlockedChebyshevPlan(baby_step=8),
    fuse_input_normalization=True,
    retain_ntt=True,
)
linear = BsgsPlan(baby_step=16, giant_step_reduction="aggregate")
plan = bootstrap.plan(
    engine,
    "7/44",
    modulus_raise=bootstrap.Centered(),
    eval_mod=eval_mod,
    coeffs_to_slots_plan=linear,
    slots_to_coeffs_plan=linear,
)
```

The profile name still selects the Fourier stage counts and the input scale; `eval_mod`, `polynomial_plan`, `coeffs_to_slots_plan` and `slots_to_coeffs_plan` replace the corresponding components. The plan's `coeffs_to_slots` and `slots_to_coeffs` fields hold the `LinearTransform` stages produced by `fhelium.algorithms.linear.radix2_dft`; replace them with `dataclasses.replace` to use another factorization. For a different stage order, compose the step functions (`modulus_raise`, `coeffs_to_slots`, `split_real_imag`, `eval_mod`, `merge_real_imag`, `slots_to_coeffs`) in an ordinary Python callable.

## Select modulus raising and secret distributions

### Centered raising

`bootstrap.Centered()` extends each component's centered CRT representative from the source Q basis $q_b$ to the target Q basis. It preserves the actual scale and returns a two-component coefficient-domain standard-RNS ciphertext that decrypts to $m+q_bI$ with $|I|\le(\lVert s\rVert_1+1)/2$. The Fourier and periodic-reduction stages remove the multiple $q_bI$. The public operation is also available on its own:

```python
raised = engine.modulus_raise(source_ciphertext, target_depth=0)
```

### Sparse-secret encapsulation

`bootstrap.Encapsulated(intermediate_weight=h)`, or `intermediate_weight=h` in `plan`, key-switches the source ciphertext from the main secret $s$ to an intermediate secret $t$ of Hamming weight $h$ over a small auxiliary basis, raises the centered components, and switches back to $s$ with a guard-scaled return key. The overflow then satisfies $|I|\le(h+1)/2$ independently of the weight of $s$. The auxiliary primes are the largest primes of the configured P catalog: one low prime, one guard prime and further return primes until their product reaches $q_b$.

`create_evaluation_keys` samples $t$, adds the low and return `EncapsulationKey` values to the inventory, and discards $t$. To provision them separately, call `engine.create_encapsulation_keys(secret_key, intermediate, source_depth=..., low_p_moduli=..., return_prime=..., return_p_moduli=...)` with a secret from `engine.create_secret_key(hamming_weight=h)`. `create_secret_key(hamming_weight=h)` selects a uniform size-$h$ support with independent signs; omitting the weight selects independent ternary coefficients.

Hamming weight, auxiliary moduli, the reducer bound and approximation schedule are separate mathematical choices. Assess their joint key relations under the intended security assumptions, as described in [Sparse-secret encapsulation](../developer/security#sparse-secret-encapsulation).

## Set entry coordinates and actual scales

`input_scale` sets the coordinate scale $\Delta_0$ of the core. An input at actual scale $\Delta_b<\Delta_0$ is multiplied by the integer $k=\max(1,\operatorname{round}(\Delta_0/\Delta_b))$, which keeps depth; the output then records scale $s_{out}k\Delta_b/\Delta_0$ so that it represents the input message. `target_scales` supplies one arithmetic target per depth; `"3/38"` derives targets that keep EvalMod near its product fixed point. A target does not overwrite a ciphertext's actual scale after an operation.

For one diagonal stage with input scale $\Delta_{in}$, prepared diagonal scale $\Delta_p$ and rescale ratio $\rho_d$ of the stage's depth, the output has

$$
\Delta_{out}=\frac{\Delta_{in}\Delta_p}{\rho_d}.
$$

On the `-boot` presets, $\rho_d$ is the ratio of the dropped and added Q products between the two depth bases, which may contain several primes. Choose depth bases, Fourier stage counts and polynomial depth together, then inspect `plan.stages()`, `output_depth` and `refresh_window`.

### Select the basis conversion

`basis_conversion` selects both key-switch basis conversions. `"exact"` uses the Garner digit lift and exact ModDown; `"crt"` uses the raw CRT digit lift with exact ModDown; `"normalized_crt"` (the `"3/38"` default) uses centered CRT sums for the digit lift and the ModDown remainder. `BsgsPlan(basis_conversion=...)` overrides the plan's choice for one linear transform.

All three lifts preserve source residues, but their integer representatives multiply the evaluation-key error differently. For digit product $D$ and $w$ source rows, the normalized lift lies in $[-wD/2,wD/2)$; the raw lift can be larger. Select a CRT alternative with a noise budget that accounts for P, digit supports, actual scale and the circuit's required precision, and validate the encrypted output.

## Establish the reduction coordinates

The raw branch coordinate is $r$, with precondition $|r|\le B$. The normalized polynomial coordinate is $x=r/B$. Cosine reduction targets

$$
\rho_B(x)=\frac{\sin(\pi Bx)}{\pi}.
$$

`reference(values)` takes normalized $x$. With `fuse_input_normalization=True`, CoeffsToSlots incorporates the division by $B$ and the reduction receives normalized coordinates; otherwise the reduction applies the division itself. `seed_shift` translates the seed polynomial coordinate consistently in the encrypted evaluation and the reference.

The encrypted raw coordinate includes the modulus-raising integer phase. Determine its range from the selected raise rather than from the message amplitude, and evaluate the complete clear reference and encrypted result on the intended message class.

## Select polynomial evaluation

`Polynomial` stores ascending coefficients in the power or Chebyshev basis over an interval. `CosineExpansion(degree)` expands the cosine seed analytically through Jacobi–Anger coefficients; `ChebyshevInterpolator(degree)` interpolates at Chebyshev nodes. The evaluation plans are:

| Plan | Schedule |
|---|---|
| `BinaryChebyshevPlan` | Shared binary Chebyshev basis |
| `BlockedChebyshevPlan(baby_step)` | Coefficient blocks over shared baby and giant polynomials |
| `EvenChebyshevPlan(inner)` | Even series through $T_{2j}(x)=T_j(T_2(x))$ with an inner plan |
| `RecursiveChebyshevPlan` | Recursive division by giant Chebyshev polynomials |
| `HornerPlan`, `BalancedPowerPlan`, `PatersonStockmeyerPlan` | Power-basis schedules |

```python
from fhelium.algorithms.polynomial import BlockedChebyshevPlan

blocked = BlockedChebyshevPlan(baby_step=8)
print(blocked.required_depths(eval_mod.polynomial))
print(blocked.operation_inventory(eval_mod.polynomial))
```

Supply a plan through `polynomial_plan=` or within `CosineDoubleAngle`. Replacing the plan retains the polynomial coefficients and changes arithmetic ordering, rounding and depth.

## Choose the linear execution schedule

The CoeffsToSlots and SlotsToCoeffs stages are cyclic-diagonal maps

$$
L(x)=\sum_k d_k\mathbin{\odot}\operatorname{Rot}_k(x).
$$

`DirectPlan` evaluates the direct sum. `BsgsPlan` partitions offsets into baby and giant steps and reuses the corresponding rotations:

```python
from fhelium.algorithms.linear import BsgsPlan

linear = BsgsPlan(
    baby_step=16,
    hoist_baby_rotations=True,
    rotation_schedule="auto",
    group_batch_size=16,
    giant_step_reduction="aggregate",
)
```

`group_major` retains reusable babies and finishes bounded blocks of groups; `baby_major` consumes babies one at a time into retained group sums; `auto` chooses from the stage's group and baby extents. `baby_steps_by_transform` supplies stage-specific partitions by stage name, and `group_batch_size` bounds simultaneous group outputs.

`giant_step_reduction="aggregate"` streams unrounded giant key products and performs one shared quotient by the auxiliary modulus product and the dropped Q product. `"independent"` completes each giant's rescale and key switch before adding the Q outputs.

For ordinary Eager composition, `engine.sum_rotations(terms, rescale=True)` consumes `(ciphertext, rotation_key)` pairs; `None` denotes an unrotated contribution. To rewrite an existing Program, insert `AggregateRotationSumsPass` into a pipeline:

```python
from fhelium import compile as fc

pipeline = fc.default_lower_and_fuse_pipeline().before(
    "rotation-hoisting", fc.AggregateRotationSumsPass()
)
```

## Provision keys and execute

```python
from fhelium.values import EvaluationKeySet

keys = EvaluationKeySet()
bootstrap.create_evaluation_keys(
    engine, plan, secret_key, into=keys, rotation_key_strategy="all_required"
)
refreshed = bootstrap.bootstrap(engine, ciphertext, plan, keys=keys)
```

`create_evaluation_keys` reuses compatible keys already in the inventory and generates the missing rotation, relinearization, conjugation and encapsulation keys. Each consumed depth receives the layout of `CkksConfig.key_layout(depth)`. `signed_power_of_two` composes rotations from a smaller key basis, and `signed_power_of_two_with_baby_steps` additionally supplies direct baby steps; inspect `plan.key_steps(strategy)` and `plan.key_requirements(strategy)` before generation. Selection uses the active Q supports, P basis, physical rows, ring dimension, dtype and device.

For `message_domain="real"`, supply real slot messages. Complex execution batches its two reduction branches when `batch_branches=True`. `product_implementation="auto"` with `product_chunk_size=16` selects the weighted-product implementation from represented data and device; `None` requests native weighted products. `compress_diagonals=True` losslessly compresses eligible encoded integers.

## Prepare repeated Eager or compiled execution

```python
from fhelium import compile as fc

prepared = bootstrap.prepare(engine, plan, ciphertext, keys=keys)
compiled = fc.compile(prepared, graph=True)
try:
    refreshed = compiled(ciphertext)
finally:
    compiled.clear()
    prepared.close()
```

`prepare` encodes the diagonals and constants for the input signature of its `like` argument and selects keys once. Later changes to `keys` leave the prepared bindings unchanged; prepare again to select different keys. Eager calls and Compile capture of the `PreparedBootstrap` share the encoded samples. Supply a custom pipeline or Backend through `fc.compile` when needed. Clear compiled callables before closing the prepared owner.

## Validate the selected composition

Check the output depth and actual scale, and compare decrypted values against the input and the selected polynomial reference. Separating approximation error from execution error makes the result interpretable when changing the raising relation, polynomial, scale plan or quotient schedule. Apply the application's precision requirement to the complete refreshed value.

The [tutorial performance table](../tutorial/composable-ckks-bootstrap#_6-performance-and-material-footprint) reports measured configurations. The [developer guide](../developer/composable-ckks-bootstrap) describes the component interfaces and numerical ownership.
