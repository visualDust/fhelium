# Compose a bootstrap callable

Use `FullSlotBootstrap` to assemble modulus raising, forward and inverse Fourier transforms, and periodic reduction. A profile assembles these components for a common circuit; direct construction exposes the individual choices. Both forms use the same input states, evaluation-key values and prepared execution lifecycle.

For a complete executable example and measured configurations, start with [Bootstrap a full-slot CKKS ciphertext](../tutorial/composable-ckks-bootstrap). This guide assumes an `Engine`, a selected modulus chain, and an input range and precision requirement.

## Choose a profile or assemble components

```python
from fhelium.experimental import bootstrap as bs

bootstrap = bs.cosine_bootstrap(engine)
print(bootstrap.input_depth, bootstrap.output_depth)
print(bootstrap.remaining_depth, bootstrap.refresh_window)
```

The default `7/44` profile uses a degree-44 seed, seven double-angle iterations, Blocked polynomial evaluation and aggregate giant-step quotients. The degree-28 profiles use Binary evaluation and independent quotients. `3/30` uses a narrower raw-coordinate range, four forward stages, an even degree-30 seed, three double-angle iterations and three inverse stages. Its raising relation and scale schedule must support that range.

To choose every component directly:

```python
forward = bs.Radix2FourierTransformCompiler(stage_count=2)
inverse = bs.Radix2FourierTransformCompiler(stage_count=2)
linear = bs.DiagonalBSGSEvaluator(
    baby_step=16,
    hoist_baby_rotations=True,
    giant_step_reduction="aggregate",
)
reduction = bs.CosineDoubleAngleReduction(
    input_bound=1024,
    double_angle_iterations=7,
    approximator=bs.ChebyshevInterpolator(degree=44),
    evaluator=bs.BlockedChebyshevEvaluator(baby_step=8),
    fuse_input_normalization=True,
    retain_ntt=True,
)
bootstrap = bs.FullSlotBootstrap(
    engine,
    modulus_raising=bs.CenteredModRaise(),
    coeffs_to_slots_compiler=forward,
    coeffs_to_slots_evaluator=linear,
    modular_reduction=reduction,
    slots_to_coeffs_compiler=inverse,
    slots_to_coeffs_evaluator=linear,
    retain_diagonals=True,
    retain_constants=True,
)
```

This selects the component families and stage counts directly; the profile also supplies its stage-specific BSGS choices. Inspect the resulting stages and `linear.schedule_for(stage)` when choosing a partition. For a different complete topology, compose the components in an ordinary Python callable and declare its input/output states and coordinate map; `FullSlotBootstrap` implements the full-slot order described above.

## Select modulus raising and secret distributions

### Centered raising

`CenteredModRaise` extends each component's centered CRT representative from the source Q basis into the configured target Q basis. It preserves the actual scale and returns a two-component coefficient-domain standard-RNS ciphertext. The decryption phase acquires a source-modulus multiple that the Fourier and periodic-reduction stages process.

The default component is centered raising. Use the public operation independently when composing another algorithm:

```python
raised = engine.modulus_raise(source_ciphertext, target_depth=0)
```

### Sparse-secret encapsulation

Choose the main secret, intermediate secret, source basis and auxiliary moduli before constructing the raising component. `create_encapsulation_keys` prepares a low-basis switch to the intermediate secret and a guard-scaled return hint under the main secret. Execution returns to the main-secret relation before CoeffsToSlots.

The following assumes `auxiliary` contains the chosen `low_p_moduli`, `return_prime` and `return_p_moduli`, and `target_scales` contains one arithmetic target per depth:

```python
secret_key = engine.create_secret_key(hamming_weight=engine.ring_dimension // 2)
intermediate = engine.create_secret_key(hamming_weight=32)
low_key, return_key = engine.create_encapsulation_keys(
    secret_key,
    intermediate,
    source_depth=engine.max_depth,
    **auxiliary,
)
raising = bs.EncapsulatedModRaise(low_key, return_key, **auxiliary)
del intermediate

bootstrap = bs.cosine_bootstrap(
    engine,
    profile="3/30",
    modulus_raising=raising,
    source_depth=engine.max_depth,
    input_depth=engine.max_depth,
    target_scales=target_scales,
)
```

A complete numerical recipe is supplied by `examples/_bootstrap_encapsulation_parameters.py`; the tutorial shows how to load it. `create_secret_key(hamming_weight=h)` selects a uniform size-$h$ support and independent signs. Omitting the weight selects independent ternary coefficients. To supply another distribution, use `prepare_secret_key(coefficients)`; `convert_secret_key` re-expresses the centered source-basis polynomial in a target configuration and word format.

The raising component retains ordinary evaluation-key values. It does not retain the secrets or generate them during execution. Hamming weight, low and return moduli, the reducer bound and approximation schedule are separate mathematical choices. Assess their joint key relations under the intended security assumptions, as described in [Sparse-secret encapsulation](../developer/security#sparse-secret-encapsulation).

## Set entry coordinates and actual scales

Choose `source_depth` for the low Q basis and `modulus_raise_target_depth` for the larger Q basis. `input_depth` may equal `source_depth` or its immediately preceding depth. The preceding-group entry performs integer scale lifting and one rescale; direct source-basis entry establishes the normalized coordinate without that transition.

`input_scale` sets the core coordinate scale. Entry returns a compensation ratio, and finish applies it to the output scale. A `target_scales` sequence supplies coefficient-product targets for every public depth. It does not overwrite a ciphertext's actual scale after an operation.

For one diagonal stage, let the input scale be $\Delta_{in}$, the prepared diagonal scale be $\Delta_p$ and the dropped Q group's product be $M_d$. The stage output has

$$
\Delta_{out}=\frac{\Delta_{in}\Delta_p}{M_d}.
$$

The entire group product is used even when a group contains several primes. Choose prime groups, Fourier stage counts and polynomial depth together, then inspect `output_depth`, `remaining_depth` and `refresh_window`.

`Radix2FourierTransformCompiler.scale_stage` selects the stage carrying the transform's numerical factor. `FullSlotBootstrap.normalization_stage` can fold slot-count normalization into a chosen forward stage. The subsequent coefficient multiplication still consumes its declared depth for scale alignment. The `3/30` profile places both factors in the last forward stage.

### Select the lift and internal scales together

`modup_method="exact"` is the default key-switch lift and uses the canonical Garner representative. `"normalized_crt"` selects a normalized linear CRT lift, while `"crt"` selects its raw form. `DiagonalBSGSEvaluator(modup_method=...)` can override the arithmetic choice for a linear transform. Encapsulation's low-basis switch remains exact.

All three lifts preserve source residues, but their integer representatives multiply the evaluation-key error differently. For digit product $D$ and $w$ source rows, the normalized lift lies in $[0,wD)$; the raw lift can be larger. Select a CRT alternative only with a noise budget that accounts for P, digit supports, actual scale and the full circuit's required precision. Validate the complete encrypted output rather than treating residue congruence as an accuracy guarantee.

`ScaledModRaise(inner, multiplier=k)` applies the chosen raising component and then represents the same message as `(k*c, k*scale)` through an actual integer payload multiplication. It consumes no additional Q group and does not reduce existing decoded error. Choose `target_scales` and diagonal scales consistently so subsequent rescaling uses the intended prime budget. A larger internal scale can reduce the normalized contribution of later arithmetic noise, but does not by itself establish a faster or more accurate Bootstrap.

## Establish the reduction coordinates

The raw branch coordinate is $r$, with precondition $|r|\le B$. The normalized polynomial coordinate is $x=r/B$. Cosine reduction targets

$$
\rho_B(x)=\frac{\sin(\pi Bx)}{\pi}.
$$

`reference(values)` takes normalized $x$. With `fuse_input_normalization=True`, Fourier preparation incorporates division by $B$, and `evaluate` also receives normalized coordinates. Otherwise the reducer applies the division itself. Its `seed_shift` is a separate polynomial-coordinate translation applied consistently by the encrypted evaluator and reference.

The encrypted raw coordinate includes the modulus-raising integer phase. Determine its range from the selected construction rather than equating it with the original message amplitude. Evaluate the complete clear reference and encrypted result on the intended message class.

## Select polynomial evaluation

`PolynomialApproximation` stores ascending coefficients. Power basis means $\sum_n a_nx^n$; Chebyshev basis means $\sum_n a_nT_n(x)$. Apply any affine map from a physical design interval to the coefficient coordinate before evaluation and include its depth cost.

Use Binary evaluation for a shared binary basis, Blocked evaluation for coefficient blocks with shared baby/giant polynomials, or Even evaluation to factor an even Chebyshev series through $T_2$:

```python
blocked = bs.BlockedChebyshevEvaluator(baby_step=8)
plan = blocked.plan(reduction.polynomial)
print(plan.required_depths)
print(blocked.operation_inventory(reduction.polynomial))

even = bs.EvenChebyshevEvaluator(blocked)
```

Supply the chosen evaluator when constructing the reduction or through the profile's `polynomial_evaluator` argument. Even decomposition requires zero odd coefficients; exact trailing zeros do not add a $T_2$ evaluation to a constant polynomial. Replacing the evaluator retains the polynomial coefficients but changes arithmetic ordering, rounding and key-switch noise.

`BootstrapArithmetic.for_input(value, depths)` derives the basis scale recurrence from the actual input. Scalar plaintext preparation requires `max(abs(real(c)), abs(imag(c))) * scale < 2**63`; addition uses the ciphertext's actual scale and multiplication uses the selected plaintext scale. Plan a real depth transition when a chosen coefficient scale cannot be encoded, rather than changing the recorded scale without arithmetic.

## Choose the linear execution schedule

The compiler supplies cyclic-diagonal maps

$$
L(x)=\sum_k d_k\mathbin{\odot}\operatorname{Rot}_k(x).
$$

A `DirectDiagonalEvaluator` performs the direct sum. `DiagonalBSGSEvaluator` partitions offsets into baby and giant steps and reuses the corresponding rotations. Both implement the map and its declared depth/scale transition; their key inventories and rounding order differ.

```python
linear = bs.DiagonalBSGSEvaluator(
    baby_step=16,
    hoist_baby_rotations=True,
    rotation_schedule="auto",
    group_batch_size=16,
    giant_step_reduction="aggregate",
)
```

`group_major` retains reusable babies and finishes bounded blocks of groups. `baby_major` consumes babies one at a time into retained group sums. The automatic schedule chooses between these traversals from the stage's group and baby extents. `baby_steps_by_transform` supplies stage-specific partitions, and `group_batch_size` bounds simultaneous group outputs.

`giant_step_reduction="aggregate"` streams unrounded giant key products and performs one shared quotient by the auxiliary modulus product and complete dropped Q group. `"independent"` completes each giant's rescale and key switch before adding the Q outputs. Direct giant keys can enter the shared quotient immediately; composed rotations execute their prefixes first. Select this numerical schedule together with the available key inventory and workspace.

For ordinary Eager composition, `engine.sum_rotations(terms, rescale=True)` consumes `(ciphertext, rotation_key)` pairs; `None` denotes an unrotated contribution. The method owns each consumed contribution so the producer may reuse its input buffer. Callable capture represents the sum with its actual key operands. For rewriting an existing Program, insert `AggregateRotationSumsPass` into an ordinary pipeline:

```python
from fhelium import compile as fc

pipeline = fc.default_lower_and_fuse_pipeline().before(
    "rotation-hoisting", fc.AggregateRotationSumsPass()
)
```

## Provision a shared key inventory and execute

Create an application-owned inventory, or reuse one already serving another compatible consumer:

```python
from fhelium.values import EvaluationKeySet

keys = EvaluationKeySet()
bootstrap.create_evaluation_keys(
    secret_key,
    into=keys,
    rotation_key_strategy="all_required",
    device=ciphertext.device,
)
refreshed = bootstrap(ciphertext, evaluation_keys=keys)
```

`create_evaluation_keys` returns the supplied inventory, reuses compatible groups and generates missing rotation, relinearization and conjugation keys. It considers additional storage across the composition's consumers, including reuse of one key across stages. Engine factories remain available to generate individual key values; add loaded or separately generated values with `keys.add(key)` or `keys.update(iterable)`.

The default `all_required` strategy supplies direct transform rotations. `signed_power_of_two` composes otherwise unavailable rotations from a smaller basis while reusing available direct keys, and `signed_power_of_two_with_baby_steps` additionally supplies direct baby steps. Inspect `key_steps(strategy)` and `evaluation_key_requirements(strategy)` before generation. A shared inventory can contain several layouts and devices for one step. Selection uses the active Q supports, P basis, physical rows, ring dimension, dtype and consuming device; hoisted groups require compatible layouts as a group.

The application maintains the inventory's parameter context and main-secret relation. Encapsulation's low and return keys are supplied through the raising component with their own auxiliary bases and intermediate-secret relation. Preparation, capture and linking consume already provisioned keys.

For `message_domain="real"`, supply real slot messages and the generator-five radix-2 reconstruction. Complex execution can batch its two modular branches through `batch_modular_branches=True`. The named cosine composition defaults to `product_implementation="auto"` and `product_chunk_size=16`. Selection follows represented data and device; `None` requests native weighted products, and a registered implementation name requests that implementation. The chunk size bounds the product expression size.

## Prepare repeated Eager or compiled execution

`retain_diagonals` and `retain_constants` control ordinary-call cache retention. Constants retain their first encoded samples, including stochastic rounding. `compress_diagonals=True` losslessly compresses eligible encoded integers. These controls govern material residency independently of the temporary BSGS traversal.

Prepare the complete refresh for one input state and device:

```python
from fhelium import compile as fc
from fhelium.runtime import CudaGraphProgram

prepared = bootstrap.prepare(ciphertext, evaluation_keys=keys)
reference_output = prepared(ciphertext)
optimized = fc.compile(prepared)
optimized.prepare(ciphertext)
graph = CudaGraphProgram.capture(
    optimized, example_inputs=(ciphertext,), warmup=1
)
try:
    refreshed = graph.replay(ciphertext, copy_output=True, synchronize=True)
finally:
    graph.close()
    prepared.close()
```

Material discovery records each selected key in ordinary dictionaries beside the encoded materials. The prepared callable retains those references, the input signature and selected components; it releases the example payload and temporary result. Adding candidates to `keys`, removing members or clearing it leaves existing prepared bindings unchanged. Prepare again to select different keys. The selected key Tensors remain live, so compatible in-place data changes are visible to both Eager and linked execution.

`PreparedBootstrap` supplies the same entry, core and finish computation to Eager execution and Compile capture. Both paths share the encoded samples. Supply a custom pipeline or Backend through `fc.compile` when needed. For direct control of normalized coordinates, use `entry`, `prepare_core` and `finish`. Close Graphs before their prepared owner; linked executables retain their own material references.

## Validate the selected composition

Check the input's component count, Q rows, coefficient domain and standard residues; inspect output depth and actual scale; and compare decrypted values against both the input and the selected polynomial reference. Separating approximation error from execution error makes the result interpretable when changing the raising relation, polynomial, scale plan or quotient schedule. Apply the application's precision requirement to the complete refreshed value.

The [tutorial performance table](../tutorial/composable-ckks-bootstrap#_6-performance-and-material-footprint) reports circuit choices with their timing and material scope. The [developer guide](../developer/composable-ckks-bootstrap) describes the component interfaces and numerical ownership.
