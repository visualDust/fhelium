# Compose a bootstrap callable

Compose CKKS bootstrapping through the experimental public component interfaces. Use `FullSlotBootstrap` with the standard full-slot topology and configurable mathematical strategies. Direct Python composition supports custom topologies.

## Establish the coordinate convention

A periodic reduction has two coordinates:

- raw branch coordinate $r$, with the application precondition $|r|\le B$;
- normalized polynomial coordinate $x=r/B\in[-1,1]$, where
  $B=$ `input_bound`.

The built-in target is

$$
\rho_B(x)=\frac{\sin(\pi Bx)}{\pi}
          =\frac{\sin(\pi r)}{\pi}.
$$

`reference(values)` always takes normalized $x$. `evaluate(...)` takes raw $r$
when `fuse_input_normalization=False`; it takes already normalized $x$ when
fusion is enabled. `FullSlotBootstrap` honors fusion by folding $1/B$ into the
first CoeffsToSlots numerical stage.

## Assemble the built-in topology

```python
from fhelium.experimental import bootstrap as bs

compiler = bs.Radix2FourierTransformCompiler(stage_count=2)
linear_evaluator = bs.DiagonalBSGSEvaluator(
    baby_step=16,
    hoist_baby_rotations=True,
)
modular_reduction = bs.CosineDoubleAngleReduction(
    input_bound=1024,
    double_angle_iterations=7,
    approximator=bs.ChebyshevInterpolator(degree=44),
    evaluator=bs.BinaryDecompositionChebyshevEvaluator(skip_near_zero=1e-15),
    fuse_input_normalization=True,
)

bootstrap = bs.FullSlotBootstrap(
    engine,
    coeffs_to_slots_compiler=compiler,
    coeffs_to_slots_evaluator=linear_evaluator,
    modular_reduction=modular_reduction,
    slots_to_coeffs_compiler=compiler,
    slots_to_coeffs_evaluator=linear_evaluator,
    modulus_raise_target_depth=0,
    retain_diagonals=False,
)
```

Before using this object, establish all of the following:

1. the input is a two-component coefficient-domain standard-RNS Q ciphertext
   at `bootstrap.input_depth`, with input precision suitable for the circuit;
2. every batch member uses all $S=N/2$ slots and the active `prime_ids`;
3. both raw branch coordinates lie within `[-input_bound, input_bound]`;
4. the selected polynomial degree and evaluator meet the application's error
   model;
5. the modulus chain accommodates `bootstrap.output_depth`;
6. the supplied `EvaluationKeySet` contains the reported rotations and the
   compatible relinearization and conjugation capabilities.

Construction checks structural-base/default-scale proximity, transform slot
counts, target depth, and depth. It cannot inspect the encrypted coordinate
range or certify an application error bound.

## Replace BSGS with direct evaluation

The compiler synthesizes each transform; the evaluator decides how its stages
are executed. Replacing BSGS with direct diagonal evaluation changes only the
evaluator argument:

```python
direct = bs.FullSlotBootstrap(
    engine,
    coeffs_to_slots_compiler=compiler,
    coeffs_to_slots_evaluator=bs.DirectDiagonalEvaluator(),
    modular_reduction=modular_reduction,
    slots_to_coeffs_compiler=compiler,
    slots_to_coeffs_evaluator=bs.DirectDiagonalEvaluator(),
)
```

For a cyclic-diagonal stage

$$
L(x)=\sum_kd_k\mathbin{\odot}\operatorname{Rot}_k(x),
$$

BSGS splits $k=g+b$ and moves the final giant rotation outside each group. This
is algebraically equivalent to direct evaluation. Both consume one Q depth per
stage. If the input scale is $\Delta_{\rm in}$ and the evaluator prepares the
diagonal at scale $\Delta_p$, the dropped Q group has product $M_d$, and the
stage uses

$
\Delta_{\rm out}=\frac{\Delta_{\rm in}\Delta_p}{M_d}.
$

`BootstrapArithmetic` selects $\Delta_p$ from the circuit's target schedule;
it is not necessarily `config.default_scale`.

They can differ in rotation inventory, operation ordering, memory use, and CKKS
rounding, so compare decoded semantics rather than requiring bitwise residues.

## Replace periodic reduction (`modular_reduction`)

```python
exponential = bs.ExponentialSquaringReduction(
    input_bound=1024,
    degree=16,
    evaluator=bs.BalancedPowerEvaluator(skip_near_zero=1e-15),
    fuse_input_normalization=True,
)

bootstrap = bs.FullSlotBootstrap(
    engine,
    coeffs_to_slots_compiler=compiler,
    coeffs_to_slots_evaluator=linear_evaluator,
    modular_reduction=exponential,
    slots_to_coeffs_compiler=compiler,
    slots_to_coeffs_evaluator=linear_evaluator,
)
```

The exponential component stores ascending power coefficients
$a_n=(i\pi)^n/n!$, squares $\log_2 B$ times, and extracts sine by conjugation.
It has its own depth and key needs. The constructor computes the resulting
output depth from the selected components.

## Respect polynomial basis conventions

`PolynomialApproximation` uses ascending coefficients:

- `basis="power"`: $p(x)=\sum_n a_nx^n$;
- `basis="chebyshev"`: $p(x)=\sum_n a_nT_n(x)$.

For an approximation designed on physical $[a,b]$, the stored coefficient
coordinate is $x=(2t-a-b)/(b-a)$. `evaluate_plaintext()` and homomorphic
evaluators do not insert this affine map. Apply the input normalization and
include the resulting depth cost in the component's declared depth budget.

Polynomial evaluators derive their basis scale recurrence from the input's
actual scale. This preserves the polynomial's coefficients and does not
require guessing a full-slot stage's target scale. For a high-degree
polynomial, inspect `arithmetic.for_input(value, depths).target_scales` when
planning precision. If the resulting scalar plaintexts cannot fit the integer
encoding range, prepare the value with `arithmetic.advance_depth(value)` and
include that real transition in the depth budget; evaluation does not add it
silently.

## Change the complete topology

A complete custom algorithm is a Python callable. `BootstrapArithmetic` owns
one Engine plus the prepared-constant cache used by Bootstrap's depth-dependent
multiplication, depth advancement, and scalar operations. Pass that owner to
component `evaluate()` methods so nested reduction and polynomial schedules
share the same material lifecycle. Application code may interleave these
components with ordinary Engine operations in any order:

```python
class MyBootstrap:
    def __init__(self, engine, reduction):
        self.engine = engine
        self.arithmetic = bs.BootstrapArithmetic(engine)
        self.reduction = reduction

    def __call__(
        self,
        ciphertext,
        *,
        rotation_keys,
        relinearization_key,
        conjugation_key,
    ):
        prepared = my_centered_raise(self.engine, ciphertext)
        slots = my_forward_transform(
            self.engine,
            prepared,
            rotation_keys=rotation_keys,
        )
        branches = my_split_branches(
            self.engine,
            slots,
            conjugation_key=conjugation_key,
        )
        reduced = [
            self.reduction.evaluate(
                self.arithmetic,
                my_normalize_raw_coordinate(branch),
                relinearization_key=relinearization_key,
            )
            for branch in branches
        ]
        return my_inverse_transform(
            self.engine,
            my_recombine_branches(reduced),
            rotation_keys=rotation_keys,
        )
```

Document the custom callable's tensor axes, depth/scale/domain/basis
transitions, raw range, normalization owner, and output target. Ordinary
dictionaries or tensors can hold caches; runtime value serialization and artifact
facilities remain available for persistence.

## Generate or supply keys

```python
from fhelium.values import EvaluationKeySet

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
refreshed = bootstrap(
    ciphertext,
    evaluation_keys=evaluation_keys,
)
```

`rotation_strategy="direct"` requests a direct key for every transform rotation.
`"power_of_two"` stores compact signed-power keys and composes missing direct
steps online. Generate or provision the relinearization and conjugation keys
independently because they serve different operations and are not part of the
rotation inventory.

## Use the versioned factories correctly

The experimental `logn16` factory names identify documented bootstrap
configurations rather than runtime validators. Their documented setup is:

```python
from fhelium.eager import Engine

config = fh.CkksConfig.parse(
    fh.Preset.slots32768_scale50_depth27_int64,
    galois_generator=5,
)
engine = Engine(config)
```

Factories do not enforce this preset and do not prove the raw branch range
or output error. Treat a different engine, range, polynomial profile, or
application tolerance as a new validation target.
