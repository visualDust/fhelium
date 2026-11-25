# CKKS bootstrap internals

This page specifies the mathematical, value-state, and actual-scale invariants of the full-slot bootstrap in `fhelium.bootstrap`, executed through `fhelium.eager.Engine`.

## Evaluation stack

```mermaid
flowchart TB
    PLAN[BootstrapPlan<br/>profile, components and scale targets]
    ALG[fhelium.algorithms<br/>linear transforms and polynomials]
    STAGES[fhelium.bootstrap stages<br/>modulus raise, CoeffsToSlots, EvalMod, SlotsToCoeffs]
    ENGINE[eager Engine operations<br/>rotate, multiply, relinearize, rescale]
    RUNTIME[RNS and NTT runtime]
    OPS[torch.ops CPU / CUDA primitives]

    PLAN --> STAGES
    ALG --> STAGES --> ENGINE --> RUNTIME --> OPS
```

`fhelium.algorithms` provides slot-wise linear transforms in cyclic-diagonal form (`fhelium.algorithms.linear`) and polynomial approximation and evaluation (`fhelium.algorithms.polynomial`). `fhelium.bootstrap` composes them with modulus raising and the periodic reduction of `fhelium.bootstrap.periodic`. Every stage is an ordinary sequence of Engine operations; the dense arithmetic reaches `fhelium_rns_ops`, `fhelium_ntt_ops` and `fhelium_ckks_ops` through the Engine. Linear stages and their evaluation plans are paired by protocol: a plan must understand its stage representation, and `required_depths` must match execution.

The source tree follows those responsibilities:

- `fhelium/bootstrap/_plan.py` defines `BootstrapPlan`, the named profiles, depth placement, default target scales and key-step queries;
- `fhelium/bootstrap/_core.py` evaluates entry, modulus raise, CoeffsToSlots, branch split, EvalMod, merge and SlotsToCoeffs, and owns `PreparedBootstrap` materials;
- `fhelium/bootstrap/_modulus_raise.py` and `_encapsulation.py` define the `Centered` and `Encapsulated` raises and the selection of encapsulation keys and auxiliary primes;
- `fhelium/bootstrap/_keys.py` derives the key consumers of a plan and adds missing keys to an inventory;
- `fhelium/bootstrap/periodic/` defines `CosineDoubleAngle` and `CosineExpansion`;
- `fhelium/algorithms/_arithmetic.py` defines `Arithmetic`, which applies depth-dependent target scales, representation choices and a shared constant table through one Engine;
- `fhelium/algorithms/linear/` contains `LinearTransform`, `DirectPlan`/`BsgsPlan`, the radix-2 factorization and key-aware execution of stages;
- `fhelium/algorithms/polynomial/` contains `Polynomial`, interpolation and the power-basis and Chebyshev evaluation plans.

Every component's `evaluate(arithmetic, ...)` method receives one `fhelium.algorithms.Arithmetic`. Nested periodic and polynomial evaluations share its target-scale schedule, representation mode and constant table. Preparation fills that table and the key and diagonal dictionaries; prepared execution repeats the same `evaluate` calls and requires existing entries. The protocols are stated in the `fhelium.algorithms.polynomial`, `fhelium.algorithms.linear` and `fhelium.bootstrap.periodic` module documentation; [Implement a bootstrap component](../how-to/implement-bootstrap-component.md) shows them in use.

## Notation

Let:

- $N=2^{\mathtt{logN}}$ and $S=N/2$;
- $D$ be `engine.max_depth`, which is `plan.source_depth`;
- $q_b$ be the product of the Q basis at depth $D$;
- $\Delta_0$ be `plan.input_scale`;
- $Q_d$ be the ordered Q basis recorded by a ciphertext's `prime_ids` at depth $d$;
- $\rho_d$ be `config.rescale_divisor(d)`, the product of the primes leaving the basis at a rescale from depth $d$ divided by the product of the primes entering it;
- $C$ and $T$ be the unscaled CoeffsToSlots and SlotsToCoeffs maps;
- $B$ be `plan.eval_mod.input_bound`;
- $F$ be `plan.eval_mod.fused_input_divisor`, either $1$ or $B$.

The radix-2 convention is

$$
T(C(a))=Sa.
$$

A plaintext reference round trip therefore factors $C$ with `scale=1.0` and $T$ with `scale=1.0 / S`.

## Entry

The input is a two-component coefficient-domain, standard-residue Q ciphertext at depth at most $D$, with axes `[component, *batch, limb, coefficient]`, coefficient extent $N$, and limb order equal to `prime_ids`. An earlier depth first reaches $D$ through `mod_switch_to_depth`. With actual scale $\Delta_b$, entry chooses

$$
k=\max\left(1,\operatorname{round}(\Delta_0/\Delta_b)\right)
$$

and multiplies the residues by the integer $k$, which keeps depth and recorded scale and represents message $km$. The identity case $k=1$ returns the input directly. A metadata view then records scale $\Delta_0$, so the fixed transform circuit operates on the coordinate $u=\tau m$ with $\tau=k\Delta_b/\Delta_0$. After refreshing $u$, the output scale is multiplied by $\tau$ to recover coordinate $m$. These two coordinate changes compensate one another. CoeffsToSlots/SlotsToCoeffs coefficients and their prepared materials are independent of the input scale.

### Arithmetic scale schedule

`plan.target_scales` holds depth-dependent targets $s_d$. Without supplied targets, the plan starts from $s_D=$ `config.scale_at(D)` and computes backward

$$
s_d=\sqrt{\rho_d s_{d+1}}.
$$

Thus a ciphertext product at scale $s_d$ rescales to $s_{d+1}$. Scalar, polynomial-coefficient, and diagonal plaintexts use scale $\rho_d s_{d+1}/\Delta_{in}$, so their products also reach $s_{d+1}$. The `"3/38"` profile instead holds the CoeffsToSlots and EvalMod targets at the largest scales whose encoded diagonals and constants stay below $2^{61}$. Every rescale retains the quotient's actual scale. Constants added to a ciphertext are encoded at that ciphertext's scale; prepared-material keys include the selected encoding scale.

Independent polynomial basis nodes use an arithmetic view anchored at the input's actual scale, $t_d=\Delta_{in}$, and propagate $t_{j+1}=t_j^2/\rho_j$. Copies of the input used in Chebyshev subtraction follow the same recurrence. Coefficient products connect those basis values to the chosen output scale. Paterson–Stockmeyer coefficient accumulators use a common scale multiple of the basis schedule, so multiplication by an unscaled basis power and addition of the next coefficient group remain consistent.

A basis scale can become too small after repeated products, requiring a scalar plaintext coefficient beyond the signed-integer encoding range. Material preparation reports that range failure instead of allowing integer overflow. `Arithmetic.advance_depth(value)` places a value on the target schedule with one declared transition; evaluation plans do not consume an undeclared depth.

## Modulus raising

### Centered extension

For every ciphertext component and polynomial coefficient, centered ModRaise chooses the unique source representative

$$
\widetilde c\in
\left[-\left\lfloor\frac{q_b}{2}\right\rfloor,
       \left\lfloor\frac{q_b}{2}\right\rfloor\right]
$$

consistent with the residue modulo $q_b$, and emits $\widetilde c\bmod q_i$ for every $q_i\in Q_{\ell_r}$, where $\ell_r=0$ is the raise target depth. It is a component-wise centered basis extension, not rescale or modulus restriction.

The transition is

```text
depth D, prime_ids Q_D, coefficient, standard, Q, two components, scale Delta_0
  ->
depth ell_r, prime_ids Q_ell_r, coefficient, standard, Q, two components,
scale Delta_0
```

Data axes and batch shape are preserved; the limb extent changes from `len(Q_D)` to `len(Q_ell_r)`. Execution uses the shared centered mixed-radix basis-extension operation, including source bases wider than a single hybrid digit. The slow `reference_centered_basis_extend()` oracle reconstructs with Python integers and returns a tensor shaped `[*batch, target_limb, coefficient]` in standard residues on the input device.

### Sparse-secret encapsulation

Let $(d_0,d_1)$ be the low-basis result of switching from main secret $s$ to intermediate secret $t$. Its digit keys satisfy $b_j+a_jt=P_{low}E_js+e_j$. Centered extension lifts $d_0$ and $d_1$ into the return basis. With guard prime $q_\star$, return auxiliary product $P_{return}$ and $D_r=q_\star P_{return}$, the return hint satisfies

$$
H_0+H_1s=D_rt+e.
$$

The output components on target Q are

$$
c'_0=\widetilde d_0+\operatorname{Round}_{D_r}(\widetilde d_1H_0),\qquad
c'_1=\operatorname{Round}_{D_r}(\widetilde d_1H_1),
$$

where $\operatorname{Round}_{D_r}$ denotes the selected rounded quotient by the complete guard/auxiliary product, reduced into target Q. Its evaluation uses NTT key products, inverse transforms and the shared trailing-basis quotient. The main-secret relation is restored before Fourier evaluation, with the added key-switch and quotient noise. Source and target depths, batch shape and actual scale follow the registered CKKS operation's state transition.

`create_secret_key(hamming_weight=h)` selects uniform support without replacement and independent signs. `create_evaluation_keys` samples $t$ with weight `intermediate_weight`, and crypto preparation converts both secrets into the low and return numerical bases and creates two `EncapsulationKey` values in the inventory. $P_{low}$ is the largest P prime, $q_\star$ the next, and $P_{return}$ the following P primes until $q_\star P_{return}\ge q_b$. Secret weight and auxiliary bases are caller choices; the associated [security model](security#sparse-secret-encapsulation) covers their joint publication.

### Numerical preparation and execution

Bootstrap key switches use the plan's `basis_conversion`, which selects both the digit lift and the ModDown remainder (see [Basis-conversion methods](../concepts/terminology-and-mathematical-model.md#basis-conversion-methods)). The `"3/38"` profile uses `"normalized_crt"`; other profiles use `"exact"`. `"crt"` and `"normalized_crt"` change evaluation-key and rounding noise, so select them with the auxiliary product, effective digit supports and actual scales, then validate complete refresh error. `BsgsPlan(basis_conversion=...)` overrides the plan's choice for one linear transform; encapsulation's low-basis switch uses exact extension.

NTT policy controls numerical table layout independently of key inventory membership. Compact int32 policies can use Shoup twiddles, with the storage and fusion constraints described in [compact NTT plans](rns-and-ntt.md#compact-plans). Montgomery twiddles remain the default. Select a table layout using complete-workload latency and retained table storage.

Centered CRT extension and trailing-prime quotients are RNS operations. CKKS Backend implementations compose the low key switch, centered extension, return key product and quotient from Tensor operands. Eager owns ciphertext state transitions and key placement; Compile preserves live key/table operands and exposes both whole-operation execution and RNS/NTT lowering. Secret sampling, basis conversion and guard-scaled hint generation belong to crypto preparation. A bootstrap plan selects a modulus-raising component and combines it with Fourier transforms, polynomial reduction and its actual-scale schedule.

## Cyclic-diagonal linear maps

A `LinearTransform` stores one CPU `complex128` vector $d_k$ with axes `[slot]` for each signed rotation offset $k$. It represents

$$
L(x)=\sum_kd_k\mathbin{\odot}\operatorname{Rot}_k(x),
$$

where `Rot_k` matches `numpy.roll(x, k)`. `reference(values)` accepts exactly one vector of shape `[slot]`, returns the same shape, and performs neither CKKS encoding nor scale or depth simulation.

`DirectPlan` rotates, plaintext-multiplies, and sums all diagonal terms, then rescales once. BSGS writes $k=g+b$ and computes the equivalent identity

$$
\operatorname{Rot}_g\left(
 \operatorname{Rot}_b(x)\mathbin{\odot}\operatorname{Rot}_{-g}(d_{g+b})
\right)
=
\operatorname{Rot}_{g+b}(x)\mathbin{\odot}d_{g+b}.
$$

`giant_step_reduction="aggregate"` streams the final giant key products into QP and accumulates each ordinary ciphertext component after multiplying by P. If $z_g$ denotes a complete unrounded contribution and $M_d$ the product of the Q primes leaving the basis at depth $d$, the result is $\lfloor(\sum_g z_g+h)/(PM_d)\rfloor\bmod Q_{d+1}$, where $h=(PM_d-1)/2$ for nearest rounding. The aggregate quotient applies to depth transitions that only remove primes. One inverse QP NTT and one quotient tail finish the stage. The independent strategy instead rescales and key-switches every giant before Q accumulation. The two schedules implement the same ideal map and output-state transition but have different rounding errors.

`backend.ckks.rotation` owns the streaming Tensor accumulator and whole-operation implementation. `RotationSumOp` exposes the same numerical operation to Compile, with live source, key and table operands. Its logical lowering exposes per-key products to existing RNS/NTT fusion and ends in `RoundedTrailingQuotientOp`. `ScaleAndPadRowsOp` supplies the P-scaled ordinary Q contribution with zero P rows. The quotient tail uses the complete product of the leaving primes, including multi-prime depth transitions.

`AggregateRotationSumsPass` optionally recognizes compatible SSA rotation/add trees and rescale transitions. The pass preserves unrelated users and rejects unknown effects, partial required state, mismatched table bindings and caller-assigned operations. It changes rounding, so it is not enabled by the default lowering recipe. Compose it with `default_lower_and_fuse_pipeline(...).before("rotation-hoisting", AggregateRotationSumsPass())`. The pass docstring records measured Graph and non-Graph performance conditions.

`BsgsPlan.rotation_schedule` controls the lifetime of shared rotations and group accumulators. `group_major` materializes the distinct baby rotations and computes up to `group_batch_size` group sums together. Each block is completed through INTT, rescale and giant rotations before producing the next block. A rotation is removed from the invocation's map after its last block consumer. `baby_major` yields one baby result at a time and updates the live group accumulators in place. With direct baby keys, the generator uses the same Backend hybrid-digit preparation and per-key product/ModDown as hoisted rotate-many, but releases each yielded result before generating the next one. Composed key inventories retain their ordinary successive keyed rotations.

The default group batch contains at most 16 groups, preserving the native 16-term/16-group weighted-sum specialization when that shape is present. `group_batch_size=1` selects one group at a time; callers may choose intermediate extents to trade shared-load execution against temporary storage. The default `auto` schedule selects baby-major when the stage has fewer giant groups than distinct baby rotations, and group-major otherwise. The groups and baby offsets come from the caller's BSGS partition; `schedule_for(stage)` exposes the decision, and callers can force either traversal. The comparison concerns two-component ciphertext storage of equal shape within one stage. It does not include key-switch scratch, encoded diagonals, or kernel-launch costs. Both traversals feed the same selected giant-step quotient schedule in increasing-offset order.

Full-slot execution releases entry coordinates and transform intermediates after branch splitting, and releases the original branch pair after packing a batched reduction. `BinaryChebyshevPlan` counts recurrence and coefficient consumers before execution. Basis nodes remain cached through their final consumer, while coefficient products are accumulated by depth as soon as their basis node is available; each depth group still receives one shared rescale. Monomial multiplication expresses the negacyclic shift, sign correction and modular reduction as functional Tensor operations. Prepared diagonals and scalar constants remain governed by the callable's retention settings.

For either plan, diagonal plaintexts are unbatched `[limb, ntt_index]` tensors in NTT/Montgomery form over the active Q basis. They broadcast over ciphertext batch axes. At depth $d$ and input scale $\Delta$, their encoding scale is $\rho_d s_{d+1}/\Delta$. One rescale therefore returns scale $s_{d+1}$, advances depth once, and moves to the basis $Q_{d+1}$. The output has two coefficient-domain standard-RNS components.

## CoeffsToSlots, branch split, and coordinates

Plan construction folds numerical factors

$$
\alpha_C=\frac{\Delta_0}{q_bF},\qquad
\alpha_T=\frac{q_b}{2\Delta_0}.
$$

The factors multiply diagonal values; they are distinct from the diagonal plaintext's encoding scale selected by the arithmetic schedule. Let $a$ denote the coefficient-coordinate value represented by the centered-raised ciphertext at metadata scale $\Delta_0$. Define

$$
w=\frac{\Delta_0}{Sq_b}C(a),\qquad
r_{\rm R}=2\operatorname{Re}(w),\qquad
r_{\rm I}=2\operatorname{Im}(w).
$$

Slot normalization contributes a factor $1/S$. The two-stage profiles apply it in a separate `normalize` step that consumes one depth; the four-stage profiles fold it into the last CoeffsToSlots stage and consume no depth for it. The represented complex coordinate is $w/F$.

Conjugation gives $\overline{w}/F$. Addition exposes $2\operatorname{Re}(w)/F$; subtraction exposes $2i\operatorname{Im}(w)/F$. Under the full-slot cyclotomic order used by the generator-5 profile, multiplication by $X^{3S}$ converts the latter to $2\operatorname{Im}(w)/F$. Therefore:

- if normalization is not fused, $F=1$ and the reducers receive raw coordinates $r_{\rm R}$ and $r_{\rm I}$;
- if normalization is fused, $F=B$ and they receive normalized coordinates $x_{\rm R}=r_{\rm R}/B$ and $x_{\rm I}=r_{\rm I}/B$.

Conjugation, branch addition/subtraction, and monomial multiplication preserve depth, actual scale, two components, Q basis, domain, residue representation, and `prime_ids`.

## Periodic reduction

The raw coordinate is $r$. The normalized polynomial coordinate is a distinct object

$$
x=r/B\in[-1,1].
$$

The built-in cosine reducer targets

$$
\rho_B(x)=\frac{\sin(\pi Bx)}{\pi}
          =\frac{\sin(\pi r)}{\pi}.
$$

For $r=2k+\epsilon$, the periodic target is $\sin(\pi\epsilon)/\pi\approx\epsilon$. Neither a reduction nor a plan can inspect encrypted $r$, so $|r|\le B$ is a caller-established precondition.

The methods consume the following inputs:

- `reference(values)` always consumes normalized $x$ and applies the fitted polynomial and recurrence directly;
- non-fused encrypted evaluation consumes raw $r$ and spends one scalar-multiply depth computing $x=r/B$;
- fused encrypted evaluation assumes the caller already supplied $x$.

### Cosine recurrence

For $R=$ `double_angle_iterations`, the fitted seed is

$$
z_0(x)\mathrel{\approx}
\pi^{-1/2^R}
\cos\left(\frac{\pi Bx}{2^R}-\frac{\pi}{2^{R+1}}\right),
$$

followed by

$$
z_j=2z_{j-1}^2-\pi^{-1/2^{R-j}},\qquad j=1,\ldots,R.
$$

For a nonzero `seed_shift` $u=x+\sigma$, the fitted seed uses the corresponding phase correction so evaluating the polynomial at $u$ preserves the same periodic target in $x$. The four-stage profiles use $\sigma=-1/(2B)$; for 3/30 this makes the degree-30 seed even. `EvenChebyshevPlan` evaluates $p(u)=q(T_2(u))$ through a degree-15 inner polynomial; constants with trailing zero coefficients delegate directly to the inner constant schedule.

The final value approximates $\rho_B(x)$. Every square uses ciphertext multiplication, relinearization, and group rescale with actual scale retained.

The reduction output is a functional two-component, coefficient-domain standard-RNS Q ciphertext at input depth plus `required_depths`, unchanged batch shape, and the corresponding active `prime_ids`. Its actual scale follows the arithmetic schedule and quotient scales; it need not equal $\Delta_0$.

## Polynomial basis and evaluator requirements

`Polynomial.coefficients` is always ascending degree:

$$
p(x)=\sum_{n=0}^{d}a_nx^n
\quad\text{or}\quad
p(x)=\sum_{n=0}^{d}a_nT_n(x).
$$

`ChebyshevInterpolator` maps a physical coordinate $t\in[a,b]$ to

$$
x=\frac{2t-(a+b)}{b-a}.
$$

Its returned coefficients are functions of $x$, even though `domain` records $[a,b]$. `evaluate_plaintext()` and the homomorphic evaluation plans do not apply this affine normalization. The caller must provide the basis coordinate and account for any depth required to compute it.

`BalancedPowerPlan` uses shared balanced powers. `BinaryChebyshevPlan` uses

$$
T_{2n}=2T_n^2-1,\qquad
T_{2n+1}=2T_nT_{n+1}-T_1.
$$

These plans accept two-component Q ciphertexts in coefficient/standard or NTT/Montgomery form. Ciphertext products use NTT/Montgomery form and produce three components. Immediate products and delayed block accumulators use `Engine.relinearize(..., rescale=True)` to restore two components and move to the next depth basis with one common quotient. The key is selected at the product depth, and the actual scale is divided by $\rho_d$. The [relinearization equations](multiplication-keyswitch-rescale.md#relinearization) define the combined numerator and independent rounding offsets. `Arithmetic.retain_ntt` controls whether intermediate products stay in NTT form. `BlockedChebyshevPlan` uses the same entry states and exposes its block schedule through `plan()`; its final block products share this transition as described below.

## Recombination, SlotsToCoeffs, and output

After periodic reduction, multiplication of the imaginary result by $X^S$ restores its imaginary placement. Branch addition represents

$$
\rho(r_{\rm R})+i\rho(r_{\rm I}),
\qquad
\rho(r)=\frac{\sin(\pi r)}{\pi}.
$$

The fixed SlotsToCoeffs factor $q_b/(2\Delta_0)$ cancels the branch-split factor for the internal coordinate $u$. Its result initially records the circuit's actual output scale. Multiplying that metadata scale by $\tau=k\Delta_b/\Delta_0$ then changes coordinates from refreshed $u$ back to $m$.

The output depth is

$$
\ell_{\rm out}=\ell_r+m_C+m_N+m_\rho+m_T,
$$

where $m_C$ and $m_T$ are the declared CoeffsToSlots and SlotsToCoeffs stage costs, $m_N\in\{0,1\}$ is `plan.normalization_depths` and $m_\rho$ is `plan.eval_mod.required_depths`. `plan.stages()` reports each placement. The output is a functional two-component coefficient-domain standard-RNS Q ciphertext with unchanged batch axes and `Q_ell_out` `prime_ids`.

## Key provisioning, selection and material ownership

`EvaluationKeySet` retains ordinary `RotationKey`, `RelinearizationKey` and `ConjugationKey` references supplied by an application. Multiple layouts and devices for the same step can coexist. Object identity deduplicates insertion; `remove` and `clear` release inventory references. The consumer's active basis and execution state determine its selection. The application maintains parameter provenance and secret relations.

`required_rotations` is the union of direct or BSGS transform offsets. `key_steps("all_required")` returns direct steps; `key_steps("signed_power_of_two")` returns a standalone signed-power basis; `key_steps("signed_power_of_two_with_baby_steps")` adds direct babies from the actual BSGS partitions. The queries normalize and deduplicate steps, exclude zero and return no key data. `key_requirements` additionally reports conjugation and the selected reduction's relinearization requirement. The reduction stage is slotwise and introduces no transform rotations.

`bootstrap.create_evaluation_keys(engine, plan, secret_key, into=pool, rotation_key_strategy=..., device=...)` provisions the selected capabilities through Engine primitive factories and returns `pool`. Existing compatible groups are reused by reference. Missing material is chosen across the composition's consumers, accounting for shared steps, active Q coverage and additional key storage. Requested direct-baby hoists constrain newly completed groups to a common stored layout. Already complete mixed coverage is reused without generation and can execute independently; disabling baby hoisting removes that constraint from new generation. A compact layout for one stage is useful only in the context of the resulting shared inventory. For `Encapsulated` raising it also adds the low and return `EncapsulationKey` values.

Selection reads the actual `digit_prime_ids`, ordered `prime_ids`, P basis, ring dimension, residue dtype and device. Independent rotations and composed prefixes can use different stored layouts. Hoisted babies share a decomposition; aggregate giants share the layout of their final key switches. Preparation represents each logical rotation as an ordered tuple of primitive keys. When available baby paths cannot share a hoisted decomposition, the evaluator executes those paths independently. Relinearization selection uses the aligned product's active basis. A stored key's surviving digit supports retain their original Tensor plane indices when execution restricts Q. Fresh key generation can partition the smaller active Q basis independently.

With all execution facts known, rotation groups are ranked by primitive key-switch application count, total active digit applications and bytes of distinct selected key payloads, with insertion order breaking ties. Repeated use of a primitive key contributes repeated applications but one retained payload. Relinearization and conjugation rank by active digits, stored payload and insertion order. This deterministic selection rule operates on represented work and storage; measured latency remains a property of the complete execution. Unknown facts permit only a unique compatible choice.

During material discovery, linear stages, polynomial products and conjugations record the selected key references in ordinary dictionaries. Built-in polynomial and reduction schedules expose their actual relinearization depths, including normalization and delayed products; undeclared custom schedules require full-basis coverage. A directly supplied key is retained for that call; other calls at the same depth keep their inventory-selected keys. A prepared core retains those selections and its encoded materials. Repeated prepared execution reads the dictionaries, and capture passes the selected payloads into ordinary operations. Backend implementations consume actual key/table Tensors. Appending to, removing from or clearing the shared pool does not change a prepared computation; a new preparation chooses from the revised inventory.

A `PreparedBootstrap` retains encoded diagonal and constant dictionaries, selected primitive rotation paths and the keys consumed by polynomial products, conjugations and the encapsulated raise. `diagonal_bytes` and `constant_bytes` report its material storage. A call of `bootstrap.bootstrap` without `prepared=` builds temporary materials for that call. RNS/NTT and modulus-raising numerical tables belong to Engine resource preparation. Each prepared owner has its own materials and closes after its Graph users.

`bootstrap.plan` selects components through its profile argument. The default is `"3/38"`: four CoeffsToSlots stages, a degree-38 `CosineExpansion` seed evaluated by `RecursiveChebyshevPlan`, three SlotsToCoeffs stages, aggregate quotients and `normalized_crt` basis conversion. The wide-range profiles (`"7/44"`, `"8/28"`, `"8/28-s2c3"`) pair with a centered raise; the narrow-range profiles require a raise that bounds the integer phase, such as `Encapsulated`. Construction requires Galois generator 5 and sufficient depth for the declared component costs. The raw-coordinate range and output precision are properties of the chosen construction and input class.

The four-stage profiles partition their coefficient-to-slot stages with power-of-two baby widths derived from each collapsed radix-layer group. If a group contains $w$ layers and $r$ layers remain below it, its rotation unit is $2^r$. The first stage uses a baby width of $2^{\lceil w/2\rceil}$; subsequent stages use $2^{\lceil(w+1)/2\rceil}$ to account for their signed diagonal range. The step is the rotation unit times that width, and traversal is group-major. At $N=65536$, the resulting steps are 8192, 1024, 64 and 4. A supplied `coeffs_to_slots_plan` retains its own partition and traversal.

## Block-factorized Chebyshev evaluation

`BlockedChebyshevPlan.plan()` owns coefficient factorization, the binary dependency graph, critical-path depth and operation inventory. Descending elimination applies

$$
T_{jm+i}(x)=2T_{jm}(x)T_i(x)-T_{jm-i}(x),\qquad
P(x)=\sum_j A_j(x)T_{jm}(x),\quad \deg A_j<m.
$$

Giants are built from the original coordinate through the binary recurrence. All basis advancement follows one `for_input` scale schedule. Where $s_{d+1}=(s_d s_d)/\rho_d$, encoded-one advancement selects plaintext scale $s_d$ directly, giving the same binary64 multiplication and division as a basis product.

Let the final depth be $f$, baby scales be $S_k$ and the aligned giant scale at $f-1$ be $G$. Each baby is restricted to depth $f-2$ without changing its scale. Its coefficient plaintext uses scale $p_k=C/S_k$, where the represented products $S_kp_k$ must all equal the common scale $C$. The block's nonconstant terms are accumulated and rescaled once to $A=C/\rho_{f-2}$; its constant coefficient is then added at scale $A$. Products $A_jT_{jm}$ for $j>0$ have three components at the common actual scale $AG$ and are accumulated before one relinearization and rescale. The $j=0$ block is multiplied by encoded one at plaintext scale $G$ and rescaled. Both paths compute the output scale $(AG)/\rho_{f-1}$ in the same operation order. Basis advances shared by recurrence and giant consumers reuse one result for each basis index and target depth. This construction consumes the plan's declared depths and handles complex coefficients and messages without a real projection.

The coefficient-scale choice retains the preferred $C$ when every binary64 division and multiplication round-trips to it. Otherwise it derives a common scale from the basis significands and checks the represented products before constructing the block. That scale propagates through the remaining polynomial, double-angle and Fourier operations. Scalar encoding limits apply to each selected coefficient scale and to constants at their addition scale.

The reordered rounding and key switching preserve the polynomial but need not preserve residue arrays or the previous noise realization. The state schedule and approximation error therefore have separate validation obligations.

## Compact scalar materials

A constant slot message has period one. The Engine's compact encoder embeds its polynomial as $a(X^{N/U})$, using $U=2$ for generator5 and $U=4$ for generator3. The actual ring roots, Q rows, scale and stochastic-rounding stream remain those of the Engine. Additions use the ciphertext's current scale and domain; multiplication plaintexts use the arithmetic's scheduled scale and NTT/Montgomery form.

`constant_cache` keys distinguish Engine ownership, scalar value, actual plaintext scale, depth/prime rows, representation, dtype and device. Monomial modulus tables share this owner-aware namespace. The cache stores an encoded rounding sample: first preparation advances the rounding state, reuse does not, and clearing permits new samples. A disabled cache uses device-side scalar message construction that can be warmed and captured without temporary host-list transfers.

Scalar preparation checks `max(abs(real(c)), abs(imag(c))) * scale < 2**63` before signed-integer coefficient conversion. For addition, `scale` is the ciphertext’s actual scale; for multiplication, it is the selected plaintext scale. Inputs outside this preparation bound raise `ValueError`.

## Prepared callable execution

`bootstrap.prepare(engine, plan, like, keys=...)` returns a `PreparedBootstrap` callable containing entry, the prepared core and output compensation. It evaluates the plan once on `like` with independent material dictionaries: each key-consuming operation records its actual selection from the shared inventory, and first-use encoding advances the encoding RNG for the discovered materials. The prepared result holds the selected key references and numerical materials; the prototype payload and temporary result are released.

Eager execution and Compile source capture of the `PreparedBootstrap` use the same Python body. Callable-object capture reads bound materials lazily; the existing pipeline receives ordinary CKKS, RNS, NTT and Tensor operations. Compilation preserves the prepared encoded samples and does not generate keys. Output compensation creates a new metadata envelope with scale multiplied by $\tau$; its payload shares the core output's storage, so repeated compensation leaves a borrowed Graph output's own metadata unchanged.

The callable accepts the prepared value signature (`input_signature`) and device, including actual scale, depth, prime identities, domains, dtype, shape and layout. Storage strides may differ when a Graph buffer packs a view. Prepared scalar, monomial, diagonal and modulus-raise lookups require existing entries. No new encoding sample or data-dependent compression probe is introduced by prepared execution or capture. Two prepared bindings can coexist independently.

The prepared callable holds the plan's components and selected keys by reference. A replaced plan requires a new preparation. Changing shared-inventory membership leaves the existing binding's selected references intact. Key and table Tensor contents remain live under the prepared shape, dtype, placement and synchronization contract. A CUDA Graph retains its callable and materials. Close all Graphs using a prepared core before calling its `close()`; close releases its bindings and ends further use.

## Automatic encoded-diagonal storage

`compress_diagonals=True` is the default. Each diagonal follows the original full-ring encoding and multiplication-preparation sequence. A single adjacent-value comparison across the encoded batch and prime rows identifies change positions; their minimum lowbit determines the largest valid contiguous repeat block. One scalar result selects the compact extent. The compact allocation copies one value per block, so the cache retains no dense backing storage. This selects the minimum extent within the NTT contiguous layout and applies to either supported Galois generator. Existing cyclic and strided-sparse plaintext representations remain available through the ordinary value APIs.

`compress_diagonals=False` retains dense preparation for direct comparisons or custom execution requirements. Both choices preserve encoding order, random-state progression, actual plaintext scale and encoded integers. Automatic preparation runs outside CUDA capture. A cold diagonal preparation during capture reports that `prepare` is required. Direct and BSGS plans consume mixed dense/compact operands, including batch broadcasting and zero-padding entries. The existing Backend implementations determine execution cost; material reduction is measured separately from latency.

## Weighted-product implementation selection

`Engine.sum_plaintext_products` and `Engine.sum_plaintext_product_groups` accept `implementation` and `term_chunk_size`. Their default `None` retains native selection. `"auto"` chooses the registered CUDA Triton weighted-sum implementation for compressed operands and native execution otherwise. `Engine.plaintext_product_implementation(device, compressed=..., implementation=...)` exposes that resolution. A requested implementation must support the operand device.

The RNS weighted-sum operation carries a layout descriptor per plaintext and a trailing implicit Tensor per sparse plaintext, after its ciphertext and plaintext operands. The native grouped-dense path remains available. `TritonWeightedSumImplementation` binds runtime shapes and strides to `Expr`, `BoundGraph` and `PointwisePlan`; its cache contains generated execution plans rather than numerical inputs. Eager dispatch does not construct a Program or invoke a Compile pipeline.

Capture retains requested weighted-sum selection and term chunking in the operation. Without a requested selection or chunk, capture continues to expose the ordinary multiply/add composition to caller-selected passes. Manual IR may describe the same mixed layouts and bind ordinary parameter and implicit Tensors. Auto-selection during capture requires a known input device; unknown placement is not treated as CUDA.

`BsgsPlan.product_implementation` and `product_chunk_size` forward these controls in both traversals. Group-major execution fuses a row of products; baby-major execution uses one-term weighted sums to preserve its live-set schedule. Group traversal and the giant rescale/rotation schedule remain separate choices. `bootstrap.plan` defaults to `product_implementation="auto"` and `product_chunk_size=16`; callers can select native products or another chunk size. Sixteen terms is one configurable execution choice, not a weighted-sum arity limit.

Backend Python export preserves all weighted-sum parameter and implicit Tensor operands. Eager Python export supports CKKS-level operations and rejects the RNS weighted-sum operation; converting it into an Engine call would replace its independently bound numerical tables and implicit rows.


### Streaming rotation execution

`BsgsPlan` schedules and yields NTT inner products. Eager `Engine.sum_rotations` owns ciphertext state transitions and key placement, while Backend accumulates the numerical QP products and applies the shared quotient. `Engine.iter_rotations_with_keys` exposes the same shared decomposition used by the registered rotate-many implementation, with one yielded result at a time. `fhelium.algorithms` and `fhelium.bootstrap` use these value operations and does not access Engine dispatchers or Backend key tables.

Rounded trailing-basis division and its inverse tables are implemented under RNS. CKKS preparation combines those tables with key-switch operands. Whole-operation execution and Compile lowering use the same quotient semantics; the lowering exposes its RNS/NTT operations to the existing scheduling and fusion passes.
