# Terminology and mathematical model

This page defines FHElium terminology for CKKS parameters, value state, tensor
layouts, and execution. API signatures remain authoritative for concrete
identifiers and types.

## Ring, modulus, and scale notation

FHElium implements the Cheon–Kim–Kim–Song (CKKS) approximate homomorphic
encryption scheme over $R=\mathbb Z[X]/(X^N+1)$.

| Symbol or term | Meaning | FHElium representation |
| --- | --- | --- |
| $N=2^{\mathtt{logN}}$ | Polynomial-ring dimension | `config.N` |
| $S=N/2$ | Complex slot capacity | `engine.num_slots` |
| $G_d$ | Q group removed by transition $d\to d+1$ when $d<D$ | `config.q_depth_groups[d]` |
| $G_D$ | Terminal Q group retained at `max_depth` | `config.q_depth_groups[-1]` |
| $Q_d$ | Product of all Q groups active at depth $d$ | `depth` plus Q `prime_ids` |
| $p_j$ | One special prime | One prime in `config.p_moduli` |
| $P=\prod_jp_j$ | Special modulus used by hybrid key switching | P portion of a QP basis |
| $B_d$ | Current modulus product, $Q_d$ or $Q_dP$ | `modulus_basis`, `depth`, and `prime_ids` |
| $D$ | Greatest public depth | `config.max_depth` or `engine.max_depth` |
| $D-d$ | Public rescale transitions remaining | `engine.depth_remaining(value)` |
| $\Delta_0$ | Default creation and planning scale | `config.default_scale` |
| $\Delta(v)$ | Actual scale carried by value $v$ | `value.scale` |
| $s(X)$ | Secret-key polynomial | `SecretKey` |
| $c(X)=(c_0,\ldots,c_{k-1})$ | Ciphertext component polynomials | Leading component axis |
| $\sigma_g$ | Galois automorphism $X\mapsto X^g$ | `galois_element` |
| $\operatorname{Rot}_r$ | Signed slot rotation | `rotation_step` |
| $\operatorname{SRound}$ | Unbiased stochastic rounding | Encoding quantizer |

A **Q prime** is a prime factor of the ciphertext modulus. Q primes are ordered
inside depth groups. A **scaling Q group** is one public group $G_d$; a public
rescale removes the complete group. The final Q group is the terminal group
at `max_depth`; it remains available for ordinary arithmetic. A **special
prime** is a factor of P. `auxiliary basis` describes a temporary
basis-conversion role such as QP; it is not a prime kind.

A **slot** is one packed complex message position. Applications decide how
vectors, matrices, padding, and masks map to the $S$ slots.

### Construction and compatibility objects

| Term | Definition |
| --- | --- |
| `Preset` | A reviewed named recipe that resolves to one exact `CkksConfig`. Its name records slot capacity, default-scale target, maximum public depth, and the default Engine residue dtype selected by its primes. |
| `CkksConfig` | The immutable mathematical and security parameter set: ring, default scale, exact Q depth groups, exact P special primes, Galois generator, Gaussian error standard deviation, security category, and budget-enforcement policy. It contains no device or machine-word policy. |
| `RnsExecutionFormat` | The device-local residue dtype and Montgomery radix selected from the exact prime set, optionally constrained by an expert Engine override. |
| `fhelium.eager.Engine` | The process-local eager evaluator. It selects an RNS execution format, lazily creates device resources, owns installed keys, and invokes registered implementations. |

Runtime values and keys do not store a parameter-set identifier. The caller
retains provenance and supplies mathematically compatible configurations,
values, and keys.

See [Configuration and modulus chain](ckks/context-and-modulus-chain.md).

## CKKS value-state coordinates

A **CKKS value state** is the combination of concrete value type, tensor
topology, depth, actual scale, ordered `prime_ids`, plaintext representation,
polynomial domain, modulus basis, residue representation, and component or key
specialization. Device placement is a separate storage property.

| Coordinate | Values | Meaning |
| --- | --- | --- |
| Plaintext representation | `slots`, `integer_coefficients`, `approximate_coefficients`, `rns` | Meaning of the plaintext payload |
| Polynomial domain | `coefficient`, `ntt` | Polynomial coefficients or NTT evaluations |
| Residue representation | `standard`, `montgomery` | Ordinary or Montgomery residues |
| Modulus basis | `Q`, `QP` | Active $Q_d$ rows or active $Q_dP$ rows |
| `prime_ids` | Ordered parameter-row identifiers | Modulus represented by each physical limb row |
| Depth | Integer $d$ in $[0,D]$ | Consumed public rescale transitions |
| Depth remaining | $D-d$ | Public rescale transitions still available |
| Actual scale | Positive finite binary64 $\Delta(v)$ | Per-value encoding factor |
| Component count | Usually two or three for ciphertexts | Degree of the ciphertext phase in $s(X)$ |
| Placement | CPU or indexed CUDA device | Tensor storage location |

Do not infer one coordinate from another. QP is a modulus basis, not a depth;
`depth` does not replace `prime_ids`; coefficient domain is not the same as an
integer-coefficient plaintext; and placement does not change the represented
value.

### Plaintext representation

| Representation | Payload and use | Dense layout |
| --- | --- | --- |
| `slots` | Real or complex CKKS message before encoding | Scalar or `[*batch, slot]` |
| `integer_coefficients` | Signed `int64` coefficients after encoding and before modular reduction | `[*batch, coefficient]` |
| `approximate_coefficients` | Binary64 tail-Q reconstruction produced by decryption for decoding | `[*batch, coefficient]` |
| `rns` | Operation-ready residues with complete RNS state | `[*batch, limb, coefficient_or_ntt_index]` |

`integer_coefficients_to_rns` is the representation boundary that reduces each
signed integer coefficient modulo every active prime and materializes the
Engine's RNS dtype. An `approximate_coefficients` plaintext is not an exact
full-$Q_d$ CRT reconstruction and cannot be converted back to RNS.

### Polynomial domain, residue representation, and basis

A coefficient-domain polynomial is indexed by coefficients. An NTT-domain
polynomial is indexed by Number Theoretic Transform evaluations. Montgomery
representation stores residues in a form suitable for modular multiplication.
Public ciphertexts couple these axes into two supported arithmetic states:

```text
(coefficient, standard)
(ntt, montgomery)
```

RNS plaintexts additionally support `(coefficient, montgomery)`. A **Residue
Number System (RNS)** represents an integer polynomial by residues modulo
pairwise-coprime primes. One residue-polynomial row is a **limb**, and
`prime_ids[limb_index]` identifies its modulus.

At depth $d$, Q contains the suffix of configured Q groups beginning at $G_d$.
QP appends all P special-prime rows to that Q suffix. One public rescale advances
to $d+1$ and divides scale by $\prod_{q\in G_d}q$; modulus switching changes
depth while preserving scale. The default scale and the prime widths inside a
depth group are independent.

## Tensor and dimension terminology

A **homogeneous batch** is zero or more local message dimensions whose members
share one value's CKKS metadata. It is not the slot axis, limb distribution,
hybrid key digits, or a set of distributed processes. `*batch` means those zero
or more leading logical batch dimensions; broadcasting is valid only where an
operation explicitly documents it.

| Value | Dense tensor layout |
| --- | --- |
| `Ciphertext.data` | `[component, *batch, limb, coefficient_or_ntt_index]` |
| RNS `Plaintext.data` | `[*batch, limb, coefficient_or_ntt_index]` |
| Integer or approximate coefficient plaintext | `[*batch, coefficient]` |
| Slots plaintext | Scalar or `[*batch, slot]` |
| `SecretKey.data` | `[limb, coefficient_or_ntt_index]` |
| `PublicKey.data` | `[key_component=2, limb, coefficient_or_ntt_index]` |
| `KeySwitchKey.data` | `[key_digit, key_component=2, limb, coefficient_or_ntt_index]` |

A **ciphertext component** is one polynomial $c_j(X)$. Fresh ciphertexts have
two components; ciphertext-ciphertext multiplication produces three until
relinearization. In key switching, `key_digit_index` is the stable key-storage
digit identity, whereas a local `digit_index` is the position among digits
active at one depth.

Keep these dimensions and identifiers distinct:

| Term | Meaning |
| --- | --- |
| Process rank | Integer position of a process globally or within a process group |
| `LOCAL_RANK` | Node-local process index commonly used to choose a local device |
| Device index | CPU/CUDA storage location identifier; not a process rank |
| `tensor.ndim` | Number of tensor dimensions |
| Slot index | Semantic packed-message position |
| Limb index | Physical RNS-row position interpreted through `prime_ids` |

## CKKS encoding, encryption, and arithmetic laws

### Encoding and decoding

Let $m\in\mathbb{C}^S$, and let $\mathcal{E}^{-1}$ map canonical slot order to
a real coefficient polynomial. Encoding at actual scale $\Delta$ uses

$$
\widehat m_i=\operatorname{SRound}\!\left(
  \Delta\,\mathcal{E}^{-1}(m)_i
\right),
$$

where

$$
\operatorname{SRound}(x)\in\{\lfloor x\rfloor,\lceil x\rceil\},
\qquad
\mathbb{E}[\operatorname{SRound}(x)]=x.
$$

`encode` returns `integer_coefficients`.
`integer_coefficients_to_rns` performs modular reduction into
coefficient-domain standard RNS. `decode` consumes integer coefficients or
bounded `approximate_coefficients`; it does not consume unreconstructed RNS.

### Ciphertext phase and key relations

A $d$-component ciphertext has decrypted phase

$$
u(X)=\sum_{j=0}^{d-1}c_j(X)s(X)^j\pmod{B_\ell},
\qquad d\in\{2,3\}.
$$

For two components this is $c_0+c_1s$; after multiplication it may include
$c_2s^2$. Decryption produces bounded binary64 approximate coefficients for
decoding rather than an exact full-modulus CRT inverse.

An **external cryptographic relation** is an application-owned relation
not represented by a symbolic lineage field. A public key is related to one
destination secret polynomial through

$$
k_0(X)+k_1(X)s_{\mathrm{dst}}(X)=e(X)\pmod{B_0}.
$$

Here $e(X)$ is the sampled public-key error polynomial.

A generic key-switch key is directed from one source secret dependency to one
destination dependency:

$$
c_0+c_1s_{\mathrm{src}}
\longmapsto
c'_0+c'_1s_{\mathrm{dst}}.
$$

A **key switch** performs that transformation up to the configured key-switch
error. **Relinearization** is its specialized three-to-two-component form:

$$
w_0+w_1s+w_2s^2
\longmapsto
c'_0+c'_1s.
$$

The object stores no generic source/destination secret identifiers, so the
application preserves the direction. A `RotationKey` does store its normalized
signed `rotation_step` specialization. See [Key lifecycle](ckks/key-lifecycle.md).

| Key material | Cryptographic relation or use |
| --- | --- |
| `SecretKey` | Stores $s(X)$ for decryption and derivation of other keys. |
| `PublicKey` | Encrypts under its externally tracked destination secret relation. |
| `KeySwitchKey` | Maps $s_{\mathrm{src}}$ dependency to $s_{\mathrm{dst}}$ dependency. |
| `RelinearizationKey` | Maps the $s^2$ multiplication dependency back to $s$. |
| `RotationKey` | Maps $\sigma_g(s)$ back to $s$ for its stored signed slot step. |
| `ConjugationKey` | Maps $\sigma_{-1}(s)$ back to $s$ for complex conjugation. |

### Domain and residue transitions

For ciphertexts, forward and inverse transitions preserve the ring element
while applying the coupled domain/residue change:

$$
(\text{coefficient},\text{standard})
\xrightarrow{\mathrm{forward\ NTT}}
(\text{ntt},\text{montgomery}),
$$

$$
(\text{ntt},\text{montgomery})
\xrightarrow{\mathrm{inverse\ NTT}}
(\text{coefficient},\text{standard}).
$$

For RNS plaintexts, `standard_residues_to_montgomery_residues` and its inverse
change residue representation in coefficient domain, while the NTT methods
change polynomial domain on Montgomery residues. These transitions preserve
depth, actual scale, basis, and `prime_ids`. An inverse NTT remains RNS and is
not CRT reconstruction. Every primitive transition requires its named source
state; an input already in the target state is an error rather than an implicit
no-op.

### Addition, subtraction, and negation

Compatible ciphertexts add or subtract componentwise:

$$
c_{\mathrm{out},j}=c_{\mathrm{lhs},j}\mathbin{\pm}c_{\mathrm{rhs},j}
\pmod{B_\ell}.
$$

The operands must already match in tensor shape, depth, exact binary64 actual
scale, component count, polynomial domain, basis, residue
representation, and `prime_ids`. FHElium does not hide depth, scale, domain, or
component alignment inside addition. The caller is responsible for selecting
operands produced with compatible CKKS parameters. Negation maps each component to
$-c_j\bmod B_\ell$ and preserves state metadata.

### Ciphertext multiplication

For two two-component NTT/Montgomery ciphertexts, multiplication is component
convolution:

$$
w_0=a_0b_0,
\qquad
w_1=a_0b_1+a_1b_0,
\qquad
w_2=a_1b_1
\pmod{B_\ell}.
$$

It returns three NTT/Montgomery components at unchanged depth with
$\Delta(w)=\Delta(a)\Delta(b)$. The primitive does not implicitly
relinearize, change domain, or rescale. Public relinearization accepts a
three-component Q ciphertext and uses internal QP key material and scratch
space; after P ModDown it returns a two-component coefficient-domain standard
Q ciphertext.

### Plaintext arithmetic

An **operation-ready plaintext** is encoded at a depth and prepared in the
arithmetic state required by an evaluator operation; write its polynomial as
$a_{\rm pt}$. For addition it is
coefficient-domain Montgomery RNS and updates only the constant component:

$$
c'_0=c_0+a_{\rm pt},
\qquad
c'_j=c_j\quad(j>0),
\qquad
\Delta(c')=\Delta(c)=\Delta(a_{\rm pt}).
$$

For multiplication it is NTT-domain Montgomery RNS and multiplies every
component:

$$
c'_j=c_ja_{\rm pt},
\qquad
\Delta(c')=\Delta(c)\Delta(a_{\rm pt}).
$$

An unbatched operation-ready plaintext may broadcast over a homogeneous
ciphertext batch where the API documents that behavior. Neither arithmetic
primitive implicitly rescales.

### Rescale and modulus switch

At depth $\ell$, let $G_\ell$ be the leading active Q depth group and
$M_\ell=\prod_{q\in G_\ell}q$. **Rescale** computes a rounded quotient on
every surviving row:

$$
c'=\operatorname{Round}\!\left(\frac{c}{M_\ell}\right)
\pmod{B_{\ell+1}},
\qquad
\ell'=\ell+1,
\qquad
\Delta(c')=\frac{\Delta(c)}{M_\ell}.
$$

`rounding="nearest"` and `rounding="floor"` identify distinct quotient laws.
A QP rescale retains P rows, so its output basis is $Q_{\ell+1}P$. The
**rescale divisor** is available through
`engine.rescale_divisor(depth=...)` for a non-final public depth; it is the
product of every prime in the group removed by that transition.

A **modulus switch** restricts rows from $Q_\ell$ or $Q_\ell P$ to target depth
$t$'s active rows without dividing coefficients:

$$
c'=c\pmod{B_t},
\qquad
\Delta(c')=\Delta(c).
$$

Message preservation requires the centered represented value not to wrap under
the smaller target modulus.

### Rotation, automorphism, and hoisting

For a signed **rotation step** $r$,

$$
\operatorname{Rot}_r(m)_j=m_{(j-r)\bmod S},
$$

matching `torch.roll(m, shifts=r)`. The backend maps $r$ to a distinct odd
**Galois element** $g$, applies $\sigma_g$, and key-switches the transformed
secret dependency. `rotation_step` is a user-visible slot displacement;
`galois_element` is a polynomial automorphism exponent. Do not use the names
interchangeably.

**Rotation hoisting** reuses decomposition, ModUp, and NTT preparation shared
by several direct rotations of one component. Each output still performs its
step-specific automorphism, key products, ModDown, and output construction.
Conjugation applies $\sigma_{-1}$ and computes
$m'_j=\overline{m_j}$ with a matching key switch.

### Hybrid key switching and bootstrapping terms

| Term | Definition |
| --- | --- |
| ModUp | Basis extension of each active hybrid digit from its source-prime subset into active $Q_\ell P$, preserving the integer represented by that digit. |
| ModDown | Hybrid key-switch down-conversion from QP to Q, including division by $P$ under its documented rounding law. |
| ModRaise | Centered RNS basis extension used by bootstrapping. It reconstructs over a depleted source basis, chooses the centered representative, and reduces it into a larger Q basis. The current full-slot entry uses the single structural row `[q_b]`. |
| Bootstrapping | CKKS refresh that raises available modulus, maps coefficients to slots, applies periodic reduction, and maps slots back to coefficients. A bootstrap callable executes this composition; its output is a refreshed ciphertext. |
| CoeffsToSlots / SlotsToCoeffs | Linear transforms between coefficient coordinates and cyclotomic slot coordinates. Full-slot branch construction obtains the conjugate coordinate by conjugation. |
| BSGS | Baby-step/giant-step decomposition of a diagonal linear transform into caller-selected rotation groups and an execution schedule, trading key families and repeated work against intermediate computation. |
| Periodic reduction | Bootstrap approximation of the periodic map that removes the encoded large-modulus quotient. The composition field is named `modular_reduction`. |
| Structural base Q prime | The single Q prime left after the private bootstrap-entry transition drops every public scale prime. `[q_b]` remains a Q basis, not a third `modulus_basis` value. |

### Basis-conversion equations

The equations in this section state the mathematical maps independently of a
particular RNS kernel or table layout. For active hybrid-digit index $\kappa$,
let $D_\kappa\subseteq I_\ell$ be its source-prime index set and let

$$
M_\kappa=\prod_{i\in D_\kappa}q_i.
$$

For one coefficient, the source residues uniquely determine
$\chi_\kappa\in[0,M_\kappa)$; mixed-radix decomposition represents
$\chi_\kappa$ for basis extension. ModUp evaluates that same integer in every
destination row:

$$
\left(\operatorname{ModUp}_{D_\kappa\rightarrow Q_\ell P}
  (\chi_\kappa)\right)_{q_i}
=\chi_\kappa\bmod q_i
\quad(i\in I_\ell),
\qquad
\left(\operatorname{ModUp}_{D_\kappa\rightarrow Q_\ell P}
  (\chi_\kappa)\right)_{p_j}
=\chi_\kappa\bmod p_j
\quad(j\in J_P).
$$

ModUp changes the RNS basis of each hybrid digit; it does not divide the
represented integer.

ModDown is a basis reduction with division by $P$. Let $y$ be a QP residue
vector and let $\widetilde y\in\mathbb Z$ be the integer representative selected
for the operation. Let $\operatorname{rep}_P^{\tau}(\widetilde y)$ be the representative of
$\widetilde y\bmod P$ selected by rounding convention $\tau$. Two standard
choices are

$$
\operatorname{rep}_P^{\mathrm{floor}}(\widetilde y)\in[0,P),
\qquad
\operatorname{rep}_P^{\mathrm{nearest}}(\widetilde y)\in(-P/2,P/2].
$$

Then

$$
\operatorname{ModDown}^{\tau}_{QP\rightarrow Q_\ell}(y)
=\frac{\widetilde y-\operatorname{rep}_P^{\tau}(\widetilde y)}{P}
\pmod{Q_\ell}.
$$

Equivalently, for every active $q_i$ row,

$$
z_i=
\left(y_i-\left(\operatorname{rep}_P^{\tau}(\widetilde y)\bmod q_i\right)\right)
\left(P^{-1}\bmod q_i\right)
\bmod q_i.
$$

Here $y_i=\widetilde y\bmod q_i$, and $z_i$ is the resulting Q-basis residue.

An implementation may evaluate this quotient one P prime at a time, but its
tables and residue recurrences must realize the selected map above.

FHElium's current hybrid key-switch path selects $\tau=\mathrm{floor}$; the
nearest form above is the standard alternative rather than the current
evaluator law.

For centered ModRaise, let $I_{\rm src}$ identify a depleted source-prime set,
let

$$
Q_{\rm src}=\prod_{j\in I_{\rm src}}q_j,
$$

and let $a_{\rm src}\in[0,Q_{\rm src})$ be the canonical CRT reconstruction of
the source residues. It selects the centered lift

$$
\widetilde a_{\rm src}=
\begin{cases}
a_{\rm src}, & a_{\rm src}\le\lfloor Q_{\rm src}/2\rfloor,\\
a_{\rm src}-Q_{\rm src}, & a_{\rm src}>\lfloor Q_{\rm src}/2\rfloor,
\end{cases}
$$

and extends it to the selected target depth $\ell_{\rm raise}$:

$$
\left(\operatorname{ModRaise}_{Q_{\rm src}\rightarrow Q_{\ell_{\rm raise}}}
  (a_{\rm src})\right)_{q_i}
=\widetilde a_{\rm src}\bmod q_i,
\qquad i\in I_{\ell_{\rm raise}}.
$$

The application must establish that the intended integer coefficient lies in
$[-\lfloor Q_{\rm src}/2\rfloor,\lfloor Q_{\rm src}/2\rfloor]$, so the source
residues identify it without aliasing. FHElium's current full-slot bootstrap is
the special case $Q_{\rm src}=q_b$ before raising to its selected target depth.

### Bootstrap and linear-transform equations

Let $a$ be a coefficient-coordinate ring vector. Under a normalized
convention, let $\mathcal C(a)$ be its cyclotomic slot-coordinate
representation and let $\mathcal C^{-1}$ consume that representation as the
corresponding SlotsToCoeffs map. Full-slot algorithms may obtain paired real,
imaginary, or conjugate branches from those slot coordinates. Ideally,

$$
\mathcal C^{-1}(\mathcal C(a))=a.
$$

Concrete algorithms may distribute a known normalization factor between these
two linear maps. Let $\operatorname{MR}$ be centered modulus raising and let
$\operatorname{Red}$ act slotwise on $\mathcal C(\operatorname{MR}(a))$ to
remove the encoded modulus multiple. The ideal bootstrap composition is

$$
\operatorname{Bootstrap}(a)
\approx
\mathcal C^{-1}\!\left(
  \operatorname{Red}\!\left(\mathcal C(\operatorname{MR}(a))\right)
\right).
$$

For period $H>0$, define the translation-invariant nearest-integer map
$\operatorname{NInt}_{+}$ by
$\operatorname{NInt}_{+}(k+\tfrac12)=k+1$ for every integer $k$. The resulting
centered modular-reduction map is

$$
\mu_H(\theta)=
\theta-H\operatorname{NInt}_{+}\!\left(\frac{\theta}{H}\right),
\qquad
\mu_H(\theta)\in[-H/2,H/2).
$$

The opposite one-sided tie convention produces $(-H/2,H/2]$ instead.
Polynomial or trigonometric approximations to $\mu_H$ are evaluated
homomorphically on a bounded input interval. A common normalized local
approximation for period two is

$$
\rho(\theta)=\frac{\sin(\pi\theta)}{\pi},
\qquad
\theta=2n+\epsilon
\Longrightarrow
\rho(\theta)\approx\epsilon
$$

for integer $n$ and sufficiently small $\epsilon$. The selected approximation,
normalization, interval, and error analysis are separate bootstrap design
choices.

For a cyclic-diagonal linear map, let $\lambda_r$ be the diagonal vector associated
with rotation offset $r$, let $v$ be the input slot vector, and let $\odot$
denote slotwise multiplication:

$$
L(v)=\sum_r \lambda_r\mathbin{\odot}\operatorname{Rot}_r(v).
$$

BSGS writes the rotation offset as $r=h+\beta$, where $h$ is a giant-step
offset and $\beta$ is a baby-step offset, and uses the identity

$$
\operatorname{Rot}_h\!\left(
  \operatorname{Rot}_\beta(v)\mathbin{\odot}
  \operatorname{Rot}_{-h}(\lambda_{h+\beta})
\right)
=
\operatorname{Rot}_{h+\beta}(v)\mathbin{\odot}\lambda_{h+\beta}.
$$

Direct and BSGS schedules therefore implement the same linear map while using
different rotation grouping and intermediate accumulation orders. See
[Composable CKKS bootstrapping](ckks/composable-bootstrapping.md) for the full
normalization, depth, scale, key, and error model.

## Distributed and multiparty terminology

FHElium's distributed model is **Single Program, Multiple Data (SPMD)**: each
process runs the same worker function over process-local data and a local
device, with application-defined ownership and collective ordering. A
**process group** is an ordered set of participating processes; **world size**
is the number of processes in that group.

A **collective** is a communication operation that every participating process
enters in compatible order. It is distinct from a collective cryptographic key
or decryption protocol. **Descriptor/payload separation** exchanges typed value
metadata first so a receiver can validate and allocate before transferring
dense tensor payloads.

### Mathematical relationships among process-local values

| Relationship or operation | Definition |
| --- | --- |
| Data parallelism | Processes evaluate independent requests or samples. Outputs remain distinct. |
| Gather | Transport that preserves independent process-local objects in a list; it performs no CKKS arithmetic. |
| Additive partial | One process-local ciphertext representing a summand of a shared logical result. |
| Additive-term parallelism | Processes own disjoint summands and combine compatible ciphertext partials by modular addition. Rotation/diagonal offsets are a common case. |
| Typed ciphertext reduction | A collective that combines additive partials through modulus-aware engine addition. |
| Limb parallelism | Processes own disjoint RNS rows of one logical value. Row-local work is possible only where the operation permits partial layouts. |
| Reconstruction | Structural concatenation of disjoint RNS rows into the complete active-row layout. |

These three operations must remain distinct:

```text
independent objects  -> gather into a list
additive partials    -> reduce with CKKS modular addition
disjoint RNS rows    -> reconstruct by ordered concatenation
```

See [Communication semantics](distributed/communication-semantics.md) for the
collective selection model.

### Cryptographic parties and processes

A **cryptographic party** is an independent trust-domain participant. A
**process rank** is an execution identity. The application maps each party to
its processes, devices, and transport endpoints.

**Collective key generation (CKG)** constructs collective public-key material.
**Relinearization-key generation (RKG)** constructs the evaluation key for
three-to-two-component relinearization. Applications supply coordination,
authenticated membership, and transport around these arithmetic phases.

Collective-decryption shares fuse into an output under the experimental
module's documented unsafe arithmetic scope. Public-key-switch shares fuse
into a ciphertext under a destination public key. These output operations are
limited to synthetic-data arithmetic studies; privacy and production security
require a reviewed protocol and parameterization.

## Execution, storage, and lifecycle terms

| Term | Definition |
| --- | --- |
| `TensorResident` | Protocol for a FHElium value whose declared Tensor fields move together, with device/byte inspection and functional `.to(...)`. |
| Value signature | Device-independent description of a nested Tensor/value structure and stored state used to validate reusable copies and graph inputs. External key relations remain properties of the concrete values. |
| `ReusableValueBuffer` | Fixed-address storage for one value-tree signature on one target device, used for eager streaming or CUDA Graph input staging. |
| `CopyHandle` | Handle for one reusable-buffer copy. It retains source storage and exposes event-based completion and stream-wait operations. |
| Eager execution | Immediate operation dispatch using the supplied values and execution resources. |
| CUDA Graph | Captured fixed GPU schedule replayed with stable addresses. `CudaGraphProgram` applies it to a deterministic process-local callable with fixed value signatures. |
| Borrowed output | Output backed by storage retained and reused by another owner, such as a CUDA Graph program. A later replay may overwrite it; an owned copy is required for retention. |

**Residency** is process-local management of logical FHElium values and their
concrete storage locations. Reusable buffers and CUDA Graph programs retain
ownership of their fixed storage.

| Residency term | Definition |
| --- | --- |
| Materialization | A ready managed value at one local location, such as pageable CPU, pinned CPU, or an indexed CUDA device. |
| Residency budget | Optional caller-supplied strict admission limit for managed charges and reservations at one location. Unbudgeted locations still retain current and peak accounting. |
| Residency request | Declarative handle/location postconditions and headroom requirements: what must be true. |
| Residency policy | Pure deterministic ordering and configured fallback choices used to evaluate a request. |
| Residency decision | Manager-issued, state-versioned plan plus policy evidence: why the selected actions are valid for the observed state. |
| Residency plan | Immutable ordered low-level intermediate representation (IR) of explicit `ensure`, `move`, `drop`, and `discard` actions plus scoped memory reservations: how placement changes. |
| Lease | Short-lived protection for active evaluator reads. It remains effective until all consumers, including asynchronous CUDA readers, complete. |
| Hold | Longer-lived retention protection for an idle materialization; active evaluator access uses a lease. |

`ResidencyManager` owns live values, accounting, transitions, and lifetimes.
`ResidencyController` derives inspectable decisions against manager state. See
[Residency lifetimes](execution/residency-lifetimes.md).

## Compiler, implementation, and build terms

| Term | Definition |
| --- | --- |
| `Program` | Source-independent `fhelium.ir.Program` containing one structurally valid mixed-dialect xDSL module. Consumers separately check semantic completeness, CKKS consistency, Backend coverage, and executable readiness. |
| Compile | Source-oriented capture and transformation that produces or updates a neutral Program while retaining compile-time evidence separately. |
| Experimental JIT | Runtime-oriented transformation and specialization using one execution device and separately supplied live bindings. |
| Pass / pipeline | A pass analyzes or transforms recognized Program patterns. A pipeline is a caller-selected ordered pass sequence; unmatched or extension operations may remain. |
| Backend | A consumer that reports coverage and builds an executable for a supported Program subset. A target such as CPU or CUDA is distinct from a region-provider name such as `native_cpu`, `native_cuda`, or `triton`. |
| Native operator stack | PyTorch dispatcher schemas, C++ registrations/bindings, CPU/CUDA implementations, and generated Python interfaces loaded by FHElium. |
| Dispatcher schema | Backend-neutral `torch.ops` contract for arguments, returns, mutation, and aliasing. CPU and CUDA registrations implement the same schema where supported. |
| Generated wrapper | Mechanical typed Python interface generated from registered native schemas; handwritten API/orchestration documentation defines the mathematical semantics. |
| ABI | Application binary interface among CPython, the exact PyTorch build and CUDA variant, the local CUDA toolkit, the C++ ABI, target GPU architectures, and FHElium native binaries. FHElium records the additional compatibility identity absent from ordinary wheel tags in its native manifest. |

The word **backend** needs a qualifier when ambiguity is possible: NTT backend
policy selects an NTT implementation; a process-group communication backend
such as NCCL or Gloo transports distributed data; a Python build backend drives
package construction; and a compiler backend builds an executable for a
covered Program. These are independent choices.

See [Open compiler stack](open-compiler-stack.md),
[Neutral IR programs](neutral-ir-programs.md), and
[Eager, Compile, and native execution](../developer/engine-native-stack.md).

## Distinctions to retain

Use the following pairs and groups as different terms, even when one maps to
another internally:

- default scale $\Delta_0$ and per-value actual scale $\Delta(v)$;
- depth and ordered `prime_ids`;
- Q/QP modulus basis and depth;
- semantic `modulus_basis` and the internal `include_p` implementation selector;
- plaintext representation, polynomial domain, and residue representation;
- inverse NTT and CRT reconstruction;
- `rotation_step` and `galois_element`;
- local `digit_index` and stable `key_digit_index`;
- process rank, device index, tensor `ndim`, slot index, and limb index;
- gather, additive ciphertext reduction, and limb reconstruction;
- cryptographic party and process rank;
- hold and lease;
- target, provider, NTT backend, communication backend, compiler backend, and
  Python build backend.
