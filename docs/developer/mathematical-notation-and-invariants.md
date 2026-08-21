# FHElium mathematical notation and cross-layer invariants

This document defines the terminology, mathematical notation, tensor vocabulary, and documentation style used from the public Python API through Python orchestration and dispatcher schemas to C++ CPU and CUDA implementations.

Its purpose is to prevent a correct mathematical operation from acquiring different names or symbols at different layers, and to make each tensor operation understandable without reading every implementation layer first.

## Scope

These rules apply to:

- public value types and `CkksEngine` operations;
- Python codec, RNS, NTT, key-switch, and bootstrap orchestration;
- generated dispatcher wrappers and FakeTensor shape rules;
- C++ operator schemas and CPU/CUDA implementation comments;
- developer documentation that describes those paths.

Generated wrappers remain mechanical typed interfaces. The semantic owner is the handwritten Python orchestration or public API docstring, with lower-level tensor details repeated only where they are required to use or maintain a native operator safely.

## Documentation style

### Mathematical notation

1. Write mathematics with Markdown/LaTeX delimiters: `$...$` inline and `$$...$$` for display equations.
2. Do not insert raw UTF-8 mathematical glyphs such as Greek letters, set symbols, inequalities, or arrows into docstrings or implementation comments.
3. Use ASCII identifiers in code spans, for example `default_scale`, `prime_ids`, and `polynomial_domain`.
4. Use one symbol for one mathematical object throughout the call chain. Do not reuse a symbol for both a CKKS scale and a secret key.
5. In Python docstrings containing LaTeX backslashes, use raw docstrings when needed to avoid invalid escape sequences.

### Required content by operation type

An arithmetic-operation docstring must state:

- the mathematical operator;
- the input preconditions;
- the output level, actual scale, component count, polynomial domain, modulus basis, and residue representation;
- whether the operation is functional or mutating;
- any approximation, rounding, range, or no-wrap condition visible to the caller.

A tensor/native-operation docstring or source comment must additionally state:

- the semantic meaning and order of every tensor axis;
- accepted shape, dtype, and device placement;
- the polynomial domain and residue representation;
- the modulus rows represented by the limb axis and how `prime_ids` maps them;
- output shape and state;
- mutation, storage aliasing, and allowed lazy residue range;
- the mathematical operation implemented by the kernel.

A docstring must not merely say "transform", "prepare", "normalize", or "convert" when the operation actually performs NTT, Montgomery conversion, RNS restriction, basis extension, quotient rounding, or key switching.

## Canonical mathematical notation

| Symbol | Meaning | Code-level term |
|---|---|---|
| $N=2^{\mathtt{logN}}$ | cyclotomic ring dimension | `config.N` |
| $S=N/2$ | complex CKKS slot count | `engine.num_slots` |
| $R=\mathbb{Z}[X]/(X^N+1)$ | integer polynomial ring | coefficient-domain polynomial |
| $q_i$ | one ordinary Q-chain prime | one Q `prime_id` |
| $Q_\ell=\prod_{i\in I_\ell}q_i$ | product of active Q primes at public level $\ell$ | active Q basis |
| $p_j$ | one special key-switch prime | one P `prime_id` |
| $P=\prod_j p_j$ | product of special key-switch primes | P basis |
| $B_\ell$ | active modulus basis, either $Q_\ell$ or $Q_\ell P$ | `modulus_basis` |
| $\Delta(v)$ | actual scale carried by value $v$ | `value.scale` |
| $\Delta_0$ | default encoding/planning scale | `config.default_scale` |
| $s(X)$ | secret-key polynomial | `SecretKey` |
| $c(X)=(c_0,\ldots,c_{d-1})$ | ciphertext components | component axis |
| $u(X)=\sum_{j=0}^{d-1}c_j(X)s(X)^j$ | decrypted ciphertext phase | decrypt phase |
| $\sigma_g$ | Galois automorphism $X\mapsto X^g$ | `galois_element` |
| $\operatorname{Rot}_r$ | user-visible signed slot rotation | `rotation_step` |
| $\operatorname{SRound}$ | unbiased stochastic rounding | encoding quantizer |

Here $I_\ell$ is the ordered set of active Q-prime identifiers at public level $\ell$. A lower-level tensor comment should name `prime_ids` rather than assume that the active rows are always a contiguous numerical interval.

## Canonical value and state vocabulary

### Plaintext representation

`PlaintextRepresentation` describes what the payload means before the independent RNS state axes are considered.

| Value | Meaning | Canonical payload axes |
|---|---|---|
| `slots` | semantic complex CKKS message | `[*batch, slot]`, final extent $S$ |
| `integer_coefficients` | signed integer polynomial coefficients | `[*batch, coefficient]`, final extent $N$ |
| `approximate_coefficients` | bounded binary64 decrypt reconstruction, valid for decoding only | `[*batch, coefficient]`, final extent $N$ |
| `rns` | operation-ready residues | `[*batch, limb, coefficient_or_ntt_index]` |

Do not use the word "coefficient" alone to distinguish CRT reconstruction from polynomial domain. The following statements are different:

- `integer_coefficients` and `approximate_coefficients` describe plaintext representation;
- `coefficient` and `ntt` describe polynomial domain;
- `standard` and `montgomery` describe residue representation;
- `Q` and `QP` describe modulus basis.

### Independent RNS state axes

| Axis | Values | Meaning |
|---|---|---|
| `polynomial_domain` | `coefficient`, `ntt` | coefficient indexing versus NTT evaluation indexing |
| `residue_representation` | `standard`, `montgomery` | standard residues versus Montgomery residues |
| `modulus_basis` | `Q`, `QP` | active $Q_\ell$ rows versus active $Q_\ell P$ rows |
| `prime_ids` | tuple of parameter-row identifiers | ordered modulus represented by each limb row |
| `level` | public Q-chain level | determines active $Q_\ell$ but does not replace `prime_ids` |
| `scale` | positive finite binary64 | actual per-value scale $\Delta(v)$ |

Canonical operation-ready states are:

- plaintext addition operand: `coefficient` + `montgomery`;
- plaintext or ciphertext multiplication operand: `ntt` + `montgomery`;
- coefficient-domain ciphertext: `coefficient` + `standard`.

Plaintext representation axes are independently composable. A plaintext may
also occupy the intermediate `coefficient` + `standard` RNS state returned by
`integer_coefficients_to_rns`. Ciphertext states are deliberately coupled: public ciphertexts use
either `coefficient` + `standard` or `ntt` + `montgomery`.

No public method named `ntt_domain_to_coefficient_domain` performs CRT reconstruction. It remains an RNS value.

## Canonical tensor layouts

| Value or operator family | Tensor layout | Notes |
|---|---|---|
| `Ciphertext.data` | `[component, *batch, limb, coefficient_or_ntt_index]` | component count is two or three |
| RNS `Plaintext.data` | `[*batch, limb, coefficient_or_ntt_index]` | limb order equals `prime_ids` |
| integer/approximate coefficient plaintext | `[*batch, coefficient]` | final extent is $N$ |
| slots plaintext | `[*batch, slot]` | final extent is $S$ |
| `SecretKey.data` | `[limb, coefficient_or_ntt_index]` | basis/domain must be stated |
| `PublicKey.data` | `[key_component, limb, coefficient_or_ntt_index]` | key-component extent is two |
| `KeySwitchKey.data` | `[digit, key_component, limb, coefficient_or_ntt_index]` | distinguish local `digit_index` from stable `key_digit_index` |

For every native schema, `*batch` means zero or more homogeneous leading batch axes. Broadcasting must be stated; it must not be inferred from a suggestive shape.

## Canonical operation requirements

### CKKS encoding and decoding

Let $m\in\mathbb{C}^S$ and let $\mathcal{E}^{-1}$ map canonical slot order to a real coefficient polynomial. Encoding at actual scale $\Delta$ uses

$$
p_i=\operatorname{SRound}\!\left(\Delta\,\mathcal{E}^{-1}(m)_i\right),
$$

where

$$
\operatorname{SRound}(x)\in\{\lfloor x\rfloor,\lceil x\rceil\},\qquad
\mathbb{E}[\operatorname{SRound}(x)]=x.
$$

`encode` returns `integer_coefficients`. `integer_coefficients_to_rns` performs modular reduction into coefficient-domain standard RNS. Decoding consumes integer coefficients or bounded `approximate_coefficients`; it does not accept an unreconstructed RNS plaintext.

### Encryption and decryption

For a two-component ciphertext, decryption forms

$$
u(X)=c_0(X)+c_1(X)s(X)\pmod{Q_\ell}.
$$

For a three-component ciphertext, it forms

$$
u(X)=c_0(X)+c_1(X)s(X)+c_2(X)s(X)^2\pmod{Q_\ell}.
$$

The current decrypt path reconstructs bounded binary64 `approximate_coefficients` for decoding. It is not an exact full-$Q_\ell$ CRT inverse and must not be documented as one. Direct decrypt-to-encrypt is therefore not a bit-preserving representation round trip; the semantic path is decrypt, decode, encode, encrypt.

### Domain and residue transitions

For a ciphertext, `coefficient_domain_to_ntt_domain` preserves the ring
element while applying the forward NTT and standard-to-Montgomery conversion:

$$
(\text{coefficient},\text{standard})
\longrightarrow
(\text{ntt},\text{montgomery}).
$$

For a ciphertext, `ntt_domain_to_coefficient_domain` applies the inverse NTT and
Montgomery-to-standard reduction:

$$
(\text{ntt},\text{montgomery})
\longrightarrow
(\text{coefficient},\text{standard}).
$$

For an RNS plaintext, residue representation is an independent axis:
`standard_residues_to_montgomery_residues` and
`montgomery_residues_to_standard_residues` convert residues in coefficient
domain, while `coefficient_domain_to_ntt_domain` and
`ntt_domain_to_coefficient_domain` apply only the forward or inverse NTT to
Montgomery residues. These composable transitions preserve the ring element,
representation, `level`, actual `scale`, modulus basis, component count where
present, and `prime_ids`.

Every primitive transition requires its named source state. Supplying a value
already in the target state is an error rather than an implicit no-op.

### Ciphertext addition and subtraction

For compatible ciphertexts,

$$
c_{\mathrm{out},j}=c_{\mathrm{lhs},j}\mathbin{\pm}c_{\mathrm{rhs},j}
\pmod{B_\ell}.
$$

The operands must already have the same level, scale, component count, domain,
basis, residue representation, context, and `prime_ids`. FHElium does not hide
level or scale alignment in addition or subtraction.

### Ciphertext multiplication

For component polynomials $a_i$ and $b_j$, unrelinearized multiplication is component convolution:

$$
d_k=\sum_{i+j=k}a_i b_j\pmod{B_\ell}.
$$

A two-component times two-component multiplication returns three components. Its actual scale is

$$
\Delta(d)=\Delta(a)\Delta(b).
$$

The public primitive does not implicitly relinearize or rescale.

### Relinearization and key switching

Relinearization transforms the phase relation

$$
d_0+d_1s+d_2s^2
$$

into a two-component ciphertext $c'_0+c'_1s$ under the destination secret key, up to the configured key-switch error. It preserves the represented message, level, and actual scale. A general key switch maps a source-key phase to an equivalent destination-key phase; its docstring must identify source and destination key relations rather than merely say "switch".

Public relinearization and key-switch operations consume and return the active
Q basis. QP is internal key material and scratch space for hybrid key switching;
the final P ModDown produces Q-only corrections. A public QP ciphertext must be
rejected before entering that path rather than being described as
basis-preserving.

### Rescale

At public level $\ell$, let $q_{\mathrm{drop}}$ be the leading active Q prime. Rescale computes a rounded quotient for every remaining row:

$$
c'=\operatorname{Round}\!\left(\frac{c}{q_{\mathrm{drop}}}\right)
\pmod{B_{\ell+1}},
$$

and updates metadata to

$$
\ell'=\ell+1,\qquad
\Delta(c')=\frac{\Delta(c)}{q_{\mathrm{drop}}}.
$$

`rounding="nearest"` and `rounding="floor"` name quotient laws; neither should be described by the ambiguous term "exact rounding". QP rescale retains the P rows and must state that its output basis is $Q_{\ell+1}P$.

### Modulus switch

`mod_switch_to_level` restricts the RNS row set from $Q_\ell$ or $Q_\ell P$ to the target active-row set. It does not divide coefficients and does not change the actual scale. Message preservation requires that the represented centered value does not wrap under the smaller target modulus.

### Plaintext arithmetic

For addition, the prepared plaintext is in coefficient-domain Montgomery RNS and updates only the constant ciphertext component:

$$
c'_0=c_0+p,\qquad c'_j=c_j\quad(j>0).
$$

For multiplication, the prepared plaintext is in NTT-domain Montgomery RNS and

$$
c'_j=c_jp,
$$

with

$$
\Delta(c')=\Delta(c)\Delta(p).
$$

Neither primitive implicitly rescales.

### Rotation and conjugation

For signed rotation step $r$ and slot vector $m$,

$$
\operatorname{Rot}_r(m)_j=m_{(j-r)\bmod S}.
$$

This matches `torch.roll(m, shifts=r)`. The backend implements the corresponding polynomial automorphism $\sigma_g$ and key switch. Public rotation consumes and returns the active Q basis. `rotation_step` and `galois_element` are related but are not interchangeable names.

Conjugation computes

$$
m'_j=\overline{m_j}
$$

through the conjugation automorphism and a matching key switch. Public
conjugation likewise consumes and returns the active Q basis.

## Frontend-to-backend naming policy

### Names that should remain distinct

| Terms | Reason |
|---|---|
| `rotation_step` and `galois_element` | user-visible slot displacement versus polynomial automorphism exponent |
| `digit_index` and `key_digit_index` | active local digit position versus stable key-storage digit identity |
| `modulus_basis` and internal `include_p` | semantic value state versus a local implementation selector |
| `ntt_domain_to_coefficient_domain` and CRT reconstruction | inverse NTT remains RNS; CRT reconstruction changes representation |
| `default_scale` and `scale` | planning/encoding default versus actual per-value scale |

## Native operation documentation template

A native schema and its handwritten semantic owner should be documentable in the following form:

```text
Operation: mixed_radix_basis_extend_to_montgomery

Math:
    Convert mixed-radix digits for one integer polynomial into residues
    modulo every destination prime, preserving the polynomial element.

Input tensor:
    mixed_radix_components:
        shape [*batch, digit, coefficient]
        signed integral dtype on one execution device
        coefficient-domain mixed-radix digits

Tables:
    basis_extension_coefficients:
        shape [digit - 1, destination_limb]
        rows correspond to mixed-radix digits r >= 1
        columns follow destination prime_ids
        digit zero uses rns_params[R2] rather than a table row

Output tensor:
    shape [*batch, destination_limb, coefficient]
    coefficient domain, Montgomery residues

Mutation and aliasing:
    functional; output does not alias an input
```

The documented shape and table orientation must match the actual schema. A generated wrapper may repeat shapes and mutation annotations, but it should not become an independent source of mathematical truth.

## Cross-layer review checklist

For each public mathematical operation:

1. Trace public Python API to orchestration owner.
2. State the public operation's semantic equation.
3. Record all value-state transitions.
4. Trace each native tensor and map every axis to the equation.
5. Verify that frontend and backend names refer to the same object.
6. Rename misleading terms or document an intentional abstraction mapping.
7. Confirm residue ranges and rounding laws against implementation and tests.
8. Check functional versus mutating behavior and storage aliasing.
9. Use the same notation in docstrings, developer docs, tests, and CUDA comments.
10. Validate generated API documentation and run the smallest numerical regression that exercises the operation requirements.

## Non-goals

- Do not create a brittle inventory test that freezes every docstring or exported name.
- Do not duplicate full derivations in public docstrings, generated wrappers, and CUDA comments.
- Do not describe tail-Q binary64 decrypt reconstruction as exact CRT.
- Do not change numerical tolerances as part of documentation or naming work.
