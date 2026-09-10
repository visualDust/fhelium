# Scale, depth, and RNS execution

A CKKS parameter set determines how encoded magnitude is tracked, how many
rescale transitions a circuit can consume, and which prime basis represents a
value. The Engine derives its native residue format from those exact primes.
This page traces that relationship from a value's scale to the kernel selected
at execution.

## Scale tracks encoded magnitude

A plaintext or ciphertext with actual scale $\Delta$ represents an approximate
message by treating its encoded polynomial as having been multiplied by
$\Delta$. `value.scale` stores this positive binary64 number.

**Scale bits** means

$$
b_\Delta=\log_2\Delta.
$$

It is a reporting and planning quantity. For example, “scale 50” means
$\Delta\approx2^{50}$. `config.default_scale` supplies the initial value when
encoding or encryption does not receive a caller-selected scale.

Multiplication and rescaling update actual scale as follows:

$$
\Delta(ab)=\Delta(a)\Delta(b),
\qquad
\Delta(\operatorname{Rescale}_d(a))
=\frac{\Delta(a)}{M_d}.
$$

$M_d$ is the modulus removed by the rescale at depth $d$.

## Depth selects the active Q basis

`CkksConfig` partitions the ciphertext modulus Q into ordered depth groups:

$$
(G_0,G_1,\ldots,G_D).
$$

The public depth interval is $0\le d\le D$, where
`config.max_depth == D`. At depth $d$, the active Q basis contains $G_d$
through the terminal group $G_D$. For $d<D$, one `rescale_to_next_depth` call
removes the complete group $G_d$, so

$$
M_d=\prod_{q\in G_d}q,
\qquad d'=d+1.
$$

The number of public transitions still available is

$$
\mathtt{depth\_remaining}=D-d.
$$

This is a count of available transitions, not an operation history. The caller
chooses when to rescale; multiple multiplications can precede one rescale.
Actual scale and the active modulus determine whether that schedule preserves
the required numerical range and precision.

FHElium exposes these quantities directly:

```python
config.max_depth
config.q_depth_groups
config.rescale_divisor(depth)
engine.depth_remaining(ciphertext)
```

## A depth group determines both scale reduction and RNS work

A parameter planner commonly chooses $M_d$ near the scale introduced by one
multiplication. A scale near $2^{50}$ can therefore use either of these group
shapes:

```text
G_d = [one prime near 2^50]
G_d = [two primes near 2^25 whose product is near 2^50]
```

Both groups consume one CKKS depth and divide scale by approximately $2^{50}$.
Their execution costs differ because the first shape stores one RNS row at that
depth and the second stores two. The exact primes also determine which residue
formats satisfy the native kernels' lazy-reduction bounds.

This is why `q_depth_groups` stores nested prime groups rather than a flat
prime count. Semantic depth follows the outer group sequence; NTT, key-switch,
and RNS work follow the flattened prime rows.

## Encoding and RNS materialization

Encoding and residue execution use two consecutive representations:

```text
slots
  -> signed int64 integer coefficients
  -> residues modulo every active Q prime
  -> Engine-selected integral RNS Tensor
```

The encoder uses `int64` to hold signed integer coefficients before modular
reduction. `integer_coefficients_to_rns` reduces each coefficient modulo the
active primes and writes the result in `engine.dtype`. Ciphertexts and live
keys retain that RNS dtype.

## Execution-format selection and kernel dispatch

When an Engine is constructed, `RnsExecutionFormat.select` examines every exact
Q and P modulus in the configuration and chooses the narrowest supported
residue Tensor and Montgomery-radix pairing. `engine.dtype` reports the result.
An expert `rns_dtype=` argument requests a supported Tensor dtype; construction
checks that every configured prime fits the corresponding lazy-reduction
range. The Engine uses one format for its lifetime. Device-local `RnsContext`
resources then
materialize Montgomery constants, NTT tables, basis-conversion tables, and key
layouts in that format.

The registered CKKS and RNS operations have shared schemas. Native dtype
dispatch selects the scalar specialization of the same implementation. The
operation graph, Ciphertext type, depth transition, and Backend registration
remain unchanged.

## Worked transition

Consider a configuration with

```text
default_scale = 2^50
max_depth = 26
G_0 = [q_0a, q_0b]
q_0a, q_0b < 2^30
q_0a * q_0b approximately 2^50
all remaining Q and P primes below 2^30
```

A fresh ciphertext begins with

```text
depth = 0
depth_remaining = 26
scale = 2^50
```

Multiplying two such ciphertexts produces scale $2^{100}$ at depth zero.
Rescaling removes both primes in $G_0$, advances to depth one, and produces

$$
\Delta'=\frac{2^{100}}{q_{0a}q_{0b}}\approx2^{50}.
$$

The result has `depth_remaining == 25`. The rescale implementation executes two
prime-row division steps internally, but they implement one CKKS depth
transition. Its RNS kernels use the format selected from the complete prime set
when the Engine was built.
