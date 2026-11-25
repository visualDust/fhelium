# Scale, depth, and RNS execution

A CKKS parameter set determines how encoded magnitude is tracked, how many rescale transitions a circuit can consume, and which prime basis represents a value. The Engine derives its native residue format from those prime values. This page traces that relationship from a value's scale to the kernel selected at execution.

## Scale tracks encoded magnitude

A plaintext or ciphertext with actual scale $\Delta$ represents an approximate message by treating its encoded polynomial as having been multiplied by $\Delta$. `value.scale` stores this positive binary64 number.

**Scale bits** means

$$
b_\Delta=\log_2\Delta.
$$

It is a reporting and planning quantity. For example, “scale 40” means $\Delta\approx2^{40}$. `config.scale_at(depth)` supplies the initial value at that depth when encoding or encryption does not receive a caller-selected scale.

Multiplication and rescaling update actual scale as follows:

$$
\Delta(ab)=\Delta(a)\Delta(b),
\qquad
\Delta(\operatorname{Rescale}_d(a))
=\frac{\Delta(a)}{\rho_d}.
$$

$\rho_d$ is the rescale ratio of the transition from depth $d$ to $d+1$, defined below.

## Depth selects the active Q basis

`CkksConfig` stores the Q primes as one ordered list and assigns each depth $d$ a contiguous interval $B_d$ of that list. The public depth interval is $0\le d\le D$, where `config.max_depth == D`, and $B_0$ contains every Q prime. For $d<D$, one `rescale_to_next_depth` call replaces $B_d$ by $B_{d+1}$, dropping $D_d=B_d\setminus B_{d+1}$ and introducing $R_d=B_{d+1}\setminus B_d$. It divides every coefficient and the actual scale by

$$
\rho_d=\frac{\prod_{q\in D_d}q}{\prod_{q\in R_d}q},
\qquad d'=d+1.
$$

The number of public transitions still available is

$$
\mathtt{depth\_remaining}=D-d.
$$

This is a count of available transitions, not an operation history. The caller chooses when to rescale; multiple multiplications can precede one rescale. Actual scale and the active modulus determine whether that schedule preserves the required numerical range and precision.

FHElium exposes these quantities directly:

```python
config.max_depth
config.depth_bases           # (start, stop) of every B_d in Q order
config.q_prime_ids(depth)    # Q prime IDs of B_d
config.scale_at(depth)       # default scale of depth d
config.transition_ratio(depth, depth + 1)  # rho_d as a Fraction
engine.depth_remaining(ciphertext)
```

## A basis transition determines both scale reduction and RNS work

A parameter planner chooses $\rho_d$ near the scale introduced by one multiplication. The built-in scale-40 presets combine primes near $2^{30}$ and $2^{25}$, for example

```text
D_d = [three primes near 2^30], R_d = [two primes near 2^25]    rho_d near 2^(90-50)
D_d = [four primes near 2^25],  R_d = [two primes near 2^30]    rho_d near 2^(100-60)
```

Each such transition consumes one CKKS depth and reduces the active modulus by approximately 40 bits. The execution cost of a depth follows the number of rows $|B_d|$, which these presets keep close to the minimum needed for the remaining modulus. The prime values also determine which residue formats satisfy the native kernels' lazy-reduction bounds.

Semantic depth follows the sequence of bases; NTT, key-switch, and RNS work follow the rows of each basis. A suffix chain, in which every $R_d$ is empty, is the special case where each depth drops a fixed group of primes.

## Residue words and the represented integer

For active prime product $Q=\prod_i q_i$, one polynomial coefficient is represented by its residues $(a\bmod q_i)_i$. The Chinese remainder theorem makes this vector equivalent to one integer class modulo $Q$. Each stored word needs to hold an individual residue and the arithmetic implementation's intermediate range; the represented class can span the bit width of the entire prime product.

RNS dtype therefore controls the word size and modular arithmetic for each row. The prime set controls the total represented modulus, while the actual scale controls the interpretation of the encoded magnitude. These quantities are independent coordinates. [Integers, RNS, and CRT](context-and-modulus-chain.md#integers-rns-and-the-chinese-remainder-theorem) defines forward reduction, reconstruction, and centered representatives.

## Encoding and RNS materialization

Encoding and residue execution use two consecutive representations:

```text
slots
  -> signed int64 integer coefficients
  -> residues modulo every active Q prime
  -> Engine-selected integral RNS Tensor
```

The encoder uses `int64` to hold signed integer coefficients before modular reduction. `integer_coefficients_to_rns` reduces each coefficient modulo the active primes and writes the result in `engine.dtype`. Ciphertexts and live keys retain that RNS dtype.

## Execution-format selection and kernel dispatch

When an Engine is constructed, `RnsExecutionFormat.select` examines every configured Q and P modulus in the configuration and chooses the narrowest supported residue Tensor and Montgomery-radix pairing: `torch.int32` when every prime is below $2^{30}$, otherwise `torch.int64`. Built-in presets use primes below $2^{30}$. `engine.dtype` reports the result. An expert `rns_dtype=` argument requests a supported Tensor dtype; construction checks that every configured prime fits the corresponding lazy-reduction range. The Engine uses one format for its lifetime. Device-local `RnsContext` resources own the dense RNS parameters and residue-representation arithmetic in that format. `NttContext` composes that RNS context and owns the selected NTT policy, transform tables, and transforms; CKKS preparation supplies the basis-conversion and key-switch materials needed by the chosen operations.

The registered CKKS and RNS operations have shared schemas. Native dtype dispatch selects the scalar specialization of the same implementation. The operation graph, Ciphertext type, depth transition, and Backend registration remain unchanged.

## Worked transition

Consider a configuration with

```text
preset = slots16384-scale40-depth16, max_depth = 16
q_0 ... q_3 near 2^25, q_4 ... q_23 near 2^30, every P prime below 2^30
B_1 = q_2 ... q_23,   B_2 = q_0 ... q_20     (Q-order intervals)
D_1 = [q_21, q_22, q_23],   R_1 = [q_0, q_1]
scales[1], scales[2] approximately 2^40
```

A ciphertext at depth one with scale $\Delta_1$ has `depth_remaining == 15`. Multiplying two such ciphertexts produces scale $\Delta_1^2$ at depth one. Rescaling computes the rounded quotient by $q_{21}q_{22}q_{23}$ after multiplication by $q_0q_1$, represents it over $B_2$, and produces

$$
\Delta'=\frac{\Delta_1^2}{\rho_1}=\Delta_2,
\qquad
\rho_1=\frac{q_{21}q_{22}q_{23}}{q_0q_1}.
$$

The result has `depth_remaining == 14`. The rescale implementation performs one rounding step for the three dropped primes and one basis conversion to the retained and introduced rows; together they implement one CKKS depth transition. Its RNS kernels use the format selected from the complete prime set when the Engine was built.
