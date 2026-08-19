# Modulus-chain depth

**Example source:** [`examples/04_modulus_chain_depth.py`](https://github.com/VisualDust/fhelium/blob/main/examples/04_modulus_chain_depth.py)

This example builds several modulus-chain depths for one preset and compares
modulus bits, active RNS rows, security estimates, and value sizes. The
tutorial connects the evaluator's required rescale/level-transition budget to
configured chain depth, security budget, and ciphertext memory.

## Run the example

Inspect low, middle, and full-depth variants of a preset:

```bash
python examples/04_modulus_chain_depth.py --preset slots32768-scale40-levels34-int64
```

Select exact depth values:

```bash
python examples/04_modulus_chain_depth.py \
  --preset slots32768-scale40-levels34-int64 \
  --depths 16,24,34
```

## 1. Scale-prime count fixes the public-level interval

```python
cfg = CkksConfig.parse(preset, num_scale_primes=depth)
```

`num_scale_primes` counts configured scale-prime rows and equals
`engine.public_level_count`. Public levels are therefore
`[0, num_scale_primes)`, and the number of ordinary public one-level
transitions available from level zero is `num_scale_primes - 1`.

The level-zero ordinary modulus contains those scale primes plus one structural
base Q prime, so

$$
\mathtt{num\_q\_primes}
=
\mathtt{num\_scale\_primes}+1.
$$

The final public level retains the last scale prime and the structural base.
Bootstrap entry owns the subsequent transition into the base-only state.

For the maintained int64 presets, a useful approximation is:

$$
\operatorname{bits}(QP)
\approx
b_sL + b_b + 60K_P,
$$

where $L$ is `num_scale_primes`, $b_s$ is `scale_bits`, $b_b$ is the
structural-base-prime width, and $K_P$ is `num_p_primes`. The maintained
int64 presets use $b_s\in\{30,40,50\}$ and the default 60-bit structural
base. Maintained int32 presets use $b_s=25$ and 28-bit structural/P primes,
giving the separate approximation

$$
\operatorname{bits}(QP)\approx25L+28(1+K_P).
$$

The exact catalog primes remain authoritative. In particular, every native
modulus must also satisfy $4q<2^w$, where $w$ is the configured residue buffer
width; security-budget capacity alone does not establish native arithmetic
validity.

`total_modulus_bits` is the exact configured value
$\lceil\log_2(Q_0P)\rceil$. It covers both the ordinary Q-chain primes $q_i$
and the special-prime product $P$, not Q alone. The configuration requires
`total_modulus_bits <= maximum_modulus_bits` when
`enforce_security_budget=True`.

The exact primes come from the immutable catalog. The configuration validates
that the selected chain remains within the requested security budget.

## 2. Q and P have different roles

- Q rows form the ordinary ciphertext modulus chain.
- One leading scale prime is consumed by each rescale.
- The base Q row remains at the end of the chain.
- P rows support hybrid key switching and are not ordinary ciphertext rows.

The table printed by the example distinguishes `Q primes`, `P primes`, and
`total primes` rather than reporting one ambiguous limb count.

## 3. Level-zero values are largest

```python
ct0 = engine.encrypt_message([1, 2, 3, 4], level=0)
print(ct0.data.nbytes)
```

At level zero, every Q row is active. A level transition drops leading Q rows,
so a later-level ciphertext is smaller.

For a two-component ciphertext, the approximate payload size is:

$$
B_{\mathrm{ct}}
\approx
2L_QN \cdot W\ \text{bytes},
$$

where $L_Q$ is the number of active Q rows, $N$ is the ring dimension, and
$W$ is four bytes for int32 or eight bytes for int64.

Allocator overhead and temporary operation storage are separate from this
payload calculation.

## 4. Compare costs at the same active level

Initial configured chain depth and current active Q-row count are different
quantities. Two configurations that have reached the same active Q-row count can have similar
current ciphertext sizes even if one started with a longer chain.

Conversely, comparing only level zero makes a longer initial chain look more
expensive because it genuinely stores more rows at that point.

## 5. Choose depth from the circuit

Count rescale operations in the intended circuit and reserve a small
engineering margin. Do not always select the largest chain depth simply because it
fits the security table:

- more initial Q rows increase ciphertext and prepared-plaintext memory;
- key-switch and relinearization work touches more active rows;
- key material can dominate serving capacity;
- unnecessary depth makes early-level operations more expensive.

::: info Level is not an abstract counter
In FHElium, level determines an exact ordered `prime_ids` interval and a
concrete dense tensor shape. Operations validate this structure rather than
trusting level metadata alone.
:::

::: details Complete runnable source
<<< @/../examples/04_modulus_chain_depth.py
:::

## Related concepts and guides

- [Scale and level lifecycle](../concepts/ckks/scale-and-level-lifecycle.md)
- [Context and modulus chain](../concepts/ckks/context-and-modulus-chain.md)
- [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
- [Choose a preset and chain depth](../how-to/choose-preset-and-depth.md)
