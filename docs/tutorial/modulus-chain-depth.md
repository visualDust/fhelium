# Modulus-chain depth

**Example source:** [`examples/03_eager_modulus_chain.py`](https://github.com/VisualDust/fhelium/blob/main/examples/03_eager_modulus_chain.py)

This example prints the active Q basis, default scale, and rescale ratio of selected depths of one preset, together with the ciphertext storage at each depth.

## Run the example

```bash
python examples/03_eager_modulus_chain.py \
  --preset slots32768-scale40-depth34
```

Select the depths to display:

```bash
python examples/03_eager_modulus_chain.py \
  --preset slots32768-scale40-depth34 \
  --depths 0,16,33
```

## 1. Each depth selects an interval of Q primes

A resolved configuration stores

```python
config.q_moduli        # ordered Q primes q_0 ... q_{L-1}
config.depth_bases     # (start, stop) of the active Q basis B_d at every depth
config.max_depth       # D = len(depth_bases) - 1
```

as prime values and interval bounds. `config.q_prime_ids(d)` lists the Q prime IDs of $B_d$, and `config.active_q_moduli(d)` lists their values. Public values use depths from zero through $D$; $B_0$ contains every Q prime, and $B_D$ is the terminal basis.

The example's `Q rows` and `log2 B_d` columns report $|B_d|$ and $\log_2\prod_{q\in B_d}q$.

## 2. A rescale drops and introduces primes

One public rescale from depth $d$ replaces $B_d$ by $B_{d+1}$. It drops $D_d=B_d\setminus B_{d+1}$, introduces $R_d=B_{d+1}\setminus B_d$, and divides actual scale by

$$
\rho_d=\frac{\prod_{q\in D_d}q}{\prod_{q\in R_d}q}.
$$

The `|D|/|R|` column reports the number of dropped and introduced primes, from `config.basis_transition(d, d + 1)`, and `log2 rho_d` reports $\log_2\rho_d$ from `config.rescale_divisor(d)`. The built-in presets combine primes near $2^{30}$ and $2^{25}$ so that $\rho_d$ is close to the preset scale while the active row count decreases by about one or two rows per depth.

## 3. Default scales follow the ratios

`config.scale_at(d)` is the scale used when a value is created at depth $d$ without an explicit scale. The `log2 scale` column shows these values. The terminal entry equals the preset scale, and earlier entries satisfy $\Delta_d=\sqrt{\Delta_{d+1}\rho_d}$, so a product of two default-scale values at depth $d$ rescales to $\Delta_{d+1}$.

## 4. Q and P have different roles

- The Q primes and their per-depth bases form the ciphertext-modulus chain.
- P contains special primes used temporarily by hybrid key switching.
- QP appends all P rows to the active Q basis without changing depth.

`config.total_modulus_bits` is the bit length of the complete QP product. When security-budget enforcement is enabled, it must not exceed `config.maximum_modulus_bits`.

## 5. Depth-zero values are largest

```python
ciphertext = engine.encrypt_message([1, 2, 3, 4], depth=0)
print(ciphertext.data.nbytes)
```

At depth zero, every Q prime is active. Later depths contain fewer Q rows. For a two-component ciphertext, payload storage is approximately

$$
2\,L_Q\,N\,W,
$$

where $L_Q=|B_d|$ is the active Q-row count and $W$ is the residue element size. Evaluation keys and operation temporaries have additional axes and lifetimes.

## 6. Choose depth from the circuit

Count the public rescale transitions on the intended execution path and reserve the required margin. Then validate precision, message range, security budget, key storage, and latency with the selected configuration:

- more depths provide more transitions;
- more prime rows increase early-depth ciphertext and prepared-plaintext size;
- key switching and relinearization touch the active QP rows;
- the rescale ratios determine the scale removed by each rescale.

::: details Source
<<< @/../examples/03_eager_modulus_chain.py
:::

## Related concepts and guides

- [Scale, depth, and native RNS dispatch](../concepts/ckks/scale-depth-and-execution-format.md)
- [Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md)
- [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md)
- [Choose a preset and chain depth](../how-to/choose-preset-and-depth.md)
