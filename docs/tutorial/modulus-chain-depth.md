# Modulus-chain depth

**Example source:** [`examples/04_modulus_chain_depth.py`](https://github.com/VisualDust/fhelium/blob/main/examples/04_modulus_chain_depth.py)

This example derives several exact Q chains from one preset and compares their
maximum public depth, modulus size, active RNS rows, selected execution dtype,
and ciphertext storage.

## Run the example

```bash
python examples/04_modulus_chain_depth.py \
  --preset slots32768-scale40-depth34-int64
```

Select maximum depths:

```bash
python examples/04_modulus_chain_depth.py \
  --preset slots32768-scale40-depth34-int64 \
  --depths 16,24,33
```

## 1. Maximum depth follows Q-group count

A resolved configuration stores

```python
config.q_depth_groups
config.max_depth
```

as exact values. If the groups are

$$
(G_0,G_1,\ldots,G_D),
$$

then `max_depth == D`. Public values use depths from zero through $D$, and
`G_D` is the ordinary terminal Q group.

The example's `prefix_config` function constructs a shorter exact parameter set
by retaining the requested public-group prefix and the same terminal Q group:

```python
CkksConfig(
    default_scale=source.default_scale,
    q_depth_groups=(
        *source.q_depth_groups[:max_depth],
        source.q_depth_groups[-1],
    ),
    p_moduli=source.p_moduli,
    logN=source.logN,
)
```

No count override regenerates an existing configuration. Each constructed
`CkksConfig` records its complete Q groups and P primes.

## 2. One depth may contain several RNS rows

One public rescale from depth $d$ removes the complete group $G_d$ and divides
actual scale by

$$
M_d=\prod_{q\in G_d}q.
$$

A one-prime group and a two-prime group both consume one depth. They have
different active row counts and may select different native execution formats.
Use `config.rescale_divisor(d)` for the group product and
`config.active_q_moduli(d)` for the active prime sequence.

## 3. Q and P have different roles

- Q depth groups form the ciphertext-modulus chain.
- P contains special primes used temporarily by hybrid key switching.
- QP appends all P rows to the active Q basis without changing depth.

`config.total_modulus_bits` is the bit length of the exact complete QP product.
When security-budget enforcement is enabled, it must not exceed
`config.maximum_modulus_bits`.

## 4. Depth-zero values are largest

```python
ciphertext = engine.encrypt_message([1, 2, 3, 4], depth=0)
print(ciphertext.data.nbytes)
```

At depth zero, every Q group is active. Later depths contain fewer Q rows. For
a two-component ciphertext, payload storage is approximately

$$
2\,L_Q\,N\,W,
$$

where $L_Q$ is the active Q-row count and $W$ is the residue element size.
Evaluation keys and operation temporaries have additional axes and lifetimes.

## 5. Choose depth from the circuit

Count the public rescale transitions on the intended execution path and reserve
the required margin. Then validate precision, message range, security budget,
key storage, and latency with the exact groups:

- more public groups provide more transitions;
- more prime rows increase early-depth ciphertext and prepared-plaintext size;
- key switching and relinearization touch the active QP rows;
- group products determine the scale removed by each rescale.

::: details Source
<<< @/../examples/04_modulus_chain_depth.py
:::

## Related concepts and guides

- [Scale, depth, and native RNS dispatch](../concepts/ckks/scale-depth-and-execution-format.md)
- [Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md)
- [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md)
- [Choose a preset and chain depth](../how-to/choose-preset-and-depth.md)
