# Explicit scale management

**Example source:** [`examples/04_eager_scale_management.py`](https://github.com/VisualDust/fhelium/blob/main/examples/04_eager_scale_management.py)

This example plans two plaintext-multiplication scales against the actual rescale ratio, tracks every per-value scale transition, and aligns a depth separately. The tutorial also applies a bounded scale-metadata reinterpretation. Scale and depth transition laws are defined in [Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md).

## Run the example

Start with the shortest preset:

```bash
python examples/04_eager_scale_management.py --preset slots8192-scale40-depth7
```

## Track the actual scale

When no scale is supplied, encoding and encryption at depth $d$ use that depth's default scale:

```python
Delta_d = engine.config.scale_at(d)
```

Every value then owns its actual binary64 scale. Two plaintext products update it without consuming a depth:

$$
\Delta(c_{\mathrm{pre}})=\Delta(c_{\mathrm{in}})\Delta(p_1)\Delta(p_2).
$$

Public `rescale_to_next_depth` divides by the rescale ratio of the transition $d\to d+1$:

$$
\Delta(c_{\mathrm{out}})=
\frac{\Delta(c_{\mathrm{pre}})}{\rho_d},
\qquad \rho_d=\frac{\prod_{q\in D_d}q}{\prod_{q\in R_d}q},
$$

where $D_d$ and $R_d$ are the Q primes that leave and enter the active basis.

The result stores this quotient as its actual scale.

## Query the arithmetic before encoding

The engine provides two pure transition queries:

```python
rescale_ratio = engine.rescale_divisor(depth=ciphertext.depth)
output_scale = engine.rescale_output_scale(
    input_scale=pre_rescale_scale,
    depth=ciphertext.depth,
)
```

`rescale_divisor` returns $\rho_d$ as a `Fraction`, used to choose operand scales. `rescale_output_scale` calculates the actual output scale after the products have been evaluated.

For two plaintext products followed by one `rescale_to_next_depth`, targeting the scale `target` gives:

```python
first_scale = 2**20
second_scale = target * float(rescale_ratio) / (ciphertext.scale * first_scale)
```

The example uses this equation before encoding either operation-ready plaintext. Its target is the scale of the source after a modulus switch to the next depth, and its planned branch reaches that scale after `rescale_to_next_depth` divides by $\rho_d$.

## Align depth independently from scale

The original branch remains at depth zero. The planned product is at depth one. A modulus switch aligns the original value to depth one. When $B_1\subseteq B_0$ it selects rows and keeps the scale; otherwise it multiplies by an integer and rescales, and the result carries `config.scale_at(1)`. The example reads the switched scale and plans the product to match it:

```python
level_aligned = engine.mod_switch_to_depth(
    ciphertext,
    planned_product.depth,
)
combined = engine.add(planned_product, level_aligned)
```

This addition succeeds because the program independently arranged:

1. the same depth, through modulus switch;
2. the same scale, through plaintext-scale planning.

The operands therefore satisfy all addition preconditions before `add` is called.

## Apply a bounded metadata reinterpretation

If the second plaintext scale replaces $\rho_d$ by the nearest power of two $2^k$, `rescale_to_next_depth` reports:

$$
\Delta_t\,2^k/\rho_d,
$$

which differs from the target scale $\Delta_t$ by the factor $2^k/\rho_d$. A bounded reinterpretation records an application-approved target scale:

```python
reinterpreted = engine.reinterpret_at_scale(
    actual_result,
    target,
    max_relative_change=5e-2,
)
```

The ciphertext residues remain unchanged. The decoded message is multiplied by `old_scale / target`, and the configured bound limits the accepted scale-ratio bias.

::: details Source

<<< @/../examples/04_eager_scale_management.py

:::

## Next step

Continue with [Late relinearization and NTT reuse](late-relinearization-and-ntt-reuse.md) to combine scale transitions with three-component products and representation reuse. The complete transition laws are defined in [Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md), and the parameter invariants are described in [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md).
