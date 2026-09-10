# Explicit scale management

**Example source:** [`examples/05_explicit_scale_management.py`](https://github.com/VisualDust/fhelium/blob/main/examples/05_explicit_scale_management.py)

This example plans two plaintext-multiplication scales against the actual
Q-group divisor, tracks every per-value scale transition, and aligns a depth
separately.
The tutorial also applies a bounded scale-metadata reinterpretation. Scale and
depth transition laws are defined in
[Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md).

## Run the example

Start with the shortest preset:

```bash
python examples/05_explicit_scale_management.py --preset slots8192-scale40-depth7-int64
```

## Track the actual scale

When no scale is supplied, encoding and encryption use:

```python
Delta = engine.config.default_scale
```

Every value then owns its actual binary64 scale. Two plaintext products update
it without consuming a depth:

$$
\Delta(c_{\mathrm{pre}})=\Delta(c_{\mathrm{in}})\Delta(p_1)\Delta(p_2).
$$

Public `rescale_to_next_depth` divides by the actual leading Q-group product:

$$
\Delta(c_{\mathrm{out}})=
\frac{\Delta(c_{\mathrm{pre}})}{M_{\mathrm{drop}}},
\qquad M_{\mathrm{drop}}=\prod_{q\in G_d}q.
$$

The result stores this quotient as its actual scale.

## Query the arithmetic before encoding

The engine provides two pure transition queries:

```python
drop_divisor = engine.rescale_divisor(depth=ciphertext.depth)
output_scale = engine.rescale_output_scale(
    input_scale=pre_rescale_scale,
    depth=ciphertext.depth,
)
```

`rescale_divisor` supplies the complete group product used to choose operand
scales. `rescale_output_scale` calculates the actual output scale after
the products have been evaluated.

For two plaintext products followed by one `rescale_to_next_depth`, targeting
`Delta` gives:

```python
first_scale = 2**20
second_scale = Delta * drop_divisor / (ciphertext.scale * first_scale)
```

The example uses this equation before encoding either operation-ready
plaintext. Its planned branch reaches `Delta` after `rescale_to_next_depth` divides
by the actual leading group product.

## Align depth independently from scale

The original branch remains at depth zero. The planned product is at depth one.
A modulus switch aligns the original value without changing its scale:

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

The operands therefore satisfy all addition preconditions before `add` is
called.

## Apply a bounded metadata reinterpretation

If the plaintext-scale product is `Delta` rather than the actual leading-group
product $M_{\mathrm{drop}}$,
`rescale_to_next_depth` reports:

$$
\Delta^2/M_{\mathrm{drop}},
$$

which differs from `Delta` by the actual-group-divisor ratio. A bounded reinterpretation
records an application-approved target scale:

```python
reinterpreted = engine.reinterpret_at_scale(
    actual_result,
    Delta,
    max_relative_change=1e-2,
)
```

The ciphertext residues remain unchanged. The decoded message is multiplied by
`old_scale / Delta`, and the configured bound limits the accepted scale-ratio
bias.

::: details Source

<<< @/../examples/05_explicit_scale_management.py

:::

## Next step

Continue with [Late relinearization and NTT reuse](late-relinearization-and-ntt-reuse.md)
to combine scale transitions with three-component products and
representation reuse. The complete transition laws are defined in
[Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md), and
the parameter invariants are described in
[Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md).
