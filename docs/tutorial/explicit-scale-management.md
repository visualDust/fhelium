# Explicit scale management

**Example source:** [`examples/05_explicit_scale_management.py`](https://github.com/VisualDust/fhelium/blob/main/examples/05_explicit_scale_management.py)

This example plans two plaintext-multiplication scales against the actual Q
prime, tracks every per-value scale transition, and aligns a level separately.
The tutorial also applies a bounded scale-metadata reinterpretation. Scale and
level transition laws are defined in
[Scale and level lifecycle](../concepts/ckks/scale-and-level-lifecycle.md).

## Run the example

Start with the shortest preset:

```bash
python examples/05_explicit_scale_management.py --preset slots8192-scale40-levels7-int64
```

## Track the actual scale

When no scale is supplied, encoding and encryption use:

```python
Delta = engine.config.default_scale
```

Every value then owns its actual binary64 scale. Two plaintext products update
it without consuming a level:

$$
\Delta(c_{\mathrm{pre}})=\Delta(c_{\mathrm{in}})\Delta(p_1)\Delta(p_2).
$$

Public `rescale_to_next_level` divides by the actual leading Q prime:

$$
\Delta(c_{\mathrm{out}})=
\frac{\Delta(c_{\mathrm{pre}})}{q_{\mathrm{drop}}}.
$$

The result stores this quotient as its actual scale.

## Query the arithmetic before encoding

The engine provides two pure transition queries:

```python
q = engine.rescale_to_next_drop_prime(level=ciphertext.level)
output_scale = engine.rescale_to_next_output_scale(
    input_scale=pre_rescale_scale,
    level=ciphertext.level,
)
```

`rescale_to_next_drop_prime` supplies the divisor used to choose operand
scales. `rescale_to_next_output_scale` calculates the actual output scale after
the products have been evaluated.

For two plaintext products followed by one `rescale_to_next_level`, targeting
`Delta` gives:

```python
first_scale = 2**20
second_scale = Delta * q / (ciphertext.scale * first_scale)
```

The example uses this equation before encoding either operation-ready
plaintext. Its planned branch reaches `Delta` after `rescale_to_next_level` divides
by the actual drop prime.

## Align level independently from scale

The original branch remains at level zero. The planned product is at level one.
A modulus switch aligns the original value without changing its scale:

```python
level_aligned = engine.mod_switch_to_level(
    ciphertext,
    planned_product.level,
)
combined = engine.add(planned_product, level_aligned)
```

This addition succeeds because the program independently arranged:

1. the same level, through modulus switch;
2. the same scale, through plaintext-scale planning.

The operands therefore satisfy all addition preconditions before `add` is
called.

## Apply a bounded metadata reinterpretation

If the plaintext-scale product is `Delta` rather than the actual prime
$q_{\mathrm{drop}}$,
`rescale_to_next_level` reports:

$$
\Delta^2/q_{\mathrm{drop}},
$$

which differs from `Delta` by the actual-prime ratio. A bounded reinterpretation
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

## Complete runnable source

<<< @/../examples/05_explicit_scale_management.py

## Next step

Continue with [Late relinearization and NTT reuse](late-relinearization-and-ntt-reuse.md)
to combine scale transitions with three-component products and
representation reuse. The complete transition laws are defined in
[Scale and level lifecycle](../concepts/ckks/scale-and-level-lifecycle.md), and
the parameter invariants are described in
[Context and modulus chain](../concepts/ckks/context-and-modulus-chain.md).
