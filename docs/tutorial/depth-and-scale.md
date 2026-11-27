# Depth and scale

**Example source:** [`examples/03_eager_depth_and_scale.py`](https://github.com/VisualDust/fhelium/blob/main/examples/03_eager_depth_and_scale.py)

This example prints the modulus chain of one preset, the active Q basis, default scale, and rescale ratio of selected depths, and then plans the scales of two plaintext products so that the rescaled result can be added to a second branch aligned by a modulus switch. It closes with a power-of-two scale allocation and a bounded scale reinterpretation.

## Run the example

```bash
python examples/03_eager_depth_and_scale.py --preset slots8192-scale40-depth7
```

Select the depths of the chain table:

```bash
python examples/03_eager_depth_and_scale.py \
  --preset slots32768-scale40-depth34 \
  --depths 0,16,33
```

## 1. Each depth selects a set of Q primes

A resolved configuration stores

```python
config.q_moduli    # ordered Q primes q_0 ... q_{L-1}
config.q_bases     # prime IDs of the active Q basis B_d at every depth
config.max_depth   # D = len(q_bases) - 1
```

`config.q_prime_ids(d)` lists the prime IDs of $B_d$ and `config.active_q_moduli(d)` their values. Values use depths from zero through $D$; $B_0$ contains every Q prime. The `Q rows` and `log2 B_d` columns report $|B_d|$ and $\log_2\prod_{q\in B_d}q$, and `remaining` reports `config.depth_remaining(d)`.

## 2. A rescale drops and introduces primes

A rescale from depth $d$ replaces $B_d$ by $B_{d+1}$. It drops $D_d=B_d\setminus B_{d+1}$, introduces $R_d=B_{d+1}\setminus B_d$, and divides the actual scale by

$$
\rho_d=\frac{\prod_{q\in D_d}q}{\prod_{q\in R_d}q}.
$$

The `|D|/|R|` column reports the numbers of dropped and introduced primes from `config.basis_transition(d, d + 1)`, and `log2 rho_d` reports $\log_2\rho_d$ from `config.rescale_divisor(d)`. The built-in presets combine primes near $2^{30}$ and $2^{25}$ so that $\rho_d$ is close to the preset scale while the active row count falls by one or two rows per depth.

`config.scale_at(d)` is the scale used when a value is created at depth $d$ without a `scale` argument; the `log2 scale` column lists it. The maximum-depth entry equals the preset scale, and earlier entries satisfy $\Delta_d=\sqrt{\Delta_{d+1}\rho_d}$, so a product of two default-scale values at depth $d$ rescales to $\Delta_{d+1}$.

The `CT size` column shows that a depth-0 ciphertext is the largest. A two-component ciphertext stores approximately $2\,|B_d|\,N\,W$ bytes, where $W$ is the residue element size; the header line reports the residue dtype selected from the prime widths, `int32` for every built-in preset.

## 3. Plan two plaintext products against the actual ratio

Every value owns its actual binary64 scale. Two plaintext products multiply it without consuming a depth, and one rescale divides by $\rho_d$:

$$
\Delta(c_{\mathrm{out}})=\frac{\Delta(c_{\mathrm{in}})\Delta(p_1)\Delta(p_2)}{\rho_d}.
$$

The example reads $\rho_0$ before encoding either plaintext and solves for the second scale that reaches a chosen target:

```python
rescale_ratio = engine.rescale_divisor(depth=source.depth)
first_scale = float(1 << (math.floor(math.log2(target_scale)) // 2))
second_scale = target_scale * float(rescale_ratio) / (source.scale * first_scale)
```

`engine.rescale_output_scale(pre_rescale_scale, depth=d)` computes the recorded scale after one rescale without residue arithmetic; the example prints it beside the target.

## 4. Align depth independently from scale

The second branch stays at depth zero until a modulus switch moves it to depth one:

```python
depth_aligned_source = engine.mod_switch_to_depth(source, source.depth + 1)
target_scale = depth_aligned_source.scale
```

When $B_1\subseteq B_0$, the switch selects rows and keeps the scale; otherwise it multiplies by an integer and rescales, and the result records `config.scale_at(1)`. The example uses the switched scale as the target of the planned product, so the two branches have the same depth and the same binary64 scale, and `engine.add(planned, depth_aligned_source)` receives compatible operands.

## 5. Reinterpret a mismatched scale within a bound

Replacing $\rho_0$ by the nearest power of two $2^k$ in the allocation gives the scale $\Delta_t2^k/\rho_0$ after rescale, which differs from the target $\Delta_t$ by the factor $2^k/\rho_0$; the example prints that relative difference. A bounded reinterpretation records the target scale and leaves the residues unchanged:

```python
reinterpreted = engine.reinterpret_at_scale(
    approximate, target_scale, max_relative_change=args.reinterpret_bound
)
```

The decoded message is multiplied by `old_scale / target_scale`, and the bound limits the accepted ratio. The final table compares each decoded result with the corresponding cleartext expression.

::: details Source
<<< @/../examples/03_eager_depth_and_scale.py
:::

## Next step

Continue with [Late relinearization and NTT reuse](late-relinearization-and-ntt-reuse.md). The chain is defined in [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md), and the scale and depth effects of every operation in [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md). [Choose a preset and chain depth](../how-to/choose-preset-and-depth.md) applies them to parameter selection.
