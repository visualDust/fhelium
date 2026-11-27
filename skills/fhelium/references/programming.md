# Programming CKKS with FHElium

## Choose a preset

1. Count slots. A ring with `logN = k` has `2^(k-1)` complex slots; count padding and intermediate layouts. Presets: `slots8192` (logN 14), `slots16384` (15), `slots32768` (16), `slots65536` (17).
2. Count depth. Each rescale consumes one depth: a ciphertext × ciphertext product followed by relinearize-and-rescale, a plaintext product followed by rescale, and each multiplicative level of a polynomial. Example: `(W x)` by a plaintext linear transform (1) then a degree-7 polynomial by Paterson-Stockmeyer (about 3–4) needs about 5 depths; `slots8192-scale40-depth7` fits.
3. Choose the scale. 40-bit scales are the usual default; 30-bit gives more depth with less precision, 50-bit the opposite. Scale bits set precision, not range; keep magnitudes of inputs and intermediates bounded (polynomial approximations assume an input interval).
4. Bootstrapping: use a `-boot{k}` preset (`slots32768-scale40-depth29-boot13` etc.); `k` application depths remain after each refresh.
5. Inspect: `config = fh.CkksConfig.parse(preset)`; `config.max_depth`, `config.q_bases`, `config.scale_at(d)`, `config.num_slots`, `config.logN`. `examples/03_eager_depth_and_scale.py --preset NAME` prints the chain.

Depth is a consumed-depth index: 0 is the full Q basis, `engine.max_depth` is the last usable depth, `engine.depth_remaining(value)` is the number of rescales left. Report depth transitions as `input depth -> output depth`.

Residues are `torch.int32` when every prime is below 2^30 (all built-in presets) and `torch.int64` otherwise; `engine.dtype` reports it.

## Keys

```python
engine = Engine(fh.Preset.slots16384_scale40_depth16)
sk = engine.create_secret_key(device=device)
pk = engine.create_public_key(sk, device=device)
relin = engine.create_relinearization_key(sk, device=device)
rot = {s: engine.create_rotation_key(s, sk, device=device) for s in steps}
conj = engine.create_conjugation_key(sk, device=device)
keys = fh.EvaluationKeySet([relin, *rot.values()])
```

- Rotation steps follow `torch.roll(message, shifts=step)`.
- Required steps: `linear.plan(t).required_rotation_offsets(t)` for a linear transform; `bootstrap.create_evaluation_keys(engine, plan, sk)` creates all bootstrap keys (`plan.key_requirements` lists them); `fhelium.ir.analyze_evaluation_key_requirements(program)` returns `EvaluationKeyRequirements(rotation_steps, requires_relinearization, requires_conjugation)` for a Program.
- Keys are created at `source_depth=0` by default and serve every lower depth. `Engine(expand_keys=True)` (default) stores expanded key layouts; compact keys loaded from files are expanded with `KeySwitchKey.expand()` when needed.
- Installed-key methods (`rotate_by_step`, `relinearize(v)` without a key, `rotation_key(step)`) use keys installed with `set_rotation_key`/`install_evaluation_key`. With the default `allow_automatic_key_generation=True` they generate missing keys from the installed secret key; use `Engine(..., allow_automatic_key_generation=False)` in applications so missing keys raise `KeyError` instead. Compiled execution and `fc.load` never generate keys.
- Docs: [Provision a key set](https://fhelium.550w.host/how-to/provision-keyset), example 02.

## State transitions

| Operation | Input requirement | Result |
| --- | --- | --- |
| `encrypt_message(m, pk, depth=d)` | — | depth d, default scale of d, coefficient domain |
| `coefficient_domain_to_ntt_domain(c)` / inverse | — | domain changes; depth and scale unchanged |
| `add`, `subtract` | same depth, scale, domain | same state |
| `multiply(a, b)` | NTT domain, same depth | 3 components, scale `sa*sb` |
| `relinearize(c, key, rescale=False)` | 3 components | 2 components; with `rescale=True` depth+1 and scale divided by the dropped primes |
| `multiply_plaintext(c, p)` | p prepared at c's depth | scale `sc*sp`, no implicit rescale |
| `rescale_to_next_depth(c)` | — | depth+1, scale `s*R/D` (D: Q primes dropped, R: Q primes added; either may be several) |
| `mod_switch_to_depth(c, d)` | d ≥ c.depth | depth d, scale unchanged |
| `rotate_with_key(c, key)` | key for the step | same depth and scale |
| `reinterpret_at_scale(c, s)` | — | scale label changes, data unchanged |

Rules:

- Bring operands to the same depth with `mod_switch_to_depth` before `add`. Operands at the same depth from different paths can carry different actual scales; `add`, `subtract`, `add_plaintext`, `sum_ciphertexts`, and `sum_plaintext_products` raise `ScaleMismatchError` for unequal scales and `StateError` for other mismatched fields, so read `value.scale` and align them.
- The actual scale is per value (`value.scale`) and differs from the preset's nominal `2^bits`. Do not assume one prime drop per depth: a depth can drop a group of primes.
- Encode plaintexts at the depth of the ciphertext they multiply: `engine.prepare_compressed_plaintext(m, depth=c.depth, device=c.device)`. A message shorter than the slot count that tiles evenly is stored compressed.
- Docs: [State transitions](https://fhelium.550w.host/concepts/ckks/value-state-transitions), [Diagnose a value-state mismatch](https://fhelium.550w.host/how-to/diagnose-value-state-mismatch), example 03.

## Structured operations

Prefer these over Python loops; each runs fused or hoisted kernels and compiles to fewer nodes.

- Sum of products `Σ c_i · p_i`: `engine.sum_plaintext_products(cts, pts)`; stack plaintexts with `fh.Plaintext.stack_batch` for many terms.
- Several rotations of one ciphertext: `engine.rotate_many_with_keys(c, keys)` (hoisted decomposition).
- `Σ rot(c_i, s_i)`: `engine.sum_rotations([(c_i, key_i), ...])` (`None` key for step 0).
- Baby-step/giant-step matvec: `engine.sum_rotated_plaintext_product_groups`, or the higher-level `fhelium.algorithms.linear`:

```python
from fhelium.algorithms import linear, polynomial
t = linear.LinearTransform({step: diagonal, ...}, engine.num_slots)  # diagonal tensors of length slots
p = linear.plan(t)                       # DirectPlan or BsgsPlan
keys = fh.EvaluationKeySet(engine.create_rotation_key(s, sk) for s in sorted(p.required_rotation_offsets(t)))
prepared = linear.prepare(engine, x, t, p, keys=keys)   # encodes diagonals once at x's depth; or linear.apply(engine, x, t, p, keys=keys)
y = linear.apply(engine, x, prepared=prepared)          # consumes one depth

f = polynomial.chebyshev_fit(np.tanh, 15, domain=(-4.0, 4.0))  # NumPy function, input interval
y = polynomial.evaluate(engine, y, f, polynomial.plan(f), keys=fh.EvaluationKeySet([relin]))  # degree 15: 5 depths
```

- Docs: [Linear layer and polynomial activation](https://fhelium.550w.host/tutorial/linear-layer-and-polynomial), [Rotation hoisting](https://fhelium.550w.host/tutorial/rotation-hoisting), [Compressed plaintexts](https://fhelium.550w.host/tutorial/compressed-plaintext), examples 04–08.

## Bootstrapping

```python
config = fh.CkksConfig.parse(fh.Preset.slots32768_scale40_depth29_boot13)
engine = Engine(config, allow_automatic_key_generation=False)
plan = bootstrap.plan(engine)                 # default profile "3/38"; intermediate_weight=... for encapsulated raise
keys = bootstrap.create_evaluation_keys(engine, plan, sk)
prepared = bootstrap.prepare(engine, plan, x, keys=keys)
fresh = bootstrap.bootstrap(engine, x, prepared=prepared)   # or fc.compile(prepared)(x)
```

`plan.output_depth` is the depth after refresh. Inputs must lie in the message range the EvalMod approximation supports. Docs: [Bootstrap a full-slot CKKS ciphertext](https://fhelium.550w.host/tutorial/composable-ckks-bootstrap), [Compose a bootstrap plan](https://fhelium.550w.host/how-to/compose-bootstrap-circuit), example 09.

## Placement and batching

- Pass `device=` to key, encryption, and encoding factories. Operations follow operands; `value.to(device)` moves values; keys must be on the operand's device (`allow_automatic_key_replication=True` replicates them on demand).
- Batch independent ciphertexts along a leading dimension for one kernel per operation: [Homogeneous batching](https://fhelium.550w.host/tutorial/homogeneous-batching), example 06.
- Multi-GPU: `fhelium.distributed`, examples 20–23, [Choose a multi-GPU partition](https://fhelium.550w.host/how-to/choose-multi-gpu-partition).
- Save and load values: `fh.save_value(value, path)` (`allow_secret=True` for secret keys), `fh.load_value(path, device=...)`, example 10; named stores: `fhelium.store.ValueStore`, example 11.
