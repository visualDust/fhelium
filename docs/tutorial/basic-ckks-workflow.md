# Basic CKKS workflow

**Example source:** [`examples/01_basic_ckks_flow.py`](https://github.com/VisualDust/fhelium/blob/main/examples/01_basic_ckks_flow.py)

This example encrypts dense tensor messages, evaluates independent addition,
multiplication, and rotation branches, then decrypts and checks each result.
The tutorial explains the CKKS state transitions in that baseline
workflow.

## 1. Create a local engine

```python
import fhelium as fh

engine = fh.CkksEngine(fh.Preset.slots8192_scale40_levels7_int64, device="cpu")
```

A `CkksEngine` is process-local and owns one device. Distributed execution is
expressed separately through `fhelium.distributed` collectives.
Selecting `device="cuda:0"` dispatches the same engine operations and
`torch.ops` schemas to CUDA; FHElium does not use backend-specific public
methods or hidden transfers.

## 2. Encrypt messages

```python
import torch

x = torch.linspace(-0.05, 0.05, engine.num_slots, dtype=torch.float64)
y = torch.linspace(0.02, -0.02, engine.num_slots, dtype=torch.float64)

ct_x = engine.encrypt_message(x)
ct_y = engine.encrypt_message(y)
```

The returned [`Ciphertext`](../api/fhelium/core/ciphertext.md#ciphertext) carries its level, scale,
prime IDs, polynomial domain, modulus basis, and residue representation alongside one
dense tensor.

## 3. Evaluate an operation that preserves state

```python
ct_sum = engine.add(ct_x, ct_y)
```

`add` is out of place and requires compatible ciphertext layouts. It does not
change the level or scale.

## 4. Prepare and multiply ciphertexts

```python
mul_x = engine.coefficient_domain_to_ntt_domain(ct_x)
mul_y = engine.coefficient_domain_to_ntt_domain(ct_y)
product_triplet = engine.multiply(mul_x, mul_y)
ct_product = engine.rescale_to_next_level(engine.relinearize(product_triplet))
```

FHElium deliberately does not hide rescale or relinearization. This makes the
level, representation, and key-switch transitions visible to algorithms that
reuse NTT-domain operands or delay relinearization. With default-scale inputs,
the product carries scale $\Delta^2$; the post-relinearization rescale consumes
one level and records the actual scale $\Delta^2/q_0$.

## 5. Rotate with an exact key

```python
rotation_key = engine.rotation_key(1)
ct_rotated = engine.rotate_with_key(ct_x, rotation_key)
```

A rotation key is bound to one canonical signed step. Applications choose
which keys exist and where they reside.

## 6. Decrypt and check approximation error

```python
sum_clear = engine.decrypt_message(ct_sum)[: engine.num_slots]
torch.testing.assert_close(sum_clear, x + y, atol=2e-5, rtol=0)
```

CKKS is approximate. Validate results with an chosen numerical tolerance appropriate
for the scale, depth, input range, and workload.

## Complete runnable source

The source below is included directly from the tested repository example, so
the tutorial does not maintain a second copy of the complete program.

<<< @/../examples/01_basic_ckks_flow.py

## Related concepts and guides

- [Scale and level lifecycle](../concepts/ckks/scale-and-level-lifecycle.md)
- [Value model and identity](../concepts/ckks/value-model-and-identity.md)
- [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
- [Choose a preset and chain depth](../how-to/choose-preset-and-depth.md)
