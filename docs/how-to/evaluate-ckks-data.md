# Encode, encrypt, and evaluate CKKS data

CKKS evaluation computes approximate arithmetic on packed encrypted slots. This procedure builds an Eager evaluator for a sum and a ciphertext product, supplies its keys, and checks the decoded result and value state.

## Prerequisites

Install a CPU-capable FHElium build and PyTorch. Choose a configuration with at least one available rescale transition using [Choose a preset and chain depth](choose-preset-and-depth.md). The example uses all physical slots, small real messages, and CPU placement so that packing and device transfers do not obscure the arithmetic.

## 1. Create data and keys on the execution device

```python
import torch
import fhelium as fh
from fhelium.eager import Engine

config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7_int64)
engine = Engine(config)
device = "cpu"
secret = engine.create_secret_key(device=device)
public = engine.create_public_key(secret, device=device)
relin = engine.create_relinearization_key(secret, device=device)
engine.set_relinearization_key(relin)

x = torch.linspace(-0.05, 0.05, config.num_slots, dtype=torch.float64, device=device)
y = torch.full_like(x, 0.03)
px = engine.encode(x, depth=0, device=device)
cx = engine.encrypt(px, public)
cy = engine.encrypt_message(y, public, depth=0, device=device)
```

Encoding represents a clear message as a scaled polynomial; encryption introduces the key relation and randomized ciphertext. Keep the secret key with the decrypting party. An evaluator receiving ciphertexts and evaluation keys need not receive it. Factories accept placement controls; subsequent arithmetic follows its Tensor operands.

## 2. Evaluate addition and multiplication

```python
summed = engine.add(cx, cy)
a = engine.coefficient_domain_to_ntt_domain(cx)
b = engine.coefficient_domain_to_ntt_domain(cy)
triplet = engine.multiply(a, b)
product = engine.rescale_to_next_depth(engine.relinearize(triplet))
```

Addition requires compatible value states and preserves depth and scale. Multiplication in NTT representation produces three ciphertext components and scale $\Delta_x\Delta_y$ at the input depth. Relinearization reduces three components to two using the supplied key. Rescaling advances one depth and divides the actual scale by the entire dropped Q group's product:

$$
\Delta_{\mathrm{out}}=\frac{\Delta_x\Delta_y}{\prod_{q\in G_d}q}.
$$

The group may contain multiple primes. Use the configuration's divisor rather than interpreting a depth transition as one prime-width subtraction.

## 3. Prepare public factors for repeated products

For a ciphertext-plaintext product, encode at the ciphertext's depth and convert the factor to operation-ready form:

```python
factor = engine.prepare_plaintext_for_multiplication(
    engine.encode(y, depth=cx.depth, device=device)
)
weighted = engine.multiply_plaintext(a, factor)
```

The product scale is `a.scale * factor.scale`; multiplication itself does not consume a depth. Choose when to rescale according to the remaining arithmetic and available magnitude headroom. Reuse the prepared factor only for compatible depth, prime rows, representation, and device. `examples/04_eager_scale_management.py` demonstrates allocating two factor scales around one rescale; `examples/05_eager_ntt_reuse.py` demonstrates retaining NTT intermediates.

## 4. Observe approximation error and output state

```python
for value, expected in ((summed, x + y), (product, x * y)):
    decoded = engine.decrypt_message(value, secret)
    print("max absolute error:", float((decoded - expected).abs().max()))
    print("depth:", value.depth, "scale:", value.scale)
    print("prime rows:", value.prime_ids, "shape:", tuple(value.data.shape))
```

The sum remains at depth zero; the rescaled product reaches depth one. Establish an application error criterion before accepting results and include imaginary residuals when real output is expected. Ciphertext residue equality is useful for comparing an unchanged deterministic arithmetic schedule, but changed rescale placement or algebraic schedules should be checked against decoded clear-message semantics and their error model.

## Continue from this evaluator

Use [Compile a callable](compile-callable.md) to specialize the same function, or [Build and transform a Program](build-program-pipeline.md) to describe and transform the calculation directly. Both paths share the same caller-provided data and key requirements. See [Value-state diagnosis](diagnose-value-state-mismatch.md) when an operand is incompatible and [CKKS multiplication, key switching, and rescale](../developer/multiplication-keyswitch-rescale.md) for the implemented transitions.
