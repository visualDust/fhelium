# Key material lifecycle

**Example source:** [`examples/02_eager_key_materials.py`](https://github.com/VisualDust/fhelium/blob/main/examples/02_eager_key_materials.py)

Example 02 creates keys from one secret, installs selected evaluator capabilities, and inspects their layouts. File persistence and artifact generations are covered by [09](value-memory-and-persistence.md) and [10](artifact-store.md).

```bash
python examples/02_eager_key_materials.py --preset slots8192-scale40-depth7-int64 --rotations=-4,-1,1,2,4
```

## 1. Create typed keys

```python
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relinearization_key = engine.create_relinearization_key(secret_key)
rotation_key = engine.create_rotation_key(1, secret_key)
```

The roles are distinct:

| Key | Primary use | Typical dense axes |
| --- | --- | --- |
| [`SecretKey`](../api/fhelium/values/keys.md#secretkey) | decryption and generation of derived keys | `[limb, coefficient]` |
| [`PublicKey`](../api/fhelium/values/keys.md#publickey) | public-key encryption | `[key component, limb, coefficient]` |
| [`RelinearizationKey`](../api/fhelium/values/keys.md#relinearizationkey) | three-component to two-component conversion | `[digit, key component, limb, coefficient]` |
| [`RotationKey`](../api/fhelium/values/keys.md#rotationkey) | one slot automorphism/key switch | `[digit, key component, limb, coefficient]` |

The example uses factory calls so key creation is visible. Code that must forbid implicit key creation can construct the engine with `allow_automatic_key_generation=False` and install only the keys it owns.

Factory methods select placement directly:

```python
secret_key = engine.create_secret_key(device="cuda:0")
public_key = engine.create_public_key(secret_key)
rotation_key = engine.create_rotation_key(1, secret_key)
```

Derived-key factories infer placement from their secret-key input. Supplying a different `device` authorizes a copy of that same secret relation; the Engine does not generate an unrelated secret key on the destination.

Operation use does not imply permission to copy a key. By default, a public, secret, or evaluation key must already be on the Tensor operation's device. Create a copy with `key.to(device)` and pass or install it when placement is caller-managed. `Engine(..., allow_automatic_key_replication=True)` instead permits the Engine to create and cache device replicas when an installed key is first needed there. The source copy remains allocated, and releasing Python references does not guarantee device-memory erasure.

## 2. Treat rotation step as stored specialization

```python
key = engine.rotation_keys[rotation_step]
assert key.rotation_step == rotation_step
```

`RotationKeySet` validates normalized signed steps when constructing or updating the mapping. A key for step `+1` must not be silently reused as a key for another step, even if both tensors happen to have the same shape.

This distinction matters in distributed and multi-user systems: the tensor layout alone is not sufficient stored key state, and neither the layout nor the runtime key object proves an external ciphertext/key relation. Preserve that relation in the application that provisions the key.

## 3. Inspect key memory

```python
shape = tuple(relinearization_key.data.shape)
byte_count = relinearization_key.data.nbytes
```

Evaluation keys are usually much larger than ciphertexts because they contain multiple decomposition digits and both Q and P basis rows. Capacity planning should use the actual `nbytes` for the active parameter set instead of a count of Python objects.

## 4. Install selected evaluator capabilities

```python
engine.set_secret_key(secret_key)
engine.set_public_key(public_key)
engine.set_relinearization_key(relinearization_key)
engine.set_rotation_key(rotation_key)
```

Creation returns a value; installation adds that value to the Engine's inventory. Key placement remains separate. A stored rotation step describes specialization, not the participant or secret that produced the key. Cryptographic correspondence remains the application's responsibility.

::: details Source
<<< @/../examples/02_eager_key_materials.py
:::

## Related concepts and guides

- [Key lifecycle](../concepts/ckks/key-lifecycle.md)
- [Serialization and artifacts](../concepts/execution/serialization-and-artifacts.md)
- [Provision the minimum required keyset](../how-to/provision-keyset.md)
