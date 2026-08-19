# Key material lifecycle

**Example source:** [`examples/02_key_materials.py`](https://github.com/VisualDust/fhelium/blob/main/examples/02_key_materials.py)

This example creates the major CKKS key types, reports their dense layouts and
sizes, and optionally persists selected material. The tutorial distinguishes
stored key state and specialization from application-maintained cryptographic
relations, ownership, and restoration.

## What you will learn

- which keys are needed for encryption, decryption, multiplication, and
  rotation;
- how key shapes expose components, decomposition digits, RNS limbs, and
  coefficients;
- why a [`RotationKey`](../api/fhelium/core/keys.md#rotationkey) is bound to one canonical signed
  step;
- how
  [`ArtifactStore`](../api/fhelium/artifacts/store.md#artifactstore) differs from
  key generation and key placement;
- why secret-key persistence requires an explicit opt-in.

## Run the example

```bash
python examples/02_key_materials.py \
  --preset slots8192-scale40-levels7-int64 \
  --rotations=-4,-1,1,2,4
```

To retain selected artifacts in a local store:

```bash
python examples/02_key_materials.py \
  --preset slots8192-scale40-levels7-int64 \
  --rotations=1,2,4 \
  --store /tmp/fhelium-key-demo
```

The second command persists public, relinearization, and rotation keys. It
does **not** persist the secret key.

## 1. Create exact key types

```python
secret_key = engine.secret_key
public_key = engine.public_key
relinearization_key = engine.relinearization_key

for rotation_step in rotation_steps:
    engine.rotation_key(rotation_step)
```

The roles are distinct:

| Key | Primary use | Typical dense axes |
| --- | --- | --- |
| [`SecretKey`](../api/fhelium/core/keys.md#secretkey) | decryption and generation of derived keys | `[limb, coefficient]` |
| [`PublicKey`](../api/fhelium/core/keys.md#publickey) | public-key encryption | `[key component, limb, coefficient]` |
| [`RelinearizationKey`](../api/fhelium/core/keys.md#relinearizationkey) | three-component to two-component conversion | `[digit, key component, limb, coefficient]` |
| [`RotationKey`](../api/fhelium/core/keys.md#rotationkey) | one slot automorphism/key switch | `[digit, key component, limb, coefficient]` |

Calling the engine properties may lazily create missing key material. Code
that must forbid secret-key creation can construct the engine with
`allow_sk_gen=False` and install only the keys it owns.

## 2. Treat rotation step as stored specialization

```python
key = engine.rotation_keys[rotation_step]
assert key.rotation_step == rotation_step
```

`RotationKeySet` validates canonical signed steps when constructing or updating the mapping. A
key for step `+1` must not be silently reused as a key for another step, even
if both tensors happen to have the same shape.

This distinction matters in distributed and multi-user systems: the tensor
layout alone is not sufficient stored key state, and neither the layout nor
`context_id` proves an external ciphertext/key relation.

## 3. Inspect key memory

```python
shape = tuple(relinearization_key.data.shape)
byte_count = relinearization_key.data.nbytes
```

Evaluation keys are usually much larger than ciphertexts because they contain
multiple decomposition digits and both Q and P basis rows. Capacity planning
should use the actual `nbytes` for the active parameter set instead of a count
of Python objects.

## 4. Persist selected public/evaluation material

```python
store = ArtifactStore(path)
store.put("keys/public", public_key, overwrite=True)
relinearization_ref = store.put(
    "keys/relinearization",
    relinearization_key,
    overwrite=True,
)

rotation_keys = store.collection("keys/rotation")
rotation_keys.put("1", engine.rotation_keys[1], overwrite=True)
```

The store uses a transactional SQLite catalog for logical names and immutable
safetensors objects for exact key payloads. It adds typed current-generation
references, collections, checksums, and local durability policy. Overwriting a
name creates a new artifact ID and makes the previous reference stale; it does
not retain prior key versions. The store does not decide which user owns a key,
where that key should be cached, or when it should move to CUDA.

## 5. Secret-key persistence is deliberately noisy

```python
store.put(
    "keys/secret",
    secret_key,
    allow_secret=True,
    overwrite=True,
)
```

The explicit `allow_secret=True` prevents an accidental generic save path from
writing a secret key. It is not encryption at rest. Production systems still
need an external key-management service (KMS), permissions, encryption, audit
policy, and deletion policy appropriate to their threat model. The store's
payload checksum detects accidental corruption; it does not authenticate data
against an actor who can modify both the catalog and payload.

## 6. Restore the exact type

```python
restored = store.get(relinearization_ref, device=engine.device)
assert type(restored) is fh.RelinearizationKey
torch.testing.assert_close(restored.data, relinearization_key.data)
```

The serialized metadata reconstructs the key type, context, modulus basis, polynomial domain,
prime IDs, and other exact state. A successful tensor load is not enough if
that metadata does not match the current engine or intended operation.

::: warning Do not make every key globally resident
A serving layer should provision exact user/model keysets, enforce a memory
budget, and lease only the keys required by the current operation. The core
engine intentionally does not infer that policy.
:::

::: details Complete runnable source
<<< @/../examples/02_key_materials.py
:::

## Related concepts and guides

- [Key lifecycle](../concepts/ckks/key-lifecycle.md)
- [Serialization and artifacts](../concepts/execution/serialization-and-artifacts.md)
- [Provision the minimum required keyset](../how-to/provision-keyset.md)
