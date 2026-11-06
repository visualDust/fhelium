# Provision the minimum required keyset

Build a shared inventory from the operations your evaluators will execute. Reuse compatible key values across consumers, generate missing material during setup, and prepare each consumer with the keys selected for its actual input state and device.

## Prerequisites

Have the evaluator schedule, CKKS parameters and source/destination secret relations for each workload. For a transformed Program, derive requirements after the passes that introduce rotations and relinearization. Captured Eager key operands already identify supplied Tensor data.

## 1. Identify the required operations

| Operation | Key requirement |
| --- | --- |
| Public-key encryption | `PublicKey` |
| Decryption | `SecretKey` |
| Relinearize a three-component product | `RelinearizationKey` |
| Rotation by normalized step $s$ | `RotationKey(s)` |
| Conjugation | `ConjugationKey` |
| Source-to-destination secret dependency change | `KeySwitchKey` |

An evaluator that receives ciphertext inputs may need only evaluation keys. Keep public and secret keys with the callers that encrypt, decrypt or generate material.

For a transformed Program, inspect the represented requirements:

```python
from fhelium import ir

requirements = ir.analyze_evaluation_key_requirements(compiled.program)
print(sorted(requirements.rotation_steps))
print(requirements.requires_relinearization)
```

Here `compiled` is the Compilation after the chosen scheduling passes. The analysis reports roles and rotation steps; provisioning also needs the consuming Q/P bases, key layouts and devices. Supply the resulting material through [Compilation preparation](persist-compiled-program.md#3-supply-missing-data-or-replace-an-assignment).

For a custom stored key layout, set `workspace["key_layouts"] = {symbol: {"prime_ids": rows, "digit_prime_ids": digits}}`. Supplied keys, whether generated or loaded, must match the layout selected for numerical lowering. See [Evaluation-key layout declarations](../developer/materials-and-preparation.md#evaluation-key-layout-declarations) for default selection and partial declarations.

## 2. Choose rotation coverage

Derive rotation steps from the actual packing and evaluator schedule, normalize them with the Engine's slot convention, and verify the directions against a cleartext reference. Padding, negative rotations and baby-step/giant-step decomposition can change the required steps even when matrix dimensions agree.

| Strategy | Stored material | Online work |
| --- | --- | --- |
| Direct keys | One compatible key for every required step | One keyed operation per requested rotation |
| Signed-power basis | A smaller set of signed-power keys | Composed rotations for other steps |
| Signed powers with direct baby steps | A basis plus the selected BSGS baby steps | Composed giants and directly hoistable babies |

Measure end-to-end latency, retained key bytes and temporary execution memory together. Shared baby decomposition requires a compatible group of direct keys. The presence of every step somewhere in an inventory does not guarantee that those entries form such a group.

## 3. Populate a shared inventory

`EvaluationKeySet` accepts ordinary rotation, relinearization and conjugation key values:

```python
from fhelium.values import EvaluationKeySet

rotation_key = engine.create_rotation_key(8, secret_key)
relinearization_key = engine.create_relinearization_key(secret_key)
conjugation_key = engine.create_conjugation_key(secret_key)
pool = EvaluationKeySet(
    [rotation_key, relinearization_key, conjugation_key]
)
pool.add(rotation_key)
pool.update([relinearization_key, conjugation_key])
```

The inventory retains references and deduplicates by object identity. The repeated additions above keep one entry for each object. Distinct keys for the same step can coexist, including alternative digit layouts or device copies. Keep all members within the secret relation and parameter context intended by the consumers; structural selection does not establish those cryptographic relations.

For a configured `FullSlotBootstrap`, use its consumer-wide provisioning helper:

```python
bootstrap.create_evaluation_keys(
    secret_key,
    into=pool,
    rotation_key_strategy="all_required",
    device=secret_key.device,
)
```

The helper returns `pool`, reuses compatible existing groups, and creates missing rotation, relinearization and conjugation material through Engine factories. Its layout choice accounts for additional storage across the composition's consumers, including shared steps and reuse across stages. When requested direct babies need new material, the helper completes a common layout for their hoisted execution. A complete mixed inventory is reused without generation and executes those babies independently; setting `hoist_baby_rotations=False` also permits independent layouts when filling gaps. Repeating provisioning can reuse material already present. Encapsulation's low and return hints are supplied through the raising component and retain their separately chosen secret relations and auxiliary bases.

Inspect `bootstrap.key_steps(strategy)` and `bootstrap.evaluation_key_requirements(strategy)` to see the requested rotation strategy and capability roles. These queries describe the standalone basis and contain no generated key data. The default `all_required` strategy requires direct keys for its logical rotations, and the baby-step strategy requires direct baby keys. Signed-power requests can reuse more direct paths already present in the pool. Generate or load the needed keys before `prepare`, capture or linking.

## 4. Select compact layouts when they reduce total storage

A primitive factory can create a key for a chosen active Q suffix:

```python
compact_key = engine.create_rotation_key(
    8, secret_key, source_depth=3, device=secret_key.device
)
pool.add(compact_key)
print(compact_key.prime_ids, compact_key.digit_prime_ids)
```

At `source_depth=3`, the fresh key stores the active Q suffix and P and uses a greedy digit partition constrained by the auxiliary product. With the default P basis, `source_depth=0` uses the configured depth-zero decomposition. `digit_prime_ids=` permits a caller-selected partition in either scope. An existing broader key instead keeps its stored digit planes when used on a smaller basis: execution intersects each stored support with active Q and uses the original plane indices.

The key-switch, rotation, relinearization and conjugation factories also accept `p_prime_ids=`. The selected IDs must be distinct catalog primes outside the key's source Q suffix. For example, a key generated at depth 3 can use the already dropped Q primes together with the configured P primes:

```python
config = engine.config
auxiliary_ids = (
    *range(config.q_row_start(3)),
    *range(config.num_q_primes, config.total_num_primes),
)
alternative = engine.create_rotation_key(
    8,
    secret_key,
    source_depth=3,
    p_prime_ids=auxiliary_ids,
    device=secret_key.device,
)
pool.add(alternative)
```

This generates new key material with Q rows followed by the selected P order. Supplying P also makes the default digit partition use that auxiliary product; omitting `p_prime_ids` retains the configured P basis. A larger P can reduce the digit count while increasing quotient and transform work. Compare the complete consuming operation and shared inventory before choosing an alternative.

The pool can retain both keys for step 8. Selection reads the actual digit supports, QP rows, ring dimension, dtype and device required by the consumer. Depth supplies the active Q basis in that consumer's context; it is not a pool lookup identity. Members of a hoisted group share the stored QP row order and digit supports required by its common decomposition.

A compact key can reduce one operation's key payload and digit work, but a separate compact key at every depth can increase the total retained inventory. Compare the additional bytes needed for the entire workload before generating alternatives. A shared broader key can be preferable when several consumers reuse it.

## 5. Bind consumers before repeated execution

Prepare a bootstrap from the populated inventory:

```python
prepared = bootstrap.prepare(ciphertext, evaluation_keys=pool)
refreshed = prepared(ciphertext)
```

Material discovery records the selected keys in ordinary dictionaries beside encoded diagonals and constants. The prepared callable retains those selected references. Adding a new candidate to `pool`, removing one member or clearing the pool leaves this callable's bindings unchanged. Prepare again when the consumer should use a different selection. In-place writes to a bound key Tensor remain visible and must preserve the prepared layout and synchronization requirements.

To release only inventory references:

```python
pool.remove(compact_key)
pool.clear()
```

Existing prepared users can still retain those keys. Close prepared callables after their Graph users and release other owners before expecting their live allocations to disappear.

Primitive Eager operations remain available when the key choice is already known. Pass a key directly or install it with `engine.set_relinearization_key(...)` or `engine.set_rotation_key(...)`. Construct `Engine(config, allow_automatic_key_generation=False)` when missing material must be resolved by application provisioning. Compile capture and linking consume supplied numerical data.

## 6. Assign devices, ranks and residency

For every worker, record its local operations, permitted secret material, required key values, host storage and CUDA residency windows. Typical allocations include:

- shared evaluation material for independent data-parallel workers;
- owned rotation offsets plus common keys for offset-parallel workers;
- a client or root owner retaining secrets while evaluator workers receive evaluation keys;
- reconstruction on an owner rank for limb-parallel complete-row stages.

An inventory can contain device-local alternatives. Place a key with `key.to(device)` or generate it on the target device before preparing its consumer. Engine-managed lazy replicas require `allow_automatic_key_replication=True`; account for the retained source and replicas independently.

Use the existing Residency facilities for pageable host holds, fixed-capacity pinned/CUDA windows, active leases and prefetch. Inventory membership retains key objects, while Residency controls physical materializations. A collective can create temporary peer copies even when only one rank ultimately retains a key, so include distribution peaks in capacity planning.

## 7. Verify specialist key relations

Generate a generic key switch from the intended source and destination secrets, record that relation in application metadata, and verify under the destination secret:

```python
source_secret = engine.create_secret_key()
destination_secret = engine.create_secret_key()
source_public = engine.create_public_key(source_secret)
source = engine.encrypt_message(message, source_public)
source_to_destination = engine.create_key_switch_key(
    source_secret, destination_secret
)
switched = engine.switch_key(source, source_to_destination)
decoded = engine.decrypt_message(switched, destination_secret)
```

Reversing the two secrets creates a different relation. Tensor layout does not establish its direction or parameter provenance.

Conjugation uses the corresponding automorphism key:

```python
conjugation_key = engine.create_conjugation_key(source_secret)
conjugated = engine.conjugate(source, conjugation_key)
```

Compare the decoded result with `message.conj()` under the selected packing convention. Provision these values before evaluation and retain their source relation with the application's inventory records.

## 8. Persist values and report key cost

Save individual key values through the existing value or artifact persistence APIs. Stored metadata preserves the actual prime rows, digit supports and operation specialization. After loading, reconstruct the application's inventory and restore its parameter and secret-relation associations. Saving a key leaves its live placement unchanged.

Before writing sensitive material, verify the destination's access controls, encryption and deletion policy. Typed secret serialization requires an opt-in. Sensitivity labels classify stored data; the storage system supplies confidentiality and access control. Keep key payloads out of logs and benchmark metadata.

Report the direct-key count, rotation decomposition strategy, actual retained bytes, aggregate and maximum per-rank storage, host/CUDA placement, temporary peaks and whether generation or movement is timed. Several layouts for one step contribute separate payloads, while shared references to one Tensor do not create another allocation.

## Verify the outcome

Run the evaluator with its provisioned material and verify the required numerical accuracy under the intended secret relation. Confirm that preparation selects compatible groups on the consuming device and that repeated execution uses those bound keys. Keep key creation, loading and movement outside a steady-state timing interval unless the reported workload includes them.

## Related documentation

- [Key lifecycle](../concepts/ckks/key-lifecycle.md)
- [Key-materials tutorial](../tutorial/key-materials.md)
- [Residency lifetimes](../concepts/execution/residency-lifetimes.md)
- [Choose a multi-GPU partition](choose-multi-gpu-partition.md)
