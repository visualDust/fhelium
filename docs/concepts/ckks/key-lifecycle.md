# Key lifecycle

Keys are typed values carrying Tensor payloads, arithmetic metadata and operation specialization. An application creates or loads these values, retains the keys needed by its consumers, places them on execution devices and supplies them to evaluation. Their lifetime follows the references held by the application, prepared computations and executables.

## Key families and secret relations

| Key | Mathematical role | Stored specialization |
| --- | --- | --- |
| `SecretKey` | Decryption and derivation of key material | Secret polynomial and arithmetic state |
| `PublicKey` | Public-key encryption | Q rows and arithmetic state |
| `RelinearizationKey` | Switch the multiplication dependency from $s^2$ to $s$ | Hybrid digit supports and QP rows |
| `RotationKey` | Switch an automorphism-transformed secret back to $s$ | Normalized signed rotation step, hybrid digit supports and QP rows |
| `ConjugationKey` | Switch the conjugation-transformed secret back to $s$ | Hybrid digit supports and QP rows |
| `KeySwitchKey` | Switch a source secret dependency to a destination dependency | Hybrid digit supports and QP rows |

A generic key switch maps the ciphertext phase

$$
c_0+c_1s_{\mathrm{src}}
\longmapsto
c'_0+c'_1s_{\mathrm{dst}},
$$

up to key-switch error. The application maintains these source and destination relations and the mathematical parameter context in which the keys were generated. Matching Tensor layouts, prime indices or rotation steps establishes structural suitability for an operation; it cannot establish the secret relation. [Terminology and mathematical model](../terminology-and-mathematical-model.md#ciphertext-phase-and-key-relations) defines the corresponding phase equations.

## Shared evaluation-key inventory

`EvaluationKeySet` is an application-owned inventory of ordinary `RotationKey`, `RelinearizationKey` and `ConjugationKey` objects. Construction accepts an iterable of these values. Several consumers can share one inventory, and one rotation step can have several stored layouts or device copies. Public and secret keys remain with the encryption, decryption and key-generation owners.

`add` and `update` retain supplied object references and deduplicate by object identity. Distinct key objects remain distinct entries even when their operation role or payload shape agrees. `remove` and `clear` release only the inventory's references. A prepared computation or another application reference can keep the same key and its Tensor storage alive afterward.

Inventory membership describes available material. A consumer selects from that material using its required operation, normalized rotation step, active Q support, P basis, physical key rows, ring dimension, residue dtype and device. Depth determines the consumer's active Q rows in its context; the inventory stores the key's actual coverage and layout. A key can therefore serve several depths without being entered once per depth.

A `RotationKeySet` represents a selected step-to-key mapping. It is useful for a group of operations that has already chosen one key per step. The shared `EvaluationKeySet` can retain several such alternatives until the consumer chooses a suitable group.

With complete execution facts, rotation selection prefers fewer primitive key-switch applications, then fewer active digit applications, then fewer bytes across the distinct selected key payloads. Insertion order breaks ties. Single-role selection prefers fewer active digits, then smaller stored payloads and insertion order. These rules give deterministic reference selection; they do not predict measured latency. With incomplete facts, selection requires a unique candidate consistent with the known requirements.

## Coverage, digit supports and group compatibility

A key-switch Tensor has axes `[digit, key component, limb, coefficient]`. `prime_ids` gives its ordered physical QP rows, while `digit_prime_ids` gives the mathematical Q support associated with each stored digit plane. Both describe the existing payload.

The key's stored basis consists of its source Q rows followed by a disjoint auxiliary P basis. P is part of the key layout and can differ between inventory members. A prime that has left the source Q suffix can belong to that key's P basis. The physical row order follows the stored Q and P sequences, so the concatenated IDs need not be globally increasing. Changing the auxiliary basis requires new key material. Restriction to a smaller Q basis retains the original P rows and digit planes.

When an existing key serves a smaller active Q basis, execution intersects each stored digit support with that basis and retains the original plane index for every nonempty intersection. Creating a fresh key for a smaller source basis can instead choose a new partition and store fewer digit planes and rows. These choices produce different key material and different key-switch noise.

Hoisted rotations share the decomposition, basis extension and NTT of one source component. Their selected keys must agree on the effective digit partition and the physical layout needed by that shared preparation. Selecting individually usable keys for each step is insufficient when the resulting group cannot share those operands. The same requirement applies to the final key switches accumulated into one shared quotient. Independent rotations and the earlier switches of composed paths can consume different stored layouts.

A normalized signed step lies in $[-S/2,S/2)$ for $S$ slots. Direct rotation keys reduce sequential key switches, while a smaller signed-power basis composes the same requested rotation from several keyed operations. The schedule determines both the required steps and which keys must form compatible groups. [Composable CKKS bootstrapping](composable-bootstrapping.md) gives one consumer of this model.

## Provisioning and preparation

Provisioning creates or loads the material required by a consumer schedule. Engine factories create individual key values; a consumer's provisioning helper can reuse compatible inventory members and generate missing groups. Its storage decision must account for all of that consumer's uses, because one larger key can be shared across stages while several smaller keys require separate allocations.

```mermaid
flowchart LR
    SOURCE[Create or load typed keys]
    POOL[Application-owned shared inventory]
    SELECT[Consumer material discovery]
    BOUND[Selected key references and numerical materials]
    EXEC[Prepared Eager or Compile execution]
    SOURCE --> POOL --> SELECT --> BOUND --> EXEC
```

Preparation selects existing keys and records those choices in ordinary material dictionaries. A prepared computation retains the selected key references and its numerical materials. Subsequent additions or removals in the inventory do not change those bindings. In-place changes to a selected Tensor remain visible to execution; changing its layout or placement requires new preparation for the resulting execution contract.

A Compile Program represents key payloads as Tensor operands, including operands referenced by material symbols. `Compilation.material_bindings` assigns live Tensors to those symbols. Capture preserves supplied key data, and optional material preparation can select missing operands from an existing inventory. Linking retains the assigned Tensors. Key generation is a provisioning action performed before these stages. [Neutral IR programs](../neutral-ir-programs.md) describes material operands and [the Compile lifecycle](../open-compiler-stack.md) describes preparation and linking.

Eager primitive operations also accept key values directly or use installed Engine keys. Creation returns a value; installation retains it for the corresponding Engine operation. These primitive interfaces remain useful when the application already knows the required key.

## Placement, persistence and lifetime

The application decides which process owns, replicates, broadcasts, stages or evicts each key. An inventory may contain keys on several devices; selection uses the consuming device. Individual key copies can be created with `key.to(device)`. Primitive Eager execution rejects a key on another device by default, while `allow_automatic_key_replication=True` enables Engine-managed replicas. Copy lifetime and device trust remain application responsibilities.

Residency management governs physical materializations and their active lifetimes. The shared inventory governs references to available key values. Clearing an inventory neither evicts a key retained by another owner nor erases device memory. [Residency lifetimes](../execution/residency-lifetimes.md) defines the corresponding storage mechanisms.

Key persistence saves the actual payload, ordered prime rows, digit supports and operation specialization. Loading restores those facts; the application restores the associated parameter context and secret relations before use. Typed secret-key serialization requires an opt-in, and storage encryption and access control belong to the deployment. [Serialization and artifacts](../execution/serialization-and-artifacts.md) defines the value format and artifact lifecycle.

Steady-state evaluator measurements exclude key creation, loading, movement and material preparation unless their timing scope includes those actions. Report retained key bytes together with temporary execution storage and any device replicas; counting rotation steps alone does not describe the memory cost.

## Related concepts

- [Distributed SPMD model](../distributed/spmd-model.md)
- [Ownership and runtime responsibilities](../architecture/ownership-and-responsibilities.md)
