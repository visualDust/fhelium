# Key lifecycle

FHElium treats keys as typed values with controlled creation, placement,
installation, persistence, and use. This separation supports production key
custody, minimal evaluator keysets, distributed ownership, and repeatable
benchmarks.

## Key families

```mermaid
graph TD
    SK[SecretKey]
    SK --> PK[PublicKey]
    SK --> RLK[RelinearizationKey]
    SK --> RK[RotationKey step]
    SK --> CK[ConjugationKey]
    SK --> KSK[KeySwitchKey]

    PK --> ENC[Public-key encryption]
    SK --> DEC[Decryption and key generation]
    RLK --> RELIN[Three components to two]
    RK --> ROT[Slot rotation]
    CK --> CONJ[Complex conjugation]
    KSK --> SWITCH[Generic secret dependency switch]
```

| Key | Main purpose | Stored state or specialization |
| --- | --- | --- |
| `SecretKey` | Decrypt and derive other keys | Dense arithmetic state |
| `PublicKey` | Public-key encryption | Q rows and arithmetic state |
| `RelinearizationKey` | Switch the multiplication $s^2$ component | QP key layout |
| `RotationKey` | Automorphism-specific key switch | Normalized signed step and QP key layout |
| `ConjugationKey` | Complex conjugation | Key-switch state |
| `KeySwitchKey` | Source-to-destination secret-key dependency switch | Digits, QP rows, and arithmetic state |

These objects validate their stored rows, representation state, and
specialization. The application tracks CKKS parameter provenance and symbolic
source/destination lineage beyond the concrete fields carried by each key
type.

## Creation, installation, and use are different actions

```python
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relin_key = engine.create_relinearization_key(secret_key)
rot_key = engine.create_rotation_key(3, secret_key)

engine.set_secret_key(secret_key)
engine.set_public_key(public_key)
engine.set_relinearization_key(relin_key)
engine.set_rotation_key(rot_key)
```

```mermaid
flowchart LR
    C[create or load]
    P[optional persist]
    M[move or broadcast]
    I[install or pass]
    U[use in evaluator]
    C --> P --> M --> I --> U
    C --> I
```

This design allows an evaluator to load externally managed keys without ever
creating a secret key locally. It also makes setup cost and steady-state
execution cost separable.

## Rotation keys bind steps

A `RotationKey` describes one normalized signed slot step. Equivalent modular
steps reduce to the range $[-S/2,S/2)$, but a key for one normalized step cannot
be used for another merely because tensor shapes match.

A `RotationKeySet` maps normalized steps to direct keys. Generate only steps the
packing/evaluator actually needs unless a measured decomposition strategy is
better.

```mermaid
flowchart LR
    DIRECT["direct keys"]
    DIRECT_MEMORY["more key memory"]
    DIRECT_ROTATIONS["fewer sequential rotations"]
    BASIS["small decomposition basis"]
    BASIS_MEMORY["less key memory"]
    BASIS_ROTATIONS["more rotations and key switches"]

    DIRECT --> DIRECT_MEMORY --> DIRECT_ROTATIONS
    BASIS --> BASIS_MEMORY --> BASIS_ROTATIONS
```

The workload selects this key-memory versus online-operation trade-off.

## Lifecycle invariants

- **The consumer plans the keyset.** Derive public, relinearization, rotation,
  conjugation, and generic key-switch requirements from the actual evaluator
  schedule.
- **Setup and use are distinct.** Creating, loading, moving, installing, and
  using a key are separate actions. Distribute the secret key only to workers
  authorized to decrypt or derive key material.
- **Persistence is authorization.** Secret-key serialization requires an
  explicit opt-in. Sensitivity labels provide classification metadata;
  applications supply encryption, a key-management service (KMS),
  access-control lists (ACLs), audit, backup, and deletion policy.
- **Placement is application-owned.** The workload decides which process owns,
  replicates, broadcasts, stages, or evicts each key. Large evaluation keys
  make this both a security and capacity decision. Eager execution rejects a
  key on another device by default. The caller may place a copy with
  `key.to(device)` or opt into Engine-managed lazy replicas with
  `allow_automatic_key_replication=True`. Source-copy lifetime and device trust
  remain application decisions.
- **Steady-state measurements exclude setup unless stated otherwise.** Report
  key creation, load, movement, and materialization separately when the named
  result is evaluator latency.

Use [Provision the minimum required keyset](../../how-to/provision-keyset.md)
for the operational checklist, specialist key-switch example, custody checks,
and reporting procedure. The [Engine API](../../api/fhelium/eager.md) defines the
construction and installation methods.

## Continue

- [Key materials tutorial](../../tutorial/key-materials.md)
- [Distributed SPMD model](../distributed/spmd-model.md)
- [Residency lifetimes](../execution/residency-lifetimes.md)
- [Provision the minimum required keyset](../../how-to/provision-keyset.md)
