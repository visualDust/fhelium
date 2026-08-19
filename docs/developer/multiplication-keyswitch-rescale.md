# Multiplication, key switching, and rescale

These paths combine many arithmetic stages and exact-state transitions. They
are frequent correctness and performance hotspots because they depend on
active rows, hybrid digits, Q/QP conversion, NTT/Montgomery representation, and
large evaluation keys.

## Implementation stack

```mermaid
graph TB
    API[CkksEngine public operation]
    STATE[Python state and key validation]
    KS[HybridKeySwitcher]
    RS[CkksRescaler]
    RNS[RnsRuntime + NTT backend]
    WRAP[Generated ckks_ops / rns_ops / ntt_ops wrappers]
    DISP[torch.ops + PyTorch dispatcher]
    CPU[C++ CPU primitives<br/>ATen + parallel_for]
    CUDA[CUDA primitives<br/>current stream kernels]

    API --> STATE
    STATE --> KS
    STATE --> RS
    STATE --> RNS
    KS --> RNS
    RS --> RNS
    KS --> WRAP
    RS --> WRAP
    RNS --> WRAP --> DISP
    DISP --> CPU
    DISP --> CUDA
```

The engine owns public state transitions and output construction.
`HybridKeySwitcher` and `CkksRescaler` compose tensor stages in Python.
Pointwise RNS arithmetic and NTT transitions enter the `fhelium_rns_ops` and
`fhelium_ntt_ops` namespaces; Galois, key-switch accumulation, ModDown, and
rescale kernels enter `fhelium_ckks_ops`. CPU and CUDA registrations implement
the same schemas where the primitive is shared across devices.

## Plaintext multiplication

Conceptual data flow:

```mermaid
flowchart LR
    CT0[c0 NTT / Montgomery]
    CT1[c1 NTT / Montgomery]
    PT[operation-ready plaintext<br/>NTT / Montgomery]
    M0[pointwise Montgomery multiply]
    M1[pointwise Montgomery multiply]
    OUT[2-component NTT / Montgomery ciphertext]
    CT0 --> M0 --> OUT
    CT1 --> M1 --> OUT
    PT --> M0
    PT --> M1
```

The output scale is multiplied, but level is unchanged until a rescale. `multiply_plaintext` does not perform hidden forward or inverse NTTs;
the caller or JIT places transitions around a multiplication region. Compatible
products may be added in NTT form and converted to coefficient-domain standard
residues once before rescale. Prepared plaintext reuse must match level, scale,
basis, prime IDs, domain, and residue representation exactly.

## Ciphertext multiplication

For two components $(c_0,c_1)$ and $(d_0,d_1)$:

$$
(e_0,e_1,e_2)=
(c_0d_0,\;c_0d_1+c_1d_0,\;c_1d_1).
$$

The engine requires compatible two-component NTT/Montgomery inputs and returns
a three-component NTT/Montgomery ciphertext. Relinearization is a later,
key-switch stage. Fresh direct-CKKS operands enter at scale $\Delta$;
the triplet carries scale $\Delta^2$, and rescale follows
relinearization.

## Relinearization

```mermaid
graph LR
    T[triplet e0 e1 e2]
    E2[e2]
    K[RelinearizationKey]
    KS[hybrid key switch]
    C0[correction 0]
    C1[correction 1]
    OUT[two-component ciphertext]
    T --> E2 --> KS
    K --> KS
    KS --> C0 --> OUT
    KS --> C1 --> OUT
    T --> OUT
```

The original first two components are combined with corrections that replace
the $s^2$ dependency represented by `e2`.

## Hybrid key-switch pipeline

```mermaid
flowchart TB
    S[source component in active Q]
    D[1 hybrid digit partition]
    MR[2 mixed-radix decomposition]
    MU[3 ModUp each digit from Q to QP]
    N[4 forward NTT / Montgomery]
    KP[5 multiply matching key digit]
    ACC[6 accumulate two QP outputs]
    IN[7 inverse NTT]
    MD[8 ModDown / divide by P]
    COR[9 combine corrections]
    S --> D --> MR --> MU --> N --> KP --> ACC --> IN --> MD --> COR
```

Each stage has distinct row, basis, and representation requirements. Fusing stages may
be useful, but a fused operator must preserve the same observable state and
residue-range assumptions.

## Hybrid digits across levels

Scale Q primes are partitioned into composite digits, while the base Q prime is
a final singleton digit. At later levels, an active digit can become shorter or
disappear.

```mermaid
graph LR
    subgraph L0[level 0]
      D0[q0 q1 q2 q3]
      D1[q4 q5 q6 q7]
      DB[q_base]
    end
    subgraph L2[later level]
      E0[q2 q3]
      E1[q4 q5 q6 q7]
      EB[q_base]
    end
    D0 -->|same key_digit_index| E0
    D1 --> E1
    DB --> EB
```

`RnsDigitSpec` keeps both the active digit index and stable level-zero
`key_digit_index` used to select the correct evaluation-key axis. A local digit
index is not necessarily the key tensor index.

## Rotation and hoisting

Rotation applies a Galois automorphism and then key-switches the transformed
secret dependency. For several steps on the same input component, preparation
can be shared:

```mermaid
flowchart TB
    C1[input c1]
    PREP[decompose + ModUp + NTT once]
    R1[step 1 automorphism + key products + ModDown]
    R2[step 2 automorphism + key products + ModDown]
    RN[step n automorphism + key products + ModDown]
    C1 --> PREP
    PREP --> R1
    PREP --> R2
    PREP --> RN
```

Step-specific work and outputs remain. Hoist chunking must account for live
prepared digits, accumulators, rotated outputs, and exact key residency.

## Rescale

For leading active prime $q_l$:

$$
c'\approx\operatorname{round}(c/q_l)\pmod{Q_{l+1}}.
$$

```mermaid
flowchart LR
    IN[coefficient residues in Q_l]
    DROP[select dropped leading row]
    ROUND[nearest/truncate correction]
    INV[multiply inverse of q_l modulo remaining primes]
    OUT[remaining rows in Q_l+1]
    IN --> DROP --> ROUND --> INV --> OUT
```

The implementation must select constants using canonical prime identity, not
an ambiguous compact row position. Output metadata must increase level, remove
the dropped prime ID, reduce row count, and update scale.

## Correctness hazards

High-risk errors include:

- using local row count to infer the wrong canonical modulus;
- selecting the wrong key digit after earlier primes are dropped;
- mixing Q and QP parameter rows;
- applying NTT tables for another active slice/device;
- treating singleton digits as a normal full group;
- violating lazy/canonical residue assumptions across fused operators;
- copying or overwriting staged data before another stream/device is done;
- reconstructing correct tensor values with wrong public metadata.

## Validation matrix

For a change in these paths, cover:

```text
fresh single operation
chained operation across several levels
level 0 / middle / last legal level
single-row digit and shortened digit
Q / QP
2 / 3 components
functional / in-place
multiple NTT backends
logN = 14 smoke and target logN = 15 or logN = 16
source build and installed wheel
world size 1 and 2+ if transport/partition is involved
```

Decrypt after every legal materialization step to localize the first
incorrect stage.

## Continue

- [Scale and level lifecycle](../concepts/ckks/scale-and-level-lifecycle.md)
- [RNS and NTT architecture](rns-and-ntt.md)
- [Native operator workflow](native-operator-workflow.md)
- [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
- [Rotation-hoisting tutorial](../tutorial/rotation-hoisting.md)

## Source map

| Path | Source owner |
| --- | --- |
| Public multiplication, relinearization, rotation, and plaintext calls | `fhelium/engine/ckks_engine.py` |
| Hybrid decomposition, ModUp, prepared rotations, key products, ModDown | `fhelium/engine/hybrid_keyswitch.py` |
| Direct fused key-digit consumption | `fhelium/engine/direct_keyswitch_consumer.py` |
| Rescale state validation and quotient construction | `fhelium/engine/ckks_rescale.py` |
| RNS/NTT arithmetic and active parameters | `fhelium/engine/rns/runtime.py`, `fhelium/engine/ntt/` |
| CKKS-local operator schemas | `csrc/ops/ckks/ckks.cpp` |
| CPU CKKS tensor primitives | `csrc/ops/ckks/cpu/ckks_cpu.cpp` |
| CUDA Galois, key-switch, plaintext, and rescale kernels | `csrc/ops/ckks/cuda/` |
