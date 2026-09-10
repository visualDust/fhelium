# RNS and NTT architecture

FHElium represents large CKKS moduli as dense residue rows and uses NTT-domain
pointwise arithmetic for polynomial multiplication. Correctness depends on
mapping every compact active row to the correct configured modulus and transform
parameters.

## Chain order

`RnsChain` assigns prime IDs in `[Q | P]` order:

```text
Q prime IDs: 0 ... num_q - 1
P prime IDs: num_q ... num_q + num_p - 1
```

At depth $d$, the active Q basis drops the rows in all preceding depth groups.
Let $s_d$ be the sum of their row counts:

```text
Q_d  = Q[s_d:]
QP_d = Q[s_d:] + P
```

```mermaid
graph TB
    L0[depth 0<br/>G0 G1 G2 ... terminal Q group]
    L1[depth 1<br/>G1 G2 ... terminal Q group]
    L2[depth 2<br/>G2 ... terminal Q group]
    QP[depth-specific Q plus special P rows]
    L0 -->|rescale| L1 -->|rescale| L2
    L1 -->|ModUp| QP
    QP -->|ModDown| L1
```

The dense tensor is compact at the current depth, while `prime_ids` and runtime
layout map each row to its parameter rows.

## Placement-independent layout

`RnsLayout` describes:

- active Q or QP prime IDs;
- depth-specific row counts and parameter slices;
- hybrid decomposition digit rows;
- stable depth-zero key-digit indices;
- component-relative digit row IDs.

It intentionally contains no device assignment or communication policy. An
SPMD workload may partition prime IDs, but that partition does not redefine the
mathematical layout or native application binary interface (ABI).

## RNS and NTT contexts

`RnsContext` binds the residue number system (RNS) layout to device-resident
arithmetic parameters. It provides:

- standard/lazy modular add/subtract;
- Montgomery multiply and conversions;
- row selection and extension/reduction helpers;

`NttContext` composes one `RnsContext` and owns the selected number theoretic
transform (NTT) policy, device tables, executor, and forward/inverse transform
methods. The two contexts share the native parameter tensor because the native
NTT application binary interface reads both RNS parameters and inverse-transform
normalization from that tensor. An NTT execution resource is separate from an
RNS execution resource; it does not inherit or substitute for one.

A caller or lowering supplies the physical basis role and any internal digit-row
mapping. A native operator must not infer a global prime solely from an
ambiguous local row count.

## Implementation path

`fhelium.eager.Engine` creates its Q/P chain and immutable `RnsLayout` once.
When a call first selects a local device, the Engine creates one `RnsContext`
that binds this shared layout to device-resident Montgomery parameters and one
composing `NttContext`. NTT construction selects the transform policy and
builds its device tables and executor. Arithmetic calls then pass tensor views
through generated wrappers to the shared PyTorch operator schemas.

```mermaid
graph TB
    ENG[eager Engine]
    RNSCTX[RnsContext]
    NTTCTX[NttContext]
    LAYOUT[RnsChain + RnsLayout]
    PARAM[RnsParameterStore<br/>aligned parameter views]
    PLAN[NTT policy + host plan]
    TABLE[Typed device tables]
    BACKEND[NttBackend Python adapter]
    WRAP[Generated rns_ops / ntt_ops wrapper]
    DISP[torch.ops + PyTorch dispatcher]
    CPU[C++ CPU implementation]
    CUDA[CUDA implementation]

    ENG --> RNSCTX
    ENG --> NTTCTX
    NTTCTX --> RNSCTX
    RNSCTX --> LAYOUT
    RNSCTX --> PARAM
    NTTCTX --> PLAN --> TABLE --> BACKEND
    PARAM --> WRAP
    BACKEND --> WRAP --> DISP
    DISP -->|CPU| CPU
    DISP -->|CUDA| CUDA
```

For an operand with `k` compact active limbs, `RnsContext` selects a zero-copy
parameter view with exactly `k` columns for the supplied physical basis.
`NttContext` uses that row mapping while slicing its transform tables before
invoking an operator. The registered C++ implementation validates tensor axes
and device; it does not receive a Python depth number or look up an engine.

## NTT backend protocol

```mermaid
graph TD
    P[NttBackend protocol]
    P --> I[Indexed radix-2]
    P --> C[Compact grouped radix-2]
    P --> R[Compact power-of-two radix]
    I --> I1[CPU production and cross-device baseline]
    I --> I2[stored indices and twiddles]
    C --> C1[group4 + smem8]
    C --> C2[group8 + smem8]
    C --> C3[group16 + smem8]
    R --> F[strict fixed radix]
    F --> F1[radix-4 / radix-8 / radix-16]
```

The current policy names are defined in
`fhelium/config/ntt.py`; read that file or the current API/CLI
instead of copying names from an old report.

Every name describes a complete policy. `group8` means three fused radix-2
stages, and `smem8` means eight radix-2 stages execute inside a shared-memory
tile. FHElium does not infer either property from a string suffix: one immutable
policy variant supplies only the factors meaningful to its algorithm family.
The policy registry is a discriminated union of indexed radix-2 execution,
compact grouped radix-2, and strict fixed-radix variants; it is deliberately
not one optional-field object covering every family.

### Indexed plans

The sole indexed policy, `radix2_indexed`, stores twiddle/index
tables. It is the CPU production backend and the cross-device validation
baseline for compact CUDA policies. CPU executes every stage for one
batch/limb row inside one native parallel region; CUDA launches one radix-2
stage at a time. Expanded twiddles store only the nontrivial odd lane.

### Compact plans

Compact policies retain smaller per-prime transform data and derive indices in
CUDA. They are CUDA backends. The selectable policies are
`radix2_compact_group4_smem8`, `radix2_compact_group8_smem8`, and
`radix2_compact_group16_smem8`; the group-8 policy is the CUDA default. CPU engines instead select
`radix2_indexed`. Their eight shared-memory stages are listed in
both the policy name and native ABI.

Plan objects are temporary construction values. `NttContext` retains only a
typed `IndexedRadix2Tables`, `CompactRadix2Tables`, or
`CompactPowerOfTwoRadixTables` device package, never an optional-field superset
or a second host-resident copy of the plan. Indexed tables can reside on CPU or
CUDA; compact table packages currently reside on CUDA.

## Grouped radix-2 stages

A grouped backend combines several radix-2 butterfly stages in one launch:

```mermaid
flowchart LR
    subgraph Separate
      A1[stage] --> A2[global store/load] --> A3[stage] --> A4[global store/load]
    end
    subgraph Grouped
      B1[load local tuple] --> B2[multiple butterfly stages] --> B3[store]
    end
```

This can reduce launches and global-memory round trips, but may increase:

- register pressure;
- shared-memory use;
- index arithmetic;
- occupancy loss;
- sensitivity to active row count and batch shape.

`group16` means four grouped radix-2 stages; it must not be described as a
distinct radix-16 algorithm.

## Genuine power-of-two radix transforms

This algorithm family has one shared mathematical plan, typed table package,
backend, and native ABI for strict fixed-radix policies. Its dedicated radix-4,
radix-8, and radix-16 CUDA butterflies have a distinct implementation identity
from the grouped radix-2 kernels.

The strict policies are `radix4_compact`, `radix8_compact`, and
`radix16_compact`. Every transform digit has exactly that radix. Consequently,
they require `logN` to be divisible by 2, 3, and 4, respectively; configuration
rejects an incompatible ring before any plan or GPU table is built. In
particular, `radix16_compact` rejects `logN = 14`. The function
`fhelium.compatible_ntt_backends(logN)` returns only names valid for a given
ring dimension.

Forward digits use decimation in frequency (DIF); inverse digits use the dual
decimation-in-time (DIT) order.

For one radix-$R$ digit, the plan chooses an outer root $\beta$ for each group
and a fixed primitive $R$-th root $\zeta_R$. The forward butterfly evaluates

$$
Y_l = \sum_{k=0}^{R-1} X_k
      \left(\beta\,\zeta_R^{\operatorname{bitrev}(l)}\right)^k.
$$

The table stores $\beta^1,\ldots,\beta^{R-1}$ for every digit group. Across a
complete transform these outer twists total exactly $N-1$ values per prime.
The fixed cyclic-root table contains only 4, 8, or 16 values per prime. Thus
the family remains $O(N)$ and does not
reintroduce an expanded indexed schedule.

Radix-4 uses a dedicated four-point cyclic NTT butterfly. Radix-8 uses a
dedicated 2x4 Cooley--Tukey butterfly: two radix-4 transforms, fixed
$\zeta_8^u$ coupling, and one combine step. Radix-16 uses a dedicated 4x4
Cooley--Tukey butterfly: four radix-4 column transforms, the fixed radix-16
coupling matrix, and four radix-4 row transforms. These dedicated butterflies
operate independently of radix-2 stage schedules and twiddles. Inverse DIT
executes the
dual fixed-width digit order, applies inverse cyclic roots, and then the
inverse outer twist; the usual single $N^{-1}$ epilogue remains unchanged.

Shared-memory capacity and the production fusion depth are recorded by the
native CUDA implementation. The compiled maximum and
current production default are both eight transform bits, corresponding to a
maximum 256-coefficient physical tile. Production Torch operators use this
compiled choice directly.

For an eligible strict schedule, the forward launcher chooses the largest
suffix of complete radix digits whose widths fit the native default; the
inverse launcher chooses the corresponding dual prefix. A realized selection may cover
fewer than eight bits because a digit is never split merely to fill the budget.
The selected digits execute consecutively after one coalesced tile load and
before one coalesced store, while preserving the digit-bit-reversed
intermediate layout after every individual DIF digit.

Eight is a measured static engineering choice. It covers the profiled
low-stride bottleneck, fits two complete
radix-16 digits or four radix-4 digits, and needs only two 256-element shared
buffers (4 KiB for int64 residues). A smaller budget misses the complete
two-radix16 region; a larger tile would reduce Cooperative Thread Array (CTA,
CUDA thread-block) supply and increase shared
memory and synchronization without demonstrated benefit.

The genuine-radix public names identify one strict-radix policy whose native
implementation includes shared fusion. The registered compact radix-2 names
expose grouping and `smem8` as distinct selectable execution policies. Result
provenance should still record the
FHElium version in case internal tuning changes.

This genuine-radix locality optimization assigns one worker to each radix-4
four-point tuple. Radix-8 and radix-16 use
four-worker groups to evaluate their 2x4 and 4x4 factorizations through shared
scratch space, reducing each worker's live register vector. A separate
`fhelium_ntt_diagnostic_ops` namespace accepts a specified
`shared_memory_log_n` override for correctness tests and cross-GPU profiling;
the production backend never calls that namespace.

The supported schedules are:

| `logN` | compatible strict genuine-radix schedules |
|---|---|
| 14 | $4^7$ |
| 15 | $8^5$ |
| 16 | $4^8$, $16^4$ |
| 17 | none |

The power-of-two radix kernels may choose a different representative in the
lazy $[0,2q)$ interval than sequential radix-2 because modular additions are
associated differently. Forward results are therefore congruent modulo $q$,
rather than necessarily bit-for-bit equal as signed integers. Standard-range inverse
outputs are equal and all domain and representation states are unchanged.

The default remains `radix2_compact_group8_smem8`. A new algorithm family is
not promoted merely because it has fewer mathematical digits; radix-4/8/16 can
trade fewer launches for more register pressure and fixed-root arithmetic and
must win full CKKS workloads before becoming the default.

## Representation invariants

Backend methods must preserve the declared transitions:

```mermaid
flowchart LR
    CS[Coefficient / standard]
    FWD[Forward NTT]
    NM[NTT / Montgomery]
    POINT[Pointwise Montgomery arithmetic]
    INV[Inverse NTT]
    OUT[Coefficient / standard<br/>or another named output form]

    CS --> FWD --> NM --> POINT --> INV --> OUT
```

NTT domain and Montgomery form are separate metadata dimensions even when a
valid public NTT ciphertext uses both.

## Batch axes and the native RNS ABI

Public RNS operands use `[..., limb, coefficient]`. A full ciphertext uses
`[component, *batch, limb, coefficient]`; the engine selects one component
before calling an RNS or NTT backend, leaving
`[*batch, limb, coefficient]`.

Native CUDA helpers collapse only that homogeneous batch prefix into a
zero-copy flattened view:

```text
[*batch, limb, N] -> [B_flat, limb, N]
```

For an unbatched component, `B_flat=1`. RNS parameters retain their independent
`[parameter, limb]` layout and are never interpreted as message batches.
Component axes, hybrid-digit axes, and message-batch axes must not be flattened
together merely because each is dense.

The collapse uses a zero-copy `view`. A caller that creates a
non-collapsible layout must expose its repacking step.
Native binary operators require equal `B_flat`, except for named shared-public
operand requirements that allow a singleton batch.

## High-risk layout and state cases

Whenever row mapping, tables, or kernels change, test:

- depth zero and a middle depth;
- the final legal active row configuration;
- one-row/singleton digit paths;
- Q and QP bases;
- compact current rows versus global parameter offsets;
- multiple `logN` values;
- indexed and compact families;
- every declared compact group width and indexed execution;
- every compatible strict radix-4, radix-8, and radix-16 schedule;
- batched/component tensor axes;
- partial-limb inputs only where the operation supports them.

## Performance methodology

Benchmark NTT policy at three layers:

1. forward/inverse microbench for configured shapes and active rows;
2. CKKS operators that use the transforms;
3. complete workloads with keys, memory, and launch policy.

Do not promote the fastest depth-zero transform automatically to every depth or
workload.

## Continue

- [Multiplication, key switching, and rescale](multiplication-keyswitch-rescale.md)
- [Native operator workflow](native-operator-workflow.md)
- [Configuration and modulus chain](../concepts/ckks/context-and-modulus-chain.md)
- [CKKS cost model](../concepts/performance/cost-model.md)

## Source map

| Responsibility | Source |
| --- | --- |
| RNS chain, layout, and hybrid digit identity | `fhelium/backend/rns/{chain,layout,decomposition}.py` |
| Parameter materialization and aligned views | `fhelium/backend/rns/{montgomery,parameters,runtime}.py` |
| NTT policy definitions | `fhelium/config/ntt.py` |
| Host plans and typed device tables | `fhelium/backend/ntt/plans/`, `fhelium/backend/ntt/tables.py` |
| Python NTT executors | `fhelium/backend/ntt/executors/` |
| RNS schemas and CPU/CUDA implementations | `csrc/ops/rns/` |
| NTT schemas and CPU/CUDA implementations | `csrc/ops/ntt/` |
| Shared modular helpers | `csrc/ops/common/` |
| Focused correctness and policy tests | `tests/backend/test_ntt_backend.py`, `tests/native/test_native_operator_invariants.py` |
