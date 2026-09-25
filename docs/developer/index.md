# Developer Guide

The Developer Guide explains FHElium's implementation mechanisms: public value semantics, graph-free Eager dispatch, Program transformations, callable specialization, Backend preparation, numerical algorithms, native execution, and storage lifetimes. Eager and Compile share operation implementations; manual Program/Compilation/Pipeline execution and callable Compile are equally supported ways to compose the Compile machinery.

## Architecture overview

<DocGrid>
  <DocCard
    title="Eager, Compile, and native execution"
    description="Follow the architecture diagrams from immediate execution or manual and callable Compile into whole operations, generated kernels, and CPU/CUDA dispatch."
    href="/developer/engine-native-stack"
  />
  <DocCard
    title="Source tree and ownership"
    description="Locate the source owner for a mechanism, including material preparation, host execution, arithmetic, runtime, and persistence."
    href="/developer/source-tree"
  />
</DocGrid>

## Values, Programs, and execution preparation

| Implementation question | Article |
| --- | --- |
| How does an immediate operation update CKKS state and dispatch Tensor payloads? | [Values, operation semantics, and Eager dispatch](compiler-state-and-eager-execution.md) |
| How are computation, partial state, effects, and captured Python represented? | [IR, capture, effects, and open state](compiler-stack-internals.md) |
| What does each pass receive, and how is the common pipeline composed? | [Compilation state and pass composition](compilation-and-passes.md) |
| How are keys, tables, and execution handles prepared and bound? | [Tensor materials and operation preparation](materials-and-preparation.md) |
| How do operation declarations, lowerings, and implementation choices connect? | [Operation registration and implementation selection](operation-registration-and-selection.md) |
| How does linking generate host calls, and how do callable specializations reuse them? | [Linking and prepared host execution](prepared-host-execution.md) |
| How are SSA regions transformed into pointwise and NTT kernels? | [Generated kernels, fusion, and NTT execution](compiled-execution-and-kernels.md) |
| Where is an operation family's numerical implementation? | [IR operations and implementations](ir-operation-implementation-index.md) |

For executable tasks, use [evaluate CKKS data](../how-to/evaluate-ckks-data.md), [build a Program pipeline](../how-to/build-program-pipeline.md), or [compile a callable](../how-to/compile-callable.md). These routes share the mechanisms documented here.

## Security

[Security](security.md) describes cryptographic parameter assessment, randomness, key custody, stored data, executable Programs, and distributed execution.

## Numerical algorithms and representations

| Implementation question | Article |
| --- | --- |
| How are slots encoded, random streams advanced, and key relations constructed? | [Encoding, randomness, and key construction](encoding-randomness-and-keys.md) |
| How do prime rows, Montgomery residues, and transform schedules map to Tensor storage? | [RNS and NTT architecture](rns-and-ntt.md) |
| How do component products, hybrid digits, rotations, and quotient rounding compose? | [Multiplication, key switching, and rescale](multiplication-keyswitch-rescale.md) |
| How is compact encoded plaintext storage consumed without dense expansion? | [CompressedPlaintext internals](compressed-plaintext-internals.md) |
| How are modulus raising, linear transforms, and periodic reduction composed? | [CKKS bootstrap internals](composable-ckks-bootstrap.md) |

## Persistence, runtime, and distribution

| Implementation question | Article |
| --- | --- |
| What Program and Tensor state survives serialization, and what must be rebound? | [Compilation persistence and rebinding](compilation-persistence.md) |
| How are persistent values cataloged, published, recovered, and retired? | [ArtifactStore internals](artifact-store-v1.md) |
| How are stable buffers, asynchronous copies, and CUDA Graph output lifetimes managed? | [Execution buffers and CUDA Graphs](execution-buffers-and-cuda-graphs.md) |
| How are live values, replicas, reconstruction sources, and ownership represented? | [Residency state and ownership](residency-state-and-ownership.md) |
| How do leases, reservations, plans, and admission decisions become transitions? | [Residency plans and execution](residency-plans-and-execution.md) |
| How do rank-local IR, descriptors, payload transport, and reductions execute? | [Distributed internals](distributed-internals.md) |

## Contributor workflows

- [Contributing to FHElium](contributing.md): environments, source checks, and mathematical validation.
- [Native operator workflow](native-operator-workflow.md): schemas, CPU/CUDA implementations, wrappers, and ABI checks.
- [Binary packaging and release](binary-packaging-and-release.md): wheel identities, installed-artifact checks, and package repositories.
- [Contributing to documentation](documentation.md): page roles, static API generation, diagrams, and site builds.

Concept definitions belong in [Concepts](../concepts/index.md), runnable workflows in [Tutorials](../tutorial/index.md), and individual tasks in [How-to](../how-to/index.md). Use the generated [API reference](../api/index.md) for signatures and operation equations alongside these implementation traces.
