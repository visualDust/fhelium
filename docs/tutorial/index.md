# Tutorials

Start with one correct process-local evaluator, then choose a learning track for
the capability you actually need. The numbered examples are a catalog of
maintained workflows; they do not define one mandatory sequential course.

## First program

1. Read [Support, maturity, and security scope](support-and-security.md).
2. [Install FHElium](installation.md) against the selected CPU-only or
   CUDA-enabled PyTorch source-build environment.
3. Run the first evaluator in [Tutorials](tutorials.md).
4. Continue with [01 — Basic CKKS workflow](basic-ckks-workflow.md).

FHElium currently supports Linux x86-64 and macOS Apple Silicon, Python 3.12 or
3.13, PyTorch `>=2.10,<2.14`, and a C++17 host toolchain. CUDA builds are Linux
x86-64 only and additionally require a CUDA-enabled PyTorch distribution, a
matching CUDA source toolkit, and a supported NVIDIA GPU at runtime. macOS uses
the native CPU backend rather than PyTorch MPS. The
[installation guide](installation.md) defines the verified endpoints.

## Choose a learning track

| Goal | Suggested sequence |
| --- | --- |
| Build a correct evaluator | [01 Basic workflow](basic-ckks-workflow.md) → [02 Key lifecycle](key-materials.md) → [04 Chain depth](modulus-chain-depth.md) → [05 Actual scales](explicit-scale-management.md) → [06 Explicit reuse](late-relinearization-and-ntt-reuse.md) |
| Understand value layout and storage | [03 Memory and persistence](value-memory-and-persistence.md) → [14 Homogeneous batching](homogeneous-batching.md) → [15 Compressed plaintexts](compressed-plaintext.md) |
| Reduce rotation cost | [07 Rotation hoisting](rotation-hoisting.md) → [Benchmark a workload](../how-to/benchmark-a-workload.md) |
| Use multiple GPUs | [08 Independent ciphertexts](spmd-independent-ciphertexts.md) → [09 Additive rotation terms](spmd-rotation-parallel-matvec.md) → [10 RNS-limb pipeline](spmd-limb-parallel-pipeline.md) |
| Repeat work within bounded memory | [11 CUDA Graph](cuda-graph-matvec.md) → [12 Reusable buffers](reusable-value-buffer.md) → [13 Residency](explicit-residency.md) |
| Evaluate a feature | Read the [bootstrapping semantics and range requirements](../concepts/ckks/composable-bootstrapping.md) before [16 Bootstrapping](composable-ckks-bootstrap.md), the [multiparty supported security scope](../how-to/use-multiparty-ckks.md) before [17 Multiparty CKKS](multiparty-ckks.md), or the [JIT program model](../concepts/unified-jit-programs.md) before [18 JIT programs](unified-jit.md) |

Use [Tutorials](tutorials.md) to map every numbered page to
its source file and main question.

## Performance policy comes after correctness

The default NTT policy is sufficient to establish a first correct evaluator.
After a representative workload exists on the target GPU, use
[Screen NTT backends](../how-to/screen-ntt-backends.md), then follow the deeper
[analysis guide](../how-to/choose-ntt-backend.md) only when rankings need an
explanation.

## When you need another document type

- [Concepts](../concepts/index.md) define state, ownership, communication, and
  architecture.
- [How-to guides](../how-to/index.md) provide focused planning and diagnostic
  procedures.
- [API reference](../api/index.md) gives signatures and docstrings for
  supported interfaces and non-private implementation modules.
- [Developer Guide](../developer/index.md) covers the native operator stack and
  internal cross-layer invariants and interfaces.
