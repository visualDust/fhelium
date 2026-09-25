# Tutorials

Start with [installation](installation.md) and [security](../developer/security.md). The [example catalog](tutorials.md) lists every numbered workflow and its source. Examples are grouped by module; select a track rather than running the catalog as a single application.

| Module | Examples | Entry point |
| --- | --- | --- |
| Eager | 01–08 | [Basic CKKS workflow](basic-ckks-workflow.md) |
| Values, Serialization, and Artifacts | 09–10 | [Value movement and files](value-memory-and-persistence.md) |
| Compile | 11–16 | [JIT compilation](compile-jit.md) |
| Runtime | 17–18 | [Double-buffered execution](reusable-value-buffer.md) |
| Residency | 19–20 | [Manual Residency](explicit-residency.md) |
| Distributed | 21–24 | [Data-parallel encrypted batches](spmd-independent-ciphertexts.md) |
| Experimental | 25–26 | [Experimental bootstrapping](composable-ckks-bootstrap.md) |

## Choose a level of control

- Start with [Eager](basic-ckks-workflow.md) for immediate numerical operations.
- Use [a Compile decorator](compile-jit.md) for reusable specialization, [a pipeline](compose-and-execute-compile-pipeline.md) to choose transformations, or [a custom pass](customize-compile-pass-and-pipeline.md) to add a rewrite.
- Use [Program persistence](compile-material-persistence.md) to separate capture from deployment and supply materials at the execution site.
- Use [buffers](reusable-value-buffer.md), [CUDA Graphs](cuda-graph-matvec.md), or [Residency](explicit-residency.md) for different execution and memory controls.
- Choose a [distributed partition](../concepts/distributed/spmd-model.md) based on whether local results are independent values, additive terms, or prime rows.

[Concepts](../concepts/index.md) explain the model; [how-to guides](../how-to/index.md) cover focused procedures; the [API reference](../api/index.md) gives current signatures; the [developer guide](../developer/index.md) covers implementation.
