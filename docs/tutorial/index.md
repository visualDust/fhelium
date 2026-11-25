# Tutorials

Start with [installation](installation.md) and [security](../developer/security.md). The [example catalog](tutorials.md) lists every numbered workflow and its source. Examples are grouped by module; select a track rather than running the catalog as a single application.

| Module | Examples | Entry point |
| --- | --- | --- |
| Eager | 01–08 | [Basic CKKS workflow](basic-ckks-workflow.md) |
| Algorithms | 09 | [Bootstrapping](composable-ckks-bootstrap.md) |
| Values, Serialization, and Artifacts | 10–11 | [Value movement and files](value-memory-and-persistence.md) |
| Compile | 12–17 | [JIT compilation](compile-jit.md) |
| Runtime | 18–19 | [Double-buffered execution](reusable-value-buffer.md) |
| Residency | 20–21 | [Manual Residency](explicit-residency.md) |
| Distributed | 22–25 | [Data-parallel encrypted batches](spmd-independent-ciphertexts.md) |
| Experimental | 26 | [Multiparty CKKS](multiparty-ckks.md) |

## Choose a level of control

- Start with [Eager](basic-ckks-workflow.md) for immediate numerical operations.
- Use [a Compile decorator](compile-jit.md) for reusable specialization, [a pipeline](compose-and-execute-compile-pipeline.md) to choose transformations, or [a custom pass](customize-compile-pass-and-pipeline.md) to add a rewrite.
- Use [Program persistence](compile-material-persistence.md) to separate capture from deployment and supply materials at the execution site.
- Use [buffers](reusable-value-buffer.md), [CUDA Graphs](cuda-graph-matvec.md), or [Residency](explicit-residency.md) for different execution and memory controls.
- Choose a [distributed partition](../concepts/distributed/spmd-model.md) based on whether local results are independent values, additive terms, or prime rows.

[Concepts](../concepts/index.md) explain the model; [how-to guides](../how-to/index.md) cover focused procedures; the [API reference](../api/index.md) gives current signatures; the [developer guide](../developer/index.md) covers implementation.
