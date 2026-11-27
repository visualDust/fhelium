# Tutorials

Start with [installation](installation.md) and [security](../developer/security.md). Each numbered example under `examples/` demonstrates one usage model, and each has a walkthrough page. Examples are grouped by module; select a track rather than running the catalog as a single application.

## Choose a level of control

- Start with [Eager](basic-ckks-workflow.md) for immediate numerical operations.
- Use [`@fc.compile`](compile-layer.md) to capture a function and reuse its specializations, [a pipeline](compose-and-execute-compile-pipeline.md) to choose transformations, or [a custom pass](customize-compile-pass-and-pipeline.md) to add a rewrite.
- Use [export and load](export-and-load.md) to deploy a compiled callable in another Python or C++ process.
- Use [double buffering](reusable-value-buffer.md), [CUDA Graph segments](compiled-cuda-graphs.md), or [Residency](manual-residency.md) for different execution and memory controls.
- Choose a [distributed partition](../concepts/distributed/spmd-model.md) based on whether local results are independent values, additive terms, or prime rows.

## Example catalog

### Eager

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [01](https://github.com/VisualDust/fhelium/blob/main/examples/01_eager_basics.py) | [Basic CKKS workflow](basic-ckks-workflow.md) | Encrypt, evaluate, and decrypt one process-local computation |
| [02](https://github.com/VisualDust/fhelium/blob/main/examples/02_eager_key_materials.py) | [Key creation and installation](key-materials.md) | Create typed keys from one secret and install selected evaluator capabilities |
| [03](https://github.com/VisualDust/fhelium/blob/main/examples/03_eager_depth_and_scale.py) | [Depth and scale](depth-and-scale.md) | Inspect per-depth Q bases and plan per-value scales |
| [04](https://github.com/VisualDust/fhelium/blob/main/examples/04_eager_ntt_reuse.py) | [NTT reuse and late relinearization](late-relinearization-and-ntt-reuse.md) | Schedule reusable NTT operands and delay three-component reduction |
| [05](https://github.com/VisualDust/fhelium/blob/main/examples/05_eager_rotation_hoisting.py) | [Rotation hoisting](rotation-hoisting.md) | Request grouped rotations and compare with independent calls |
| [06](https://github.com/VisualDust/fhelium/blob/main/examples/06_eager_batching.py) | [Homogeneous batching](homogeneous-batching.md) | Use leading Tensor batch axes instead of an evaluation loop |
| [07](https://github.com/VisualDust/fhelium/blob/main/examples/07_eager_compressed_plaintext.py) | [Compressed plaintexts](compressed-plaintext.md) | Evaluate losslessly compressed operation-ready plaintexts |

### Algorithms

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [08](https://github.com/VisualDust/fhelium/blob/main/examples/08_algorithms_linear_polynomial.py) | [Linear layer and polynomial activation](linear-layer-and-polynomial.md) | Apply a planned matrix-vector product and a Chebyshev activation |
| [09](https://github.com/VisualDust/fhelium/blob/main/examples/09_algorithms_bootstrap.py) | [Bootstrapping](composable-ckks-bootstrap.md) | Refresh a maximum-depth ciphertext on a built-in bootstrap preset |

### Values and storage

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [10](https://github.com/VisualDust/fhelium/blob/main/examples/10_value_files.py) | [Value movement and files](value-memory-and-persistence.md) | Move typed values and restore them from caller-owned files |
| [11](https://github.com/VisualDust/fhelium/blob/main/examples/11_value_store.py) | [Named values and generations](value-store.md) | Store values under logical names, collections, and generation-specific references |

### Compile

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [12](https://github.com/VisualDust/fhelium/blob/main/examples/12_compile_layer.py) | [Compile a layer](compile-layer.md) | Compile an encrypted layer, reuse its specializations, and read its execution statistics |
| [13](https://github.com/VisualDust/fhelium/blob/main/examples/13_compile_pipeline.py) | [Program pipelines and textual IR](compose-and-execute-compile-pipeline.md) | Select built-in passes, transform textual IR, and link a Program |
| [14](https://github.com/VisualDust/fhelium/blob/main/examples/14_compile_custom_pass.py) | [Custom BSGS transformation](customize-compile-pass-and-pipeline.md) | Implement one matrix-to-BSGS rewriting pass |
| [15](https://github.com/VisualDust/fhelium/blob/main/examples/15_compile_export_load.py) | [Export and load a compiled callable](export-and-load.md) | Export a compiled function to one file and call it in another process |

### Runtime

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [16](https://github.com/VisualDust/fhelium/blob/main/examples/16_runtime_double_buffer.py) | [Double-buffered serving](reusable-value-buffer.md) | Serve queued requests with a compiled layer while pinned-host tiles stream into two device buffers |
| [17](https://github.com/VisualDust/fhelium/blob/main/examples/17_runtime_cuda_graph.py) | [Compiled CUDA Graphs](compiled-cuda-graphs.md) | Replay compiled Graph segments, chain compiled functions, and keep results across calls |

### Residency

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [18](https://github.com/VisualDust/fhelium/blob/main/examples/18_residency_manual.py) | [Manual Residency](manual-residency.md) | Plan placements between compiled calls and protect asynchronous readers with leases |
| [19](https://github.com/VisualDust/fhelium/blob/main/examples/19_residency_automatic.py) | [Automatic Residency admission](automatic-residency.md) | Inspect and execute automatic admission under managed memory pressure |

### Distributed

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [20](https://github.com/VisualDust/fhelium/blob/main/examples/20_distributed_batch_inputs.py) | [Data-parallel encrypted batches](spmd-independent-ciphertexts.md) | Split a global encrypted batch across ranks and restore sample order |
| [21](https://github.com/VisualDust/fhelium/blob/main/examples/21_distributed_partial_results.py) | [Additive partial results](spmd-rotation-parallel-matvec.md) | Partition additive terms and reduce ciphertext partials |
| [22](https://github.com/VisualDust/fhelium/blob/main/examples/22_distributed_rns_shards.py) | [RNS-sharded execution](spmd-limb-parallel-pipeline.md) | Partition one ciphertext by prime rows and reconstruct its full basis |
| [23](https://github.com/VisualDust/fhelium/blob/main/examples/23_distributed_collective_ir.py) | [Rank-local collective IR](rank-local-collective-ir.md) | Express a collective using a specialized op or a generic combine region |

### Experimental

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [24](https://github.com/VisualDust/fhelium/blob/main/examples/24_experimental_multiparty.py) | [Experimental multiparty CKKS](multiparty-ckks.md) | Compose multiparty arithmetic with application-owned protocol state |

### C++

| Example | Demonstrates |
| --- | --- |
| [`cpp/ckks_matrix_vector.cpp`](https://github.com/VisualDust/fhelium/blob/main/examples/cpp/ckks_matrix_vector.cpp) | Evaluate an encrypted matrix-vector product through `libfhelium`; see [Call the native library from C++](../how-to/call-native-library-from-cpp.md) |
| [`cpp/load_artifact.cpp`](https://github.com/VisualDust/fhelium/blob/main/examples/cpp/load_artifact.cpp) | Load and call a compiled artifact written by example 15; see [Deploy a compiled callable](../how-to/deploy-compiled-callable.md) |
| [`cpp/runner_schedule.cpp`](https://github.com/VisualDust/fhelium/blob/main/examples/cpp/runner_schedule.cpp) · [`cpp/dispatcher_operators.cpp`](https://github.com/VisualDust/fhelium/blob/main/examples/cpp/dispatcher_operators.cpp) | Build an exec schedule for the Runner and call RNS operators through the c10 dispatcher; see [Linking, exec schedules, and the Runner](../developer/linking-and-runner.md) |

[Concepts](../concepts/index.md) explain the model; [how-to guides](../how-to/index.md) cover focused procedures; the [API reference](../api/index.md) gives current signatures; the [developer guide](../developer/index.md) covers implementation.
