# Example catalog

Choose a module and a usage model. Each row links to one executable example and its tutorial. [Start with the basic evaluator](basic-ckks-workflow.md), or go to [JIT compilation](compile-jit.md) for reusable function execution.

## Eager

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [01](https://github.com/VisualDust/fhelium/blob/main/examples/01_eager_basics.py) | [Basic CKKS workflow](basic-ckks-workflow.md) | Encrypt, evaluate, and decrypt one process-local computation |
| [02](https://github.com/VisualDust/fhelium/blob/main/examples/02_eager_key_materials.py) | [Key creation and installation](key-materials.md) | Create typed keys from one secret and install selected evaluator capabilities |
| [03](https://github.com/VisualDust/fhelium/blob/main/examples/03_eager_modulus_chain.py) | [Modulus-chain depth](modulus-chain-depth.md) | Inspect configured Q groups and the available depth budget |
| [04](https://github.com/VisualDust/fhelium/blob/main/examples/04_eager_scale_management.py) | [Actual scale management](explicit-scale-management.md) | Plan per-value scales and align depth independently |
| [05](https://github.com/VisualDust/fhelium/blob/main/examples/05_eager_ntt_reuse.py) | [NTT reuse and late relinearization](late-relinearization-and-ntt-reuse.md) | Schedule reusable NTT operands and delay three-component reduction |
| [06](https://github.com/VisualDust/fhelium/blob/main/examples/06_eager_rotation_hoisting.py) | [Rotation hoisting](rotation-hoisting.md) | Request grouped rotations and compare with independent calls |
| [07](https://github.com/VisualDust/fhelium/blob/main/examples/07_eager_batching.py) | [Homogeneous batching](homogeneous-batching.md) | Use leading Tensor batch axes instead of an evaluation loop |
| [08](https://github.com/VisualDust/fhelium/blob/main/examples/08_eager_compressed_plaintext.py) | [Compressed plaintexts](compressed-plaintext.md) | Evaluate losslessly compressed operation-ready plaintexts |

## Values, Serialization, and Artifacts

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [09](https://github.com/VisualDust/fhelium/blob/main/examples/09_value_files.py) | [Value movement and files](value-memory-and-persistence.md) | Move typed values and restore them from caller-owned files |
| [10](https://github.com/VisualDust/fhelium/blob/main/examples/10_artifact_store.py) | [Named artifacts and generations](artifact-store.md) | Use logical names, collections, and generation-specific references |

## Compile

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [11](https://github.com/VisualDust/fhelium/blob/main/examples/11_compile_jit.py) | [JIT compilation](compile-jit.md) | Reuse a decorated function across input contents and static specializations |
| [12](https://github.com/VisualDust/fhelium/blob/main/examples/12_compile_pipeline.py) | [Caller-composed Compile pipeline](compose-and-execute-compile-pipeline.md) | Select built-in transformation and scheduling passes before direct linking |
| [13](https://github.com/VisualDust/fhelium/blob/main/examples/13_compile_textual_ir.py) | [Textual Program IR](ir-textual-program.md) | Parse, inspect, transform, and print an open mixed-level Program |
| [14](https://github.com/VisualDust/fhelium/blob/main/examples/14_compile_custom_pass.py) | [Custom BSGS transformation](customize-compile-pass-and-pipeline.md) | Implement one matrix-to-BSGS rewriting pass |
| [15](https://github.com/VisualDust/fhelium/blob/main/examples/15_compile_python_codegen.py) | [Generated Python](generate-python.md) | Export editable Eager and Backend Python at selected IR stages |
| [16](https://github.com/VisualDust/fhelium/blob/main/examples/16_compile_material_persistence.py) | [Program materials and persistence](compile-material-persistence.md) | Name materials, save none/some/all data, and fill bindings after load |

## Runtime

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [17](https://github.com/VisualDust/fhelium/blob/main/examples/17_runtime_double_buffer.py) | [Double-buffered execution](reusable-value-buffer.md) | Overlap pinned-host transfers with computation in fixed CUDA buffers |
| [18](https://github.com/VisualDust/fhelium/blob/main/examples/18_runtime_cuda_graph.py) | [CUDA Graph replay](cuda-graph-matvec.md) | Capture a fixed evaluator and replay with changing input data |

## Residency

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [19](https://github.com/VisualDust/fhelium/blob/main/examples/19_residency_manual.py) | [Manual Residency](explicit-residency.md) | Plan placements and protect asynchronous readers with leases |
| [20](https://github.com/VisualDust/fhelium/blob/main/examples/20_residency_automatic.py) | [Automatic Residency admission](automatic-residency.md) | Inspect and execute automatic admission under managed memory pressure |

## Distributed

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [21](https://github.com/VisualDust/fhelium/blob/main/examples/21_distributed_batch_inputs.py) | [Data-parallel encrypted batches](spmd-independent-ciphertexts.md) | Split a global encrypted batch across ranks and restore sample order |
| [22](https://github.com/VisualDust/fhelium/blob/main/examples/22_distributed_partial_results.py) | [Additive partial results](spmd-rotation-parallel-matvec.md) | Partition additive terms and reduce ciphertext partials |
| [23](https://github.com/VisualDust/fhelium/blob/main/examples/23_distributed_rns_shards.py) | [RNS-sharded execution](spmd-limb-parallel-pipeline.md) | Partition one ciphertext by prime rows and reconstruct its full basis |
| [24](https://github.com/VisualDust/fhelium/blob/main/examples/24_distributed_collective_ir.py) | [Rank-local collective IR](rank-local-collective-ir.md) | Express a collective using a specialized op or a generic combine region |

## Experimental

| Example | Tutorial | Demonstrates |
| --- | --- | --- |
| [25](https://github.com/VisualDust/fhelium/blob/main/examples/25_experimental_bootstrap.py) | [Experimental bootstrapping](composable-ckks-bootstrap.md) | Refresh a depleted ciphertext using a composable bootstrap preset |
| [26](https://github.com/VisualDust/fhelium/blob/main/examples/26_experimental_multiparty.py) | [Experimental multiparty CKKS](multiparty-ckks.md) | Compose multiparty arithmetic with application-owned protocol state |
