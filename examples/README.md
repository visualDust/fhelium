# FHElium examples

Each numbered file demonstrates one usage model or method. Related modules have consecutive numbers; the sequence is a catalog, not a mandatory course.

Run from the repository root with the Python/PyTorch environment used to build FHElium. Start with a small CPU workflow or the corresponding CUDA invocation:

```bash
python examples/01_eager_basics.py --preset slots8192-scale40-depth7-int64
python examples/01_eager_basics.py --device cuda:0 --preset slots8192-scale40-depth7-int64
```

Scripts with configurable workloads expose `--help`. Examples 12–15 and 24 use fixed CPU demonstrations; Runtime and Residency examples 17–20 require CUDA. Experimental bootstrap 25 also requires CUDA and a substantially larger working set. General Eager scripts expose `--preset` and `--device`. Distributed scripts 21–23 expose `--preset` and select rank-local devices through `dist.init`: CUDA when available, otherwise CPU.

## 01–08 · Eager

| Example | Demonstrates |
| --- | --- |
| [`01_eager_basics.py`](01_eager_basics.py) · [Tutorial](../docs/tutorial/basic-ckks-workflow.md) | Encrypt, evaluate, and decrypt one process-local computation |
| [`02_eager_key_materials.py`](02_eager_key_materials.py) · [Tutorial](../docs/tutorial/key-materials.md) | Create typed keys from one secret and install selected evaluator capabilities |
| [`03_eager_modulus_chain.py`](03_eager_modulus_chain.py) · [Tutorial](../docs/tutorial/modulus-chain-depth.md) | Inspect configured Q groups and the available depth budget |
| [`04_eager_scale_management.py`](04_eager_scale_management.py) · [Tutorial](../docs/tutorial/explicit-scale-management.md) | Plan per-value scales and align depth independently |
| [`05_eager_ntt_reuse.py`](05_eager_ntt_reuse.py) · [Tutorial](../docs/tutorial/late-relinearization-and-ntt-reuse.md) | Schedule reusable NTT operands and delay three-component reduction |
| [`06_eager_rotation_hoisting.py`](06_eager_rotation_hoisting.py) · [Tutorial](../docs/tutorial/rotation-hoisting.md) | Request grouped rotations and compare with independent calls |
| [`07_eager_batching.py`](07_eager_batching.py) · [Tutorial](../docs/tutorial/homogeneous-batching.md) | Use leading Tensor batch axes instead of an evaluation loop |
| [`08_eager_compressed_plaintext.py`](08_eager_compressed_plaintext.py) · [Tutorial](../docs/tutorial/compressed-plaintext.md) | Evaluate losslessly compressed operation-ready plaintexts |

## 09–10 · Values, Serialization, and Artifacts

| Example | Demonstrates |
| --- | --- |
| [`09_value_files.py`](09_value_files.py) · [Tutorial](../docs/tutorial/value-memory-and-persistence.md) | Move typed values and restore them from caller-owned files |
| [`10_artifact_store.py`](10_artifact_store.py) · [Tutorial](../docs/tutorial/artifact-store.md) | Use logical names, collections, and generation-specific references |

## 11–16 · Compile

| Example | Demonstrates |
| --- | --- |
| [`11_compile_jit.py`](11_compile_jit.py) · [Tutorial](../docs/tutorial/compile-jit.md) | Reuse a decorated function across input contents and static specializations |
| [`12_compile_pipeline.py`](12_compile_pipeline.py) · [Tutorial](../docs/tutorial/compose-and-execute-compile-pipeline.md) | Select built-in transformation and scheduling passes before direct linking |
| [`13_compile_textual_ir.py`](13_compile_textual_ir.py) · [Tutorial](../docs/tutorial/ir-textual-program.md) | Parse, inspect, transform, and print an open mixed-level Program |
| [`14_compile_custom_pass.py`](14_compile_custom_pass.py) · [Tutorial](../docs/tutorial/customize-compile-pass-and-pipeline.md) | Implement one matrix-to-BSGS rewriting pass |
| [`15_compile_python_codegen.py`](15_compile_python_codegen.py) · [Tutorial](../docs/tutorial/generate-python.md) | Export editable Eager and Backend Python at selected IR stages |
| [`16_compile_material_persistence.py`](16_compile_material_persistence.py) · [Tutorial](../docs/tutorial/compile-material-persistence.md) | Name materials, save none/some/all data, and fill bindings after load |

## 17–18 · Runtime

| Example | Demonstrates |
| --- | --- |
| [`17_runtime_double_buffer.py`](17_runtime_double_buffer.py) · [Tutorial](../docs/tutorial/reusable-value-buffer.md) | Overlap pinned-host transfers with computation in fixed CUDA buffers |
| [`18_runtime_cuda_graph.py`](18_runtime_cuda_graph.py) · [Tutorial](../docs/tutorial/cuda-graph-matvec.md) | Capture a fixed evaluator and replay with changing input data |

## 19–20 · Residency

| Example | Demonstrates |
| --- | --- |
| [`19_residency_manual.py`](19_residency_manual.py) · [Tutorial](../docs/tutorial/explicit-residency.md) | Plan placements and protect asynchronous readers with leases |
| [`20_residency_automatic.py`](20_residency_automatic.py) · [Tutorial](../docs/tutorial/automatic-residency.md) | Inspect and execute automatic admission under managed memory pressure |

## 21–24 · Distributed

| Example | Demonstrates |
| --- | --- |
| [`21_distributed_batch_inputs.py`](21_distributed_batch_inputs.py) · [Tutorial](../docs/tutorial/spmd-independent-ciphertexts.md) | Split a global encrypted batch across ranks and restore sample order |
| [`22_distributed_partial_results.py`](22_distributed_partial_results.py) · [Tutorial](../docs/tutorial/spmd-rotation-parallel-matvec.md) | Partition additive terms and reduce ciphertext partials |
| [`23_distributed_rns_shards.py`](23_distributed_rns_shards.py) · [Tutorial](../docs/tutorial/spmd-limb-parallel-pipeline.md) | Partition one ciphertext by prime rows and reconstruct its full basis |
| [`24_distributed_collective_ir.py`](24_distributed_collective_ir.py) · [Tutorial](../docs/tutorial/rank-local-collective-ir.md) | Express a collective using a specialized op or a generic combine region |

## 25–26 · Experimental

| Example | Demonstrates |
| --- | --- |
| [`25_experimental_bootstrap.py`](25_experimental_bootstrap.py) · [Tutorial](../docs/tutorial/composable-ckks-bootstrap.md) | Refresh a depleted ciphertext using a composable bootstrap preset |
| [`26_experimental_multiparty.py`](26_experimental_multiparty.py) · [Tutorial](../docs/tutorial/multiparty-ckks.md) | Compose multiparty arithmetic with application-owned protocol state |

## Entry points

- **Eager versus Compile:** 01 executes operations immediately; 11 JIT-compiles a decorated function; 12 controls the complete pass sequence. Example 14 adds a custom transformation, rather than another default compilation wrapper.
- **Persistence:** 09 owns individual file paths; 10 owns named artifact generations; 16 persists a Program with optional data. Key creation stays in 02.
- **Repeated execution:** 17 schedules copies and buffer reuse; 18 captures and replays device execution. Residency 19–20 adds managed placement and admission.
- **Distributed:** 21 gathers independent outputs, 22 reduces additive partials, 23 reconstructs disjoint RNS rows, and 24 describes a collective in Program IR.

## Run distributed examples

Examples 21–23 support world size one and one process per GPU:

```bash
python examples/21_distributed_batch_inputs.py --batch-size 7
python examples/22_distributed_partial_results.py --size 8
python examples/23_distributed_rns_shards.py

torchrun --standalone --nproc-per-node=2 examples/21_distributed_batch_inputs.py --batch-size 7
torchrun --standalone --nproc-per-node=2 examples/22_distributed_partial_results.py --size 8
torchrun --standalone --nproc-per-node=2 examples/23_distributed_rns_shards.py
```

Gather, ciphertext reduction, and RNS reconstruction express different mathematical relationships. Read the [SPMD model](../docs/concepts/distributed/spmd-model.md) before adapting the partition.

## Experimental scope

Examples 25–26 remain last and opt-in. The bootstrap example does not prove an application's encrypted input range. The multiparty example uses synthetic data and throwaway keys; its output protocols have no supported privacy or production security guarantee. Read the linked tutorials before running them.
