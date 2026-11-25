# FHElium examples

Each numbered file demonstrates one usage model or method. Related modules have consecutive numbers; the sequence is a catalog, not a mandatory course.

Run from the repository root with the Python/PyTorch environment used to build FHElium. Start with a small CPU workflow or the corresponding CUDA invocation:

```bash
python examples/01_eager_basics.py --preset slots8192-scale40-depth7
python examples/01_eager_basics.py --device cuda:0 --preset slots8192-scale40-depth7
```

Scripts with configurable workloads expose `--help`. Examples 13–16 and 25 use fixed CPU demonstrations; Runtime and Residency examples 18–21 require CUDA. Bootstrap example 09 also requires CUDA and a substantially larger working set. General Eager scripts expose `--preset` and `--device`. Distributed scripts 22–24 expose `--preset` and select rank-local devices through `dist.init`: CUDA when available, otherwise CPU.

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

## 09 · Algorithms

| Example | Demonstrates |
| --- | --- |
| [`09_algorithms_bootstrap.py`](09_algorithms_bootstrap.py) · [Tutorial](../docs/tutorial/composable-ckks-bootstrap.md) | Refresh a maximum-depth ciphertext on a built-in bootstrap preset through Eager, Compile, or CUDA Graph execution |

## 10–11 · Values, Serialization, and Artifacts

| Example | Demonstrates |
| --- | --- |
| [`10_value_files.py`](10_value_files.py) · [Tutorial](../docs/tutorial/value-memory-and-persistence.md) | Move typed values and restore them from caller-owned files |
| [`11_artifact_store.py`](11_artifact_store.py) · [Tutorial](../docs/tutorial/artifact-store.md) | Use logical names, collections, and generation-specific references |

## 12–17 · Compile

| Example | Demonstrates |
| --- | --- |
| [`12_compile_jit.py`](12_compile_jit.py) · [Tutorial](../docs/tutorial/compile-jit.md) | Reuse a decorated function across input contents and static specializations |
| [`13_compile_pipeline.py`](13_compile_pipeline.py) · [Tutorial](../docs/tutorial/compose-and-execute-compile-pipeline.md) | Select built-in transformation and scheduling passes before direct linking |
| [`14_compile_textual_ir.py`](14_compile_textual_ir.py) · [Tutorial](../docs/tutorial/ir-textual-program.md) | Parse, inspect, transform, and print an open mixed-level Program |
| [`15_compile_custom_pass.py`](15_compile_custom_pass.py) · [Tutorial](../docs/tutorial/customize-compile-pass-and-pipeline.md) | Implement one matrix-to-BSGS rewriting pass |
| [`16_compile_python_codegen.py`](16_compile_python_codegen.py) · [Tutorial](../docs/tutorial/generate-python.md) | Export editable Eager and Backend Python at selected IR stages |
| [`17_compile_material_persistence.py`](17_compile_material_persistence.py) · [Tutorial](../docs/tutorial/compile-material-persistence.md) | Name materials, save none/some/all data, and fill bindings after load |

## 18–19 · Runtime

| Example | Demonstrates |
| --- | --- |
| [`18_runtime_double_buffer.py`](18_runtime_double_buffer.py) · [Tutorial](../docs/tutorial/reusable-value-buffer.md) | Overlap pinned-host transfers with computation in fixed CUDA buffers |
| [`19_runtime_cuda_graph.py`](19_runtime_cuda_graph.py) · [Tutorial](../docs/tutorial/cuda-graph-matvec.md) | Capture a fixed evaluator and replay with changing input data |

## 20–21 · Residency

| Example | Demonstrates |
| --- | --- |
| [`20_residency_manual.py`](20_residency_manual.py) · [Tutorial](../docs/tutorial/explicit-residency.md) | Plan placements and protect asynchronous readers with leases |
| [`21_residency_automatic.py`](21_residency_automatic.py) · [Tutorial](../docs/tutorial/automatic-residency.md) | Inspect and execute automatic admission under managed memory pressure |

## 22–25 · Distributed

| Example | Demonstrates |
| --- | --- |
| [`22_distributed_batch_inputs.py`](22_distributed_batch_inputs.py) · [Tutorial](../docs/tutorial/spmd-independent-ciphertexts.md) | Split a global encrypted batch across ranks and restore sample order |
| [`23_distributed_partial_results.py`](23_distributed_partial_results.py) · [Tutorial](../docs/tutorial/spmd-rotation-parallel-matvec.md) | Partition additive terms and reduce ciphertext partials |
| [`24_distributed_rns_shards.py`](24_distributed_rns_shards.py) · [Tutorial](../docs/tutorial/spmd-limb-parallel-pipeline.md) | Partition one ciphertext by prime rows and reconstruct its full basis |
| [`25_distributed_collective_ir.py`](25_distributed_collective_ir.py) · [Tutorial](../docs/tutorial/rank-local-collective-ir.md) | Express a collective using a specialized op or a generic combine region |

## 26 · Experimental

| Example | Demonstrates |
| --- | --- |
| [`26_experimental_multiparty.py`](26_experimental_multiparty.py) · [Tutorial](../docs/tutorial/multiparty-ckks.md) | Compose multiparty arithmetic with application-owned protocol state |

## Entry points

- **Eager versus Compile:** 01 executes operations immediately; 12 demonstrates Eager-first or Compile-first callable execution with disk reuse and opt-in automatic Graph capture; 13 controls the complete pass sequence. Example 15 adds a custom transformation, rather than another default compilation wrapper.
- **Persistence:** 10 owns individual file paths; 11 owns named artifact generations; 17 persists a Program with optional data. Key creation stays in 02.
- **Repeated execution:** 18 schedules copies and buffer reuse; 19 captures and replays device execution, with `--compile` selecting callable-owned capture. Residency 20–21 adds managed placement and admission.
- **Distributed:** 22 gathers independent outputs, 23 reduces additive partials, 24 reconstructs disjoint RNS rows, and 25 describes a collective in Program IR.

## Run distributed examples

Examples 22–24 support world size one and one process per GPU:

```bash
python examples/22_distributed_batch_inputs.py --batch-size 7
python examples/23_distributed_partial_results.py --size 8
python examples/24_distributed_rns_shards.py

torchrun --standalone --nproc-per-node=2 examples/22_distributed_batch_inputs.py --batch-size 7
torchrun --standalone --nproc-per-node=2 examples/23_distributed_partial_results.py --size 8
torchrun --standalone --nproc-per-node=2 examples/24_distributed_rns_shards.py
```

Gather, ciphertext reduction, and RNS reconstruction express different mathematical relationships. Read the [SPMD model](../docs/concepts/distributed/spmd-model.md) before adapting the partition.

## C++ library

[`cpp/`](cpp) is a CMake project that links the installed `libfhelium` through `find_package(FHElium)`. It evaluates an encrypted 8 × 8 matrix-vector product in C++ with a built-in preset: key generation, encoding and encryption, rotations by hybrid key switching, a weighted sum of encoded diagonals, rescale, decryption, and decoding, compared with the double-precision product. It runs on CPU and, when available, on the first CUDA device. Build it with the Python environment that provides FHElium and Torch:

```bash
cmake -S examples/cpp -B build/cpp-example \
  -DCMAKE_PREFIX_PATH="$(python -c 'import fhelium.native, torch.utils; print(fhelium.native.cmake_prefix_path + ";" + torch.utils.cmake_prefix_path)')"
cmake --build build/cpp-example
build/cpp-example/ckks_matrix_vector
```

See [Call the native library from C++](../docs/how-to/call-native-library-from-cpp.md) for the interface and linking requirements.

## Bootstrap and experimental scope

Example 26 remains last and opt-in. The bootstrap example 09 does not prove an application's encrypted input range. The multiparty example uses synthetic data and throwaway keys; its output protocols have no supported privacy or production security guarantee. Read the linked tutorials before running them.
