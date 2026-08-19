# FHElium examples

The numbered examples are maintained runnable companions to the documentation
tutorials. Run them from the repository root in the same Python/PyTorch
environment used to build FHElium:

```bash
python examples/01_basic_ckks_flow.py --preset slots8192-scale40-levels7-int64
python examples/01_basic_ckks_flow.py --device cuda:0 \
  --preset slots8192-scale40-levels7-int64
```

Use `python examples/<file>.py --help` to inspect a script's preset, device, and
workload options. General local examples default to CPU. CUDA-specific Graph,
buffer, and Residency examples retain a CUDA default and reject CPU with a parser error.
Each maintained example exposes its CKKS state and workload policy in a
self-contained workflow.

## Example index

| File | Purpose |
| --- | --- |
| [`01_basic_ckks_flow.py`](../docs/tutorial/basic-ckks-workflow.md) | Encrypt, add, multiply, relinearize, rescale, rotate, decrypt, and compare with cleartext |
| [`02_key_materials.py`](../docs/tutorial/key-materials.md) | Inspect key layouts and optionally persist selected public/evaluation material |
| [`03_plaintext_ciphertext_memory.py`](../docs/tutorial/value-memory-and-persistence.md) | Compare value sizes, move a live activation, and round-trip exact value files |
| [`04_modulus_chain_depth.py`](../docs/tutorial/modulus-chain-depth.md) | Relate configured chain depth, modulus width, active rows, security budget, and ciphertext size |
| [`05_explicit_scale_management.py`](../docs/tutorial/explicit-scale-management.md) | Plan actual per-value scales against dropped Q primes and align level separately |
| [`06_explicit_state_late_relinearization_ntt.py`](../docs/tutorial/late-relinearization-and-ntt-reuse.md) | Reuse NTT operands and delay relinearization under declared CKKS state constraints |
| [`07_rotation_hoisting_benchmark.py`](../docs/tutorial/rotation-hoisting.md) | Compare independent rotations with grouped rotation hoisting |
| [`08_spmd_independent_ciphertexts.py`](../docs/tutorial/spmd-independent-ciphertexts.md) | Scatter/evaluate/gather independent encrypted samples |
| [`09_spmd_rotation_parallel_mxv.py`](../docs/tutorial/spmd-rotation-parallel-matvec.md) | Partition additive diagonal terms and reduce ciphertext partials |
| [`10_spmd_limb_parallel_pipeline.py`](../docs/tutorial/spmd-limb-parallel-pipeline.md) | Partition RNS rows and reconstruct every expected active row at reconstruction points |
| [`11_cuda_graph_matrix_vector.py`](../docs/tutorial/cuda-graph-matvec.md) | Capture a fixed evaluator and replay it with staged ciphertext inputs |
| [`12_reusable_value_buffer.py`](../docs/tutorial/reusable-value-buffer.md) | Stream pinned-host plaintext tiles through fixed CUDA buffers |
| [`13_explicit_residency.py`](../docs/tutorial/explicit-residency.md) | Manage opaque local handles with optional pinned/CUDA budgets, a scoped reservation, and an event-backed CUDA lease |
| [`14_automatic_residency.py`](../docs/tutorial/automatic-residency.md) | Prepare and review deterministic reclaim and admission for an exact CUDA working set under a strict managed budget |
| [`15_homogeneous_batching.py`](../docs/tutorial/homogeneous-batching.md) | Compare homogeneous message batches with unbatched loops |
| [`16_compressed_plaintext.py`](../docs/tutorial/compressed-plaintext.md) | Validate lossless operation-ready plaintext compression and direct evaluation |
| [`17_ckks_bootstrap_logn16.py`](../docs/tutorial/composable-ckks-bootstrap.md) | Refresh a depleted ciphertext with a versioned composable bootstrap factory |
| [`18_multiparty_ckks.py`](../docs/tutorial/multiparty-ckks.md) | Exercise stateless multiparty arithmetic with two in-process party records, synthetic data, and throwaway keys |
| [`19_unified_jit.py`](../docs/tutorial/unified-jit.md) | Trace a typed PyTorch matrix-vector quadratic into one mixed-dialect `Program`, apply a selected lowering pipeline, provision current key requirements, and execute with a retained workspace |
| [`20_jit_textual_ir.py`](../docs/tutorial/jit-textual-ir.md) | Parse and round-trip mixed-dialect text, record requirements in a custom pass, bind an application operation handler, and execute on CPU |
| [`21_jit_custom_pipeline.py`](../docs/tutorial/jit-custom-pipeline.md) | Extend the default pipeline with an explicit CKKS audit and executable validator, retain workspace analysis, and run an encrypted quadratic |

The [tutorial catalog](../docs/tutorial/tutorials.md) groups these files into core
evaluator, value/storage, performance, distributed, repeated-execution, and
feature tracks.

## Run the SPMD examples

Every SPMD example also runs with world size one:

```bash
python examples/08_spmd_independent_ciphertexts.py
python examples/09_spmd_rotation_parallel_mxv.py --size 8
python examples/10_spmd_limb_parallel_pipeline.py
```

Run one process per GPU with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=2 \
  examples/08_spmd_independent_ciphertexts.py

torchrun --standalone --nproc-per-node=2 \
  examples/09_spmd_rotation_parallel_mxv.py --size 8

torchrun --standalone --nproc-per-node=2 \
  examples/10_spmd_limb_parallel_pipeline.py
```

Choose the collective from the mathematical relation among process-local
values:

| Relation | Correct operation | Example |
| --- | --- | --- |
| Independent logical ciphertexts | Scatter/gather typed values | 08 |
| Disjoint additive terms of one result | Broadcast input, then typed ciphertext reduction | 09 |
| Disjoint RNS rows of one value | Limb scatter/gather and structural reconstruction | 10 |

Raw machine-integer all-reduce is not a valid ciphertext reduction, and limb
reconstruction is not addition. Read the
[SPMD model](../docs/concepts/distributed/spmd-model.md) before adapting these
programs.

## Feature examples

Examples 17 through 21 are opt-in evaluation workflows. Bootstrap factories do
not prove an application's encrypted input range or precision target. The
current multiparty output operations have no supported privacy guarantee or
production-security guarantee; read the [multiparty supported security scope](../docs/how-to/use-multiparty-ckks.md)
and use synthetic data and throwaway keys. JIT retains one
mixed-dialect `Program`; applications choose passes, handlers,
runtime materials, engines, and evaluation keys before execution. Read the
[bootstrapping](../docs/tutorial/composable-ckks-bootstrap.md),
[multiparty](../docs/tutorial/multiparty-ckks.md), or
[JIT](../docs/tutorial/unified-jit.md) tutorial before adapting the
corresponding workflow.
