# FHElium examples

The numbered examples are runnable companions to the documentation
tutorials. Run them from the repository root in the same Python/PyTorch
environment used to build FHElium:

```bash
python examples/01_basic_ckks_flow.py --preset slots8192-scale40-levels7-int64
python examples/01_basic_ckks_flow.py --device cuda:0 \
  --preset slots8192-scale40-levels7-int64
```

Use `python examples/<file>.py --help` to inspect a script's configuration,
device, and workload options. Some evidence-backed examples fix their CKKS
configuration rather than expose it as a command-line option. General local
examples default to CPU. CUDA-specific Graph, buffer, and Residency examples
use a CUDA default and reject CPU with a parser error. Each example exposes
its CKKS state and workload policy in a self-contained workflow.

For a practical reusable-buffer smoke run, keep Example 12's fixed
`slots32768-scale40-levels34-int64` preset at level `20` while reducing only
its residency allocation:

```bash
python examples/12_reusable_value_buffer.py \
  --num-tiles 4 --plaintexts-per-tile 4 --message-size 32
```

## Example index

| File | Purpose |
| --- | --- |
| [`01_basic_ckks_flow.py`](../docs/tutorial/basic-ckks-workflow.md) | Encrypt, add, multiply, relinearize, rescale, rotate, decrypt, and compare with cleartext |
| [`02_key_materials.py`](../docs/tutorial/key-materials.md) | Inspect key layouts and optionally persist selected public/evaluation material |
| [`03_plaintext_ciphertext_memory.py`](../docs/tutorial/value-memory-and-persistence.md) | Compare value sizes, move a live activation, and round-trip serialized value files |
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
| [`14_automatic_residency.py`](../docs/tutorial/automatic-residency.md) | Prepare and review deterministic reclaim and admission for a CUDA working set under a strict managed budget |
| [`15_homogeneous_batching.py`](../docs/tutorial/homogeneous-batching.md) | Compare homogeneous message batches with unbatched loops |
| [`16_compressed_plaintext.py`](../docs/tutorial/compressed-plaintext.md) | Validate lossless operation-ready plaintext compression and direct evaluation |
| [`17_compose_and_execute.py`](../docs/tutorial/compose-and-execute-compile-pipeline.md) | Compose built-in passes, lower a captured rotated quadratic to Backend operations, link its keys and resources, and execute it |
| [`18_ir_textual_program.py`](../docs/tutorial/ir-textual-program.md) | Parse and round-trip textual mixed-level IR, insert a caller-defined analysis pass, and inspect partial lowering |
| [`19_customize_compile_pass.py`](../docs/tutorial/customize-compile-pass-and-pipeline.md) | Define a BSGS rewriting pass, compose a caller-selected pipeline, link its resources, and execute the resulting Program |
| [`20_generate_python.py`](../docs/tutorial/generate-python.md) | Emit editable Eager API calls from CKKS IR and direct Backend implementation calls from a lowered Program |
| [`21_rank_local_collective_ir.py`](../docs/tutorial/rank-local-collective-ir.md) | Compare specialized ciphertext reduction with generic rank-local all-reduce containing a visible CKKS-add region |
| [`22_ckks_bootstrap_logn16.py`](../docs/tutorial/composable-ckks-bootstrap.md) | Refresh a depleted ciphertext with a versioned composable bootstrap factory |
| [`23_multiparty_ckks.py`](../docs/tutorial/multiparty-ckks.md) | Exercise stateless multiparty arithmetic with two in-process party records, synthetic data, and throwaway keys |

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

Examples 17 through 20 cover Compile capture, textual IR, a custom BSGS pass,
Backend execution, and editable Python emission. Example 21 shows a
specialized-to-generic rank-local collective transformation. Examples 22 and
23 are opt-in evaluation workflows. Bootstrap factories do not prove an
application's encrypted input range or precision target. The current
multiparty output operations have no supported privacy guarantee or
production-security guarantee; read the
[multiparty supported security scope](../docs/how-to/use-multiparty-ckks.md)
and use synthetic data and throwaway keys. Read the
[built-in Compile execution](../docs/tutorial/compose-and-execute-compile-pipeline.md),
[custom Compile pass and pipeline](../docs/tutorial/customize-compile-pass-and-pipeline.md),
[generated Python](../docs/tutorial/generate-python.md),
[rank-local collective IR](../docs/tutorial/rank-local-collective-ir.md),
[bootstrapping](../docs/tutorial/composable-ckks-bootstrap.md), or
[multiparty](../docs/tutorial/multiparty-ckks.md) tutorial before adapting the
corresponding workflow.
