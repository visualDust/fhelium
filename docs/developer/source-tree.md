# Source tree

This map identifies the first implementation entry for common development
tasks. It is intentionally curated; generated files and private helpers may
change within a release series.

## Top-level areas

```text
fhelium/
  config/          CKKS parameters, NTT policies, security assessment
  core/            context identity, exact values, keys, rotation planning
  engine/          CKKS algorithms and engine-owned RNS/NTT implementation
    rns/            chain/layout metadata, parameters, arithmetic runtime
    ntt/            host plans, materialized tables, CPU/CUDA backends
  native/          extension loading, ABI diagnostics, CUDA inspection, wrappers
  rng/             CSPRNG interface and adapter
  serialization/   exact versioned value files
  artifacts/       logical names, current generations, and local repository policy
  distributed/     process state, typed transport, HE collectives
  execution/       signatures, reusable buffers, CUDA Graphs
  residency/       local live-value ownership, admission, plans, and lifetimes
  experimental/    opt-in features whose APIs and behavior can change directly
    bootstrap/      composable CKKS bootstrap mechanisms
      presets/      measured versioned bootstrap compositions
    mpc/            experimental multiparty CKKS arithmetic
    jit/            unified mixed-dialect xDSL programs, passes, and execution
  benchmarks/      public definitions/registry and built-in benchmark runners
csrc/ops/
  rns/             schemas plus CPU/CUDA residue arithmetic
  ntt/             schemas plus CPU/CUDA forward/inverse NTT
  ckks/            schemas plus CPU/CUDA CKKS tensor primitives
  common/          shared tensor/RNS validation and parameters
    cpu/           CPU Montgomery helpers
    cuda/          CUDA Montgomery, repetition, and launch helpers
csrc/runtime/      CUDA device and peer-topology inspection extension
examples/          maintained runnable workflows
tests/             public, state, ABI, execution, distributed tests
packaging/         release matrix, manylinux builders, artifact/index publication
cloudflare/        read-only Python package-index Worker
.github/workflows/ release orchestration
```

## Stable value layer

| Goal | First file(s) |
| --- | --- |
| Context identity | `fhelium/core/context.py` |
| Plaintext representations | `fhelium/core/plaintext.py` |
| Ciphertext components/limbs | `fhelium/core/ciphertext.py` |
| Key layouts and rotation step identity | `fhelium/core/keys.py` |
| Representation-state vocabulary | `fhelium/core/state.py` |
| Tensor movement and value-local byte accounting | `fhelium/core/tensor_resident.py` |

Core values should not acquire engine, process-group, artifact-path, or
application cache ownership.

## Engine and arithmetic layer

| Goal | First file(s) |
| --- | --- |
| Public CKKS operations and validation | `fhelium/engine/ckks_engine.py` |
| Encode/decode | `fhelium/engine/ckks_plaintext_codec.py`, `fhelium/engine/slot_embedding.py` |
| Encryption/decryption | `fhelium/engine/ckks_encryptor.py`, `fhelium/engine/ckks_decryptor.py` |
| Key creation | `fhelium/engine/key_generator.py` |
| Rescale | `fhelium/engine/ckks_rescale.py` |
| Hybrid decomposition | `fhelium/engine/rns/decomposition.py` |
| Key-switch orchestration | `fhelium/engine/hybrid_keyswitch.py` |
| RNS chain/layout/parameters | `fhelium/engine/rns/chain.py`, `fhelium/engine/rns/layout.py`, `fhelium/engine/rns/parameters.py` |
| RNS arithmetic facade | `fhelium/engine/rns/runtime.py` |
| Rotation/Galois mapping | `fhelium/engine/galois.py` |
| Reusable rotation-step planning | `fhelium/core/rotation.py` |
| NTT backend implementations | `fhelium/engine/ntt/` |
| NTT plan/table preparation | `fhelium/engine/ntt/plans/` |
| Host Montgomery constants | `fhelium/engine/rns/montgomery.py` |

## Native ABI and kernels

| Layer | Location |
| --- | --- |
| Torch operator loading/status | `fhelium/native/runtime.py` |
| Shared ABI identity/manifest helpers | `fhelium/native/_abi.py` |
| Compiled Torch operator and ABI manifest | `fhelium/native/torchops/` |
| Python wrappers | `fhelium/native/wrapper/{rns_ops,ntt_ops,ckks_ops}.py` |
| Python CUDA inspection API | `fhelium/native/cuda/__init__.py` |
| Wrapper generator | `scripts/generate_native_wrappers.py` |
| Backend-neutral Torch schemas | `csrc/ops/<family>/*.cpp` |
| CPU dispatcher registrations and implementations | `csrc/ops/<family>/cpu/` |
| CUDA dispatcher registrations and implementations | `csrc/ops/<family>/cuda/` |
| Shared tensor/RNS helpers | `csrc/ops/common/` |
| CUDA runtime inspection | `csrc/runtime/cuda_info.{h,cpp}` |

Invoke wrapper generation directly with
`python scripts/generate_native_wrappers.py`. Do not hand-edit generated
wrapper output without changing the source schema or generator that owns it.

## Distributed and execution

| Goal | Location |
| --- | --- |
| Rank/device/process-group init | `fhelium/distributed/_state.py` |
| Typed descriptors and allocation | `fhelium/distributed/_transfer.py` |
| Collective transport/group primitives | `fhelium/distributed/_collective_common.py` |
| Whole-value collectives | `fhelium/distributed/_value_collectives.py` |
| Limb scatter/gather | `fhelium/distributed/_limb_collectives.py` |
| Ciphertext reduction | `fhelium/distributed/_ciphertext_reduction.py` |
| Private collective aggregation point | `fhelium/distributed/_typed_collectives.py` |
| Exact execution signatures | `fhelium/execution/signature.py` |
| Reusable buffers and copy handles | `fhelium/execution/buffer.py` |
| CUDA Graph program | `fhelium/execution/cuda_graph.py` |
| Composable CKKS bootstrap | `fhelium/experimental/bootstrap/` |
| Versioned bootstrap presets | `fhelium/experimental/bootstrap/presets/` |
| Multiparty CKKS arithmetic | `fhelium/experimental/mpc/` |
| JIT program, capture, passes, readiness, and execution | `fhelium/experimental/jit/` |
| Artifact refs/store | `fhelium/artifacts/` |
| Opaque residency handles, replica rules, and recoverability rules | `fhelium/residency/model.py` |
| Residency transitions, local accounting, and optional budgets | `fhelium/residency/manager.py` |
| Ordered residency plan IR | `fhelium/residency/plan.py` |
| Declarative working-set requirements | `fhelium/residency/request.py` |
| Deterministic automatic policy and controller | `fhelium/residency/policy.py`, `fhelium/residency/controller.py` |
| Leases, holds, and reservations | `fhelium/residency/lease.py` |
| Tensor-free snapshots, explanations, and reports | `fhelium/residency/snapshot.py` |

## Test entry points

Start with the narrowest relevant test, then broaden:

| Validation area | Representative tests |
| --- | --- |
| Value/state semantics | `tests/test_value_representation_invariants.py`, `tests/test_batch_value_model.py` |
| Context identity | `tests/test_context_identity.py` |
| CKKS arithmetic | `tests/test_ckks_operation_correctness.py`, `tests/test_scale_management.py` |
| Composable bootstrap | `tests/test_ckks_bootstrap.py` |
| Native schemas and mutation | `tests/test_native_operator_invariants.py` |
| NTT policy and correctness | `tests/test_ntt_backend.py` |
| Serialization | `tests/test_serialization.py` |
| Artifact repository | `tests/test_artifact_store.py` |
| Tensor-resident values | `tests/test_residency.py` |
| Managed residency handles, transitions, plans, and lifetimes | `tests/test_resource_residency.py` |
| Reusable buffers / graphs | `tests/test_execution_buffer.py`, `tests/test_cuda_graph_execution.py` |
| Typed transport | `tests/test_distributed_transfer.py` |
| Packaged prime catalog | `tests/test_prime_catalog.py` |
| JIT | `tests/test_jit_ir.py`, `tests/test_jit_passes.py`, `tests/test_jit_execution.py`, `tests/test_jit_api.py` |

Use repository test discovery as the final authority because file names can
evolve.

## Recommended reading order

```mermaid
graph LR
    EX[README and examples]
    CORE[core values]
    ENGINE[CkksEngine]
    FLOW[rescale / key switch / NTT]
    TESTS[focused tests]
    NATIVE[csrc and wrappers]
    EX --> CORE --> ENGINE --> FLOW --> TESTS --> NATIVE
```

Read tests before optimizing a native path: they often encode singleton-row,
last-level, mutation, and Q/QP invariants not obvious from a benchmark.
