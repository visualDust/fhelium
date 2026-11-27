# API reference

The API reference covers the Python package and the C++ interface of `libfhelium`, the native library that ships inside it. The [C++ reference](cpp/index.md) documents the public headers.

## Python

The Python reference covers the public module tree: modules whose path contains no component starting with an underscore. Names that a package publishes through `__all__` are rendered on that package's page, including names defined in private implementation modules. The reference is generated from the current checkout without importing FHElium or compiling CUDA code; signatures, docstrings, source links, module pages, and sidebar entries come from the same abstract-syntax-tree scan.

## Choose by task

| Task | Start with | Related guidance |
| --- | --- | --- |
| Configure CKKS parameters | [`fhelium.config.ckks`](fhelium/config/ckks.md) and [`fhelium.config.ntt`](fhelium/config/ntt.md) | [Choose a preset and depth](../how-to/choose-preset-and-depth.md) |
| Assess a supported security-table row | [`fhelium.config.security`](fhelium/config/security.md) | [Security](../developer/security.md) |
| Create and use an eager engine | [`fhelium.eager`](fhelium/eager.md) | [Basic CKKS workflow](../tutorial/basic-ckks-workflow.md) |
| Control scale and depth with separate operations | [`fhelium.eager`](fhelium/eager.md) | [Value-state transitions](../concepts/ckks/value-state-transitions.md) |
| Inspect typed values and keys | [`fhelium.values.ciphertext`](fhelium/values/ciphertext.md), [`fhelium.values.plaintext`](fhelium/values/plaintext.md), and [`fhelium.values.keys`](fhelium/values/keys.md) | [Value model and identity](../concepts/ckks/value-model-and-identity.md) |
| Persist a value | [`fhelium.serialization.value`](fhelium/serialization/value.md) | [Persistence and artifacts](../concepts/execution/serialization-and-artifacts.md) |
| Compile, export, and load a callable | [`fhelium.compile`](fhelium/compile.md) | [Compiled execution](../concepts/execution/compiled-execution.md) |
| Stream inputs or capture Eager code | [`fhelium.runtime.buffer`](fhelium/runtime/buffer.md) and [`fhelium.runtime.cuda_graph`](fhelium/runtime/cuda_graph.md) | [Value signatures and buffers](../concepts/execution/signatures-and-buffers.md) |
| Inspect CPU/CUDA topology and current host/CUDA memory | [`fhelium.runtime.topology`](fhelium/runtime/topology.md) and [`fhelium.runtime.memory`](fhelium/runtime/memory.md) | [Inspect runtime, memory, and CUDA topology](../how-to/inspect-runtime-and-cuda.md) |
| Manage local value placement, admission, lifetimes, plans, and deterministic automation | [`fhelium.residency.manager`](fhelium/residency/manager.md), [`model`](fhelium/residency/model.md), [`plan`](fhelium/residency/plan.md), [`request`](fhelium/residency/request.md), and [`controller`](fhelium/residency/controller.md) | [Residency lifetimes](../concepts/execution/residency-lifetimes.md) |
| Build and execute IR-based CKKS programs | [`fhelium.ir`](fhelium/ir.md), [`fhelium.compile`](fhelium/compile.md), [`compile.passes.lowering`](fhelium/compile/passes/lowering.md), and [`fhelium.backend`](fhelium/backend.md) | [Open compiler stack](../concepts/open-compiler-stack.md) |
| Coordinate processes and collectives | [`fhelium.distributed`](fhelium/distributed.md) | [SPMD model](../concepts/distributed/spmd-model.md) |
| Inspect native-extension availability and ABI diagnostics | [`fhelium.native`](fhelium/native.md) | [Inspect runtime, memory, and CUDA topology](../how-to/inspect-runtime-and-cuda.md) |
| Inspect CUDA devices and peer topology programmatically | [`fhelium.native.cuda`](fhelium/native/cuda.md) | [Inspect runtime, memory, and CUDA topology](../how-to/inspect-runtime-and-cuda.md) |
| Refresh ciphertexts by bootstrapping | [`fhelium.bootstrap`](fhelium/bootstrap.md) and [`fhelium.algorithms`](fhelium/algorithms.md) | [Bootstrapping semantics and range requirements](../concepts/ckks/composable-bootstrapping.md) |
| Use experimental multiparty CKKS | [`fhelium.experimental.mpc`](fhelium/experimental/mpc.md) | [Multiparty supported security scope](../how-to/use-multiparty-ckks.md) |

## Generation rule

`scripts/generate_api_docs.py` generates one page per public Python module. A module whose path contains a component starting with an underscore has no page; its public names appear on the page of the package that exports them. Package initializers with a defined `__all__` are included; other package initializers are omitted.

For each discovered module, a defined `__all__` defines the members when it is present. Otherwise, the generator includes non-underscored classes, functions, and data definitions. Module-path components and page titles are kept as they occur in Python.

The sidebar follows the package tree. Each entry shows the last component of its module name, and the page title shows the complete name. Pages under `/api/cpp/` use a separate sidebar of C++ headers. Adding, removing, or renaming a source module changes the generated pages and sidebar on the next build. The API index is reached from the top navigation and is not repeated as an “Overview” sidebar item.

The configured module-tree exclusions are `fhelium.native.wrapper`, which contains generated Torch-operator bindings, and the synthetic and standalone benchmark entry modules under `fhelium.benchmarks.synthetic` and `fhelium.benchmarks.standalone` listed in `EXCLUDED_API_MODULE_PREFIXES`.
