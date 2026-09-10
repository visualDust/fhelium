# API reference

The API reference follows the non-private Python module tree. It is generated
from the current checkout without importing FHElium or compiling CUDA code.
Signatures, docstrings, source links, module pages, and sidebar entries all
come from the same abstract-syntax-tree scan. Some pages document
implementation modules for contributors; supported import surfaces are the
package initializers and modules that publish interfaces through ``__all__``.

## Choose by task

| Task | Start with | Related guidance |
| --- | --- | --- |
| Configure CKKS parameters | [`fhelium.config.ckks`](fhelium/config/ckks.md) and [`fhelium.config.ntt`](fhelium/config/ntt.md) | [Choose a preset and depth](../how-to/choose-preset-and-depth.md) |
| Assess a supported security-table row | [`fhelium.config.security`](fhelium/config/security.md) | [Security scope](../tutorial/support-and-security.md) |
| Create and use an eager engine | [`fhelium.eager`](fhelium/eager.md) | [Quickstart](../tutorial/tutorials.md) |
| Control scale and depth with separate operations | [`fhelium.eager`](fhelium/eager.md) | [Scale and depth lifecycle](../concepts/ckks/scale-and-depth-lifecycle.md) |
| Inspect typed values and keys | [`fhelium.values.ciphertext`](fhelium/values/ciphertext.md), [`fhelium.values.plaintext`](fhelium/values/plaintext.md), and [`fhelium.values.keys`](fhelium/values/keys.md) | [Value model and identity](../concepts/ckks/value-model-and-identity.md) |
| Persist a value | [`fhelium.serialization.value`](fhelium/serialization/value.md) | [Serialization and artifacts](../concepts/execution/serialization-and-artifacts.md) |
| Capture or reuse repeated work | [`fhelium.runtime.cuda_graph`](fhelium/runtime/cuda_graph.md) and [`fhelium.runtime.buffer`](fhelium/runtime/buffer.md) | [Execution concepts](../concepts/execution/cuda-graph-model.md) |
| Inspect CPU/CUDA topology and current host/CUDA memory | [`fhelium.runtime.topology`](fhelium/runtime/topology.md) and [`fhelium.runtime.memory`](fhelium/runtime/memory.md) | [Inspect runtime, memory, and CUDA topology](../how-to/inspect-runtime-and-cuda.md) |
| Manage local value placement, admission, lifetimes, plans, and deterministic automation | [`fhelium.residency.manager`](fhelium/residency/manager.md), [`model`](fhelium/residency/model.md), [`plan`](fhelium/residency/plan.md), [`request`](fhelium/residency/request.md), and [`controller`](fhelium/residency/controller.md) | [Residency lifetimes](../concepts/execution/residency-lifetimes.md) |
| Build and execute IR-based CKKS programs | [`fhelium.ir`](fhelium/ir.md), [`fhelium.compile`](fhelium/compile.md), [`compile.passes.lowering`](fhelium/compile/passes/lowering.md), and [`fhelium.backend`](fhelium/backend.md) | [Open compiler stack](../concepts/open-compiler-stack.md) |
| Coordinate processes and collectives | [`fhelium.distributed`](fhelium/distributed.md) | [SPMD model](../concepts/distributed/spmd-model.md) |
| Inspect native-extension availability and ABI diagnostics | [`fhelium.native`](fhelium/native.md) | [Inspect runtime, memory, and CUDA topology](../how-to/inspect-runtime-and-cuda.md) |
| Inspect CUDA devices and peer topology programmatically | [`fhelium.native.cuda`](fhelium/native/cuda.md) | [Inspect runtime, memory, and CUDA topology](../how-to/inspect-runtime-and-cuda.md) |
| Use experimental CKKS facilities | [`fhelium.experimental.bootstrap`](fhelium/experimental/bootstrap.md), [`fhelium.experimental.mpc`](fhelium/experimental/mpc.md), and [`fhelium.experimental.jit`](fhelium/experimental/jit.md) | [Bootstrapping semantics and range requirements](../concepts/ckks/composable-bootstrapping.md) and [multiparty supported security scope](../how-to/use-multiparty-ckks.md) |

## Generation rule

`scripts/generate_api_docs.py` discovers Python source modules regardless of
whether a module-path component starts with an underscore. Package initializers
with a defined `__all__` are included; other package initializers are omitted.

For each discovered module, a defined `__all__` defines the members when it
is present. Otherwise, the generator includes non-underscored classes,
functions, and data definitions. Module-path components and page titles are
kept as they occur in Python.

Sidebar grouping is mechanical: modules are grouped by their first package
segment, while every item retains its complete Python module name. Adding,
removing, or renaming a source module changes the generated pages and sidebar
on the next build. The API index is reached from the top navigation and is not
repeated as an “Overview” sidebar item.

The sole configured module-tree exclusion is `fhelium.native.wrapper`, which
contains generated Torch-operator bindings.
