# Linking and prepared host execution

Backend linking converts a transformed Compilation into a `ProgramExecutable` with resolved implementation calls, bound materials, and prepared Python control flow. Manual Compile and callable Compile execute through this same host mechanism. This article follows preparation through repeated invocation and distinguishes host preparation from device-kernel compilation.

## How is an executable linked?

`OperationBackend.link(compilation)` delegates to the caller-composable `backend_linking_pipeline`. It copies the workspace mapping and material dictionary, removes transient dispatch/link products, and runs a Pipeline that clones the Program. The source Compilation's binding assignments and workspace mapping remain unchanged; referenced Tensor storage and custom objects remain shared.

The standard linking pipeline in `compile/passes/backend/_pipeline.py` runs:

1. `ResolveTensorPlaceholdersPass` to record available Tensor facts.
2. `ResolveBackendOperationsPass` to resolve implementations and produce a `ProgramDispatchTable`.
3. `InitializeResourceBindingsPass` to initialize named non-Tensor handles.
4. `MaterializeResourcesPass`, when a Backend materializer is supplied, to construct missing handles.
5. `LinkProgramPass` to resolve every external reference and construct `ProgramExecutable`.

Resolution consumes the Program's selected operations and assignments. Numerical lowering and key/table preparation belong in the preceding transformation pipeline. A missing or ambiguous implementation stops resolution; unresolved material symbols, absent required handles, or handle-kind mismatches stop linking.

The following helper links an already transformed and supplied Compilation:

```python
from fhelium.backend import OperationBackend

def link_program(compilation, backend=None):
    backend = OperationBackend() if backend is None else backend
    executable = backend.link(compilation)
    return executable, executable.host_source, executable.manifest
```

A supplied `pipeline=` replaces the linking recipe. It must produce a `ProgramExecutable` in the resulting workspace. `backend_linking_pipeline(backend)` exposes the standard sequence for inspection and editing.

## What is prepared once?

`compile/passes/backend/_prepare_host.py` binds material Tensor objects, resource objects, invocation descriptors, and implementation methods into a Python namespace. `compile/passes/codegen/_host.py` emits supported straight-line and structured control flow. Python compilation produces the directly callable host function stored by `ProgramExecutable`.

SSA values become local variables. Calls use preselected implementation methods and fixed operand order. Supported structured regions become Python control flow or prepared region callables. `PreparingOperationImplementation.prepare_operation` can compile a represented fusion region into a complete executable operation during resolution.

`host_source` exposes the generated control flow, and `manifest` identifies resolved operations and resources. The bound Python namespace retains the live objects used by those calls.

## How are public inputs and outputs adapted?

`ProgramExecutable.run(*inputs)` checks the Program arity and adapts declared public values to Tensor payloads. The prepared host function then performs numerical execution. Output adapters rebuild declared public CKKS values from represented result state.

Adapters are cached properties. Input adapters are prepared when first used; output adapters are built after numerical execution on first invocation. Incomplete public output state can therefore fail at execution time even when linking succeeded.

Ordinary Tensor results retain ordinary Tensor ownership. Public ciphertext or plaintext reconstruction uses Program result metadata, while Eager reconstruction uses the called method's metadata transition.

## How long do intermediates live?

For flat blocks, the host emitter computes last uses and releases local references after their final consumer. Externally returned values remain live. Structured regions retain lexically captured values for their scope. Tensor aliases and storage reclamation follow PyTorch ownership and allocator behavior.

PyTorch determines when released storage becomes reusable. Runtime-owned [execution buffers and CUDA Graphs](execution-buffers-and-cuda-graphs.md) provide stable addresses and replay lifetimes when those properties are required.

A material reference binds a live Tensor object. Compatible in-place content updates are visible to the executable; replacing a dictionary entry requires relinking. Non-Tensor handles are similarly retained as the linked live objects.

## How does a compiled callable reuse this work?

`CompiledCallable` in `compile/_callable.py` adds Python argument binding, specialization matching, and output-structure restoration. `compile/_specialization.py` defines invocation signatures, and `compile/_preparation.py` prepares and binds variants.

A compiled variant retains input conditions, source and transformed Compilations, and interface mapping. A linked `Specialization` pairs that variant with a Backend and executable. The matcher compares represented argument metadata and static scalar values. Ciphertext contents and storage addresses do not form the specialization key.

`prepare(*args, **kwargs)` establishes the matching specialization. `on_miss="error"` makes subsequent unprepared input conditions an error; the default miss policy permits preparation. An omitted pipeline uses `default_lower_and_fuse_pipeline`; a supplied Pipeline or signature-to-Pipeline function replaces it. Programs and Compilations skip Python source capture.

`with_backend` relinks the existing transformed variants to another Backend and snapshots that set of variants. Later variants are prepared independently. Relinking does not retune an already transformed Program for different implementation coverage or hardware.

## Which compilation costs remain lazy?

Host preparation can create Triton functions before their GPU binaries have compiled. Generated implementations may prepare a dynamic layout when physical facts first become available. First execution can therefore incur device-code compilation or launch-plan preparation even for an already prepared callable specialization.

Prepare and execute the intended startup workload before steady-state timing or CUDA Graph capture. Report cold preparation separately from warmed execution. Host source generation, Backend kernel generation, and CUDA Graph capture are distinct mechanisms with different products.

`EmitEagerPythonPass` exports public Eager calls; `EmitBackendPythonPass` exports Backend-oriented Python. Backend export and linked execution share host control-flow emission. [Generated kernels and fusion](compiled-execution-and-kernels.md) describes device code, while [Compilation persistence](compilation-persistence.md) describes portable Program and data storage.
