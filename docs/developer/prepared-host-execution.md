# Linking and prepared host execution

Backend linking converts a transformed Compilation into a `ProgramExecutable` with resolved implementation calls, bound materials, and prepared Python control flow. Manual Compile and callable Compile execute through this same host mechanism. This article follows preparation through repeated invocation and distinguishes host preparation from device-kernel compilation.

## How is an executable linked?

`OperationBackend.link(compilation)` delegates to the caller-composable `backend_linking_pipeline`. It copies the workspace mapping and material dictionary, removes transient dispatch/link products, and runs a Pipeline that clones the Program. The source Compilation's binding assignments and workspace mapping remain unchanged; referenced Tensor storage and custom objects remain shared.

The standard linking pipeline in `compile/passes/backend/_pipeline.py` runs:

1. `ResolveTensorPlaceholdersPass` to record available Tensor facts, including the arguments and results of called functions.
2. `LinkCalledFunctionsPass` to link each function called from `main` once as its own `ProgramExecutable`.
3. `ResolveBackendOperationsPass` to resolve implementations and produce a `ProgramDispatchTable`; a `func.call` dispatches to its callee's executable.
4. `InitializeResourceBindingsPass` to initialize named non-Tensor handles.
5. `MaterializeResourcesPass`, when a Backend materializer is supplied, to construct missing handles.
6. `LinkProgramPass` to resolve every external reference and construct `ProgramExecutable`.

Called functions arise when the lowering pipeline places key switches that differ only in their Galois element into one shared function body. Each call passes its own automorphism gather and sign tables, so the callee's dispatch, kernel preparation and host code are constructed once for all of its calls.

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

Linking validates statically bound indexed NTT schedules, including tables consumed inside whole CKKS operations and periodic plaintext preparation. The check follows material references, storage-preserving casts, and fusion inputs. Schedules supplied as runtime inputs or computed values must be prepared by their producer before execution. Their index values and layout remain fixed while the prepared computation is used; violating this precondition is undefined behavior. Twiddle, arithmetic-parameter, and key Tensor contents remain live numerical inputs.

## How are public inputs and outputs adapted?

`ProgramExecutable.run(*inputs)` checks the Program arity and adapts declared public values to Tensor payloads. The prepared host function then performs numerical execution. Output adapters rebuild declared public CKKS values from represented result state.

Adapters are cached properties. Input adapters are prepared when first used; output adapters are built after numerical execution on first invocation. Incomplete public output state can therefore fail at execution time even when linking succeeded.

Ordinary Tensor results retain ordinary Tensor ownership. Public ciphertext or plaintext reconstruction uses Program result metadata, while Eager reconstruction uses the called method's metadata transition.

## How long do intermediates live?

For flat blocks, the host emitter computes last uses and releases local references after their final consumer. Externally returned values remain live. Structured regions retain lexically captured values for their scope. Tensor aliases and storage reclamation follow PyTorch ownership and allocator behavior.

PyTorch determines when released storage becomes reusable. Runtime-owned [execution buffers and CUDA Graphs](execution-buffers-and-cuda-graphs.md) provide stable addresses and replay lifetimes when those properties are required.

A material reference binds a live Tensor object. Compatible in-place numerical content updates are visible to the executable; replacing a dictionary entry requires relinking. Indexed NTT schedules follow the structural lifetime requirement described above. Non-Tensor handles are similarly retained as the linked live objects.

## How does a compiled callable reuse this work?

`CompiledCallable` in `compile/_callable.py` adds Python argument binding, specialization matching, and output-structure restoration. `compile/_specialization.py` defines invocation signatures, and `compile/_preparation.py` prepares and binds variants.

A compiled variant retains input conditions, source and transformed Compilations, and interface mapping. A linked `Specialization` pairs that variant with a Backend and executable. The matcher compares represented argument metadata and static scalar values. Ciphertext contents and storage addresses do not form the specialization key.

A call whose signature has no resident specialization prepares one in the calling thread and then executes it. Preparation captures or loads the Program, transforms it when needed, binds materials, links and warms supported CUDA kernels; the Triton kernels of one executable compile concurrently in a shared thread pool. Preparation errors propagate to the caller.

`prepare(*args, **kwargs)` establishes the matching specialization without executing it. `on_miss="error"` requires a prior matching `prepare`. An omitted pipeline uses the default recipe, with source-lowering passes omitted for state-instantiated CKKS templates. Semantic templates retain full instance scheduling. A supplied Pipeline or signature-to-Pipeline function replaces the recipe.

`with_backend` snapshots existing transformed variants into a new callable, which links them with the selected Backend when prepared. Later variants are prepared independently. Relinking does not retune an already transformed Program for different implementation coverage or hardware.

## Structural instantiation and guarded selection plans

A new call signature first checks the optimized-instance cache. On a miss, preparation looks for a compatible source-family template in process memory or the structural disk cache. `compile/_template.py` clones its high-level Program and applies registered operation-state rules to current inputs and material owners. The accompanying provenance description classifies facts as `Known`, `InputField`, `Derived` and `Unknown`. Python metadata reads are guarded against their capture-time values. A guard miss or uncovered operation returns to ordinary capture; it does not make the underlying callable unsupported.

Templates cover CKKS arithmetic and scalar operations, NTT and residue transitions, depth operations, key operations and modulus raising, along with supported static Python loops and helper calls. Registered Tensor calls and encrypted semantic arithmetic and rotation also participate. Codec-specific paths, grouped-RNS operations and arbitrary replacements need ordinary capture when their state rules are unavailable. Source identity eligibility remains narrower than frontend capture support.

A CKKS instance uses `instantiated_ckks_pipeline`, which removes source semantic/logical lowering and transition insertion already represented by template state evaluation. Semantic templates use the full default recipe. Current material requirements, state checks, common-subexpression elimination, dead-value elimination and implementation selection remain instance work.

`compile/_selection_plan.py` stores decisions rather than prior operation objects. Built-in lowering selection uses per-operation guards; fusion uses a guard over the complete lowered block, including order, edges, escaping values, effects and matcher-visible facts. Applying a plan reconstructs the chosen lowering or fusion region from the current graph and current types. Custom policies outside these guard contracts use ordinary selection.

Arithmetic and NTT guards permit supported prime-row variation; topology-sensitive key operations retain the concrete facts their lowerings consume. A changed key or digit layout can miss locally while neighboring arithmetic decisions still hit. Selection and fusion hit/miss counts and a reason are retained in specialization statistics. [Compilation persistence](compilation-persistence.md) defines the source-family storage and material-rebinding scope.

### Local recipe and current-window facts reuse

`stats()["specializations"][i]["local_plans"]` reports local matcher analysis performed during that specialization’s preparation. The current consumer describes one operation and its directly connected operand/result ports, including operand sharing. It reuses a matcher recipe for that local structure and fixed facts, then evaluates the recipe against current values. Python capture and lowering origins can identify candidates; they do not establish semantic equivalence by themselves.

| Counter | Counting unit |
| --- | --- |
| `occurrences` | Requests for a local recipe over one operation and its direct ports, accumulated across participating pass scopes; an operation can contribute more than once |
| `recipe_hits` / `recipe_misses` | Recipe lookups satisfied by the pass-local or source-family store, versus lookups requiring a new recipe |
| `facts_hits` / `facts_misses` | Repeated queries satisfied by current-operation facts, versus first evaluations in a read-only graph-analysis window |

Facts include the result of analyzing an unsupported operation. Multiple probes or matchers can query the same operation in one window, so facts counts can exceed recipe counts. A new window evaluates its own current facts; evaluated SSA values and layouts are discarded before graph rewriting. `guard_bypasses` records recipe requests that use ordinary analysis outside the local guard contract.

Global candidate scheduling and final fusion eligibility checks continue when using local hits. The whole-block fusion plan remains a separate reuse decision. Local counters describe work reused at their respective granularities; compare preparation-stage timings and `PassReport.seconds` to establish a time saving. A resident invocation retains its specialization’s preparation record rather than performing those analyses again.

## Kernel warming and automatic Graph capture

Callable preparation warms supported Triton kernels from the linked executable and physical input facts. A warmup failure is retained in stage statistics; operations outside the warming coverage can still initialize during execution. Manually linking a Program prepares its host function and leaves device-kernel initialization to the caller’s execution workflow.

Callable execution defaults to `graph=False`. `compile/_graph.py` manages capture when the caller selects `graph="auto"` or `graph=True`. One prepared invocation establishes the result structure and eligibility. The next compatible invocation attempts capture on the caller’s thread and execution context. CPU inputs, autograd outputs and capture failures continue as prepared calls; `stats()["graphs"]` records state, execution counts, capture costs and fallback reasons. `graph=False` keeps the prepared route.

Each callable owns device-local shared Graph pools. Replays return owned copies by default; `copy_output=False` exposes storage valid until the corresponding Graph replays or the callable is cleared. Clearing closes Graphs, then closes pools and releases resident specializations. Persistent cache files remain intact.

## Preparation and execution statistics

`stats()` reports per-specialization host durations for source capture, pipeline execution, material binding, linking, cache load/store and kernel warming where those stages ran. `PassReport.seconds` measures individual pass invocations and their configured structural verification. Instance-cache status distinguishes hits, stored entries, misses, disabled persistence and source/material ineligibility. Structural observations separately report hit, guard miss or unsupported capture, the memory/disk lookup layer, and template instantiation cost. `resident_cache_hits` counts reuse of linked specializations; selection-plan observations count lowering and fusion reuse.

Execution timing separates host dispatch from completed work. CPU calls complete on return. For inputs identifying one CUDA device, events measure the caller’s CUDA stream, including work joined onto that stream; completed events are collected without a device synchronization. Independent streams and devices reachable only through closures lie outside that scope. Unsupported multi-device completion remains unavailable rather than being estimated from launch time. Use synchronized external timing for application-wide latency.

`timing` separates compiled and steady compiled observations. The steady group excludes each specialization’s first two compiled calls, covering initial prepared execution and, when enabled, the automatic-capture transition. `prepare_seconds` includes source identity, capture, template construction and instantiation, pipeline, binding, linking, cache read/write and kernel warming; `graph_capture_seconds` records the additional Graph construction cost.

`EmitEagerPythonPass` exports public Eager calls; `EmitBackendPythonPass` exports Backend-oriented Python. Backend export and linked execution share host control-flow emission. [Generated kernels and fusion](compiled-execution-and-kernels.md) describes device code, while [Compilation persistence](compilation-persistence.md) describes portable Program and data storage.

Backend-exported Python consumes caller-prepared materials. When supplying custom indexed NTT tables, call `ntt_ops.validate_indexed_schedule(even_indices, odd_indices, N)` during material preparation, before invoking the generated function. The generated function reads the supplied tensors without repeating schedule validation.
