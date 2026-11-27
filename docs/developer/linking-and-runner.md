# Linking, exec schedules, and the Runner

Backend linking converts a transformed Compilation into a `ProgramExecutable` with resolved implementation calls, bound materials, and an `exec` schedule executed by a libfhelium Runner. Manual Compile, callable Compile, and loaded artifacts execute through this Runner mechanism; Eager calls the registered implementations directly. This article follows preparation through repeated invocation and distinguishes host preparation from device-kernel compilation.

## How is an executable linked?

`OperationBackend.link(compilation)` delegates to the caller-composable `backend_linking_pipeline`. It copies the workspace mapping and material dictionary, removes transient dispatch/link products, and runs a Pipeline that clones the Program. The source Compilation's binding assignments and workspace mapping remain unchanged; referenced Tensor storage and custom objects remain shared.

The standard linking pipeline in `compile/passes/backend/_pipeline.py` runs:

1. `ResolveTensorPlaceholdersPass` to record available Tensor facts, including the arguments and results of called functions.
2. `LinkCalledFunctionsPass` to link each function called from `main` once as its own `ProgramExecutable`.
3. `ResolveBackendOperationsPass` to resolve implementations and produce a `ProgramDispatchTable`; a `func.call` dispatches to its callee's executable.
4. `InitializeResourceBindingsPass` to initialize named non-Tensor handles.
5. `MaterializeResourcesPass`, when a Backend materializer is supplied, to construct missing handles.
6. `LinkProgramPass` to resolve every external reference and construct `ProgramExecutable`.

Called functions arise when the lowering pipeline places key switches that differ only in their Galois element into one shared function body. Each call passes its own automorphism gather and sign tables, so the callee's dispatch, kernel preparation and Runner are constructed once for all of its calls.

Resolution consumes the Program's selected operations and assignments. Numerical lowering and key/table preparation belong in the preceding transformation pipeline. A missing or ambiguous implementation stops resolution; unresolved material symbols, absent required handles, or handle-kind mismatches stop linking.

The following helper links an already transformed and supplied Compilation:

```python
from fhelium.backend import OperationBackend

def link_program(compilation, backend=None):
    backend = OperationBackend() if backend is None else backend
    executable = backend.link(compilation)
    return executable, executable.schedule_text, executable.manifest
```

A supplied `pipeline=` replaces the linking recipe. It must produce a `ProgramExecutable` in the resulting workspace. `backend_linking_pipeline(backend)` exposes the standard sequence for inspection and editing.

## What is prepared once?

`compile/passes/backend/_prepare_host.py` lowers the linked Program into an `exec` schedule for one set of input layouts. Each dispatched operation calls its implementation's `lower` with placeholder Tensors and splices the recorded `exec` operations in its place; structured control flow and index arithmetic remain. Input layouts come from the shape and strides of the Program's argument types, and lowering propagates the result layouts of each operation forward. When every input layout is declared, the executable lowers during linking; otherwise it lowers at the first call with each runtime layout and reuses that schedule for later calls with the same layout.

`backend/runner.py` translates the schedule into the builder calls of a `torch.classes.fhelium.Runner` (`csrc/include/fhelium/runner.h`): one buffer per value with its storage class and static layout, host integer and float registers for index arithmetic, and one instruction per operator call, kernel launch, view, allocation, copy, schedule call and structured `scf.for`/`scf.if`. An operator call resolves its dispatcher operator and builds the constant part of its argument stack when the Runner is built; each execution fills only the buffer Tensors and registers. A kernel launch carries the kernel's record extracted from the Triton compilation (device image, launch configuration, runtime parameter types and scratch sizes); the Runner loads the image through the CUDA or HIP runtime and launches it on the current stream of its device. Material Tensors, link-time constants and mutable resource state are bound to their buffers. The Runner is built at the first call or by kernel warming. `PreparingOperationImplementation.prepare_operation` can compile a represented fusion region into one operation whose lowering launches its generated kernels.

Lowering a linked Program records functional operator calls that have a destination-passing overload (`aten::name.out`, or `namespace::name_out` with a written `out` argument for FHElium operators) as an allocation followed by a call of that overload, so that their results can live in the arena described below. An FHElium destination overload may also take a written byte `workspace` for the temporaries of the call; lowering queries its size with `namespace::name_workspace_bytes`, which takes the functional arguments and reads only their sizes, strides, dtypes, devices and option values, and allocates a byte buffer of that size that is live only during the call. The native operation derives the size and the layout of its temporaries inside the workspace from one plan, and its functional form allocates the same workspace itself. Eager fragments record the functional calls.

An `exec.python_call` runs a Python implementation inside the schedule: the Runner pauses its execution at the instruction, the Python driver calls the function with the instruction's operands, and the execution resumes with the returned Tensors. A C++ caller builds and runs only schedules without Python calls.

`schedule_text` prints the `exec` Program of a statically lowered executable, and `manifest` identifies resolved operations and resources.

Linking validates statically bound indexed NTT schedules, including tables consumed inside whole CKKS operations and periodic plaintext preparation. The check follows material references, storage-preserving casts, and fusion inputs. Schedules supplied as runtime inputs or computed values must be prepared by their producer before execution. Their index values and layout remain fixed while the prepared computation is used; violating this precondition is undefined behavior. Twiddle, arithmetic-parameter, and key Tensor contents remain live numerical inputs.

## Build and run an exec schedule in C++

`<fhelium/runner.h>` declares `fhelium::exec::Runner`, which executes an exec schedule: Tensor buffers, int64 and double host registers, and operations that call dispatcher operators, launch compiled device kernels, create views, allocate, copy and zero storage, call other schedules, and branch or loop on host registers. A program links `FHElium::torch`, builds a schedule with the builder methods, calls `finalize()`, and then calls `run(inputs, integers, floats)` any number of times. `finalize` resolves every operator schema through `c10::Dispatcher` and prepares the constant part of each argument stack, so a run fills only buffer Tensors and registers before each boxed call. Material and state buffers are bound once with `bind`; the Runner releases every other buffer when a run returns, apart from the outputs it returns.

```cpp
#include <fhelium/runner.h>

using fhelium::exec::ArgumentKind;
constexpr auto buffer = static_cast<int64_t>(ArgumentKind::buffer);

// y = x + x mod q for residues x and a bound parameter table
auto runner = c10::make_intrusive<fhelium::exec::Runner>();
const auto x = runner->buffer("input", {}, {}, {}, {});
const auto table = runner->buffer("material", {}, {}, {}, {});
const auto y = runner->buffer("internal", {}, {}, {}, {});
runner->bind(table, params);
runner->inputs({x}, {}, {});
runner->outputs({y});
runner->call_op("fhelium_rns_ops::add_standard.default",
                {buffer, buffer, buffer}, {x, x, table}, {y});
runner->finalize();
const auto result = runner->run({residues}, {}, {})[0];
```

A device kernel is registered as a `KernelRecord`: its image (cubin or PTX for CUDA, a code object for HIP), entry name, warp count and size, blocks per program, dynamic shared memory, runtime parameter types, and scratch sizes. The Runner loads the image with the CUDA runtime library API or the HIP module API on its first launch and launches it on the current stream of the selected device. A schedule that contains `python_call` operations runs only from Python, which supplies each call's results; `run` rejects it. Python accesses the same class as `torch.classes.fhelium.Runner`. The repository example [`examples/cpp/runner_schedule.cpp`](https://github.com/VisualDust/fhelium/tree/main/examples/cpp/runner_schedule.cpp) builds a schedule with a host loop and a parity branch over `fhelium_rns_ops` operators and compares its result with direct libfhelium calls.

## How are public inputs and outputs adapted?

`ProgramExecutable.run(*inputs)` checks the Program arity and adapts declared public values to Tensor payloads. The Runner then performs numerical execution. Output adapters rebuild declared public CKKS values from represented result state.

Adapters are cached properties. Input adapters are prepared when first used; output adapters are built after numerical execution on first invocation. Incomplete public output state can therefore fail at execution time even when linking succeeded.

Ordinary Tensor results retain ordinary Tensor ownership. Public ciphertext or plaintext reconstruction uses Program result metadata, while Eager reconstruction uses the called method's metadata transition.

## How long do intermediates live?

`backend/_planning.py` assigns the storage of a linked executable's schedule when its Runner is built. An allocation that is neither returned, carried through a loop or branch, nor returned by an operation that may alias its arguments becomes an *arena buffer*: a fixed view of one region per device that the Runner allocates once and keeps while it lives. Each arena buffer is live from the first to the last instruction that uses it or a view of it; a buffer first used before a loop and used inside it stays live until the loop ends. Buffers are placed largest first at the lowest 256-byte-aligned offset that does not overlap a buffer with an intersecting lifetime, and their allocation and release instructions are removed. The arena therefore holds the largest set of simultaneously live temporaries of the schedule. The Runners of top-level linked schedules on one CUDA device use that device's shared `torch.classes.fhelium.Arena` (`backend/runner.py:shared_arena`), which grows to the largest requirement among them; a growth makes each Runner derive its arena views again and drop its captures at its next execution, and the Arena orders executions that use it across streams. A schedule call's callee runs on a Runner with a private arena, because the caller's arena buffers stay live across the call. An execution that starts while another execution on the shared arena has not returned, through a Python implementation inside a schedule or from another thread, also uses a private-arena Runner of its schedule.

Allocations that reach the schedule results are `output` buffers allocated by each execution, and results of functional operator calls are `internal` buffers that the caching allocator provides. Each of these is released after its last use within its block; the returned outputs remain owned by the caller. Eager fragments do not use an arena, so their temporaries return to the caching allocator after each call. Tensor aliases and storage reclamation follow PyTorch ownership and allocator behavior.

PyTorch determines when released storage becomes reusable. Runtime-owned [execution buffers and CUDA Graphs](execution-buffers-and-cuda-graphs.md) provide stable addresses and replay lifetimes when those properties are required.

A material reference binds a live Tensor object. Compatible in-place numerical content updates are visible to the executable; replacing a dictionary entry requires relinking. Indexed NTT schedules follow the structural lifetime requirement described above. Non-Tensor handles are similarly retained as the linked live objects.

## How does a compiled callable reuse this work?

`CompiledCallable` in `compile/_callable.py` adds Python argument binding, specialization matching, and output-structure restoration. `compile/_specialization.py` defines invocation signatures, and `compile/_preparation.py` prepares and binds variants.

A compiled variant retains input conditions, source and transformed Compilations, and interface mapping. A linked `Specialization` pairs that variant with a Backend and executable. The matcher compares represented argument metadata and static scalar values. Ciphertext contents and storage addresses do not form the specialization key.

A call whose signature has no resident specialization prepares one in the calling thread and then executes it. Preparation captures or loads the Program, transforms it when needed, binds materials, links and warms supported CUDA kernels; the Triton kernels of one executable compile concurrently in a shared thread pool. Preparation errors propagate to the caller.

`prepare(*args, **kwargs)` establishes the matching specialization without executing it. `on_miss="error"` requires a prior matching `prepare`. An omitted pipeline uses the default recipe. A supplied Pipeline or signature-to-Pipeline function replaces the recipe.

`with_backend` snapshots existing transformed variants into a new callable, which links them with the selected Backend when prepared. Later variants are prepared independently. Relinking does not retune an already transformed Program for different implementation coverage or hardware.

## Reuse of selection work within a preparation

Execution selection probes each distinct local signature once. `SelectionDecisions` in `compile/passes/lowering/_select_execution.py` is a workspace entry keyed by the selecting pass configuration, the operation type, its attributes other than `rotation_step`, and the operand types. Operand types carry depth, actual scale and prime rows, so an entry applies to one depth. Later operations with the same key reuse the accepted or preserved decision and are lowered directly; the workspace key `selection_decision_cache=False` disables the entry. `stats()["specializations"][i]["selection_decisions"]` reports hits, misses, entries, key construction seconds and probe seconds.

Fusion matching reads each operation's registered `OperationStructure` (`ir/_operation_structure.py`): operand roles, result layout and scalar component bodies. Structures are memoized by operation type, operand count and attributes. The Triton code generator emits the `same`, `select`, `pack`, `key_product` and `transform` layouts; operations whose layout is `permute` (coefficient automorphism) or `reduce_rows` (ModDown, rescale) remain region boundaries executed by their whole-operation implementations.

## Kernel warming and automatic Graph capture

Callable preparation warms the Triton kernels launched by the executable's lowered schedules, using the launch signatures recorded during lowering, and then builds the schedules' Runners from the compiled kernel images. A warmup failure is retained in stage statistics; a Runner that was not built during warming is built at its first call. Manually linking a Program lowers its schedule and leaves kernel compilation and Runner construction to the first call.

With `graph="auto"` (the default) or `graph=True`, `compile/_graph.py` routes a specialization whose Tensor inputs share one CUDA device through a second Runner of the same schedule built with Graph segments, one for each grad, inference and autocast context. CPU inputs, inputs that require gradients and failures before the first capture continue as prepared calls; `stats()["graphs"]` records state, execution counts, capture costs, segment and replay counts and fallback reasons.

Capture safety is a static property of each schedule instruction. Kernel launches, views, allocations, copies, zeroing and operator calls whose Tensors are on one CUDA device are safe unless the operator carries `torch.Tag.cudagraph_unsafe` or reads device values on the host, such as `aten::item` or `aten::nonzero`; a schedule call is safe when its callee is. Collectives on CUDA buffers are safe; a segment runs directly when its process group cannot be captured. Python calls and CPU operations are unsafe. Planning encloses each maximal run of safe top-level instructions, where a loop or branch counts as one instruction, in a Graph segment. The Runner executes a segment at its first execution, captures it on a side stream at the second, and replays it afterwards. A Runner in mode `auto` (`graph="auto"`) records events around the segment's first direct execution and its next three replays and, at the first execution after they have completed, keeps the Graph if the median replay is shorter than the direct execution; otherwise it releases the segment's captures and executes the segment directly from then on; the choice persists across later recaptures. Buffers that a segment reads from outside it, other than arena, material and state buffers and inputs held in storage owned by a Runner (its arena, output region or captured allocations), are staged into Runner-owned storage before each replay. An input held in Runner-owned storage is read in place; the Runner keeps up to four captures of a segment for different addresses of such inputs, captures a new address at its second occurrence and evicts the least recently replayed capture; host registers that a segment reads are recorded at capture, and a change captures the segments again. Binding-time preparation allocates new material Tensors at its first refresh and copies the results of later refreshes into them, so later refreshes keep the executable, its Runners and their captures; rebinding any other material or state buffer to a different Tensor releases the captures.

With Graph segments, planning places each result allocated once with a static CUDA layout in the Runner's output region (`place_output`), which the Runner allocates at construction. The Runner returns such results, and results held in captured storage, as Tensors whose storage refers to the region without a copy. At the start of its next execution, before any instruction writes the region, it copies each previously returned result whose storage the caller still references into newly allocated memory and redirects that storage to the copy; views follow because they share the storage. Released results cost nothing. `graph_stats` counts these as `detached_outputs`. Allocations made inside segments by operations without a destination-passing form come from one Graph pool per device, shared by the Runners on that device's arena, whose executions never interleave; segments capture on one stream per device. Temporaries that a segment no longer reads are released after its replay; a result read after the segment and held in pool memory is copied into Runner-owned storage after each replay, which `live_out_copies` counts, so no pool memory outlives its Graph. Planning avoids these copies by placing such results in the arena or output region. `private_pool_bytes` reports the pool reservation. `ProgramExecutable.segment_stats` and the `segments` entry of `stats()["graphs"]` report each segment's node count, capture and instantiation host times and variant count; with `fhelium.backend.runner.PROFILE_SEGMENTS` set before the Runners are built, they also report the device times of the latest direct execution and replay. Clearing releases the Graph Runners, their pools and output regions and then the resident specializations. Persistent cache files remain intact.

## Transfer and CPU-task planning

`backend/_overlap.py` rewrites the top-level instructions of a planned schedule without adding or removing a transfer or collective and without changing the device of an operation. Every instruction keeps its order relative to the instructions it depends on, so results equal those of the synchronous schedule bitwise. A `copy` or `aten::_to_copy` between a CPU and a CUDA buffer of one dtype runs on a stream of role `h2d` or `d2h`, which first waits for an event of the compute stream. Device work that writes the device buffer or reads a host-to-device destination waits for the copy's event, and the host waits for it before it accesses the host buffer and before the schedule returns. A transfer is issued after the writers of its source and the earlier accesses of its destination, together with the allocation of its destination, at the earliest such position at which the device memory the schedule needs (its arena plus the other buffers it allocates) exceeds that of the schedule without moved transfers by at most `TRANSFER_BUDGET`, by default the size of the schedule's largest transfer. A device-to-host transfer whose source is in the arena delays the reuse of that arena range until the copy completes, within the same limit. A collective on buffers of one CUDA device runs on a stream of role `comm` in the same way, and Graph segments end at the stream instructions around it. Host buffers that the schedule allocates are page-locked; a caller-owned pageable buffer stays correct because CUDA stages it within the copy call, and its transfer overlaps only device work already queued.

A run of CPU instructions becomes a task on the process's CPU lane, one persistent worker thread shared by all schedules, which runs tasks in issue order and starts each task after the events of the transfers producing its inputs. Host-to-device transfers of a task's results are issued at the end of the task, and the host thread joins the task before it accesses a buffer the task accesses and before the schedule returns. CPU operators in a task use the configured intra-op thread count; the persistent thread keeps its OpenMP team warm, so a task runs as fast as the same instructions on the calling thread. Loops and branches are planned as single instructions with synchronous transfers. `OverlapReport` records what planning changed, and `ProgramExecutable.overlap_stats()` reports `moved_transfers`, `h2d_transfers`, `d2h_transfers`, `stream_collectives`, `planned_tasks`, and the accumulated `tasks_run`, `task_seconds`, `join_wait_seconds`, and `event_waits`. Streams, events, and tasks are recorded in exported entries.

## Preparation and execution statistics

`stats()` reports per-specialization host durations for source capture, pipeline execution, material binding, linking, cache load/store and kernel warming where those stages ran. `PassReport.seconds` measures individual pass invocations and their configured structural verification. Instance-cache status distinguishes hits, stored entries, misses, disabled persistence and source/material ineligibility. `resident_cache_hits` counts reuse of linked specializations; `selection_decisions` counts reused selection probes.

Execution timing separates host dispatch from completed work. CPU calls complete on return. For inputs identifying one CUDA device, events measure the caller’s CUDA stream, including work joined onto that stream; completed events are collected without a device synchronization. Independent streams and devices reachable only through closures lie outside that scope. Unsupported multi-device completion remains unavailable rather than being estimated from launch time. Use synchronized external timing for application-wide latency.

`timing` separates compiled and steady compiled observations. The steady group excludes each specialization’s first two compiled calls, covering initial prepared execution and, when enabled, the automatic-capture transition. `prepare_seconds` includes source identity, capture, template construction and instantiation, pipeline, binding, linking, cache read/write and kernel warming; `graph_capture_seconds` records the additional Graph construction cost.

`EmitEagerPythonPass` exports public Eager calls; `EmitBackendPythonPass` exports Program-level Python in which each operation call passes through a `LoweredCall`, which lowers the implementation for the received payload layouts and runs the fragment on a Runner. [Generated kernels and fusion](compiled-execution-and-kernels.md) describes device code, while [Program files and the compile cache](compilation-persistence.md) describes portable Program and data storage.

Backend-exported Python consumes caller-prepared materials. When supplying custom indexed NTT tables, call `ntt_ops.validate_indexed_schedule(even_indices, odd_indices, N)` during material preparation, before invoking the generated function. The generated function reads the supplied tensors without repeating schedule validation.
