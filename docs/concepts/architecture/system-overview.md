# Architecture

FHElium is a full-stack framework for research on Cheon–Kim–Kim–Song (CKKS) approximate-number homomorphic encryption. Its architecture connects public value semantics, two usage models, shared operation definitions, execution backends, runtime mechanisms, and native kernels. Each layer owns a specific class of decisions and exposes the information needed by adjacent layers.

## System structure

```mermaid
flowchart TB
    APP[Application or research system]

    subgraph SYSTEMS[System composition]
        direction TB
        BUFFERS[Runtime buffers<br/>and DoubleBuffer]
        RESIDENCY[Residency<br/>between calls]
        DIST[Process groups and<br/>compiled collectives]
        STORE[Persistence<br/>and artifacts]
    end

    subgraph SEM[Semantics]
        direction TB
        VALUES[Public values<br/>and CKKS state]
        OPS[Registered operation<br/>semantics]
    end

    subgraph USAGE[Usage models]
        direction TB
        EAGER[Eager<br/>operation call]
        MANUAL[Manual Compilation<br/>and Pipeline]
        CALLABLE[Callable capture<br/>and specialization]
        CACHE[(Specialization cache<br/>resident or on disk<br/>artifact format)]
        CALLABLE <--> CACHE
    end

    subgraph PIPE[Compile pipeline]
        direction LR
        PROGRAM[Program]
        PASSES[Analysis and<br/>transformation passes]
        SELECT[Implementation<br/>selection and lowering<br/>whole operation, RNS/NTT,<br/>fusion regions]
        SCHED[Exec schedule<br/>operator calls, kernel<br/>launches, copies, collectives,<br/>buffers, streams]
        PROGRAM --> PASSES --> SELECT --> SCHED
    end

    subgraph EXECUTION[Execution]
        direction LR
        DISPATCH[Eager dispatch to<br/>registered implementations]
        RUNNER[libfhelium Runner<br/>arena, CUDA Graph segments,<br/>transfer overlap, CPU lane]
        LOADED[fc.load in Python<br/>fhelium::Artifact in C++]
        LOADED --> RUNNER
    end

    subgraph BACKEND[Backend implementations]
        direction LR
        RESOURCES[Supplied Tensor operands,<br/>tables, execution handles]
        COMPOSED[Lower-level<br/>operations]
        WHOLE[Whole<br/>operations]
        GENERATED[Generated<br/>kernels]
        RESOURCES --> COMPOSED
        RESOURCES --> WHOLE
        RESOURCES --> GENERATED
    end

    subgraph NATIVE[Native operator layer]
        direction LR
        SCHEMAS[Typed PyTorch operator<br/>schemas and dispatcher<br/>libfhelium_torch]
        CPU[CPU native<br/>operators]
        GPU[CUDA native<br/>operators]
        DEVICECODE[Generated device<br/>kernels, Triton cubin]
        SCHEMAS --> CPU
        SCHEMAS --> GPU
    end

    SYSTEMS -. application composition .-> APP
    APP --> USAGE
    SEM -. defines .-> USAGE
    USAGE -->|cache miss or manual Program| PIPE
    USAGE -->|Eager call or cache hit| EXECUTION
    PIPE -->|exec schedule, store, export| EXECUTION
    BACKEND -. lowered into .-> PIPE
    EXECUTION -->|Eager dispatch| BACKEND
    EXECUTION -->|Runner call_op and launch| NATIVE
    BACKEND --> NATIVE
```

Semantics define what every route computes. The usage models decide when CKKS state and scheduling are fixed: an Eager call executes immediately, and a compiled callable consults its specialization cache before entering the Compile pipeline. The pipeline lowers selected Backend implementations into an exec schedule, which the Runner executes in the current process or, after export, in another Python or C++ process. Eager dispatch and the Runner both reach the typed native operators; the Runner also launches generated device kernels. Runtime buffers, Residency, process groups, and persistence compose with these layers through the application.

## Semantic and program layers

A public `Ciphertext`, `Plaintext`, `CompressedPlaintext`, or key combines Tensor storage with the state needed to interpret it. A sparse compressed plaintext retains both compact and implicit Tensor data. Depending on the value type, this state includes depth, actual scale, active prime identities, polynomial domain, modulus basis, residue representation, component count, batch shape, key role, and device placement.

Registered operations define their input requirements, result types, and state effects. Selected implementations declare the numerical operands and non-Tensor execution handles they consume. These definitions establish operation meaning across every execution route.

Applications enter this layer through two usage models. **Eager** applies one operation and its CKKS state transition immediately. **Compile** represents operations and values in a Program, where selected passes can analyze, schedule, preserve, or lower them before Backend linking produces an exec schedule. Manual Program/Compilation/Pipeline execution and callable specialization use the same Compile machinery. A compiled callable additionally keys each specialization by its call signature and keeps it resident and, for a cacheable source, in an on-disk entry stored in the artifact format; a hit runs the stored schedule without capture or linking. Both usage models retain the meaning declared by the registered operation catalog.

## Backend composition layer

The **Backend** owns registered operation implementations and a workspace of named non-Tensor execution handles. Compile passes select implementations, prepare their numerical operands, and link the Program through the Backend's execution interface.

A Program can retain a whole CKKS operation, expose lower-level RNS/NTT operations, or group compatible operations into a fusion region. Whole implementations can compose the same RNS/NTT algorithms used by exposed operations. Generated implementations consume selected regions and produce device kernels or sequences combining generated and native stages.

Keys, arithmetic parameters, transform tables, and rounding state are Tensor operands supplied through `Compilation.material_bindings` for Compile or direct arguments for Eager. Process groups and encryption samplers are non-Tensor handles supplied through Backend resource bindings. These inputs preserve their respective data and execution lifetimes.

## Execution layer

Eager resolves and caches one device-local call at a time and invokes the registered implementation directly. Compile lowers the selected implementations into an exec schedule: operator calls, generated-kernel launches, views and copies, collectives, schedule calls, and host control flow over buffers with assigned storage and streams. The libfhelium Runner executes the schedule from native code. It places temporaries in a per-device arena, captures capture-safe segments as CUDA Graphs on CUDA, overlaps host-device transfers with computation, and runs consecutive CPU instructions as tasks on a CPU worker lane. Manual execution, compiled callables, `fhelium.compile.load` in Python, and `fhelium::Artifact` in C++ run schedules through this Runner; an exported artifact stores the linked schedules so that another process runs them without capture or linking.

## Native CPU and GPU layers

Typed PyTorch operator schemas define the native execution interface. CPU and CUDA register separate implementations for those schemas, and PyTorch selects the implementation from Tensor placement. The two native backends can use different kernel structures, parallel execution strategies, and microarchitecture-specific optimizations while preserving the schema's Tensor, mutation, and resource contract.

The schemas and the Runner are built into `libfhelium_torch`. Generated implementations are compiled by Triton into device binaries (cubin on CUDA), which the Runner launches directly. The native layer uses PyTorch's allocator, CPU execution runtime, CUDA streams, device dispatch, and profiling tools. Additional execution providers can register Backend implementations and connect their own kernels through the same operation and resource interfaces.

## Runtime and system composition

The operation path composes with several system owners:

- `fhelium.runtime` supplies hardware and memory observations, reusable and double-buffered device storage, and manual CUDA Graph capture;
- `fhelium.compile` also supplies callable specialization, the persistent compile cache, and export and loading of compiled artifacts;
- `fhelium.distributed` supplies process setup, typed value transport, and CKKS-aware collectives for rank-local execution;
- `fhelium.residency` owns live materializations, byte accounting, placement admission, and asynchronous lifetimes;
- serialization preserves public values or Program text with selected Tensor bindings, and `fhelium.store` manages named generations of those persistent representations.

Applications compose these mechanisms around Eager calls and compiled calls: buffers stage the next request while the current one runs, Residency holds keys and weights between calls, process groups carry the collectives compiled into rank-local schedules, and artifacts carry schedules to other processes. They select devices, keys, placement policies, distributed schedules, and persistence behavior according to the workload.

## Ownership map

| Owner | Architectural responsibility |
| --- | --- |
| `fhelium.values` | Public payload-bearing values, represented CKKS state, and keys |
| `fhelium.eager` | Immediate operation transitions, key inventory, and device-local direct dispatch |
| `fhelium.ir` | Program structure, registered operation semantics, types, and analyses |
| `fhelium.compile` | Source capture, Compilation materials/workspaces, passes, lowering, implementation selection, preparation, linking, the compile cache, and export and loading of compiled artifacts |
| `fhelium.backend` | Implementation registry, numerical execution and kernel generation, non-Tensor resource workspace, exec schedules, and the Runner with its arena and Graph segments |
| `fhelium.native` | Native extension loading, application binary interface (ABI) diagnostics, typed wrappers, and device inspection |
| `fhelium.runtime` | Hardware and memory observations, reusable and double-buffered device storage, and manual CUDA Graph capture |
| `fhelium.compile.CompiledCallable` | Input specialization, preparation and executable reuse over the Compile pipeline |
| `fhelium.distributed` | Process setup, typed transport, and CKKS-aware collectives |
| `fhelium.residency` | Live-value ownership, byte accounting, placement admission, and lifetimes |
| `fhelium.serialization` and `fhelium.store` | Durable value and Compilation representation, logical names, and generation management |

Applications compose these owners by selecting passes, implementations, resources, devices, keys, placement policies, and distributed schedules. This keeps research choices visible at the layer where they are made and allows their effects to be measured across the complete workload.

## Related concepts

- CKKS values: [Value model and identity](../ckks/value-model-and-identity.md)
- Operation effects: [Evaluator operation transitions](../ckks/evaluator-operation-transitions.md)
- Program representation: [Neutral IR programs](../neutral-ir-programs.md)
- Compile lifecycle: [Open compiler stack](../open-compiler-stack.md)
- Runtime ownership: [Ownership and responsibilities](ownership-and-responsibilities.md)
- Execution lifetimes: [Signatures and buffers](../execution/signatures-and-buffers.md)
