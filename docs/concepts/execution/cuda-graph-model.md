# CUDA Graph execution model

`CudaGraphProgram` adapts an ordinary deterministic rank-local `CkksEngine`
callable for fixed-address capture and replay while preserving its CKKS
semantics. Capture covers that callable's fixed schedule, buffers, and
statically bound resources.

## Static and dynamic state

```mermaid
graph TB
    F[ordinary evaluator callable]
    STATIC[static closure state<br/>engine, keys, weights, schedule]
    DYNAMIC[dynamic positional inputs]
    CAP[Capture]
    BUFFER[stable input buffer]
    GRAPH[torch.cuda.CUDAGraph]
    OUTPUT[retained borrowed output]

    STATIC --> F
    DYNAMIC --> CAP
    F --> CAP
    CAP --> BUFFER
    CAP --> GRAPH
    GRAPH --> OUTPUT
```

A good capture candidate has:

- fixed operation sequence and control flow;
- fixed tensor shapes and CKKS states;
- keys and operation-ready weights bound as static state;
- deterministic rank-local arithmetic;
- a small, well-defined set of dynamic inputs.

## Capture lifecycle

```mermaid
sequenceDiagram
    participant App
    participant Program as CudaGraphProgram
    participant Side as side stream
    participant Graph as CUDA Graph

    App->>Program: capture(function, example_inputs)
    Program->>Program: build value signature and input buffer
    Program->>Side: warm up with fresh buffers
    Side-->>Program: lazy initialization complete
    Program->>Graph: capture function on stable inputs
    Graph-->>Program: retain output objects and storage
    Program->>Graph: synchronized validation replay
    Program-->>App: program and capture statistics
```

Warmup occurs outside capture so lazy initialization, allocator activity, and
kernel setup do not unexpectedly enter the captured region.

## Replay lifecycle

```mermaid
sequenceDiagram
    participant App
    participant Program
    participant Stream as copy/current stream
    participant Graph

    App->>Program: replay(next_inputs)
    Program->>Program: validate value signature
    Program->>Stream: copy into stable input addresses
    Program->>Stream: wait for overwrite safety
    Program->>Graph: replay
    Graph-->>Program: update retained output storage
    Program-->>App: borrowed output or owned clone
```

The convenience `replay(...)` path combines input staging and replay. Advanced
schedules may split them:

- `copy_inputs_from(...)` prepares stable inputs and returns a copy handle;
- `replay_prepared(...)` consumes that prepared handle and launches replay.

Use the [Execution API reference](../../api/fhelium/execution/cuda_graph.md) for stream,
event, and output-copy options.

## Borrowed outputs

Captured output tensors are retained at stable addresses. The default output is
therefore borrowed:

```mermaid
stateDiagram-v2
    [*] --> Replay1
    Replay1 --> Borrowed1
    Borrowed1 --> Replay2: same output storage overwritten
    Borrowed1 --> Owned: copy_output = true
    Replay2 --> Borrowed2
    Owned --> Retained
```

If a caller must retain one result across the next replay, request an owned
copy. Merely keeping the Python output object does not preserve its previous
contents.

## Sequential program instances

One `CudaGraphProgram` instance owns one set of stable inputs, graph state, and
retained outputs. Treat it as sequential. Concurrent workers should own
separate program instances, buffers, and scheduling state.

Calling the raw underlying CUDA graph's replay method bypasses FHElium's:

- value-signature validation;
- dynamic input staging;
- event dependencies;
- overwrite protection;
- output ownership policy.

Use the program wrapper unless deliberately implementing a lower-level runtime
with equivalent guarantees.

## Capture region

```mermaid
graph LR
    subgraph Outside[Usually outside capture]
      ENC[encryption]
      KEY[key generation and loading]
      IO[request I/O]
      DIST[dynamic collectives]
      CACHE[admission and cache misses]
    end
    subgraph Inside[Fixed rank-local schedule]
      ROT[rotations]
      PM[plaintext multiplication]
      RS[rescale]
      ADD[accumulation]
    end
    Outside --> Inside
```

Randomized key generation/encryption, dynamic shapes or levels, variable
communication topology, storage I/O, and cache miss paths are poor capture
candidates.

A distributed workload normally captures each rank's stable local evaluator
and leaves typed reduction in eager execution.

## When graphs help

Graphs target repeated host/Python/dispatcher launch overhead. They tend to
help when:

- the schedule is replayed many times;
- there are many relatively small launches;
- input signatures remain stable;
- graph-private and retained memory fit the budget.

They may provide little benefit when one large kernel, host-to-device (H2D) input transfer, or
inter-rank communication already dominates. Always compare a synchronized eager
baseline with the same correctness and memory accounting.

## Common failures

- Capturing key generation or fresh-randomness encryption.
- Changing level, scale, or key step between replays.
- Retaining a borrowed output across another replay.
- Concurrent replay through one program instance.
- Assuming graph capture automatically includes distributed collectives.
- Reporting graph speedup without including input staging or checking allocator
  peaks.

## Continue

- [CUDA Graph matvec tutorial](../../tutorial/cuda-graph-matvec.md)
- [Value signatures and buffers](signatures-and-buffers.md)
- [Communication semantics](../distributed/communication-semantics.md)
- [CKKS cost model](../performance/cost-model.md)
