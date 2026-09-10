# Distributed internals

`fhelium.distributed` extends PyTorch distributed with collectives that preserve
FHElium value metadata or apply CKKS-specific arithmetic. Process-group
initialization, communication backends, rank identity, point-to-point
operations, and ordinary tensor collectives remain PyTorch facilities.

## Runtime stack

```mermaid
graph TB
    APP[SPMD Python application]
    API[fhelium.distributed public API]
    META[Descriptor and group validation]
    DATA[Tensor payload operations]
    PG[torch.distributed ProcessGroup]
    CPU[Gloo or selected CPU backend]
    GPU[NCCL or selected CUDA backend]

    APP --> API --> META --> DATA --> PG
    PG --> CPU
    PG --> GPU
```

`fhelium.distributed.init()` calls
`torch.distributed.init_process_group`. Under `torchrun`, standard `RANK`,
`WORLD_SIZE`, and `LOCAL_RANK` environment values supply the default process
identity. CUDA initialization selects `cuda:LOCAL_RANK`; CPU execution defaults
to Gloo. A direct world-size-one launch uses a local `HashStore` when no
rendezvous is supplied, but still creates an ordinary PyTorch `ProcessGroup`.

The package also re-exports selected `torch.distributed` APIs. Raw tensor calls
retain their PyTorch signatures, mutation behavior, group semantics, and
`Work` handles. FHElium-specific names such as `broadcast_ciphertext` and
`all_reduce_ciphertext` identify operations that require value metadata
or CKKS arithmetic.

## Rank and device identity

Three integer namespaces appear in distributed code:

- **global rank** identifies a process in the global job;
- **process-group-relative rank** identifies its position in one subgroup;
- **local rank** commonly identifies a process on one host and selects a local
  CUDA device.

Public collective arguments such as `src` and `dst` use the rank namespace
stated by their PyTorch or FHElium interface. CUDA device indices identify
process-local placement.

Each rank constructs a local `fhelium.eager.Engine` and local values. Engines and process
groups have independent lifetimes:

```mermaid
graph LR
    RANK[One process / rank]
    DEV[One selected local device]
    ENG[Rank-local eager Engine]
    VALUES[Local values and keys]
    GROUP[PyTorch ProcessGroup]

    RANK --> DEV --> ENG --> VALUES
    RANK --> GROUP
```

## Rank-local distributed IR

A Compile `Program` represents one rank. A distributed pass receives a
participating-device count and emits a rank-local graph that may read its group
rank at launch. An SPMD launcher binds the same graph template, or a
rank-specialized derivative, to one process-group member per rank. Process-group
creation and destruction remain outside the Program.

The registered distributed vocabulary is:

| Operator | Rank-local meaning |
| --- | --- |
| `fhelium_dist.rank` | Read the current process-group-relative rank. |
| `fhelium_dist.group_size` | Read the number of participating group ranks. |
| `fhelium_dist.broadcast` | Functionally broadcast one local Tensor value from a group-relative root. |
| `fhelium_dist.all_reduce` | Reduce local Tensor values using a visible two-argument combine region. |
| `fhelium_dist.all_reduce_add_ciphertext` | Preserve ciphertext-add reduction intent for a whole-operation implementation or later lowering. |
| `fhelium_dist.yield` | Return one result from a generic combine region. |

`!fhelium_dist.group` is a launch-bound resource type. Its live binding is a
`ProcessGroupExecutionResource`; the Program does not own the PyTorch
`ProcessGroup` lifecycle. Rank and group size are distinct from the local CUDA
device index. A separate `!fhelium_memory.device` resource selects transfer
destinations.

The IR also loads xDSL `arith` and structured control flow (`scf`). A rank may
therefore select its local loop interval without embedding one multi-rank
module:

```text
group rank -> rank-local loop bounds -> local arithmetic -> collective
```

Compile transformations recurse into `scf.for`, `scf.if`, and generic
all-reduce combine regions. Unknown vendor region owners stay opaque. Current
Backend execution supports scalar index arithmetic, one-block `scf.for` and
`scf.if` regions, and registered Tensor implementations inside those regions.
This structured executor is independent of the current JIT planner design.

Distributed IR is permissive. Structural verification checks local operand,
result, attribute, and region shape only. It does not prove that predicates or
loop trip counts are uniform, that ranks execute matching collective sequences,
that a combine function is associative, or that execution is deadlock-free.
Caller-selected analysis passes may diagnose those properties; Backend and
launch do not silently add handshakes, barriers, or corrective collectives.

`LowerSpecializedCollectivesPass` may replace
`fhelium_dist.all_reduce_add_ciphertext` with generic
`fhelium_dist.all_reduce`. The generated combine region contains a visible
`fhelium_ckks.add`. Callers may instead preserve the specialized operator and
bind a whole-operation implementation. Both representations use the same
operation registry and arithmetic resources.

The initial generic implementation is a correctness baseline: it all-gathers
equal-layout local Tensor payloads and folds them in process-group rank order
through the compiled combine region. It does not establish a permanent
tree/ring/hierarchical algorithm policy. Provider or pass implementations may
select another inspectable communication plan later.

## Rank-local transfer IR

`fhelium_memory.transfer` is the placement-changing operation for an existing
SSA value. It consumes a value and a launch-bound
`!fhelium_memory.device` resource and produces the same IR value type.
Ordinary arithmetic operations do not receive a generic device attribute.

The first implementation is synchronous and functional. Its `memory_space`
attribute accepts `default`, `pageable_host`, or `pinned_host`; CPU and
concrete-index CUDA targets are runtime resources. A same-device transfer may
return the input storage. Cross-device movement delegates to PyTorch. Async
copy tokens, streams, fixed-buffer allocation, and Residency integration are
not part of this slice.

The Backend transfer ABI consumes one Tensor payload. Public Ciphertext,
Plaintext, and key objects remain public boundaries whose adapters decompose
and reconstruct Tensor leaves. In particular, this primitive does not turn
cross-device key replication into an automatic placement policy.

## Descriptor and payload phases

A typed value transfer separates control metadata from tensor payloads. The
sender converts a value into a `ValueEnvelope`-derived descriptor containing:

- transfer protocol version;
- concrete FHElium value type and value-schema version;
- non-tensor arithmetic metadata;
- tensor names, shapes, dtypes, and CPU/CUDA device type.

The receiver validates the descriptor, allocates the corresponding tensor
leaves on its rank-local device, transfers those leaves, and reconstructs the
typed value through the value schema.

```mermaid
sequenceDiagram
    participant S as Source rank
    participant C as Group-consistent control phase
    participant R as Receiver rank
    participant P as Tensor payload collectives

    S->>C: value descriptor
    C->>C: validate protocol, arguments, and rank agreement
    C->>R: accepted descriptor
    R->>R: allocate typed receiver storage
    S->>P: ordered dense tensor leaves
    P->>R: transfer payloads
    R->>R: reconstruct FHElium value
```

The transfer protocol and durable serialization schema have independent
versions. Raw `torch.Tensor` has its own transport descriptor kind outside the
FHElium value serialization registry.

Control errors must become group-consistent before a rank enters a payload
collective. Otherwise, a rank-local exception can leave peers waiting in NCCL
or Gloo.

## Whole-value collectives

Whole-value collectives transmit one complete typed object per logical
position:

- broadcast supports ciphertexts, plaintexts, compressed plaintexts, and
  selected key types;
- scatter distributes a source sequence of complete ciphertext values;
- gather and all-gather reconstruct complete values in process-group order.

These are transport operations. They preserve payload bits and value metadata;
they do not add ciphertexts, concatenate RNS rows, align depths, or change
scale.

Receiver allocation follows descriptor device type. A CUDA-described tensor
requires a CUDA rank-local device; CPU-described tensors remain CPU tensors.
The transfer layer does not silently change the sender's declared device type.

## Limb collectives

`scatter_ciphertext_limbs` and `gather_ciphertext_limbs` implement structural
RNS partitioning. A shard carries a subset of the ordered `prime_ids` and the
matching tensor limb rows. Reconstruction requires:

- one public depth and caller-established CKKS parameter provenance;
- identical component and batch axes;
- identical scale, polynomial domain, modulus basis, and residue form;
- non-overlapping requested prime IDs;
- complete requested coverage in configured prime order.

Gather concatenates rows into the declared mathematical layout. It does not
perform modular addition. Conversely, whole-value gather does not reconstruct a
ciphertext from disjoint limb shards.

## Ciphertext reductions

Raw `torch.distributed.ReduceOp.SUM` is incorrect for ciphertext payloads
because signed machine-integer addition does not implement per-limb modular
addition. `reduce_ciphertext` uses an arbitrary-world-size binomial tree. Each
receiver obtains one complete ciphertext into temporary storage and calls its
rank-local `engine.add_`:

```mermaid
graph LR
    A[Rank-local ciphertext partials]
    P2P[batch_isend_irecv tree edge]
    TMP[Temporary complete ciphertext]
    ADD[Engine.add_<br/>modular native operation]
    ROOT[Root ciphertext sum]
    BCAST[Payload broadcast]
    ALL[Sum on every rank]

    A --> P2P --> TMP --> ADD --> ROOT
    ROOT --> BCAST --> ALL
```

The tree supports non-power-of-two group sizes, emits `P - 1` ciphertext
messages, has `O(log P)` critical-path rounds, and needs at most one incoming
ciphertext buffer per active receiver. `all_reduce_ciphertext` performs that
modular reduction, then broadcasts the completed ciphertext payload from the
first process-group rank.

These typed reductions are synchronous composite operations. They do not
return a `torch.distributed.Work`; introducing useful overlap would require an
lifetime model covering both communication and local modular
addition.

## Source layout

| Layer | Source |
| --- | --- |
| Process-group initialization and local device | `fhelium/distributed/_state.py` |
| Transfer descriptor and receiver allocation | `fhelium/distributed/_transfer.py` |
| Group/rank validation and tensor staging | `fhelium/distributed/_collective_common.py` |
| Whole-value collectives | `fhelium/distributed/_value_collectives.py` |
| Limb scatter/gather | `fhelium/distributed/_limb_collectives.py` |
| Modular ciphertext reduce/all-reduce | `fhelium/distributed/_ciphertext_reduction.py` |
| Private public-API aggregation | `fhelium/distributed/_typed_collectives.py` |
| Public PyTorch-compatible facade | `fhelium/distributed/__init__.py` |
| Distributed IR declarations | `fhelium/ir/dialects/distributed.py` |
| Memory-transfer IR declaration | `fhelium/ir/dialects/memory.py` |
| Structured Compile traversal | `fhelium/compile/passes/_operation_transforms.py` |
| Specialized collective lowering | `fhelium/compile/passes/distributed/_lower_specialized_collectives.py` |
| Process-group resource binding | `fhelium/backend/distributed/resources.py` |
| Registered collective implementations | `fhelium/backend/distributed/operations.py` |
| Registered Tensor transfer implementation | `fhelium/backend/memory/operations.py` |
| Structured Program execution | `fhelium/backend/execution.py` |

## Validation

Distributed implementation changes should cover:

- world size one, two, and a non-power-of-two size where applicable;
- default group and subgroups;
- global versus group-relative rank arguments;
- CPU/Gloo and CUDA/NCCL paths supported by the change;
- descriptor or argument mismatch before payload transfer;
- reconstructed value state and device;
- whole-value, additive, and limb-partition semantics as distinct cases;
- cleanup and timeout diagnostics after a failed collective.

IR-specific behavior coverage additionally checks:

- textual parse/print and dialect catalog registration;
- CKKS lowering and implementation assignment inside known structured regions;
- specialized-to-generic collective lowering;
- CPU/CUDA transfer and pinned-host allocation;
- rank query, `scf` execution, broadcast, and visible combine-region execution;
- Gloo and NCCL operation execution without an Eager Engine fallback.

The focused implementation suite starts at
`tests/distributed/test_distributed_transfer.py`. Public multi-rank examples and benchmark
workers provide workload-level validation after the protocol tests.

## Continue

- [Rank-local SPMD model](../concepts/distributed/spmd-model.md)
- [Communication semantics](../concepts/distributed/communication-semantics.md)
- [Execution buffers and CUDA Graphs](execution-buffers-and-cuda-graphs.md)
- [Eager, Compile, and native execution](engine-native-stack.md)
