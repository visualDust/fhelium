# Compiler stack internals

FHElium's compiler stack separates mixed-level representation,
source-oriented transformation, runtime observation, and executable
construction across four packages:

- `fhelium.ir` owns the xDSL `Program`, registered dialects,
  operation specifications, analyses, and shared pass protocol;
- `fhelium.compile` owns PyTorch source capture, compile-time
  attachments, shared CKKS-to-RNS/NTT lowering, and pass adapters;
- `fhelium.runtime` reads CPU/CUDA topology and point-in-time memory counters,
  and supplies buffer, signature, and CUDA Graph mechanisms;
- `fhelium.experimental.jit` owns live bindings, IR
  interpretation, backend assignment, and executable build.

A `Program` remains permissive and may mix registered abstraction levels with
unregistered extensions. Structural validity, operation semantics, mathematical
analyses, and provider coverage are distinct checks.

## Package and source map

| Responsibility | Implementation | Principal objects |
| --- | --- | --- |
| Program ownership and interchange | `fhelium/ir/_program.py` | `Program` |
| Dialect context and value helpers | `fhelium/ir/_dialect.py` | `create_dialect_context`, `create_operation`, `value_type`, `value_role` |
| Registered dialects | `fhelium/ir/dialects/` | core, semantic, logical, CKKS, RNS, NTT, and Torch classes |
| Registered operation semantics | `fhelium/ir/_operation_specs.py` | `OperationSpec`, `OperationSpecRegistry`, `DEFAULT_OPERATION_SPECS` |
| Representation analyses | `fhelium/ir/_analysis.py` | inventory, value-state, and evaluation-key analyses |
| Compile pipeline protocol | `fhelium/compile/_pipeline.py` | `Pass`, `PassResult`, and `Pipeline` |
| Source capture | `fhelium/compile/frontend/` | `capture`, `CapturedCallable`, input specifications, PyTorch encoding |
| Compilation state and driver | `fhelium/compile/_compilation.py`, `_workspace.py`, `_driver.py` | `Compilation`, `CompileWorkspace`, `ConstantBundle`, `compile` |
| Frontend passes | `fhelium/compile/passes/frontend/` | source-semantic normalization into logical operations |
| CKKS passes | `fhelium/compile/passes/ckks/` | logical-to-CKKS transformation, scheduling insertion, and caller-composed rotation hoisting |
| Neutral CKKS lowering library and pass adapter | `fhelium/compile/passes/lowering/` | `CkksLoweringRegistry`, `lower_ckks_program`, `LowerCkksToRnsNttPass` |
| Eager CKKS lifecycle assembly | `fhelium/eager/` | `Engine`, lazy device-local resources, key inventory, and configured numerical services |
| CKKS numerical algorithms | `fhelium/backend/ckks/` | codec, cryptography, rotation implementations, and material factories |
| Registered operation execution | `fhelium/backend/` | `OperationBackend`, implementation registry, Backend passes, Program-wide resources, and executables |
| CPU and CUDA topology | `fhelium/runtime/topology.py` | `CpuTopology`, `CudaDeviceInfo`, `CudaTopology` |
| Point-in-time memory counters | `fhelium/runtime/memory.py` | `MemorySnapshot` |
| Experimental JIT | `fhelium/experimental/jit/` | Runtime bindings, provider coverage, region planning, and executable construction |

The focused stable tests are `tests/compile/test_ir_stack.py` and
`tests/compile/test_compile_stack.py`.

The IR operation-spec registry defines operation meaning and effects.
`fhelium.backend.ckks` supplies numerical codec and cryptographic algorithms,
while `fhelium.eager` adapts public objects to Tensor payloads and owns their
immediate-execution lifecycle. The current JIT adapter remains Eager-backed
whenever those runtime services are required.

## Program ownership and structural validity

`Program` owns one `xdsl.dialects.builtin.ModuleOp`. `Program.empty()` supplies
`fhelium.schema_version = "1"` and `fhelium.dialect_version = "0.2"`.
`Program.from_function()` wraps a caller-built block in `func.func`; parse,
load, clone, print, and save all return or consume the same representation.

Construction and pipeline execution call xDSL verification for registered
IRDL constraints, region/block structure, and static single assignment (SSA)
use-def integrity. CKKS analyses, resource resolution, Backend coverage, and
performance evaluation are separate consumers of the verified Program.
`Program.single_block(name)` selects the form used by current execution
consumers; the base representation accepts richer structure.

## Registered multi-level dialects

`create_dialect_context()` loads xDSL `builtin` and `func`, then all registered first-party FHElium
dialects, with `allow_unregistered=True`. Dialect version 0.2 registers:

| Namespace | Representative registered types and operations |
| --- | --- |
| `fhelium` | `!fhelium.encrypted`, `!fhelium.message`, `!fhelium.plaintext`, material/resource types and references, `fhelium.constant` |
| `fhelium_semantic` | role-aware frontend values and add/subtract/multiply/negate/roll before CKKS representation is selected |
| `fhelium_logical` | encrypted/public types and supported encrypted/public operand combinations |
| `fhelium_ckks` | ciphertext/plaintext/key types, add/subtract/multiply, NTT transitions, preparation, relinearize, rescale, rotate |
| `torch` | structured `torch.call` |

Specialized namespaces use underscore spellings such as `fhelium_ckks`.
Representative operation names are `fhelium_semantic.multiply`,
`fhelium_logical.multiply.encrypted_encrypted`, `fhelium_ckks.to_ntt`,
and `torch.call`.

`fhelium_semantic.add` and `multiply` operate on values that already carry
encrypted/public frontend roles while their CKKS representation remains open.
Public-public multiplication may remain ordinary Tensor work;
encrypted-encrypted multiplication may lower to CKKS multiplication,
relinearization, and rescale.

Registration supplies constructors and structural verification while allowing
operations from different abstraction levels in one Program:

```text
%product = "fhelium_semantic.multiply"(%left, %right)
    : (!fhelium_semantic.public<{}>, !fhelium_semantic.public<{}>)
   -> !fhelium_semantic.public<{}>
%candidate = "vendor.experimental.refresh"(%ciphertext)
    : (!fhelium_ckks.ciphertext<{}>) -> !fhelium_ckks.ciphertext<{}>
```

The vendor operation remains unregistered and inert unless a later pass,
provider, backend, or caller-registered handler supplies its meaning. Unknown
content can survive parse, print, clone, and unrelated transforms.

## Shared transformations and source capture

A pass implements:

```python
run(program: Program, shared_data: dict[object, object]) -> PassResult
```

`Compilation` carries the current Program, one mutable `CompileWorkspace`, and
the ordered pass reports accumulated so far. `Pipeline.run(compilation)` clones
the Program once, retains the same workspace, and returns another Compilation.
This keeps symbolic materials and pass state paired with the Program that
references them.

Each `CompileWorkspace` entry owner defines its key, value, and validation
contract.

Built-in code normally uses a value's class as its key. A Compile workspace
stores `CkksConfig` under that class key. Custom passes may use other objects
or strings. None of these entries is serialized into Program text.

`LowerCkksToRnsNttPass` reads `CkksConfig` when it applies. The current hybrid
key-switch digit layout is reconstructed from the Q/P chain in that
configuration and is device-independent. Under the one currently
supported decomposition policy, one configuration determines one complete
per-level key-digit-index table. The reverse mapping is many-to-one because
configurations with different unrelated parameters can produce the same table.
If the configuration is absent, the pass reports the missing input and leaves
CKKS operations in the Program. `Engine` directly owns the fixed resources
needed by immediate Eager execution.

`Pipeline` is an ordered tuple. It runs each pass over the evolving clone,
verifies every returned Program structurally, and appends ordered reports to
the Compilation. A pass may return unchanged IR when it finds no applicable
pattern. The default compile pipeline performs local semantic-to-logical and
logical-to-CKKS transformations, plaintext preparation, multiply NTT
transitions, relinearization/rescale insertion, and conservative dead-value
elimination. The resulting Program may contain operations at multiple levels.

`capture()` uses PyTorch FX with declared input roles:

| Input declaration | Capture effect |
| --- | --- |
| `encrypted()` | encrypted runtime block argument with declared frontend state |
| `message()` | public runtime argument |
| `plaintext()` | pre-encoded plaintext runtime argument |
| `static(value)` | finite immutable specialization omitted from the runtime signature |

Recognized arithmetic is emitted with registered semantic operation classes;
other calls use registered `torch.call` with structured target and argument
attributes. Tensor constants are detached into the `ConstantBundle` stored at
`captured.workspace[ConstantBundle]`, while `fhelium.material.ref` carries their symbolic
names in IR. Moving captured constants into `RuntimeBindings.materials` is a
caller action.

## OperationSpecRegistry stores registered operation semantics

Each first-party dialect owns an immutable `OPERATION_SPECS` tuple beside its
operation classes. `fhelium.ir._operation_specs` checks those declarations
against every registered dialect operation, then constructs
`DEFAULT_OPERATION_SPECS`. `OperationSpecRegistry` is an immutable name map
and rejects duplicate entries. Each
`OperationSpec` contains:

```python
OperationSpec(
    name,
    family,
    operand_arity,
    result_arity,
    operand_roles=(),
    result_roles=(),
    effect="pure",
    validator=None,
    operation_type=None,
)
```

The descriptive `family` classifies semantics for reporting and provider-local
checks. Provider selection uses registered implementation identities. `effect` is `pure` for operations
without externally visible state changes, `rng-write` for operations that
advance a random stream, `mutation` for operations that modify existing
values, or `opaque` for operations whose registered semantics have no narrower
effect classification. The optional validator checks local
attributes or structure not already expressed by the registered class and
role/arity fields. `operation_type` ties a registered specification to its IRDL
class.

`DEFAULT_OPERATION_SPECS` covers core references/constants, Torch calls,
semantic pointwise operations, logical and CKKS operations, lower-level
RNS/key-switch/NTT/execution operations, and the registered conversion bridge.
A Program may also contain extension names. When composite execution
encounters an absent specification, it reports `unknown-operation-spec` even if
a provider offered a proposal. Provider availability never defines operation
meaning.

Implementation-specific differential tests evaluate whether the IR
interpreter and region providers implement the registered meaning.

[Operation declaration and implementation selection](./operation-registration-and-selection.md)
documents the source ownership, named lowering, Compile assignment, and
Backend registration path.

## Execution device, bindings, and Session

A JIT Session owns one concrete `session.device` used to bind resources before
runtime Tensor inputs are available. Session reads the CPU or CUDA topology
needed for that device.

For a backend call, Session presents the selected device and observed topology
as `ExecutionInputs`. This value carries no lifecycle or selection authority;
the Session remains the owner of the execution request. Execution inputs are
read lazily for the Session device.

`RuntimeBindings` contains live execution capabilities: Eager Engine, public/evaluation
keys, materials, resources, trusted operation or Torch handlers, resolvers, and
application extension objects.

JIT runtime bindings are caller-supplied live objects. They are distinct from
`fhelium.backend.resources.ResourceBindings`, which contains typed resources
linked by symbol and kind for a concrete Backend executable.

A `Session` internally creates an initially empty workspace dictionary and
shares it across transforms. The Session separately owns one execution device,
one bindings object, a `BackendRegistry`, and an optional default Backend. The
following methods receive execution inputs and bindings through their runtime
interfaces:

| Method | Responsibility |
| --- | --- |
| `transform` | run a selected shared pipeline with `session.workspace` |
| `coverage` | ask one backend to inspect the current Program, environment, bindings, and policy |
| `build` | return coverage and an executable only when no coverage error remains |
| `run` | build and invoke the selected executable |

`analyze_requirements()` is a separate non-executing selected-entry scan for
operation names, unknown operations, symbolic references, Torch targets,
rotation/relinearization requirements, Eager Engine need, and return arity.

## Captured Python reference and IR interpreter

`CapturedCallable` is stored at
`compilation.workspace[CapturedCallable]`. Its `reference(...)` method invokes
the stored Python or PyTorch callable on ordinary inputs and computes the
pre-transform result used as a correctness oracle or debugging aid. It does
not contain the Program or CompileWorkspace.

`InterpreterBackend` remains separate from composite provider planning. It
checks one selected single-block function, schema and dialect versions,
operation schemas, symbolic bindings, runtime and key compatibility, and
trusted extension handlers. Its executable interprets registered operations in
SSA order.

Public pointwise operations use PyTorch. Registered CKKS operations use the
supplied `fhelium.eager.Engine`, which invokes the same operation registry and
resources available to composite regions.

Each `BackendProvider` owns a `RegionCompilerRegistry` that maps registered
source operations to provider-local region compilers. This registry assigns
and builds source regions. The current `native_cpu` and `native_cuda` provider
IDs select an Eager-backed JIT adapter. That adapter registers CKKS lowering,
preserved CKKS operations, public-boundary operations, RNS/NTT compilation,
and remaining Eager operations. The `triton` provider registers
`TritonPointwiseRegionCompiler` for the public pointwise operation classes it
can compile. Standalone CKKS operations remain available for Eager assignment.

## Caller provider policy and proposal selection

`ProviderPolicy` contains exactly two decisions:

```python
policy = ProviderPolicy(
    "cuda",
    ("triton", "native_cuda"),
)
```

The first field is the CPU or CUDA target. The second is an ordered tuple of
provider names. The order expresses caller preference when more than one
provider implements an operation; there is no operation-family route table.

Each provider reports support for individual registered operations. The
planner first forms adjacent candidates within each provider and runs their
non-executing region checks. It attaches any rejection to the affected
operation assignments before selecting one implementation per operation in
caller provider order. This lets a later caller-enabled provider cover a
region rejected by an earlier provider. Selected adjacent operations with the
same provider and implementation then form one execution region. CKKS lowering
remains a separate Compile transformation.

After selection, `CompositeBackend` resolves an `OperationSpec` for every
source operation, validates local semantics, requires exactly one selected
proposal to own it, asks each provider to lower and verify its region, and
constructs an `ExecutionPlan`. The plan records:

- Program input and output value IDs;
- every SSA value's role, producer, consumers, and Python-object ABI;
- every operation's name, semantic family, provider, implementation, support,
  and diagnostics;
- each selected region's operation IDs and crossing input/output values.

The executable manifest adds all assignment decisions, region executable
manifests, `connection_abi = "python-object-ssa-v1"`,
`cuda_stream = "current-pytorch-stream"`, and `fallback = false`.

At invocation, the wrapper binds Program arguments once, stores SSA results as
Python objects, invokes region executables in selected source order, and passes
escaping results to downstream regions. The plan records the provider,
transfer, layout, and stream actions performed by those regions.

## Triton pointwise lowering

The CUDA `triton` provider registers `TritonPointwiseRegionCompiler` for
`fhelium_semantic.add`, `fhelium_semantic.multiply`, and
`fhelium_semantic.negate`. Adjacent operations selected for Triton form one
region. `TritonPointwiseBackend` lowers that region to one generated kernel.
Runtime inputs must
be same-shaped contiguous CUDA `float32` public tensors; a region returns one
Tensor. The generated source and digest are inspectable implementation assets.

When policy is `ProviderPolicy("cuda", ("triton", "native_cuda"))`, a supported
Triton assignment precedes the overlapping native assignment. Dependency,
target, role, and region verification occurs before build; execution never
substitutes another provider.

## Shared CKKS lowering and native execution

`fhelium.compile.passes.lowering.lower_ckks_program` is the common
neutral lowering entry that eager execution, compiler pipelines, and JIT can
call independently. Eager execution and its JIT provider call this library
directly.
`fhelium.compile.LowerCkksToRnsNttPass` adapts the same library to
the compile pass protocol. Each
CKKS evaluator operation with shared lowering expands to a composition
of `fhelium_rns` and `fhelium_ntt` operations plus resource references.
Backend assignment separately selects CPU, CUDA, radix, kernel, mutation, and
fusion implementations. Current registrations cover ciphertext and dense-plaintext
arithmetic, coefficient/NTT and standard/Montgomery transitions, rescale,
modulus switching, scale reinterpretation, relinearization, caller-named key
switching, rotation, and conjugation. The lower-level operations may also
appear directly in an input Program whose operands
name the required resources.

`fhelium.execution.implementation` on a CKKS source operation selects a
whole-operation Backend implementation. Compile pipelines preserve that
operation; lowering rejects a conflicting transformation and retains the
recorded selection. Eager applies the same rule before choosing a named lowering or
preserving the operation for its assigned implementation.

At runtime, execution SSA values carry Tensor payloads. Program operand and
result types retain their represented CKKS and RNS state. `RnsContext`, `NttContext`,
`RescaleExecutionResource`,
`KeySwitchExecutionResource`, and ordinary key values provide concrete runtime
parameters and key storage. A Backend pipeline links them by symbol and kind
when it materializes the Program-wide resource environment. Execution uses the
linked live objects directly, including caller mutations made after linking.

`OperationImplementationRegistry` maps every implemented operation class and
implementation name. An `OperationBackend` owns one such registry plus an
immutable `BackendWorkspace` containing keys, named resources, a materializer,
and material overrides. `OperationBackend()` assembles the built-in
implementation registry with no preselected NTT executor. Specialized
execution owners may instead supply their own registry.
`ResolveBackendOperationsPass` scans without lowering and builds a
`ProgramDispatchTable`. `BindCkksKeysPass` can then match ordinary caller key
objects to the key roles required by that Program. Rotation keys match the
structured rotation step on their resource operands; a public, secret,
relinearization, conjugation, or generic switch key matches only a unique
unbound role of its concrete kind. Application user identities remain in
caller-owned key selection. Optional resource materialization runs once, after which
`LinkProgramPass` resolves graph-external materials, matches the complete
resource binding set, and directly builds the prelinked executable. The
resulting `ResourceBindings` is the post-match table used by Backend execution.
`OperationBackend.link(compilation)` runs the standard linking pipeline against
a shallow copy of Compile workspace data and returns `ProgramExecutable`.
The pipeline starts each link with
`InitializeResourceBindingsPass`, so every link starts from fresh live
bindings. Material references become ordinary SSA values.
`ProgramExecutable` follows SSA topology,
associates Tensor and resource operands, and invokes the implementations in its
dispatch table. The selected NTT executor belongs to `NttContext`; Compile
records its name as `fhelium.execution.implementation` on the existing NTT
operation.

`fhelium.eager.Engine` checks public values, computes the called operation's
result metadata, and invokes a registered implementation through an
`OperationInvocation`. Compile and JIT callers apply their selected frontend, middle-level,
and Backend passes, then use `OperationBackend.link()` to produce an
executable. JIT region compilers reuse the same mechanism.
Encode, decode, integer-coefficient conversion,
encrypt, and decrypt adapt public objects in Eager and dispatch their numerical
Tensor payloads through that registry. Compressed plaintext arithmetic follows
the same rule. Eager two-component ciphertext multiplication selects
`native-ct2-convolution`, while Compile passes may instead lower multiplication
to its RNS composition. Relinearization similarly has the
`native-relinearize-streaming` whole-operation implementation and the
`rns-ntt-relinearize` lowering. The scheduled
`fhelium_ckks.hoisted_rotate_many` operation selects
`native-rotate-many-hoisted`, which consumes the recorded offsets and keys
without deciding group membership. A caller may select `HoistRotationsPass`
to form these groups at compile time; Eager forms one only when
the caller selects `use_hoisting=True`. Manifests record the chosen
implementation, resources, effects, and `fallback = false`.
Key generation calls `CkksKeyGenerator` directly with one device's
`KeyGenerationResource`.

## Extension points and trust

| Extension | Required mechanism |
| --- | --- |
| New first-party IR construct | define an IRDL class in the appropriate dialect, add it to that dialect, and add its dialect-local `OperationSpec` |
| Registered executable operation | let the catalog check include the dialect-owned specification, then add lowering or implementations as required |
| Analysis or transformation | implement `Pass` and compose it in a `Pipeline` |
| RNS/NTT operation | define its dialect operation, operand/result types, resource requirements, Backend implementation, and CKKS lowering uses |
| Whole-Program executable strategy | implement `Backend.coverage` and `Backend.build` |
| Operation implementation | declare its name and operation classes beside the Backend implementation and add it to device-resource registry assembly |
| Application operation | lower it, provide a backend/provider, or bind a trusted `RuntimeBindings.handlers[name]` callable |
| FHE-touching Torch call | bind a reviewed `RuntimeBindings.torch_handlers[target]` callable |

Permissive parsing retains operation names as representation data. Backend
plugins, handlers, resolvers, and generated-code loaders supply trusted
execution authority and run
Python or native code and belong in the application's trust model.

A new provider should report precise assignment diagnostics, lower without
mutating the caller Program, make its artifact inspectable, verify environment and
bindings without executing user work, and preserve invocation ABI behavior.
