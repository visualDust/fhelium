# Neutral IR programs

`fhelium.ir.Program` is the compiler stack's source-independent representation of a
computation. It owns one structurally valid xDSL `ModuleOp` and can contain
standard structural operations, FHElium types and references, partially lowered
arithmetic, and extension dialects in the same module.

A Program contains the source-independent IR structure. Python source, Compile
workspaces, live CKKS objects, target devices, Backends, and executables remain
separate inputs to the consumers that need them. PyTorch capture, textual
import, direct xDSL construction, Compile transforms, and JIT transforms all
produce or consume this same class.

Import these interfaces from `fhelium.ir`, `fhelium.compile`, and
`fhelium.experimental.jit`.

## Representation anatomy

A Program contains serializable structure and symbolic identities:

| Element | Representation |
| --- | --- |
| Module and functions | xDSL `builtin.module` and registered `func` operations |
| Dataflow | Blocks, regions, operands, results, and SSA use-def relationships |
| Known FHElium values | Open encrypted, message, and plaintext types |
| External values | Material and resource reference operations with symbolic names |
| Partial state | Attribute dictionaries attached to open FHElium types and operations |
| Extensions | Unregistered operations and types preserved by the permissive xDSL context |
| Interchange versions | FHElium schema and dialect version module attributes |

`Program.empty(...)` constructs a module, and `Program.from_function(...)` wraps
a caller-built block in a top-level function. `Program.parse(...)` and
`Program.load(...)` construct the same object from text.

```python
from fhelium import ir

program = ir.parse(text, source_name="experiment.mlir")
program.verify_structure()
copy = program.clone()
```

The `module` property exposes the underlying xDSL object when direct
construction or mutation is required. `walk()`, `functions`, `function(name)`,
and `single_block(name)` provide common structural access. The last accessor
serves consumers that specifically require one block while the base Program
continues to accept richer structure.

## Structural integrity

Program construction calls xDSL verification. Structural integrity means that
the module can be represented according to xDSL's rules: registered operations
and attributes satisfy their structural definitions, regions and blocks form a
valid hierarchy, and SSA operands and results have valid relationships.

Pipeline execution repeats this structural check after every pass. This catches
a pass that returns malformed IR before a later pass consumes it.

Structural integrity covers the xDSL representation. Semantic completeness and
numerical validity require their respective analyses.

### Semantic incompleteness

A structurally valid Program may contain an operation whose meaning is unknown
to FHElium, a known operation with prerequisites that have not been introduced,
or a mixture of operations for which no complete lowering has been selected.
For example, an unregistered `vendor.ckks.bootstrap` operation can be a valid
SSA producer even though no current pass or backend interprets it.

For the registered executable vocabulary, semantic meaning belongs to a
provider-neutral `OperationSpec`. Extensions may instead define a contract in a
dialect, pass, caller-supplied specification registry, backend, external
implementation, or trusted handler. Consumers obtain the contract from one of
those declared semantic owners.

### Numerical incompleteness or inconsistency

Known encrypted types use open state dictionaries. A value may omit a scale,
depth, basis, polynomial domain, or other CKKS fact because the fact is not yet
known. A Program may also represent a candidate schedule in which two addition
operands have incompatible scales, a rescale has no valid Q depth group to drop, or an
approximation has no established error bound.

These states are useful inputs to analysis, diagnostics, and experimental
transforms. CKKS consistency,
approximation error, security parameters, key sufficiency, and schedule cost
require analyses with explicit assumptions and scopes.

## Mixed abstraction levels and registered dialects

Each operation or region carries its own abstraction level. Dialect version 0.2
registers the following first-party
namespaces in addition to xDSL `builtin` and `func`:

| Namespace | Represented responsibility |
| --- | --- |
| `fhelium` | Core open value/reference types, material/resource references, and constants |
| `fhelium_semantic` | Provider-neutral secret/public tensor arithmetic |
| `fhelium_logical` | Encrypted/public operation-role combinations |
| `fhelium_ckks` | CKKS values, preparation, representation transitions, and evaluator operations |
| `torch` | Preserved Torch calls with structured call descriptors |

The textual namespace uses underscores between `fhelium` and a specialized
level. For example, the registered spellings include
`fhelium_semantic.multiply` and `fhelium_ckks.to_ntt`.

One module can contain operations from all of these levels together with
application, vendor, or research extensions. A pass transforms the registered
patterns it recognizes and leaves the rest unchanged. Different functions may be
lowered at different CKKS depths, and a single function may connect several abstraction levels.
Structural verification accepts this mixed-level composition.

Registration gives first-party operations and types structural constructors and
IRDL verification. Explicit analyses check CKKS schedule consistency, while
Backend coverage checks external bindings and implementation availability.

## Open value types and external state

Registered value types carry open state dictionaries so a Program can represent
known facts without requiring every fact at construction. The core `fhelium`
types provide compatibility roles such as `EncryptedType`, `MessageType`, and
`PlaintextType`; specialized dialects provide depth-specific types including
`semantic.SecretType`, `semantic.PublicType`, `logical.EncryptedType`,
`ckks.CiphertextType`, `ckks.PlaintextType`, `rns.BundleType`, and
`keyswitch.KeyType`. `MaterialType` and `ResourceType` represent graph-external
identities.

`value_type(role, state)` constructs a known role type, while
`value_role(value)` returns a recognized role or `None` for an extension type.
The state dictionary stores serializable attributes.

`MaterialRefOp` and `ResourceRefOp` introduce graph-external identities. A
symbol can identify a captured Tensor constant, key-related material,
application value, buffer, or another caller-defined object. The caller-owned
object is resolved through the Compile workspace, runtime bindings, or Backend
resource mechanism.

## Textual round trip

Program text is the editable interchange form:

```python
program.save("before-specialization.mlir")
edited = ir.load("before-specialization.mlir")
print(edited.to_text(generic=True))
```

`to_text(...)` supports ordinary or generic printing and optional locations;
`save(...)` writes UTF-8 text with locations. Parsing uses an xDSL dialect context that
registers `builtin`, `func`, and FHElium's current structural vocabulary while
allowing unknown dialects.

The round trip preserves IR structure, attributes, symbolic references, and
unregistered content accepted by xDSL. Parsing and printing may normalize
formatting, comments, and other lexical choices. A parsed Program is checked
for structural integrity.

FHElium uses xDSL, a Python implementation of MLIR-style IR infrastructure.
Integration with the upstream MLIR C++ toolchain or additional dialect
ecosystems requires a caller-supplied interchange adapter.

## Representation analysis

The base IR package supplies analyses that describe represented content without
classifying backend support:

```python
inventory = ir.inventory_program(program)
states = ir.analyze_value_states(program, function="main")
key_requirements = ir.analyze_evaluation_key_requirements(program)
```

`inventory_program` inventories operation counts, dialect names, and registered
top-level functions. `analyze_value_states` reports each SSA value's xDSL type,
recognized role, and open metadata for one single-block function.
`analyze_evaluation_key_requirements` lists rotation and relinearization key
capabilities requested by primitive CKKS operations in one selected entry. Key
generation, loading, binding, and validation occur in their execution owners.

JIT adds selected-entry requirement analyses, including symbolic materials,
resources, Torch targets, rotation steps, and relinearization requirements.
Symbol resolution, key creation, provider selection, and execution consume
those reports in later stages.

An analysis result is interpreted according to its stated scope. A
selected-entry requirement scan covers that entry; a Program inventory reports
represented structure; the operation registry supplies semantics.

## Registered operation semantics

`OperationSpecRegistry` is the centralized lookup for registered executable
operation semantics. Each `OperationSpec` records:

- the exact registered operation name and a descriptive semantic family;
- operand and result arities where they are fixed;
- expected value roles;
- an effect classification of `pure`, `rng-write`, `mutation`, or `opaque`;
- the registered operation class and an optional local validator.

`DEFAULT_OPERATION_SPECS` covers the registered core references and constants,
Torch calls, semantic pointwise operations, logical and CKKS operations,
registered RNS/key-switch/NTT/execution operations, and the registered
conversion bridge. The registry is immutable and rejects duplicate names.
It supplies operation meaning and local checks. Caller policy and passes select
providers and lowering sequences.

A Program can contain a name absent from the selected registry. Parsing,
printing, structural verification, and unrelated transformations continue to
work because representation is permissive. A consumer that requires a registered
specification must report the missing specification. `CompositeBackend`, for
example, reports `unknown-operation-spec` and requires a declared semantic
contract before assignment.

IR interpretation and backend execution are separate uses of the same
specifications. The IR interpreter supplies per-operation execution for its
registered operations. Backend assignment can join adjacent equal assignments
before backend lowering. Triton can therefore implement a contiguous region of
`fhelium_semantic.add`, `fhelium_semantic.multiply`, and
`fhelium_semantic.negate` without changing those operations' registered
semantics or introducing a `fhelium.kernel.*` vocabulary.

Component-sensitive CKKS specifications also define the executable component
contract. Ciphertext multiplication requires two-component operands and a
three-component result; relinearization requires a three-component operand and
a two-component result; key switching, rotation, grouped rotation, and
conjugation require two-component operands and results. Permissive
representation accepts absent or conflicting component state for analysis;
execution requires the complete component contract.

CKKS state consistency, numerical accuracy, cryptographic security, and
implementation equivalence each require analyses and validation with a stated
scope in addition to operation specifications and provider coverage.

## Transform protocol

`Pass` and `Pipeline` belong to Compile and transform a `Compilation`, which
carries the current Program, workspace data, and ordered reports:

```python
from fhelium import compile as fh_compile

compilation = fh_compile.Compilation(
    program,
    fh_compile.CompileWorkspace(
        {"application/policy": "candidate-a"}
    ),
)
result = fh_compile.Pipeline((first_pass, second_pass)).run(
    compilation
)

transformed = result.program
reports = result.reports
```

A pass returns `PassResult`, containing its Program, `PassStats`, diagnostics,
and optional `DecisionRecord` values. An unchanged result is valid when a pass
finds no matching pattern or elects not to transform one. Pipeline execution
clones the source once, so the original Program remains unchanged.

A Pipeline accepts an ordinary mutable `dict` and passes that same object to
every selected pass. Code that writes and reads an entry defines its key,
value, and validation contract. The dictionary travels beside the Program as
Compile workspace state.

## Workspaces, execution device, and live bindings

External state is divided according to ownership and serialization behavior:

| Container | Contents | Relationship to Program |
| --- | --- | --- |
| `CompileWorkspace` | Arbitrary entries shared across one capture and Compile pipeline | Passed beside a Program as Python state |
| `ConstantBundle` | Captured constant snapshots | Stored under the `ConstantBundle` class key in a Compile workspace; IR uses material symbols |
| JIT Session workspace | Arbitrary entries shared across Session transforms | Initially empty and passed beside a Program |
| JIT execution device | One concrete CPU/CUDA device | Stored in `session.device`; Session reads the topology needed to check it |
| `RuntimeBindings` | Live engine, keys, materials, resources, resolvers, and trusted handlers | Stored directly in `session.bindings`; supplied to a selected backend |

A captured tensor constant is stored in the `ConstantBundle` found at
`captured.workspace[ConstantBundle]`, while IR contains its material symbol. An
application may later copy that symbol map into `RuntimeBindings.materials`.
This transfer is a caller action because a compile-time snapshot and a live
runtime capability have different ownership.

Program serialization writes Program text. Compile/JIT workspaces and live
bindings remain process-owned Python objects; typed-value serialization and
`ArtifactStore` persist their respective public value and artifact formats.

## Extension trust

Unregistered operations remain inert representation data. They become
executable only when the caller supplies one of the following trusted
capabilities:

- a pass that lowers the operation to another represented form;
- a backend that implements its declared semantics;
- a caller-defined backend that loads a compiled symbol under its own ABI and
  target checks;
- a runtime handler explicitly installed in `RuntimeBindings`.

Operation and Torch handlers are trusted live capabilities. FHElium can check
that a required handler is present and callable, but the caller owns its
semantic equivalence, side effects, input and result types, memory safety, and
resource behavior. Code execution begins only through a capability explicitly
installed by the caller.

## Continue

- [Open compiler stack](open-compiler-stack.md) explains the architectural
  lifecycle, transformation model, linking, and relationships to runtime
  subsystems.
- [IR API](../api/fhelium/ir.md) lists the public Program, analysis, and pass
  interfaces.
- [Textual IR tutorial](../tutorial/ir-textual-program.md) demonstrates parsing,
  stable serialization, a caller-defined analysis pass, and partial lowering.
- [Rank-local collective IR](../tutorial/rank-local-collective-ir.md) compares a
  specialized collective with a generic visible combine region.
- [Compiler stack internals](../developer/compiler-stack-internals.md) documents
  current operation schemas and extension contracts.
