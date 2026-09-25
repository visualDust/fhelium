# Neutral IR programs

`fhelium.ir.Program` is the compiler stack's source-independent representation of a computation. It owns one structurally valid xDSL `ModuleOp` and can contain standard structural operations, FHElium types and references, partially lowered arithmetic, and extension dialects in the same module.

A Program contains the source-independent IR structure. Known placement and layout can be recorded as serializable value facts. Python source, workspaces, live Tensor data, execution handles, Backends, and executables are supplied separately to the consumers that need them. PyTorch capture, textual import, direct xDSL construction, and Compile transformations all produce or consume this same class.

Import representation interfaces from `fhelium.ir` and compilation interfaces from `fhelium.compile`.

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

`Program.empty(...)` constructs a module, and `Program.from_function(...)` wraps a caller-built block in a top-level function. `Program.parse(...)` and `Program.load(...)` construct the same object from text.

```python
from fhelium import ir

program = ir.parse(text, source_name="experiment.mlir")
program.verify_structure()
copy = program.clone()
```

The `module` property exposes the underlying xDSL object when direct construction or mutation is required. `walk()`, `functions`, `function(name)`, and `single_block(name)` provide common structural access. The last accessor serves consumers that specifically require one block while the base Program continues to accept richer structure.

## Structural integrity

Program construction calls xDSL verification. Structural integrity means that the module can be represented according to xDSL's rules: registered operations and attributes satisfy their structural definitions, regions and blocks form a valid hierarchy, and SSA operands and results have valid relationships.

Pipeline execution repeats this structural check after every pass. This catches a pass that returns malformed IR before a later pass consumes it.

Structural integrity covers the xDSL representation. Semantic completeness and numerical validity require their respective analyses.

### Semantic incompleteness

A structurally valid Program may contain an operation whose meaning is unknown to FHElium, a known operation with prerequisites that have not been introduced, or a mixture of operations for which no complete lowering has been selected. For example, an unregistered `vendor.ckks.bootstrap` operation can be a valid SSA producer even though no current pass lowers it or Backend implements it.

For the registered executable vocabulary, semantic meaning belongs to a provider-neutral `OperationSpec`. Extensions may instead define a contract in a dialect, pass, caller-supplied specification registry, backend, external implementation, or trusted handler. Consumers obtain the contract from one of those declared semantic owners.

### Numerical incompleteness or inconsistency

Known encrypted types use open state dictionaries. A value may omit a scale, depth, basis, polynomial domain, or other CKKS fact because the fact is not yet known. A Program may also represent a candidate schedule in which two addition operands have incompatible scales, a rescale has no valid Q depth group to drop, or an approximation has no established error bound.

These states are useful inputs to analysis, diagnostics, and experimental transforms. CKKS consistency, approximation error, security parameters, key sufficiency, and schedule cost require analyses with explicit assumptions and scopes.

## Mixed abstraction levels and registered dialects

Each operation or region carries its own abstraction level. Dialect version 0.2 registers the following first-party namespaces alongside xDSL `builtin`, `func`, `arith`, and `scf`:

| Namespace | Represented responsibility |
| --- | --- |
| `fhelium` | Core open value/reference types, material/resource references, and constants |
| `fhelium_semantic` | Provider-neutral secret/public tensor arithmetic |
| `fhelium_logical` | Encrypted/public operation-role combinations |
| `fhelium_ckks` | CKKS values, preparation, representation transitions, and evaluator operations |
| `fhelium_rns` | Residue arithmetic, component composition, basis conversion, and key-digit products |
| `fhelium_ntt` | Forward and normalized inverse polynomial transforms |
| `fhelium_memory` | Tensor movement to supplied destinations |
| `fhelium_dist` | Rank-local group queries, collectives, and combine regions |
| `fhelium_fusion` | Joint execution regions retaining their represented operations |
| `torch` | Preserved Torch calls with structured call descriptors |

The textual namespace uses underscores between `fhelium` and a specialized level. For example, the registered spellings include `fhelium_semantic.multiply` and `fhelium_ckks.to_ntt`.

One module can contain operations from all of these levels together with application, vendor, or research extensions. A pass transforms the registered patterns it recognizes and leaves the rest unchanged. Different functions may retain different abstraction levels, and a single function may connect several levels while its values carry independent CKKS state. Structural verification accepts this mixed-level composition.

Registration gives first-party operations and types structural constructors and IRDL verification. Explicit analyses check CKKS schedule consistency, while Backend coverage checks external bindings and implementation availability.

## Open value types and external state

Registered value types carry open state dictionaries so a Program can represent known facts without requiring every fact at construction. Core types provide encrypted, message, and plaintext roles. Specialized types describe semantic roles, logical arithmetic, and numerical representations: `semantic.SecretType`, `semantic.PublicType`, `logical.EncryptedType`, `logical.PublicType`, `ckks.CiphertextType`, `ckks.PlaintextType`, `ckks.CompressedPlaintextType`, `ckks.EvaluationKeyType`, and `rns.RnsBundleType`. Their abstraction level identifies the represented kind of computation; depth is an independently recorded CKKS fact. `MaterialType` and `ResourceType` describe external identities, while `memory.DeviceType` and `distributed.GroupType` identify execution-handle roles.

`value_type(role, state)` constructs a known role type, while `value_role(value)` returns a recognized role or `None` for an extension type. The state dictionary stores serializable attributes.

`MaterialRefOp` introduces a Tensor supplied through `Compilation.material_bindings`, such as a captured constant, evaluation-key payload, arithmetic table, or rounding-state Tensor. `ResourceRefOp` introduces a non-Tensor execution handle supplied through Backend resource bindings, such as a process group or encryption sampler. Both references preserve symbolic identity in the Program while their live objects retain caller-controlled lifetimes.

## Textual round trip

Program text is the editable interchange form:

```python
program.save("before-specialization.mlir")
edited = ir.load("before-specialization.mlir")
print(edited.to_text(generic=True))
```

`to_text(...)` supports ordinary or generic printing and optional locations; `save(...)` writes UTF-8 text with locations. Parsing uses an xDSL dialect context that registers `builtin`, `func`, `arith`, `scf`, and FHElium's current structural vocabulary while allowing unknown dialects.

The round trip preserves IR structure, attributes, symbolic references, and unregistered content accepted by xDSL. Parsing and printing may normalize formatting, comments, and other lexical choices. A parsed Program is checked for structural integrity.

FHElium uses xDSL, a Python implementation of MLIR-style IR infrastructure. Integration with the upstream MLIR C++ toolchain or additional dialect ecosystems requires a caller-supplied interchange adapter.

## Representation analysis

The base IR package supplies analyses that describe represented content without classifying backend support:

```python
inventory = ir.inventory_program(program)
states = ir.analyze_value_states(program, function="main")
key_requirements = ir.analyze_evaluation_key_requirements(program)
```

`inventory_program` inventories operation counts, dialect names, and registered top-level functions. `analyze_value_states` reports each SSA value's xDSL type, recognized role, and open metadata for one single-block function. `analyze_evaluation_key_requirements` reports rotation, relinearization, and conjugation capabilities represented by logical, CKKS, and lowered key-operand uses in a selected single-block entry and its nested regions. Generic key-switch relations remain caller-named. The returned capability requirements guide subsequent key provision.

Program analyses describe selected-entry requirements, including symbolic materials, resources, Torch targets, rotation steps, and relinearization or conjugation requirements. Symbol resolution, key creation, provider selection, and execution consume those reports in later stages.

An analysis result is interpreted according to its stated scope. A selected-entry requirement scan covers that entry; a Program inventory reports represented structure; the operation registry supplies semantics.

## Registered operation semantics

`OperationSpecRegistry` is the centralized lookup for registered executable operation semantics. Each `OperationSpec` records:

- the exact registered operation name and a descriptive semantic family;
- operand and result arities where they are fixed;
- expected value roles;
- an effect classification of `pure`, `rng-write`, `mutation`, or `opaque`;
- the registered operation class and an optional local validator.

`DEFAULT_OPERATION_SPECS` assembles the dialect-owned specifications for core references and constants, Torch calls, semantic and logical arithmetic, CKKS, RNS, NTT, memory movement, distributed operations, and fusion regions. The registry is immutable and rejects duplicate names. It supplies operation meaning and local checks. Caller policy and passes select implementations and lowering sequences.

A Program can contain a name absent from the selected registry. Parsing, printing, structural verification, and unrelated transformations continue to work because representation is permissive. A consumer that requires a registered specification must report the missing specification rather than inventing execution semantics.

Compile may group compatible RNS/NTT operations in a fusion region and select a Backend implementation for that region. The Triton implementation generates component-indexed arithmetic kernels and fused NTT endpoint stages while retaining the original operations inside the region for inspection. A transform region can require several synchronized kernels. Arbitrary captured Torch or semantic operations still require a lowering or their own Backend implementation.

Component-sensitive CKKS specifications also define the executable component contract. Ciphertext multiplication requires two-component operands and a three-component result; relinearization requires a three-component operand and a two-component result; key switching, rotation, grouped rotation, and conjugation require two-component operands and results. Permissive representation accepts absent or conflicting component state for analysis; execution requires the complete component contract.

CKKS state consistency, numerical accuracy, cryptographic security, and implementation equivalence each require analyses and validation with a stated scope in addition to operation specifications and provider coverage.

## Transform protocol

`Pass` and `Pipeline` belong to Compile and transform a `Compilation`, which carries the current Program, workspace data, one material-binding dictionary, and ordered reports:

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

A pass returns `PassResult`, containing its Program, `PassStats`, diagnostics, and optional `DecisionRecord` values. An unchanged result is valid when a pass finds no matching pattern or elects not to transform one. Pipeline execution clones the source once, so the original Program remains unchanged.

`Pipeline.run(compilation)` calls each `Pass.run(current_compilation)` with the evolving Program and retained workspace and material-binding mappings. Entry owners define workspace contracts; numerical Tensor data uses the shared material dictionary. Independent assignments use separate workspace and binding mappings. A shallow mapping copy retains the same Tensor objects and custom Python objects.

## Workspaces, execution device, and live bindings

External state is divided according to ownership and serialization behavior:

| Container | Contents | Relationship to Program |
| --- | --- | --- |
| `CompileWorkspace` | Arbitrary entries shared across one capture and Compile pipeline | Passed beside a Program as Python state |
| `Compilation.material_bindings` | Live Tensor data | A dictionary keyed by Program material symbol |
| `BackendWorkspace` | Non-Tensor execution handles and their preparation | Supplied to `OperationBackend` for linking |
| Callable specialization | Input conditions, source/transformed Compilations, and linked executable | Cached by a reusable callable |

`Program.material_descriptions` stores optional JSON annotations keyed by material symbol. Capture and material-producing passes record known information; callers may add labels and other fields. Symbols survive operation reordering. Descriptions do not constrain the data subsequently assigned to them.

`Compilation.material_bindings` supplies the actual Tensors. `prepare_material_bindings` can fill absent entries from supplied numerical providers and keys, preserving current assignments. Linking resolves the resulting references to live Tensor objects. Compatible in-place content updates remain visible; replacing an entry requires relinking to bind the new object.

`Program.save` writes textual IR, including descriptions. `serialization.save_compilation` additionally supports no, all, or selected Tensor bindings. It excludes workspaces, Python callables, pass reports, and executables. ArtifactStore persists that same Compilation representation under a logical name.

## Extension trust

Unregistered operations remain inert representation data. They become executable only when the caller supplies one of the following trusted capabilities:

- a pass that lowers the operation to another represented form;
- a backend that implements its declared semantics;
- a caller-defined Backend implementation that loads a compiled symbol under its own ABI and target checks.

Operation implementations are trusted executable extensions. Registration does not prove semantic equivalence, correct side effects, memory safety, or resource behavior. Callers own those properties for their extensions.

## Related concepts

- [Open compiler stack](open-compiler-stack.md) explains capture, transformation, material preparation, linking, and callable specialization.
- [State transitions and orthogonality](ckks/state-transitions-and-orthogonality.md) relates represented mathematical state to execution choices.
- [Rank-local SPMD](distributed/spmd-model.md) explains how a Program participates in a distributed computation.
- [Serialization and artifacts](execution/serialization-and-artifacts.md) explains durable representations and live rebinding.
