# Operation registration and implementation selection

Operation declarations define computation, lowering recipes rewrite it, and Backend implementations execute selected operations on Tensor payloads. FHElium records these three extension points separately so a Program can retain whole operations, expose RNS/NTT composition, or select generated implementations without changing their mathematical meaning.

## Where is each declaration owned?

| Declaration | Source owner | Identity |
| --- | --- | --- |
| Operation class and `OperationSpec` | `ir/dialects/<family>.py` | Dialect operation name and class |
| `CkksLoweringDefinition` | `compile/passes/lowering/` | Source operation class and lowering name |
| `OperationImplementation` | Owning `backend/` family | Operation class and implementation name |
| Built-in assembly | `backend/assembly.py` | Ordered implementation contributions |
| Native schema and registration | `csrc/ops/<family>/` | Torch namespace, schema, and dispatch key |

The [IR article](compiler-stack-internals.md) explains schema and effect ownership. The [operation index](ir-operation-implementation-index.md) links families to source definitions containing their signatures and equations.

## Element dependencies

Each operation class declares how its result elements depend on its operand elements. `registered_operation_spec` collects the class's `dependencies` declaration along with its signature and effects. Built-in operations implement a `dependencies()` method returning `OperationDependencies`. The method can read instance attributes such as a polynomial domain. The catalog also accepts fixed `OperationDependencies` values for specifications supplied directly.

`ValueDependency(result_index, operand_index, axes)` describes one result–operand pair. Named axes belong to the operation's coordinates: `slot` for logical encrypted values, `coefficient` and `limb` for RNS payloads, and `tensor_position` for ordinary Tensor indexing. The `coefficient` axis is the last dimension of an RNS polynomial payload. Its entries are coefficients in coefficient form and evaluations in NTT form.

| Relationship | Possible reads for an output position |
| --- | --- |
| `element` | The corresponding input position |
| `reindexed` | A mapped input position, including selection, permutation, and broadcast |
| `mixing` | Multiple input positions |
| `unknown` | An unresolved relationship |

An omitted declaration, result–operand pair, or axis is unknown. Each operand is described separately: a compact plaintext multiplication reads corresponding ciphertext positions and mapped compact-plaintext positions; its numerical parameter tables retain their own relationships. Effects such as mutation and random-state updates are recorded separately.

For example, a class can declare fixed relationships alongside its operands:

```python
from fhelium.ir import OperationDependencies, ValueDependency

# Method on an operation class with one result and two data operands:
def dependencies(self) -> OperationDependencies:
    return OperationDependencies((
        ValueDependency(0, 0, {"coefficient": "element", "limb": "element"}),
        ValueDependency(0, 1, {"coefficient": "element", "limb": "element"}),
    ))
```

`operation_dependencies(operation).kind(result_index, operand_index, axis)` resolves a relationship for the current instance. Coefficient-domain ModUp mixes limbs while preserving polynomial positions. NTT mixes positions while preserving limbs. NTT-domain ModDown mixes both. A row restriction reindexes limbs without combining their values. Shape equality and changes to `prime_ids` therefore do not determine these relationships.

Fusion-region descriptions compose the body SSA paths and merge paths reaching the same output. Unknown relationships on contributing paths remain unknown. Descriptions are resolved from registered operation semantics and current IR; cloning and textual persistence carry the operands and attributes needed to query them again.

Generated-kernel matchers check these relationships against the indexing and arithmetic that their emitters support. The RNS matcher accepts direct position reads, compact-plaintext expansion, and its supported NTT stages; the Tensor matcher accepts its supported pointwise and mapped reads. Missing required facts leave the operation outside that fusion region. Implementation coverage, effects, layout, placement, and table requirements remain part of selection.

## What does a Backend implementation expose?

`backend/implementation.py` defines the execution protocol. An implementation declares `name`, `operation_types`, and `supports_in_place`. Its `resource_requirements(invocation)` lists genuine non-Tensor handles. `execute(invocation, inputs, resources, *, in_place)` returns Tensor results using numerical operands and bound handles.

`OperationInvocation` contains the operation class, operand/result counts, operation attributes, and represented operand/result state required by execution. Eager constructs it directly from a method call; Compile constructs it while resolving IR operations.

Optional interfaces add narrowly defined preparation:

- `tensor_requirements(operation, config)` declares missing numerical operands and related attributes.
- `PreparingOperationImplementation.prepare_operation(operation)` prepares a represented operation or region for direct execution.
- `FusionImplementation.match_fusion(operations)` reports support for candidate regions.
- `RegionOperationImplementation.execute_regions(...)` executes an operation with prepared Tensor-region callables.

Numerical algorithms live beside their implementations. `backend/assembly.py` assembles contributions; it contains no arithmetic execution loops. Table construction belongs to the corresponding RNS, NTT, or CKKS numerical owner.

## How does registry resolution work?

`OperationImplementationRegistry` indexes contributed implementations by operation class and stable name. Duplicate identities are rejected. Candidate discovery reports names in declaration order. A requested name must exist; without an assignment, exactly one candidate must be available. Ambiguity is an execution-selection requirement.

`AssignImplementationsPass` maps full textual operation names to implementation names. It writes assignments without asserting executable coverage. Existing assignments are preserved unless replacement is requested through its `overwrite` control. A custom pass may assign different implementations to individual instances.

This is an executable configuration snippet for a later Pipeline:

```python
from fhelium import compile as fc
from fhelium.backend import OperationBackend
from fhelium.ir.dialects import rns

backend = OperationBackend()
choices = backend.registry.available(rns.AddStandardOp)
assignment = fc.AssignImplementationsPass({
    rns.AddStandardOp.name: "native-rns-linear",
})
```

Check the selected registry's candidate names before applying an assignment in an application. Implementation assignments remain constraints through transformation and linking. A selected implementation's execution failure is reported rather than retried through another implementation.

## How is whole execution distinguished from lowering?

`CkksLoweringDefinition` associates a source operation with a named rewrite and optional default designation. Its callback receives the operation, optional configuration, and current Compilation, then returns replacement operations, the result SSA value, and material descriptions. `CkksLoweringRegistry` resolves these definitions independently of execution registration.

A whole key-switch implementation invokes the shared RNS algorithms internally. A key-switch lowering exposes digit ModUp, NTT, key products, accumulation, and ModDown as Program operations. Both implement the same key-switch relation; the exposed form gives Compile additional scheduling and fusion opportunities.

`SelectExecutionLoweringsPass` in the common default pipeline selects locally using available whole implementations and generated fusion support, under existing assignments and represented facts. Native whole-operation coverage can remain selected while surrounding operations are lowered or fused. Pass reports expose these selection decisions.

## How are NTT choices expressed?

NTT semantics specify direction, normalization, and input/output residue forms. `AssignNttImplementationPass` expresses algorithm and schedule constraints, while `SelectNttImplementationsPass` asks the selected implementation to complete supported choices from available facts.

```python
from fhelium import compile as fc
from fhelium.backend import OperationBackend

backend = OperationBackend()
ntt_selection = fc.Pipeline((
    fc.AssignNttImplementationPass(
        algorithm="radix2_compact", group_width=8
    ),
    fc.SelectNttImplementationsPass(backend.registry),
))
```

Native algorithm families include indexed radix-2, compact radix-2, and fixed radix. Algorithm/table layout, schedule parameters, and Backend implementation identity are related but distinct choices. A concrete implementation assignment applies to logical NTT operations, so place it after a lowering that introduces those operations. Missing device facts can leave selection unresolved.

[Materials and preparation](materials-and-preparation.md) describes declared table operands. [RNS and NTT](rns-and-ntt.md) describes transform layouts and arithmetic invariants.

## Where does native device dispatch occur?

A registered Backend implementation can compose several generated wrappers under `native/wrapper/`. Each wrapper invokes a `torch.ops.fhelium_*` schema. Backend-neutral C++ defines that schema; CPU and CUDA translation units attach kernels with `TORCH_LIBRARY_IMPL`. Torch selects the device registration from Tensor dispatch keys.

An IR implementation name therefore identifies a numerical execution mechanism, while Torch's dispatch key selects a native device implementation of a schema. The [architecture overview](engine-native-stack.md) traces this shared native path and ABI loading.

## Adding an extension

For a new operation, define its schema, mathematical transition, effects, and dialect-local specification. For a lowering, add a named definition beside the rewrite and preserve external uses, supplied operands, and represented facts. For an implementation, contribute the execution object through its owning assembly and state numerical Tensor and non-Tensor requirements separately.

Validate the implemented behavior and selection outcome for the supported layouts and devices. Native schema changes additionally require wrapper regeneration and ABI-aware validation described in [native operator workflow](native-operator-workflow.md). A task-oriented example is provided by [select an operation implementation](../how-to/select-operation-implementation.md).
