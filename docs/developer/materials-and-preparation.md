# Tensor materials and operation preparation

A material is Tensor data referenced by a Program symbol. `Compilation.material_bindings` maps those symbols to live Tensors, including evaluation keys, RNS parameters, NTT tables, codec tables, and mutable rounding state. Preparation declares missing operands and optionally obtains data from caller-supplied providers before execution is linked.

## What is data, and what is an execution handle?

| Representation | Stored value | Owner |
| --- | --- | --- |
| `MaterialRefOp` and `Compilation.material_bindings` | Numerical Tensor | Compilation and caller's Tensor ownership |
| `Program.material_descriptions` | Optional description of a material's purpose | Program metadata used for inspection and preparation |
| `ResourceRefOp` and `ResourceRequirement` | Symbol and kind of a non-Tensor execution handle | IR or selected implementation |
| `BackendWorkspace.named_resources` | `BoundResource` values with symbol, kind, and live object | Backend workspace |
| `BackendWorkspace.materializer` | Optional constructor of missing execution handles | Backend workspace |

Process groups and encryption samplers are execution handles. RNS parameter arrays and evaluation-key payloads are numerical data even when they are long-lived or expensive to construct. `BackendWorkspace` contains named handles and an optional handle materializer; the Tensor binding dictionary belongs to Compilation.

## How do material references enter a Program?

Capture retains actual Tensor objects from values, keys, and tables. It does not copy their contents. Lowering can introduce additional `MaterialRefOp` operands when it exposes arithmetic previously hidden inside a whole operation. `compile/_materials.py` allocates symbols and attaches optional descriptions without renumbering existing references.

Descriptions can record material kind, key role, algorithm-table layout, or parameter identity. They support optional data selection. A directly supplied binding remains the caller's assignment; descriptions guide optional selection without inspecting or overwriting that data.

`ResolveTensorPlaceholdersPass` reads the current binding dictionary and propagates available physical facts into placeholder types. Symbols absent from the dictionary remain unresolved during inspection and transformation; linking requires every material referenced by the executable.

## How are material requirements resolved?

`CompileWorkspace` stores the `OperationImplementationRegistry` used to query implementation requirements. Callable Compile supplies its selected Backend registry; manual Programs can provide the entry when constructing the workspace. Lowering, rotation hoisting and implementation selection use the shared material-resolution functions for their affected operations. The selected implementation's optional `tensor_requirements(operation, config)` supplies attributes and named material descriptions. Resolution adds missing Tensor operands while preserving supplied operands, existing symbols and implementation assignments. Insufficient configuration, placement or algorithm facts leave the corresponding requirements unresolved.

This mechanism applies to retained native whole operations as well as newly exposed lower-level operations. For example, a whole key switch needs its internal NTT schedule and parameter/table operands. A lowered key switch separates RNS ModUp and ModDown tables from the selected logical NTT operations' table requirements. RNS-only preparation can use `RnsContext`; transform tables require a compatible NTT provider and layout.

`ResolveMaterialRequirementsPass` applies the same resolution to an existing Program as a standalone operation. `SelectNttImplementationsPass` selects missing schedules under existing constraints and declares corresponding table references. Supplied table operands remain dataflow inputs. Selecting a compact radix-2 group width can reuse the compatible compact table layout; incompatible layouts must not be silently substituted.

## Evaluation-key layout declarations

Key operands carry their mathematical layout in `EvaluationKeyType.state`. `prime_ids` lists the stored Q prime IDs followed by the auxiliary prime IDs; each entry of `digit_prime_ids` lists the Q support of one stored key plane. Lowering uses these fields to derive active digit intersections and key-plane indices. Layout declarations and Tensor bindings can be supplied separately.

`workspace["key_layouts"]` holds a mapping from material symbols or labels to type-state fields such as `{"prime_ids": rows, "digit_prime_ids": digits}`. The standalone `ResolveMaterialRequirementsPass` also accepts this mapping as `key_layouts`. Preparation applies these constraints before selecting supplied keys. For symbols with neither a caller layout entry nor represented row or digit metadata, it selects the depth-zero Q/P layout from the available CKKS configuration. Partial declarations retain unspecified facts as unknown; layout-dependent lowering requires those facts to be supplied. References to the same material symbol share one layout.

Programs retain key and table inputs as material symbols. Their Tensor bindings may be provided during compilation or during execution preparation. Saving a Program without those bindings allows it to be loaded and bound to compatible key sets separately.

## How does optional data provision work?

`prepare_material_bindings(compilation, resources=..., keys=...)` fills absent entries and returns the symbols still unresolved. Resources can be existing `CkksDeviceResources`, `RnsContext` or `NttContext` providers. Keys can be named entries, an iterable of key values or an application-owned `EvaluationKeySet`.

Named entries matching a material symbol or label supply that binding directly and must satisfy the key operand's declared layout. An iterable supplies candidates whose represented metadata is matched against the material description; absent or ambiguous matches remain unresolved. Existing bindings are preserved. An `EvaluationKeySet` additionally exposes selection among alternative layouts and devices using the actual requirements of the consuming key operands. Active Q supports, P basis, physical rows, digit supports, ring dimension, dtype and device determine a suitable key. Shared material uses must admit one consistent selection. When some execution facts are unknown, selection requires a unique candidate compatible with all known facts; ambiguous or incompatible choices leave the symbol unresolved.

The inventory contains existing rotation, relinearization and conjugation values. Callers create or load those values before preparation and maintain their parameter context and secret relations. Capability or layout matching cannot establish those relations. `prepare_material_bindings` consumes the selected key Tensors, fills missing facts, and checks supplied key objects against already declared row, digit and physical-layout facts before binding them. Named raw Tensor bindings retain the caller's responsibility for the represented key semantics. Binding performs no key generation.

```python
from fhelium import compile as fc

def supply_materials(compilation, *, resources=(), keys=()):
    return fc.prepare_material_bindings(
        compilation, resources=resources, keys=keys
    )
```

Providers may construct or retrieve the numerical tables requested by this step. Subsequent material-table lookup returns the supplied Tensor. Callable Compile accepts `resources=` and `keys=` for execution preparation; supplied key metadata constrains layout selection, and binding fills the resulting material symbols before linking.

Prepared consumers such as Bootstrap discover encoded materials and selected key references in ordinary dictionaries. Compile capture of that prepared computation retains the resulting Tensor dataflow. Once selected, key references are independent of later inventory membership changes. The shared inventory supplies candidates; the prepared dictionaries and Compilation bindings retain the selected objects for their respective executions.

## What remains live after linking?

Linking resolves references to Tensor objects and builds a host callable retaining those objects. Replacing `compilation.material_bindings[symbol]` later does not rebind an existing executable; relink to use a compatible replacement. Loading a material-free Program into separate Compilations gives independent bindings for different key sets without repeating lowering. Changing the Q/P layout or digit partition requires a matching specialization. In-place writes to the bound Tensor remain visible because linking did not snapshot its contents.

Layout-specialized execution still requires its prepared shape, stride, dtype, extent, and placement contract. Mutating storage structure is different from updating compatible contents. Synchronization with active device work or concurrent writers remains the caller's responsibility.

Mutable rounding streams are numerical Tensor state with effects. Encryption's sampler is a live handle with effects. Preparation must not replace either with one frozen random sample. Saving selected bindings can capture sensitive or mutable data; see [Compilation persistence](compilation-persistence.md) for inclusion and security semantics.

## Extension points

An implementation that needs numerical operands can declare `tensor_requirements` and consume the resulting Tensors through the ordinary execution interface. Place table construction beside the numerical owner, such as `backend/rns/tables.py` or `backend/ntt/tables.py`. Use `resource_requirements` only for genuine non-Tensor handles. [Registration and selection](operation-registration-and-selection.md) describes how implementations enter the registry, and [prepared host execution](prepared-host-execution.md) describes final linking.
