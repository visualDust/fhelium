# IR, capture, effects, and open state

`fhelium.ir.Program` owns an xDSL module representing computation through typed SSA values, operations, blocks, and regions. Programs can be constructed, parsed, captured from Python, cloned, transformed, and exported. This article describes the represented computation; [Compilation and passes](compilation-and-passes.md) describes the request state carried alongside it.

## What does a Program establish?

`ir/_program.py` provides the module interface, parsing and text formatting, structural verification, and function inspection. A Program may mix FHElium dialects with supported upstream xDSL operations. `ir/dialects/` owns FHElium's operation classes and state-bearing types.

| Representation | Question answered |
| --- | --- |
| Semantic input roles | Which values are encrypted, plaintext, messages, or static? |
| Logical arithmetic | Which encrypted/public arithmetic is requested? |
| CKKS operations | Which polynomial, key, scale, and modulus transitions are requested? |
| RNS and NTT operations | Which residue arithmetic, basis conversion, and transforms are represented? |
| Core, memory, distributed, and structured operations | Which external values, movement, effects, and control flow participate? |
| Fusion regions | Which represented operations will execute jointly through a selected implementation? |

These abstraction levels can coexist. A caller can retain a whole CKKS key switch beside separately represented RNS arithmetic. Structural validity does not require every operation to have a Backend implementation.

## Which state can remain unknown?

State dictionaries describe known mathematical facts such as depth, scale, components, prime identities, basis, domain, and residue form, together with physical facts relevant to preparation. Open types preserve missing configuration, state, and device facts for later assignment.

A transformation derives only the facts supported by its operands and operation semantics. A pass may report that an operation awaits configuration, layout, or another assignment. An execution consumer eventually requires enough information for its Tensor ABI or public value adapter. The time at which a fact becomes required depends on that consumer: successful parsing, transformation, linking, and execution are different checks.

`ir/_state.py` defines state representation, `ir/_analysis.py` supplies Program analyses, and `ir/ckks_state.py` supplies the CKKS metadata equations. A transparent IR cast can refine a type without changing Tensor bytes; NTT, Montgomery conversion, and rescale must remain numerical operations when they change those bytes.

## How are semantics and effects recorded?

Dialect modules declare operation classes and their `OPERATION_SPECS`. `ir/_operation_catalog.py` defines `OperationSpec` and the catalog interface; `ir/_operation_specs.py` assembles the dialect-owned descriptions. Specifications state operand roles, arity, semantics, and effects for validation and inspection. The [operation index](ir-operation-implementation-index.md) links operation families to their mathematical and numerical owners.

Effects determine which transformations are legal. `compile/passes/program/_purity.py` interprets registered effects, and dead-value elimination retains operations whose effects must occur even when their results are unused. Mutable random streams, transfers, communication, and unknown effects constrain removal, reuse, and movement. A pure arithmetic region is eligible only when its nested operations satisfy the applicable rules.

Storage identity also matters. Common-subexpression reuse must preserve externally visible allocation and mutation behavior. A fusion region must retain an intermediate as an output when an external use still needs it. SSA dependency analysis alone cannot justify reordering an effectful operation.

## What do the capture frontends produce?

Both public frontends return a `Compilation` containing a Program and live Tensor material bindings.

| Frontend | Mechanism | Material and state handling |
| --- | --- | --- |
| `compile.capture` | PyTorch FX capture with declared input roles | Emits semantic arithmetic and supported Torch calls; retains captured Tensor data |
| `compile.capture_eager` | Symbolic execution of supported Engine-call compositions | Preserves independent CKKS argument states and actual key/table Tensor dataflow |

FX input specifications are declared with `encrypted`, `plaintext`, `message`, and `static`. Recognized arithmetic is available to semantic lowering. A retained `torch.call` still needs supported execution or a later transformation.

Eager capture substitutes Engine adapters through a private copy of referenced globals and closure bindings. Supported pure Python control flow can use captured scalar arguments as static values. Ordinary Tensor metadata propagation uses FakeTensor support. Data-dependent Python branching on encrypted payloads, arbitrary object behavior, unsupported mutation, encryption, and key generation are outside this frontend's captured subset.

Supported encoding and plaintext-preparation calls become numerical Program operations. Key-dependent calls retain actual Tensor operands from supplied key values or captured Engine data. Missing data stays represented as a material reference for later preparation.

A compiled ordinary-function helper is captured through its original source under the outer pipeline. Its prepared executable is not invoked during tracing. Captured callable metadata records the source and Python argument/output structure while the current Program interface remains compatible with that record.

## How do manual and callable Compile use this IR?

Manual execution starts from a Program or captured Compilation, applies a Pipeline, then links through an `OperationBackend`. Callable Compile uses the same representation and pass machinery while managing input specialization and executable reuse. Neither route requires an alternative IR or execution engine.

When a manual transformation changes the Program interface, its current arguments become authoritative. A captured Python signature cannot describe an arbitrary replacement interface. Construct a callable for the transformed interface when needed. [Prepared host execution](prepared-host-execution.md) explains argument adaptation and specialization reuse.

## Extending the representation

Add an operation to its dialect and dialect-local specification collection, with the equations and effects needed by consumers. Implement transformations in the owning pass family and execution in the appropriate Backend family. Keep unknown operations visible to external xDSL/MLIR transformations until a consumer supports them.

Parsing a Program does not sandbox extensions. Caller-defined passes, Backend implementations, Torch calls, native code, and generated kernels execute code with the process's authority. Validate provenance separately from structural correctness.
