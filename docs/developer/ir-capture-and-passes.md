# IR, capture, and passes

`fhelium.ir.Program` owns an xDSL module representing computation through typed SSA values, operations, blocks, and regions. Programs can be constructed, parsed, captured from Python, cloned, transformed, and exported. This article describes the represented computation, the capture frontends that produce it, and the `Compilation` and `Pipeline` state that passes transform.

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

When a manual transformation changes the Program interface, its current arguments become authoritative. A captured Python signature cannot describe an arbitrary replacement interface. Construct a callable for the transformed interface when needed. [Linking, exec schedules, and the Runner](linking-and-runner.md) explains argument adaptation and specialization reuse.

## Compilation state

`Compilation` carries a Program, a caller-extensible `CompileWorkspace`, ordered pass reports, and one `material_bindings` dictionary. `Pipeline` composes passes over this state. Manual Program execution and callable Compile use the same protocol and the same default lowering-and-fusion recipe.

## What is copied by a Pipeline?

`compile/_compilation.py` defines the request container and `compile/_pipeline.py` defines `Pass`, `PassResult`, and `Pipeline`. `Pipeline.run(compilation)` first clones the Program. It retains the workspace and material dictionary, then passes the current Compilation to each `Pass.run(compilation)` in order. Each result supplies the next Program and contributes a `PassReport`.

The frozen Compilation dataclass retains mutable workspace and Tensor data. Changes to the retained dictionary are visible to subsequent passes and to callers sharing it. For independent binding assignments, construct a Compilation with a copied dictionary and workspace mapping. Such shallow copies preserve live Tensor storage and custom object identity.

The following pass is executable Python and illustrates the current protocol without changing IR:

```python
from dataclasses import dataclass
from fhelium import compile as fc

@dataclass(frozen=True)
class InspectOperations:
    name: str = "inspect-operations"

    def run(self, compilation: fc.Compilation) -> fc.PassResult:
        count = sum(1 for _ in compilation.program.walk())
        return fc.PassResult.unchanged(
            compilation.program,
            matched=count,
            diagnostics=(f"Observed {count} operations",),
        )

inspection = fc.Pipeline((InspectOperations(),))
```

`PassResult` records counts, diagnostics, and optional `DecisionRecord` values. Pipeline runs passes in the supplied order. The default `verify=True` checks the returned Program's structure after each pass; `verify=False` checks after the final pass. Each `PassReport.seconds` records host wall time for the pass and its configured verification; device completion and numerical execution have separate timing scopes.

## Where do configuration and intermediate products live?

`CompileWorkspace` is a mapping for caller inputs and pass-owned products. For example, CKKS transformations read `CkksConfig`; material and implementation selection read `OperationImplementationRegistry`; resolution writes `ProgramDispatchTable`; linking writes `ProgramExecutable`. A workspace entry's owner defines its meaning and freshness requirements.

Tensor data belongs in `Compilation.material_bindings`, indexed by Program symbols. A pass that creates a new reference and one that consumes it see the same dictionary. Configuration and material descriptions guide operand preparation. Material lookup retrieves the Tensor bound to a symbol. See [materials and preparation](materials-and-preparation.md).

Reports accumulate across Pipeline invocations. They describe what passes observed or changed. Inspect the current Program together with diagnostics and implementation assignments when a later consumer cannot proceed.

## What does the common default recipe do?

`default_lower_and_fuse_pipeline` in `compile/_lower_and_fuse.py` returns an ordinary editable Pipeline. Its principal phases are:

1. Resolve available Tensor facts, remove dead values, and reuse compatible intermediates.
2. Lower semantic intent, resolve rotation-key operands, insert required multiplication and plaintext representation preparation, and lower logical arithmetic to CKKS.
3. Assign represented depth and scale, lower message preparation, and select available NTT schedules.
4. Prepare actual key facts before hoisting, hoist compatible rotations, and prepare operands of newly introduced semantic operations before probing their lowerings.
5. Select per-operation whole execution or a lowering, prepare newly exposed low-level operands, reuse intermediates, and form supported fusion regions.
6. Remove values made dead by the transformations.

The baseline preserves caller-inserted numerical transitions. It does not insert rescale or relinearization. Applications requesting those schedules can insert the relevant CKKS passes before dependent depth and scale assignment.

```python
from fhelium import compile as fc
from fhelium.backend import OperationBackend

backend = OperationBackend()
pipeline = fc.default_lower_and_fuse_pipeline(backend)
pass_names = pipeline.names
```

An omitted Backend creates the standard registry. The recipe selects per-operation execution through implementation coverage and fusion matching. For callable execution, pass optional `resources=` and `keys=` to `fc.compile`; execution preparation supplies the resulting material symbols before linking. Manual Programs use `prepare_material_bindings` for the same step.

`SelectExecutionLoweringsPass.execution_cost` optionally receives a whole operation and its isolated, fusion-processed lowering candidate. Return `(whole_cost, lowered_cost)` in one common unit, or `None` when no estimate applies. A lower cost wins; ties retain whole execution. Caller implementation assignments take precedence. `DecisionRecord` retains the supplied costs and selection. With no estimate, ordinary implementation-coverage and fusion rules remain in force.

A measured cost is specific to the executed implementation, parameter state and physical input conditions. Unknown strides or placement remain unknown in the Program. A caller using observed layouts must establish their correspondence to the current candidate; operation counts or fewer materialized bytes alone do not constitute a latency estimate.

Lowering and implementation selection complete material declarations for their affected operations using the registry in the workspace. `ResolveMaterialRequirementsPass` exposes the same logic for an explicit whole-Program resolution. The default recipe uses the operation-local integration.

## How are passes customized?

`then`, `before`, `after`, and `replace` return new Pipelines. Named insertion and replacement require one matching pass name. The default recipe repeats some fact-resolution and cleanup passes, so inspect `pipeline.names` and construct a tuple directly when the desired location is ambiguous.

A caller-supplied callable pipeline replaces the default recipe. It may also be selected by a function of `CallSignature`. A supplied linking pipeline separately replaces the standard linking sequence. Optimization and linking are independently composable because the former transforms computation while the latter binds execution.

## What makes a transformation correct?

A pass must preserve the computation's equations, effects, external uses, and public interface unless its documented purpose changes that interface. Rescale divides by the configured rescale ratio of its transition, and scale propagation uses actual per-value scales. Representation transitions must match the consuming implementation's domain and residue form.

Keep unresolved facts unresolved when a partial transformation can proceed. Report a concrete missing requirement when the requested transformation cannot. A requested implementation remains fixed through transformation and linking; an execution failure is reported to the caller.

For an extension, inspect the owning family under `compile/passes/` and implement the smallest transformation with observable mathematical behavior. Use the reports to expose decisions; avoid embedding execution resources or generated Tensor contents in portable operation attributes.

## Extending the representation

Add an operation to its dialect and dialect-local specification collection, with the equations and effects needed by consumers. Implement transformations in the owning pass family and execution in the appropriate Backend family. Keep unknown operations visible to external xDSL/MLIR transformations until a consumer supports them.

Parsing a Program does not sandbox extensions. Caller-defined passes, Backend implementations, Torch calls, native code, and generated kernels execute code with the process's authority. Validate provenance separately from structural correctness.
