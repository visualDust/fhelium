# Compilation state and pass composition

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

`PassResult` records counts, diagnostics, and optional `DecisionRecord` values. Pipeline runs passes in the supplied order and verifies the returned Program's structure after each pass.

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

A pass must preserve the computation's equations, effects, external uses, and public interface unless its documented purpose changes that interface. Rescale uses the product of a complete depth group, and scale propagation uses actual per-value scales. Representation transitions must match the consuming implementation's domain and residue form.

Keep unresolved facts unresolved when a partial transformation can proceed. Report a concrete missing requirement when the requested transformation cannot. A requested implementation remains fixed through transformation and linking; an execution failure is reported to the caller.

For an extension, inspect the owning family under `compile/passes/` and implement the smallest transformation with observable mathematical behavior. Use the reports to expose decisions; avoid embedding execution resources or generated Tensor contents in portable operation attributes.
