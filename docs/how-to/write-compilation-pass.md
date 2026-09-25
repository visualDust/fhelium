# Write and insert a Compilation pass

A Compile pass inspects or transforms the current Compilation and returns a `PassResult` containing its Program and activity report. Use a pass to implement an application transformation, add a domain-specific lowering, or collect analysis at a selected stage.

## Prerequisites

Start with the `compilation` and `inspection_pipeline` from [Build and transform a Program](build-program-pipeline.md), or use a Compilation produced by either capture frontend. Identify the operation family and the stage at which the required facts are available.

## 1. Implement an analysis pass

```python
from collections import Counter
from dataclasses import dataclass
from fhelium import compile as fc

@dataclass(frozen=True)
class CountOperationsPass:
    name: str = "count-operations"

    def run(self, compilation: fc.Compilation) -> fc.PassResult:
        counts = Counter(op.name for op in compilation.program.walk())
        compilation.workspace["analysis/operation-counts"] = dict(counts)
        return fc.PassResult.unchanged(
            compilation.program,
            matched=sum(counts.values()),
            diagnostics=(f"observed {len(counts)} operation names",),
        )

analyzed = inspection_pipeline.then(CountOperationsPass()).run(compilation)
print(analyzed.workspace["analysis/operation-counts"])
print(analyzed.reports[-1])
```

`run(compilation)` reads the current Program, `workspace`, and `material_bindings` together. Store transient analysis in the workspace and portable numerical meaning in IR. Keep the pass's return report consistent with the operations it matched and transformed.

## 2. Insert it at the relevant stage

Use `pipeline.before(name, pass_)`, `after`, or `replace` when the target name occurs once. Use `then` to append an analysis after all current steps. Inspect `pipeline.names` rather than assuming that another recipe uses the same order.

The pipeline checks Program structure after each pass. Validate a numerical rewrite with the clear-message comparison described below.

## 3. Add a numerical rewrite when needed

For a rewrite, traverse a snapshot such as `tuple(program.walk())` when replacing operations during traversal. Match the operation and the facts the algebra requires, create replacement operations with the appropriate value states, replace uses, and report the changes. Leave unsupported patterns unchanged or report a concrete unmet requirement.

If a pass derives a constant Tensor, assign it once under a Program material symbol in `compilation.material_bindings` and introduce a matching material-reference operand. A material description communicates purpose; it does not generate the Tensor or prevent a caller from replacing it. Preserve unknown state when the rewrite does not establish that state.

`examples/14_compile_custom_pass.py` implements a complete constant-matrix baby-step/giant-step lowering. It reads a captured matrix Tensor, constructs cyclic diagonals, creates named material operands, rewrites the matrix operation, and applies a caller-selected CKKS schedule. Use that example to carry the packing assumptions through a complete numerical transformation.

## 4. Validate the transformation's meaning

First compare the transformed algebra with a clear-message reference using the same packing and cyclic-rotation convention. Then execute the CKKS result and measure its error against that reference. Check component count, depth, actual scale, prime rows, and representation at relevant transitions. Compare ciphertext residues only when both paths intentionally execute the same deterministic numerical schedule.

Retain before/after Programs and pass reports for [graph inspection](visualize-mixed-level-ir.md). The observable outcome is the intended operation change with a validated result. See [Compilation and passes](../developer/compilation-and-passes.md) and [operation declarations and selection](../developer/operation-registration-and-selection.md) for implementation contracts.
