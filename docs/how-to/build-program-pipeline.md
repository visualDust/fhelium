# Build a Program pipeline and write passes

A `Program` describes a calculation as typed operations and dataflow. A `Compilation` carries that Program, a workspace, one symbol-to-Tensor material dictionary, and pass reports. Use this procedure to import or capture a calculation, apply a chosen pipeline, and retain an inspectable transformed Program.

## Prerequisites

Install FHElium's Python dependencies. Parsing and the analysis-only pipeline below require neither a CKKS keyset nor device execution. Decide whether the calculation is most naturally expressed as a Python function or directly as IR; both produce the same compilation input.

## 1. Describe the calculation directly

The following complete textual Program computes a public sum. State fields can remain unknown while a consumer does not need them:

```python
from fhelium import compile as fc, ir

program = ir.parse(r'''builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.2"
} {
  func.func @main(
    %x: !fhelium_semantic.public<{}>,
    %y: !fhelium_semantic.public<{}>
  ) -> !fhelium_semantic.public<{}> {
    %sum = "fhelium_semantic.add"(%x, %y)
      : (!fhelium_semantic.public<{}>, !fhelium_semantic.public<{}>)
      -> !fhelium_semantic.public<{}>
    func.return %sum : !fhelium_semantic.public<{}>
  }
}''', source_name="sum.mlir")
compilation = fc.Compilation(program)
```

Use `ir.Program.load(path)` to read textual IR from a file. Program construction through dialect operations is another option when an application generates IR. The Program describes numerical material operands by symbols; live payloads belong in `compilation.material_bindings`.

## 2. Alternatively, capture Python expressions

Role-declared capture describes which input is encrypted without first allocating a ciphertext:

```python
import torch
import fhelium as fh

config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7)

def rotated_sum(x: torch.Tensor) -> torch.Tensor:
    return x + torch.roll(x, shifts=1, dims=-1)

captured = fc.capture(
    rotated_sum,
    inputs={"x": fc.encrypted(
        slots=config.num_slots,
        polynomial_domain="coefficient",
        residue_representation="standard",
    )},
    workspace=fc.CompileWorkspace({fh.CkksConfig: config}),
)
```

`fc.capture_eager(function, arguments={...})` instead captures Eager calls and ordinary Tensor expressions with representative values. It records the actual dataflow and fixed Tensor materials. Choose that frontend when CKKS transitions are already expressed by operations in the Python function. Capture does not supply missing keys or move values to a selected device.

## 3. Run only the transformations the task needs

```python
inspection_pipeline = fc.Pipeline((
    fc.EliminateDeadValuesPass(),
    fc.LowerSemanticToLogicalPass(),
))
transformed = inspection_pipeline.run(compilation)
print(transformed.program.to_text())
for report in transformed.reports:
    print(report.name, report.stats, report.diagnostics)
```

`Pipeline.run` clones the input Program and returns a new Compilation. The returned object retains the workspace and the same material-binding dictionary, so pass-produced materials remain available to later passes. Copy the dictionary when preparing independent binding assignments; that shallow copy still shares Tensor storage.

This partial pipeline is an observable transformation product. It need not produce an executable, and unknown value facts remain unknown until a pass or input specialization supplies them.

## 4. Prepare a numerical execution schedule

For a standard lowering-and-fusion recipe, construct and inspect an ordinary pipeline:

```python
from fhelium.backend import OperationBackend

backend = OperationBackend()
recipe = fc.default_lower_and_fuse_pipeline(backend)
print(recipe.names)
```

The default recipe does not insert rescaling or relinearization. For high-level encrypted multiplication, choose those policies before assigning CKKS depths and scales:

```python
scheduled = recipe.before(
    "assign-ckks-depths",
    fc.InsertRelinearizationPass(),
    fc.InsertRescalePass(),
)
```

Supply the entry depth and actual scale where they are known, using `AssignCkksDepthsPass(entry_depth=...)` and `AssignCkksScalesPass(entry_scale=...)`, or provide represented input state through callable specialization. Use `replace` to replace the corresponding recipe passes. Do not assign guessed facts merely to make an intermediate Program appear executable.

## 5. Execute the directly constructed Program

The public-sum Program from step 1 can use the full recipe and link directly. Its arithmetic takes ordinary Tensor operands and needs no external numerical materials:

```python
ready = recipe.run(compilation)
executable = backend.link(ready)
x = torch.arange(8, dtype=torch.float64, device="cpu")
y = torch.full_like(x, 0.5)
result = executable.run(x, y)
torch.testing.assert_close(result, x + y)
print(result)
print(executable.manifest)
```

The result contains `0.5, 1.5, ..., 7.5`. The semantic public addition becomes a registered Torch call; its input and output remain public Tensors. This route constructs, transforms, links, and executes a calculation without Python capture or a callable wrapper. For an encrypted calculation, complete its CKKS schedule and supply the required keys and tables before the same linking step.

## 6. Write a pass

A Compile pass inspects or transforms the current Compilation and returns a `PassResult` containing its Program and activity report. Use a pass to implement an application transformation, add a domain-specific lowering, or collect analysis at a selected stage. The examples below continue with `compilation` and `inspection_pipeline` from the previous steps.

### Implement an analysis pass

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

### Insert it at the relevant stage

Use `pipeline.before(name, pass_)`, `after`, or `replace` when the target name occurs once. Use `then` to append an analysis after all current steps. Inspect `pipeline.names` rather than assuming that another recipe uses the same order.

The pipeline checks Program structure after each pass. Validate a numerical rewrite with the clear-message comparison in the last step.

### Add a numerical rewrite

For a rewrite, traverse a snapshot such as `tuple(program.walk())` when replacing operations during traversal. Match the operation and the facts the algebra requires, create replacement operations with the appropriate value states, replace uses, and report the changes. Leave unsupported patterns unchanged or report a concrete unmet requirement.

If a pass derives a constant Tensor, assign it once under a Program material symbol in `compilation.material_bindings` and introduce a matching material-reference operand. A material description communicates purpose; it does not generate the Tensor or prevent a caller from replacing it. Preserve unknown state when the rewrite does not establish that state.

`examples/14_compile_custom_pass.py` implements a complete constant-matrix baby-step/giant-step lowering. It reads a captured matrix Tensor, constructs cyclic diagonals, creates named material operands, rewrites the matrix operation, and applies a caller-selected CKKS schedule. Use that example to carry the packing assumptions through a complete numerical transformation.

### Validate the transformation

First compare the transformed algebra with a clear-message reference using the same packing and cyclic-rotation convention. Then execute the CKKS result and measure its error against that reference. Check component count, depth, actual scale, prime rows, and representation at relevant transitions. Compare ciphertext residues only when both paths execute the same deterministic numerical schedule.

Retain before/after Programs and pass reports for [graph inspection](visualize-mixed-level-ir.md). The observable outcome is the intended operation change with a validated result. See [IR, capture, and passes](../developer/ir-capture-and-passes.md) and [operation declarations and selection](../developer/operation-registration-and-selection.md) for implementation contracts.

## 7. Choose the next product

To execute, [supply materials and link](save-and-relink-program.md), then call `executable.run(...)`. To use a function interface over this same Compilation, construct `fc.compile(compilation, backend=backend, pipeline=recipe)`. To inspect graph structure, [render a selected Program](visualize-mixed-level-ir.md). To extend the transformation, write a pass as in step 6.

[Example 13](../tutorial/compose-and-execute-compile-pipeline.md) runs a captured pipeline with caller-generated keys, material preparation, and linking, and a textual Program through a partial pipeline. See [IR, capture, and passes](../developer/ir-capture-and-passes.md) for representation and pass mechanisms.
