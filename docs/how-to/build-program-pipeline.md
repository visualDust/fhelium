# Build and transform a Program

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

config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7_int64)

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

## 6. Choose the next product

To execute, [supply materials and link](persist-compiled-program.md), then call `executable.run(...)`. To use a function interface over this same Compilation, construct `fc.compile(compilation, backend=backend, pipeline=recipe)`. To inspect graph structure, [render a selected Program](visualize-mixed-level-ir.md). To extend the transformation, [write a Compilation pass](write-compilation-pass.md).

`examples/12_compile_pipeline.py` demonstrates a complete encrypted rotation-and-square schedule, caller-generated keys, material preparation, linking, and decoded validation. `examples/13_compile_textual_ir.py` demonstrates textual round trips and a partial pipeline without execution. See [Compilation and passes](../developer/compilation-and-passes.md) for representation and pass mechanisms.
