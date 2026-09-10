# Import and transform textual Program IR

**Example source:** [`examples/18_ir_textual_program.py`](https://github.com/VisualDust/fhelium/blob/main/examples/18_ir_textual_program.py)

Example 18 starts from textual mixed-level IR. It parses and prints the Program, verifies a stable textual round trip, inserts a caller-defined analysis pass into the Compile pipeline, and inspects the transformed CKKS operations.

## Run the example

```bash
python examples/18_ir_textual_program.py
```

No cryptographic runtime or device is required because the example stops at the transformed Program.

## Textual Program

The input module contains one function with:

- an encrypted semantic value;
- a public message gain;
- a cyclic roll;
- encrypted addition;
- encrypted-by-public multiplication.

The module also carries FHElium schema and dialect versions. `fhelium.ir.parse(...)` loads the registered FHElium and upstream xDSL operations while preserving the mixed abstraction levels represented in the text.

## Stable serialization

The example performs:

```python
imported = ir.parse(text, source_name="inline-compile-example.mlir")
serialized = imported.to_text()
round_tripped = ir.parse(serialized)
```

It then checks that printing the second Program reproduces the first serialized form. This verifies structural round-trip stability for the selected Program; it does not prove numerical equivalence or executability.

## Caller-defined analysis pass

`RecordDialectInventoryPass` implements the public pass protocol:

```python
from fhelium import compile as fh_compile

@dataclass(frozen=True)
class RecordDialectInventoryPass:
    name: str = "record-dialect-inventory"

    def run(self, program, shared_data):
        ...
        shared_data["textual-ir/dialect-counts"] = counts
        return fh_compile.PassResult.unchanged(...)
```

The pass reads the current Program, counts operation namespaces, publishes its result through the request's schema-free `CompileWorkspace`, and returns an unchanged `PassResult`. It does not add attributes to the Program merely to transport analysis data.

The example places this pass before dead-value elimination:

```python
pipeline = fh_compile.Pipeline(
    (
        RecordDialectInventoryPass(),
        fh_compile.EliminateDeadValuesPass(),
        fh_compile.LowerSemanticToLogicalPass(),
        fh_compile.InsertPlaintextPreparationPass(),
        fh_compile.InsertMultiplyNttTransitionsPass(),
        fh_compile.LowerLogicalToCkksPass(),
        fh_compile.InsertRelinearizationPass(),
        fh_compile.InsertRescalePass(),
    )
)
```

Pipeline composition preserves the listed execution order. The analysis pass receives the same workspace as every subsequent Compile pass.

## Partial lowering

The remaining pipeline lowers semantic roll, addition, and mixed multiplication through logical operations into CKKS operations. The resulting Program contains the representation transitions and plaintext preparation selected by those passes.

The example also derives evaluation-key requirements from the transformed IR. This operation-depth analysis remains separate from key creation and runtime resource binding.

## Why use textual IR

Textual Program IR is useful when a compiler developer needs to:

- inspect a transformation independently of source capture;
- construct a minimal reproduction for a pass;
- exchange a source-independent Program;
- preserve unknown extension operations for another consumer;
- compare structural output before and after a transformation.

The parser accepts permissive mixed-level Programs. It does not turn structural acceptance into a CKKS correctness or distributed-safety proof.

::: details Source

<<< @/../examples/18_ir_textual_program.py

:::

## Next steps

- [Compose and execute built-in Compile passes](compose-and-execute-compile-pipeline.md) creates the same Program representation from a callable and lowers it for Backend execution.
- [Customize a Compile pass and pipeline](customize-compile-pass-and-pipeline.md) demonstrates a rewriting pass.
- [Neutral IR Programs](../concepts/neutral-ir-programs.md) defines Program structure and serialization.
