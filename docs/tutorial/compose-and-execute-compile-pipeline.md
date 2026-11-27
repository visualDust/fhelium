# Program pipelines and textual IR

**Example source:** [`examples/13_compile_pipeline.py`](https://github.com/VisualDust/fhelium/blob/main/examples/13_compile_pipeline.py)

Example 13 exposes the individual stages of the Compile workflow. Its first part captures a supported PyTorch computation, composes existing FHElium passes, lowers the Program to operations covered by the Backend, links keys and arithmetic resources, executes encrypted inputs, and reports error against the clear result. Its second part starts from textual mixed-level IR, prints it back, and runs the front of the pipeline with a caller-defined analysis pass.

## Run the example

From the repository root:

```bash
python examples/13_compile_pipeline.py
```

The example runs on CPU. The first part prints captured and backend-ready Program inventories, evaluation-key requirements, result state, numerical error, and pass reports with host durations; the second prints the textual round-trip result, the recorded dialect counts, the pass reports, and the printed Program.

For a managed function interface over the same compiler, use [callable compilation](compile-layer.md). Manual `Compilation`, `Pipeline` and Backend linking remain available when the application owns transformation order and material binding. Passing a Program or Compilation to `fc.compile` also uses synchronous preparation.

## Source computation

The source adds a vector to one cyclic rotation and squares the sum:

```python
def rotated_quadratic(x, rotation):
    mixed = x + torch.roll(x, shifts=rotation, dims=-1)
    return mixed * mixed
```

Capture assigns `x` the encrypted role and treats `rotation` as a compile-time integer. The logical 16-slot input is tiled across the complete CKKS slot ring before encryption, so a full-ring rotation implements the same periodic layout used by the clear computation.

## Caller-selected pipeline

Pipelines express workload-specific choices. `fh_compile.default_lower_and_fuse_pipeline(backend)` supplies an editable default recipe for manual Programs; inspect its `.names` or compose its passes when those defaults suit the workload. This example writes the pass sequence to show its relinearization, rescale and NTT choices, and reports each `PassReport.seconds` after execution. The example supplies the configuration and implementation registry through the capture workspace:

```python
from fhelium.backend import OperationBackend, OperationImplementationRegistry
from fhelium import compile as fh_compile

backend = OperationBackend()
workspace = fh_compile.CompileWorkspace({
    CkksConfig: config,
    OperationImplementationRegistry: backend.registry,
})
```

Pass the workspace to `capture`. The pipeline then composes the built-in transformations:

```python
from fhelium import compile as fh_compile

pipeline = fh_compile.Pipeline(
    (
        fh_compile.EliminateDeadValuesPass(),
        fh_compile.LowerSemanticToLogicalPass(),
        fh_compile.InsertMultiplyNttTransitionsPass(),
        fh_compile.LowerLogicalToCkksPass(),
        fh_compile.ResolveRotationKeyOperandsPass(),
        fh_compile.InsertRelinearizationPass(),
        fh_compile.InsertRescalePass(),
        fh_compile.AssignCkksDepthsPass(entry_depth=0),
        fh_compile.AssignCkksScalesPass(
            entry_scale=config.scale_at(0),
        ),
        fh_compile.AssignNttImplementationPass("native-ntt", ntt_backend="radix2_indexed"),
        fh_compile.LowerCkksToRnsNttPass(),
        fh_compile.SelectNttImplementationsPass(),
    )
)
compiled = pipeline.run(captured)
```

This sequence records the following caller-selected pipeline choices:

- it inserts immediate relinearization and rescale operations for the ciphertext product;
- it assigns concrete depths and per-value actual scales after transition placement;
- it declares the configured key layout;
- it selects the indexed radix-2 NTT schedule and lowers CKKS arithmetic to RNS/NTT operations.

Another caller may choose late transition placement, additional analyses, a different lowering route, or a pipeline that returns an intermediate Program.

Lowering and implementation selection declare their required materials using the workspace registry. Custom key layouts can be supplied through `workspace["key_layouts"]`, keyed by material symbol or label; see [Materials and preparation](../developer/materials-and-preparation.md#evaluation-key-layout-declarations) for the fields and selection rules.

## Keys, resources, and linking

`analyze_evaluation_key_requirements(...)` finds the required rotation and relinearization uses in the compiled Program. The example creates compatible keys and uses `prepare_material_bindings` to supply the numerical materials for execution:

```python
fh_compile.prepare_material_bindings(
    compiled,
    resources=resources,
    keys=(*rotation_keys, *((relinearization_key,) if relinearization_key is not None else ())),
)
executable = backend.link(compiled)
```

`prepare_material_bindings` supplies Tensor data for the declared symbols, and linking builds an executable from the resulting bindings.

## Encrypted execution

The example encrypts one tiled input, invokes the executable through the public `Ciphertext` boundary, decrypts the result, and compares it with the retained `CapturedCallable` reference:

```python
result = executable.run(encrypted_x)
decoded = engine.decrypt_message(result, secret_key)
expected = captured_callable.reference(clear_x)
```

Registered Backend implementations remain Tensor-oriented internally. `ProgramExecutable` performs the public value adaptation at the outer execution interface.

## Textual Program input

`textual_pipeline` parses a module that contains one function with an encrypted semantic value, a public message gain, a cyclic roll, encrypted addition, and encrypted-by-public multiplication. The module carries FHElium schema and dialect versions; `fhelium.ir.parse(...)` loads the registered FHElium and upstream xDSL operations and keeps the mixed abstraction levels of the text.

```python
imported = ir.parse(text, source_name="inline-compile-example.mlir")
serialized = imported.to_text()
round_tripped = ir.parse(serialized)
```

Printing the second Program reproduces the first serialized form, which is the structural round trip the example reports. A parsed Program enters the same pipeline as a captured one through `fh_compile.Compilation(program, workspace)`.

## Caller-defined analysis pass

`RecordDialectInventoryPass` implements the pass protocol: it reads the current Program, counts operation namespaces, publishes the counts in the request's `CompileWorkspace`, and returns an unchanged `PassResult`.

```python
@dataclass(frozen=True)
class RecordDialectInventoryPass:
    name: str = "record-dialect-inventory"

    def run(self, compilation):
        counts = Counter(op.name.split(".", maxsplit=1)[0] for op in compilation.program.walk())
        compilation.workspace["textual-ir/dialect-counts"] = dict(sorted(counts.items()))
        return fh_compile.PassResult.unchanged(compilation.program, matched=sum(counts.values()))
```

The example places this pass first and follows it with dead-value elimination, semantic-to-logical lowering, plaintext preparation, NTT transition insertion, logical-to-CKKS lowering, relinearization, and rescale insertion. Every pass in the pipeline receives the same workspace. The pipeline stops at CKKS operations, so the second part needs no keys or device; evaluation-key requirements are derived from the transformed IR.

Textual IR serves to inspect a transformation independently of source capture, to write a minimal reproduction for a pass, and to exchange a source-independent Program. The parser accepts permissive mixed-level Programs; CKKS state assignment and Backend linking perform the checks that make a Program executable.

## Relation to Example 14

Example 13 uses only operations and passes supplied by FHElium. [Example 14](customize-compile-pass-and-pipeline.md) defines a pass that rewrites captured `torch.matmul` into a baby-step/giant-step schedule before composing, linking, and executing its pipeline.

::: details Source

<<< @/../examples/13_compile_pipeline.py

:::

## Leaving NTT choices open

Logical NTT operations can leave their algorithm unassigned. Add `SelectNttImplementationsPass()` where the required facts become available, then supply any newly introduced table materials before linking. `AssignNttImplementationPass` can constrain an algorithm, group width or fixed radix without assigning every execution detail. See [NTT semantics and selection](../developer/compiled-execution-and-kernels#ntt-semantics-and-selection) for the selection order and table-data responsibilities.
