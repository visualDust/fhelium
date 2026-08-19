# Import and execute textual JIT IR

**Example source:** [`examples/20_jit_textual_ir.py`](https://github.com/VisualDust/fhelium/blob/main/examples/20_jit_textual_ir.py)

Example 20 imports a versioned mixed-dialect xDSL `Program` from text, verifies
a stable textual round trip, records requirements with a custom pass, binds an
application operation handler through a retained `Workspace`, checks readiness,
and executes an exact public Tensor computation.

This workflow is IR-first and CPU-only. It demonstrates textual interchange and
extension operations without constructing a CKKS engine or generating keys.

## Run the complete example

From the repository root:

```bash
python examples/20_jit_textual_ir.py
```

The script prints:

- the imported operation, unknown-operation, and Torch-target requirements;
- readiness diagnostics before and after the extension handler is bound;
- reports from the custom pass and executable validator;
- the exact input, output, and expected Tensor values;
- the canonical round-tripped textual `Program`.

No CUDA device is required.

## 1. Define a versioned mixed-dialect module

The example starts from textual xDSL rather than a PyTorch callable:

```text
builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.1"
} {
  func.func @main(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
    %bias = "fhelium.constant"() {fhelium.literal = "0.75"}
      : () -> !fhelium.message<{}>
    %negated = "torch.call"(%x) {
      fhelium.call.kind = "function",
      fhelium.call.target = "torch.neg",
      fhelium.call.arguments = "..."
    } : (!fhelium.message<{}>) -> !fhelium.message<{}>
    %result = "application.scale_and_shift"(%negated, %bias)
      : (!fhelium.message<{}>, !fhelium.message<{}>)
     -> !fhelium.message<{}>
    func.return %result : !fhelium.message<{}>
  }
}
```

One SSA graph contains four vocabularies:

| Operation/type | Responsibility |
| --- | --- |
| `builtin.module`, `func.func`, `func.return` | xDSL module, function, and return structure |
| `!fhelium.message<{}>`, `fhelium.constant` | FHElium public-value role and literal |
| `torch.call` targeting `torch.neg` | Preserved audited public Torch computation |
| `application.scale_and_shift` | Custom operation preserved for a bound handler |

The schema and dialect version attributes select the current versioned
execution semantics at readiness. The open parser retains the application operation even
though FHElium does not register its dialect.

`fhelium.call.arguments` contains a JSON descriptor for positional and keyword
arguments. The complete escaped attribute is maintained in the example source;
it states that operand zero is the sole positional argument to `torch.neg`.

## 2. Parse and verify the structure

Import the package and parse the module:

```python
from fhelium.experimental import jit

imported = jit.parse(
    _PROGRAM_TEXT,
    source_name="inline-application.mlir",
)
```

`jit.parse()` creates the package's permissive xDSL context, parses registered
and unregistered dialect content, and constructs a `Program`. Construction
verifies xDSL structure, including SSA uses and registered operation
constraints. It does not require an operation handler, engine, key inventory,
or numerically complete CKKS state.

The `source_name` is diagnostic provenance. It does not become a live file or a
runtime material binding.

## 3. Check a stable textual round trip

The example prints the parsed module, parses that canonical form again, and
requires the second print to be byte-for-byte equal:

```python
canonical_text = imported.to_text()
round_tripped = jit.parse(
    canonical_text,
    source_name="round-tripped-application.mlir",
)
if round_tripped.to_text() != canonical_text:
    raise RuntimeError("textual JIT Program round trip was not stable")
```

This check establishes stability for the current parser/printer representation
used by the example. It also proves that the unknown
`application.scale_and_shift` operation survives interchange.

Printing serializes the current module and symbolic references. It does not
serialize a `Workspace`, Python callable, Tensor, engine, key, or handler.
Callers that mutate `Program.module` directly own structural validity until a
constructor, pass pipeline, or readiness check verifies it again.

## 4. Analyze requirements without deciding readiness

Requirements are a pure scan of the selected entry:

```python
requirements = round_tripped.requirements()
```

For this program, the scan reports:

- operations: `application.scale_and_shift`, `fhelium.constant`, and
  `torch.call`;
- unknown operations: `application.scale_and_shift`;
- Torch targets: the exact `torch.neg` symbol;
- no CKKS engine requirement.

Requirements describe what the current graph contains. They do not authorize
an extension operation or decide whether its handler has correct semantics.

The example confirms that the unbound program is not runnable:

```python
unbound = round_tripped.readiness()
if unbound.runnable:
    raise RuntimeError(
        "the extension operation was ready without a handler"
    )
```

The executable validator also reports that the unknown operation lacks an
bound handler. Structural import therefore remains permissive while
execution fails closed.

## 5. Write a custom pass that retains analysis

`RecordRequirementsPass` implements the public pass protocol:

```python
@dataclass(frozen=True)
class RecordRequirementsPass:
    name: str = "record-imported-requirements"

    def run(
        self,
        program: jit.Program,
        workspace: MutableMapping[Any, Any],
    ) -> jit.PassResult:
        requirements = program.requirements()
        operation_surface = tuple(sorted(requirements.operations))
        workspace["analysis/imported-operation-surface"] = operation_surface
        return jit.PassResult.unchanged(
            program,
            matched=len(operation_surface),
            diagnostics=(
                f"recorded {len(operation_surface)} executable operation names",
            ),
        )
```

The pass does not rewrite IR. It publishes an application-owned analysis value
under a caller-chosen workspace key and returns an unchanged result with
observable counts and diagnostics.

A legal unchanged `PassResult` is a completed pass outcome. It does not pretend
that every operation is executable; the independent validator and readiness
gate make that decision.

## 6. Bind application policy outside the graph

The application operation is implemented by a trusted Python handler:

```python
def scale_and_shift(
    operation: Operation,
    operands: tuple[object, ...],
    workspace: MutableMapping[Any, Any],
) -> object:
    del operation
    if len(operands) != 2:
        raise ValueError("application.scale_and_shift requires two operands")
    source, bias = operands
    if not isinstance(source, torch.Tensor) or not isinstance(
        bias, (int, float)
    ):
        raise TypeError("scale_and_shift expects Tensor and real operands")
    gain = workspace.get("application/gain")
    if not isinstance(gain, (int, float)):
        raise TypeError("workspace['application/gain'] must be real")
    return source * gain + bias
```

The retained workspace supplies both the handler and its caller-owned policy:

```python
workspace = jit.Workspace(
    {
        "application/gain": 0.5,
        "handlers": {
            "application.scale_and_shift": scale_and_shift,
        },
    }
)
```

`handlers` maps an exact extension operation name to a callable with the
signature `(operation, evaluated_operands, workspace)`. Binding the handler is a
trust decision: FHElium checks that it is callable, but the application owns its
arithmetic, type behavior, side effects, and resource safety.

Built-in operation names are reserved and cannot be overridden through this
mapping. Preserved `torch.call` targets use the separate audited Torch path or
an exact `workspace["torch_handlers"]` binding.

## 7. Compose the custom pipeline

The example combines the analysis pass with the executable validator:

```python
pipeline = jit.PassPipeline(
    (
        RecordRequirementsPass(),
        jit.ValidateExecutableGraphPass(),
    )
)
transformed = pipeline.run(round_tripped, workspace)
```

`PassPipeline.run()` clones the source `Program` once, invokes both passes in
order, verifies the `Program` returned by each pass, and passes the exact same
workspace object to every step. The example asserts retained identity:

```python
if transformed.workspace is not workspace:
    raise RuntimeError("the custom pipeline did not retain its Workspace")
```

The first report records the imported operation surface. The second report
shows that the graph is executable with the exact handler map now present.
Neither pass invokes the handler.

## 8. Check readiness after binding

Readiness is observational:

```python
ready = transformed.program.readiness(transformed.workspace)
if not ready.runnable:
    detail = "; ".join(item.message for item in ready.diagnostics)
    raise RuntimeError(f"the bound textual Program is not ready: {detail}")
```

The check validates the selected entry, current operation schemas, version
attributes, audited Torch target, extension handler presence, and workspace
container schemas. It does not execute `torch.neg`, call
`scale_and_shift`, or mutate application caches.

The expected transition is:

| Workspace state | `runnable` | Relevant diagnostics |
| --- | --- | --- |
| No extension handler | `False` | `invalid-executable-graph`, `missing-handler` |
| Retained handler and gain | `True` | none |

The gain itself is handler-owned policy. Readiness cannot infer that a
particular handler will read `workspace["application/gain"]`; the handler
performs that check when invoked.

## 9. Execute and require exact output

The runtime input is a public CPU Tensor:

```python
value = torch.tensor([1.0, -2.0, 0.25], dtype=torch.float64)
actual = jit.run(
    transformed.program,
    value,
    workspace=transformed.workspace,
)
expected = -value * 0.5 + 0.75
torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
```

Execution evaluates:

$$
y = 0.5(-x) + 0.75.
$$

For the maintained input, both the actual and expected results are

```text
[0.25, 1.75, 0.625]
```

The comparison uses zero relative and absolute tolerance because this is an
ordinary public Tensor computation with the same PyTorch operations on both
paths. This example makes no CKKS approximation claim.

`jit.run()` repeats readiness, binds the entry argument, interprets the audited
`torch.neg`, then invokes the extension handler at its exact operation. A
handler failure remains an execution failure rather than being converted into
successful readiness.

## Complete source

<<< @/../examples/20_jit_textual_ir.py

## Continue

- [Trace, transform, and run a JIT program](unified-jit.md) covers typed PyTorch
  capture and encrypted CKKS execution.
- [Compose a custom JIT pass pipeline](jit-custom-pipeline.md) covers pass
  insertion, explicit CKKS auditing, key planning, and CUDA execution.
- [JIT programs](../concepts/unified-jit-programs.md) defines structural and
  readiness validity, Program/Workspace ownership, and control modes.
- [JIT internals](../developer/unified-jit-internals.md) specifies the exact
  textual schemas, pass scope, and handler authorization.
