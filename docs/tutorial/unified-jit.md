# Trace, transform, and run a JIT program

**Example source:** [`examples/19_unified_jit.py`](https://github.com/VisualDust/fhelium/blob/main/examples/19_unified_jit.py)

Example 19 traces a typed PyTorch square matrix-vector computation into one
mixed-dialect xDSL [`Program`](../concepts/unified-jit-programs.md), applies an
selected lowering pipeline, provisions the evaluation keys required by the
resulting operations, and runs the program with a retained workspace.

The example evaluates an `8 × 8` affine map in repeated packed slot blocks,
then applies

$$
p(z)=(z+0.25)(z-0.5)
$$

and a caller-owned plaintext output gain. Public matrix preparation remains as
preserved Torch operations. Arithmetic that mixes encrypted values is represented
by FHElium semantic operations and then lowered to explicit CKKS operations.

## Run the complete example

From the repository root:

```bash
python examples/19_unified_jit.py \
  --preset slots8192-scale40-levels7-int64
```

The script runs on the selected CPU or CUDA engine and prints the final textual
program, a report for every selected pass, the output level and actual scale,
and the measured maximum absolute error against its semantic reference.

## 1. Define one semantic function

The input function uses ordinary PyTorch syntax:

```python
def square_matvec_quadratic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output_gain: torch.Tensor,
    matrix_size: int,
    repeats: int,
) -> torch.Tensor:
    main_diagonal = torch.diagonal(weight).repeat(repeats)
    affine = x * main_diagonal

    for shift in range(1, matrix_size):
        wrapped_head = torch.diagonal(
            weight,
            offset=matrix_size - shift,
        )
        ordinary_tail = torch.diagonal(weight, offset=-shift)
        cyclic_diagonal = torch.cat((wrapped_head, ordinary_tail))
        packed_diagonal = cyclic_diagonal.repeat(repeats)
        rotated = torch.roll(x, shifts=shift, dims=-1)
        affine = affine + rotated * packed_diagonal

    affine = affine + bias.repeat(repeats)
    quadratic = (affine + 0.25) * (affine - 0.5)
    return quadratic * output_gain
```

For rotation $r$ and matrix row $j$, the packed diagonal is

$$
d_r[j] = W[j,(j-r)\bmod 8].
$$

The packed affine computation is therefore

$$
a = \sum_{r=0}^{7} d_r \odot \operatorname{roll}(x,r) + b.
$$

Each public `torch.diagonal`, `torch.cat`, and `repeat` call remains in the
mixed-dialect program as `torch.call`. Each supported operation whose result
depends on the encrypted input is captured as an `fhelium.semantic.*`
operation. The static loop bounds let FX unroll the seven nonzero rotations.

## 2. Trace with typed input roles

Import the unified package directly:

```python
from fhelium.experimental import jit
```

Every function parameter receives one input declaration:

```python
captured = jit.trace(
    square_matvec_quadratic,
    inputs={
        "x": jit.encrypted(),
        "weight": jit.message(),
        "bias": jit.message(),
        "output_gain": jit.plaintext(),
        "matrix_size": jit.static(8),
        "repeats": jit.static(engine.num_slots // 8),
    },
)
```

| Declaration | Runtime meaning | Semantic reference meaning |
| --- | --- | --- |
| `encrypted()` | A compatible core `Ciphertext`, or a Tensor encrypted online with `workspace["public_key"]`; the declaration carries level, scale, slot-extent, and batch policy | Tensor |
| `message()` | Public Python/PyTorch data; public-only computation remains a preserved Torch call until an encrypted consumer requires preparation | The same public value |
| `plaintext()` | A caller-owned core [`Plaintext`](../api/fhelium/core/plaintext.md#plaintext) whose representation and CKKS state match its consumer | Public Tensor/scalar shadow |
| `static(value)` | A finite immutable scalar specialized during capture and omitted from the runtime signature | The specialized value restored by `CaptureResult.reference()` |

`encrypted()` defaults to level zero and the selected engine's default scale at
execution. `slots="full"` requires the complete engine slot axis; a positive
integer declares a fixed logical final-axis extent. `batch_mode="none"`
requires one slot axis, while `"any"` permits leading batch axes.

`message()` and `plaintext()` describe distinct public roles. A message is
semantic Python/PyTorch data that can pass through public preprocessing. A
plaintext is already a FHElium encoded value with its representation,
level, scale, basis, domain, and residue state.

## 3. Inspect the capture result

`jit.trace()` returns a `CaptureResult`, not a second program type:

```python
source_program = captured.program
workspace = captured.workspace

print(source_program.to_text())
print(captured.fx_code)
print(captured.runtime_signature)
```

The result provides:

- `program`: the canonical mixed-dialect xDSL `Program`;
- `workspace`: retained graph-external materials and caller state;
- `signature`, `specs`, and `fx_code`: frontend evidence;
- `reference(...)`: execution of the original callable with static values
  restored.

Tensor constants captured from Python are cloned into
`workspace["materials"]`. Their SSA positions contain only symbolic
`fhelium.material.ref` operations, so `Program.save()` does not serialize live
Tensor values.

The source program is structurally valid immediately after capture. It can
contain semantic operations and therefore need not yet be ready for the current
interpreter:

```python
source_report = source_program.readiness(workspace)
print(source_report.runnable)
print(source_report.diagnostics)
```

## 4. Apply a selected pass pipeline

Example 19 selects the general lowering policy:

```python
lowered = jit.default_pipeline().run(
    captured.program,
    captured.workspace,
)
program = lowered.program
workspace = lowered.workspace
```

A pipeline clones the source `Program` once, runs the ordered pass tuple over
the clone, and returns the same workspace object. The source program remains
available for comparison.

The default pipeline currently runs:

1. unreachable pure-value elimination;
2. semantic-to-logical role classification;
3. operation-specific plaintext preparation insertion;
4. ciphertext-multiply NTT transition insertion;
5. logical-to-explicit-CKKS lowering;
6. relinearization insertion;
7. rescale insertion;
8. conservative late-rescale and late-relinearization passes.

Each pass handles only the local operations and roles it recognizes. A pass
with no applicable pattern returns a legal unchanged result. Inspect the full
behavior through `lowered.reports`:

```python
for report in lowered.reports:
    print(report.name, report.stats, report.diagnostics)
```

The pass counters distinguish matches, transformations, insertions, removals,
and skips. The pipeline structurally verifies every pass result before invoking
the next pass. Completion records the selected transformations; the independent
readiness check decides whether the result can run.

### Visualize the lowered Program

`SvgGraphVisualizationPass` renders the selected entry's lowered SSA/dataflow
graph without changing the `Program`:

```python
from pathlib import Path

from fhelium.experimental.jit.passes.visualize_svg import (
    SvgGraphPresentation,
)

output_path = Path("graph_exports/example19-lowered.svg")
output_path.parent.mkdir(parents=True, exist_ok=True)

visualization = jit.SvgGraphVisualizationPass(
    output_path,
    overwrite=True,
    presentation=SvgGraphPresentation(
        fields={
            "name",
            "opcode",
            "role",
            "operands",
            "attributes",
            "scheduling_obligations",
            "num_users",
        },
        attribute_names={
            "condition",
            "fhelium.call.target",
            "operation",
            "scale_mode",
            "shift",
        },
    ),
).run(program, workspace)
print(visualization.stats)
```

The pass draws entry arguments, explicit operations, SSA dependencies, result
roles, and the function output. Arguments, constants, and outputs have
dedicated colors. Operations of the same kind share a stable color selected
from a diverse light palette derived from FHElium's cobalt, spectral-blue,
helium-amber, warm-bridge, and neutral colors. Every fill uses the same
high-contrast deep-neutral text color. Preserved `torch.call` nodes also
distinguish their function or method target. `fields` selects complete
node-record sections, while `attribute_names` optionally limits the xDSL
attributes shown inside the `attributes` section.

Rendering requires the Python `pydot` package and the system Graphviz `dot`
executable. The default `overwrite=False` refuses to replace an existing file;
the example opts into replacement so the command is repeatable.

[![Example 19 lowered JIT Program rendered as an SSA/dataflow graph](/figures/jit-example19-lowered-program.svg)](/figures/jit-example19-lowered-program.svg)

*Figure 1. The Example 19 `main` entry after the default lowering pipeline.
Open the SVG to inspect the complete 87-operation graph at full resolution.*

Use [Visualize and inspect a JIT Program](../how-to/visualize-jit-program.md)
to select state/type fields, control long attributes, compare pass snapshots,
or customize node-record rows.

## 5. Analyze the transformed requirements

Analyze evaluation-key requirements after lowering, because the analysis scans
the current explicit CKKS operations:

```python
key_requirements = jit.analyze_evaluation_key_requirements(program)

evaluation_keys = fh.EvaluationKeySet(
    rotations=fh.RotationKeySet(
        {
            step: engine.rotation_key(step)
            for step in key_requirements.rotation_steps
        }
    ),
    relinearization=(
        engine.relinearization_key
        if key_requirements.requires_relinearization
        else None
    ),
)
```

Example 19's packed matrix-vector program requires rotation steps 1 through 7.
Its quadratic contains a ciphertext-ciphertext multiplication and therefore
requires relinearization.

`program.requirements()` provides the broader pure scan, including current
operation names, symbolic materials/resources, preserved Torch targets,
unknown operation names, engine requirement, and return count. This analysis
does not inspect the workspace or decide whether execution should proceed.

## 6. Populate the retained workspace

Add live evaluator services to the same workspace:

```python
workspace.update(
    {
        "engine": engine,
        "evaluation_keys": evaluation_keys,
    }
)
```

The program contains the computation and symbolic identities. The workspace
contains the live engine, keys, captured materials, handlers, resources,
policies, and caches for this request. Keeping these objects graph-external
allows the same textual program to be paired with another compatible runtime.

When an encrypted argument is supplied as a Tensor, online encryption also
requires:

```python
workspace["public_key"] = engine.public_key
```

Example 19 supplies a caller-created `Ciphertext`, so online encryption is not
part of its execution requirements.

## 7. Check readiness before execution

Readiness compares the selected entry with the current workspace without running
or materializing anything:

```python
report = program.readiness(workspace)
if not report.runnable:
    for diagnostic in report.diagnostics:
        print(diagnostic.code, diagnostic.subject, diagnostic.message)
    raise RuntimeError("program is not ready")
```

The report covers:

- the versioned program schema and dialect;
- one structurally executable selected entry;
- built-in operation schemas and cleared scheduling obligations;
- trusted handlers for extension operations or FHE-touching Torch
  targets;
- symbolic material and resource bindings;
- an engine and the required evaluation keys.

`program.run()` performs the same gate and raises `ProgramNotReadyError` with
its complete report when a requirement is missing.

## 8. Build semantic and encrypted inputs

The gain has separate semantic and encoded forms:

```python
output_gain_value = 0.75
output_gain_plaintext = engine.plaintext(
    output_gain_value,
    level=2,
    scale=engine.config.default_scale,
)
```

The capture result's reference receives the semantic scalar:

```python
reference = captured.reference(
    packed_x,
    weight,
    bias,
    output_gain_value,
)
```

The JIT program receives the core plaintext and an encrypted packed input:

```python
encrypted_x = engine.encrypt_message(packed_x, engine.public_key)

result = program.run(
    encrypted_x,
    weight,
    bias,
    output_gain_plaintext,
    workspace=workspace,
)
```

Input names recorded by capture allow either positional or keyword binding. A
caller-owned `Ciphertext` must match the workspace engine's context, device,
dtype, ring dimension, declared level and actual scale, and batch policy. A
`plaintext()` argument must be a core `Plaintext`; its representation
state is handled at the consuming preparation operation.

The interpreter executes preserved public Torch calls through its audited
public target table. A preserved Torch call that touches encrypted or plaintext
roles requires a target binding under `workspace["torch_handlers"]`.
Unknown extension operations require a handler under
`workspace["handlers"]`.

## 9. Decrypt and apply the example's acceptance criterion

Example 19 decrypts and compares every packed slot:

```python
decoded = engine.decrypt_message(
    result,
    engine.secret_key,
    is_real=True,
)
error = torch.abs(decoded - reference)
max_abs_error = float(error.max())
if max_abs_error > 3e-5:
    raise RuntimeError(
        "JIT execution exceeded its fixed CKKS validation threshold: "
        f"max_abs_error={max_abs_error:.3e}, atol={3e-5:.3e}"
    )
```

The fixed absolute tolerance belongs to this preset, circuit, and deterministic
input construction. It is evidence for this maintained example rather than a
general numerical guarantee for every JIT program.

## Alternative entry paths

The same `Program` model supports workflows that start outside PyTorch capture:

- [Example 20: Import and execute textual JIT IR](jit-textual-ir.md)
  parses versioned mixed-dialect text, retains an application operation, binds
  its handler, and executes the selected entry.
- [Example 21: Customize and audit a JIT pass pipeline](jit-custom-pipeline.md)
  demonstrates custom analysis/pass composition and the retained workspace.

For direct textual input:

```python
program = jit.parse(text, source_name="application.mlir")
# or: program = jit.load("application.mlir")

transformed = jit.default_pipeline().run(program, jit.Workspace())
print(transformed.program.to_text())
```

Parsing establishes structural validity. Select passes, handlers, materials,
and runtime services separately for the intended execution request.

## Complete source

<<< @/../examples/19_unified_jit.py

## Continue

- [JIT programs](../concepts/unified-jit-programs.md) defines the
  `Program`, workspace, validity, and control-level concepts.
- [JIT internals](../developer/unified-jit-internals.md) specifies xDSL,
  schemas, pass scope, and extension interfaces.
- [Textual JIT IR](jit-textual-ir.md) and
  [custom JIT pipelines](jit-custom-pipeline.md) continue with independent
  advanced workflows.
- [Input-role API](../api/fhelium/experimental/jit.md),
  [Program API](../api/fhelium/experimental/jit.md),
  [execution/readiness API](../api/fhelium/experimental/jit.md), and
  [pass API](../api/fhelium/experimental/jit/passes.md) provide current
  current signatures.
