# Visualize mixed-level IR

`SvgGraphVisualizationPass` renders one selected `fhelium.ir.Program` function
as an SSA/dataflow SVG. Use it to compare Program snapshots, inspect repeated
operation patterns, examine lowering attributes, and trace producer-consumer
relationships without executing or modifying IR.

The pass owns entry traversal, SSA naming, dependency edges, Graphviz
construction, and file handling. `SvgGraphPresentation` owns node records and
tooltips; `SvgGraphTheme` owns colors and operation classification.

Visualization is a Compile pass over an IR Program. It requires a
`Compilation` only to participate in the ordinary pipeline and publish its
output path; it does not require source capture, lowering, a JIT session,
execution device, bindings, or a backend.

## 1. Obtain a Program snapshot

Parse, load, construct, or receive any structurally valid `Program`. This
standalone example parses a small semantic-level graph:

```python
from pathlib import Path

from fhelium import compile as fh_compile
from fhelium import ir

program = ir.parse(
    r'''builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.2"
} {
  func.func @main(
    %x: !fhelium_semantic.public<{}>
  ) -> !fhelium_semantic.public<{}> {
    %bias = "fhelium.constant"() {fhelium.literal = "0.75"}
      : () -> !fhelium_semantic.public<{}>
    %result = "fhelium_semantic.add"(%x, %bias)
      : (!fhelium_semantic.public<{}>, !fhelium_semantic.public<{}>)
     -> !fhelium_semantic.public<{}>
    func.return %result : !fhelium_semantic.public<{}>
  }
}
''',
    source_name="visualization-example.mlir",
)
output_dir = Path("graph_exports")
```

`ir.Program.load(path)` reads the same textual representation from a file.
Captured and compile-transformed Programs are also ordinary snapshots, but
neither producer is required for visualization.

## 2. Render a focused graph

```python
presentation = fh_compile.SvgGraphPresentation(
    fields={
        "name",
        "opcode",
        "role",
        "operands",
        "attributes",
        "num_users",
    },
    attribute_names={
        "condition",
        "fhelium.call.target",
        "operation",
        "scale_mode",
        "shift",
    },
)

visualized = fh_compile.Pipeline(
    (
        fh_compile.SvgGraphVisualizationPass(
            output_dir / "inspection.svg",
            presentation=presentation,
        ),
    )
).run(fh_compile.Compilation(program))
svg_output = visualized.workspace[fh_compile.SvgGraphOutput]
```

Arguments, constants, operations, and outputs remain visually distinct. Equal
operation color keys receive equal colors. Preserved `torch.call` operations
also include call kind and target in their default color key.

Rendering requires the Python `pydot` package and the system Graphviz `dot`
executable. It needs no JIT session, engine, key, material binding, or backend.

Omit `output_path` to create a unique file under the operating system's
temporary directory:

```python
visualized = fh_compile.Pipeline(
    (fh_compile.SvgGraphVisualizationPass(),)
).run(fh_compile.Compilation(program))
svg_output = visualized.workspace[fh_compile.SvgGraphOutput]
assert svg_output.temporary
print(svg_output.path)
```

When `output_path` is provided, the pass writes to that path instead and
records `temporary=False`. `CompileWorkspace` records the result but does not
own a general temporary directory or redirect files produced by other passes.
The pass does not delete a temporary SVG after returning; callers may inspect,
copy, or remove the recorded path when it is no longer needed.

## 3. Select node fields

`fields=None` displays every supported section. A nonempty set selects the
record layout:

| Field | Rendered evidence |
| --- | --- |
| `name` | Stable SSA result or block-argument name |
| `opcode` | xDSL operation name, such as `fhelium_ckks.add` |
| `role` | Encrypted, message, plaintext, material, or resource role |
| `operands` | Ordered SSA producer names |
| `result_types` | xDSL result types and open state metadata |
| `attributes` | Selected operation attributes, one row per attribute |
| `num_users` | Total SSA uses of the operation results |

A rewrite-pattern view commonly needs `opcode`, `operands`, `attributes`, and
`num_users`. Add `result_types` when a decision depends on value roles or state.
Omit sections that do not contribute to the current inspection.

Inspect CKKS transition placement through the actual
`fhelium_ckks.relinearize`, `fhelium_ckks.rescale`, and
`fhelium_ckks.mod_switch` nodes, their SSA edges, and the state carried by
result types. The visualization does not synthesize a separate pending-work
field.

Select rank direction at construction:

```python
fh_compile.Pipeline(
    (
        fh_compile.SvgGraphVisualizationPass(
            output_dir / "left-to-right.svg",
            presentation=presentation,
            rank_direction="LR",
        ),
    )
).run(fh_compile.Compilation(program))
```

Supported directions are top-to-bottom (`TB`), bottom-to-top (`BT`),
left-to-right (`LR`), and right-to-left (`RL`).

## 4. Control operation attributes

The `attributes` field and `attribute_names` answer different questions:

- omit `attributes` to remove all attribute rows;
- use `attribute_names=None` to include every selected operation attribute;
- provide an allowlist to include only listed attributes that are present.

The default presentation excludes xDSL's dynamic-operation implementation
marker from general attribute rows.

Each included attribute receives one `attr:<name>` row. Missing allowlist names
are ignored. Attributes and result types are separate evidence: operation
attributes carry operation parameters and implementation selections, while
result types carry SSA value role and arithmetic state.

### Limit long values

`attribute_preview_chars=180` includes at most 180 source characters from each
attribute value in the record. A shortened row reports the omitted character
count, while the complete text remains in the aggregate node tooltip.

```python
full_attributes = fh_compile.SvgGraphPresentation(
    fields={"name", "opcode", "attributes"},
    attribute_preview_chars=None,
)
```

For a large captured Program, select attribute names before disabling the
preview limit. `fhelium.call.arguments`, for example, can contain a complete
structured call descriptor.

## 5. Customize operation records

Subclass the presentation rather than the visualization pass. The presentation
exposes pure hooks:

| Hook | Responsibility |
| --- | --- |
| `select_attributes(operation)` | Return sorted `(name, Attribute)` pairs |
| `format_attribute_value(operation, name, attribute)` | Produce a record preview |
| `operation_sections(context)` | Return final `SvgNodeSection` rows |
| `operation_tooltip(context, sections)` | Build hover text from rendered rows |

This example promotes `scale` to a dedicated first row:

```python
class ScaleFirstPresentation(fh_compile.SvgGraphPresentation):
    def operation_sections(self, context):
        sections = super().operation_sections(context)
        scale_rows = tuple(
            fh_compile.SvgNodeSection("scale", row.value, row.tooltip)
            for row in sections
            if row.name == "attr:scale"
        )
        other_rows = tuple(
            row for row in sections if row.name != "attr:scale"
        )
        return (*scale_rows, *other_rows)


scale_view = ScaleFirstPresentation(
    fields={"name", "opcode", "role", "attributes"},
)
```

`SvgOperationContext` supplies the operation plus assigned result and operand
names. `SvgNodeSection.value` participates in layout; its optional tooltip adds
complete or alternate evidence to the containing node.

Presentation hooks must not mutate operations or the Program.
`operation_sections()` must return only `SvgNodeSection` values.

## 6. Customize colors

`SvgGraphTheme` contains the operation palette, color overrides, color-key
callable, canvas, input/output fills, text, strokes, and edges.

```python
def ckks_family(context):
    if context.opcode.startswith("fhelium_ckks."):
        return "ckks"
    return fh_compile.default_svg_operation_color_key(context)


custom_theme = fh_compile.SvgGraphTheme(
    operation_palette=("#DCE4FF", "#CCE7EE", "#F7E5BC"),
    operation_colors={
        "ckks": "#CBD7FA",
        "fhelium.constant": "#E6E7EC",
    },
    operation_color_key=ckks_family,
    canvas_color="transparent",
    input_fill_color="#DCE4FF",
    output_fill_color="#F3D89D",
    node_font_color="#181B26",
    node_stroke_color="#596178",
    edge_color="#7B8193",
)

custom_view = fh_compile.SvgGraphPresentation(
    fields={"name", "opcode", "operands"},
    theme=custom_theme,
)
```

A matching override takes precedence over the palette. Other keys map
deterministically into the nonempty palette. When a classifier handles only
selected families, delegate other operations to
`default_svg_operation_color_key()` so preserved Torch call targets use
their default distinctions.

## 7. Compare snapshots

When another tool has produced `source_program` and `transformed_program`,
render both with one presentation. The visualization code depends only on
their `Program` interface:

```python
comparison_view = fh_compile.SvgGraphPresentation(
    fields={"name", "opcode", "operands", "attributes", "num_users"},
    attribute_names={"condition", "scale_mode", "shift"},
)

for filename, program_snapshot in (
    ("captured.svg", source_program),
    ("transformed.svg", transformed_program),
):
    fh_compile.Pipeline(
        (
            fh_compile.SvgGraphVisualizationPass(
                output_dir / filename,
                presentation=comparison_view,
            ),
        )
    ).run(fh_compile.Compilation(program_snapshot))
```

Stable color keys support visual comparison, while SSA names, operands,
attributes, and user counts expose structural changes. The visualization pass
returns the Program unchanged and reports the rendered operation count. It does
not establish CKKS consistency, numerical accuracy, key availability, or
backend coverage. It also does not lower the graph, build a native executable,
or provide a CKKS Triton implementation.

## Related documentation

- [Open compiler stack](../concepts/open-compiler-stack.md)
- [Neutral IR programs](../concepts/neutral-ir-programs.md)
- [Compose and execute built-in Compile passes](../tutorial/compose-and-execute-compile-pipeline.md)
- [Customize a Compile pass and pipeline](../tutorial/customize-compile-pass-and-pipeline.md)
