# Visualize and inspect a JIT Program

`SvgGraphVisualizationPass` renders one selected JIT entry as an inspectable SSA/dataflow SVG. Use it to compare Program snapshots, identify repeated operation patterns, inspect lowering attributes, and trace producer-consumer relationships without executing or modifying the Program.

The pass owns entry traversal, stable SSA naming, dependency edges, Graphviz construction, and file handling. `SvgGraphPresentation` independently owns operation rows and tooltips, while `SvgGraphTheme` owns colors and operation color classification. This separation allows a custom presentation without copying traversal code.

## 1. Choose the Program snapshot

Render the snapshot that answers the current question. A captured Program exposes semantic and preserved Torch operations; a lowered Program exposes lowered CKKS operations and scheduling decisions.

```python
from pathlib import Path

from fhelium.experimental import jit
from fhelium.experimental.jit.passes.visualize_svg import (
    SvgGraphPresentation,
    SvgGraphTheme,
    SvgNodeSection,
    default_svg_operation_color_key,
)

source_program = captured.program
lowered = jit.default_pipeline().run(
    source_program,
    captured.workspace,
)
lowered_program = lowered.program
workspace = lowered.workspace

output_dir = Path("graph_exports")
```

Use different output paths for before/after comparisons. The pass creates missing parent directories but refuses to overwrite an existing file unless `overwrite=True` is selected.

## 2. Render a focused lowered graph

```python
presentation = SvgGraphPresentation(
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
)

jit.SvgGraphVisualizationPass(
    output_dir / "lowered.svg",
    presentation=presentation,
).run(lowered_program, workspace)
```

Arguments, constants, operations, and outputs remain distinct. Equal operation color keys receive equal colors. The default key is the exact operation name; preserved `torch.call` operations also include call kind and target, allowing repeated public subgraphs to remain recognizable. The SVG canvas is transparent, and node fills use a varied FHElium-derived light palette with deep-neutral text.

Rendering requires the Python `pydot` package and the system Graphviz `dot` executable.

## 3. Select node-record fields

`fields=None` displays every supported section. Pass a non-empty set to `SvgGraphPresentation` to control the record layout:

| Field | Rendered evidence |
| --- | --- |
| `name` | Stable SSA result or block-argument name. |
| `opcode` | Exact xDSL operation name, such as `fhelium.ckks.add`. |
| `role` | Encrypted, message, plaintext, material, or resource result role. |
| `operands` | Ordered SSA producer names consumed by the node. |
| `result_types` | Exact xDSL result types and their carried representation-state metadata. |
| `attributes` | Selected non-internal xDSL operation attributes, one row per attribute. |
| `scheduling_obligations` | Lowering work still outstanding on the operation. |
| `num_users` | Total number of SSA uses of the operation results. |

A graph for rewrite-pattern analysis commonly needs `opcode`, `operands`, `attributes`, and `num_users`. Add `result_types` when the pattern depends on role or representation state. Omit sections that do not contribute to the current decision; complete type and attribute text can make a large Program unnecessarily wide.

Select top-to-bottom, bottom-to-top, left-to-right, or right-to-left layout when invoking the pass:

```python
jit.SvgGraphVisualizationPass(
    output_dir / "left-to-right.svg",
    presentation=presentation,
    rank_direction="LR",
).run(lowered_program, workspace)
```

## 4. Control operation attributes

The `attributes` field and `attribute_names` perform different selections:

- omitting `attributes` from `fields` removes all attribute selection and rows;
- including `attributes` with `attribute_names=None` retains every selected operation attribute;
- an `attribute_names={...}` allowlist retains only exact names present on each operation.

The default presentation always excludes:

- `fhelium.scheduling_obligations`, because it has an independent field; and
- `op_name__`, because it is an xDSL dynamic-operation implementation detail.

Every retained attribute is printed in its own `attr:<name>` row. Missing allowlist names are ignored. Operation attributes and result types remain different evidence: attributes record operation parameters and lowering markers, while `result_types` records SSA result type/state.

### Long attribute values

The default `attribute_preview_chars=180` retains at most 180 source characters from each attribute value in the Graphviz record. A shortened row appends the exact omitted-character count, so the final row text is deliberately longer than the retained prefix:

```text
attr:fhelium.call.arguments=... [247 chars omitted]
```

The row's complete attribute text remains in the aggregate SVG node tooltip. Set `attribute_preview_chars=None` only when complete values should participate directly in Graphviz layout:

```python
full_attributes = SvgGraphPresentation(
    fields={"name", "opcode", "attributes"},
    attribute_preview_chars=None,
)

jit.SvgGraphVisualizationPass(
    output_dir / "full-attributes.svg",
    presentation=full_attributes,
).run(lowered_program, workspace)
```

For a large traced Program, prefer an `attribute_names` allowlist before disabling the preview bound. In particular, `fhelium.call.arguments` may contain a complete structured call description.

## 5. Customize complete operation rows

Subclass the presentation rather than the visualization pass. `SvgGraphVisualizationPass` should continue to own traversal and file production. The presentation exposes these pure hooks:

| Hook | Responsibility |
| --- | --- |
| `select_attributes(operation)` | Return sorted `(name, Attribute)` pairs after filtering. |
| `format_attribute_value(operation, name, attribute)` | Produce one attribute's record preview. |
| `operation_sections(context)` | Return all final `SvgNodeSection` rows for one operation. |
| `operation_tooltip(context, sections)` | Build aggregate hover text from the already-rendered rows. |

`operation_sections()` is the general row-customization hook. A subclass can call the base implementation and then rename, reorder, remove, or inject rows. This example promotes `scale` to the first dedicated row while preserving its complete tooltip evidence:

```python
class ScaleFirstPresentation(SvgGraphPresentation):
    def operation_sections(self, context):
        sections = super().operation_sections(context)
        scale_rows = tuple(
            SvgNodeSection("scale", row.value, row.tooltip)
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

jit.SvgGraphVisualizationPass(
    output_dir / "scale-first.svg",
    presentation=scale_view,
).run(lowered_program, workspace)
```

`SvgOperationContext` supplies the operation plus stable result and operand names. `SvgNodeSection.value` participates in layout; its optional `tooltip` contributes complete or alternate evidence to the containing node's hover text. Presentation hooks must not mutate the operation, Program, or Workspace. `operation_sections()` must return a tuple containing only `SvgNodeSection` values; invalid output fails before Graphviz construction.

## 6. Customize colors coherently

`SvgGraphTheme` contains the operation palette, operation-key color overrides, color-key callable, transparent canvas color, input/output fills, text color, node stroke, and edge color. Pass one theme into a presentation instead of overriding module constants.

```python
def ckks_family(context):
    name = context.opcode
    if name.startswith("fhelium.ckks."):
        return "ckks"
    return default_svg_operation_color_key(context)

custom_theme = SvgGraphTheme(
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

custom_view = SvgGraphPresentation(
    fields={"name", "opcode", "operands"},
    theme=custom_theme,
)

jit.SvgGraphVisualizationPass(
    output_dir / "custom-theme.svg",
    presentation=custom_view,
).run(lowered_program, workspace)
```

An exact `operation_colors` entry takes precedence over the palette. Other keys are deterministically mapped into the non-empty palette. Supplying `operation_colors={}` intentionally removes the default special color for `fhelium.constant`; omitting it retains that default. The hash algorithm remains an implementation detail, while equal keys under one theme are guaranteed to receive equal colors.

When a custom classifier handles only selected operation families, delegate all other contexts to `default_svg_operation_color_key()` as above. This preserves the default distinction among `torch.call` function and method targets instead of collapsing every preserved Torch operation into one color.

## 7. Compare pass snapshots

Render adjacent snapshots with the same presentation:

```python
comparison_view = SvgGraphPresentation(
    fields={"name", "opcode", "operands", "attributes", "num_users"},
    attribute_names={"condition", "scale_mode", "shift"},
)

jit.SvgGraphVisualizationPass(
    output_dir / "captured.svg",
    presentation=comparison_view,
).run(source_program, workspace)

jit.SvgGraphVisualizationPass(
    output_dir / "lowered.svg",
    presentation=comparison_view,
).run(lowered_program, workspace)
```

Stable operation colors make repeated kinds comparable, while SSA names, operands, attributes, and user counts expose structural changes. The visualization pass reports matched operations and returns the Program unchanged; it does not establish structural validity, execution readiness, numerical correctness, or key availability beyond the checks already performed by the Program and selected pipeline.

## Related documentation

- [JIT programs](../concepts/unified-jit-programs.md)
- [Trace, transform, and run a JIT Program](../tutorial/unified-jit.md)
- [Customize and audit a JIT pass pipeline](../tutorial/jit-custom-pipeline.md)
- [`SvgGraphVisualizationPass` API](../api/fhelium/experimental/jit/passes/visualize_svg.md)
