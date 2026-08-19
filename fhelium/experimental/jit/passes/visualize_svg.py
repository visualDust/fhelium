"""Render a selected JIT Program entry as an SSA/dataflow SVG."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Mapping, MutableMapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Operation, SSAValue

from .._dialect import operation_name, value_role
from .._errors import JitPassError
from .._program import Program
from ._base import PassResult
from ._utils import OBLIGATIONS_ATTRIBUTE, display_name, obligations

SvgGraphField = Literal[
    "name",  # Stable SSA result or block-argument name.
    "opcode",  # Exact xDSL operation name, such as fhelium.ckks.add.
    "role",  # FHElium encrypted/message/plaintext/material/resource role.
    "operands",  # Ordered SSA value names consumed by the node.
    "result_types",  # Exact xDSL result types and carried state metadata.
    "attributes",  # Selected xDSL attributes, one record row per attribute.
    "scheduling_obligations",  # Explicit lowering work still outstanding.
    "num_users",  # Total number of SSA uses of the node's results.
]
SvgGraphDirection = Literal["TB", "BT", "LR", "RL"]

# Keep node record sections in one stable order regardless of caller set order.
_FIELD_ORDER: tuple[SvgGraphField, ...] = (
    "name",
    "opcode",
    "role",
    "operands",
    "result_types",
    "attributes",
    "scheduling_obligations",
    "num_users",
)
_ALL_FIELDS: frozenset[SvgGraphField] = frozenset(_FIELD_ORDER)

# Light operation fills derived from FHElium's cobalt, spectral-blue,
# helium-amber, warm-bridge, and neutral colors.
_DEFAULT_OPERATION_PALETTE = (
    "#CBD7FA",
    "#BAC8F1",
    "#E1DFF7",
    "#D0CCEE",
    "#CCE7EE",
    "#B9DCE5",
    "#CDE8E1",
    "#BBDDD3",
    "#D8E8CC",
    "#C9DEBB",
    "#F7E5BC",
    "#F1D499",
    "#EBC889",
    "#EAD8C7",
    "#DFC4AD",
    "#F1D8DD",
    "#E7C5CE",
    "#D6D9E3",
)
_DEFAULT_OPERATION_COLORS = {"fhelium.constant": "#E6E7EC"}


@dataclass(frozen=True, slots=True)
class SvgNodeSection:
    """Represent one labeled record row inside an SVG operation node.

    ``value`` participates in Graphviz layout. ``tooltip`` optionally retains
    complete or alternate evidence for the containing node's aggregate hover
    text; Graphviz does not attach a separate tooltip to each record cell.
    """

    name: str
    value: str
    tooltip: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("SVG node section name must be a non-empty string")
        if not isinstance(self.value, str):
            raise TypeError("SVG node section value must be a string")
        if self.tooltip is not None and not isinstance(self.tooltip, str):
            raise TypeError("SVG node section tooltip must be a string or None")


@dataclass(frozen=True, slots=True)
class SvgOperationContext:
    """Provide stable operation evidence to an SVG presentation policy."""

    operation: Operation
    result_names: tuple[str, ...]
    operand_names: tuple[str, ...]

    def __post_init__(self) -> None:
        result_names = tuple(self.result_names)
        operand_names = tuple(self.operand_names)
        if not all(isinstance(name, str) for name in result_names):
            raise TypeError("SVG operation result names must be strings")
        if not all(isinstance(name, str) for name in operand_names):
            raise TypeError("SVG operation operand names must be strings")
        object.__setattr__(self, "result_names", result_names)
        object.__setattr__(self, "operand_names", operand_names)

    @property
    def opcode(self) -> str:
        """Return the exact registered or dynamic operation name."""

        return operation_name(self.operation)


def _string_attribute(operation: Operation, name: str) -> str | None:
    attribute = operation.attributes.get(name)
    return attribute.data if isinstance(attribute, StringAttr) else None


def default_svg_operation_color_key(context: SvgOperationContext) -> str:
    """Return the default stable operation color key.

    Exact operation names define ordinary keys. Preserved ``torch.call``
    operations additionally include call kind and target so custom classifiers
    can delegate without reproducing that rule.
    """

    name = context.opcode
    if name == "torch.call":
        call_kind = (
            _string_attribute(context.operation, "fhelium.call.kind") or ""
        )
        target = (
            _string_attribute(context.operation, "fhelium.call.target") or ""
        )
        return f"{name}:{call_kind}:{target}"
    return name


def _validate_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SVG graph theme {field} must be a non-empty string")
    return value


class _ImmutableColorMap(Mapping[str, str]):
    """Store a canonical read-only color mapping for theme value semantics."""

    __slots__ = ("_items",)
    _items: tuple[tuple[str, str], ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, values: Mapping[str, str]) -> None:
        object.__setattr__(self, "_items", tuple(sorted(values.items())))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("SVG graph operation colors are immutable")

    def __getitem__(self, key: str) -> str:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self._items) == dict(other.items())

    def __deepcopy__(self, memo: dict[int, object]) -> _ImmutableColorMap:
        del memo
        return self


@dataclass(frozen=True, init=False, slots=True)
class SvgGraphTheme:
    """Define colors and operation color classification for an SVG graph.

    An operation first receives a key from ``operation_color_key``. An exact
    ``operation_colors`` entry wins; otherwise the key is deterministically
    mapped into ``operation_palette``. The hash implementation is private, but
    equal keys under one theme always receive equal colors. ``None`` selects
    the FHElium default palette or color-key function. ``operation_colors=None``
    retains the default constant color, while an empty mapping removes it.
    Theme configuration is read-only and deliberately unhashable because a
    caller-supplied color-key callable does not define a general cache identity.
    """

    operation_palette: tuple[str, ...]
    operation_colors: Mapping[str, str]
    operation_color_key: Callable[[SvgOperationContext], str]
    canvas_color: str
    input_fill_color: str
    output_fill_color: str
    node_font_color: str
    node_stroke_color: str
    edge_color: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        *,
        operation_palette: Collection[str] | None = None,
        operation_colors: Mapping[str, str] | None = None,
        operation_color_key: Callable[[SvgOperationContext], str] | None = None,
        canvas_color: str = "transparent",
        input_fill_color: str = "#DCE4FF",
        output_fill_color: str = "#F3D89D",
        node_font_color: str = "#181B26",
        node_stroke_color: str = "#596178",
        edge_color: str = "#7B8193",
    ) -> None:
        palette: tuple[str, ...]
        if operation_palette is None:
            palette = _DEFAULT_OPERATION_PALETTE
        elif isinstance(operation_palette, (str, bytes)):
            raise TypeError(
                "SVG graph operation_palette must be a collection of colors"
            )
        else:
            palette = tuple(operation_palette)
        if not palette:
            raise ValueError("SVG graph operation_palette must be non-empty")
        for index, color in enumerate(palette):
            _validate_nonempty_string(
                color,
                field=f"operation_palette[{index}]",
            )
        if operation_colors is None:
            selected_colors = dict(_DEFAULT_OPERATION_COLORS)
        else:
            selected_colors = dict(operation_colors)
        for key, color in selected_colors.items():
            _validate_nonempty_string(key, field="operation_colors key")
            _validate_nonempty_string(
                color,
                field=f"operation_colors[{key!r}]",
            )
        selected_color_key: Callable[[SvgOperationContext], str]
        if operation_color_key is None:
            selected_color_key = default_svg_operation_color_key
        elif not callable(operation_color_key):
            raise TypeError("SVG graph operation_color_key must be callable")
        else:
            selected_color_key = operation_color_key
        object.__setattr__(self, "operation_palette", palette)
        object.__setattr__(
            self,
            "operation_colors",
            _ImmutableColorMap(selected_colors),
        )
        object.__setattr__(self, "operation_color_key", selected_color_key)
        for field, color in (
            ("canvas_color", canvas_color),
            ("input_fill_color", input_fill_color),
            ("output_fill_color", output_fill_color),
            ("node_font_color", node_font_color),
            ("node_stroke_color", node_stroke_color),
            ("edge_color", edge_color),
        ):
            object.__setattr__(
                self,
                field,
                _validate_nonempty_string(color, field=field),
            )

    def operation_fill_color(self, context: SvgOperationContext) -> str:
        """Return the configured explicit or stable palette color."""

        key = self.operation_color_key(context)
        if not isinstance(key, str):
            raise TypeError(
                "SVG graph operation_color_key must return a string"
            )
        if not key.strip():
            raise ValueError(
                "SVG graph operation_color_key must return a non-empty string"
            )
        explicit = self.operation_colors.get(key)
        if explicit is not None:
            return explicit
        color_hash = int(
            hashlib.md5(
                key.encode(),
                usedforsecurity=False,
            ).hexdigest()[:8],
            16,
        )
        return self.operation_palette[color_hash % len(self.operation_palette)]


class SvgGraphPresentation:
    """Select and format operation evidence for an SVG graph.

    The presentation is independent of graph traversal and file production.
    Subclasses can override ``select_attributes`` for filtering,
    ``format_attribute_value`` for one-value display policy,
    ``operation_sections`` for complete operation-row transformation, or
    ``operation_tooltip`` for aggregate hover evidence. These methods must not
    mutate their operation, Program, or Workspace.
    """

    def __init__(
        self,
        *,
        fields: Collection[SvgGraphField] | None = None,
        attribute_names: Collection[str] | None = None,
        attribute_preview_chars: int | None = 180,
        theme: SvgGraphTheme | None = None,
    ) -> None:
        if isinstance(fields, (str, bytes)):
            raise TypeError(
                "JIT SVG fields must be a collection of field names"
            )
        selected_fields = _ALL_FIELDS if fields is None else frozenset(fields)
        unknown_fields = selected_fields - _ALL_FIELDS
        if unknown_fields:
            raise ValueError(
                "Unknown JIT SVG visualization fields: "
                f"{sorted(unknown_fields)}"
            )
        if not selected_fields:
            raise ValueError(
                "JIT SVG visualization requires at least one displayed field"
            )
        selected_attributes: frozenset[str] | None = None
        if attribute_names is not None:
            if isinstance(attribute_names, (str, bytes)):
                raise TypeError(
                    "JIT SVG attribute_names must be a collection of names"
                )
            selected_attributes = frozenset(attribute_names)
            if not all(
                isinstance(attribute, str) and attribute.strip()
                for attribute in selected_attributes
            ):
                raise ValueError(
                    "JIT SVG attribute_names must contain non-empty strings"
                )
        if attribute_preview_chars is not None:
            if isinstance(attribute_preview_chars, bool) or not isinstance(
                attribute_preview_chars, int
            ):
                raise TypeError(
                    "JIT SVG attribute_preview_chars must be an integer or None"
                )
            if attribute_preview_chars <= 0:
                raise ValueError(
                    "JIT SVG attribute_preview_chars must be positive"
                )
        if theme is None:
            theme = SvgGraphTheme()
        elif not isinstance(theme, SvgGraphTheme):
            raise TypeError("JIT SVG theme must be SvgGraphTheme")
        self.fields = selected_fields
        self.attribute_names = selected_attributes
        self.attribute_preview_chars = attribute_preview_chars
        self.theme = theme

    def select_attributes(
        self,
        operation: Operation,
    ) -> tuple[tuple[str, Attribute], ...]:
        """Return sorted non-internal attributes selected for ``operation``."""

        hidden = {OBLIGATIONS_ATTRIBUTE, "op_name__"}
        return tuple(
            (name, attribute)
            for name, attribute in sorted(operation.attributes.items())
            if name not in hidden
            and (self.attribute_names is None or name in self.attribute_names)
        )

    def format_attribute_value(
        self,
        operation: Operation,
        name: str,
        attribute: Attribute,
    ) -> str:
        """Format one selected attribute for its bounded record preview."""

        del operation, name
        text = str(attribute)
        preview_chars = self.attribute_preview_chars
        if preview_chars is None or len(text) <= preview_chars:
            return text
        omitted = len(text) - preview_chars
        return f"{text[:preview_chars]}… [{omitted} chars omitted]"

    def operation_sections(
        self,
        context: SvgOperationContext,
    ) -> tuple[SvgNodeSection, ...]:
        """Return all displayed record rows for one operation.

        Override this complete-row hook to reorder, rename, remove, or inject
        derived sections while reusing ``super().operation_sections(context)``.
        """

        operation = context.operation
        sections: list[SvgNodeSection] = []
        for field in _FIELD_ORDER:
            if field not in self.fields:
                continue
            if field == "name":
                sections.append(
                    SvgNodeSection(
                        "name",
                        ", ".join(context.result_names)
                        if context.result_names
                        else display_name(operation),
                    )
                )
            elif field == "opcode":
                sections.append(
                    SvgNodeSection("opcode", operation_name(operation))
                )
            elif field == "role":
                sections.append(
                    SvgNodeSection("roles", _roles(operation.results))
                )
            elif field == "operands":
                sections.append(
                    SvgNodeSection(
                        "operands",
                        "(" + ", ".join(context.operand_names) + ")",
                    )
                )
            elif field == "result_types":
                sections.append(
                    SvgNodeSection("result_types", _types(operation.results))
                )
            elif field == "attributes":
                for name, attribute in self.select_attributes(operation):
                    sections.append(
                        SvgNodeSection(
                            f"attr:{name}",
                            self.format_attribute_value(
                                operation,
                                name,
                                attribute,
                            ),
                            tooltip=f"{name}={attribute}",
                        )
                    )
            elif field == "scheduling_obligations":
                sections.append(
                    SvgNodeSection(
                        "scheduling_obligations",
                        "(" + ", ".join(sorted(obligations(operation))) + ")",
                    )
                )
            elif field == "num_users":
                sections.append(
                    SvgNodeSection(
                        "num_users",
                        str(_num_users(operation.results)),
                    )
                )
        return tuple(sections)

    def operation_tooltip(
        self,
        context: SvgOperationContext,
        sections: Collection[SvgNodeSection],
    ) -> str | None:
        """Return one aggregate node tooltip from already-rendered sections."""

        details = tuple(
            section.tooltip
            for section in sections
            if section.tooltip is not None
        )
        return "\n".join((operation_name(context.operation), *details))


def _record_text(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("|", "\\|")
        .replace("<", "\\<")
        .replace(">", "\\>")
        .replace("\n", " ")
    )


def _section_text(section: SvgNodeSection) -> str:
    return f"{_record_text(section.name)}={_record_text(section.value)}"


def _record_label(sections: Collection[SvgNodeSection]) -> str:
    rendered = tuple(_section_text(section) for section in sections)
    return "{" + "|".join(rendered) + "}" if rendered else " "


def _roles(values: Collection[SSAValue]) -> str:
    return "(" + ", ".join(str(value_role(value)) for value in values) + ")"


def _types(values: Collection[SSAValue]) -> str:
    return "(" + ", ".join(str(value.type) for value in values) + ")"


def _num_users(values: Collection[SSAValue]) -> int:
    count = 0
    for value in values:
        for _ in value.uses:
            count += 1
    return count


def _io_sections(
    *,
    name: str,
    opcode: str,
    values: Collection[SSAValue],
    fields: frozenset[SvgGraphField],
    operand_names: Collection[str],
    num_users: int,
) -> tuple[SvgNodeSection, ...]:
    sections: list[SvgNodeSection] = []
    for field in _FIELD_ORDER:
        if field not in fields:
            continue
        if field == "name":
            sections.append(SvgNodeSection("name", name))
        elif field == "opcode":
            sections.append(SvgNodeSection("opcode", opcode))
        elif field == "role":
            sections.append(SvgNodeSection("roles", _roles(values)))
        elif field == "operands":
            sections.append(
                SvgNodeSection(
                    "operands",
                    "(" + ", ".join(operand_names) + ")",
                )
            )
        elif field == "result_types":
            sections.append(SvgNodeSection("result_types", _types(values)))
        elif field == "attributes":
            continue
        elif field == "scheduling_obligations":
            sections.append(SvgNodeSection("scheduling_obligations", "()"))
        elif field == "num_users":
            sections.append(SvgNodeSection("num_users", str(num_users)))
    return tuple(sections)


def _validated_operation_sections(
    presentation: SvgGraphPresentation,
    context: SvgOperationContext,
) -> tuple[SvgNodeSection, ...]:
    sections = presentation.operation_sections(context)
    if not isinstance(sections, tuple):
        raise TypeError(
            "JIT SVG operation_sections must return a tuple of "
            "SvgNodeSection values"
        )
    if not all(isinstance(section, SvgNodeSection) for section in sections):
        raise TypeError(
            "JIT SVG operation_sections must return only SvgNodeSection values"
        )
    return sections


@dataclass(frozen=True, init=False)
class SvgGraphVisualizationPass:
    """Write one selected entry's SSA/dataflow graph to an SVG file.

    The pass owns entry traversal, stable SSA naming, dependency edges,
    Graphviz construction, overwrite policy, and SVG production. A composed
    ``SvgGraphPresentation`` owns operation rows, tooltips, color classification,
    and theme values. Rendering returns the Program unchanged and does not
    establish execution readiness or numerical correctness.

    Args:
        output_path: Exact ``.svg`` file to write.
        overwrite: Replace an existing output file when true.
        entry: Unique single-block function entry to render.
        presentation: Operation evidence and theme policy. ``None`` constructs
            the default presentation.
        rank_direction: Graphviz rank direction: top-to-bottom, bottom-to-top,
            left-to-right, or right-to-left.
        name: Diagnostic pass name.
    """

    output_path: Path
    presentation: SvgGraphPresentation
    overwrite: bool = False
    entry: str = "main"
    rank_direction: SvgGraphDirection = "TB"
    name: str = "visualize-svg"

    def __init__(
        self,
        output_path: str | PathLike[str],
        *,
        overwrite: bool = False,
        entry: str = "main",
        presentation: SvgGraphPresentation | None = None,
        rank_direction: SvgGraphDirection = "TB",
        name: str = "visualize-svg",
    ) -> None:
        path = Path(output_path)
        if path.suffix.lower() != ".svg":
            raise ValueError("JIT SVG output_path must end with '.svg'")
        if presentation is None:
            presentation = SvgGraphPresentation()
        elif not isinstance(presentation, SvgGraphPresentation):
            raise TypeError("JIT SVG presentation must be SvgGraphPresentation")
        if rank_direction not in {"TB", "BT", "LR", "RL"}:
            raise ValueError(
                "JIT SVG rank_direction must be 'TB', 'BT', 'LR', or 'RL'"
            )
        object.__setattr__(self, "output_path", path)
        object.__setattr__(self, "overwrite", overwrite)
        object.__setattr__(self, "entry", entry)
        object.__setattr__(self, "presentation", presentation)
        object.__setattr__(self, "rank_direction", rank_direction)
        object.__setattr__(self, "name", name)

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Render one entry function and return an unchanged pass result."""

        del workspace
        path = self.output_path
        if path.exists() and not self.overwrite:
            raise JitPassError(
                f"JIT SVG visualization would overwrite existing file {path}"
            )
        try:
            import pydot
        except ImportError as exc:
            raise JitPassError(
                "JIT SVG visualization requires pydot>=4.0.1"
            ) from exc

        presentation = self.presentation
        theme = presentation.theme
        block = program.entry_block(self.entry)
        operations = tuple(
            operation
            for operation in block.ops
            if operation_name(operation) != "func.return"
        )
        dot = pydot.Dot(
            "FHEliumJitProgram",
            graph_type="digraph",
            rankdir=self.rank_direction,
            bgcolor=theme.canvas_color,
        )
        value_nodes: dict[SSAValue, str] = {}
        value_names: dict[SSAValue, str] = {}
        used_names: dict[str, int] = {}

        def assign_value_name(value: SSAValue, fallback: str) -> str:
            base = value.name_hint or fallback
            occurrence = used_names.get(base, 0)
            used_names[base] = occurrence + 1
            unique = base if occurrence == 0 else f"{base}_{occurrence}"
            name = f"%{unique}"
            value_names[value] = name
            return name

        for index, argument in enumerate(block.args):
            node_id = f"arg_{index}"
            value_nodes[argument] = node_id
            value_name = assign_value_name(argument, f"arg{index}")
            sections = _io_sections(
                name=value_name,
                opcode="input",
                values=(argument,),
                fields=presentation.fields,
                operand_names=(),
                num_users=_num_users((argument,)),
            )
            dot.add_node(
                pydot.Node(
                    node_id,
                    label=_record_label(sections),
                    shape="record",
                    style="filled,rounded",
                    fillcolor=theme.input_fill_color,
                    fontcolor=theme.node_font_color,
                    color=theme.node_stroke_color,
                )
            )

        for index, operation in enumerate(operations):
            node_id = f"op_{index}"
            for result_index, result in enumerate(operation.results):
                assign_value_name(result, f"value{index}_{result_index}")
            context = SvgOperationContext(
                operation,
                tuple(value_names[result] for result in operation.results),
                tuple(value_names[operand] for operand in operation.operands),
            )
            sections = _validated_operation_sections(presentation, context)
            tooltip = presentation.operation_tooltip(context, sections)
            if tooltip is not None and not isinstance(tooltip, str):
                raise TypeError(
                    "JIT SVG operation_tooltip must return a string or None"
                )
            node_attributes: dict[str, Any] = {
                "label": _record_label(sections),
                "shape": "record",
                "style": "filled,rounded",
                "fillcolor": theme.operation_fill_color(context),
                "fontcolor": theme.node_font_color,
                "color": theme.node_stroke_color,
            }
            if tooltip is not None:
                node_attributes["tooltip"] = tooltip
            dot.add_node(pydot.Node(node_id, **node_attributes))
            for result in operation.results:
                value_nodes[result] = node_id
            for operand in operation.operands:
                source = value_nodes.get(operand)
                if source is not None:
                    dot.add_edge(
                        pydot.Edge(
                            source,
                            node_id,
                            color=theme.edge_color,
                        )
                    )

        for operation in block.ops:
            if operation_name(operation) != "func.return":
                continue
            operand_names = tuple(
                value_names[operand] for operand in operation.operands
            )
            sections = _io_sections(
                name="output",
                opcode="output",
                values=tuple(operation.operands),
                fields=presentation.fields,
                operand_names=operand_names,
                num_users=0,
            )
            dot.add_node(
                pydot.Node(
                    "output",
                    label=_record_label(sections),
                    shape="record",
                    style="filled,rounded",
                    fillcolor=theme.output_fill_color,
                    fontcolor=theme.node_font_color,
                    color=theme.node_stroke_color,
                )
            )
            for operand in operation.operands:
                source = value_nodes.get(operand)
                if source is not None:
                    dot.add_edge(
                        pydot.Edge(
                            source,
                            "output",
                            color=theme.edge_color,
                        )
                    )

        try:
            svg = cast(Any, dot).create_svg()
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(svg, str):
                path.write_text(svg, encoding="utf-8")
            else:
                path.write_bytes(svg)
        except Exception as exc:
            raise JitPassError(
                f"JIT SVG visualization failed for {path}: {exc}"
            ) from exc
        if not path.is_file() or path.stat().st_size == 0:
            raise JitPassError(
                f"JIT SVG visualization did not produce a file at {path}"
            )
        return PassResult.unchanged(
            program,
            matched=len(operations),
            diagnostics=(f"wrote JIT Program SVG to {path.resolve()}",),
        )


__all__ = [
    "SvgGraphDirection",
    "SvgGraphField",
    "SvgGraphPresentation",
    "SvgGraphTheme",
    "SvgGraphVisualizationPass",
    "SvgNodeSection",
    "SvgOperationContext",
    "default_svg_operation_color_key",
]
