#!/usr/bin/env python3
"""Generate VitePress API fragments and package navigation without imports."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from urllib.parse import quote

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "fhelium"
API_ROOT = REPOSITORY_ROOT / "docs" / "api"
GENERATED_ROOT_PAGE = API_ROOT / "fhelium.md"
GENERATED_PACKAGE_ROOT = API_ROOT / "fhelium"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs" / ".vitepress" / "api-reference.json"
DEFAULT_SIDEBAR_OUTPUT = (
    REPOSITORY_ROOT / "docs" / ".vitepress" / "api-sidebar.json"
)
SOURCE_ROOT = "https://github.com/VisualDust/fhelium/blob"
EXCLUDED_API_MODULE_PREFIXES = ("fhelium.native.wrapper",)

DIRECTIVE_RE = re.compile(r"^:::\s+(fhelium(?:\.[A-Za-z_]\w*)*)\s*$")
ROLE_RE = re.compile(r":(?:py:)?(?:class|func|meth|attr|mod):`~?([^`]+)`")
SECTION_RE = re.compile(
    r"^(Args|Arguments|Parameters|Attributes|Returns|Yields|Raises|"
    r"Examples|Example|Notes|Note|Warnings|Warning):\s*$"
)
FIELD_RE = re.compile(
    r"^\s{4}(\*{0,2}[A-Za-z_][\w.]*)(?:\s+\(([^)]+)\))?:\s*(.*)$"
)


@dataclass(frozen=True)
class ImportTarget:
    module: str
    symbol: str | None


@dataclass
class ModuleDefinition:
    name: str
    path: Path
    tree: ast.Module
    is_package: bool
    objects: dict[str, ast.AST]
    imports: dict[str, ImportTarget]


@dataclass(frozen=True)
class ObjectDefinition:
    module: ModuleDefinition
    node: ast.AST


@dataclass(frozen=True)
class DataclassOptions:
    enabled: bool = False
    init: bool = True
    kw_only: bool = False


@dataclass(frozen=True)
class DataclassParameter:
    name: str
    annotation: str
    default: str | None
    init: bool
    kw_only: bool


ResolvedDefinition = Union[ModuleDefinition, ObjectDefinition]


def module_name_for(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def resolve_import_module(
    current_module: str,
    *,
    is_package: bool,
    imported_module: str | None,
    level: int,
) -> str:
    if level == 0:
        return imported_module or ""

    package_parts = current_module.split(".")
    if not is_package:
        package_parts.pop()
    keep = len(package_parts) - (level - 1)
    if keep < 0:
        raise ValueError(f"Invalid relative import in {current_module}")
    parts = package_parts[:keep]
    if imported_module:
        parts.extend(imported_module.split("."))
    return ".".join(parts)


def build_module_index() -> dict[str, ModuleDefinition]:
    modules: dict[str, ModuleDefinition] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module_name, is_package = module_name_for(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        objects: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                objects[node.name] = node
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                objects[node.targets[0].id] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target,
                ast.Name,
            ):
                objects[node.target.id] = node
        modules[module_name] = ModuleDefinition(
            name=module_name,
            path=path,
            tree=tree,
            is_package=is_package,
            objects=objects,
            imports={},
        )

    for module in modules.values():
        for node in module.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound_name = alias.asname or alias.name.split(".")[0]
                    target_name = alias.name if alias.asname else bound_name
                    module.imports[bound_name] = ImportTarget(target_name, None)
                continue

            if not isinstance(node, ast.ImportFrom):
                continue
            target_module = resolve_import_module(
                module.name,
                is_package=module.is_package,
                imported_module=node.module,
                level=node.level,
            )
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound_name = alias.asname or alias.name
                candidate_module = f"{target_module}.{alias.name}"
                if candidate_module in modules:
                    module.imports[bound_name] = ImportTarget(
                        candidate_module,
                        None,
                    )
                else:
                    module.imports[bound_name] = ImportTarget(
                        target_module,
                        alias.name,
                    )
    return modules


def resolve_reference(
    reference: str,
    modules: dict[str, ModuleDefinition],
) -> ResolvedDefinition:
    parts = reference.split(".")
    for split_index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_index])
        module = modules.get(module_name)
        if module is None:
            continue
        remainder = parts[split_index:]
        if not remainder:
            return module
        return resolve_from_module(
            module,
            remainder,
            modules,
            visited=set(),
        )
    raise ValueError(f"Cannot resolve API reference {reference!r}")


def resolve_from_module(
    module: ModuleDefinition,
    remainder: list[str],
    modules: dict[str, ModuleDefinition],
    *,
    visited: set[tuple[str, tuple[str, ...]]],
) -> ResolvedDefinition:
    key = (module.name, tuple(remainder))
    if key in visited:
        raise ValueError(f"Cyclic API re-export while resolving {module.name}")
    visited.add(key)

    symbol = remainder[0]
    node = module.objects.get(symbol)
    if node is not None:
        if len(remainder) != 1:
            raise ValueError(
                f"Nested API member resolution is unsupported: "
                f"{module.name}.{'.'.join(remainder)}"
            )
        return ObjectDefinition(module=module, node=node)

    imported = module.imports.get(symbol)
    if imported is None:
        raise ValueError(f"Cannot resolve {module.name}.{symbol}")

    if imported.symbol is None:
        target_module = modules.get(imported.module)
        if target_module is None:
            raise ValueError(f"Unknown imported module {imported.module}")
        if len(remainder) == 1:
            return target_module
        return resolve_from_module(
            target_module,
            remainder[1:],
            modules,
            visited=visited,
        )

    target_module = modules.get(imported.module)
    if target_module is None:
        raise ValueError(f"Unknown imported module {imported.module}")
    return resolve_from_module(
        target_module,
        [imported.symbol, *remainder[1:]],
        modules,
        visited=visited,
    )


def parse_api_directives() -> dict[str, tuple[str, ...]]:
    directives: dict[str, tuple[str, ...]] = {}
    for path in sorted(API_ROOT.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            match = DIRECTIVE_RE.match(lines[index])
            if match is None:
                index += 1
                continue

            reference = match.group(1)
            members: list[str] = []
            cursor = index + 1
            in_members = False
            while cursor < len(lines):
                line = lines[cursor]
                if not line.strip() or not line[:1].isspace():
                    break
                if line.strip() == "members:":
                    in_members = True
                elif in_members:
                    member_match = re.match(r"^\s*-\s+([A-Za-z_]\w*)\s*$", line)
                    if member_match:
                        members.append(member_match.group(1))
                cursor += 1

            value = tuple(members)
            previous = directives.get(reference)
            if previous is not None and previous != value:
                if previous and value:
                    raise ValueError(
                        f"Conflicting API directives for {reference}: "
                        f"{previous!r} and {value!r}"
                    )
                value = ()
            directives[reference] = value
            index = cursor
    return directives


def normalize_inline_markup(text: str) -> str:
    text = ROLE_RE.sub(lambda match: f"`{match.group(1)}`", text)
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def dedent_section(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    return textwrap.dedent("\n".join(lines)).splitlines()


def parse_fields(lines: list[str]) -> list[tuple[str, str | None, str]]:
    fields: list[tuple[str, str | None, str]] = []
    for line in lines:
        match = FIELD_RE.match(line)
        if match:
            fields.append((match.group(1), match.group(2), match.group(3)))
            continue
        if not fields:
            continue
        continuation = line.strip()
        if continuation:
            name, type_name, description = fields[-1]
            separator = " " if description else ""
            fields[-1] = (
                name,
                type_name,
                f"{description}{separator}{continuation}",
            )
    return fields


def render_field_section(title: str, lines: list[str]) -> str:
    fields = parse_fields(lines)
    if not fields:
        body = "\n".join(dedent_section(lines))
        return f"**{title}**\n\n{body}" if body else ""

    rendered = [f"**{title}**", ""]
    for name, type_name, description in fields:
        label = f"`{name}`"
        if type_name:
            label += f" (`{type_name}`)"
        rendered.append(f"- {label}: {description}".rstrip())
    return "\n".join(rendered)


def render_examples(lines: list[str]) -> str:
    content = dedent_section(lines)
    if not content:
        return ""
    if any(line.lstrip().startswith(">>>") for line in content):
        return "**Examples**\n\n```python\n" + "\n".join(content) + "\n```"
    return "**Examples**\n\n" + "\n".join(content)


def render_docstring(value: str | None) -> str:
    if not value:
        return ""
    cleaned = normalize_inline_markup(inspect.cleandoc(value))
    lines = cleaned.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = SECTION_RE.match(line)
        if match:
            sections.append((current_name, current_lines))
            current_name = match.group(1)
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_name, current_lines))

    rendered: list[str] = []
    for name, section_lines in sections:
        if name is None:
            body = "\n".join(dedent_section(section_lines))
            if body:
                rendered.append(body)
            continue

        if name in {"Args", "Arguments", "Parameters"}:
            section = render_field_section("Parameters", section_lines)
        elif name == "Attributes":
            section = render_field_section("Attributes", section_lines)
        elif name == "Raises":
            section = render_field_section("Raises", section_lines)
        elif name in {"Examples", "Example"}:
            section = render_examples(section_lines)
        else:
            title = {
                "Returns": "Returns",
                "Yields": "Yields",
                "Notes": "Note",
                "Note": "Note",
                "Warnings": "Warning",
                "Warning": "Warning",
            }[name]
            body = "\n".join(dedent_section(section_lines))
            section = f"**{title}**\n\n{body}" if body else ""
        if section:
            rendered.append(section)
    return "\n\n".join(rendered)


def source_link(definition: ObjectDefinition) -> str:
    relative = definition.module.path.relative_to(REPOSITORY_ROOT).as_posix()
    node = definition.node
    start_line = getattr(node, "lineno", None)
    if not isinstance(start_line, int):
        raise ValueError(f"API definition has no source line: {ast.dump(node)}")
    end_line = getattr(node, "end_lineno", None)
    if not isinstance(end_line, int):
        end_line = start_line
    source_ref = quote(os.environ.get("DOCS_SOURCE_REF") or "main", safe="")
    return f"{SOURCE_ROOT}/{source_ref}/{relative}#L{start_line}-L{end_line}"


def decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return ast.unparse(decorator)


def has_decorator(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> bool:
    return any(
        decorator_name(item).split(".")[-1] == name
        for item in node.decorator_list
    )


def format_arguments(arguments: ast.arguments, *, drop_first: bool) -> str:
    rendered = ast.unparse(arguments)
    if not drop_first:
        return rendered
    rendered = re.sub(r"^(?:self|cls)(?:,\s*)?", "", rendered)
    rendered = re.sub(r"^/\s*,?\s*", "", rendered)
    return rendered


def format_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    name: str | None = None,
    drop_first: bool = False,
) -> str:
    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    arguments = format_arguments(node.args, drop_first=drop_first)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{async_prefix}def {name or node.name}({arguments}){returns}: ..."


def dataclass_options(node: ast.ClassDef) -> DataclassOptions:
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        target = call.func if call else decorator
        if ast.unparse(target).split(".")[-1] != "dataclass":
            continue
        kw_only = False
        init = True
        if call:
            for keyword in call.keywords:
                if keyword.arg == "kw_only" and isinstance(
                    keyword.value,
                    ast.Constant,
                ):
                    kw_only = bool(keyword.value.value)
                if keyword.arg == "init" and isinstance(
                    keyword.value,
                    ast.Constant,
                ):
                    init = bool(keyword.value.value)
        return DataclassOptions(enabled=True, init=init, kw_only=kw_only)
    return DataclassOptions()


def expression_base_name(expression: ast.expr) -> str:
    while isinstance(expression, ast.Subscript):
        expression = expression.value
    return ast.unparse(expression)


def resolve_class_bases(
    definition: ObjectDefinition,
    modules: dict[str, ModuleDefinition],
) -> list[ObjectDefinition]:
    node = definition.node
    assert isinstance(node, ast.ClassDef)
    resolved: list[ObjectDefinition] = []
    for base in node.bases:
        try:
            candidate = resolve_from_module(
                definition.module,
                expression_base_name(base).split("."),
                modules,
                visited=set(),
            )
        except ValueError:
            continue
        if isinstance(candidate, ObjectDefinition) and isinstance(
            candidate.node,
            ast.ClassDef,
        ):
            resolved.append(candidate)
    return resolved


def annotation_name(annotation: ast.expr) -> str:
    expression = (
        annotation.value
        if isinstance(annotation, ast.Subscript)
        else annotation
    )
    return ast.unparse(expression).split(".")[-1]


def dataclass_field_options(
    field: ast.AnnAssign,
    *,
    default_kw_only: bool,
) -> tuple[bool, bool, str | None]:
    init = True
    kw_only = default_kw_only
    value = field.value
    if (
        not isinstance(value, ast.Call)
        or expression_base_name(value.func).split(".")[-1] != "field"
    ):
        return init, kw_only, ast.unparse(value) if value is not None else None

    default: str | None = None
    for keyword in value.keywords:
        if keyword.arg == "init" and isinstance(keyword.value, ast.Constant):
            init = bool(keyword.value.value)
        elif keyword.arg == "kw_only" and isinstance(
            keyword.value,
            ast.Constant,
        ):
            kw_only = bool(keyword.value.value)
        elif keyword.arg == "default":
            default = ast.unparse(keyword.value)
        elif keyword.arg == "default_factory":
            default = f"field(default_factory={ast.unparse(keyword.value)})"
    return init, kw_only, default


def declared_dataclass_parameters(
    node: ast.ClassDef,
    *,
    default_kw_only: bool,
) -> list[DataclassParameter]:
    parameters: list[DataclassParameter] = []
    fields_are_kw_only = default_kw_only
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(
            item.target,
            ast.Name,
        ):
            continue
        if annotation_name(item.annotation) == "KW_ONLY":
            fields_are_kw_only = True
            continue
        if (
            item.target.id.startswith("_")
            or annotation_name(item.annotation) == "ClassVar"
        ):
            continue
        init, kw_only, default = dataclass_field_options(
            item,
            default_kw_only=fields_are_kw_only,
        )
        parameters.append(
            DataclassParameter(
                name=item.target.id,
                annotation=ast.unparse(item.annotation),
                default=default,
                init=init,
                kw_only=kw_only,
            )
        )
    return parameters


def merge_dataclass_parameters(
    destination: list[DataclassParameter],
    additions: list[DataclassParameter],
) -> None:
    indices = {
        parameter.name: index for index, parameter in enumerate(destination)
    }
    for parameter in additions:
        index = indices.get(parameter.name)
        if index is None:
            indices[parameter.name] = len(destination)
            destination.append(parameter)
        else:
            destination[index] = parameter


def dataclass_fields_for(
    definition: ObjectDefinition,
    modules: dict[str, ModuleDefinition],
    *,
    visited: set[tuple[str, str]],
) -> list[DataclassParameter] | None:
    node = definition.node
    assert isinstance(node, ast.ClassDef)
    key = (definition.module.name, node.name)
    if key in visited:
        raise ValueError(
            f"Cyclic class inheritance while resolving {node.name}"
        )
    visited.add(key)

    inherited: list[DataclassParameter] = []
    has_dataclass_base = False
    # Dataclasses collect base fields in reverse MRO order.
    for base in reversed(resolve_class_bases(definition, modules)):
        base_fields = dataclass_fields_for(
            base,
            modules,
            visited=set(visited),
        )
        if base_fields is None:
            continue
        has_dataclass_base = True
        merge_dataclass_parameters(inherited, base_fields)

    options = dataclass_options(node)
    if not options.enabled:
        return inherited if has_dataclass_base else None

    merge_dataclass_parameters(
        inherited,
        declared_dataclass_parameters(
            node,
            default_kw_only=options.kw_only,
        ),
    )
    return inherited


def dataclass_constructor_parameters(
    definition: ObjectDefinition,
    modules: dict[str, ModuleDefinition],
) -> list[DataclassParameter] | None:
    node = definition.node
    assert isinstance(node, ast.ClassDef)
    options = dataclass_options(node)
    bases = resolve_class_bases(definition, modules)

    if options.enabled and options.init:
        fields = dataclass_fields_for(definition, modules, visited=set())
        assert fields is not None
        return [field for field in fields if field.init]

    # A dataclass subclass without its own generated initializer inherits the
    # first dataclass base initializer. This covers marker subclasses such as
    # RelinearizationKey without exposing constructors from arbitrary bases.
    for base in bases:
        parameters = dataclass_constructor_parameters(base, modules)
        if parameters is not None:
            return parameters
    return [] if options.enabled else None


def format_class_signature(
    definition: ObjectDefinition,
    modules: dict[str, ModuleDefinition],
) -> str:
    node = definition.node
    assert isinstance(node, ast.ClassDef)
    constructor = next(
        (
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "__init__"
        ),
        None,
    )
    if constructor is not None:
        arguments = format_arguments(constructor.args, drop_first=True)
        return f"{node.name}({arguments})"

    dataclass_parameters = dataclass_constructor_parameters(definition, modules)
    if dataclass_parameters is not None:
        positional: list[str] = []
        keyword_only: list[str] = []
        for parameter in dataclass_parameters:
            rendered = f"{parameter.name}: {parameter.annotation}"
            if parameter.default is not None:
                rendered += f" = {parameter.default}"
            (keyword_only if parameter.kw_only else positional).append(rendered)
        parameters = positional
        if keyword_only:
            parameters = [*parameters, "*", *keyword_only]
        return f"{node.name}({', '.join(parameters)})"
    return f"{node.name}()"


def class_attributes(node: ast.ClassDef) -> list[tuple[str, str, str]]:
    attributes: list[tuple[str, str, str]] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(
            item.target, ast.Name
        ):
            if item.target.id.startswith("_"):
                continue
            value = ast.unparse(item.value) if item.value is not None else ""
            attributes.append(
                (item.target.id, ast.unparse(item.annotation), value)
            )
        elif isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
            if not isinstance(target, ast.Name) or target.id.startswith("_"):
                continue
            attributes.append((target.id, "", ast.unparse(item.value)))
    return attributes


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_attributes(node: ast.ClassDef) -> str:
    attributes = class_attributes(node)
    if not attributes:
        return ""
    lines = [
        "### Attributes",
        "",
        "| Name | Type | Default/value |",
        "| --- | --- | --- |",
    ]
    for name, annotation, value in attributes:
        lines.append(
            f"| `{name}` | {f'`{escape_table_cell(annotation)}`' if annotation else ''} "
            f"| {f'`{escape_table_cell(value)}`' if value else ''} |"
        )
    return "\n".join(lines)


def render_method(
    definition: ObjectDefinition,
    nodes: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> str:
    implementation = next(
        (
            node
            for node in reversed(nodes)
            if not has_decorator(node, "overload")
        ),
        nodes[-1],
    )
    signature_nodes = [
        node for node in nodes if has_decorator(node, "overload")
    ]
    if not signature_nodes:
        signature_nodes = [implementation]

    node = implementation
    is_static = has_decorator(node, "staticmethod")
    is_property = has_decorator(node, "property") or has_decorator(
        node,
        "cached_property",
    )
    if is_property:
        annotation = ast.unparse(node.returns) if node.returns else "Any"
        signature = f"{node.name}: {annotation}"
    else:
        signature = "\n".join(
            format_function_signature(
                signature_node,
                drop_first=not is_static,
            )
            for signature_node in signature_nodes
        )
    kind = "property" if is_property else "method"
    parts = [
        f"### `{node.name}`",
        "",
        f'<span class="api-kind">{kind}</span>',
        "",
        "```python",
        signature,
        "```",
    ]
    docstring = render_docstring(ast.get_docstring(node, clean=False))
    if docstring:
        parts.extend(["", docstring])
    return "\n".join(parts)


def render_class(
    definition: ObjectDefinition,
    display_path: str,
    modules: dict[str, ModuleDefinition],
) -> str:
    node = definition.node
    assert isinstance(node, ast.ClassDef)
    bases = [ast.unparse(base) for base in node.bases]
    parts = [
        f"## `{node.name}`",
        "",
        '<span class="api-kind">class</span> '
        f"[View source]({source_link(definition)})",
        "",
        "```python",
        format_class_signature(definition, modules),
        "```",
    ]
    if bases:
        parts.extend(
            ["", "**Bases:** " + ", ".join(f"`{base}`" for base in bases)]
        )

    docstring = render_docstring(ast.get_docstring(node, clean=False))
    if docstring:
        parts.extend(["", docstring])

    attributes = render_attributes(node)
    if attributes:
        parts.extend(["", attributes])

    methods: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name.startswith("_"):
            continue
        methods.setdefault(item.name, []).append(item)
    for method_nodes in methods.values():
        parts.extend(["", render_method(definition, method_nodes)])
    return "\n".join(parts)


def render_function(definition: ObjectDefinition, display_path: str) -> str:
    node = definition.node
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    parts = [
        f"## `{node.name}`",
        "",
        '<span class="api-kind">function</span> '
        f"[View source]({source_link(definition)})",
        "",
        "```python",
        format_function_signature(node),
        "```",
    ]
    docstring = render_docstring(ast.get_docstring(node, clean=False))
    if docstring:
        parts.extend(["", docstring])
    return "\n".join(parts)


def render_data(definition: ObjectDefinition, display_path: str) -> str:
    node = definition.node
    if isinstance(node, ast.Assign):
        target = node.targets[0]
        assert isinstance(target, ast.Name)
        name = target.id
        declaration = f"{name} = {ast.unparse(node.value)}"
    else:
        assert isinstance(node, ast.AnnAssign)
        assert isinstance(node.target, ast.Name)
        name = node.target.id
        declaration = f"{name}: {ast.unparse(node.annotation)}"
        if node.value is not None:
            declaration += f" = {ast.unparse(node.value)}"
    if name.isupper():
        kind = "constant"
    else:
        value = node.value
        value_name = (
            expression_base_name(value).split(".")[-1]
            if value is not None
            else ""
        )
        kind = "type alias" if value_name in {"Literal", "Union"} else "data"

    return "\n".join(
        [
            f"## `{name}`",
            "",
            f'<span class="api-kind">{kind}</span> '
            f"[View source]({source_link(definition)})",
            "",
            "```python",
            declaration,
            "```",
        ]
    )


def render_object(
    definition: ObjectDefinition,
    display_path: str,
    modules: dict[str, ModuleDefinition],
) -> str:
    if isinstance(definition.node, ast.ClassDef):
        return render_class(definition, display_path, modules)
    if isinstance(definition.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return render_function(definition, display_path)
    return render_data(definition, display_path)


def explicit_module_exports(module: ModuleDefinition) -> tuple[str, ...] | None:
    node = module.objects.get("__all__")
    value: ast.expr | None = None
    if isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign):
        value = node.value
    if value is None:
        return None
    exports = ast.literal_eval(value)
    if not isinstance(exports, (list, tuple)) or not all(
        isinstance(item, str) for item in exports
    ):
        raise ValueError(f"{module.name}.__all__ must be a string sequence")
    return tuple(exports)


def public_module_members(module: ModuleDefinition) -> tuple[str, ...]:
    exports = explicit_module_exports(module)
    if exports is not None:
        return exports

    names: list[str] = []
    for node in module.tree.body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith(
                    "_"
                ):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(
                node.target, ast.Name
            ) and not node.target.id.startswith("_"):
                names.append(node.target.id)
    return tuple(dict.fromkeys(names))


def render_module_member(
    module: ModuleDefinition,
    member: str,
    reference: str,
    modules: dict[str, ModuleDefinition],
) -> str:
    try:
        definition = resolve_from_module(
            module,
            [member],
            modules,
            visited=set(),
        )
    except ValueError:
        imported = module.imports.get(member)
        if imported is None:
            raise
        target = imported.module
        if imported.symbol is not None:
            target = f"{target}.{imported.symbol}"
        return "\n".join(
            [
                f"## `{member}`",
                "",
                '<span class="api-kind">re-export</span>',
                "",
                "```python",
                f"{member} = {target}",
                "```",
            ]
        )

    if isinstance(definition, ObjectDefinition):
        return render_object(definition, f"{reference}.{member}", modules)
    return "\n".join(
        [
            f"## `{member}`",
            "",
            '<span class="api-kind">module</span>',
            "",
            f"`{definition.name}`",
        ]
    )


def render_reference(
    reference: str,
    members: tuple[str, ...],
    modules: dict[str, ModuleDefinition],
) -> str:
    resolved = resolve_reference(reference, modules)
    if isinstance(resolved, ObjectDefinition):
        if members:
            raise ValueError(
                f"Object directive {reference} cannot select members"
            )
        return render_object(resolved, reference, modules)

    selected = members or public_module_members(resolved)
    rendered: list[str] = []
    module_docstring = render_docstring(
        ast.get_docstring(resolved.tree, clean=False)
    )
    if module_docstring:
        rendered.append(module_docstring)
    for member in selected:
        rendered.append(
            render_module_member(resolved, member, reference, modules)
        )
    return "\n\n".join(rendered)


def public_api_modules(
    modules: dict[str, ModuleDefinition],
) -> tuple[str, ...]:
    discovered: list[str] = []
    for name, module in modules.items():
        if name.split(".")[0] != PACKAGE_ROOT.name:
            continue
        if any(part.startswith("_") for part in name.split(".")[1:]):
            continue
        if module.is_package and "__all__" not in module.objects:
            continue
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in EXCLUDED_API_MODULE_PREFIXES
        ):
            continue
        discovered.append(name)
    return tuple(sorted(discovered))


def generated_module_page(module_name: str) -> Path:
    components = module_name.split(".")
    if len(components) == 1:
        return GENERATED_ROOT_PAGE
    return GENERATED_PACKAGE_ROOT.joinpath(*components[1:]).with_suffix(".md")


def generated_module_link(module_name: str) -> str:
    components = module_name.split(".")
    if len(components) == 1:
        return f"/api/{components[0]}"
    return "/api/" + "/".join(components)


def write_generated_module_pages(module_names: tuple[str, ...]) -> None:
    GENERATED_ROOT_PAGE.unlink(missing_ok=True)
    if GENERATED_PACKAGE_ROOT.exists():
        shutil.rmtree(GENERATED_PACKAGE_ROOT)

    for module_name in module_names:
        path = generated_module_page(module_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "---",
                    "editLink: false",
                    "---",
                    "",
                    f"# `{module_name}`",
                    "",
                    f"::: {module_name}",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def render_api_sidebar(
    module_names: tuple[str, ...],
) -> list[dict[str, object]]:
    root_name = PACKAGE_ROOT.name
    group_names = sorted(
        {
            root_name
            if len(module_name.split(".")) <= 2
            else ".".join(module_name.split(".")[:2])
            for module_name in module_names
        },
        key=lambda group: (group != root_name, group.casefold()),
    )
    sidebar: list[dict[str, object]] = []
    for group in group_names:
        group_modules = [
            module_name
            for module_name in module_names
            if (
                root_name
                if len(module_name.split(".")) <= 2
                else ".".join(module_name.split(".")[:2])
            )
            == group
        ]
        sidebar.append(
            {
                "text": group,
                "items": [
                    {
                        "text": module_name,
                        "link": generated_module_link(module_name),
                    }
                    for module_name in group_modules
                ],
            }
        )
    return sidebar


def generate(output: Path, sidebar_output: Path) -> None:
    modules = build_module_index()
    module_names = public_api_modules(modules)
    write_generated_module_pages(module_names)
    directives = parse_api_directives()
    rendered = {
        reference: render_reference(reference, members, modules)
        for reference, members in sorted(directives.items())
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rendered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidebar = render_api_sidebar(module_names)
    sidebar_output.parent.mkdir(parents=True, exist_ok=True)
    sidebar_output.write_text(
        json.dumps(sidebar, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(rendered)} API references at {output}")
    print(f"Generated {len(module_names)} API module pages")
    print(f"Generated {len(sidebar)} API package groups at {sidebar_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON file consumed by the VitePress Markdown plugin",
    )
    parser.add_argument(
        "--sidebar-output",
        type=Path,
        default=DEFAULT_SIDEBAR_OUTPUT,
        help="JSON file consumed by the VitePress sidebar configuration",
    )
    args = parser.parse_args()
    generate(args.output.resolve(), args.sidebar_output.resolve())


if __name__ == "__main__":
    main()
