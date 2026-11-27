"""Generate the C++ API reference pages from libfhelium's public headers.

Header text supplies documentation: `//!` lines before the namespace describe
the header, and `///` lines document the function, enumerator, struct, class or
member declaration that follows them. Torch
operator schemas come from the registration sources in `csrc/torchops`, and
device implementations come from the routers in `csrc/ops/<family>/api.cpp`.
Like the Python reference, generation scans source text and needs neither a
compiler nor an importable `fhelium` package.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOT = REPOSITORY_ROOT / "csrc" / "include"
TORCHOPS_ROOT = REPOSITORY_ROOT / "csrc" / "torchops"
OPS_ROOT = REPOSITORY_ROOT / "csrc" / "ops"
OUTPUT_ROOT = REPOSITORY_ROOT / "docs" / "api" / "cpp"
LINK_ROOT = "/api/cpp"
SOURCE_ROOT = "https://github.com/VisualDust/fhelium/blob"

# Header, generated page name. The order is the sidebar order.
HEADERS = (
    ("fhelium/ckks.h", "ckks"),
    ("fhelium/rns.h", "rns"),
    ("fhelium/ntt.h", "ntt"),
    ("fhelium/ntt_diagnostic.h", "ntt-diagnostic"),
    ("fhelium/random.h", "random"),
    ("fhelium/rns_parameters.h", "rns-parameters"),
    ("fhelium/rns_tables.h", "rns-tables"),
    ("fhelium/ntt_tables.h", "ntt-tables"),
    ("fhelium/ckks_tables.h", "ckks-tables"),
    ("fhelium/ckks_parameters.h", "ckks-parameters"),
    ("fhelium/operation_tables.h", "operation-tables"),
    ("fhelium/ntt_operations.h", "ntt-operations"),
    ("fhelium/rns_operations.h", "rns-operations"),
    ("fhelium/ckks_operations.h", "ckks-operations"),
    ("fhelium/ckks_keys.h", "ckks-keys"),
)

DISPATCH_DEVICES = {
    "CPU_GPU": "CPU, CUDA, HIP",
    "GPU": "CUDA, HIP",
    "CUDA": "CUDA",
}

ANY_DEVICE = "Any device"

NAMESPACE_RE = re.compile(r"^namespace\s+([\w:]+)\s*\{")
TYPE_RE = re.compile(r"^(struct|class)\s+(?:FHELIUM_API\s+)?(\w+)")
STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Declaration:
    name: str
    kind: str
    text: str
    doc: tuple[str, ...]
    start_line: int
    end_line: int
    parent: str = ""


@dataclass(frozen=True)
class Header:
    include: str
    page: str
    namespace: str
    intro: tuple[str, ...]
    declarations: tuple[Declaration, ...]


def paragraphs(lines: list[str]) -> tuple[str, ...]:
    """Join comment lines into Markdown paragraphs and list items."""

    blocks: list[str] = []
    current: list[str] = []
    for line in [*lines, ""]:
        if not line.strip():
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        if line.startswith("- ") and current:
            blocks.append(" ".join(current))
            current = []
        current.append(line.strip())
    rendered: list[str] = []
    for block in blocks:
        if (
            block.startswith("- ")
            and rendered
            and rendered[-1].startswith("- ")
        ):
            rendered[-1] += "\n" + block
        else:
            rendered.append(block)
    return tuple(rendered)


def parse_header(include: str, page: str) -> Header:
    path = INCLUDE_ROOT / include
    lines = path.read_text(encoding="utf-8").splitlines()
    namespace = ""
    intro: list[str] = []
    pending: list[str] = []
    declarations: list[Declaration] = []
    parent = ""
    # Documentation of the preceding function, shared by the overloads that
    # directly follow it without their own comment.
    shared: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("FHELIUM_API") and not pending:
            pending = shared
        shared = []
        if stripped.startswith("//!"):
            intro.append(stripped[3:].removeprefix(" "))
        elif stripped.startswith("///"):
            pending.append(stripped[3:].removeprefix(" "))
        elif match := NAMESPACE_RE.match(stripped):
            namespace = match.group(1)
        elif stripped == "};":
            parent = ""
        elif pending and (match := TYPE_RE.match(stripped)):
            declarations.append(
                Declaration(
                    name=match.group(2),
                    kind=match.group(1),
                    text=f"{match.group(1)} {match.group(2)}",
                    doc=paragraphs(pending),
                    start_line=index + 1,
                    end_line=index + 1,
                )
            )
            parent = match.group(2)
            pending = []
        elif pending:
            start = index
            if parent:
                # A member function may carry its whole body on one line.
                terminators: tuple[str, ...] = (";", "{", "}")
            elif stripped.startswith("FHELIUM_API"):
                terminators = (";",)
            else:
                terminators = (",",)
            text = [stripped]
            while not text[-1].endswith(terminators):
                index += 1
                text.append(lines[index].strip())
            declaration = " ".join(text)
            if parent:
                declaration = re.sub(r"\s*\{[^{}]*\}$", "", declaration)
                declaration = declaration.rstrip("{;").strip()
                name = re.search(r"(\w+)\(", declaration) or re.search(
                    r"(\w+)(?:\s*=.*)?$", declaration
                )
                kind = "member"
            elif terminators == (";",):
                declaration = declaration.removeprefix("FHELIUM_API").strip()
                name = re.search(r"(\w+)\(", declaration)
                kind = "function"
            else:
                name = re.match(r"(\w+)", declaration)
                kind = "enumerator"
            assert name is not None, f"{include}:{start + 1}"
            if kind == "function":
                shared = pending
            declarations.append(
                Declaration(
                    name=name.group(1),
                    kind=kind,
                    text=declaration.rstrip(",;"),
                    doc=paragraphs(pending),
                    start_line=start + 1,
                    end_line=index + 1,
                    parent=parent,
                )
            )
            pending = []
        index += 1
    return Header(
        include=include,
        page=page,
        namespace=namespace,
        intro=paragraphs(intro),
        declarations=tuple(declarations),
    )


def balanced_calls(text: str, function: str) -> list[str]:
    """Return the argument text of every `function(...)` call in `text`."""

    calls: list[str] = []
    for match in re.finditer(rf"\b{re.escape(function)}\(", text):
        depth = 1
        index = match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(text[index], 0)
            index += 1
        calls.append(text[match.end() : index - 1])
    return calls


def torch_operators() -> dict[str, tuple[str, str]]:
    """Map qualified C++ functions to their Torch library and schema."""

    schemas: dict[tuple[str, str], str] = {}
    functions: dict[str, tuple[str, str]] = {}
    block_re = re.compile(
        r"TORCH_LIBRARY(?:_FRAGMENT|_IMPL)?\((\w+)\s*,[^{]*\{(.*?)\n\}", re.S
    )
    for path in sorted(TORCHOPS_ROOT.glob("*.cpp")):
        for library, body in block_re.findall(path.read_text(encoding="utf-8")):
            for arguments in balanced_calls(body, "m.def"):
                schema = "".join(STRING_RE.findall(arguments))
                schemas[(library, schema.split("(", 1)[0])] = schema
            for arguments in balanced_calls(body, "m.impl"):
                name, target = (
                    part.strip() for part in arguments.split(",", 1)
                )
                target = re.sub(r"\s+", "", target).lstrip("&")
                name = name.strip('"')
                # A file-local adapter registers the public function of the
                # same name in the library's namespace.
                if "::" not in target:
                    family = library.removeprefix("fhelium_").removesuffix(
                        "_ops"
                    )
                    target = f"fhelium::{family}::{name}"
                functions[target] = (library, name)
    return {
        qualified: (library, schemas[(library, name)])
        for qualified, (library, name) in functions.items()
    }


def device_implementations() -> dict[str, str]:
    """Map qualified C++ functions to the devices their router dispatches."""

    devices: dict[str, str] = {}
    for path in sorted(OPS_ROOT.glob("*/api.cpp")):
        namespace = ""
        text = path.read_text(encoding="utf-8")
        for line_match in re.finditer(
            r"^namespace\s+([\w:]+)\s*\{|FHELIUM_DISPATCH_(CPU_GPU|GPU|CUDA)\(\s*(\w+)",
            text,
            re.M,
        ):
            if line_match.group(1):
                namespace = line_match.group(1)
            else:
                devices[f"{namespace}::{line_match.group(3)}"] = (
                    DISPATCH_DEVICES[line_match.group(2)]
                )
    return devices


def split_parameters(text: str) -> list[str]:
    parameters: list[str] = []
    depth = 0
    current = ""
    for character in text:
        if character in "<(":
            depth += 1
        elif character in ">)":
            depth -= 1
        if character == "," and depth == 0:
            parameters.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        parameters.append(current.strip())
    return parameters


def format_signature(declaration: Declaration, namespace: str) -> str:
    head, parameters = declaration.text.split("(", 1)
    parameters = parameters.rsplit(")", 1)[0]
    result, name = head.rsplit(None, 1)
    qualified = f"{result} {namespace}::{name}("
    parts = split_parameters(parameters)
    if not parts:
        return qualified + ");"
    return qualified + "\n" + ",\n".join(f"    {part}" for part in parts) + ");"


def source_link(include: str, declaration: Declaration) -> str:
    source_ref = quote(os.environ.get("DOCS_SOURCE_REF") or "main", safe="")
    relative = (INCLUDE_ROOT / include).relative_to(REPOSITORY_ROOT).as_posix()
    return (
        f"{SOURCE_ROOT}/{source_ref}/{relative}"
        f"#L{declaration.start_line}-L{declaration.end_line}"
    )


def first_sentence(doc: tuple[str, ...]) -> str:
    return doc[0] if doc else ""


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def render_function(
    header: Header,
    declaration: Declaration,
    anchor: str,
    operators: dict[str, tuple[str, str]],
    devices: dict[str, str],
) -> str:
    qualified = f"{header.namespace}::{declaration.name}"
    parts = [
        f"## `{declaration.name}` {{#{anchor}}}",
        "",
        '<span class="api-kind">function</span> '
        f"[View source]({source_link(header.include, declaration)})",
        "",
        "```cpp",
        format_signature(declaration, header.namespace),
        "```",
        "",
        f"**Devices:** {devices.get(qualified, ANY_DEVICE)}  ",
    ]
    if qualified in operators:
        library, schema = operators[qualified]
        parts.extend(
            [
                f"**Torch operator:** `torch.ops.{library}.{declaration.name}`",
                "",
                "```text",
                f"{library}::{schema}",
                "```",
            ]
        )
    parts.extend(["", *[f"{paragraph}\n" for paragraph in declaration.doc]])
    return "\n".join(parts).rstrip() + "\n"


def render_type(header: Header, declaration: Declaration) -> str:
    members = [
        d
        for d in header.declarations
        if d.kind == "member" and d.parent == declaration.name
    ]
    parts = [
        f"## `{declaration.name}` {{#{declaration.name}}}",
        "",
        f'<span class="api-kind">{declaration.kind}</span> '
        f"[View source]({source_link(header.include, declaration)})",
        "",
        *[f"{paragraph}\n" for paragraph in declaration.doc],
        "| Member | Description |",
        "| --- | --- |",
        *[
            f"| [`{escape_cell(d.text)}`]({source_link(header.include, d)}) "
            f"| {escape_cell(' '.join(d.doc))} |"
            for d in members
        ],
    ]
    return "\n".join(parts).rstrip() + "\n"


def render_enumerators(header: Header) -> str:
    rows = [
        "| Enumerator | Value | Row content |",
        "| --- | --- | --- |",
    ]
    for declaration in header.declarations:
        if declaration.kind != "enumerator":
            continue
        value = declaration.text.split("=", 1)[1].strip()
        rows.append(
            f"| [`{declaration.name}`]({source_link(header.include, declaration)}) "
            f"| {value} | {escape_cell(' '.join(declaration.doc))} |"
        )
    return "\n".join(rows) + "\n"


def render_page(
    header: Header,
    operators: dict[str, tuple[str, str]],
    devices: dict[str, str],
) -> str:
    functions = [d for d in header.declarations if d.kind == "function"]
    types = [d for d in header.declarations if d.kind in {"struct", "class"}]
    enumerators = [d for d in header.declarations if d.kind == "enumerator"]
    parts = [
        "---",
        "editLink: false",
        "---",
        "",
        f"# `{header.include}`",
        "",
        f"`#include <{header.include}>` · namespace `{header.namespace}`",
        "",
        *[f"{paragraph}\n" for paragraph in header.intro],
    ]
    parts.extend(render_type(header, d) for d in types)
    if functions:
        # Overloads share a name, so later ones get numbered anchors.
        anchors: list[str] = []
        seen: dict[str, int] = {}
        for d in functions:
            seen[d.name] = seen.get(d.name, 0) + 1
            anchors.append(
                d.name if seen[d.name] == 1 else f"{d.name}-{seen[d.name]}"
            )
        parts.extend(
            [
                "| Function | Devices | Summary |",
                "| --- | --- | --- |",
                *[
                    f"| [`{d.name}`](#{anchor}) "
                    f"| {devices.get(f'{header.namespace}::{d.name}', ANY_DEVICE)} "
                    f"| {escape_cell(first_sentence(d.doc))} |"
                    for d, anchor in zip(functions, anchors)
                ],
                "",
            ]
        )
        parts.extend(
            render_function(header, d, anchor, operators, devices)
            for d, anchor in zip(functions, anchors)
        )
    if enumerators:
        parts.append(render_enumerators(header))
    return "\n".join(parts).rstrip() + "\n"


def generate() -> list[dict[str, object]]:
    """Write the C++ reference pages and return their sidebar group."""

    operators = torch_operators()
    devices = device_implementations()
    for path in OUTPUT_ROOT.glob("*.md"):
        if path.name != "index.md":
            path.unlink()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    headers = [parse_header(include, page) for include, page in HEADERS]
    documented = {
        f"{header.namespace}::{d.name}"
        for header in headers
        for d in header.declarations
        if d.kind == "function"
    }
    # Every registered operator is a public function and needs a documented
    # declaration; an undocumented one would be missing from the reference.
    if missing := set(operators) - documented:
        raise ValueError(
            "Registered libfhelium functions lack header documentation: "
            + ", ".join(sorted(missing))
        )
    for header in headers:
        (OUTPUT_ROOT / f"{header.page}.md").write_text(
            render_page(header, operators, devices), encoding="utf-8"
        )
    print(
        f"Generated {len(headers)} C++ API pages covering "
        f"{len(documented)} functions"
    )
    return [
        {
            "text": "C++ (libfhelium)",
            "link": f"{LINK_ROOT}/",
            "items": [
                {"text": header.include, "link": f"{LINK_ROOT}/{header.page}"}
                for header in headers
            ],
        }
    ]


if __name__ == "__main__":
    generate()
