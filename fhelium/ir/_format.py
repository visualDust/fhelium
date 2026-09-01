"""Human-readable presentation for mixed-level Programs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from io import StringIO
from typing import Literal

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IntegerAttr,
    NoneAttr,
    StringAttr,
    f64,
    i64,
)
from xdsl.ir import Attribute
from xdsl.printer import Printer

from ._program import Program

ProgramFormatDetail = Literal["summary", "full"]

_SUMMARY_OMITS = frozenset(
    {
        # Entry types already carry the input specifications.
        "fhelium.input_specs",
        # func.return shows the ordinary single-result shape used by examples.
        "fhelium.output_structure",
        # The material type and symbol contain the useful presentation fields.
        "fhelium.material.descriptor",
    }
)


def _json_attribute(value: object) -> Attribute:
    """Convert decoded JSON into presentation-only structured attributes."""

    if value is None:
        return NoneAttr()
    if isinstance(value, bool):
        return IntegerAttr.from_bool(value)
    if isinstance(value, int):
        return IntegerAttr(value, i64)
    if isinstance(value, float):
        return FloatAttr(value, f64)
    if isinstance(value, str):
        return StringAttr(value)
    if isinstance(value, list):
        return ArrayAttr(_json_attribute(item) for item in value)
    if isinstance(value, dict):
        return DictionaryAttr(
            {str(name): _json_attribute(item) for name, item in value.items()}
        )
    raise TypeError(f"Unsupported decoded JSON value {type(value).__name__}")


def _decoded_json(attribute: StringAttr) -> Attribute | None:
    text = attribute.data
    if not text or text[0] not in "[{":
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, (dict, list)):
        return None
    return _json_attribute(value)


class _ReadablePrinter(Printer):
    """Render machine JSON strings as structured presentation attributes."""

    def __init__(
        self,
        *,
        detail: ProgramFormatDetail,
        stream: StringIO,
        print_generic_format: bool,
        print_debuginfo: bool,
    ) -> None:
        super().__init__(
            stream=stream,
            print_generic_format=print_generic_format,
            print_debuginfo=print_debuginfo,
        )
        self._detail = detail

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, StringAttr):
            decoded = _decoded_json(attribute)
            if decoded is not None:
                super().print_attribute(decoded)
                return
        super().print_attribute(attribute)

    def print_op_attributes(
        self,
        attributes: Mapping[str, Attribute],
        *,
        reserved_attr_names: Iterable[str] = (),
        print_keyword: bool = False,
    ) -> bool:
        if self._detail == "summary":
            attributes = {
                name: attribute
                for name, attribute in attributes.items()
                if name not in _SUMMARY_OMITS
            }
        return super().print_op_attributes(
            attributes,
            reserved_attr_names=reserved_attr_names,
            print_keyword=print_keyword,
        )


def format_program(
    program: Program,
    *,
    detail: ProgramFormatDetail = "summary",
    generic: bool = False,
    include_locations: bool = False,
) -> str:
    """Format a Program for reading rather than parsing or persistence.

    JSON payloads stored in machine-oriented string attributes are rendered as
    nested attribute dictionaries and arrays. ``summary`` also omits duplicated
    module/material metadata; ``full`` includes every field. Use
    :meth:`Program.to_text` when output must round-trip through the parser.
    """

    if not isinstance(program, Program):
        raise TypeError("format_program expects a Program")
    if detail not in {"summary", "full"}:
        raise ValueError("format_program detail must be 'summary' or 'full'")
    stream = StringIO()
    _ReadablePrinter(
        detail=detail,
        stream=stream,
        print_generic_format=generic,
        print_debuginfo=include_locations,
    ).print_op(program.module)
    return stream.getvalue()


__all__ = ["ProgramFormatDetail", "format_program"]
