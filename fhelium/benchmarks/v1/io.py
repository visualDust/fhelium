"""Strict JSON persistence for FHElium Benchmark v1 reports."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from fhelium.benchmarks.io import write_json_atomic

from .manifest import validate_report_specification
from .model import BenchmarkReport


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def write_report_atomic(
    path: str | os.PathLike[str], report: BenchmarkReport
) -> None:
    """Validate and atomically persist a :class:`BenchmarkReport`."""

    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a BenchmarkReport")
    validate_report_specification(report)
    write_json_atomic(path, report.to_dict())


def read_report(path: str | os.PathLike[str]) -> BenchmarkReport:
    """Read one strict Benchmark v1 report from UTF-8 JSON."""

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(
            stream,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r} is forbidden")
            ),
        )
    if not isinstance(payload, Mapping):
        raise TypeError("Benchmark v1 report JSON must contain an object")
    report = BenchmarkReport.from_dict(payload)
    validate_report_specification(report)
    return report


__all__ = ["read_report", "write_report_atomic"]
