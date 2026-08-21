#!/usr/bin/env python3
"""Publish one completed formal Benchmark v1 report to its static catalog.

The immutable raw report remains the authoritative evidence. This program
validates it without importing FHElium or initializing CUDA, then derives one
whole-run rendering projection. Individual benchmark cases are never published
as standalone portal results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = "v1"
SPECIFICATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "fhelium"
    / "benchmarks"
    / "v1"
    / "specification.json"
)
MAX_REPORT_BYTES = 128 * 1024 * 1024
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
LATENCY_UNITS = {"ns", "us", "ms", "s"}
MEMORY_UNITS = {"bytes", "KiB", "MiB", "GiB", "KB", "MB", "GB"}
ALLOWED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_CACHE_DISABLE",
    "CUDA_FORCE_PTX_JIT",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_LAUNCH_MODE",
    "NCCL_NET",
    "NCCL_CUMEM_ENABLE",
}
SENSITIVE_KEY_FRAGMENTS = (
    "authorization",
    "cookie",
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "privatekey",
    "apikey",
    "sshkey",
)
SENSITIVE_EXACT_KEYS = {
    "auth",
    "cwd",
    "fqdn",
    "host",
    "hostname",
    "session",
    "sessionid",
    "user",
    "userid",
    "username",
    "workspace",
    "workingdirectory",
}
ENVIRONMENT_KEYS = {
    "env",
    "environ",
    "environment",
    "environmentvariables",
    "processenvironment",
}
HOME_PATH = re.compile(
    r"(?:^|[\s=\"'\(])~/|"
    r"(?:^|[\s=\"'\(]|file://)(?:/home/[^/\s]+/|/Users/[^/\s]+/|"
    r"/root/|[A-Za-z]:\\Users\\[^\\\s]+\\)"
)
ENVIRONMENT_EXPANSION = re.compile(
    r"(?<!\\)\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|"
    r"[A-Za-z_][A-Za-z0-9_]*)|%[A-Za-z_][A-Za-z0-9_]*%"
)
CREDENTIAL_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bgh[opusr]_[A-Za-z0-9]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE,
)
ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s=\"'\(]|file://)(?:/(?!/)[A-Za-z0-9._-]+(?:/[^\s\"'\)]*)?|"
    r"[A-Za-z]:[\\/][^\s\"']+|"
    r"\\\\[^\\/\s]+[\\/][^\s\"']+)"
)

REPORT_FIELDS = {
    "benchmark_version",
    "manifest_sha256",
    "execution",
    "platform",
    "started_at",
    "finished_at",
    "status",
    "cases",
}
CASE_FIELDS = {
    "id",
    "title",
    "category",
    "benchmark",
    "workload_id",
    "profile",
    "parameters",
    "requirements",
    "comparison",
    "status",
    "started_at",
    "finished_at",
    "result",
    "unavailable",
    "failure",
}
RESULT_FIELDS = {
    "benchmark",
    "profile",
    "workload_id",
    "metrics",
    "correctness",
    "effective_parameters",
    "timed_boundary",
    "rows",
    "scalars",
    "metadata",
    "notes",
    "evidence",
}
METRIC_FIELDS = {
    "name",
    "value",
    "unit",
    "statistic",
    "direction",
    "dimensions",
    "samples",
}
CORRECTNESS_FIELDS = {
    "name",
    "passed",
    "oracle",
    "metric",
    "observed",
    "comparison",
    "limit",
    "unit",
    "details",
}
TIMED_BOUNDARY_FIELDS = {
    "id",
    "description",
    "includes",
    "excludes",
    "synchronization",
}
UNAVAILABLE_FIELDS = {"reason", "details"}
SPECIFICATION_FIELDS = {
    "benchmark_version",
    "title",
    "description",
    "cases",
    "manifest_sha256",
}
SPECIFICATION_CASE_FIELDS = {
    "id",
    "title",
    "category",
    "benchmark",
    "workload_id",
    "profile",
    "parameters",
    "requirements",
    "comparison",
    "unavailable_reason",
}
MANIFEST_FIELDS = {
    "benchmark_version",
    "title",
    "description",
    "cases",
}
CATALOG_FIELDS = {"benchmark_version", "generated_at", "runs"}
CATALOG_RUN_FIELDS = {
    "case_counts",
    "cases",
    "fhelium",
    "execution",
    "highlights",
    "id",
    "manifest_sha256",
    "platform",
    "published_at",
    "raw_path",
    "raw_sha256",
    "recorded_at",
    "slug",
    "status",
}
RETIRED_REPORT_FIELDS = {
    "comparison_key",
    "measurement_version",
    "report_schema_version",
    "schema_version",
    "suite_name",
    "suite_version",
    "workload_version",
}
PLATFORM_SYSTEM_FIELDS = (
    "system",
    "release",
    "version",
    "machine",
    "node",
    "platform",
)


class ValidationError(ValueError):
    """A report or catalog violates the publication validation rules."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON number {value!r} is not allowed")


def _read_json(path: Path) -> tuple[bytes, Any]:
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise ValidationError(f"cannot read {path}: {error}") from error
    if len(contents) > MAX_REPORT_BYTES:
        raise ValidationError(
            f"{path} exceeds the {MAX_REPORT_BYTES}-byte publication limit"
        )
    try:
        value = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise ValidationError(
            f"{path} is not strict UTF-8 JSON: {error}"
        ) from error
    return contents, value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must be an object")
    return value


def _exact_fields(
    value: dict[str, Any], expected: set[str], field: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ValidationError(f"{field} fields differ: {'; '.join(details)}")


def _reject_retired_report_fields(value: Any, field: str = "root") -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            if name in RETIRED_REPORT_FIELDS:
                raise ValidationError(
                    f"{field}.{name} is not part of Benchmark v1"
                )
            _reject_retired_report_fields(child, f"{field}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_retired_report_fields(child, f"{field}[{index}]")


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{field} must be an array")
    return value


def _text(
    value: Any,
    field: str,
    *,
    maximum: int = 2000,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or (not allow_empty and not value)
    ):
        qualifier = "a trimmed string" if allow_empty else "a non-empty string"
        raise ValidationError(f"{field} must be {qualifier}")
    if len(value) > maximum:
        raise ValidationError(f"{field} exceeds {maximum} characters")
    if any(ord(character) < 32 for character in value):
        raise ValidationError(f"{field} contains a control character")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValidationError(f"{field} has an invalid format")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a JSON number")
    if not math.isfinite(value):
        raise ValidationError(f"{field} must be finite")
    return value


def _scalar(value: Any, field: str) -> bool | int | float | str | None:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValidationError(f"{field} must be a finite JSON scalar")


def _utc_timestamp(value: Any, field: str) -> datetime:
    original = _text(value, field, maximum=40)
    candidate = original[:-1] + "+00:00" if original.endswith("Z") else original
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValidationError(
            f"{field} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValidationError(f"{field} must include a UTC designator")
    return parsed.astimezone(UTC)


def _normalized_timestamp(value: Any, field: str) -> str:
    parsed = _utc_timestamp(value, field)
    return parsed.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"value is not finite JSON: {error}") from error


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _validate_json_value(value: Any, field: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for name, item in value.items():
            if not isinstance(
                name, str
            ):  # JSON parsing already guarantees this.
                raise ValidationError(f"{field} keys must be strings")
            _validate_json_value(item, f"{field}.{name}")
        return
    raise ValidationError(
        f"{field} contains unsupported {type(value).__name__}"
    )


def _json_object(value: Any, field: str) -> dict[str, Any]:
    result = _mapping(value, field)
    _validate_json_value(result, field)
    return result


def _load_specification(
    path: Path = SPECIFICATION_PATH,
) -> dict[str, Any]:
    """Load and authenticate the dependency-free Benchmark v1 specification."""

    _, value = _read_json(path)
    specification = _mapping(value, "specification")
    _exact_fields(specification, SPECIFICATION_FIELDS, "specification")
    _reject_retired_report_fields(specification, "specification")
    if specification.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValidationError(
            "specification benchmark_version must be exactly "
            f"{BENCHMARK_VERSION!r}"
        )
    _text(specification.get("title"), "specification.title", maximum=300)
    _text(
        specification.get("description"),
        "specification.description",
        maximum=4000,
    )
    digest = _text(
        specification.get("manifest_sha256"),
        "specification.manifest_sha256",
        maximum=64,
        pattern=DIGEST,
    )
    covered = {name: specification[name] for name in MANIFEST_FIELDS}
    actual_digest = hashlib.sha256(_canonical_json_bytes(covered)).hexdigest()
    if digest != actual_digest:
        raise ValidationError(
            "specification.manifest_sha256 does not match its canonical "
            "covered payload"
        )

    cases = _array(specification.get("cases"), "specification.cases")
    if len(cases) != 5:
        raise ValidationError(
            "specification.cases must contain exactly five cases"
        )
    ids: set[str] = set()
    for index, value in enumerate(cases):
        field = f"specification.cases[{index}]"
        case = _mapping(value, field)
        _exact_fields(case, SPECIFICATION_CASE_FIELDS, field)
        case_id = _text(
            case.get("id"), f"{field}.id", maximum=100, pattern=IDENTIFIER
        )
        if case_id in ids:
            raise ValidationError(
                f"specification contains duplicate case {case_id!r}"
            )
        ids.add(case_id)
        for name in ("benchmark", "workload_id"):
            _text(
                case.get(name),
                f"{field}.{name}",
                maximum=100,
                pattern=IDENTIFIER,
            )
        _text(case.get("profile"), f"{field}.profile", maximum=100)
        _text(case.get("title"), f"{field}.title", maximum=300)
        _text(case.get("category"), f"{field}.category", maximum=100)
        for name in ("parameters", "requirements", "comparison"):
            _json_object(case.get(name), f"{field}.{name}")
        if case.get("unavailable_reason") is not None:
            raise ValidationError(
                f"{field}.unavailable_reason must be null because the report "
                "case format has no manifest-level unavailability field"
            )
    return specification


def _scan_string(value: str, field: str) -> None:
    if HOME_PATH.search(value):
        raise ValidationError(f"{field} contains a machine-local home path")
    if ENVIRONMENT_EXPANSION.search(value):
        raise ValidationError(f"{field} contains environment expansion")
    if CREDENTIAL_VALUE.search(value):
        raise ValidationError(f"{field} contains a credential-like value")
    if ABSOLUTE_PATH.search(value):
        raise ValidationError(f"{field} contains a machine-local absolute path")


def _scan_environment(value: Any, field: str) -> None:
    for name, item in _mapping(value, field).items():
        if name not in ALLOWED_ENVIRONMENT:
            raise ValidationError(
                f"{field}.{name} is not in the CUDA/NCCL publication allowlist"
            )
        text = _text(item, f"{field}.{name}", maximum=500, allow_empty=True)
        _scan_string(text, f"{field}.{name}")


def _scan_for_sensitive_data(value: Any, field: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if field == "root.platform" and key == "environment":
                _scan_environment(child, f"{field}.{key}")
                continue
            if normalized in ENVIRONMENT_KEYS:
                raise ValidationError(
                    f"{field}.{key} is an environment dump; only the platform "
                    "CUDA/NCCL allowlist is publishable"
                )
            if (
                any(
                    fragment in normalized
                    for fragment in SENSITIVE_KEY_FRAGMENTS
                )
                or normalized in SENSITIVE_EXACT_KEYS
            ):
                raise ValidationError(f"{field}.{key} is a sensitive field")
            if "score" in normalized and not (
                normalized == "globalscore" and child is False
            ):
                raise ValidationError(
                    f"{field}.{key} is a score field; composite scores are not "
                    "part of FHElium Benchmark"
                )
            _scan_for_sensitive_data(child, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_for_sensitive_data(child, f"{field}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{field} contains a non-finite number")
    if isinstance(value, str):
        _scan_string(value, field)


def _scalar_mapping(value: Any, field: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, item in _mapping(value, field).items():
        result[_text(name, f"{field} key", maximum=100)] = _scalar(
            item, f"{field}.{name}"
        )
    return result


def _string_array(value: Any, field: str, *, maximum: int = 100) -> list[str]:
    items = _array(value, field)
    if len(items) > maximum:
        raise ValidationError(f"{field} contains more than {maximum} items")
    return [
        _text(item, f"{field}[{index}]", maximum=2000)
        for index, item in enumerate(items)
    ]


def _performance_category(unit: str, dimensions: dict[str, Any]) -> str | None:
    declared = dimensions.get("category")
    if declared is not None:
        if declared not in {"latency", "throughput", "memory"}:
            raise ValidationError(
                "metric dimensions.category must be latency, throughput, or "
                "memory when present"
            )
        return str(declared)
    if unit in LATENCY_UNITS:
        return "latency"
    if unit in MEMORY_UNITS:
        return "memory"
    if unit.endswith("/s"):
        return "throughput"
    return None


def _project_metric(value: Any, field: str) -> dict[str, Any] | None:
    metric = _mapping(value, field)
    _exact_fields(metric, METRIC_FIELDS, field)
    name = _text(
        metric.get("name"), f"{field}.name", maximum=100, pattern=IDENTIFIER
    )
    if "score" in name.lower():
        raise ValidationError(f"{field}.name must not define a score")
    unit = _text(metric.get("unit"), f"{field}.unit", maximum=40)
    direction = _text(metric.get("direction"), f"{field}.direction", maximum=10)
    if direction not in {"lower", "higher", "none"}:
        raise ValidationError(f"{field}.direction is invalid")
    dimensions = _scalar_mapping(
        metric.get("dimensions"), f"{field}.dimensions"
    )
    for index, sample in enumerate(
        _array(metric.get("samples"), f"{field}.samples")
    ):
        _number(sample, f"{field}.samples[{index}]")
    metric_value = _number(metric.get("value"), f"{field}.value")
    statistic = _text(
        metric.get("statistic"), f"{field}.statistic", maximum=100
    )
    category = _performance_category(unit, dimensions)
    if category is None:
        return None
    dimensions.pop("category", None)
    return {
        "name": name,
        "category": category,
        "value": metric_value,
        "unit": unit,
        "statistic": statistic,
        "direction": direction,
        "dimensions": dimensions,
    }


def _project_correctness(value: Any, field: str) -> dict[str, Any]:
    check = _mapping(value, field)
    _exact_fields(check, CORRECTNESS_FIELDS, field)
    passed = check.get("passed")
    if type(passed) is not bool:
        raise ValidationError(f"{field}.passed must be a boolean")
    if not passed:
        raise ValidationError(f"{field} did not pass")
    unit_value = check.get("unit")
    unit = (
        ""
        if unit_value is None
        else _text(unit_value, f"{field}.unit", maximum=40)
    )
    _json_object(check.get("details"), f"{field}.details")
    return {
        "name": _text(check.get("name"), f"{field}.name", maximum=200),
        "passed": True,
        "oracle": _text(check.get("oracle"), f"{field}.oracle", maximum=2000),
        "metric": _text(check.get("metric"), f"{field}.metric", maximum=100),
        "observed": _scalar(check.get("observed"), f"{field}.observed"),
        "comparison": _text(
            check.get("comparison"), f"{field}.comparison", maximum=100
        ),
        "limit": _scalar(check.get("limit"), f"{field}.limit"),
        "unit": unit,
        # Detailed per-check states remain in the immutable raw report. The
        # catalog keeps the check identity and acceptance result needed by the
        # default-collapsed validation panel without duplicating large prime
        # and ciphertext-state objects hundreds of times.
        "details": {},
    }


def _project_timed_boundary(value: Any, field: str) -> dict[str, Any]:
    timed_boundary = _mapping(value, field)
    _exact_fields(timed_boundary, TIMED_BOUNDARY_FIELDS, field)
    return {
        "id": _text(
            timed_boundary.get("id"),
            f"{field}.id",
            maximum=100,
            pattern=IDENTIFIER,
        ),
        "description": _text(
            timed_boundary.get("description"),
            f"{field}.description",
            maximum=2000,
        ),
        "includes": _string_array(
            timed_boundary.get("includes"), f"{field}.includes"
        ),
        "excludes": _string_array(
            timed_boundary.get("excludes"), f"{field}.excludes"
        ),
        "synchronization": _text(
            timed_boundary.get("synchronization"),
            f"{field}.synchronization",
            maximum=2000,
        ),
    }


def _project_unavailable(value: Any, field: str) -> dict[str, Any]:
    unavailable = _mapping(value, field)
    _exact_fields(unavailable, UNAVAILABLE_FIELDS, field)
    return {
        "reason": _text(
            unavailable.get("reason"), f"{field}.reason", maximum=1000
        ),
        "details": copy.deepcopy(
            _json_object(unavailable.get("details"), f"{field}.details")
        ),
    }


def _project_case(
    value: Any,
    index: int,
    specification_case_value: Any,
    *,
    report_started_at: datetime,
    report_finished_at: datetime,
) -> dict[str, Any]:
    field = f"cases[{index}]"
    case = _mapping(value, field)
    _exact_fields(case, CASE_FIELDS, field)
    specification_case = _mapping(
        specification_case_value,
        f"specification.cases[{index}]",
    )
    for name in (
        "id",
        "title",
        "category",
        "benchmark",
        "workload_id",
        "profile",
        "parameters",
        "requirements",
        "comparison",
    ):
        if not _json_values_equal(case.get(name), specification_case.get(name)):
            raise ValidationError(
                f"{field}.{name} differs from the fixed Benchmark v1 "
                "specification"
            )

    status = _text(case.get("status"), f"{field}.status", maximum=20)
    if status not in {"measured", "unavailable"}:
        raise ValidationError(
            f"{field} is {status!r}; a completed publication may contain only "
            "measured or unavailable cases"
        )
    case_id = _text(
        case.get("id"), f"{field}.id", maximum=100, pattern=IDENTIFIER
    )
    benchmark = _text(
        case.get("benchmark"),
        f"{field}.benchmark",
        maximum=100,
        pattern=IDENTIFIER,
    )
    workload = _text(
        case.get("workload_id"),
        f"{field}.workload_id",
        maximum=100,
        pattern=IDENTIFIER,
    )
    profile = _text(case.get("profile"), f"{field}.profile", maximum=100)
    started_at = _utc_timestamp(case.get("started_at"), f"{field}.started_at")
    finished_at = _utc_timestamp(
        case.get("finished_at"), f"{field}.finished_at"
    )
    if started_at > finished_at:
        raise ValidationError(
            f"{field}.started_at must not be later than finished_at"
        )
    if started_at < report_started_at or finished_at > report_finished_at:
        raise ValidationError(
            f"{field} timestamps must fall within the report interval"
        )
    if case.get("failure") is not None:
        raise ValidationError(
            f"{field}.failure must be null in a completed report"
        )

    parameters = copy.deepcopy(
        _json_object(case.get("parameters"), f"{field}.parameters")
    )
    requirements = copy.deepcopy(
        _json_object(case.get("requirements"), f"{field}.requirements")
    )
    comparison = copy.deepcopy(
        _json_object(case.get("comparison"), f"{field}.comparison")
    )
    common = {
        "id": case_id,
        "title": _text(case.get("title"), f"{field}.title", maximum=300),
        "category": _text(
            case.get("category"), f"{field}.category", maximum=100
        ),
        "status": status,
        "benchmark": benchmark,
        "workload_id": workload,
        "profile": profile,
        "effective_parameters": parameters,
        "requirements": requirements,
        "comparison": comparison,
    }
    if status == "unavailable":
        if case.get("result") is not None:
            raise ValidationError(
                f"{field}.result must be null when unavailable"
            )
        return {
            **common,
            "unavailable": _project_unavailable(
                case.get("unavailable"), f"{field}.unavailable"
            ),
            "metrics": [],
            "correctness": [],
            "timed_boundary": None,
            "benchmark_context": None,
            "metadata": None,
        }

    if case.get("unavailable") is not None:
        raise ValidationError(f"{field}.unavailable must be null when measured")
    result = _mapping(case.get("result"), f"{field}.result")
    _exact_fields(result, RESULT_FIELDS, f"{field}.result")
    for name, expected in (
        ("benchmark", benchmark),
        ("profile", profile),
        ("workload_id", workload),
    ):
        if result.get(name) != expected:
            raise ValidationError(
                f"{field}.result.{name} does not match its case"
            )
    effective = _json_object(
        result.get("effective_parameters"),
        f"{field}.result.effective_parameters",
    )
    if not _json_values_equal(effective, parameters):
        raise ValidationError(
            f"{field}.result.effective_parameters does not match case.parameters"
        )
    metrics = [
        projected
        for metric_index, metric in enumerate(
            _array(result.get("metrics"), f"{field}.result.metrics")
        )
        if (
            projected := _project_metric(
                metric, f"{field}.result.metrics[{metric_index}]"
            )
        )
        is not None
    ]
    if not metrics:
        raise ValidationError(f"{field} has no renderable performance metrics")
    identities: set[tuple[str, str]] = set()
    for metric in metrics:
        identity = (
            metric["name"],
            json.dumps(
                metric["dimensions"], sort_keys=True, separators=(",", ":")
            ),
        )
        if identity in identities:
            raise ValidationError(
                f"{field} contains duplicate metric {identity!r}"
            )
        identities.add(identity)
    correctness_values = _array(
        result.get("correctness"), f"{field}.result.correctness"
    )
    if not correctness_values:
        raise ValidationError(f"{field}.result.correctness must not be empty")
    correctness = [
        _project_correctness(
            check, f"{field}.result.correctness[{check_index}]"
        )
        for check_index, check in enumerate(correctness_values)
    ]
    rows = _array(result.get("rows"), f"{field}.result.rows")
    for row_index, row in enumerate(rows):
        _json_object(row, f"{field}.result.rows[{row_index}]")
    _json_object(result.get("scalars"), f"{field}.result.scalars")
    notes = _array(result.get("notes"), f"{field}.result.notes")
    for note_index, note in enumerate(notes):
        if not isinstance(note, str):
            raise ValidationError(
                f"{field}.result.notes[{note_index}] must be a string"
            )
    evidence = _array(result.get("evidence"), f"{field}.result.evidence")
    for evidence_index, item in enumerate(evidence):
        _json_object(item, f"{field}.result.evidence[{evidence_index}]")
    metadata = _json_object(result.get("metadata"), f"{field}.result.metadata")
    declared_workload = metadata.get("workload_id")
    if declared_workload is not None and declared_workload != workload:
        raise ValidationError(
            f"{field}.result.metadata.workload_id does not match its case"
        )
    context = copy.deepcopy(
        _mapping(
            metadata.get("benchmark_context"),
            f"{field}.result.metadata.benchmark_context",
        )
    )
    selected_metadata = {
        name: copy.deepcopy(metadata[name])
        for name in (
            "case_highlight",
            "not_applicable_cases",
            "operation_order",
            "world_size",
            "operand_mode",
        )
        if name in metadata
    }
    return {
        **common,
        "unavailable": None,
        "metrics": metrics,
        "correctness": correctness,
        "timed_boundary": _project_timed_boundary(
            result.get("timed_boundary"), f"{field}.result.timed_boundary"
        ),
        "benchmark_context": context,
        "metadata": selected_metadata,
    }


def _project_platform(
    value: Any, *, allow_dirty: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    platform = _mapping(value, "platform")
    build = _mapping(platform.get("fhelium_build"), "platform.fhelium_build")
    source = _mapping(
        build.get("source_git"), "platform.fhelium_build.source_git"
    )
    commit = _text(
        source.get("commit"),
        "platform.fhelium_build.source_git.commit",
        maximum=64,
        pattern=COMMIT,
    )
    dirty = source.get("dirty")
    if type(dirty) is not bool:
        raise ValidationError(
            "platform.fhelium_build.source_git.dirty must be boolean"
        )
    if dirty and not allow_dirty:
        raise ValidationError(
            "platform.fhelium_build.source_git.dirty must be false for publication"
        )
    system = _json_object(platform.get("system"), "platform.system")
    unexpected_system_fields = sorted(set(system) - set(PLATFORM_SYSTEM_FIELDS))
    if unexpected_system_fields:
        raise ValidationError(
            "platform.system contains non-public fields: "
            + ", ".join(unexpected_system_fields)
        )
    node = system.get("node")
    if node not in (None, "", "<redacted>"):
        raise ValidationError("platform.system.node must be empty or redacted")
    _scan_environment(platform.get("environment"), "platform.environment")
    _array(platform.get("invocation"), "platform.invocation")
    cuda = _mapping(platform.get("cuda"), "platform.cuda")
    devices_value = cuda.get("devices")
    if isinstance(devices_value, dict):
        devices = [
            {
                "index": str(index),
                **copy.deepcopy(
                    _mapping(device, f"platform.cuda.devices.{index}")
                ),
            }
            for index, device in sorted(
                devices_value.items(),
                key=lambda item: (
                    int(item[0]) if str(item[0]).isdigit() else str(item[0])
                ),
            )
        ]
    elif isinstance(devices_value, list):
        devices = [
            {
                "index": str(index),
                **copy.deepcopy(
                    _mapping(device, f"platform.cuda.devices[{index}]")
                ),
            }
            for index, device in enumerate(devices_value)
        ]
    elif devices_value is None:
        devices = []
    else:
        raise ValidationError(
            "platform.cuda.devices must be an object or array"
        )
    p2p_value = cuda.get("p2p")
    p2p = {} if p2p_value is None else _mapping(p2p_value, "platform.cuda.p2p")
    projected_platform = {
        "system": {
            name: copy.deepcopy(system[name])
            for name in PLATFORM_SYSTEM_FIELDS
            if name in system
        },
        "cpu": copy.deepcopy(_mapping(platform.get("cpu"), "platform.cpu")),
        "memory": copy.deepcopy(
            _mapping(platform.get("memory"), "platform.memory")
        ),
        "python": copy.deepcopy(
            _mapping(platform.get("python"), "platform.python")
        ),
        "torch": copy.deepcopy(
            _mapping(platform.get("torch"), "platform.torch")
        ),
        "cuda": {
            "devices": devices,
            "p2p": copy.deepcopy(p2p),
        },
        "environment": copy.deepcopy(
            _mapping(platform.get("environment"), "platform.environment")
        ),
    }
    identity = {
        "version": _text(
            build.get("version"), "platform.fhelium_build.version", maximum=100
        ),
        "commit": commit,
        "dirty": dirty,
        "native": copy.deepcopy(
            _mapping(build.get("native"), "platform.fhelium_build.native")
        ),
    }
    return projected_platform, identity


def _highlight_for_case(case: dict[str, Any]) -> dict[str, Any] | None:
    if case["status"] != "measured":
        return None
    spec_value = case["comparison"].get("portal_highlight")
    if not isinstance(spec_value, dict):
        return None
    label = spec_value.get("label")
    if not isinstance(label, str) or not label:
        return None
    if spec_value.get("source") == "case_highlight":
        metadata = case.get("metadata") or {}
        case_highlight = metadata.get("case_highlight")
        field = spec_value.get("field")
        if not isinstance(case_highlight, dict) or not isinstance(field, str):
            return None
        value = case_highlight.get(field)
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return None
        return {
            "id": case["id"],
            "label": label,
            "value": value,
            "unit": "",
            "direction": "none",
            "case_id": case["id"],
        }
    metric_name = spec_value.get("metric")
    if not isinstance(metric_name, str):
        return None
    dimensions = {
        name: value
        for name, value in spec_value.items()
        if name not in {"label", "metric", "source", "field"}
    }
    metric = next(
        (
            item
            for item in case["metrics"]
            if item["name"] == metric_name
            and all(
                item["dimensions"].get(name) == value
                for name, value in dimensions.items()
            )
        ),
        None,
    )
    if metric is None:
        return None
    return {
        "id": case["id"],
        "label": label,
        "value": metric["value"],
        "unit": metric["unit"],
        "direction": metric["direction"],
        "case_id": case["id"],
    }


def validate_report(
    payload: Any, *, allow_dirty: bool = False
) -> dict[str, Any]:
    """Validate one formal v1 report and return its whole-run projection."""

    specification = _load_specification()
    report = _mapping(payload, "root")
    _exact_fields(report, REPORT_FIELDS, "root")
    _reject_retired_report_fields(report)
    _scan_for_sensitive_data(report)
    if report.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValidationError(
            f"benchmark_version must be exactly {BENCHMARK_VERSION!r}"
        )
    if report.get("status") != "completed":
        raise ValidationError(
            "only a completed formal Benchmark v1 report can be published"
        )
    manifest_sha256 = _text(
        report.get("manifest_sha256"),
        "manifest_sha256",
        maximum=64,
        pattern=DIGEST,
    )
    if manifest_sha256 != specification["manifest_sha256"]:
        raise ValidationError(
            "manifest_sha256 does not identify the fixed Benchmark v1 "
            "specification"
        )
    report_started_at = _utc_timestamp(report.get("started_at"), "started_at")
    report_finished_at = _utc_timestamp(
        report.get("finished_at"), "finished_at"
    )
    if report_started_at > report_finished_at:
        raise ValidationError("started_at must not be later than finished_at")
    recorded_at = _normalized_timestamp(
        report.get("finished_at"), "finished_at"
    )
    execution = _mapping(report.get("execution"), "execution")
    _exact_fields(execution, {"backend", "device"}, "execution")
    backend = execution.get("backend")
    device = execution.get("device")
    if backend not in {"cpu", "cuda"}:
        raise ValidationError("execution.backend must be cpu or cuda")
    if backend == "cpu" and device != "cpu":
        raise ValidationError("CPU execution requires execution.device='cpu'")
    if backend == "cuda" and (
        not isinstance(device, str)
        or re.fullmatch(r"cuda:[0-9]+", device) is None
    ):
        raise ValidationError(
            "CUDA execution requires an indexed cuda:N device"
        )
    platform, fhelium = _project_platform(
        report.get("platform"), allow_dirty=allow_dirty
    )
    if backend == "cuda" and not platform["cuda"]["devices"]:
        raise ValidationError(
            "CUDA execution requires published CUDA device provenance"
        )
    raw_cases = _array(report.get("cases"), "cases")
    specification_cases = _array(
        specification.get("cases"), "specification.cases"
    )
    if len(raw_cases) != len(specification_cases):
        raise ValidationError(
            "cases must contain the five fixed Benchmark v1 positions"
        )
    cases = [
        _project_case(
            case,
            index,
            specification_cases[index],
            report_started_at=report_started_at,
            report_finished_at=report_finished_at,
        )
        for index, case in enumerate(raw_cases)
    ]
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValidationError("case ids must be unique")
    counts = {
        name: 0 for name in ("measured", "unavailable", "failed", "interrupted")
    }
    for case in cases:
        counts[case["status"]] += 1
    if counts["measured"] == 0:
        raise ValidationError("a published report must contain a measured case")
    highlights = [
        highlight
        for case in cases
        if (highlight := _highlight_for_case(case)) is not None
    ]
    return {
        "manifest_sha256": manifest_sha256,
        "execution": copy.deepcopy(execution),
        "recorded_at": recorded_at,
        "status": "completed",
        "fhelium": fhelium,
        "platform": platform,
        "case_counts": counts,
        "highlights": highlights[:6],
        "cases": cases,
    }


def _empty_catalog() -> dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": None,
        "runs": [],
    }


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_catalog()
    _, value = _read_json(path)
    catalog = _mapping(value, "catalog")
    _exact_fields(catalog, CATALOG_FIELDS, "catalog")
    if catalog.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValidationError(
            f"catalog benchmark_version must be exactly {BENCHMARK_VERSION!r}"
        )
    generated_at = catalog.get("generated_at")
    if generated_at is not None:
        _normalized_timestamp(generated_at, "catalog.generated_at")
    runs = _array(catalog.get("runs"), "catalog.runs")
    ids: set[str] = set()
    rebuilt_runs: list[dict[str, Any]] = []
    for index, value in enumerate(runs):
        run = _mapping(value, f"catalog.runs[{index}]")
        _exact_fields(run, CATALOG_RUN_FIELDS, f"catalog.runs[{index}]")
        digest = _text(
            run.get("raw_sha256"),
            f"catalog.runs[{index}].raw_sha256",
            maximum=64,
            pattern=DIGEST,
        )
        if run.get("id") != digest or run.get("slug") != f"sha256-{digest}":
            raise ValidationError(f"catalog.runs[{index}] identity is invalid")
        if digest in ids:
            raise ValidationError(f"catalog contains duplicate run {digest}")
        ids.add(digest)
        raw_name = f"sha256-{digest}.json"
        if run.get("raw_path") != f"/benchmarks/v1/runs/{raw_name}":
            raise ValidationError(f"catalog.runs[{index}].raw_path is invalid")
        published_at = _normalized_timestamp(
            run.get("published_at"),
            f"catalog.runs[{index}].published_at",
        )
        raw_file = path.parent / "runs" / raw_name
        if not raw_file.is_file():
            raise ValidationError(
                f"catalog.runs[{index}] raw report {raw_file} does not exist"
            )
        raw_bytes, raw_report = _read_json(raw_file)
        actual_digest = hashlib.sha256(raw_bytes).hexdigest()
        if actual_digest != digest:
            raise ValidationError(
                f"catalog.runs[{index}] raw report digest does not match"
            )
        compact = validate_report(raw_report, allow_dirty=True)
        expected = {
            **compact,
            "id": digest,
            "slug": f"sha256-{digest}",
            "published_at": published_at,
            "raw_path": f"/benchmarks/v1/runs/{raw_name}",
            "raw_sha256": digest,
        }
        if not _json_values_equal(run, expected):
            raise ValidationError(
                f"catalog.runs[{index}] does not equal the projection of its raw report"
            )
        rebuilt_runs.append(expected)
    recorded_times = [run["recorded_at"] for run in rebuilt_runs]
    if recorded_times != sorted(recorded_times, reverse=True):
        raise ValidationError(
            "catalog.runs must be ordered by recorded_at descending"
        )
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": generated_at,
        "runs": rebuilt_runs,
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(
                value, stream, ensure_ascii=False, indent=2, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != contents:
            raise ValidationError(
                f"immutable raw path {path} exists with different bytes"
            )
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def publish(
    report_path: Path,
    data_root: Path,
    *,
    published_at: str | None = None,
    allow_dirty: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Validate and publish one immutable whole Benchmark v1 report."""

    report_bytes, report = _read_json(report_path)
    compact = validate_report(report, allow_dirty=allow_dirty)
    digest = hashlib.sha256(report_bytes).hexdigest()
    raw_name = f"sha256-{digest}.json"
    raw_path = data_root / "runs" / raw_name
    catalog_path = data_root / "catalog.json"
    catalog = _load_catalog(catalog_path)
    timestamp = _normalized_timestamp(
        published_at
        or datetime.now(UTC)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "published_at",
    )
    entry = {
        **compact,
        "id": digest,
        "slug": f"sha256-{digest}",
        "published_at": timestamp,
        "raw_path": f"/benchmarks/v1/runs/{raw_name}",
        "raw_sha256": digest,
    }
    existing = next(
        (run for run in catalog["runs"] if run.get("raw_sha256") == digest),
        None,
    )
    if existing is not None:
        expected = {**entry, "published_at": existing.get("published_at")}
        if not _json_values_equal(existing, expected):
            raise ValidationError(
                f"catalog projection for raw report {digest} does not match"
            )
        _write_immutable(raw_path, report_bytes)
        return digest, existing
    _write_immutable(raw_path, report_bytes)
    runs = [*catalog["runs"], entry]
    runs.sort(key=lambda run: run["recorded_at"], reverse=True)
    _atomic_write_json(
        catalog_path,
        {
            "benchmark_version": BENCHMARK_VERSION,
            "generated_at": timestamp,
            "runs": runs,
        },
    )
    return digest, entry


def _default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "docs/public/benchmarks/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        help="completed formal Benchmark v1 report JSON",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_data_root(),
        help="benchmark catalog directory",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="developer-only: accept a dirty local source report",
    )
    arguments = parser.parse_args()
    try:
        digest, entry = publish(
            arguments.report,
            arguments.data_root.resolve(),
            allow_dirty=arguments.allow_dirty,
        )
    except ValidationError as error:
        parser.exit(2, f"publication rejected: {error}\n")
    print(
        f"published sha256:{digest} ({len(entry['cases'])} Benchmark v1 cases)"
    )
    print(f"raw JSON: /benchmarks/v1/runs/sha256-{digest}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
