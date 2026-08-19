"""Data model for independent FHElium benchmark workloads and results."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

ProgressCallback = Callable[[str], None]
BenchmarkRunner = Callable[
    ["BenchmarkProfile", ProgressCallback], "BenchmarkResult"
]
MetricDirection = Literal["lower", "higher", "none"]


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _finite_number(value: Any, path: str) -> int | float:
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"{path} must be a finite int or float")


def _json_scalar(value: Any, path: str) -> Any:
    if value is None or type(value) in (bool, int) or isinstance(value, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"{path} must be a finite JSON scalar")


def _normalize_json(value: Any, path: str) -> Any:
    if value is None or type(value) in (bool, int) or isinstance(value, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _normalize_json(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _normalize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    result = _normalize_json(value, path)
    assert isinstance(result, dict)
    return result


@dataclass(frozen=True)
class BenchmarkProfile:
    """Named parameter set exposed by the CLI and TUI.

    ``parameters`` contains JSON-compatible defaults understood by the
    corresponding benchmark runner. :meth:`with_overrides` returns a new
    profile and does not mutate the registered definition.
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "profile.name"))
        object.__setattr__(
            self,
            "description",
            _string(self.description, "profile.description"),
        )
        object.__setattr__(
            self,
            "parameters",
            _object(self.parameters, "profile.parameters"),
        )

    def with_overrides(self, overrides: Mapping[str, Any]) -> BenchmarkProfile:
        parameters = dict(self.parameters)
        parameters.update(overrides)
        return replace(self, parameters=parameters)


@dataclass(frozen=True)
class BenchmarkDefinition:
    """One discoverable benchmark workload and its execution profiles."""

    name: str
    title: str
    description: str
    profiles: tuple[BenchmarkProfile, ...]
    runner: BenchmarkRunner
    workload_id: str
    category: str = "workload"
    requirements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "definition.name"))
        object.__setattr__(
            self, "title", _string(self.title, "definition.title")
        )
        object.__setattr__(
            self,
            "description",
            _string(self.description, "definition.description"),
        )
        object.__setattr__(
            self,
            "workload_id",
            _string(self.workload_id, "definition.workload_id"),
        )
        object.__setattr__(
            self, "category", _string(self.category, "definition.category")
        )
        profiles = tuple(self.profiles)
        if not profiles or not all(
            isinstance(profile, BenchmarkProfile) for profile in profiles
        ):
            raise TypeError(
                "definition.profiles must contain BenchmarkProfile values"
            )
        names = [profile.name for profile in profiles]
        if len(names) != len(set(names)):
            raise ValueError("definition profile names must be unique")
        object.__setattr__(self, "profiles", profiles)
        if not callable(self.runner):
            raise TypeError("definition.runner must be callable")
        object.__setattr__(
            self,
            "requirements",
            _object(self.requirements, "definition.requirements"),
        )

    def profile(self, name: str | None = None) -> BenchmarkProfile:
        if name is None:
            return self.profiles[0]
        for profile in self.profiles:
            if profile.name == name:
                return profile
        choices = ", ".join(profile.name for profile in self.profiles)
        raise KeyError(
            f"Unknown profile {name!r} for {self.name!r}; choices: {choices}"
        )


@dataclass(frozen=True)
class BenchmarkTimedBoundary:
    """Stable definition of what one benchmark measurement includes."""

    id: str
    description: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    synchronization: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _string(self.id, "timed_boundary.id"))
        object.__setattr__(
            self,
            "description",
            _string(self.description, "timed_boundary.description"),
        )
        for field_name in ("includes", "excludes"):
            values = tuple(
                _string(value, f"timed_boundary.{field_name}[{index}]")
                for index, value in enumerate(getattr(self, field_name))
            )
            object.__setattr__(self, field_name, values)
        object.__setattr__(
            self,
            "synchronization",
            _string(self.synchronization, "timed_boundary.synchronization"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "includes": list(self.includes),
            "excludes": list(self.excludes),
            "synchronization": self.synchronization,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkTimedBoundary:
        required = {
            "id",
            "description",
            "includes",
            "excludes",
            "synchronization",
        }
        if set(payload) != required:
            raise ValueError("invalid BenchmarkTimedBoundary fields")
        return cls(
            id=payload["id"],
            description=payload["description"],
            includes=tuple(payload["includes"]),
            excludes=tuple(payload["excludes"]),
            synchronization=payload["synchronization"],
        )


@dataclass(frozen=True)
class BenchmarkMetric:
    """One normalized finite benchmark measurement."""

    name: str
    value: int | float
    unit: str
    statistic: str
    direction: MetricDirection
    dimensions: Mapping[str, Any] = field(default_factory=dict)
    samples: tuple[int | float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "metric.name"))
        object.__setattr__(
            self, "value", _finite_number(self.value, "metric.value")
        )
        object.__setattr__(self, "unit", _string(self.unit, "metric.unit"))
        object.__setattr__(
            self,
            "statistic",
            _string(self.statistic, "metric.statistic"),
        )
        if self.direction not in {"lower", "higher", "none"}:
            raise ValueError(
                "metric.direction must be 'lower', 'higher', or 'none'"
            )
        dimensions = {
            key: _json_scalar(value, f"metric.dimensions.{key}")
            for key, value in _object(
                self.dimensions, "metric.dimensions"
            ).items()
        }
        object.__setattr__(self, "dimensions", dimensions)
        samples = tuple(
            _finite_number(value, f"metric.samples[{index}]")
            for index, value in enumerate(self.samples)
        )
        object.__setattr__(self, "samples", samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "statistic": self.statistic,
            "direction": self.direction,
            "dimensions": dict(self.dimensions),
            "samples": list(self.samples),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkMetric:
        required = {
            "name",
            "value",
            "unit",
            "statistic",
            "direction",
            "dimensions",
            "samples",
        }
        if set(payload) != required:
            raise ValueError("invalid benchmark metric fields")
        return cls(
            name=payload["name"],
            value=payload["value"],
            unit=payload["unit"],
            statistic=payload["statistic"],
            direction=payload["direction"],
            dimensions=payload["dimensions"],
            samples=tuple(payload["samples"]),
        )


@dataclass(frozen=True)
class BenchmarkCheck:
    """One recorded correctness observation and acceptance rule."""

    name: str
    passed: bool
    oracle: str
    metric: str
    observed: Any
    comparison: str
    limit: Any | None
    unit: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "check.name"))
        if type(self.passed) is not bool:
            raise TypeError("check.passed must be a bool")
        object.__setattr__(self, "oracle", _string(self.oracle, "check.oracle"))
        object.__setattr__(self, "metric", _string(self.metric, "check.metric"))
        object.__setattr__(
            self,
            "observed",
            _json_scalar(self.observed, "check.observed"),
        )
        object.__setattr__(
            self,
            "comparison",
            _string(self.comparison, "check.comparison"),
        )
        object.__setattr__(
            self, "limit", _json_scalar(self.limit, "check.limit")
        )
        if self.unit is not None:
            object.__setattr__(self, "unit", _string(self.unit, "check.unit"))
        object.__setattr__(
            self, "details", _object(self.details, "check.details")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "oracle": self.oracle,
            "metric": self.metric,
            "observed": self.observed,
            "comparison": self.comparison,
            "limit": self.limit,
            "unit": self.unit,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkCheck:
        required = {
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
        if set(payload) != required:
            raise ValueError("invalid benchmark correctness-check fields")
        return cls(
            name=payload["name"],
            passed=payload["passed"],
            oracle=payload["oracle"],
            metric=payload["metric"],
            observed=payload["observed"],
            comparison=payload["comparison"],
            limit=payload["limit"],
            unit=payload["unit"],
            details=payload["details"],
        )


@dataclass
class BenchmarkResult:
    """Common result header plus workload-specific benchmark evidence.

    Normalized metrics and correctness checks are the stable comparison and
    publication surface. ``rows``, ``scalars``, ``metadata``, and ``evidence``
    remain extensible for workload-specific diagnostics.
    """

    benchmark: str
    profile: str
    workload_id: str
    effective_parameters: Mapping[str, Any]
    timed_boundary: BenchmarkTimedBoundary
    metrics: Sequence[BenchmarkMetric]
    correctness: Sequence[BenchmarkCheck]
    rows: list[dict[str, Any]] = field(default_factory=list)
    scalars: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.benchmark = _string(self.benchmark, "result.benchmark")
        self.profile = _string(self.profile, "result.profile")
        self.workload_id = _string(self.workload_id, "result.workload_id")
        self.effective_parameters = _object(
            self.effective_parameters, "result.effective_parameters"
        )
        if not isinstance(self.timed_boundary, BenchmarkTimedBoundary):
            raise TypeError(
                "result.timed_boundary must be BenchmarkTimedBoundary"
            )
        self.metrics = tuple(self.metrics)
        if not all(
            isinstance(metric, BenchmarkMetric) for metric in self.metrics
        ):
            raise TypeError(
                "result.metrics must contain BenchmarkMetric values"
            )
        self.correctness = tuple(self.correctness)
        if not all(
            isinstance(check, BenchmarkCheck) for check in self.correctness
        ):
            raise TypeError(
                "result.correctness must contain BenchmarkCheck values"
            )
        rows = _normalize_json(self.rows, "result.rows")
        if not isinstance(rows, list) or not all(
            isinstance(row, dict) for row in rows
        ):
            raise TypeError("result.rows must be an array of objects")
        self.rows = rows
        self.scalars = _object(self.scalars, "result.scalars")
        self.metadata = _object(self.metadata, "result.metadata")
        notes = _normalize_json(self.notes, "result.notes")
        if not isinstance(notes, list) or not all(
            isinstance(note, str) for note in notes
        ):
            raise TypeError("result.notes must be an array of strings")
        self.notes = notes
        evidence = _normalize_json(self.evidence, "result.evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(row, dict) for row in evidence
        ):
            raise TypeError("result.evidence must be an array of objects")
        self.evidence = evidence
        for key, expected in {"workload_id": self.workload_id}.items():
            declared = self.metadata.get(key)
            if declared is not None and declared != expected:
                raise ValueError(
                    f"result.metadata.{key} does not match result.{key}"
                )

    @property
    def correctness_passed(self) -> bool:
        """Whether at least one check exists and all checks passed."""

        return bool(self.correctness) and all(
            check.passed for check in self.correctness
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "profile": self.profile,
            "workload_id": self.workload_id,
            "effective_parameters": dict(self.effective_parameters),
            "timed_boundary": self.timed_boundary.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "correctness": [check.to_dict() for check in self.correctness],
            "rows": self.rows,
            "scalars": self.scalars,
            "metadata": self.metadata,
            "notes": self.notes,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkResult:
        required = {
            "benchmark",
            "profile",
            "workload_id",
            "effective_parameters",
            "timed_boundary",
            "metrics",
            "correctness",
            "rows",
            "scalars",
            "metadata",
            "notes",
            "evidence",
        }
        if set(payload) != required:
            raise ValueError("invalid benchmark-result fields")
        metrics = payload["metrics"]
        correctness = payload["correctness"]
        if not isinstance(metrics, Sequence) or isinstance(
            metrics, (str, bytes)
        ):
            raise TypeError("result.metrics must be an array")
        if not isinstance(correctness, Sequence) or isinstance(
            correctness, (str, bytes)
        ):
            raise TypeError("result.correctness must be an array")
        return cls(
            benchmark=payload["benchmark"],
            profile=payload["profile"],
            workload_id=payload["workload_id"],
            effective_parameters=payload["effective_parameters"],
            timed_boundary=BenchmarkTimedBoundary.from_dict(
                payload["timed_boundary"]
            ),
            metrics=tuple(BenchmarkMetric.from_dict(row) for row in metrics),
            correctness=tuple(
                BenchmarkCheck.from_dict(row) for row in correctness
            ),
            rows=[dict(row) for row in payload["rows"]],
            scalars=dict(payload["scalars"]),
            metadata=dict(payload["metadata"]),
            notes=[str(note) for note in payload["notes"]],
            evidence=[dict(row) for row in payload["evidence"]],
        )
