"""Strict data model for the immutable FHElium Benchmark v1 specification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from fhelium.benchmarks.model import BenchmarkResult

from ._validation import (
    normalize_json,
    require_fields,
    strict_name,
    strict_object,
    strict_optional_string,
    strict_optional_timestamp,
    strict_string,
    strict_timestamp,
)

BENCHMARK_VERSION = "v1"
_RESULT_FIELDS = {
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


def _result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    """Snapshot the Benchmark v1 leaf-result wire fields."""

    payload = result.to_dict()
    require_fields(payload, required=_RESULT_FIELDS, model="BenchmarkResult")
    normalized = normalize_json(payload, path="result")
    assert isinstance(normalized, dict)
    return normalized


def _result_from_dict(payload: Any) -> BenchmarkResult:
    """Parse the Benchmark v1 leaf-result wire fields without coercion."""

    if not isinstance(payload, Mapping):
        raise TypeError("BenchmarkResult must be an object")
    require_fields(payload, required=_RESULT_FIELDS, model="BenchmarkResult")
    for field_name in (
        "effective_parameters",
        "timed_boundary",
        "scalars",
        "metadata",
    ):
        if not isinstance(payload[field_name], Mapping):
            raise TypeError(f"BenchmarkResult.{field_name} must be an object")
    for field_name in ("metrics", "correctness", "rows", "evidence"):
        values = payload[field_name]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"BenchmarkResult.{field_name} must be an array")
        if not all(isinstance(value, Mapping) for value in values):
            raise TypeError(
                f"BenchmarkResult.{field_name} must contain only objects"
            )
    notes = payload["notes"]
    if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes)):
        raise TypeError("BenchmarkResult.notes must be an array")
    if not all(isinstance(note, str) for note in notes):
        raise TypeError("BenchmarkResult.notes must contain only strings")
    return BenchmarkResult.from_dict(payload)


class CaseStatus(StrEnum):
    """Lifecycle state of one Benchmark v1 case."""

    PENDING = "pending"
    RUNNING = "running"
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ReportStatus(StrEnum):
    """Lifecycle state of one Benchmark v1 report."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ExecutionBackend(StrEnum):
    """Native execution backend selected for one complete v1 run."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class BenchmarkExecution:
    """Report-level backend and indexed device shared by every v1 case."""

    backend: ExecutionBackend
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", ExecutionBackend(self.backend))
        object.__setattr__(
            self, "device", strict_string(self.device, "execution.device")
        )
        if self.backend is ExecutionBackend.CPU:
            if self.device != "cpu":
                raise ValueError("CPU execution device must be exactly 'cpu'")
        elif not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("CUDA execution device must be indexed cuda:N")

    def to_dict(self) -> dict[str, str]:
        return {"backend": self.backend.value, "device": self.device}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkExecution:
        require_fields(
            payload,
            required={"backend", "device"},
            model="BenchmarkExecution",
        )
        return cls(backend=payload["backend"], device=payload["device"])


@dataclass(frozen=True)
class BenchmarkCase:
    """One fixed leaf-benchmark invocation in Benchmark v1.

    ``parameters`` overrides the selected leaf profile. The v1 resolver
    records the resulting effective parameter object in its canonical
    manifest and in every case record. ``unavailable_reason`` declares a
    deterministic manifest-level absence; runners can additionally raise
    :class:`~fhelium.benchmarks.v1.BenchmarkCaseUnavailable` after preflight.
    """

    id: str
    title: str
    category: str
    benchmark: str
    workload_id: str
    profile: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requirements: Mapping[str, Any] = field(default_factory=dict)
    comparison: Mapping[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", strict_name(self.id, "case.id"))
        object.__setattr__(
            self, "title", strict_string(self.title, "case.title")
        )
        object.__setattr__(
            self, "category", strict_string(self.category, "case.category")
        )
        object.__setattr__(
            self, "benchmark", strict_name(self.benchmark, "case.benchmark")
        )
        object.__setattr__(
            self,
            "workload_id",
            strict_name(self.workload_id, "case.workload_id"),
        )
        if self.profile is not None:
            object.__setattr__(
                self, "profile", strict_name(self.profile, "case.profile")
            )
        object.__setattr__(
            self,
            "parameters",
            strict_object(self.parameters, "case.parameters"),
        )
        object.__setattr__(
            self,
            "requirements",
            strict_object(self.requirements, "case.requirements"),
        )
        object.__setattr__(
            self,
            "comparison",
            strict_object(self.comparison, "case.comparison"),
        )
        object.__setattr__(
            self,
            "unavailable_reason",
            strict_optional_string(
                self.unavailable_reason, "case.unavailable_reason"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "benchmark": self.benchmark,
            "workload_id": self.workload_id,
            "profile": self.profile,
            "parameters": normalize_json(self.parameters),
            "requirements": normalize_json(self.requirements),
            "comparison": normalize_json(self.comparison),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkCase:
        require_fields(
            payload,
            required={
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
            },
            model="BenchmarkCase",
        )
        return cls(
            id=payload["id"],
            title=payload["title"],
            category=payload["category"],
            benchmark=payload["benchmark"],
            workload_id=payload["workload_id"],
            profile=payload["profile"],
            parameters=payload["parameters"],
            requirements=payload["requirements"],
            comparison=payload["comparison"],
            unavailable_reason=payload["unavailable_reason"],
        )


@dataclass(frozen=True)
class BenchmarkSpecification:
    """The ordered, immutable Benchmark v1 case specification."""

    benchmark_version: str
    title: str
    description: str
    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self) -> None:
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(
                f"benchmark_version must be exactly {BENCHMARK_VERSION!r}"
            )
        object.__setattr__(
            self, "title", strict_string(self.title, "benchmark.title")
        )
        object.__setattr__(
            self,
            "description",
            strict_string(self.description, "benchmark.description"),
        )
        if not isinstance(self.cases, tuple):
            object.__setattr__(self, "cases", tuple(self.cases))
        for index, case in enumerate(self.cases):
            if not isinstance(case, BenchmarkCase):
                raise TypeError(
                    f"benchmark.cases[{index}] must be a BenchmarkCase"
                )
        ids = [case.id for case in self.cases]
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        if duplicates:
            raise ValueError(
                "benchmark case ids must be unique: " + ", ".join(duplicates)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "title": self.title,
            "description": self.description,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkSpecification:
        require_fields(
            payload,
            required={
                "benchmark_version",
                "title",
                "description",
                "cases",
            },
            model="BenchmarkSpecification",
        )
        cases = payload["cases"]
        if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
            raise TypeError("BenchmarkSpecification.cases must be an array")
        return cls(
            benchmark_version=payload["benchmark_version"],
            title=payload["title"],
            description=payload["description"],
            cases=tuple(BenchmarkCase.from_dict(case) for case in cases),
        )


@dataclass(frozen=True)
class ProbeError:
    """Non-fatal error from an optional platform-information probe."""

    probe: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe", strict_name(self.probe, "probe"))
        object.__setattr__(
            self, "error_type", strict_string(self.error_type, "error_type")
        )
        object.__setattr__(
            self, "message", strict_string(self.message, "message")
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "probe": self.probe,
            "error_type": self.error_type,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeError:
        require_fields(
            payload,
            required={"probe", "error_type", "message"},
            model="ProbeError",
        )
        return cls(
            probe=payload["probe"],
            error_type=payload["error_type"],
            message=payload["message"],
        )


@dataclass(frozen=True)
class FHEliumBuildIdentity:
    """FHElium distribution, source, and native-build provenance."""

    version: str
    distribution: Mapping[str, Any]
    source_git: Mapping[str, Any]
    native: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "version", strict_string(self.version, "version")
        )
        for name in ("distribution", "source_git", "native"):
            object.__setattr__(
                self,
                name,
                strict_object(getattr(self, name), f"fhelium_build.{name}"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "distribution": normalize_json(self.distribution),
            "source_git": normalize_json(self.source_git),
            "native": normalize_json(self.native),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FHEliumBuildIdentity:
        require_fields(
            payload,
            required={"version", "distribution", "source_git", "native"},
            model="FHEliumBuildIdentity",
        )
        return cls(
            version=payload["version"],
            distribution=payload["distribution"],
            source_git=payload["source_git"],
            native=payload["native"],
        )


@dataclass(frozen=True)
class PlatformSnapshot:
    """Normalized hardware, runtime, build, and invocation provenance."""

    system: Mapping[str, Any]
    cpu: Mapping[str, Any]
    memory: Mapping[str, Any]
    python: Mapping[str, Any]
    fhelium_build: FHEliumBuildIdentity
    torch: Mapping[str, Any]
    cuda: Mapping[str, Any]
    environment: Mapping[str, Any]
    invocation: tuple[str, ...]
    probe_errors: tuple[ProbeError, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "system",
            "cpu",
            "memory",
            "python",
            "torch",
            "cuda",
            "environment",
        ):
            object.__setattr__(
                self,
                name,
                strict_object(getattr(self, name), f"platform.{name}"),
            )
        if not isinstance(self.fhelium_build, FHEliumBuildIdentity):
            raise TypeError(
                "platform.fhelium_build must be FHEliumBuildIdentity"
            )
        invocation = tuple(
            strict_string(value, f"platform.invocation[{index}]")
            for index, value in enumerate(self.invocation)
        )
        object.__setattr__(self, "invocation", invocation)
        errors = tuple(self.probe_errors)
        if not all(isinstance(error, ProbeError) for error in errors):
            raise TypeError(
                "platform.probe_errors must contain ProbeError values"
            )
        object.__setattr__(self, "probe_errors", errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": normalize_json(self.system),
            "cpu": normalize_json(self.cpu),
            "memory": normalize_json(self.memory),
            "python": normalize_json(self.python),
            "fhelium_build": self.fhelium_build.to_dict(),
            "torch": normalize_json(self.torch),
            "cuda": normalize_json(self.cuda),
            "environment": normalize_json(self.environment),
            "invocation": list(self.invocation),
            "probe_errors": [error.to_dict() for error in self.probe_errors],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlatformSnapshot:
        require_fields(
            payload,
            required={
                "system",
                "cpu",
                "memory",
                "python",
                "fhelium_build",
                "torch",
                "cuda",
                "environment",
                "invocation",
                "probe_errors",
            },
            model="PlatformSnapshot",
        )
        invocation = payload["invocation"]
        errors = payload["probe_errors"]
        if not isinstance(invocation, Sequence) or isinstance(
            invocation, (str, bytes)
        ):
            raise TypeError("PlatformSnapshot.invocation must be an array")
        if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes)):
            raise TypeError("PlatformSnapshot.probe_errors must be an array")
        return cls(
            system=payload["system"],
            cpu=payload["cpu"],
            memory=payload["memory"],
            python=payload["python"],
            fhelium_build=FHEliumBuildIdentity.from_dict(
                payload["fhelium_build"]
            ),
            torch=payload["torch"],
            cuda=payload["cuda"],
            environment=payload["environment"],
            invocation=tuple(invocation),
            probe_errors=tuple(ProbeError.from_dict(error) for error in errors),
        )


@dataclass(frozen=True)
class CaseUnavailable:
    """Structured explanation for a case that cannot be measured."""

    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reason", strict_string(self.reason, "unavailable.reason")
        )
        object.__setattr__(
            self, "details", strict_object(self.details, "unavailable.details")
        )

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "details": normalize_json(self.details)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseUnavailable:
        require_fields(
            payload,
            required={"reason", "details"},
            model="CaseUnavailable",
        )
        return cls(reason=payload["reason"], details=payload["details"])


@dataclass(frozen=True)
class CaseFailure:
    """Structured exception information retained in a Benchmark v1 run."""

    stage: str
    error_type: str
    message: str
    traceback: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", strict_name(self.stage, "failure.stage")
        )
        object.__setattr__(
            self,
            "error_type",
            strict_string(self.error_type, "failure.error_type"),
        )
        object.__setattr__(
            self, "message", strict_string(self.message, "failure.message")
        )
        object.__setattr__(
            self,
            "traceback",
            tuple(
                strict_string(line, f"failure.traceback[{index}]")
                for index, line in enumerate(self.traceback)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": list(self.traceback),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseFailure:
        require_fields(
            payload,
            required={"stage", "error_type", "message", "traceback"},
            model="CaseFailure",
        )
        trace = payload["traceback"]
        if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
            raise TypeError("CaseFailure.traceback must be an array")
        return cls(
            stage=payload["stage"],
            error_type=payload["error_type"],
            message=payload["message"],
            traceback=tuple(trace),
        )


@dataclass
class CaseRecord:
    """Resolved case identity, execution state, and optional leaf result."""

    id: str
    title: str
    category: str
    benchmark: str
    workload_id: str
    profile: str
    parameters: Mapping[str, Any]
    requirements: Mapping[str, Any]
    comparison: Mapping[str, Any]
    status: CaseStatus = CaseStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    result: BenchmarkResult | None = None
    unavailable: CaseUnavailable | None = None
    failure: CaseFailure | None = None

    def __post_init__(self) -> None:
        self.id = strict_name(self.id, "case_record.id")
        self.title = strict_string(self.title, "case_record.title")
        self.category = strict_string(self.category, "case_record.category")
        self.benchmark = strict_name(self.benchmark, "case_record.benchmark")
        self.workload_id = strict_name(
            self.workload_id, "case_record.workload_id"
        )
        self.profile = strict_name(self.profile, "case_record.profile")
        self.parameters = strict_object(
            self.parameters, "case_record.parameters"
        )
        self.requirements = strict_object(
            self.requirements, "case_record.requirements"
        )
        self.comparison = strict_object(
            self.comparison, "case_record.comparison"
        )
        self.status = CaseStatus(self.status)
        self.started_at = strict_optional_timestamp(
            self.started_at, "case_record.started_at"
        )
        self.finished_at = strict_optional_timestamp(
            self.finished_at, "case_record.finished_at"
        )
        if self.result is not None and not isinstance(
            self.result, BenchmarkResult
        ):
            raise TypeError(
                "case_record.result must be BenchmarkResult or null"
            )
        if self.result is not None:
            _result_to_dict(self.result)
        if self.unavailable is not None and not isinstance(
            self.unavailable, CaseUnavailable
        ):
            raise TypeError(
                "case_record.unavailable must be CaseUnavailable or null"
            )
        if self.failure is not None and not isinstance(
            self.failure, CaseFailure
        ):
            raise TypeError("case_record.failure must be CaseFailure or null")
        self.validate_state()

    def validate_state(self) -> None:
        """Reject contradictory terminal-state payloads."""

        if self.status is CaseStatus.MEASURED:
            if (
                self.result is None
                or self.unavailable is not None
                or self.failure is not None
            ):
                raise ValueError("measured case requires only a result")
        elif self.status is CaseStatus.UNAVAILABLE:
            if (
                self.unavailable is None
                or self.result is not None
                or self.failure is not None
            ):
                raise ValueError(
                    "unavailable case requires only unavailable information"
                )
        elif self.status in (CaseStatus.FAILED, CaseStatus.INTERRUPTED):
            if self.failure is None or self.unavailable is not None:
                raise ValueError(
                    "failed/interrupted case requires failure information, "
                    "permits an optional partial result, and forbids "
                    "unavailable information"
                )
        elif any(
            value is not None
            for value in (self.result, self.unavailable, self.failure)
        ):
            raise ValueError(
                "pending/running case cannot contain a terminal payload"
            )

        if self.status is CaseStatus.PENDING:
            if self.started_at is not None or self.finished_at is not None:
                raise ValueError("pending case cannot contain timestamps")
        elif self.status is CaseStatus.RUNNING:
            if self.started_at is None or self.finished_at is not None:
                raise ValueError(
                    "running case requires a start and no finish timestamp"
                )
        elif self.started_at is None or self.finished_at is None:
            raise ValueError(
                "terminal case requires start and finish timestamps"
            )
        if (
            self.started_at is not None
            and self.finished_at is not None
            and datetime.fromisoformat(self.finished_at.replace("Z", "+00:00"))
            < datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
        ):
            raise ValueError("case finish timestamp precedes its start")

    def to_dict(self) -> dict[str, Any]:
        self.validate_state()
        result = None
        if self.result is not None:
            result = _result_to_dict(self.result)
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "benchmark": self.benchmark,
            "workload_id": self.workload_id,
            "profile": self.profile,
            "parameters": normalize_json(self.parameters),
            "requirements": normalize_json(self.requirements),
            "comparison": normalize_json(self.comparison),
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": result,
            "unavailable": (
                None if self.unavailable is None else self.unavailable.to_dict()
            ),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseRecord:
        require_fields(
            payload,
            required={
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
            },
            model="CaseRecord",
        )
        result = payload["result"]
        unavailable = payload["unavailable"]
        failure = payload["failure"]
        return cls(
            id=payload["id"],
            title=payload["title"],
            category=payload["category"],
            benchmark=payload["benchmark"],
            workload_id=payload["workload_id"],
            profile=payload["profile"],
            parameters=payload["parameters"],
            requirements=payload["requirements"],
            comparison=payload["comparison"],
            status=payload["status"],
            started_at=payload["started_at"],
            finished_at=payload["finished_at"],
            result=None if result is None else _result_from_dict(result),
            unavailable=(
                None
                if unavailable is None
                else CaseUnavailable.from_dict(unavailable)
            ),
            failure=(
                None if failure is None else CaseFailure.from_dict(failure)
            ),
        )


@dataclass
class BenchmarkReport:
    """Outer report for one complete or checkpointed Benchmark v1 run.

    ``benchmark_version`` versions the entire Benchmark specification.
    FHElium, Python, Torch, CUDA, and native-build versions remain platform
    provenance. No total or composite score is defined.
    """

    benchmark_version: str
    manifest_sha256: str
    execution: BenchmarkExecution
    platform: PlatformSnapshot
    started_at: str
    finished_at: str | None
    status: ReportStatus
    cases: list[CaseRecord]

    def __post_init__(self) -> None:
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(
                f"benchmark_version must be exactly {BENCHMARK_VERSION!r}"
            )
        self.manifest_sha256 = strict_string(
            self.manifest_sha256, "manifest_sha256"
        )
        if len(self.manifest_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.manifest_sha256
        ):
            raise ValueError(
                "manifest_sha256 must be a lowercase SHA-256 digest"
            )
        if not isinstance(self.platform, PlatformSnapshot):
            raise TypeError("platform must be PlatformSnapshot")
        if not isinstance(self.execution, BenchmarkExecution):
            raise TypeError("execution must be BenchmarkExecution")
        self.started_at = strict_timestamp(self.started_at, "started_at")
        self.finished_at = strict_optional_timestamp(
            self.finished_at, "finished_at"
        )
        self.status = ReportStatus(self.status)
        if not isinstance(self.cases, list):
            self.cases = list(self.cases)
        if not all(isinstance(case, CaseRecord) for case in self.cases):
            raise TypeError("cases must contain CaseRecord values")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("report case ids must be unique")
        self.validate_state()

    @property
    def requires_nonzero_exit(self) -> bool:
        """Whether a CLI should fail due to failure or interruption."""

        return self.status is ReportStatus.INTERRUPTED or any(
            case.status is CaseStatus.FAILED for case in self.cases
        )

    @property
    def suggested_exit_code(self) -> int:
        """Return the decision for a CLI without exiting the process."""

        return 1 if self.requires_nonzero_exit else 0

    def to_dict(self) -> dict[str, Any]:
        self.validate_state()
        return {
            "benchmark_version": self.benchmark_version,
            "manifest_sha256": self.manifest_sha256,
            "execution": self.execution.to_dict(),
            "platform": self.platform.to_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status.value,
            "cases": [case.to_dict() for case in self.cases],
        }

    def validate_state(self) -> None:
        """Reject contradictory report lifecycle and outcome fields."""

        if self.status is ReportStatus.RUNNING:
            if self.finished_at is not None:
                raise ValueError(
                    "running report cannot have a finish timestamp"
                )
            return
        if self.finished_at is None:
            raise ValueError("terminal report requires a finish timestamp")
        if datetime.fromisoformat(
            self.finished_at.replace("Z", "+00:00")
        ) < datetime.fromisoformat(self.started_at.replace("Z", "+00:00")):
            raise ValueError("report finish timestamp precedes its start")
        failed = any(case.status is CaseStatus.FAILED for case in self.cases)
        interrupted = any(
            case.status is CaseStatus.INTERRUPTED for case in self.cases
        )
        active = any(
            case.status in (CaseStatus.PENDING, CaseStatus.RUNNING)
            for case in self.cases
        )
        if self.status is ReportStatus.COMPLETED and (
            failed or interrupted or active
        ):
            raise ValueError(
                "completed report may contain only measured and unavailable "
                "cases"
            )
        if self.status is ReportStatus.FAILED:
            if not failed:
                raise ValueError("failed report requires a failed case")
            if interrupted or active:
                raise ValueError(
                    "failed report cannot contain active or interrupted cases"
                )
        if self.status is ReportStatus.INTERRUPTED and not interrupted:
            raise ValueError("interrupted report requires an interrupted case")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkReport:
        require_fields(
            payload,
            required={
                "benchmark_version",
                "manifest_sha256",
                "execution",
                "platform",
                "started_at",
                "finished_at",
                "status",
                "cases",
            },
            model="BenchmarkReport",
        )
        cases = payload["cases"]
        if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
            raise TypeError("BenchmarkReport.cases must be an array")
        report = cls(
            benchmark_version=payload["benchmark_version"],
            manifest_sha256=payload["manifest_sha256"],
            execution=BenchmarkExecution.from_dict(payload["execution"]),
            platform=PlatformSnapshot.from_dict(payload["platform"]),
            started_at=payload["started_at"],
            finished_at=payload["finished_at"],
            status=payload["status"],
            cases=[CaseRecord.from_dict(case) for case in cases],
        )
        from .manifest import validate_report_specification

        validate_report_specification(report)
        return report
