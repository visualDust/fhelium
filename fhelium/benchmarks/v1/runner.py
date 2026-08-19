"""Sequential, checkpointing execution of FHElium Benchmark v1."""

from __future__ import annotations

import traceback as traceback_module
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

from fhelium.benchmarks.model import BenchmarkResult

from ._validation import normalize_json, strict_object, strict_string
from .io import write_report_atomic
from .manifest import (
    ResolvedCase,
    resolve_benchmark,
    validate_report_specification,
)
from .model import (
    BENCHMARK_VERSION,
    BenchmarkExecution,
    BenchmarkReport,
    CaseFailure,
    CaseRecord,
    CaseStatus,
    CaseUnavailable,
    PlatformSnapshot,
    ReportStatus,
)
from .platform import collect_platform

ProgressCallback = Callable[[str, str], None]
ReportWriter = Callable[[str | PathLike[str], BenchmarkReport], None]


class BenchmarkCaseUnavailable(Exception):
    """Signal that leaf preflight cannot measure a case on this machine."""

    def __init__(
        self,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = strict_string(reason, "unavailable.reason")
        self.details = strict_object(details or {}, "unavailable.details")
        super().__init__(self.reason)


class BenchmarkCaseFailed(Exception):
    """Signal a failure while retaining an optional partial leaf result."""

    def __init__(
        self,
        message: str,
        *,
        result: BenchmarkResult | None = None,
    ) -> None:
        self.message = strict_string(message, "failure.message")
        if result is not None and not isinstance(result, BenchmarkResult):
            raise TypeError("failure result must be BenchmarkResult or null")
        self.result = result
        super().__init__(self.message)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _failure(error: BaseException, *, stage: str) -> CaseFailure:
    home = str(Path.home())

    def portable(value: str) -> str:
        return value.replace(home, "<home>") if home else value

    lines: list[str] = []
    for block in traceback_module.format_exception(error):
        lines.extend(
            portable(line) for line in block.rstrip().splitlines() if line
        )
    return CaseFailure(
        stage=stage,
        error_type=type(error).__name__,
        message=portable(str(error)) or "(exception carried no message)",
        traceback=tuple(lines) or (f"{type(error).__name__}",),
    )


def _record(resolved: ResolvedCase) -> CaseRecord:
    case = resolved.case
    return CaseRecord(
        id=case.id,
        title=case.title,
        category=case.category,
        benchmark=case.benchmark,
        workload_id=case.workload_id,
        profile=resolved.profile.name,
        parameters=resolved.parameters,
        requirements=case.requirements,
        comparison=case.comparison,
    )


def _unavailable_requirement(
    resolved: ResolvedCase,
    platform: PlatformSnapshot,
) -> CaseUnavailable | None:
    """Evaluate the small portable capability vocabulary used by v1."""

    requirements = resolved.case.requirements
    allowed = requirements.get("native_backends_any")
    if allowed is not None and tuple(allowed) != ("cpu", "cuda"):
        raise ValueError(
            f"case {resolved.case.id!r} has invalid native_backends_any"
        )
    return None


def _validate_result(result: Any, resolved: ResolvedCase) -> BenchmarkResult:
    if not isinstance(result, BenchmarkResult):
        raise TypeError("leaf runner must return BenchmarkResult")
    if result.benchmark != resolved.definition.name:
        raise ValueError(
            "leaf result benchmark does not match the resolved definition: "
            f"{result.benchmark!r} != {resolved.definition.name!r}"
        )
    if result.workload_id != resolved.case.workload_id:
        raise ValueError(
            "leaf result workload_id does not match the Benchmark v1 case: "
            f"{result.workload_id!r} != {resolved.case.workload_id!r}"
        )
    if result.profile != resolved.profile.name:
        raise ValueError(
            "leaf result profile does not match the effective profile: "
            f"{result.profile!r} != {resolved.profile.name!r}"
        )
    declared_workload_id = result.metadata.get("workload_id")
    if (
        declared_workload_id is not None
        and declared_workload_id != resolved.case.workload_id
    ):
        raise ValueError(
            "leaf result metadata workload_id does not match the Benchmark v1 "
            f"case: {declared_workload_id!r} != {resolved.case.workload_id!r}"
        )
    if normalize_json(result.effective_parameters) != normalize_json(
        resolved.parameters
    ):
        raise ValueError(
            "leaf result effective_parameters do not match the resolved "
            "Benchmark v1 parameters"
        )
    if not result.metrics:
        raise ValueError("leaf result must contain at least one metric")

    normalize_json(result.to_dict(), path=f"result.{resolved.case.id}")
    return result


class BenchmarkRunner:
    """Run the fixed Benchmark v1 specification with atomic checkpoints.

    The constructor binds the five package-owned cases and definitions
    directly. The CLI and :func:`run_benchmark` accept no registry or
    case/profile/parameter overrides.
    """

    def __init__(
        self,
        *,
        platform_collector: Callable[..., PlatformSnapshot] = collect_platform,
        report_writer: ReportWriter = write_report_atomic,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._platform_collector = platform_collector
        self._report_writer = report_writer
        self._clock = clock

    def run(
        self,
        *,
        execution: BenchmarkExecution,
        output_path: str | PathLike[str],
        invocation: Sequence[str] | None = None,
        progress: ProgressCallback | None = None,
        platform_snapshot: PlatformSnapshot | None = None,
    ) -> BenchmarkReport:
        """Run Benchmark v1, checkpoint after every case, and return its report."""

        from .cases import FIXED_DEFINITIONS, SPECIFICATION

        manifest = resolve_benchmark(SPECIFICATION, FIXED_DEFINITIONS)
        if platform_snapshot is None:
            platform_snapshot = self._platform_collector(invocation=invocation)
        if not isinstance(platform_snapshot, PlatformSnapshot):
            raise TypeError("platform collector must return PlatformSnapshot")
        if not isinstance(execution, BenchmarkExecution):
            raise TypeError("execution must be BenchmarkExecution")

        report = BenchmarkReport(
            benchmark_version=BENCHMARK_VERSION,
            manifest_sha256=manifest.sha256,
            execution=execution,
            platform=platform_snapshot,
            started_at=self._clock(),
            finished_at=None,
            status=ReportStatus.RUNNING,
            cases=[_record(case) for case in manifest.cases],
        )
        validate_report_specification(report)

        interrupted = False
        for resolved, record in zip(manifest.cases, report.cases, strict=True):
            if resolved.case.unavailable_reason is not None:
                timestamp = self._clock()
                record.started_at = timestamp
                record.finished_at = timestamp
                record.status = CaseStatus.UNAVAILABLE
                record.unavailable = CaseUnavailable(
                    reason=resolved.case.unavailable_reason,
                    details={"source": "benchmark-manifest"},
                )
                self._report_writer(output_path, report)
                continue

            try:
                capability_unavailable = _unavailable_requirement(
                    resolved, platform_snapshot
                )
            except Exception as error:
                timestamp = self._clock()
                record.started_at = timestamp
                record.finished_at = timestamp
                record.status = CaseStatus.FAILED
                record.failure = _failure(error, stage="preflight")
                self._report_writer(output_path, report)
                continue
            if capability_unavailable is not None:
                timestamp = self._clock()
                record.started_at = timestamp
                record.finished_at = timestamp
                record.status = CaseStatus.UNAVAILABLE
                record.unavailable = capability_unavailable
                self._report_writer(output_path, report)
                continue

            record.status = CaseStatus.RUNNING
            record.started_at = self._clock()
            try:

                def callback(message: str, case_id: str = record.id) -> None:
                    if progress is not None:
                        progress(case_id, message)

                runner: Any = resolved.definition.runner
                result = runner(
                    resolved.profile,
                    callback,
                    execution=execution,
                )
                record.result = _validate_result(result, resolved)
                if not record.result.correctness_passed:
                    raise BenchmarkCaseFailed(
                        "leaf result did not pass non-empty validation criteria",
                        result=record.result,
                    )
                record.status = CaseStatus.MEASURED
            except BenchmarkCaseUnavailable as unavailable:
                record.unavailable = CaseUnavailable(
                    reason=unavailable.reason,
                    details=unavailable.details,
                )
                record.status = CaseStatus.UNAVAILABLE
            except BenchmarkCaseFailed as error:
                if error.result is not None:
                    try:
                        record.result = _validate_result(error.result, resolved)
                    except Exception as validation_error:
                        record.failure = _failure(
                            validation_error, stage="result-validation"
                        )
                    else:
                        record.failure = _failure(error, stage="execution")
                else:
                    record.failure = _failure(error, stage="execution")
                record.status = CaseStatus.FAILED
            except KeyboardInterrupt as error:
                record.failure = _failure(error, stage="execution")
                record.status = CaseStatus.INTERRUPTED
                interrupted = True
            except Exception as error:
                record.failure = _failure(error, stage="execution")
                record.status = CaseStatus.FAILED
            finally:
                record.finished_at = self._clock()
                self._report_writer(output_path, report)
            if interrupted:
                break

        report.finished_at = self._clock()
        if interrupted:
            report.status = ReportStatus.INTERRUPTED
        elif report.requires_nonzero_exit:
            report.status = ReportStatus.FAILED
        else:
            report.status = ReportStatus.COMPLETED
        self._report_writer(output_path, report)
        return report


def run_benchmark(
    *,
    execution: BenchmarkExecution,
    output_path: str | PathLike[str],
    invocation: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkReport:
    """Run the package-owned Benchmark v1 specification."""

    return BenchmarkRunner().run(
        execution=execution,
        output_path=output_path,
        invocation=invocation,
        progress=progress,
    )
