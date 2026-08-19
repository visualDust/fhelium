"""FHElium Benchmark v1 model, report I/O, and fixed-run entrypoint."""

from .io import read_report, write_report_atomic
from .model import (
    BenchmarkExecution,
    BENCHMARK_VERSION,
    BenchmarkReport,
    CaseFailure,
    CaseRecord,
    CaseStatus,
    CaseUnavailable,
    FHEliumBuildIdentity,
    PlatformSnapshot,
    ProbeError,
    ReportStatus,
)
from .platform import collect_platform, sanitize_invocation
from .runner import (
    BenchmarkRunner,
    run_benchmark,
)

__all__ = [
    "BENCHMARK_VERSION",
    "BenchmarkReport",
    "BenchmarkRunner",
    "BenchmarkExecution",
    "CaseFailure",
    "CaseRecord",
    "CaseStatus",
    "CaseUnavailable",
    "FHEliumBuildIdentity",
    "PlatformSnapshot",
    "ProbeError",
    "ReportStatus",
    "collect_platform",
    "read_report",
    "run_benchmark",
    "sanitize_invocation",
    "write_report_atomic",
]
