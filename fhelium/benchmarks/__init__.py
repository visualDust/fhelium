"""Public benchmark definitions, registry, and custom-workload interfaces."""

from fhelium.benchmarks.model import (
    BenchmarkCheck,
    BenchmarkDefinition,
    BenchmarkMetric,
    BenchmarkProfile,
    BenchmarkResult,
    BenchmarkTimedBoundary,
)
from fhelium.benchmarks.registry import (
    BenchmarkRegistry,
    load_builtin_benchmarks,
    register_benchmark,
    registry,
)

__all__ = [
    "BenchmarkCheck",
    "BenchmarkDefinition",
    "BenchmarkMetric",
    "BenchmarkProfile",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkTimedBoundary",
    "load_builtin_benchmarks",
    "register_benchmark",
    "registry",
]
