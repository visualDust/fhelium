"""Helpers used by the ``fhelium benchmark`` CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

from fhelium.benchmarks.registry import BenchmarkRegistry


def load_benchmark_file(path: Path, registry: BenchmarkRegistry) -> None:
    """Load custom registrations from a Python file.

    A file may call :func:`register_benchmark` at import time, or expose a
    ``register_benchmarks(registry)`` function.
    """

    module_name = f"fhelium_custom_benchmark_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load benchmark file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hook = getattr(module, "register_benchmarks", None)
    if hook is not None:
        hook(registry)


def parse_overrides(values: tuple[str, ...]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for expression in values:
        if "=" not in expression:
            raise ValueError(
                f"Invalid override {expression!r}; expected KEY=VALUE"
            )
        key, raw_value = expression.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override keys cannot be empty")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides
