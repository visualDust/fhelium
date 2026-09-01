"""Generated Python source emitted from one Compile Program stage."""

from __future__ import annotations

from dataclasses import dataclass


class PythonCodegenError(RuntimeError):
    """Report that a Python emitter cannot represent the scanned Program."""


@dataclass(frozen=True)
class GeneratedPythonSource:
    """Hold editable Python source and its external symbol interface."""

    source: str
    entry_point: str
    input_names: tuple[str, ...]
    material_symbols: tuple[str, ...]
    resource_symbols: tuple[str, ...]
    operation_count: int


@dataclass(frozen=True)
class EagerPythonSource(GeneratedPythonSource):
    """Python source expressed through the public Eager Engine API."""


@dataclass(frozen=True)
class BackendPythonSource(GeneratedPythonSource):
    """Python source expressed through resolved Backend implementation calls."""

    resource_requirements: tuple[tuple[str, str], ...] = ()


__all__ = [
    "BackendPythonSource",
    "EagerPythonSource",
    "GeneratedPythonSource",
    "PythonCodegenError",
]
