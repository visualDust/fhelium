"""Errors reported by source-oriented compiler frontends and drivers."""


class CompileError(Exception):
    """Base error for the source-oriented compile package."""


class CompileInputError(CompileError, ValueError):
    """Reject a malformed source or frontend declaration."""


class CaptureError(CompileError, RuntimeError):
    """Reject a source callable that the selected frontend cannot capture."""


class PlanningError(CompileError, RuntimeError):
    """Report that a selected transform cannot make its requested decision."""


__all__ = [
    "CaptureError",
    "CompileError",
    "CompileInputError",
    "PlanningError",
]
