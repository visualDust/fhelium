"""Errors raised by JIT capture, passes, planning, and execution."""

from fhelium.errors import FHEliumError


class JitError(FHEliumError):
    """Base error for JIT capture, transformation, and execution."""


class JitTraceError(JitError, RuntimeError):
    """A Python or PyTorch construct cannot be captured safely."""


class JitInputError(JitError, ValueError):
    """An input role, shape, or runtime value violates its declared requirements."""


class JitPlanningError(JitError, RuntimeError):
    """A valid captured graph cannot satisfy the requested CKKS state plan."""


class JitPassError(JitError, RuntimeError):
    """A graph pass, pipeline, or explicit validation gate failed."""
