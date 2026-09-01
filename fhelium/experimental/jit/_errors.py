"""Errors raised by runtime-oriented JIT transformation and interpretation."""

from fhelium.errors import FHEliumError


class JitError(FHEliumError):
    """Base error for JIT transformation, build, and execution."""


class JitInputError(JitError, ValueError):
    """A runtime input cannot be bound to the represented entry interface."""


class JitInterpreterError(JitError, RuntimeError):
    """The IR interpreter cannot represent or execute a request."""


__all__ = ["JitError", "JitInputError", "JitInterpreterError"]
