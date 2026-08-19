"""Persistent dictionary workspace shared by JIT passes and execution."""

from __future__ import annotations

from typing import Any


class Workspace(dict[Any, Any]):
    """Retain graph-external state across JIT passes and execution.

    Callers and passes may exchange analyses, policies, diagnostics, caches,
    and extension-defined objects through arbitrary keys. The JIT runtime
    reserves ``materials``, ``resources``, ``handlers``, ``torch_handlers``,
    ``engine``, ``evaluation_keys``, ``public_key``, ``material_resolver``, and
    ``resource_resolver`` for execution bindings and services.

    A Program stores only material and resource symbols. Readiness verifies
    that each selected symbol has a binding and that supplied resolvers are
    callable; it leaves both bindings and resolvers untouched. Execution calls
    the corresponding resolver when each reference operation is encountered,
    passing ``(symbol, kind, binding, workspace)``. The resolver's return value
    becomes that operation's runtime value. Pipeline and execution APIs retain
    the exact Workspace object supplied by the caller.
    """


__all__ = ["Workspace"]
