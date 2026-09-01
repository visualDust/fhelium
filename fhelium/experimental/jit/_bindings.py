"""Live objects and extension handlers supplied to JIT execution."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from xdsl.ir import Operation

OperationHandler: TypeAlias = Callable[
    [Operation, tuple[object, ...], MutableMapping[Any, Any]], object
]
BindingResolver: TypeAlias = Callable[
    [str, str | None, object, MutableMapping[Any, Any]], object
]


@dataclass
class RuntimeBindings:
    """Supply live objects and resolvers to JIT coverage and execution.

    A binding set may contain an Eager Engine, keys, materials, resources,
    operation handlers, Torch handlers, and application-defined resolvers.
    Backend resource bindings are separate typed resources assembled for one
    executable; this object supplies caller-owned JIT inputs before that step.
    """

    eager_engine: object | None = None
    evaluation_keys: object | None = None
    public_key: object | None = None
    materials: MutableMapping[str, object] = field(default_factory=dict)
    resources: MutableMapping[str, object] = field(default_factory=dict)
    handlers: MutableMapping[str, OperationHandler] = field(
        default_factory=dict
    )
    torch_handlers: MutableMapping[str, Callable[..., object]] = field(
        default_factory=dict
    )
    material_resolver: BindingResolver | None = None
    resource_resolver: BindingResolver | None = None
    extensions: MutableMapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("materials", "resources", "handlers", "torch_handlers"):
            if not isinstance(getattr(self, name), MutableMapping):
                raise TypeError(
                    f"RuntimeBindings {name} must be a mutable mapping"
                )
        for name in ("material_resolver", "resource_resolver"):
            resolver = getattr(self, name)
            if resolver is not None and not callable(resolver):
                raise TypeError(f"RuntimeBindings {name} must be callable")
        if not isinstance(self.extensions, MutableMapping):
            raise TypeError(
                "RuntimeBindings extensions must be a mutable mapping"
            )

    def interpreter_workspace(self) -> dict[str, object]:
        """Project live bindings into a fresh interpreter workspace.

        Mutable application state may live inside objects stored in
        ``extensions``. The projection itself is not a second state owner and
        coverage does not mutate ``RuntimeBindings``.
        """

        environment = dict(self.extensions)
        environment.update(
            {
                "materials": self.materials,
                "resources": self.resources,
                "handlers": self.handlers,
                "torch_handlers": self.torch_handlers,
            }
        )
        for name in (
            "eager_engine",
            "evaluation_keys",
            "public_key",
        ):
            value = getattr(self, name)
            if value is not None:
                environment[name] = value
        if self.material_resolver is not None:
            environment["material_resolver"] = self.material_resolver
        if self.resource_resolver is not None:
            environment["resource_resolver"] = self.resource_resolver
        return environment


__all__ = [
    "BindingResolver",
    "OperationHandler",
    "RuntimeBindings",
]
