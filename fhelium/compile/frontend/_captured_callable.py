"""Python callable state retained after frontend capture."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from ._specs import InputSpec

ReturnT = TypeVar("ReturnT")


@dataclass(frozen=True)
class CapturedCallable(Generic[ReturnT]):
    """Retain a Python callable as the pre-transform reference computation.

    The full Python signature and input specifications record which arguments
    capture specialized as static values. ``reference`` restores those values
    before invoking ``function``. ``fx_code`` is the captured PyTorch FX source
    retained for diagnostics. The transformed Program and CompileWorkspace
    belong to ``Compilation`` rather than this frontend record.
    """

    function: Callable[..., ReturnT]
    signature: inspect.Signature
    input_specs: Mapping[str, InputSpec]
    fx_code: str

    @property
    def runtime_signature(self) -> inspect.Signature:
        """Return the callable signature after specialized static inputs."""

        return self.signature.replace(
            parameters=[
                parameter
                for name, parameter in self.signature.parameters.items()
                if self.input_specs[name].role != "static"
            ]
        )

    def reference(self, *args: object, **kwargs: object) -> ReturnT:
        """Execute the captured callable with static inputs restored."""

        dynamic = self.runtime_signature.bind(*args, **kwargs)
        dynamic.apply_defaults()
        full = self.signature.bind_partial()
        for name in self.signature.parameters:
            spec = self.input_specs[name]
            full.arguments[name] = (
                spec.static_value
                if spec.role == "static"
                else dynamic.arguments[name]
            )
        return self.function(*full.args, **full.kwargs)


__all__ = ["CapturedCallable"]
