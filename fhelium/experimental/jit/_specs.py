"""Semantic input-role declarations for JIT programs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ._ir import ValueRole

SlotExtent = int | Literal["full"]
BatchMode = Literal["none", "any"]
StaticValue = bool | int | float | complex | str | None


@dataclass(frozen=True)
class InputSpec:
    """Declare one function input's role in the PyTorch-to-FHE interface.

    ``encrypted`` declares a logical slot extent, batch policy, level, and
    scale for a runtime Tensor or core ``Ciphertext``. ``message`` declares
    public Python/PyTorch data passed through directly until an
    plaintext-preparation operation consumes it. ``plaintext`` declares a
    caller-owned core ``Plaintext`` whose state is validated at its encrypted
    consumer. ``static`` declares an immutable scalar specialized during
    capture and removed from the Program's runtime argument list.
    """

    role: ValueRole
    level: int = 0
    scale: float | None = None
    slots: SlotExtent = "full"
    batch_mode: BatchMode = "none"
    static_value: StaticValue = None

    def __post_init__(self) -> None:
        if self.role not in ("encrypted", "message", "plaintext", "static"):
            raise ValueError(f"Unsupported JIT input role: {self.role!r}")

        if self.role != "encrypted" and (
            self.level != 0
            or self.scale is not None
            or self.slots != "full"
            or self.batch_mode != "none"
        ):
            raise ValueError(
                f"{self.role} inputs do not declare encrypted slot/state "
                "requirements"
            )

        if self.role == "static":
            if not isinstance(
                self.static_value,
                (bool, int, float, complex, str, type(None)),
            ):
                raise TypeError(
                    "JIT static inputs are restricted to immutable scalar "
                    "values: bool, int, float, complex, str, or None"
                )
            if isinstance(self.static_value, float) and not math.isfinite(
                self.static_value
            ):
                raise ValueError("JIT static real values must be finite")
            if isinstance(self.static_value, complex) and not (
                math.isfinite(self.static_value.real)
                and math.isfinite(self.static_value.imag)
            ):
                raise ValueError("JIT static complex values must be finite")
            return

        if self.static_value is not None:
            raise ValueError(
                f"{self.role} inputs do not declare a static value"
            )
        if self.role in ("message", "plaintext"):
            return

        if isinstance(self.level, bool) or not isinstance(self.level, int):
            raise TypeError("JIT encrypted input level must be an integer")
        if self.level < 0:
            raise ValueError("JIT encrypted input level must be nonnegative")
        if self.scale is not None:
            if isinstance(self.scale, bool) or not isinstance(
                self.scale, (int, float)
            ):
                raise TypeError("JIT encrypted input scale must be real")
            if not math.isfinite(float(self.scale)) or float(self.scale) <= 0:
                raise ValueError(
                    "JIT encrypted input scale must be positive and finite"
                )
        if self.slots != "full":
            if isinstance(self.slots, bool) or not isinstance(self.slots, int):
                raise TypeError(
                    "JIT encrypted input slots must be 'full' or an integer"
                )
            if self.slots <= 0:
                raise ValueError("JIT encrypted input slots must be positive")
        if self.batch_mode not in ("none", "any"):
            raise ValueError(
                "JIT encrypted input batch mode must be 'none' or 'any'"
            )


def encrypted(
    *,
    level: int = 0,
    scale: float | None = None,
    slots: SlotExtent = "full",
    batch_mode: BatchMode = "none",
) -> InputSpec:
    """Declare a secret slot tensor bound from Tensor or ``Ciphertext``.

    ``level`` and a non-``None`` ``scale`` define the exact runtime CKKS input
    state; ``scale=None`` selects the bound engine's default scale. ``slots``
    specifies either the engine's full capacity or an exact final-axis extent.
    ``batch_mode='none'`` requires a one-dimensional Tensor and an unbatched
    Ciphertext; ``'any'`` permits leading batch axes.

    Runtime Tensor inputs are encrypted online and therefore require
    ``workspace['engine']`` and ``workspace['public_key']``. Runtime Ciphertext
    inputs must match the engine's context, device, dtype, ring dimension,
    level, scale, and batch policy. A fixed non-full logical slot extent cannot
    be verified from an opaque Ciphertext, so this input form accepts Tensor
    input only. The retained Python reference callable consumes its ordinary
    public Tensor argument.
    """

    return InputSpec(
        "encrypted",
        level=level,
        scale=scale,
        slots=slots,
        batch_mode=batch_mode,
    )


def message() -> InputSpec:
    """Declare public Tensor/scalar data processed by ordinary PyTorch.

    Message-only subgraphs execute as public calls. A mixed encrypted operation
    introduces operation-specific encoding and plaintext preparation;
    that preparation derives the required CKKS representation from its ciphertext
    consumer.
    """

    return InputSpec("message")


def plaintext() -> InputSpec:
    """Declare a caller-owned FHElium Plaintext with exact runtime state.

    JIT execution requires a core ``Plaintext`` entry value and validates its
    level, scale, basis, domain, residues, prime identities, and batch
    requirements at its exact encrypted consumer. The retained Python
    reference callable consumes the corresponding public Tensor or scalar
    shadow supplied by the caller.
    """

    return InputSpec("plaintext")


def static(value: StaticValue) -> InputSpec:
    """Specialize an immutable scalar Python input during capture.

    A finite ``bool``, ``int``, ``float``, ``complex``, ``str``, or ``None``
    value participates in Python control and graph construction. Capture stores
    its serialized value in Program metadata and removes the parameter from
    ``CaptureResult.runtime_signature`` and Program execution inputs.
    """

    return InputSpec("static", static_value=value)


__all__ = [
    "BatchMode",
    "InputSpec",
    "SlotExtent",
    "StaticValue",
    "encrypted",
    "message",
    "plaintext",
    "static",
]
