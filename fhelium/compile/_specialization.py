"""Describe input conditions used to reuse a prepared computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import torch

from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    Plaintext,
    KeySwitchKey,
    RotationKey,
)

from .frontend._eager_values import value_state


@dataclass(frozen=True)
class ArgumentSignature:
    """Record one argument's type, layout, and represented value state.

    Tensor contents and storage addresses are excluded. Immutable Python scalar
    values are included because capture may use them to construct the Program.
    The initial matcher conservatively requires equality of all recorded fields.
    """

    name: str
    kind: str
    properties: tuple[tuple[str, object], ...]

    def get(self, name: str, default: object = None) -> object:
        """Read one recorded property for a caller-selected pass pipeline."""

        return next(
            (value for key, value in self.properties if key == name), default
        )


@dataclass(frozen=True)
class CallSignature:
    """Collect the conditions for one flat callable or Program invocation."""

    arguments: tuple[ArgumentSignature, ...]

    def argument(self, name: str) -> ArgumentSignature:
        """Return a named argument description."""

        return next(
            argument for argument in self.arguments if argument.name == name
        )


class SpecializationMiss(RuntimeError):
    """Report that execution requires a specialization not prepared yet."""


def _tensor_properties(value: torch.Tensor) -> tuple[tuple[str, object], ...]:
    return (
        ("shape", tuple(value.shape)),
        ("strides", tuple(value.stride())),
        ("dtype", str(value.dtype)),
        ("device", str(value.device)),
        ("requires_grad", value.requires_grad),
    )


def _scalar_key(value: object) -> object:
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, complex):
        return (value.real.hex(), value.imag.hex())
    return value


def describe_argument(name: str, value: object) -> ArgumentSignature:
    """Read metadata without inspecting payload contents or copying storage."""

    if isinstance(value, (Ciphertext, Plaintext, CompressedPlaintext)):
        if isinstance(value, Plaintext):
            payload = (
                value.message if value.representation == "slots" else value.data
            )
        else:
            payload = value.data
        properties = tuple(value_state(value).items())
        if payload is not None:
            properties += _tensor_properties(payload)
        if (
            isinstance(value, CompressedPlaintext)
            and value.implicit_data is not None
        ):
            properties += tuple(
                ("implicit_" + key, item)
                for key, item in _tensor_properties(value.implicit_data)
            )
        return ArgumentSignature(name, type(value).__name__, properties)
    if isinstance(value, KeySwitchKey):
        properties = _tensor_properties(value.data) + (
            ("prime_ids", value.prime_ids),
            ("polynomial_domain", value.polynomial_domain),
            ("basis", value.modulus_basis),
            ("residue_representation", value.residue_representation),
        )
        if isinstance(value, RotationKey):
            properties += (("rotation_step", value.rotation_step),)
        return ArgumentSignature(name, type(value).__name__, properties)
    if isinstance(value, torch.Tensor):
        return ArgumentSignature(name, "Tensor", _tensor_properties(value))
    if value is None or type(value) in (bool, int, float, complex, str):
        return ArgumentSignature(
            name, type(value).__name__, (("value", _scalar_key(value)),)
        )
    raise TypeError(
        f"Compiled argument {name!r} has unsupported type {type(value).__name__}; "
        "use flat Tensor/CKKS arguments or immutable scalar parameters, and "
        "supply execution resources through Backend"
    )


def describe_call(arguments: Mapping[str, object]) -> CallSignature:
    return CallSignature(
        tuple(
            describe_argument(name, value) for name, value in arguments.items()
        )
    )


__all__ = ["ArgumentSignature", "CallSignature", "SpecializationMiss"]


def prepare_match(
    signature: CallSignature, argument_types: tuple[type, ...]
) -> Callable[[tuple[object, ...]], bool]:
    """Compile direct comparisons for one prepared invocation's conditions.

    The signature remains inspectable; matching reads current metadata without
    reconstructing descriptors or hashing nested property tuples.
    """
    namespace: dict[str, object] = {}
    terms: list[str] = []
    fields = {"basis": "modulus_basis", "components": "component_count"}
    tensor_fields = {"shape", "strides", "dtype", "device", "requires_grad"}
    for index, (argument, expected_type) in enumerate(
        zip(signature.arguments, argument_types, strict=True)
    ):
        value = f"values[{index}]"
        namespace[f"type{index}"] = expected_type
        terms.append(f"type({value}) is type{index}")
        properties = dict(argument.properties)
        if argument.kind == "Tensor":
            payload = value
        elif issubclass(expected_type, Plaintext):
            payload = value + (
                ".message"
                if properties.get("representation") == "slots"
                else ".data"
            )
        elif issubclass(
            expected_type, (Ciphertext, CompressedPlaintext, KeySwitchKey)
        ):
            payload = value + ".data"
        else:
            payload = None
        for ordinal, (field, expected) in enumerate(argument.properties):
            symbol = f"expected{index}_{ordinal}"
            if field == "value":
                expression = (
                    f"{value}.hex()"
                    if expected_type is float
                    else f"({value}.real.hex(), {value}.imag.hex())"
                    if expected_type is complex
                    else value
                )
            elif field in tensor_fields or field.startswith("implicit_"):
                active_payload = (
                    value + ".implicit_data"
                    if field.startswith("implicit_")
                    else payload
                )
                field = field.removeprefix("implicit_")
                if active_payload is None:
                    continue
                expression = (
                    f"{active_payload}.stride()"
                    if field == "strides"
                    else f"{active_payload}.{field}"
                )
                if field == "dtype":
                    expected = getattr(
                        torch, str(expected).removeprefix("torch.")
                    )
                elif field == "device":
                    expected = torch.device(str(expected))
            elif field == "ring_dimension" and issubclass(
                expected_type, Plaintext
            ):
                expression = f"{payload}.size(-1)"
            else:
                expression = f"{value}.{fields.get(field, field)}"
            namespace[symbol] = expected
            terms.append(f"{expression} == {symbol}")
    source = (
        "def matches(values):\n    return "
        + (" and ".join(f"({term})" for term in terms) or "True")
        + "\n"
    )
    exec(compile(source, "<fhelium-specialization-match>", "exec"), namespace)
    return cast(Callable[[tuple[object, ...]], bool], namespace["matches"])
