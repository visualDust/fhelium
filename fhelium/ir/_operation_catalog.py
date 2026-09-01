"""Operation semantic definitions assembled by FHElium dialects."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Operation, SSAValue

ValueRole = Literal["encrypted", "message", "plaintext", "static"]

OperationEffect = Literal["pure", "rng-write", "mutation", "opaque"]
OperationValidator: TypeAlias = Callable[[Operation], tuple[str, ...]]

EXECUTION_IMPLEMENTATION_ATTRIBUTE = "fhelium.execution.implementation"


def _operation_name(operation: Operation) -> str:
    op_name = getattr(operation, "op_name", None)
    return op_name.data if isinstance(op_name, StringAttr) else operation.name


def _value_role(value_or_type: SSAValue | Attribute) -> ValueRole | None:
    attribute = (
        value_or_type.type
        if isinstance(value_or_type, SSAValue)
        else value_or_type
    )
    state = getattr(attribute, "state", None)
    data = getattr(state, "data", {})
    represented_role = data.get("role")
    if represented_role is not None:
        if not isinstance(represented_role, StringAttr):
            raise ValueError(
                f"{attribute.name} state 'role' must be a StringAttr"
            )
        if represented_role.data not in {
            "encrypted",
            "message",
            "plaintext",
            "static",
        }:
            raise ValueError(
                f"{attribute.name} state has unsupported role "
                f"{represented_role.data!r}"
            )
        return cast(ValueRole, represented_role.data)
    role = getattr(type(attribute), "ROLE", None)
    if role in {"encrypted", "message", "plaintext", "static"}:
        return cast(ValueRole, role)
    return None


@dataclass(frozen=True)
class OperationSpec:
    """Describe one operation's family, signature, roles, and effects."""

    name: str
    family: str
    operand_arity: int | None
    result_arity: int | None
    operand_roles: tuple[str | None, ...] = ()
    result_roles: tuple[str | None, ...] = ()
    effect: OperationEffect = "pure"
    validator: OperationValidator | None = None
    operation_type: type[Operation] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.family:
            raise ValueError("OperationSpec name and family must be non-empty")
        if self.operand_arity is not None and self.operand_arity < 0:
            raise ValueError("OperationSpec operand arity must be nonnegative")
        if self.result_arity is not None and self.result_arity < 0:
            raise ValueError("OperationSpec result arity must be nonnegative")
        if self.operand_roles and self.operand_arity != len(self.operand_roles):
            raise ValueError(
                "OperationSpec operand roles differ from its arity"
            )
        if self.result_roles and self.result_arity != len(self.result_roles):
            raise ValueError("OperationSpec result roles differ from its arity")
        if self.effect not in {"pure", "rng-write", "mutation", "opaque"}:
            raise ValueError("OperationSpec effect is unsupported")
        if self.validator is not None and not callable(self.validator):
            raise TypeError("OperationSpec validator must be callable")
        if (
            self.operation_type is not None
            and self.operation_type.name != self.name
        ):
            raise ValueError(
                "OperationSpec registered operation type and name differ"
            )

    def diagnostics(self, operation: Operation) -> tuple[str, ...]:
        """Return local signature and role mismatches for ``operation``."""

        diagnostics: list[str] = []
        actual_name = _operation_name(operation)
        if actual_name != self.name:
            diagnostics.append(
                f"operation name {actual_name!r} differs from {self.name!r}"
            )
        if self.operation_type is not None and not isinstance(
            operation, self.operation_type
        ):
            diagnostics.append(
                "operation is not registered class "
                f"{self.operation_type.__name__}"
            )
        if (
            self.operand_arity is not None
            and len(operation.operands) != self.operand_arity
        ):
            diagnostics.append(
                f"expected {self.operand_arity} operands, got "
                f"{len(operation.operands)}"
            )
        if (
            self.result_arity is not None
            and len(operation.results) != self.result_arity
        ):
            diagnostics.append(
                f"expected {self.result_arity} results, got "
                f"{len(operation.results)}"
            )
        if self.operand_roles and len(operation.operands) == len(
            self.operand_roles
        ):
            actual = tuple(_value_role(value) for value in operation.operands)
            for index, (expected, observed) in enumerate(
                zip(self.operand_roles, actual, strict=True)
            ):
                if expected is not None and observed != expected:
                    diagnostics.append(
                        f"operand {index} requires role {expected!r}, got "
                        f"{observed!r}"
                    )
        if self.result_roles and len(operation.results) == len(
            self.result_roles
        ):
            actual = tuple(_value_role(value) for value in operation.results)
            for index, (expected, observed) in enumerate(
                zip(self.result_roles, actual, strict=True)
            ):
                if expected is not None and observed != expected:
                    diagnostics.append(
                        f"result {index} requires role {expected!r}, got "
                        f"{observed!r}"
                    )
        implementation = operation.attributes.get(
            EXECUTION_IMPLEMENTATION_ATTRIBUTE
        )
        if implementation is not None and (
            not isinstance(implementation, StringAttr)
            or not implementation.data
        ):
            diagnostics.append(
                f"{EXECUTION_IMPLEMENTATION_ATTRIBUTE!r} must be a nonempty "
                "string"
            )
        if self.validator is not None:
            diagnostics.extend(self.validator(operation))
        return tuple(diagnostics)


class OperationSpecRegistry:
    """Immutable lookup of uniquely named semantic operation specifications."""

    def __init__(self, specifications: Iterable[OperationSpec] = ()) -> None:
        entries: dict[str, OperationSpec] = {}
        for specification in specifications:
            if not isinstance(specification, OperationSpec):
                raise TypeError(
                    "operation specifications must be OperationSpec"
                )
            if specification.name in entries:
                raise ValueError(
                    f"OperationSpec {specification.name!r} is already registered"
                )
            entries[specification.name] = specification
        self._entries: Mapping[str, OperationSpec] = MappingProxyType(entries)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered operation names in declaration order."""

        return tuple(self._entries)

    def get(self, name: str) -> OperationSpec | None:
        """Return a specification, or ``None`` for permissive unknown IR."""

        return self._entries.get(name)

    def require(self, name: str) -> OperationSpec:
        """Return a specification or raise a lookup error."""

        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"No OperationSpec is registered for {name!r}"
            ) from None


def operation_spec(
    name: str,
    family: str,
    operands: tuple[str | None, ...],
    results: tuple[str | None, ...] = ("encrypted",),
    *,
    effect: OperationEffect = "pure",
    validator: OperationValidator | None = None,
) -> OperationSpec:
    """Construct a role-bearing specification for an open operation."""

    return OperationSpec(
        name,
        family,
        len(operands),
        len(results),
        operands,
        results,
        effect,
        validator,
    )


def registered_operation_spec(
    operation_type: type[Operation],
    family: str,
    *,
    effect: OperationEffect = "pure",
    validator: OperationValidator | None = None,
) -> OperationSpec:
    """Construct a specification for one registered xDSL operation class."""

    return OperationSpec(
        operation_type.name,
        family,
        None,
        None,
        effect=effect,
        validator=validator,
        operation_type=operation_type,
    )


def unsupported_attributes(
    operation: Operation,
    allowed: Iterable[str],
) -> tuple[str, ...]:
    """Return attributes outside common metadata and ``allowed`` names."""

    common = {
        "op_name__",
        EXECUTION_IMPLEMENTATION_ATTRIBUTE,
    }
    return tuple(sorted(set(operation.attributes) - common - set(allowed)))


def flat_operation(operation: Operation) -> tuple[str, ...]:
    """Require an operation without properties, regions, or successors."""

    diagnostics: list[str] = []
    if operation.properties:
        diagnostics.append("properties are unsupported")
    if operation.regions:
        diagnostics.append("regions are unsupported")
    if operation.successors:
        diagnostics.append("successors are unsupported")
    return tuple(diagnostics)


def flat_without_attributes(operation: Operation) -> tuple[str, ...]:
    """Require a flat operation without operation-specific attributes."""

    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(
        operation, ("fhelium.frontend.target",)
    )
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    return tuple(diagnostics)


def flat_with_attributes(*names: str) -> OperationValidator:
    """Build a validator for a flat operation with named attributes."""

    def validate(operation: Operation) -> tuple[str, ...]:
        diagnostics = list(flat_operation(operation))
        unsupported = unsupported_attributes(operation, names)
        if unsupported:
            diagnostics.append(f"unsupported attributes {list(unsupported)}")
        return tuple(diagnostics)

    return validate


def required_string_attribute(
    operation: Operation,
    name: str,
) -> str | None:
    """Return one required nonempty string attribute when present."""

    value = operation.attributes.get(name)
    return value.data if isinstance(value, StringAttr) and value.data else None


def literal_diagnostics(value: object) -> tuple[str, ...]:
    """Validate one JSON-compatible captured literal descriptor."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return ()
    if not isinstance(value, dict):
        return (f"unsupported literal {value!r}",)
    kind = value.get("kind")
    if (
        kind == "complex"
        and isinstance(value.get("real"), (int, float))
        and isinstance(value.get("imag"), (int, float))
    ):
        return ()
    if kind == "ellipsis":
        return ()
    if kind in {"torch.dtype", "torch.device", "torch.layout"} and isinstance(
        value.get("value"), str
    ):
        return ()
    return (f"unsupported literal descriptor {value!r}",)


def decode_json_attribute(
    operation: Operation,
    name: str,
) -> tuple[object | None, str | None]:
    """Decode one required JSON string attribute."""

    encoded = required_string_attribute(operation, name)
    if encoded is None:
        return None, f"requires string attribute {name!r}"
    try:
        return json.loads(encoded), None
    except json.JSONDecodeError:
        return None, f"attribute {name!r} contains malformed JSON"


def argument_descriptor_diagnostics(
    value: object,
    operand_count: int,
) -> tuple[str, ...]:
    """Validate captured nested-call argument metadata."""

    if not isinstance(value, dict):
        return literal_diagnostics(value)
    kind = value.get("kind")
    if kind == "ssa":
        index = value.get("operand")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= operand_count
        ):
            return ("SSA argument descriptor has an invalid operand index",)
        return ()
    if kind == "literal":
        return literal_diagnostics(value.get("value"))
    if kind in {"tuple", "list"}:
        items = value.get("items")
        if not isinstance(items, list):
            return (f"{kind} argument descriptor lacks an items list",)
        return tuple(
            diagnostic
            for item in items
            for diagnostic in argument_descriptor_diagnostics(
                item, operand_count
            )
        )
    if kind == "mapping":
        entries = value.get("entries")
        if not isinstance(entries, list):
            return ("mapping argument descriptor lacks an entries list",)
        diagnostics: list[str] = []
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                diagnostics.append(
                    "mapping argument descriptor has a malformed entry"
                )
                continue
            diagnostics.extend(
                argument_descriptor_diagnostics(entry[0], operand_count)
            )
            diagnostics.extend(
                argument_descriptor_diagnostics(entry[1], operand_count)
            )
        return tuple(diagnostics)
    if kind == "slice":
        return tuple(
            diagnostic
            for field in ("start", "stop", "step")
            for diagnostic in argument_descriptor_diagnostics(
                value.get(field), operand_count
            )
        )
    return (f"unknown argument descriptor kind {kind!r}",)


__all__ = [
    "OperationEffect",
    "OperationSpec",
    "OperationSpecRegistry",
    "OperationValidator",
    "argument_descriptor_diagnostics",
    "decode_json_attribute",
    "flat_operation",
    "flat_with_attributes",
    "flat_without_attributes",
    "literal_diagnostics",
    "operation_spec",
    "registered_operation_spec",
    "required_string_attribute",
    "unsupported_attributes",
]
