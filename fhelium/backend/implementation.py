"""Index operation implementations contributed by backend components.

Implementation classes declare their operation classes and stable name where
the implementation is defined. This registry stores those declarations,
rejects duplicate identities, and resolves caller selections without importing
or discovering implementation modules implicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr
from xdsl.ir import Operation

from fhelium.ir import (
    DEFAULT_OPERATION_SPECS,
    EXECUTION_IMPLEMENTATION_ATTRIBUTE,
    OperationEffect,
    OperationSpecRegistry,
)
from fhelium.ir.dialects import core

from .resources import BoundResource, ResourceRequirement


def _attribute_value(attribute: object) -> object:
    """Convert one builtin IR attribute while linking an invocation."""

    if isinstance(attribute, StringAttr):
        return attribute.data
    if isinstance(attribute, IntegerAttr):
        return attribute.value.data
    if isinstance(attribute, FloatAttr):
        return attribute.value.data
    if isinstance(attribute, ArrayAttr):
        return tuple(_attribute_value(item) for item in attribute)
    return attribute


@dataclass(frozen=True)
class OperationInvocation:
    """Describe a backend operation call without an SSA graph."""

    operation_type: type[Operation]
    operand_count: int
    result_count: int
    attributes: Mapping[str, object] = field(default_factory=dict)
    operand_prime_ids: tuple[tuple[int, ...] | None, ...] = ()
    operand_bases: tuple[str | None, ...] = ()
    operand_components: tuple[int | None, ...] = ()
    result_prime_ids: tuple[tuple[int, ...] | None, ...] = ()

    @classmethod
    def _from_eager(
        cls,
        operation_type: type[Operation],
        operand_count: int,
        result_count: int,
        attributes: Mapping[str, object],
        operand_bases: tuple[str | None, ...] = (),
    ) -> OperationInvocation:
        """Construct a descriptor from Engine-owned typed call metadata."""

        invocation = object.__new__(cls)
        object.__setattr__(invocation, "operation_type", operation_type)
        object.__setattr__(invocation, "operand_count", operand_count)
        object.__setattr__(invocation, "result_count", result_count)
        object.__setattr__(invocation, "attributes", attributes)
        object.__setattr__(invocation, "operand_prime_ids", ())
        object.__setattr__(
            invocation,
            "operand_bases",
            operand_bases or (None,) * operand_count,
        )
        object.__setattr__(invocation, "operand_components", ())
        object.__setattr__(invocation, "result_prime_ids", ())
        return invocation

    def __post_init__(self) -> None:
        if not isinstance(self.operation_type, type) or not issubclass(
            self.operation_type, Operation
        ):
            raise TypeError("Backend invocation requires an Operation class")
        if type(self.operand_count) is not int or self.operand_count < 0:
            raise ValueError(
                "Backend invocation operand_count must be nonnegative"
            )
        if type(self.result_count) is not int or self.result_count < 0:
            raise ValueError(
                "Backend invocation result_count must be nonnegative"
            )
        values = dict(self.attributes)
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError(
                "Backend invocation attribute names must be non-empty"
            )
        object.__setattr__(self, "attributes", MappingProxyType(values))
        for field_name in (
            "operand_prime_ids",
            "operand_bases",
            "operand_components",
        ):
            entries = getattr(self, field_name)
            if not entries:
                object.__setattr__(
                    self,
                    field_name,
                    (None,) * self.operand_count,
                )
            elif len(entries) != self.operand_count:
                raise ValueError(
                    f"Backend invocation {field_name} must match operand_count"
                )
        if not self.result_prime_ids:
            object.__setattr__(
                self,
                "result_prime_ids",
                (None,) * self.result_count,
            )
        elif len(self.result_prime_ids) != self.result_count:
            raise ValueError(
                "Backend invocation result_prime_ids must match result_count"
            )


def operation_invocation(operation: Operation) -> OperationInvocation:
    """Convert one linked IR operation to its runtime invocation descriptor."""

    def optional_field(value_type: object, name: str) -> object | None:
        state = getattr(value_type, "state", None)
        data = getattr(state, "data", None)
        if not isinstance(data, Mapping) or name not in data:
            return None
        value = _attribute_value(data[name])
        return None if value == "unknown" else value

    value_operands = tuple(
        operand
        for operand in operation.operands
        if not isinstance(operand.owner, core.ResourceRefOp)
    )
    prime_ids: list[tuple[int, ...] | None] = []
    bases: list[str | None] = []
    components: list[int | None] = []
    for operand in value_operands:
        value_type = operand.type
        represented_prime_ids = optional_field(value_type, "prime_ids")
        prime_ids.append(
            None
            if represented_prime_ids is None
            else tuple(int(value) for value in represented_prime_ids)  # type: ignore[union-attr]
        )
        represented_basis = optional_field(value_type, "basis")
        bases.append(
            None if represented_basis is None else str(represented_basis)
        )
        represented_components = optional_field(value_type, "components")
        if represented_components is None:
            represented_components = optional_field(
                value_type, "component_count"
            )
        components.append(
            None
            if represented_components is None
            else int(represented_components)  # type: ignore[arg-type]
        )
    result_prime_ids = []
    for result in operation.results:
        represented = optional_field(result.type, "prime_ids")
        result_prime_ids.append(
            None
            if represented is None
            else tuple(int(value) for value in represented)  # type: ignore[union-attr]
        )
    return OperationInvocation(
        operation_type=type(operation),
        operand_count=len(value_operands),
        result_count=len(operation.results),
        attributes={
            name: _attribute_value(attribute)
            for name, attribute in operation.attributes.items()
        },
        operand_prime_ids=tuple(prime_ids),
        operand_bases=tuple(bases),
        operand_components=tuple(components),
        result_prime_ids=tuple(result_prime_ids),
    )


@runtime_checkable
class OperationImplementation(Protocol):
    """Execute one or more registered operations on Tensor payloads."""

    @property
    def name(self) -> str: ...

    @property
    def operation_types(self) -> tuple[type[Operation], ...]: ...

    @property
    def supports_in_place(self) -> bool: ...

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]: ...

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        /,
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]: ...


RegionCallable = Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]


@runtime_checkable
class RegionOperationImplementation(OperationImplementation, Protocol):
    """Execute an operation whose regions contain callable Tensor programs."""

    def execute_regions(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        regions: tuple[RegionCallable, ...],
        /,
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]: ...


def requested_implementation(
    operation: Operation,
    caller_selection: str | None = None,
) -> str | None:
    """Return the implementation recorded in IR or selected by the caller.

    An assignment written by a Compile pass is a hard constraint. A caller may
    select an unassigned operation, but it may not contradict the Program.
    """

    attribute = operation.attributes.get(EXECUTION_IMPLEMENTATION_ATTRIBUTE)
    recorded = attribute.data if isinstance(attribute, StringAttr) else None
    if recorded is not None and caller_selection not in {None, recorded}:
        raise ValueError(
            f"Operation {operation.name!r} records implementation "
            f"{recorded!r}, which conflicts with caller selection "
            f"{caller_selection!r}"
        )
    return recorded if recorded is not None else caller_selection


class ImplementationRegistry[ImplementationT: OperationImplementation]:
    """Store named implementations by operation class and identity.

    The registry has no module-discovery or import-time mutation behavior.
    Backend components construct implementation objects beside their kernels,
    and an assembly function passes those objects to this registry.
    """

    def __init__(
        self,
        implementations: Sequence[ImplementationT] = (),
        *,
        category: str,
    ) -> None:
        if not isinstance(category, str) or not category:
            raise ValueError(
                "Implementation registry category must be non-empty"
            )
        entries: dict[tuple[type[Operation], str], ImplementationT] = {}
        available: dict[type[Operation], list[str]] = {}
        operation_types: list[type[Operation]] = []
        seen_operation_types: set[type[Operation]] = set()
        for implementation in implementations:
            name = implementation.name
            supported = implementation.operation_types
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"{category} implementation name must be non-empty"
                )
            if not isinstance(supported, tuple) or not supported:
                raise ValueError(
                    f"{category} implementation must declare operation types"
                )
            if any(
                not isinstance(operation_type, type)
                or not issubclass(operation_type, Operation)
                for operation_type in supported
            ):
                raise TypeError(
                    f"{category} operation types must be Operation classes"
                )
            for operation_type in supported:
                identity = (operation_type, name)
                if identity in entries:
                    raise ValueError(
                        f"{category} implementation is registered twice: "
                        f"{operation_type.name}/{name}"
                    )
                entries[identity] = implementation
                available.setdefault(operation_type, []).append(name)
                if operation_type not in seen_operation_types:
                    seen_operation_types.add(operation_type)
                    operation_types.append(operation_type)
        self._category = category
        self._implementations = tuple(implementations)
        self._entries = MappingProxyType(entries)
        self._available = MappingProxyType(
            {identity: tuple(names) for identity, names in available.items()}
        )
        self._operation_types = tuple(operation_types)

    @property
    def operation_types(self) -> tuple[type[Operation], ...]:
        """Return operation classes in first declaration order."""

        return self._operation_types

    @property
    def implementations(self) -> tuple[ImplementationT, ...]:
        """Return implementation contributions in construction order."""

        return self._implementations

    def available(self, operation_type: type[Operation]) -> tuple[str, ...]:
        """Return implementation names in declaration order."""

        return self._available.get(operation_type, ())

    def supports(
        self,
        operation_type: type[Operation],
        *,
        name: str | None = None,
    ) -> bool:
        """Return whether at least one matching implementation is registered."""

        if name is None:
            return bool(self.available(operation_type))
        return (operation_type, name) in self._entries

    def resolve(
        self,
        operation_type: type[Operation],
        *,
        requested: str | None,
    ) -> ImplementationT:
        """Resolve one named implementation, refusing ambiguous defaults."""

        choices = self.available(operation_type)
        selected = requested
        if selected is None:
            if len(choices) != 1:
                raise ValueError(
                    f"{self._category} operation {operation_type.name!r} has "
                    f"{len(choices)} available implementations; "
                    f"selection is required: {choices}"
                )
            selected = choices[0]
        try:
            return self._entries[(operation_type, selected)]
        except KeyError:
            raise KeyError(
                f"{self._category} operation {operation_type.name!r} has no "
                f"implementation {selected!r}; available={choices}"
            ) from None


class OperationImplementationRegistry:
    """Resolve implementations by operation class and identity."""

    def __init__(
        self,
        implementations: Sequence[OperationImplementation] = (),
        *,
        operation_specs: OperationSpecRegistry = DEFAULT_OPERATION_SPECS,
    ) -> None:
        if not isinstance(operation_specs, OperationSpecRegistry):
            raise TypeError("operation_specs must be OperationSpecRegistry")
        for implementation in implementations:
            if not isinstance(implementation, OperationImplementation):
                raise TypeError(
                    "Operation implementation does not satisfy its protocol"
                )
            for operation_type in implementation.operation_types:
                operation_specs.require(operation_type.name)
        self._implementations = ImplementationRegistry(
            tuple(implementations), category="Operation"
        )
        self._operation_specs = operation_specs

    @property
    def operation_types(self) -> tuple[type[Operation], ...]:
        return self._implementations.operation_types

    @property
    def implementations(self) -> tuple[OperationImplementation, ...]:
        """Return implementation contributions in construction order."""

        return self._implementations.implementations

    def with_implementations(
        self,
        implementations: Sequence[OperationImplementation],
    ) -> OperationImplementationRegistry:
        """Return this registry extended by caller-supplied contributions."""

        return OperationImplementationRegistry(
            (*self.implementations, *implementations),
            operation_specs=self._operation_specs,
        )

    def available(self, operation_type: type[Operation]) -> tuple[str, ...]:
        return self._implementations.available(operation_type)

    def supports(
        self,
        operation: Operation,
        *,
        implementation: str | None = None,
    ) -> bool:
        return self._implementations.supports(
            type(operation), name=implementation
        )

    def resolve(
        self,
        operation: Operation,
        *,
        requested: str | None,
        in_place: bool = False,
    ) -> OperationImplementation:
        return self.resolve_type(
            type(operation),
            requested=requested,
            in_place=in_place,
        )

    def resolve_type(
        self,
        operation_type: type[Operation],
        *,
        requested: str | None,
        in_place: bool = False,
    ) -> OperationImplementation:
        implementation = self._implementations.resolve(
            operation_type, requested=requested
        )
        if in_place and not implementation.supports_in_place:
            raise ValueError(
                f"Implementation {implementation.name!r} does not support "
                "in-place execution"
            )
        return implementation

    def effect(self, operation_type: type[Operation]) -> OperationEffect:
        """Return the registered effect of one operation class."""

        return self._operation_specs.require(operation_type.name).effect


__all__ = [
    "ImplementationRegistry",
    "OperationImplementation",
    "OperationImplementationRegistry",
    "OperationInvocation",
    "RegionCallable",
    "RegionOperationImplementation",
    "operation_invocation",
    "requested_implementation",
]
