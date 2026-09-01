"""Define the public records and registry for shared CKKS lowering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.ir import Operation, SSAValue

from fhelium.config import CkksConfig
from ....ir._operation_specs import (
    DEFAULT_OPERATION_SPECS,
    OperationSpecRegistry,
)


@dataclass(frozen=True)
class LoweredCkksOperation:
    """Hold replacement operations and the logical result of one CKKS op."""

    operations: tuple[Operation, ...]
    result: SSAValue


CkksLowering = Callable[[Operation, CkksConfig], LoweredCkksOperation]


@dataclass(frozen=True)
class CkksLoweringDefinition:
    """Declare one named CKKS lowering beside its implementation."""

    name: str
    operation_type: type[Operation]
    lower: CkksLowering
    is_default: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CKKS lowering name must be non-empty")
        if not isinstance(self.operation_type, type) or not issubclass(
            self.operation_type, Operation
        ):
            raise TypeError(
                "CKKS lowering operation_type must be an Operation class"
            )
        if not callable(self.lower):
            raise TypeError("CKKS lowering implementation must be callable")
        if type(self.is_default) is not bool:
            raise TypeError("CKKS lowering is_default must be bool")


class CkksLoweringRegistry:
    """Resolve implementation-local lowerings by operation class and name."""

    def __init__(
        self,
        definitions: Sequence[CkksLoweringDefinition] = (),
        *,
        operation_specs: OperationSpecRegistry = DEFAULT_OPERATION_SPECS,
    ) -> None:
        if not isinstance(operation_specs, OperationSpecRegistry):
            raise TypeError("operation_specs must be OperationSpecRegistry")
        entries: dict[tuple[type[Operation], str], CkksLoweringDefinition] = {}
        available: dict[type[Operation], list[str]] = {}
        defaults: dict[type[Operation], str] = {}
        for definition in definitions:
            if not isinstance(definition, CkksLoweringDefinition):
                raise TypeError(
                    "CKKS lowering registry entries must be "
                    "CkksLoweringDefinition"
                )
            operation_specs.require(definition.operation_type.name)
            key = (definition.operation_type, definition.name)
            if key in entries:
                raise ValueError(
                    "CKKS lowering is registered twice: "
                    f"{definition.operation_type.name!r}/{definition.name!r}"
                )
            if definition.is_default:
                previous = defaults.get(definition.operation_type)
                if previous is not None:
                    raise ValueError(
                        f"CKKS operation {definition.operation_type.name!r} "
                        "has multiple default lowerings: "
                        f"{previous!r}, {definition.name!r}"
                    )
                defaults[definition.operation_type] = definition.name
            entries[key] = definition
            available.setdefault(definition.operation_type, []).append(
                definition.name
            )
        self._entries = MappingProxyType(entries)
        self._available = MappingProxyType(
            {
                operation_type: tuple(names)
                for operation_type, names in available.items()
            }
        )
        self._defaults = MappingProxyType(defaults)
        self._operation_specs = operation_specs

    def supports(self, operation: Operation) -> bool:
        """Return whether one operation has a registered CKKS lowering."""

        return bool(self.available(type(operation)))

    @property
    def definitions(self) -> tuple[CkksLoweringDefinition, ...]:
        """Return lowering declarations in registry order."""

        return tuple(self._entries.values())

    @property
    def operation_types(self) -> tuple[type[Operation], ...]:
        """Return CKKS operation classes with shared lowering definitions."""

        return tuple(self._available)

    def available(
        self,
        operation_type: type[Operation],
    ) -> tuple[str, ...]:
        """Return registered lowering names for one operation class."""

        return self._available.get(operation_type, ())

    def resolve(
        self,
        operation_type: type[Operation],
        requested: str | None = None,
    ) -> CkksLoweringDefinition:
        """Resolve a requested, default, or unambiguous lowering."""

        choices = self.available(operation_type)
        if requested is None:
            requested = self._defaults.get(operation_type)
        if requested is None:
            if len(choices) != 1:
                raise ValueError(
                    f"CKKS operation {operation_type.name!r} has "
                    f"{len(choices)} registered lowerings; a lowering name "
                    f"is required: {choices}"
                )
            requested = choices[0]
        try:
            return self._entries[(operation_type, requested)]
        except KeyError:
            raise KeyError(
                f"CKKS operation {operation_type.name!r} has no lowering "
                f"{requested!r}; available={choices}"
            ) from None

    def lower(
        self,
        operation: Operation,
        config: CkksConfig,
        *,
        requested: str | None = None,
    ) -> LoweredCkksOperation:
        """Apply one selected lowering without choosing an implementation."""

        definition = self.resolve(type(operation), requested)
        diagnostics = self._operation_specs.require(operation.name).diagnostics(
            operation
        )
        if diagnostics:
            raise ValueError(
                f"CKKS lowering rejected {operation.name!r}: "
                f"{'; '.join(diagnostics)}"
            )
        return definition.lower(operation, config)
