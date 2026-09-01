"""Immutable records for mixed-provider coverage and execution plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .._contracts import CoverageDiagnostic


@dataclass(frozen=True)
class OperationSupport:
    """Record one operation's semantic family and selected implementation."""

    operation_id: str
    operation: str
    family: str | None
    provider: str | None
    implementation: str | None
    supported: bool
    diagnostics: tuple[CoverageDiagnostic, ...] = ()


@dataclass(frozen=True)
class PlanValue:
    """Describe one SSA value crossing a Program or region interface."""

    value_id: str
    role: str | None
    abi: str
    producer: str | None
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class PlanRegion:
    """Describe one selected contiguous region assigned to one provider."""

    region_id: str
    provider: str
    implementation: str
    operation_ids: tuple[str, ...]
    input_values: tuple[str, ...]
    output_values: tuple[str, ...]
    supported: bool = True
    diagnostics: tuple[CoverageDiagnostic, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable provider partition and Python-object SSA connection plan."""

    entry: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    values: tuple[PlanValue, ...]
    operations: tuple[OperationSupport, ...]
    regions: tuple[PlanRegion, ...]

    def manifest(self) -> Mapping[str, object]:
        """Return a JSON-compatible description of assignments and edges."""

        return MappingProxyType(
            {
                "entry": self.entry,
                "inputs": list(self.inputs),
                "outputs": list(self.outputs),
                "values": [
                    {
                        "id": value.value_id,
                        "role": value.role,
                        "abi": value.abi,
                        "producer": value.producer,
                        "consumers": list(value.consumers),
                    }
                    for value in self.values
                ],
                "operations": [
                    {
                        "id": item.operation_id,
                        "operation": item.operation,
                        "family": item.family,
                        "provider": item.provider,
                        "implementation": item.implementation,
                        "supported": item.supported,
                        "diagnostics": [
                            diagnostic.code for diagnostic in item.diagnostics
                        ],
                    }
                    for item in self.operations
                ],
                "regions": [
                    {
                        "id": region.region_id,
                        "provider": region.provider,
                        "implementation": region.implementation,
                        "operations": list(region.operation_ids),
                        "inputs": list(region.input_values),
                        "outputs": list(region.output_values),
                        "supported": region.supported,
                        "diagnostics": [
                            diagnostic.code for diagnostic in region.diagnostics
                        ],
                    }
                    for region in self.regions
                ],
            }
        )


__all__ = ["ExecutionPlan", "OperationSupport", "PlanRegion", "PlanValue"]
