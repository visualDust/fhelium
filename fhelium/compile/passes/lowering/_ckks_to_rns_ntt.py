"""Adapt neutral CKKS-to-RNS/NTT lowering to the compile pass protocol."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from fhelium.config import CkksConfig
from fhelium.ir import (
    Program,
)
from .._operation_transforms import program_operations
from ._core import CkksLoweringRegistry
from ._driver import DEFAULT_CKKS_LOWERINGS, lower_ckks_program


@dataclass(frozen=True)
class LowerCkksToRnsNttPass:
    """Apply caller-selected CKKS-to-RNS/NTT lowering definitions."""

    selections: Mapping[str, str] = field(default_factory=dict)
    preserve: frozenset[str] = frozenset()
    registry: CkksLoweringRegistry = DEFAULT_CKKS_LOWERINGS
    name: str = field(default="lower-ckks-to-rns-ntt", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.registry, CkksLoweringRegistry):
            raise TypeError("registry must be a CkksLoweringRegistry")
        selections = dict(self.selections)
        if any(
            not isinstance(operation_name, str)
            or not operation_name
            or not isinstance(lowering_name, str)
            or not lowering_name
            for operation_name, lowering_name in selections.items()
        ):
            raise ValueError("CKKS lowering selections require non-empty names")
        preserve = frozenset(self.preserve)
        if any(not isinstance(name, str) or not name for name in preserve):
            raise ValueError("Preserved CKKS operation names must be non-empty")
        overlap = preserve.intersection(selections)
        if overlap:
            raise ValueError(
                "CKKS operations cannot be both preserved and assigned a "
                f"lowering: {tuple(sorted(overlap))}"
            )
        object.__setattr__(self, "selections", MappingProxyType(selections))
        object.__setattr__(self, "preserve", preserve)

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        config = workspace.get(CkksConfig)
        if config is None:
            matched = sum(
                1
                for operation in program_operations(program)
                if self.registry.supports(operation)
            )
            return PassResult.unchanged(
                program,
                matched=matched,
                skipped=matched,
                diagnostics=("CKKS configuration is not available",),
            )
        if not isinstance(config, CkksConfig):
            raise TypeError(
                "Compile workspace CkksConfig entry has incompatible value"
            )
        result = lower_ckks_program(
            program,
            config,
            registry=self.registry,
            selections=self.selections,
            preserve=self.preserve,
        )
        if not result.changed:
            return PassResult.unchanged(
                result.program,
                matched=result.stats.matched,
                skipped=result.stats.skipped,
                decisions=result.decisions,
            )
        return PassResult(
            result.program,
            PassStats(
                matched=result.stats.matched,
                transformed=result.stats.transformed,
                inserted=result.stats.inserted,
                skipped=result.stats.skipped,
            ),
            decisions=result.decisions,
        )


__all__ = ["LowerCkksToRnsNttPass"]
