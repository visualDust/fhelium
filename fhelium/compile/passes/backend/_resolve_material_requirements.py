"""Resolve material declarations across an existing Program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from fhelium.backend.implementation import OperationImplementationRegistry
from ..._material_requirements import resolve_material_requirements
from ..._pipeline import PassResult
from .._operation_transforms import program_operations

if TYPE_CHECKING:
    from ..._compilation import Compilation


@dataclass(frozen=True)
class ResolveMaterialRequirementsPass:
    """Resolve key layouts and implementation material operands across a Program.

    The registry defaults to the Compilation workspace. ``key_layouts`` adds caller layout declarations keyed by material symbol or label. Existing operands and compatible metadata are retained; insufficient facts leave the corresponding requirements unresolved.
    """

    registry: OperationImplementationRegistry | None = None
    key_layouts: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )
    name: str = "resolve-material-requirements"

    def run(self, compilation: Compilation) -> PassResult:
        if self.key_layouts:
            compilation.workspace["key_layouts"] = {
                **cast(
                    Mapping[str, Mapping[str, object]],
                    compilation.workspace.get("key_layouts", {}),
                ),
                **self.key_layouts,
            }
        return resolve_material_requirements(
            compilation,
            program_operations(compilation.program),
            self.registry,
        )


__all__ = ["ResolveMaterialRequirementsPass"]
