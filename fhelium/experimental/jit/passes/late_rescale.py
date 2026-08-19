"""Reporting pass for backend-specific late-rescale optimization."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from .._dialect import operation_name
from .._program import Program
from ._base import PassResult
from ._utils import program_operations


@dataclass(frozen=True)
class LateRescalePass:
    """Count explicit rescale candidates and preserve their placement.

    Rescale movement changes CKKS rounding and accumulation behavior and
    requires backend-specific legality analysis. This module-wide reporting
    step returns a legal no-op with every candidate counted as both matched and
    skipped, plus one explanatory diagnostic when candidates exist. A caller
    composes a backend optimizer to perform any authorized movement.
    """

    name: str = "late-rescale"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Return an unchanged Program with candidate counts and diagnostics."""

        del workspace
        candidates = sum(
            operation_name(operation) == "fhelium.ckks.rescale"
            for operation in program_operations(program)
        )
        diagnostics = (
            (
                "late rescale requires a separately composed backend pass; "
                "candidate operations were left unchanged",
            )
            if candidates
            else ()
        )
        return PassResult.unchanged(
            program,
            matched=candidates,
            skipped=candidates,
            diagnostics=diagnostics,
        )
