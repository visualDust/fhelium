"""Invoke an provided ciphertext-state validator."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

from .._program import Program
from ._base import PassResult
from ._utils import program_operations

StateValidator = Callable[[Program, MutableMapping[Any, Any]], None]


@dataclass(frozen=True)
class ValidateCipherStatesPass:
    """Apply one caller-selected numerical ciphertext-state validator.

    The validator receives the current Program and retained Workspace and owns
    backend-specific policy for engines, parameter sets, and exact CKKS state.
    Successful validation returns the Program unchanged and reports the number
    of direct module-wide operation candidates.
    """

    validator: StateValidator
    name: str = "validate-cipher-states"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Validate exact state through the selected service."""

        self.validator(program, workspace)
        return PassResult.unchanged(
            program, matched=len(program_operations(program))
        )
