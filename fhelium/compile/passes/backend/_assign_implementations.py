"""Record caller-selected backend implementations on matching operations."""

from __future__ import annotations

from ..._pipeline import (
    DecisionRecord,
    PassResult,
    PassStats,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from xdsl.dialects.builtin import StringAttr

from fhelium.ir import (
    EXECUTION_IMPLEMENTATION_ATTRIBUTE,
    Program,
    operation_name,
)
from .._operation_transforms import program_operations_and_known_region_owners


@dataclass(frozen=True)
class AssignImplementationsPass:
    """Attach requested implementation names to selected operation classes.

    ``selections`` maps exact textual operation names to implementation names.
    Unselected operations and unknown mixed-level IR remain unchanged. The
    attribute is a backend build constraint; this pass neither discovers
    implementations nor asserts executable coverage.
    """

    selections: Mapping[str, str] = field(default_factory=dict)
    overwrite: bool = False
    name: str = "assign-implementations"

    def __post_init__(self) -> None:
        if not isinstance(self.selections, Mapping):
            raise TypeError("implementation selections must be a mapping")
        selections = dict(self.selections)
        if any(
            not isinstance(operation, str)
            or not operation
            or not isinstance(implementation, str)
            or not implementation
            for operation, implementation in selections.items()
        ):
            raise ValueError(
                "operation and implementation names must be nonempty strings"
            )
        if type(self.overwrite) is not bool:
            raise TypeError("implementation assignment overwrite must be bool")
        object.__setattr__(
            self,
            "selections",
            MappingProxyType(selections),
        )

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Annotate exact operation-name matches and report every assignment."""

        del workspace
        matched = transformed = skipped = 0
        decisions: list[DecisionRecord] = []
        for operation in program_operations_and_known_region_owners(program):
            subject = operation_name(operation)
            selected = self.selections.get(subject)
            if selected is None:
                continue
            matched += 1
            previous = operation.attributes.get(
                EXECUTION_IMPLEMENTATION_ATTRIBUTE
            )
            if previous is not None and not isinstance(previous, StringAttr):
                raise ValueError(
                    f"operation {subject!r} has a non-string implementation "
                    "assignment"
                )
            previous_name = previous.data if previous is not None else None
            if previous_name == selected:
                skipped += 1
                decisions.append(
                    DecisionRecord(
                        subject,
                        selected,
                        (selected,),
                        ("assignment already present",),
                    )
                )
                continue
            if previous_name is not None and not self.overwrite:
                raise ValueError(
                    f"operation {subject!r} already requests implementation "
                    f"{previous_name!r}; set overwrite=True to replace it"
                )
            operation.attributes[EXECUTION_IMPLEMENTATION_ATTRIBUTE] = (
                StringAttr(selected)
            )
            transformed += 1
            details = (
                (f"replaced={previous_name}",)
                if previous_name is not None
                else ()
            )
            decisions.append(
                DecisionRecord(subject, selected, (selected,), details)
            )
        if transformed == 0:
            return PassResult.unchanged(
                program,
                matched=matched,
                skipped=skipped,
                decisions=tuple(decisions),
            )
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=transformed,
                skipped=skipped,
            ),
            decisions=tuple(decisions),
        )


__all__ = ["AssignImplementationsPass"]
