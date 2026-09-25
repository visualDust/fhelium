"""State carried through one caller-composed compilation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fhelium.ir import Program

from ._pipeline import PassReport

from ._workspace import CompileWorkspace


@dataclass(frozen=True)
class Compilation:
    """Carry one Program and the state accumulated while transforming it.

    ``material_bindings`` maps Program symbols to live Tensor data. Every pass
    receives the current Compilation and uses this same dictionary.
    ``workspace`` holds caller inputs and pass-produced data that do not belong
    in portable IR. ``reports`` records the ordered pass history. A Pipeline
    returns a new Compilation with a transformed Program while retaining the
    same workspace and binding dictionary. Copy the dictionary when preparing
    independent assignments; copying it does not copy Tensor storage.
    """

    program: Program
    workspace: CompileWorkspace = field(default_factory=CompileWorkspace)
    reports: tuple[PassReport, ...] = ()
    material_bindings: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.program, Program):
            raise TypeError("Compilation program must be a Program")
        if not isinstance(self.workspace, CompileWorkspace):
            raise TypeError("Compilation workspace must be a CompileWorkspace")
        reports = tuple(self.reports)
        if not all(isinstance(report, PassReport) for report in reports):
            raise TypeError(
                "Compilation reports must contain PassReport values"
            )
        object.__setattr__(self, "reports", reports)

    def _with_pipeline_result(
        self,
        program: Program,
        reports: tuple[PassReport, ...],
    ) -> Compilation:
        """Return the Compilation produced by one Pipeline invocation."""

        return Compilation(
            program,
            self.workspace,
            (*self.reports, *reports),
            self.material_bindings,
        )


__all__ = ["Compilation"]
