"""Permissive pass protocol for the canonical xDSL ``Program``.

A pipeline clones its source Program once, gives each ordered pass the
same mutable Workspace, and accepts both transformations and reported legal
no-ops. Each pass defines its own matching scope. The standard local rewriting
passes scan direct operations in every block of every top-level function;
entry-oriented validation and visualization passes select their configured
entry. A completed pipeline records per-pass analysis results, while readiness is a
separate execution decision.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .._errors import JitPassError
from .._program import Program
from .._workspace import Workspace


@dataclass(frozen=True)
class PassStats:
    """Count one pass's local matching and rewrite activity."""

    matched: int = 0
    transformed: int = 0
    inserted: int = 0
    removed: int = 0
    skipped: int = 0

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"PassStats {field_name} must be an integer")
            if value < 0:
                raise ValueError(
                    f"PassStats {field_name} must be nonnegative, got {value}"
                )
        if self.transformed + self.skipped > self.matched:
            raise ValueError(
                "PassStats transformed + skipped cannot exceed matched: "
                f"{self.transformed} + {self.skipped} > {self.matched}"
            )


@dataclass(frozen=True)
class PassResult:
    """Carry one pass's Program, activity counts, and diagnostics."""

    program: Program
    stats: PassStats = PassStats()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.program, Program):
            raise TypeError(
                "PassResult program must be Program, got "
                f"{type(self.program).__name__}"
            )
        diagnostics = tuple(self.diagnostics)
        if not all(isinstance(item, str) for item in diagnostics):
            raise TypeError("PassResult diagnostics must contain strings")
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def changed(self) -> bool:
        """Whether reported transformation counts changed the Program."""

        return bool(
            self.stats.transformed or self.stats.inserted or self.stats.removed
        )

    @classmethod
    def unchanged(
        cls,
        program: Program,
        *,
        matched: int = 0,
        skipped: int = 0,
        diagnostics: tuple[str, ...] = (),
    ) -> PassResult:
        """Record a successful pass invocation that preserved the Program."""

        return cls(
            program,
            PassStats(matched=matched, skipped=skipped),
            diagnostics,
        )


@dataclass(frozen=True)
class PassReport:
    """Persist one named pass's counts and diagnostics as compact evidence."""

    name: str
    stats: PassStats
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    """Return the transformed Program, retained workspace, and pass reports."""

    program: Program
    workspace: MutableMapping[Any, Any]
    reports: tuple[PassReport, ...]


@runtime_checkable
class Pass(Protocol):
    """Define one named Program transformation or analysis step.

    A successful implementation may return the input Program unchanged when no
    operation matches or when matched operations lack local prerequisites.
    """

    @property
    def name(self) -> str:
        """Stable name used in pipeline composition and reports."""

        ...

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Run one pass and return its Program and behavior evidence."""

        ...


@dataclass(frozen=True)
class PassPipeline:
    """Run an inspectable ordered tuple of independent Program passes.

    One clone is made before the first pass, preserving the caller's source
    Program while allowing xDSL rewriters to mutate the private module in
    place. Every pass receives the same Workspace object, including when
    a pass returns a replacement Program. The pipeline retains all Workspace
    entries with their caller- or pass-defined interpretation and invalidation
    policy. After each pass, xDSL structural verification checks the returned
    Program; numerical and execution readiness remain separate decisions.
    """

    passes: tuple[Pass, ...] = ()

    def __post_init__(self) -> None:
        passes = tuple(self.passes)
        for program_pass in passes:
            if not isinstance(program_pass, Pass):
                raise TypeError(
                    "PassPipeline entries must expose name and "
                    "run(program, workspace)"
                )
            if not isinstance(program_pass.name, str):
                raise TypeError("JIT pass name must be a string")
            if not program_pass.name.strip():
                raise ValueError("JIT pass name must be non-empty")
        object.__setattr__(self, "passes", passes)

    @property
    def names(self) -> tuple[str, ...]:
        """Return pass names in execution order."""

        return tuple(program_pass.name for program_pass in self.passes)

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any] | None = None,
    ) -> PipelineResult:
        """Run every pass over one clone with retained Workspace identity."""

        if not isinstance(program, Program):
            raise TypeError(
                "PassPipeline program must be Program, got "
                f"{type(program).__name__}"
            )
        if workspace is None:
            workspace = Workspace()
        elif not isinstance(workspace, MutableMapping):
            raise TypeError("PassPipeline workspace must be a mutable mapping")

        current = program.clone()
        reports: list[PassReport] = []
        for program_pass in self.passes:
            result = program_pass.run(current, workspace)
            if not isinstance(result, PassResult):
                raise JitPassError(
                    f"Pass {program_pass.name!r} returned "
                    f"{type(result).__name__}, expected PassResult"
                )
            if not isinstance(result.program, Program):
                raise JitPassError(
                    f"Pass {program_pass.name!r} returned a PassResult with "
                    f"{type(result.program).__name__} instead of Program"
                )
            try:
                result.program.module.verify()
            except Exception as error:
                raise JitPassError(
                    f"Pass {program_pass.name!r} returned structurally "
                    f"invalid xDSL: {error}"
                ) from error
            reports.append(
                PassReport(
                    program_pass.name,
                    result.stats,
                    result.diagnostics,
                )
            )
            current = result.program
        return PipelineResult(current, workspace, tuple(reports))

    def then(self, *passes: Pass) -> PassPipeline:
        """Return a pipeline with passes appended in order."""

        return PassPipeline((*self.passes, *passes))

    def before(self, target: str, *passes: Pass) -> PassPipeline:
        """Insert passes before one uniquely named existing pass."""

        index = self._unique_index(target)
        return PassPipeline(
            (*self.passes[:index], *passes, *self.passes[index:])
        )

    def after(self, target: str, *passes: Pass) -> PassPipeline:
        """Insert passes after one uniquely named existing pass."""

        index = self._unique_index(target) + 1
        return PassPipeline(
            (*self.passes[:index], *passes, *self.passes[index:])
        )

    def replace(self, target: str, *passes: Pass) -> PassPipeline:
        """Replace one uniquely named pass with zero or more passes."""

        index = self._unique_index(target)
        return PassPipeline(
            (*self.passes[:index], *passes, *self.passes[index + 1 :])
        )

    def _unique_index(self, target: str) -> int:
        if not isinstance(target, str):
            raise TypeError("Pipeline target name must be a string")
        matches = [
            index
            for index, program_pass in enumerate(self.passes)
            if program_pass.name == target
        ]
        if len(matches) != 1:
            raise JitPassError(
                f"Pipeline pass name {target!r} matched {len(matches)} steps; "
                "expected exactly one"
            )
        return matches[0]


__all__ = [
    "Pass",
    "PassPipeline",
    "PassReport",
    "PassResult",
    "PassStats",
    "PipelineResult",
]
