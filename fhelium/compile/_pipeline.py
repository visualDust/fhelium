"""Compose Compile passes over Programs and shared workspace data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

from fhelium.ir import Program


class TransformError(RuntimeError):
    """Report a malformed Compile transformation request or result."""


@dataclass(frozen=True)
class PassStats:
    """Count one pass's observed and changed patterns."""

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
                "PassStats transformed + skipped cannot exceed matched"
            )


@dataclass(frozen=True)
class DecisionRecord:
    """Describe one inspectable choice made during a transformation."""

    subject: str
    selected: str | None = None
    candidates: tuple[str, ...] = ()
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject:
            raise ValueError("DecisionRecord subject must be non-empty")
        if self.selected is not None and not isinstance(self.selected, str):
            raise TypeError("DecisionRecord selected must be a string or None")
        for name, values in (
            ("candidates", self.candidates),
            ("details", self.details),
        ):
            if not isinstance(values, tuple) or not all(
                isinstance(item, str) for item in values
            ):
                raise TypeError(
                    f"DecisionRecord {name} must be a tuple of strings"
                )


@dataclass(frozen=True)
class PassResult:
    """Program, counts, diagnostics, and choices returned by one pass."""

    program: Program
    stats: PassStats = PassStats()
    diagnostics: tuple[str, ...] = ()
    decisions: tuple[DecisionRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.program, Program):
            raise TypeError("PassResult program must be a Program")
        if not isinstance(self.stats, PassStats):
            raise TypeError("PassResult stats must be PassStats")
        if not isinstance(self.diagnostics, tuple) or not all(
            isinstance(item, str) for item in self.diagnostics
        ):
            raise TypeError("PassResult diagnostics must be a tuple of strings")
        if not isinstance(self.decisions, tuple) or not all(
            isinstance(item, DecisionRecord) for item in self.decisions
        ):
            raise TypeError(
                "PassResult decisions must be a tuple of DecisionRecord"
            )

    @property
    def changed(self) -> bool:
        """Whether the reported counts describe an IR change."""

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
        decisions: tuple[DecisionRecord, ...] = (),
    ) -> PassResult:
        """Record a normal pass result that preserves the Program."""

        return cls(
            program,
            PassStats(matched=matched, skipped=skipped),
            diagnostics,
            decisions,
        )


@dataclass(frozen=True)
class PassReport:
    """Record one pass's activity, diagnostics, and choices."""

    name: str
    stats: PassStats
    diagnostics: tuple[str, ...] = ()
    decisions: tuple[DecisionRecord, ...] = ()


@runtime_checkable
class _PipelineCompilation(Protocol):
    """Structural interface implemented by Compilation."""

    @property
    def program(self) -> Program: ...

    @property
    def workspace(self) -> dict[object, object]: ...

    @property
    def reports(self) -> tuple[PassReport, ...]: ...

    def _with_pipeline_result(
        self,
        program: Program,
        reports: tuple[PassReport, ...],
    ) -> Self: ...


CompilationT = TypeVar("CompilationT", bound=_PipelineCompilation)


@runtime_checkable
class Pass(Protocol):
    """Define one locally applicable transformation or analysis step."""

    @property
    def name(self) -> str:
        """Stable name used in pipeline composition and reports."""

        ...

    def run(
        self,
        program: Program,
        shared_data: dict[Any, Any],
        /,
    ) -> PassResult:
        """Inspect or transform a Program and report the outcome."""

        ...


@dataclass(frozen=True)
class Pipeline:
    """Run an ordered, dependency-free tuple of partial transformations."""

    passes: tuple[Pass, ...] = ()

    def __post_init__(self) -> None:
        passes = tuple(self.passes)
        for program_pass in passes:
            if not isinstance(program_pass, Pass):
                raise TypeError("Pipeline entries must implement Pass")
            if not isinstance(program_pass.name, str):
                raise TypeError("Pass name must be a string")
            if not program_pass.name.strip():
                raise ValueError("Pass name must be non-empty")
        object.__setattr__(self, "passes", passes)

    @property
    def names(self) -> tuple[str, ...]:
        """Return pass names in execution order."""

        return tuple(program_pass.name for program_pass in self.passes)

    def run(
        self,
        compilation: CompilationT,
    ) -> CompilationT:
        """Clone and transform one Compilation while retaining its workspace."""

        if not isinstance(compilation, _PipelineCompilation):
            raise TypeError("Pipeline input must be a Compilation")
        current = compilation.program.clone()
        shared_data = compilation.workspace
        reports: list[PassReport] = []
        for program_pass in self.passes:
            result = program_pass.run(current, shared_data)
            if not isinstance(result, PassResult):
                raise TransformError(
                    f"Pass {program_pass.name!r} returned "
                    f"{type(result).__name__}, expected PassResult"
                )
            try:
                result.program.verify_structure()
            except Exception as error:
                raise TransformError(
                    f"Pass {program_pass.name!r} returned structurally "
                    f"invalid IR: {error}"
                ) from error
            reports.append(
                PassReport(
                    program_pass.name,
                    result.stats,
                    tuple(result.diagnostics),
                    tuple(result.decisions),
                )
            )
            current = result.program
        return compilation._with_pipeline_result(current, tuple(reports))

    def then(self, *passes: Pass) -> Pipeline:
        """Append passes in order."""

        return Pipeline((*self.passes, *passes))

    def before(self, target: str, *passes: Pass) -> Pipeline:
        """Insert passes before one uniquely named pass."""

        index = self._unique_index(target)
        return Pipeline((*self.passes[:index], *passes, *self.passes[index:]))

    def after(self, target: str, *passes: Pass) -> Pipeline:
        """Insert passes after one uniquely named pass."""

        index = self._unique_index(target) + 1
        return Pipeline((*self.passes[:index], *passes, *self.passes[index:]))

    def replace(self, target: str, *passes: Pass) -> Pipeline:
        """Replace one uniquely named pass."""

        index = self._unique_index(target)
        return Pipeline(
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
            raise TransformError(
                f"Pipeline pass name {target!r} matched {len(matches)} steps"
            )
        return matches[0]


__all__ = [
    "DecisionRecord",
    "Pass",
    "PassReport",
    "PassResult",
    "PassStats",
    "Pipeline",
    "TransformError",
]
