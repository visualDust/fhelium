"""Interpreter backend for neutral mixed-level IR Programs."""

from __future__ import annotations


from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from fhelium.ir import Program

from .._contracts import (
    CoverageDiagnostic,
    CoverageReport,
    Executable,
    ProviderPolicy,
)
from .._execution import ExecutionInputs

from .._bindings import RuntimeBindings
from ._interpreter_runtime import check_interpreter_coverage, interpret_program


@dataclass
class InterpreterExecutable:
    """Execute one neutral Program with the IR interpreter."""

    _program: Program
    _bindings: RuntimeBindings
    _execution: ExecutionInputs
    _entry: str = "main"

    @property
    def backend(self) -> str:
        """Return the interpreter backend registry name."""

        return "interpreter"

    @property
    def manifest(self) -> Mapping[str, object]:
        """Describe the interpreter target and selected entry."""

        return MappingProxyType(
            {
                "backend": self.backend,
                "entry": self._entry,
                "execution_inputs": self._execution.as_dict(),
            }
        )

    def run(self, *args: object, **kwargs: object) -> Any:
        """Interpret the selected entry with the supplied live bindings."""

        return interpret_program(
            self._program,
            *args,
            bindings=self._bindings,
            execution_device=self._execution.device,
            entry=self._entry,
            **kwargs,
        )

    def __call__(self, *args: object, **kwargs: object) -> Any:
        return self.run(*args, **kwargs)


@dataclass(frozen=True)
class InterpreterBackend:
    """Build interpreter executables for currently supported IR.

    The single-block and operation-schema limitations belong to this backend;
    they do not constrain neutral IR or JIT transformation.
    """

    entry: str = "main"
    name: str = "interpreter"

    def coverage(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> CoverageReport:
        """Report interpreter implementation and live-binding coverage."""

        diagnostics: list[CoverageDiagnostic] = []
        if policy is not None:
            diagnostics.extend(policy.validate(execution))
            if self.name not in policy.providers:
                diagnostics.append(
                    CoverageDiagnostic(
                        "provider-not-enabled",
                        f"Provider policy does not enable {self.name!r}.",
                        self.name,
                    )
                )
        try:
            report = check_interpreter_coverage(
                program, bindings, entry=self.entry
            )
        except Exception as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "interpreter-analysis-failed",
                    f"Interpreter could not inspect the current Module: {error}",
                    self.entry,
                )
            )
            return CoverageReport(self.name, tuple(diagnostics))

        diagnostics.extend(
            CoverageDiagnostic(
                item.code,
                item.message,
                item.subject,
                item.severity,
            )
            for item in report.diagnostics
        )
        return CoverageReport(
            self.name,
            tuple(diagnostics),
            report.requirements.operations,
        )

    def build(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> Executable:
        """Construct an interpreter executable after caller-checked coverage."""

        report = self.coverage(
            program,
            execution=execution,
            bindings=bindings,
            policy=policy,
        )
        if not report.covered:
            detail = "; ".join(item.message for item in report.diagnostics)
            raise RuntimeError(
                "Interpreter backend does not cover the supplied Program: "
                + detail
            )
        return InterpreterExecutable(
            program.clone(), bindings, execution, self.entry
        )


__all__ = ["InterpreterBackend", "InterpreterExecutable"]
