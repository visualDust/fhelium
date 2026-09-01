"""Runtime-oriented transformation, coverage, build, and execution session."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from fhelium.compile import Compilation, CompileWorkspace, Pipeline
from fhelium.ir import Program

from ._contracts import (
    Backend,
    BackendRegistry,
    BuildResult,
    CoverageDiagnostic,
    CoverageReport,
    ProviderPolicy,
)

from ._bindings import RuntimeBindings
from ._execution import ExecutionInputs, _observe_execution_inputs


class Session:
    """Coordinate JIT transforms, coverage, builds, and execution.

    One session owns a concrete execution device, ``RuntimeBindings``, a
    backend registry, and an optional default backend for related requests.
    Device inventory is observed on the first coverage or build request.
    """

    device: torch.device
    _execution_inputs: ExecutionInputs | None
    bindings: RuntimeBindings
    backends: BackendRegistry
    workspace: CompileWorkspace
    default_backend: str | None = None

    def __init__(
        self,
        *,
        device: str | torch.device,
        bindings: RuntimeBindings | None = None,
        backends: Sequence[Backend] = (),
        default_backend: str | None = None,
    ) -> None:
        selected_device = torch.device(device)
        if selected_device.type == "cpu":
            selected_device = torch.device("cpu")
        elif selected_device.type == "cuda":
            if selected_device.index is None:
                raise ValueError(
                    "JIT Session CUDA device must have a concrete index"
                )
        else:
            raise ValueError(
                "JIT Session supports CPU and CUDA devices, got "
                f"{selected_device.type!r}"
            )
        if bindings is None:
            bindings = RuntimeBindings()
        elif not isinstance(bindings, RuntimeBindings):
            raise TypeError("JIT Session bindings must be RuntimeBindings")
        self.device = selected_device
        self._execution_inputs = None
        self.bindings = bindings
        self.backends = BackendRegistry(backends)
        self.workspace = CompileWorkspace()
        if default_backend is not None:
            self.backends.require(default_backend)
        self.default_backend = default_backend

    def transform(
        self,
        program: Program,
        pipeline: Pipeline,
    ) -> Compilation:
        """Run one shared partial pipeline with this Session's workspace."""

        if not isinstance(program, Program):
            raise TypeError("Session.transform program must be a Program")
        if not isinstance(pipeline, Pipeline):
            raise TypeError("Session.transform pipeline must be a Pipeline")
        return pipeline.run(Compilation(program, self.workspace))

    def coverage(
        self,
        program: Program,
        *,
        backend: str | None = None,
        policy: ProviderPolicy | None = None,
    ) -> CoverageReport:
        """Return one backend's operation and binding coverage for ``program``."""

        selected = self._backend(backend)
        return selected.coverage(
            program,
            execution=self._execution(),
            bindings=self.bindings,
            policy=policy,
        )

    def build(
        self,
        program: Program,
        *,
        backend: str | None = None,
        policy: ProviderPolicy | None = None,
    ) -> BuildResult:
        """Build when the selected backend covers operations and bindings.

        An uncovered program is returned as a normal BuildResult with no
        executable, permitting the caller to transform or bind it further.
        """

        if not isinstance(program, Program):
            raise TypeError("Session.build program must be a Program")
        selected = self._backend(backend)
        execution = self._execution()
        report = selected.coverage(
            program,
            execution=execution,
            bindings=self.bindings,
            policy=policy,
        )
        if not report.covered:
            return BuildResult(
                program,
                selected.name,
                report,
                decisions=report.provider_decisions,
            )
        try:
            executable = selected.build(
                program,
                execution=execution,
                bindings=self.bindings,
                policy=policy,
            )

        except Exception as error:
            failed = CoverageReport(
                backend=selected.name,
                diagnostics=(
                    *report.diagnostics,
                    CoverageDiagnostic(
                        "backend-build-failed",
                        f"Backend build failed: {error}",
                        selected.name,
                    ),
                ),
                operations=report.operations,
                operation_support=report.operation_support,
                regions=report.regions,
                provider_decisions=report.provider_decisions,
            )
            return BuildResult(
                program,
                selected.name,
                failed,
                decisions=failed.provider_decisions,
            )
        return BuildResult(
            program,
            selected.name,
            report,
            executable,
            decisions=report.provider_decisions,
        )

    def _execution(self) -> ExecutionInputs:
        execution = self._execution_inputs
        if execution is None:
            execution = _observe_execution_inputs(self.device)
            self._execution_inputs = execution
        return execution

    def run(
        self,
        program: Program,
        *args: object,
        backend: str | None = None,
        policy: ProviderPolicy | None = None,
        **kwargs: object,
    ) -> Any:
        """Build one selected backend and invoke its executable."""

        result = self.build(program, backend=backend, policy=policy)
        if result.executable is None:
            detail = "; ".join(
                item.message for item in result.coverage.diagnostics
            )
            raise RuntimeError(
                f"JIT backend {result.backend!r} did not build an executable: "
                + detail
            )
        return result.executable.run(*args, **kwargs)

    def _backend(self, name: str | None) -> Backend:
        selected = name or self.default_backend
        if selected is None:
            raise ValueError(
                "A backend name is required; Session does not select one "
                "implicitly"
            )
        return self.backends.require(selected)


__all__ = ["Session"]
