"""Backend coverage, provider policy, build results, and executable protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fhelium.ir import Program
from ._execution import ExecutionInputs

from ._bindings import RuntimeBindings

if TYPE_CHECKING:
    from .planning._plan import OperationSupport, PlanRegion


@dataclass(frozen=True)
class CoverageDiagnostic:
    """Describe one backend coverage or binding limitation."""

    code: str
    message: str
    subject: str | None = None
    severity: str = "error"

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError(
                "CoverageDiagnostic code and message must be non-empty"
            )
        if self.severity not in {"error", "warning", "note"}:
            raise ValueError(
                "CoverageDiagnostic severity must be error, warning, or note"
            )


@dataclass(frozen=True)
class ProviderDecision:
    """Record one backend assignment outcome for coverage inspection."""

    provider: str
    implementation: str
    operation_ids: tuple[str, ...]
    selected: bool
    reason: str
    proposal_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not self.implementation or not self.reason:
            raise ValueError("ProviderDecision text fields must be non-empty")
        object.__setattr__(
            self,
            "proposal_metadata",
            MappingProxyType(dict(self.proposal_metadata)),
        )


@dataclass(frozen=True)
class CoverageReport:
    """Operation and binding implementation coverage for one Program."""

    backend: str
    diagnostics: tuple[CoverageDiagnostic, ...] = ()
    operations: frozenset[str] = frozenset()
    operation_support: tuple[OperationSupport, ...] = ()
    regions: tuple[PlanRegion, ...] = ()
    provider_decisions: tuple[ProviderDecision, ...] = ()

    @property
    def covered(self) -> bool:
        """Whether no backend-coverage error was reported."""

        return not any(item.severity == "error" for item in self.diagnostics)


@dataclass(frozen=True)
class ProviderPolicy:
    """Select providers and whether CKKS operations remain unlowered."""

    target: str
    providers: tuple[str, ...] = ()
    preserve_ckks_operations: bool = False

    def __post_init__(self) -> None:
        if self.target not in {"cpu", "cuda"}:
            raise ValueError("ProviderPolicy target must be 'cpu' or 'cuda'")
        if not self.providers:
            raise ValueError("ProviderPolicy must enable at least one provider")
        if len(set(self.providers)) != len(self.providers) or any(
            not provider for provider in self.providers
        ):
            raise ValueError(
                "ProviderPolicy providers must be distinct non-empty names"
            )
        if type(self.preserve_ckks_operations) is not bool:
            raise TypeError(
                "ProviderPolicy preserve_ckks_operations must be bool"
            )

    def validate(
        self, execution: ExecutionInputs
    ) -> tuple[CoverageDiagnostic, ...]:
        """Compare the provider target with the Session execution target."""

        diagnostics: list[CoverageDiagnostic] = []
        if execution.device.type != self.target:
            diagnostics.append(
                CoverageDiagnostic(
                    "target-mismatch",
                    f"Provider policy target {self.target!r} differs from "
                    f"Session execution target {execution.device.type!r}.",
                    self.target,
                )
            )
        return tuple(diagnostics)


@runtime_checkable
class Executable(Protocol):
    """Run one backend-built computation with supplied runtime inputs."""

    @property
    def backend(self) -> str:
        """Stable backend name that built this executable."""

        ...

    @property
    def manifest(self) -> Mapping[str, object]:
        """Inspect target, implementation, and requirement metadata."""

        ...

    def run(self, *args: object, **kwargs: object) -> Any:
        """Execute with backend-defined result typing."""

        ...


@runtime_checkable
class Backend(Protocol):
    """Analyze coverage and build an executable for one IR Program."""

    @property
    def name(self) -> str:
        """Stable backend registry name."""

        ...

    def coverage(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> CoverageReport:
        """Return implementation and binding coverage for ``program``."""

        ...

    def build(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> Executable:
        """Build after a successful coverage report."""

        ...


@dataclass(frozen=True)
class BuildResult:
    """Return backend coverage and an optional successfully built executable."""

    program: Program
    backend: str
    coverage: CoverageReport
    executable: Executable | None = None
    generated_assets: tuple[object, ...] = ()
    decisions: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.coverage.backend != self.backend:
            raise ValueError("BuildResult backend and coverage backend differ")
        if self.executable is not None and not self.coverage.covered:
            raise ValueError("BuildResult cannot carry an uncovered executable")


class BackendRegistry:
    """Record caller-selected backend implementations by stable name."""

    def __init__(self, backends: Sequence[Backend] = ()) -> None:
        self._backends: dict[str, Backend] = {}
        for backend in backends:
            self.register(backend)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered backend names in insertion order."""

        return tuple(self._backends)

    def register(self, backend: Backend) -> None:
        """Register one backend, rejecting ambiguous replacement."""

        if not isinstance(backend, Backend):
            raise TypeError("Backend must implement name, coverage, and build")
        if not backend.name:
            raise ValueError("Backend name must be non-empty")
        if backend.name in self._backends:
            raise ValueError(f"Backend {backend.name!r} is already registered")
        self._backends[backend.name] = backend

    def require(self, name: str) -> Backend:
        """Return one named backend or report the known choices."""

        try:
            return self._backends[name]
        except KeyError:
            raise KeyError(
                f"Backend {name!r} is not registered; available={self.names}"
            ) from None


__all__ = [
    "Backend",
    "BackendRegistry",
    "BuildResult",
    "CoverageDiagnostic",
    "CoverageReport",
    "Executable",
    "ProviderDecision",
    "ProviderPolicy",
]
