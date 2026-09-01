"""Typed public exceptions raised by FHElium.

The exception hierarchy is intentionally small and centralized. Callers may
catch :class:`FHEliumError` for package-level failures or a narrower category
such as :class:`ConfigurationError` or :class:`StateError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fhelium.residency.plan import (
        MemoryReservation,
        ResidencyAction,
    )
    from fhelium.residency.snapshot import ResidencyPlanReport
else:
    # Keep runtime annotation resolution cycle-free. Static analysis and the
    # source API generator resolve the concrete stable Residency types.
    ResidencyAction = Any
    MemoryReservation = Any
    ResidencyPlanReport = Any


class FHEliumError(Exception):
    """Base class for FHElium-specific failures."""


class ConfigurationError(FHEliumError, ValueError):
    """Invalid or unavailable cryptographic configuration."""


class PrimeCatalogError(ConfigurationError):
    """Base class for prime-catalog lookup, capacity, or resource failures."""


class PrimeCatalogResourceError(PrimeCatalogError):
    """An installed immutable prime-catalog resource is missing or invalid."""

    def __init__(self, *, resource_name: str, detail: str) -> None:
        self.resource_name = str(resource_name)
        self.detail = str(detail)
        super().__init__(
            f"Prime-catalog resource {self.resource_name!r} is unavailable "
            f"or invalid: {self.detail}"
        )


class MessagePrimeCatalogEntryNotFoundError(PrimeCatalogError):
    """No message-prime catalog entry matches the requested parameters."""

    def __init__(self, *, coefficient_bits: int, ring_dimension: int) -> None:
        self.coefficient_bits = int(coefficient_bits)
        self.ring_dimension = int(ring_dimension)
        super().__init__(
            "No message-prime catalog entry for "
            f"coefficient_bits={self.coefficient_bits}, "
            f"ring_dimension={self.ring_dimension}."
        )


class ScalePrimeCatalogEntryNotFoundError(PrimeCatalogError):
    """No scale-prime catalog entry matches the requested parameters."""

    def __init__(self, *, scale_bits: int, ring_dimension: int) -> None:
        self.scale_bits = int(scale_bits)
        self.ring_dimension = int(ring_dimension)
        super().__init__(
            "No scale-prime catalog entry for "
            f"scale_bits={self.scale_bits}, "
            f"ring_dimension={self.ring_dimension}."
        )


class InsufficientPrimeCatalogError(PrimeCatalogError):
    """The catalog cannot supply the requested modulus-chain width."""

    def __init__(
        self,
        *,
        prime_kind: Literal["message", "scale"],
        ring_dimension: int,
        required_count: int,
        available_count: int,
    ) -> None:
        self.prime_kind = prime_kind
        self.ring_dimension = int(ring_dimension)
        self.required_count = int(required_count)
        self.available_count = int(available_count)
        super().__init__(
            f"Insufficient {prime_kind}-prime catalog capacity for "
            f"ring_dimension={self.ring_dimension}: "
            f"required_count={self.required_count}, "
            f"available_count={self.available_count}."
        )


class SecurityBudgetExceededError(ConfigurationError):
    """The requested modulus chain exceeds the configured security budget."""

    def __init__(
        self,
        *,
        scale_bits: int,
        ring_dimension: int,
        num_scale_primes: int,
        maximum_modulus_bits: int,
        requested_modulus_bits: int,
    ) -> None:
        self.scale_bits = int(scale_bits)
        self.ring_dimension = int(ring_dimension)
        self.num_scale_primes = int(num_scale_primes)
        self.maximum_modulus_bits = int(maximum_modulus_bits)
        self.requested_modulus_bits = int(requested_modulus_bits)
        super().__init__(
            "Requested modulus chain exceeds the security budget: "
            f"requested_modulus_bits={self.requested_modulus_bits}, "
            f"maximum_modulus_bits={self.maximum_modulus_bits}, "
            f"scale_bits={self.scale_bits}, "
            f"ring_dimension={self.ring_dimension}, "
            f"num_scale_primes={self.num_scale_primes}."
        )


class SecurityParametersUnsupportedError(ConfigurationError):
    """No supported security-table row matches the configuration."""

    def __init__(
        self,
        *,
        ring_dimension: int,
        target_bits: int,
        secret_distribution: str,
        error_stddev: float,
        reason: str,
    ) -> None:
        self.ring_dimension = int(ring_dimension)
        self.target_bits = int(target_bits)
        self.secret_distribution = str(secret_distribution)
        self.error_stddev = float(error_stddev)
        self.reason = str(reason)
        super().__init__(
            "No supported security assessment matches the "
            f"configuration: ring_dimension={self.ring_dimension}, "
            f"target_bits={self.target_bits}, "
            f"secret_distribution={self.secret_distribution!r}, "
            f"error_stddev={self.error_stddev!r}. {self.reason}"
        )


class StateError(FHEliumError, RuntimeError):
    """Base class for an invalid cryptographic value state."""


class ScaleError(StateError, ValueError):
    """Base class for invalid or incompatible CKKS scale metadata."""


class InvalidScaleError(ScaleError):
    """A CKKS scale is not representable as a positive finite float."""

    def __init__(self, *, value_name: str, scale: object) -> None:
        self.value_name = value_name
        self.scale = scale
        super().__init__(
            f"{value_name} scale must be a positive finite real number; "
            f"got {scale!r}."
        )


class ScaleMismatchError(ScaleError):
    """Two values have different scales for an equal-scale operation."""

    def __init__(
        self,
        *,
        operation: str,
        lhs_name: str,
        lhs_scale: float,
        rhs_name: str,
        rhs_scale: float,
    ) -> None:
        self.operation = operation
        self.lhs_name = lhs_name
        self.lhs_scale = float(lhs_scale)
        self.rhs_name = rhs_name
        self.rhs_scale = float(rhs_scale)
        self.scale_ratio = self.lhs_scale / self.rhs_scale
        super().__init__(
            f"{operation} requires exactly equal scales: "
            f"{lhs_name}.scale={self.lhs_scale!r}, "
            f"{rhs_name}.scale={self.rhs_scale!r}, "
            f"ratio={self.scale_ratio!r}."
        )


class ExecutionError(FHEliumError, RuntimeError):
    """Base class for a failure in a FHElium execution program."""


class ExecutionInputError(ExecutionError):
    """An input tree is incompatible with an execution buffer or program."""


class CudaGraphCaptureError(ExecutionError):
    """A program failed during CUDA Graph warmup, capture, or first replay."""

    def __init__(
        self,
        *,
        stage: str,
        function_name: str,
        detail: str,
    ) -> None:
        self.stage = stage
        self.function_name = function_name
        self.detail = detail
        super().__init__(
            f"CUDA Graph {stage} failed for {function_name}: {detail}"
        )


class CudaGraphInputError(ExecutionInputError):
    """Replay input structure or value state differs from capture."""


class ArtifactError(FHEliumError, RuntimeError):
    """Base class for structured artifact identity and version failures."""


class UnsupportedArtifactStoreVersionError(ArtifactError):
    """A local artifact catalog uses an unsupported on-disk version."""

    def __init__(
        self,
        *,
        found_version: int,
        supported_versions: tuple[int, ...],
        migration_available: bool = False,
    ) -> None:
        self.found_version = int(found_version)
        self.supported_versions = tuple(
            int(version) for version in supported_versions
        )
        self.migration_available = bool(migration_available)
        migration = (
            "A migration is available."
            if self.migration_available
            else "Automatic migration is not implemented."
        )
        super().__init__(
            "Unsupported FHElium artifact-store format: "
            f"found_version={self.found_version}, "
            f"supported_versions={self.supported_versions}. {migration}"
        )


class StaleArtifactReferenceError(ArtifactError, ValueError):
    """An artifact reference no longer names the store's active generation."""

    def __init__(
        self,
        *,
        name: str,
        expected_artifact_id: str,
        current_artifact_id: str | None,
        differences: dict[str, tuple[object, object]] | None = None,
    ) -> None:
        self.name = str(name)
        self.expected_artifact_id = str(expected_artifact_id)
        self.current_artifact_id = (
            None if current_artifact_id is None else str(current_artifact_id)
        )
        self.differences = {} if differences is None else dict(differences)
        detail = (
            ""
            if not self.differences
            else f" differences={self.differences!r}."
        )
        super().__init__(
            f"ArtifactRef for {self.name!r} is stale or belongs to another "
            "store generation: "
            f"expected_artifact_id={self.expected_artifact_id!r}, "
            f"current_artifact_id={self.current_artifact_id!r}."
            f"{detail}"
        )


class ResidencyError(FHEliumError, RuntimeError):
    """Base class for managed value-residency failures."""


class ResidencyBudgetError(ResidencyError):
    """A materialization or reservation exceeds one location budget."""

    def __init__(
        self,
        *,
        location: str,
        budget_bytes: int,
        used_bytes: int,
        reserved_bytes: int,
        requested_bytes: int,
    ) -> None:
        self.location = location
        self.budget_bytes = int(budget_bytes)
        self.used_bytes = int(used_bytes)
        self.reserved_bytes = int(reserved_bytes)
        self.requested_bytes = int(requested_bytes)
        super().__init__(
            f"Residency location {location!r} exceeds its manager budget: "
            f"budget_bytes={self.budget_bytes}, "
            f"used_bytes={self.used_bytes}, "
            f"reserved_bytes={self.reserved_bytes}, "
            f"requested_bytes={self.requested_bytes}."
        )


class ResidencyHandleError(ResidencyError):
    """A residency handle is unknown, foreign, discarded, or malformed."""


class ResidencyUnavailableError(ResidencyError):
    """A required value materialization is absent from a location."""


class ResidencyInUseError(ResidencyError):
    """An active read, hold, or pending CUDA event prevents a transition."""


class ResidencyLifetimeClosedError(ResidencyError):
    """A released residency lease, hold, or reservation was reused."""


class ResidencyMaterializationError(ResidencyError):
    """A source or transition produced an invalid materialization."""


class ResidencyOwnershipError(ResidencyError):
    """A transition would violate replica or recoverability ownership."""


class ResidencyPlanError(ResidencyError):
    """A residency plan is invalid or infeasible for the current state."""


class ResidencyStaleStateError(ResidencyPlanError):
    """A residency decision targets an obsolete manager state."""

    def __init__(self, *, expected_version: int, actual_version: int) -> None:
        self.expected_version = int(expected_version)
        self.actual_version = int(actual_version)
        super().__init__(
            "Residency manager state changed after the decision: "
            f"expected_version={self.expected_version}, "
            f"actual_version={self.actual_version}."
        )


class ResidencySearchLimitError(ResidencyPlanError):
    """Automatic residency planning stopped at its deterministic state limit."""

    def __init__(
        self,
        *,
        request_name: str,
        state_limit: int,
        explored_states: int,
        detail: str,
    ) -> None:
        self.request_name = str(request_name)
        self.state_limit = int(state_limit)
        self.explored_states = int(explored_states)
        self.detail = str(detail)
        super().__init__(
            f"Automatic residency search for {self.request_name!r} is "
            "inconclusive because it reached its deterministic state limit: "
            f"state_limit={self.state_limit}, "
            f"explored_states={self.explored_states}. {self.detail}"
        )


class ResidencyPlanExecutionError(ResidencyError):
    """A preflighted residency plan failed after execution began.

    ``partial_report`` contains every transition that completed before the
    failure. Those transitions remain committed. ``failed_action`` and
    ``failed_action_index`` identify the requested action that failed during
    action execution. ``failed_reservation`` and
    ``failed_reservation_index`` identify a scoped reservation admission
    failure. The fields for the other operation kind are ``None``. The original
    runtime exception is retained as ``__cause__`` by the manager.
    """

    partial_report: ResidencyPlanReport
    failed_action: ResidencyAction | None
    failed_reservation: MemoryReservation | None

    def __init__(
        self,
        *,
        plan_name: str,
        phase: Literal["execute", "reclaim", "reserve", "enter", "exit"],
        partial_report: ResidencyPlanReport,
        failed_action: ResidencyAction | None,
        failed_action_index: int | None,
        detail: str,
        failed_reservation: MemoryReservation | None = None,
        failed_reservation_index: int | None = None,
    ) -> None:
        self.plan_name = plan_name
        self.phase = phase
        self.partial_report = partial_report
        self.failed_action = failed_action
        self.failed_action_index = failed_action_index
        self.failed_reservation = failed_reservation
        self.failed_reservation_index = failed_reservation_index
        self.detail = detail
        super().__init__(
            f"Residency plan {plan_name!r} failed during {phase} after "
            f"{len(partial_report.transitions)} completed transition(s): "
            f"{detail}"
        )


class ResidencyClosedError(ResidencyError):
    """A residency manager was used after it closed."""


class ResidencyReentrancyError(ResidencyError):
    """External source or user code re-entered an active transition."""


class PolynomialDomainError(StateError):
    """A value is in the wrong polynomial domain for an operation."""

    def __init__(
        self,
        *,
        value_name: str,
        expected: Literal["coefficient", "ntt"],
        actual: str | None,
    ) -> None:
        self.value_name = value_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{value_name} requires polynomial domain {expected!r}; "
            f"actual domain is {actual!r}."
        )


class ResidueRepresentationError(StateError):
    """A value has the wrong RNS residue representation for an operation."""

    def __init__(
        self,
        *,
        value_name: str,
        expected: Literal["standard", "montgomery"],
        actual: str | None,
    ) -> None:
        self.value_name = value_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{value_name} requires residue representation {expected!r}; "
            f"actual representation is {actual!r}."
        )


class SecretKeyModulusBasisError(StateError):
    """A secret key does not contain the modulus basis an operation needs."""

    def __init__(
        self,
        *,
        operation: str,
        expected: Literal["Q", "QP"],
        actual: str,
    ) -> None:
        self.operation = operation
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{operation} requires secret-key modulus_basis {expected!r}; "
            f"actual modulus_basis is {actual!r}."
        )


class MaximumLevelError(StateError):
    """No further level drop is available for a ciphertext."""

    def __init__(self, *, level: int, maximum_level: int) -> None:
        self.level = int(level)
        self.maximum_level = int(maximum_level)
        super().__init__(
            "Ciphertext modulus-chain depth is exhausted: "
            f"level={self.level}, maximum_level={self.maximum_level}."
        )


__all__ = [
    "ArtifactError",
    "ConfigurationError",
    "CudaGraphCaptureError",
    "CudaGraphInputError",
    "ExecutionError",
    "ExecutionInputError",
    "FHEliumError",
    "InsufficientPrimeCatalogError",
    "InvalidScaleError",
    "MaximumLevelError",
    "MessagePrimeCatalogEntryNotFoundError",
    "PolynomialDomainError",
    "PrimeCatalogError",
    "PrimeCatalogResourceError",
    "ResidencyBudgetError",
    "ResidencyClosedError",
    "ResidencyError",
    "ResidencyHandleError",
    "ResidencyInUseError",
    "ResidencyLifetimeClosedError",
    "ResidencyMaterializationError",
    "ResidencyOwnershipError",
    "ResidencyPlanError",
    "ResidencyPlanExecutionError",
    "ResidencyReentrancyError",
    "ResidencySearchLimitError",
    "ResidencyStaleStateError",
    "ResidencyUnavailableError",
    "ResidueRepresentationError",
    "ScaleError",
    "ScaleMismatchError",
    "ScalePrimeCatalogEntryNotFoundError",
    "SecretKeyModulusBasisError",
    "SecurityBudgetExceededError",
    "StaleArtifactReferenceError",
    "StateError",
    "UnsupportedArtifactStoreVersionError",
]
