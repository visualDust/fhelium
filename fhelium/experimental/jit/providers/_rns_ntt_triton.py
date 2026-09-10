"""Lifecycle-neutral RNS/NTT and Triton JIT providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch
from xdsl.ir import Operation

from fhelium.backend.assembly import create_builtin_operation_registry
from fhelium.backend.ckks.resources import (
    CONJUGATION_KEY_RESOURCE_KIND,
    KEY_SWITCH_KEY_RESOURCE_KIND,
    RELINEARIZATION_KEY_RESOURCE_KIND,
    ROTATION_KEY_RESOURCE_KIND,
    KeySwitchExecutionResource,
    RescaleExecutionResource,
)
from fhelium.backend.execution import OperationBackend, ProgramExecutable
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.resources import (
    BoundResource,
)
from fhelium.backend.rns.context import RnsContext
from fhelium.compile import Compilation
from fhelium.ir import OperationSpec, Program, value_role
from fhelium.ir.dialects import ntt, rns, semantic
from fhelium.values import (
    ConjugationKey,
    KeySwitchKey,
    RelinearizationKey,
    RotationKey,
)

from .._contracts import CoverageDiagnostic
from .._execution import ExecutionInputs

from .._bindings import RuntimeBindings
from ..planning._records import (
    CompiledRegion,
    LoweredRegion,
    RegionProposal,
    extract_proposal_program,
)
from ._registry import BackendProvider, RegionCompilerRegistry

_RNS_NTT_OPERATION_TYPES = (
    rns.AddStandardOp,
    rns.SubtractStandardOp,
    rns.NegateStandardOp,
    rns.RescaleDropLeadingPrimesOp,
    rns.ExtractComponentOp,
    rns.PackTwoComponentsOp,
    rns.HybridModUpDigitOp,
    rns.KeySwitchDigitProductOp,
    rns.AddMontgomeryLazyOp,
    rns.ModDownQpToQOp,
    rns.CoefficientAutomorphismOp,
    ntt.CoefficientStandardToNttMontgomeryOp,
    ntt.CoefficientMontgomeryToNttMontgomeryOp,
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.InverseMontgomeryOp,
)

_EVALUATION_KEY_KINDS: dict[type[KeySwitchKey], str] = {
    KeySwitchKey: KEY_SWITCH_KEY_RESOURCE_KIND,
    RelinearizationKey: RELINEARIZATION_KEY_RESOURCE_KIND,
    RotationKey: ROTATION_KEY_RESOURCE_KIND,
    ConjugationKey: CONJUGATION_KEY_RESOURCE_KIND,
}


@dataclass(frozen=True)
class _RnsNttExecutable:
    """Bind low-IR resource operands and run one RNS/NTT Program."""

    executable: ProgramExecutable
    input_types: tuple[object, ...]
    provider: str

    @property
    def backend(self) -> str:
        return self.provider

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "provider": self.provider,
                "implementation": "rns-ntt",
                "operation_executable": dict(self.executable.manifest),
                "fallback": False,
            }
        )

    def run(self, *args: object, **kwargs: object) -> object:
        if kwargs:
            raise TypeError("RNS/NTT regions accept positional values")
        if len(args) != len(self.input_types):
            raise ValueError(
                f"RNS/NTT region requires {len(self.input_types)} inputs, "
                f"got {len(args)}"
            )
        bound: list[object] = []
        for index, (value, value_type) in enumerate(
            zip(args, self.input_types, strict=True)
        ):
            if isinstance(value_type, rns.RnsBundleType):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"RNS input {index} must be torch.Tensor")
                bound.append(value)
                continue
            if isinstance(value, BoundResource):
                bound.append(value)
                continue
            if isinstance(value_type, ntt.NttPlanType):
                if not isinstance(value, NttContext):
                    raise TypeError(f"NTT input {index} must be NttContext")
                bound.append(
                    BoundResource(f"argument-{index}", "ntt-plan", value)
                )
                continue
            if isinstance(value_type, rns.RnsParametersType):
                if not isinstance(value, RnsContext):
                    raise TypeError(
                        f"RNS parameter input {index} has wrong type"
                    )
                bound.append(
                    BoundResource(f"argument-{index}", "rns-parameters", value)
                )
                continue
            if isinstance(value_type, rns.RescalePlanType):
                if not isinstance(value, RescaleExecutionResource):
                    raise TypeError(f"Rescale input {index} has wrong type")
                bound.append(
                    BoundResource(f"argument-{index}", "rescale-plan", value)
                )
                continue
            if isinstance(value_type, rns.KeySwitchPlanType):
                if not isinstance(value, KeySwitchExecutionResource):
                    raise TypeError(
                        f"Key-switch plan input {index} has wrong type"
                    )
                bound.append(
                    BoundResource(f"argument-{index}", "key-switch-plan", value)
                )
                continue
            if isinstance(value_type, rns.EvaluationKeyResourceType):
                if not isinstance(value, KeySwitchKey):
                    raise TypeError(
                        f"Evaluation-key input {index} has wrong type"
                    )
                try:
                    kind = _EVALUATION_KEY_KINDS[type(value)]
                except KeyError:
                    raise TypeError(
                        f"Evaluation-key input {index} has wrong type"
                    ) from None
                bound.append(BoundResource(f"argument-{index}", kind, value))
                continue
            raise TypeError(
                f"RNS/NTT input {index} has unsupported IR type {value_type!r}"
            )
        return self.executable.run(*bound)


@dataclass(frozen=True)
class RnsNttRegionCompiler:
    """Compile RNS/NTT Programs through registered native implementations."""

    target: Literal["cpu", "cuda"]
    name: str = "rns-ntt"
    operation_types: tuple[type[Operation], ...] = _RNS_NTT_OPERATION_TYPES

    def diagnostics(
        self,
        operation: Operation,
        specification: OperationSpec,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del operation, specification, bindings
        if execution.device.type != self.target:
            return (
                CoverageDiagnostic(
                    "rns-ntt-target-mismatch",
                    f"RNS/NTT provider target {self.target!r} differs from "
                    f"Session execution target {execution.device.type!r}.",
                    self.name,
                ),
            )
        return ()

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del program, proposal, bindings
        if execution.device.type != self.target:
            return (
                CoverageDiagnostic(
                    "rns-ntt-target-mismatch",
                    f"RNS/NTT provider target {self.target!r} differs from "
                    f"Session execution target {execution.device.type!r}.",
                    self.name,
                ),
            )
        return ()

    def lower(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> LoweredRegion:
        del execution, bindings
        return LoweredRegion(
            proposal,
            extract_proposal_program(program, proposal),
            manifest={
                "provider": proposal.provider,
                "implementation": self.name,
                "lowering": None,
            },
        )

    def verify(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del bindings
        diagnostics: list[CoverageDiagnostic] = []
        if execution.device.type != self.target:
            diagnostics.append(
                CoverageDiagnostic(
                    "rns-ntt-target-mismatch",
                    f"RNS/NTT provider target {self.target!r} differs from "
                    f"Session execution target {execution.device.type!r}.",
                    self.name,
                )
            )
        try:
            lowered.program.verify_structure()
        except Exception as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "invalid-rns-ntt-program",
                    str(error),
                    self.name,
                )
            )
        return tuple(diagnostics)

    def build(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> CompiledRegion:
        diagnostics = self.verify(
            lowered, execution=execution, bindings=bindings
        )
        if any(item.severity == "error" for item in diagnostics):
            raise RuntimeError("; ".join(item.message for item in diagnostics))
        registry = create_builtin_operation_registry(
            ntt_backend_name=None,
        )
        backend = OperationBackend(registry)
        executable = backend.link(Compilation(lowered.program))
        block = lowered.program.single_block("main")
        return CompiledRegion(
            lowered,
            _RnsNttExecutable(
                executable,
                tuple(argument.type for argument in block.args),
                lowered.proposal.provider,
            ),
        )


_TRITON_OPERATION_TYPES = (
    semantic.AddOp,
    semantic.MultiplyOp,
    semantic.NegateOp,
)


@dataclass(frozen=True)
class TritonPointwiseRegionCompiler:
    """Compile adjacent assigned public pointwise operations as one kernel."""

    name: str = "triton-pointwise"
    operation_types: tuple[type[Operation], ...] = _TRITON_OPERATION_TYPES

    def diagnostics(
        self,
        operation: Operation,
        specification: OperationSpec,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        from ..backends._triton import _dependency_available

        del specification, execution, bindings
        diagnostics: list[CoverageDiagnostic] = []
        if not _dependency_available():
            diagnostics.append(
                CoverageDiagnostic(
                    "triton-unavailable",
                    "Triton pointwise implementation requires the optional "
                    "triton package.",
                    "triton",
                )
            )
        if any(
            value_role(value) != "message"
            for value in (*operation.operands, *operation.results)
        ):
            diagnostics.append(
                CoverageDiagnostic(
                    "triton-role-mismatch",
                    "Triton pointwise execution requires public message values.",
                    "triton",
                )
            )
        return tuple(diagnostics)

    def lower(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> LoweredRegion:
        del execution, bindings
        return LoweredRegion(
            proposal,
            extract_proposal_program(program, proposal),
            manifest={
                "provider": proposal.provider,
                "implementation": self.name,
                "lowering": "generated-triton-region",
            },
        )

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        """Check complete pointwise-region structure before assignment."""

        from ..backends._triton import TritonPointwiseBackend

        report = TritonPointwiseBackend().coverage(
            extract_proposal_program(program, proposal),
            execution=execution,
            bindings=bindings,
        )
        return report.diagnostics

    def verify(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        from ..backends._triton import TritonPointwiseBackend

        report = TritonPointwiseBackend().coverage(
            lowered.program, execution=execution, bindings=bindings
        )
        return report.diagnostics

    def build(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> CompiledRegion:
        diagnostics = self.verify(
            lowered, execution=execution, bindings=bindings
        )
        if any(item.severity == "error" for item in diagnostics):
            raise RuntimeError("; ".join(item.message for item in diagnostics))
        from ..backends._triton import TritonPointwiseBackend

        executable = TritonPointwiseBackend().build(
            lowered.program, execution=execution, bindings=bindings
        )
        return CompiledRegion(lowered, executable)


def triton_backend_provider() -> BackendProvider:
    """Construct the CUDA Triton provider for public pointwise operations."""

    return BackendProvider(
        "triton",
        "cuda",
        regions=RegionCompilerRegistry((TritonPointwiseRegionCompiler(),)),
    )


__all__ = [
    "RnsNttRegionCompiler",
    "TritonPointwiseRegionCompiler",
    "triton_backend_provider",
]
