"""Current Eager-backed adapters for provider-assigned JIT regions.

This module adapts the existing Eager engine and its lifecycle-owned resources
to JIT region compilers. It is not a generic continuation of Compile and does
not define a universal JIT context or execution protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import torch
from xdsl.dialects.builtin import (
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import BlockArgument, Operation, SSAValue

from fhelium.backend.execution import ProgramExecutable, OperationBackend
from fhelium.backend.implementation import requested_implementation
from fhelium.backend.assembly import create_builtin_operation_registry
from fhelium.backend.ckks.resources import RescaleExecutionResource
from fhelium.backend.resources import ResourceBindings
from fhelium.eager._program_values import (
    public_value_from_tensor,
    verify_runtime_value_type,
)
from fhelium.values import (
    Ciphertext,
    ConjugationKey,
    EvaluationKeySet,
    KeySwitchKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from fhelium.eager import Engine
from fhelium.ir import (
    OperationSpec,
    Program,
    operation_name,
    value_role,
)
from fhelium.ir.dialects import ckks, core, rns
from fhelium.compile import Compilation
from fhelium.compile.passes.lowering import (
    DEFAULT_CKKS_LOWERINGS,
    lower_ckks_program,
)

from .._analysis import (
    RUNTIME_CKKS_OPERATION_TYPES,
    RUNTIME_OPERATION_TYPES,
    rotation_key_reference,
)
from ..backends._interpreter_runtime import (
    _execute_operation_region,
    _materialize_interpreter_region_inputs,
    check_interpreter_coverage,
)
from .._contracts import CoverageDiagnostic
from .._execution import ExecutionInputs

from .._bindings import RuntimeBindings
from ..planning._records import (
    CompiledRegion,
    LoweredRegion,
    RegionProposal,
    extract_proposal_program,
    proposal_inputs_outputs,
)
from ._rns_ntt_triton import RnsNttRegionCompiler
from ._registry import BackendProvider, RegionCompilerRegistry

_SHARED_CKKS_OPERATION_TYPES = DEFAULT_CKKS_LOWERINGS.operation_types


def _eager_engine(
    bindings: RuntimeBindings,
) -> Engine | None:
    engine = bindings.eager_engine
    return engine if isinstance(engine, Engine) else None


def _eager_backend(
    engine: Engine,
    execution: ExecutionInputs,
) -> OperationBackend:
    """Return the shared operation backend selected by JIT."""

    return engine._operation_backend(execution.device)


def _evaluation_key_for_symbol(
    symbol: str,
    kind: str,
    inventory: object,
) -> KeySwitchKey | None:
    if isinstance(inventory, Mapping):
        value = inventory.get(symbol)
        return value if isinstance(value, KeySwitchKey) else None
    if isinstance(inventory, EvaluationKeySet):
        if kind == "relinearization-key":
            return inventory.relinearization
        if kind == "conjugation-key":
            return inventory.conjugation
        if kind == "rotation-key" and symbol.startswith("rotation-key:"):
            try:
                return inventory.rotations[int(symbol.partition(":")[2])]
            except (KeyError, ValueError):
                return None
        return None
    expected_type: type[KeySwitchKey] | None = {
        "key-switch-key": KeySwitchKey,
        "relinearization-key": RelinearizationKey,
        "rotation-key": RotationKey,
        "conjugation-key": ConjugationKey,
    }.get(kind)
    return (
        cast(KeySwitchKey, inventory)
        if expected_type is not None and type(inventory) is expected_type
        else None
    )


def _key_mapping_for_lowered_program(
    program: Program,
    bindings: RuntimeBindings,
) -> dict[str, KeySwitchKey]:
    keys: dict[str, KeySwitchKey] = {}
    for operation in program.single_block("main").ops:
        if not isinstance(operation, core.ResourceRefOp):
            continue
        if operation.symbol is None or operation.kind is None:
            continue
        symbol = operation.symbol.data
        kind = operation.kind.data
        if kind not in {
            "key-switch-key",
            "relinearization-key",
            "rotation-key",
            "conjugation-key",
        }:
            continue
        key = _evaluation_key_for_symbol(
            symbol,
            kind,
            bindings.evaluation_keys,
        )
        if key is None:
            raise KeyError(
                f"Shared CKKS region requires {kind} resource {symbol!r}"
            )
        keys[symbol] = key
    return keys


@dataclass(frozen=True)
class _PublicCkksExecutable:
    """Adapt public CKKS values to a selected backend executable."""

    executable: ProgramExecutable
    result_types: tuple[object, ...]
    provider: str
    implementation: str

    @property
    def backend(self) -> str:
        return self.provider

    @property
    def manifest(self) -> Mapping[str, object]:
        manifest = getattr(self.executable, "manifest", {})
        return MappingProxyType(
            {
                "provider": self.provider,
                "implementation": self.implementation,
                "backend_executable": dict(manifest),
                "fallback": False,
            }
        )

    def _temporary_result_scales(
        self,
        args: tuple[object, ...],
    ) -> tuple[float | None, ...]:
        """Propagate scales for the current Eager-backed JIT boundary.

        JIT Programs may carry ``scale="unknown"`` until invocation. This
        local fallback follows the already lowered straight-line region; it is
        intentionally not a general JIT state model.
        """

        block = self.executable.program.single_block("main")
        scales: dict[SSAValue, float] = {}
        for argument, value in zip(block.args, args, strict=True):
            if isinstance(value, (Ciphertext, Plaintext)):
                scales[argument] = value.scale
        for operation in block.ops:
            if isinstance(operation, UnrealizedConversionCastOp):
                if len(operation.inputs) == 1 and operation.inputs[0] in scales:
                    scales[operation.outputs[0]] = scales[operation.inputs[0]]
                continue
            if isinstance(operation, (core.ResourceRefOp, ReturnOp)):
                continue
            operand_scales = [
                scales[operand]
                for operand in operation.operands
                if operand in scales
            ]
            if not operand_scales:
                continue
            if isinstance(
                operation,
                (
                    ckks.MultiplyOp,
                    rns.MontgomeryMultiplyOp,
                    rns.MultiplyPlaintextOp,
                ),
            ):
                output_scale = operand_scales[0] * operand_scales[1]
            elif isinstance(operation, ckks.MultiplyScalarOp):
                output_scale = operand_scales[0] * float(
                    operation.scalar_scale.value.data
                )
            elif isinstance(operation, rns.ReinterpretScaleOp):
                output_scale = float(operation.scale.value.data)
            elif isinstance(operation, rns.RescaleDropLeadingPrimesOp):
                plan_owner = operation.plan.owner
                if not isinstance(plan_owner, core.ResourceRefOp):
                    raise ValueError(
                        "JIT rescale plan is not a resource reference"
                    )
                bound = self.executable.bound_resources[plan_owner]
                resource = bound.value
                if not isinstance(resource, RescaleExecutionResource):
                    raise TypeError("JIT rescale resource has another type")
                source_state = cast(
                    rns.RnsBundleType,
                    operation.value.type,
                ).state.data
                depth_attribute = source_state.get("depth")
                if not isinstance(depth_attribute, IntegerAttr):
                    raise ValueError("JIT rescale requires represented depth")
                depth = int(depth_attribute.value.data)
                prime_ids = resource.rns_context.rns_layout.prime_ids(depth)
                divisor = 1
                for prime_id in prime_ids[:operation.drop_count.value.data]:
                    divisor *= int(resource.rns_context.montgomery_parameters.moduli[prime_id])
                output_scale = operand_scales[0] / divisor

            else:
                output_scale = operand_scales[0]
            for result in operation.results:
                scales[result] = output_scale
        return_operation = tuple(block.ops)[-1]
        if not isinstance(return_operation, ReturnOp):
            raise ValueError("Shared CKKS region has no func.return")
        return tuple(scales.get(value) for value in return_operation.arguments)

    def run(self, *args: object, **kwargs: object) -> object:
        if kwargs:
            raise TypeError("Shared CKKS regions accept positional values")
        block = self.executable.program.single_block("main")
        if len(args) != len(block.args):
            raise ValueError(
                f"Shared CKKS region requires {len(block.args)} inputs, "
                f"got {len(args)}"
            )
        tensors: list[torch.Tensor] = []
        for index, (argument, value) in enumerate(
            zip(block.args, args, strict=True)
        ):
            if not isinstance(value, (Ciphertext, Plaintext)):
                raise TypeError(
                    "Shared CKKS lowering received a non-RNS public value"
                )
            verify_runtime_value_type(
                argument.type,
                value,
                label=f"Shared CKKS input {index}",
            )
            if value.data is None:
                raise ValueError("Shared CKKS lowering requires RNS inputs")
            tensors.append(value.data)
        result = self.executable.run(*tensors)
        results = result if isinstance(result, tuple) else (result,)
        if len(results) != len(self.result_types):
            raise ValueError(
                "CKKS lowering executable returned a different arity"
            )
        public_results: list[object] = []
        inferred_scales = self._temporary_result_scales(args)
        for value, result_type, inferred_scale in zip(
            results,
            self.result_types,
            inferred_scales,
            strict=True,
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "CKKS lowering executable returned a non-Tensor payload"
                )
            template = next(
                (
                    argument
                    for argument in args
                    if (
                        isinstance(result_type, ckks.CiphertextType)
                        and isinstance(argument, Ciphertext)
                    )
                    or (
                        isinstance(result_type, ckks.PlaintextType)
                        and isinstance(argument, Plaintext)
                    )
                ),
                None,
            )
            public_results.append(
                public_value_from_tensor(
                    result_type,
                    value,
                    template=template,
                    scale_if_unrepresented=inferred_scale,
                )
            )
        return (
            public_results[0]
            if len(public_results) == 1
            else tuple(public_results)
        )


@dataclass(frozen=True)
class EagerCkksLoweringRegionCompiler:
    """Compile CKKS ops through the same lowering and backend used by eager."""

    name: str = "shared-ckks-lowering"
    priority: int = 200
    operation_types: tuple[type[Operation], ...] = _SHARED_CKKS_OPERATION_TYPES

    def diagnostics(
        self,
        operation: Operation,
        specification: OperationSpec,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del operation, specification
        return self._binding_diagnostics(execution, bindings)

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del program, proposal
        return self._binding_diagnostics(execution, bindings)

    def lower(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> LoweredRegion:
        diagnostics = self.region_diagnostics(
            program,
            proposal,
            execution=execution,
            bindings=bindings,
        )
        if any(item.severity == "error" for item in diagnostics):
            raise RuntimeError("; ".join(item.message for item in diagnostics))
        lowered = extract_proposal_program(program, proposal)
        engine = _eager_engine(bindings)
        assert engine is not None
        result = lower_ckks_program(
            lowered,
            engine.config,
        )
        if not result.changed:
            raise ValueError(
                "Shared CKKS region contains no registered lowering"
            )
        return LoweredRegion(
            proposal,
            lowered,
            manifest={
                "provider": proposal.provider,
                "implementation": self.name,
                "lowering": "shared-ckks-to-rns-ntt",
            },
        )

    def verify(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        diagnostics = list(self._binding_diagnostics(execution, bindings))
        try:
            lowered.program.verify_structure()
        except Exception as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "invalid-shared-ckks-lowering",
                    str(error),
                    lowered.proposal.provider,
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
        engine = _eager_engine(bindings)
        assert engine is not None
        backend = _eager_backend(engine, execution)
        key_mapping = _key_mapping_for_lowered_program(
            lowered.program,
            bindings,
        )
        additional_resources = None
        if key_mapping:
            additional_resources = engine._key_bindings(
                key_mapping,
                device=execution.device,
                operation_name="JIT region build",
            )
        if additional_resources is not None:
            backend = backend.with_named_resources(additional_resources)
        executable = backend.link(Compilation(lowered.program))
        block = lowered.program.single_block("main")
        return_op = tuple(block.ops)[-1]
        assert isinstance(return_op, ReturnOp)
        result_types = tuple(argument.type for argument in return_op.arguments)
        return CompiledRegion(
            lowered,
            _PublicCkksExecutable(
                executable,
                result_types,
                lowered.proposal.provider,
                self.name,
            ),
        )

    @staticmethod
    def _binding_diagnostics(
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        engine = _eager_engine(bindings)
        if engine is None:
            return (
                CoverageDiagnostic(
                    "missing-eager-engine",
                    "Shared CKKS lowering requires an Engine binding.",
                    "eager_engine",
                ),
            )
        try:
            _eager_backend(engine, execution)
        except (RuntimeError, TypeError, ValueError) as error:
            return (
                CoverageDiagnostic(
                    "unavailable-eager-execution-device",
                    str(error),
                    "eager_engine",
                ),
            )
        return ()


@dataclass(frozen=True)
class _PublicBoundaryExecutable:
    """Adapt public objects around one Engine boundary-operation call."""

    engine: Engine
    operation: Operation
    bindings: RuntimeBindings
    device: torch.device
    provider: str
    implementation: str

    @property
    def backend(self) -> str:
        return self.provider

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "provider": self.provider,
                "implementation": self.implementation,
                "operation": self.operation.name,
                "device": str(self.device),
                "fallback": False,
            }
        )

    def run(self, *args: object, **kwargs: object) -> object:
        if kwargs:
            raise TypeError("CKKS boundary regions accept positional values")
        if len(args) != 1:
            raise ValueError("CKKS boundary region requires one input")
        operation = self.operation
        value = args[0]
        if isinstance(operation, ckks.EncodeOp):
            return self.engine.encode(
                value,  # type: ignore[arg-type]
                depth=int(operation.depth.value.data),
                scale=float(operation.scale.value.data),
                device=self.device,
            )
        if isinstance(operation, ckks.DecodeOp):
            if not isinstance(value, Plaintext):
                raise TypeError("CKKS decode requires Plaintext")
            return self.engine.decode(
                value,
                is_real=bool(int(operation.is_real.value.data)),
                device=self.device,
            )
        if isinstance(operation, ckks.IntegerCoefficientsToRnsOp):
            if not isinstance(value, Plaintext):
                raise TypeError("Integer-to-RNS requires Plaintext")
            return self.engine.integer_coefficients_to_rns(
                value.to(self.device),
                modulus_basis=cast(
                    Literal["Q", "QP"], operation.modulus_basis.data
                ),
            )
        if isinstance(operation, ckks.EncryptOp):
            if not isinstance(value, Plaintext):
                raise TypeError("CKKS encrypt requires Plaintext")
            key = self.bindings.public_key
            if not isinstance(key, PublicKey):
                raise TypeError("CKKS encrypt requires bindings.public_key")
            return self.engine.encrypt(value, key, device=self.device)
        if isinstance(operation, ckks.DecryptOp):
            if not isinstance(value, Ciphertext):
                raise TypeError("CKKS decrypt requires Ciphertext")
            key = self.bindings.resources.get(operation.key_symbol.data)
            if not isinstance(key, SecretKey):
                raise TypeError(
                    "CKKS decrypt requires a SecretKey in bindings.resources"
                )
            return self.engine.decrypt(value, key, device=self.device)
        raise TypeError(
            f"Unsupported CKKS boundary operation {operation.name!r}"
        )


_PUBLIC_BOUNDARY_OPERATION_TYPES = (
    ckks.EncodeOp,
    ckks.DecodeOp,
    ckks.IntegerCoefficientsToRnsOp,
    ckks.EncryptOp,
    ckks.DecryptOp,
)


@dataclass(frozen=True)
class EagerPublicBoundaryRegionCompiler:
    """Execute one public CKKS boundary operation through an Engine."""

    name: str = "public-ckks-boundary"
    priority: int = 300
    merge_adjacent: bool = False
    operation_types: tuple[type[Operation], ...] = (
        _PUBLIC_BOUNDARY_OPERATION_TYPES
    )

    def diagnostics(
        self,
        operation: Operation,
        specification: OperationSpec,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del specification
        diagnostics = list(
            EagerCkksLoweringRegionCompiler._binding_diagnostics(
                execution, bindings
            )
        )
        if isinstance(operation, ckks.EncryptOp) and not isinstance(
            bindings.public_key, PublicKey
        ):
            diagnostics.append(
                CoverageDiagnostic(
                    "missing-public-key",
                    "CKKS encrypt requires bindings.public_key.",
                    self.name,
                )
            )
        if isinstance(operation, ckks.DecryptOp) and not isinstance(
            bindings.resources.get(operation.key_symbol.data), SecretKey
        ):
            diagnostics.append(
                CoverageDiagnostic(
                    "missing-secret-key",
                    "CKKS decrypt requires its named SecretKey resource.",
                    self.name,
                )
            )
        return tuple(diagnostics)

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del program
        diagnostics = list(
            EagerCkksLoweringRegionCompiler._binding_diagnostics(
                execution, bindings
            )
        )
        if len(proposal.operation_indices) != 1:
            diagnostics.append(
                CoverageDiagnostic(
                    "invalid-public-boundary-region",
                    "A public CKKS boundary region contains one operation.",
                    self.name,
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
        diagnostics = self.region_diagnostics(
            program, proposal, execution=execution, bindings=bindings
        )
        if any(item.severity == "error" for item in diagnostics):
            raise RuntimeError("; ".join(item.message for item in diagnostics))
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
        diagnostics = list(
            EagerCkksLoweringRegionCompiler._binding_diagnostics(
                execution, bindings
            )
        )
        try:
            lowered.program.verify_structure()
        except Exception as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "invalid-public-boundary-program",
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
        engine = _eager_engine(bindings)
        assert engine is not None
        backend = _eager_backend(engine, execution)
        operation = next(
            op
            for op in lowered.program.single_block("main").ops
            if not isinstance(op, ReturnOp)
        )
        selected = backend.registry.resolve(
            operation,
            requested=requested_implementation(operation),
        ).name
        return CompiledRegion(
            lowered,
            _PublicBoundaryExecutable(
                engine,
                operation,
                bindings,
                execution.device,
                lowered.proposal.provider,
                selected,
            ),
        )


@dataclass(frozen=True)
class EagerPreservedCkksRegionCompiler:
    """Execute one preserved CKKS operation with a registered implementation."""

    name: str = "preserved-ckks-operation"
    priority: int = 100
    merge_adjacent: bool = False
    operation_types: tuple[type[Operation], ...] = ()

    @staticmethod
    def _additional_resources(
        engine: Engine,
        operation: Operation,
        bindings: RuntimeBindings,
        execution: ExecutionInputs,
    ) -> ResourceBindings | None:
        key_mapping: dict[str, KeySwitchKey] = {}
        if isinstance(operation, ckks.RelinearizeOp):
            key = _evaluation_key_for_symbol(
                "relinearization-key",
                "relinearization-key",
                bindings.evaluation_keys,
            )
            if type(key) is not RelinearizationKey:
                raise TypeError(
                    "Relinearize requires a RelinearizationKey binding"
                )
            key_mapping["relinearization-key"] = key
        elif isinstance(operation, ckks.RotateOp):
            symbol, step = rotation_key_reference(operation.key)
            key = bindings.resources.get(symbol)
            if type(key) is not RotationKey:
                raise TypeError(
                    f"Rotate requires RotationKey resource {symbol!r}"
                )
            if key.rotation_step != step:
                raise ValueError(f"Rotation key {symbol!r} has another step")
            key_mapping[symbol] = key
        elif isinstance(operation, ckks.RotateManyOp):
            for key_operand in operation.keys:
                symbol, step = rotation_key_reference(key_operand)
                key = bindings.resources.get(symbol)
                if type(key) is not RotationKey:
                    raise TypeError(
                        f"Rotate-many requires RotationKey resource {symbol!r}"
                    )
                if key.rotation_step != step:
                    raise ValueError(
                        f"Rotation key {symbol!r} has another step"
                    )
                key_mapping[symbol] = key
        return (
            engine._key_bindings(
                key_mapping,
                device=execution.device,
                operation_name="JIT public-boundary build",
            )
            if key_mapping
            else None
        )

    def diagnostics(
        self,
        operation: Operation,
        specification: OperationSpec,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del specification
        diagnostics = list(
            EagerCkksLoweringRegionCompiler._binding_diagnostics(
                execution, bindings
            )
        )
        engine = _eager_engine(bindings)
        if engine is not None:
            try:
                additional = self._additional_resources(
                    engine, operation, bindings, execution
                )
                backend = _eager_backend(engine, execution)
                if additional is not None:
                    backend = backend.with_named_resources(additional)
                messages = backend.diagnostics(operation)
            except (KeyError, TypeError, ValueError) as error:
                messages = (str(error),)
            diagnostics.extend(
                CoverageDiagnostic(
                    "unavailable-preserved-ckks-implementation",
                    message,
                    self.name,
                )
                for message in messages
            )
        return tuple(diagnostics)

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del program
        diagnostics = list(
            EagerCkksLoweringRegionCompiler._binding_diagnostics(
                execution, bindings
            )
        )
        if len(proposal.operation_indices) != 1:
            diagnostics.append(
                CoverageDiagnostic(
                    "invalid-preserved-ckks-region",
                    "A preserved CKKS region contains one operation.",
                    self.name,
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
        diagnostics = self.region_diagnostics(
            program, proposal, execution=execution, bindings=bindings
        )
        if any(item.severity == "error" for item in diagnostics):
            raise RuntimeError("; ".join(item.message for item in diagnostics))
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
        diagnostics = list(
            EagerCkksLoweringRegionCompiler._binding_diagnostics(
                execution, bindings
            )
        )
        try:
            lowered.program.verify_structure()
        except Exception as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "invalid-preserved-ckks-program",
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
        engine = _eager_engine(bindings)
        assert engine is not None
        operation = next(
            op
            for op in lowered.program.single_block("main").ops
            if not isinstance(op, ReturnOp)
        )
        backend = _eager_backend(engine, execution)
        additional = self._additional_resources(
            engine, operation, bindings, execution
        )
        selected = backend.registry.resolve(
            operation,
            requested=requested_implementation(operation),
        ).name
        if additional is not None:
            backend = backend.with_named_resources(additional)
        executable = backend.link(Compilation(lowered.program))
        block = lowered.program.single_block("main")
        return_op = tuple(block.ops)[-1]
        assert isinstance(return_op, ReturnOp)
        return CompiledRegion(
            lowered,
            _PublicCkksExecutable(
                executable,
                tuple(argument.type for argument in return_op.arguments),
                lowered.proposal.provider,
                selected,
            ),
        )


@dataclass
class _NativeProgramExecutable:
    program: Program
    operations: tuple[Operation, ...]
    inputs: tuple[SSAValue, ...]
    outputs: tuple[SSAValue, ...]
    source_input_is_entry: tuple[bool, ...]
    bindings: RuntimeBindings
    provider: str
    implementation: str
    execution: ExecutionInputs
    lowering_manifest: Mapping[str, object]

    @property
    def backend(self) -> str:
        return self.provider

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "provider": self.provider,
                "implementation": self.implementation,
                "operations": [operation_name(op) for op in self.operations],
                "lowering": dict(self.lowering_manifest),
                "fallback": False,
            }
        )

    def run(self, *args: object, **kwargs: object) -> object:
        if kwargs:
            raise TypeError("Native regions accept positional values")
        diagnostics = _native_region_diagnostics(
            self.program, self.execution, self.bindings
        )
        errors = tuple(
            item.message for item in diagnostics if item.severity == "error"
        )
        if errors:
            raise RuntimeError(
                "Native region bindings changed after build: "
                + "; ".join(errors)
            )
        _validate_message_tensor_placement(
            self.inputs,
            args,
            self.execution,
            self.provider,
            direction="input",
        )
        materialized = _materialize_interpreter_region_inputs(
            self.inputs,
            args,
            self.bindings,
            self.source_input_is_entry,
            execution_device=self.execution.device,
        )
        values = _execute_operation_region(
            self.operations,
            self.inputs,
            self.outputs,
            materialized,
            self.bindings,
            execution_device=self.execution.device,
        )
        _validate_message_tensor_placement(
            self.outputs,
            values,
            self.execution,
            self.provider,
            direction="output",
        )
        if len(values) == 1:
            return values[0]
        return values


@dataclass(frozen=True)
class EagerOperationRegionCompiler:
    """Execute assigned operations through configured eager APIs."""

    name: str = "eager-operation"
    operation_types: tuple[type[Operation], ...] = ()

    def diagnostics(
        self,
        operation: Operation,
        specification: OperationSpec,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        del operation, specification, execution, bindings
        return ()

    def region_diagnostics(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        """Check interpreter coverage and native device bindings for a region."""

        return _native_region_diagnostics(
            extract_proposal_program(program, proposal), execution, bindings
        )

    def lower(
        self,
        program: Program,
        proposal: RegionProposal,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> LoweredRegion:
        source_program = extract_proposal_program(program, proposal)
        source_inputs, _ = proposal_inputs_outputs(program, proposal)
        return LoweredRegion(
            proposal,
            source_program,
            source_input_is_entry=tuple(
                isinstance(value, BlockArgument) for value in source_inputs
            ),
            manifest={
                "provider": proposal.provider,
                "implementation": self.name,
                "lowering": None,
                "execution_adapter": "eager-operations",
            },
        )

    def verify(
        self,
        lowered: LoweredRegion,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
    ) -> tuple[CoverageDiagnostic, ...]:
        source = lowered.program
        if not isinstance(source, Program):
            return (
                CoverageDiagnostic(
                    "invalid-native-lowering",
                    "Native lowering does not include a source Program.",
                    lowered.proposal.provider,
                ),
            )
        diagnostics = list(
            _native_region_diagnostics(source, execution, bindings)
        )
        try:
            lowered.program.verify_structure()
        except Exception as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "invalid-backend-ir",
                    str(error),
                    lowered.proposal.provider,
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
        source = lowered.program
        assert isinstance(source, Program)
        block = source.single_block("main")
        operations = tuple(
            operation
            for operation in block.ops
            if not isinstance(operation, ReturnOp)
        )
        return_op = tuple(block.ops)[-1]
        assert isinstance(return_op, ReturnOp)
        executable = _NativeProgramExecutable(
            source,
            operations,
            tuple(block.args),
            tuple(return_op.arguments),
            lowered.source_input_is_entry,
            bindings,
            lowered.proposal.provider,
            self.name,
            execution,
            lowered.manifest,
        )
        return CompiledRegion(lowered, executable)


def _native_region_diagnostics(
    program: Program,
    execution: ExecutionInputs,
    bindings: RuntimeBindings,
) -> tuple[CoverageDiagnostic, ...]:
    """Validate interpreter bindings and the selected Eager execution bundle."""

    report = check_interpreter_coverage(program, bindings)
    diagnostics = [
        CoverageDiagnostic(item.code, item.message, item.subject, item.severity)
        for item in report.diagnostics
    ]
    uses_ckks = any(
        isinstance(operation, RUNTIME_CKKS_OPERATION_TYPES)
        for operation in program.single_block("main").ops
    )
    engine = bindings.eager_engine
    if uses_ckks and isinstance(engine, Engine):
        try:
            _eager_backend(engine, execution)
        except (RuntimeError, TypeError, ValueError) as error:
            diagnostics.append(
                CoverageDiagnostic(
                    "unavailable-eager-execution-device",
                    str(error),
                    "eager_engine",
                )
            )
    return tuple(diagnostics)


def _validate_message_tensor_placement(
    value_specs: tuple[SSAValue, ...],
    values: tuple[object, ...],
    execution: ExecutionInputs,
    provider: str,
    *,
    direction: str,
) -> None:
    """Enforce native provider placement for tensor-valued messages."""

    expected = torch.device(
        "cpu" if provider == "native_cpu" else execution.device
    )
    for index, (specification, value) in enumerate(
        zip(value_specs, values, strict=True)
    ):
        if value_role(specification) != "message" or not isinstance(
            value, torch.Tensor
        ):
            continue
        if value.device != expected:
            raise ValueError(
                f"Native provider {provider!r} requires {direction} message "
                f"Tensor {index} on {expected}, got {value.device}"
            )


def eager_backend_provider(
    target: Literal["cpu", "cuda"],
) -> BackendProvider:
    """Construct an Eager-backed JIT provider for one native target."""

    registered_types = create_builtin_operation_registry(
        ntt_backend_name=None,
    ).operation_types
    ckks_operation_types = frozenset(ckks.FHEliumCkks.operations)
    preserved_types = tuple(
        operation_type
        for operation_type in registered_types
        if operation_type in ckks_operation_types
        and operation_type not in _PUBLIC_BOUNDARY_OPERATION_TYPES
    )
    eager_types = tuple(
        operation_type
        for operation_type in RUNTIME_OPERATION_TYPES
        if operation_type
        not in (
            *_SHARED_CKKS_OPERATION_TYPES,
            *preserved_types,
            *_PUBLIC_BOUNDARY_OPERATION_TYPES,
        )
    )
    return BackendProvider(
        f"native_{target}",
        target,
        regions=RegionCompilerRegistry(
            (
                EagerCkksLoweringRegionCompiler(),
                RnsNttRegionCompiler(target),
                EagerPublicBoundaryRegionCompiler(),
                EagerPreservedCkksRegionCompiler(
                    operation_types=preserved_types
                ),
                EagerOperationRegionCompiler(operation_types=eager_types),
            )
        ),
    )


__all__ = [
    "EagerCkksLoweringRegionCompiler",
    "EagerOperationRegionCompiler",
    "EagerPreservedCkksRegionCompiler",
    "EagerPublicBoundaryRegionCompiler",
    "eager_backend_provider",
]
