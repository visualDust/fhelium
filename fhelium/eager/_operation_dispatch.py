"""Select and cache Backend calls for graph-free Eager operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.ckks import (
    CkksDeviceResources,
)
from fhelium.backend.ckks.crypto import KeyGenerationResource
from fhelium.backend.execution import OperationBackend
from fhelium.backend.implementation import (
    OperationImplementation,
    OperationInvocation,
)
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.resources import (
    BoundResource,
)
from fhelium.backend.rns.context import RnsContext
from fhelium.ir.dialects import ckks, ntt, rns
from fhelium.rng import Csprng


class _EagerOperationDispatcher:
    """Dispatch graph-free Eager operations through one device's Backend."""

    def __init__(
        self,
        *,
        device_resources: CkksDeviceResources,
        backend: OperationBackend,
    ) -> None:
        self.device_resources = device_resources
        self.backend = backend
        self.random_lock = RLock()
        self._direct_calls: dict[
            tuple[object, ...],
            tuple[OperationImplementation, tuple[BoundResource, ...]],
        ] = {}
        self._invocations: dict[tuple[object, ...], OperationInvocation] = {}

    @property
    def rns_context(self) -> RnsContext:
        return self.device_resources.rns_context

    @property
    def ntt_context(self) -> NttContext:
        return self.device_resources.ntt_context

    @property
    def rng(self) -> Csprng:
        return self.device_resources.rng

    @property
    def key_generation(self) -> KeyGenerationResource:
        return self.device_resources.key_generation

    def materialize_all(self) -> None:
        """Make every built-in CKKS binding available to Program builders."""

        self.backend = self.backend.with_named_resources(
            self.device_resources.bindings
        )

    def _invocation(
        self,
        operation_type: type[Operation],
        operand_count: int,
        result_count: int,
        attributes: Mapping[str, object] | None,
        bases: Sequence[str | None],
    ) -> tuple[OperationInvocation, tuple[object, ...]]:
        represented_bases = tuple(bases)
        represented_attributes = (
            () if attributes is None else tuple(attributes.items())
        )
        key = (
            operation_type,
            operand_count,
            result_count,
            represented_attributes,
            represented_bases,
        )
        invocation = self._invocations.get(key)
        if invocation is None:
            invocation = OperationInvocation._from_eager(
                operation_type,
                operand_count,
                result_count,
                {} if attributes is None else attributes,
                represented_bases,
            )
            self._invocations[key] = invocation
        return invocation, key

    def _prepare_call(
        self,
        invocation: OperationInvocation,
        *,
        implementation: str | None,
        in_place: bool,
    ) -> tuple[OperationImplementation, tuple[BoundResource, ...]]:
        """Resolve the implementation and resources for one uncached call."""

        selected = self.backend.registry.resolve_type(
            invocation.operation_type,
            requested=implementation,
            in_place=in_place,
        )
        requirements = selected.resource_requirements(invocation)
        available_symbols = set(self.backend.named_resources.symbols)
        missing = tuple(
            requirement
            for requirement in requirements
            if requirement.symbol not in available_symbols
        )
        if missing:
            self.backend = self.backend.with_named_resources(
                self.device_resources.materialize(missing)
            )
        effective = self.backend.named_resources
        required = effective.resolve_all(requirements)
        return selected, required

    def execute(
        self,
        operation_type: type[Operation],
        *inputs: torch.Tensor,
        attributes: Mapping[str, object] | None = None,
        bases: Sequence[str | None] = (),
        prime_ids: tuple[int, ...] = (),
        result_count: int = 1,
        implementation: str | None = None,
        in_place: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute one registered operation through this device's Backend."""

        invocation, invocation_key = self._invocation(
            operation_type,
            len(inputs),
            result_count,
            attributes,
            bases,
        )
        key = (invocation_key, implementation, in_place)
        prepared = self._direct_calls.get(key)
        if prepared is None:
            prepared = self._prepare_call(
                invocation,
                implementation=implementation,
                in_place=in_place,
            )
            self._direct_calls[key] = prepared
        selected, prepared_resources = prepared
        if operation_type in {ckks.EncryptOp, ckks.DecryptOp}:
            context = self.rns_context
            depth = int(cast(int, invocation.attributes["depth"]))
            ids = context.rns_layout.prime_ids(
                depth, include_p=invocation.attributes["modulus_basis"] == "QP"
            )
            tensors = {"parameters": context.rns_parameters_for_prime_ids(ids)}
            tensors.update(
                {
                    f"forward_{i}": value
                    for i, value in enumerate(
                        self.ntt_context.tensor_operands(ids, inverse=False)
                    )
                }
            )
            tensors.update(
                {
                    f"inverse_{i}": value
                    for i, value in enumerate(
                        self.ntt_context.tensor_operands(ids, inverse=True)
                    )
                }
            )
            if operation_type is ckks.DecryptOp:
                tensors.update(
                    zip(
                        (
                            "reconstruction_parameters",
                            "normalizers",
                            "propagation",
                            "half_digits",
                        ),
                        self.device_resources.reconstruction_operands(depth),
                        strict=True,
                    )
                )
            inputs = (*inputs, *tensors.values())
            invocation = OperationInvocation._from_eager(
                operation_type,
                len(inputs),
                result_count,
                {
                    **invocation.attributes,
                    "parameter_names": tuple(tensors),
                    "key_row_start": ids[0],
                    "ntt_backend": self.ntt_context.ntt_backend_name,
                    "min_modulus": min(context.config.moduli[i] for i in ids),
                },
            )
            if operation_type is ckks.DecryptOp:
                prepared_resources = ()
        elif operation_type is ckks.IntegerCoefficientsToRnsOp:
            context = self.rns_context
            ids = context.rns_layout.prime_ids(
                int(cast(int, invocation.attributes["depth"])),
                include_p=invocation.attributes["modulus_basis"] == "QP",
            )
            inputs = (*inputs, context.rns_parameters_for_prime_ids(ids)[0])
            invocation = OperationInvocation._from_eager(
                operation_type,
                len(inputs),
                result_count,
                {
                    **invocation.attributes,
                    "min_modulus": min(context.config.moduli[i] for i in ids),
                },
            )
            prepared_resources = ()
        elif operation_type is ckks.PrepareCompressedPlaintextOp:
            tables = self.device_resources.periodic_encode_operands(
                inputs[0].size(-1),
                int(cast(int, invocation.attributes["depth"])),
                str(invocation.attributes["modulus_basis"]),
            )
            inputs = (*inputs, *tables)
            prepared_resources = ()
        elif operation_type in {ckks.EncodeOp, ckks.DecodeOp}:
            tables = (
                self.device_resources.encode_operands()
                if operation_type is ckks.EncodeOp
                else self.device_resources.decode_operands()
            )
            inputs = (*inputs, *tables)
            prepared_resources = ()
        elif operation_type in {
            ckks.AddScalarOp,
            ckks.MultiplyScalarOp,
            ckks.MultiplyIntegerScalarOp,
        }:
            from fhelium.backend.ckks.scalar import scalar_operands

            context = self.rns_context
            integer = operation_type is ckks.MultiplyIntegerScalarOp
            tensors, facts = scalar_operands(
                context,
                prime_ids,
                cast(int | float, invocation.attributes["scalar"]),
                None
                if integer
                else cast(float, invocation.attributes["scalar_scale"]),
                None if integer else self.rng.rounding_state,
            )
            inputs = (*inputs, *tensors)
            invocation = OperationInvocation._from_eager(
                operation_type,
                len(inputs),
                result_count,
                {**invocation.attributes, **facts},
            )
            prepared_resources = ()
        elif operation_type is rns.RescaleDropLeadingPrimesOp:
            tensors, facts = self.device_resources.rescale_operands(
                inputs[0].size(-2),
                cast(int, invocation.attributes.get("drop_count", 1)),
                include_p=bool(bases and bases[0] == "QP"),
                input_domain=str(
                    invocation.attributes.get("input_domain", "coefficient")
                ),
            )
            invocation = OperationInvocation._from_eager(
                operation_type,
                len(inputs) + len(tensors),
                result_count,
                {
                    **invocation.attributes,
                    **facts,
                    "parameter_names": tuple(tensors),
                },
            )
            inputs = (*inputs, *tensors.values())
            prepared_resources = ()
        elif operation_type in {
            ckks.RotateManyOp,
            ckks.GroupedRotationWeightedSumOp,
            ckks.SwitchKeyOp,
            ckks.ConjugateOp,
            ckks.RelinearizeOp,
            ckks.RotateOp,
        }:
            depth = self.rns_context.rns_layout.depth_for_active_row_count(
                inputs[0].size(-2), include_p=False
            )
            tensors, facts = self.device_resources.key_switch_operands(depth)
            inputs = (*inputs, *tensors.values())
            invocation = OperationInvocation._from_eager(
                operation_type,
                len(inputs),
                result_count,
                {
                    **invocation.attributes,
                    **facts,
                    "parameter_names": tuple(tensors),
                },
            )
            prepared_resources = ()
        elif operation_type in {
            ckks.AddCompressedPlaintextOp,
            ckks.MultiplyCompressedPlaintextOp,
            rns.AddStandardOp,
            rns.SubtractStandardOp,
            rns.NegateStandardOp,
            rns.SumStandardBatchOp,
            rns.AddPlaintextOp,
            rns.MultiplyPlaintextOp,
            rns.MontgomeryMultiplyOp,
            rns.MontgomeryWeightedSumOp,
            rns.MontgomeryWeightedSumsOp,
            rns.AddMontgomeryLazyOp,
            rns.StandardToMontgomeryOp,
            rns.MontgomeryToStandardOp,
            ckks.MultiplyOp,
        }:
            context = self.rns_context
            parameters = context.rns_parameters_for_prime_ids(prime_ids)
            inputs = (*inputs, parameters)
            prepared_resources = ()
        elif operation_type in {
            ntt.CoefficientStandardToNttMontgomeryOp,
            ntt.CoefficientMontgomeryToNttMontgomeryOp,
            ntt.NttMontgomeryToCoefficientStandardOp,
            ntt.NttMontgomeryToCoefficientMontgomeryOp,
        }:
            context = self.ntt_context
            tables = context.tensor_operands(
                prime_ids,
                inverse=operation_type
                in {
                    ntt.NttMontgomeryToCoefficientStandardOp,
                    ntt.NttMontgomeryToCoefficientMontgomeryOp,
                },
            )
            invocation = OperationInvocation._from_eager(
                operation_type,
                len(inputs) + len(tables),
                result_count,
                {
                    **invocation.attributes,
                    "ntt_backend": context.ntt_backend_name,
                },
            )
            inputs = (*inputs, *tables)
            prepared_resources = ()
        outputs = selected.execute(
            invocation,
            inputs,
            prepared_resources,
            in_place=in_place,
        )
        return outputs[0] if result_count == 1 else outputs


__all__ = ["_EagerOperationDispatcher"]
