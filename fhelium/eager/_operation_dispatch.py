"""Select and cache Backend calls for graph-free Eager operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.ckks import (
    DECRYPT_RECONSTRUCTION_RESOURCE_KIND,
    CkksDeviceResources,
    DecryptReconstructionResource,
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
    ResourceBindings,
    ResourceRequirement,
)
from fhelium.backend.rns.context import RnsContext
from fhelium.rng import Csprng


def _resource_identity(
    resources: Sequence[BoundResource],
) -> tuple[int, ...]:
    return tuple(id(resource) for resource in resources)


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
        self.evaluation_key_resources: dict[str, tuple[int, BoundResource]] = {}
        self.operation_key_resources: dict[str, tuple[int, BoundResource]] = {}
        self.random_lock = RLock()
        self._fixed_resources: dict[
            tuple[tuple[str, str], ...], tuple[BoundResource, ...]
        ] = {}
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
        operand_resources: tuple[BoundResource, ...],
        bindings: ResourceBindings | None,
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
        if bindings is not None:
            available_symbols.update(bindings.symbols)
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
        if bindings is not None:
            effective = effective.overlay(bindings, override=True)
        required = effective.resolve_all(requirements)
        return selected, (*operand_resources, *required)

    def discard_resources(self, resources: Sequence[BoundResource]) -> None:
        """Discard direct-dispatch cache entries that reference replaced resources."""

        stale = {id(resource) for resource in resources}
        if not stale:
            return
        self._direct_calls = {
            key: prepared
            for key, prepared in self._direct_calls.items()
            if not any(id(resource) in stale for resource in prepared[1])
        }

    def resources(
        self,
        requirements: Sequence[tuple[str, str]],
    ) -> tuple[BoundResource, ...]:
        """Resolve fixed device resources once by symbol and kind."""

        identity = tuple(requirements)
        selected = self._fixed_resources.get(identity)
        if selected is None:
            created = self.device_resources.materialize(
                tuple(
                    ResourceRequirement(symbol, kind)
                    for symbol, kind in identity
                )
            )
            self.backend = self.backend.with_named_resources(created)
            selected = self.backend.named_resources.resolve_all(
                tuple(
                    ResourceRequirement(symbol, kind)
                    for symbol, kind in identity
                )
            )
            self._fixed_resources[identity] = selected
        return selected

    def execute(
        self,
        operation_type: type[Operation],
        *inputs: torch.Tensor,
        resources: Sequence[BoundResource] = (),
        bindings: ResourceBindings | None = None,
        attributes: Mapping[str, object] | None = None,
        bases: Sequence[str | None] = (),
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
        operand_resources = tuple(resources)
        binding_resources = () if bindings is None else bindings.resources
        key = (
            invocation_key,
            implementation,
            in_place,
            _resource_identity(operand_resources),
            _resource_identity(binding_resources),
        )
        prepared = self._direct_calls.get(key)
        if prepared is None:
            prepared = self._prepare_call(
                invocation,
                operand_resources,
                bindings,
                implementation=implementation,
                in_place=in_place,
            )
            self._direct_calls[key] = prepared
        selected, prepared_resources = prepared
        outputs = selected.execute(
            invocation,
            inputs,
            prepared_resources,
            in_place=in_place,
        )
        return outputs[0] if result_count == 1 else outputs

    def decrypt_reconstruction(self) -> DecryptReconstructionResource:
        resource = self.resources(
            (
                (
                    "ckks-decrypt-reconstruction",
                    DECRYPT_RECONSTRUCTION_RESOURCE_KIND,
                ),
            )
        )[0]
        return cast(DecryptReconstructionResource, resource.value)


__all__ = ["_EagerOperationDispatcher"]
