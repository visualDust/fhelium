"""Synchronous rank-local implementations of distributed IR operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.execution import OperationBackend
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import distributed, rns

from .resources import ProcessGroupExecutionResource

TensorRegion = Callable[
    [tuple[torch.Tensor, ...]],
    tuple[torch.Tensor, ...],
]
"""Execute one already-compiled region over Tensor payloads."""

TensorCombine = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Combine two rank-local Tensor payloads into a functional result."""


def prepare_ciphertext_add_combine(
    backend: OperationBackend,
    rns_resource: BoundResource,
    *,
    modulus_basis: str = "Q",
    implementation: str | None = None,
) -> TensorCombine:
    """Bind ciphertext addition through an existing operation backend.

    The returned callable owns no Engine or public CKKS value.  It reuses the
    execution owner's selected RNS implementation and concrete arithmetic
    resource for every local combine performed by a distributed reduction.
    """

    invocation = OperationInvocation(
        rns.AddStandardOp,
        2,
        1,
        operand_bases=(modulus_basis, modulus_basis),
    )
    selected = backend.registry.resolve_type(
        invocation.operation_type,
        requested=implementation,
        in_place=False,
    )
    required = backend.named_resources.resolve_all(
        selected.resource_requirements(invocation)
    )
    resources = (rns_resource, *required)

    def combine(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return selected.execute(
            invocation,
            (lhs, rhs),
            resources,
            in_place=False,
        )[0]

    return combine


def _group(
    resources: tuple[BoundResource, ...],
) -> ProcessGroupExecutionResource:
    if not resources:
        raise ValueError(
            "Distributed operation requires a process-group resource"
        )
    return cast(ProcessGroupExecutionResource, resources[0].value)


def _collective_fold(
    value: torch.Tensor,
    group: ProcessGroupExecutionResource,
    combine: TensorCombine,
) -> torch.Tensor:
    """All-gather and fold equal-layout payloads in process-group rank order."""

    if group.size == 1:
        return value.clone()
    gathered = [torch.empty_like(value) for _ in range(group.size)]
    torch.distributed.all_gather(gathered, value, group=group.group)
    result = gathered[0]
    for operand in gathered[1:]:
        result = combine(result, operand)
    return result


class _OperandResourceImplementation:
    """Declare that the process group arrives as an IR resource operand."""

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()


@dataclass(frozen=True)
class TorchProcessGroupQueryImplementation(_OperandResourceImplementation):
    """Read group rank and size as scalar index Tensors."""

    name: str = "torch-process-group-query"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        distributed.RankOp,
        distributed.GroupSizeOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del inputs, in_place
        group = _group(resources)
        value = (
            group.rank
            if invocation.operation_type is distributed.RankOp
            else group.size
        )
        return (torch.tensor(value, dtype=torch.int64),)


@dataclass(frozen=True)
class TorchBroadcastImplementation(_OperandResourceImplementation):
    """Functionally broadcast one equal-layout Tensor from a group rank."""

    name: str = "torch-broadcast"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (distributed.BroadcastOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        group = _group(resources)
        root = cast(int, invocation.attributes["root"])
        output = inputs[0].clone()
        torch.distributed.broadcast(
            output,
            src=group.global_rank(root),
            group=group.group,
        )
        return (output,)


@dataclass(frozen=True)
class TorchGenericAllReduceImplementation(_OperandResourceImplementation):
    """Fold all rank-local payloads with the operation's combine region."""

    name: str = "torch-generic-all-reduce"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (distributed.AllReduceOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, inputs, resources, in_place
        raise RuntimeError(
            "fhelium_dist.all_reduce requires a region-capable executable"
        )

    def execute_regions(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        regions: tuple[TensorRegion, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, in_place
        if len(regions) != 1:
            raise ValueError("Generic all-reduce requires one combine region")
        combine_region = regions[0]

        def combine(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            outputs = combine_region((lhs, rhs))
            if len(outputs) != 1:
                raise ValueError(
                    "Generic all-reduce combine region returned "
                    f"{len(outputs)} values"
                )
            return outputs[0]

        return (_collective_fold(inputs[0], _group(resources), combine),)


@dataclass(frozen=True)
class TorchCiphertextAddAllReduceImplementation(_OperandResourceImplementation):
    """Reduce ciphertext payloads with an injected Backend Tensor addition."""

    combine: TensorCombine = field(repr=False, compare=False)
    name: str = "torch-ciphertext-add-all-reduce"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        distributed.AllReduceAddCiphertextOp,
    )

    def __post_init__(self) -> None:
        if not callable(self.combine):
            raise TypeError("ciphertext all-reduce combine must be callable")

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, in_place
        return (_collective_fold(inputs[0], _group(resources), self.combine),)


def distributed_operation_contributions(
    *,
    ciphertext_add: TensorCombine | None = None,
) -> tuple[
    TorchProcessGroupQueryImplementation
    | TorchBroadcastImplementation
    | TorchGenericAllReduceImplementation
    | TorchCiphertextAddAllReduceImplementation,
    ...,
]:
    """Return distributed implementations for one execution owner.

    The specialized ciphertext-add implementation is contributed only when an
    execution owner supplies addition through its existing Tensor Backend.  A
    Compile pass may instead lower the specialized operation to generic
    all-reduce with a visible combine region.
    """

    implementations: list[
        TorchProcessGroupQueryImplementation
        | TorchBroadcastImplementation
        | TorchGenericAllReduceImplementation
        | TorchCiphertextAddAllReduceImplementation
    ] = [
        TorchProcessGroupQueryImplementation(),
        TorchBroadcastImplementation(),
        TorchGenericAllReduceImplementation(),
    ]
    if ciphertext_add is not None:
        implementations.append(
            TorchCiphertextAddAllReduceImplementation(ciphertext_add)
        )
    return tuple(implementations)


__all__ = [
    "TorchBroadcastImplementation",
    "TorchCiphertextAddAllReduceImplementation",
    "TorchGenericAllReduceImplementation",
    "TorchProcessGroupQueryImplementation",
    "TensorCombine",
    "TensorRegion",
    "distributed_operation_contributions",
    "prepare_ciphertext_add_combine",
]
