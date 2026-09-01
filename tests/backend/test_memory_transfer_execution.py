"""Behavior tests for rank-local memory-transfer execution."""

from __future__ import annotations

import pytest
import torch
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium.backend.memory import (
    DEVICE_RESOURCE_KIND,
    TorchMemoryTransferImplementation,
)
from fhelium.backend.execution import OperationBackend, ProgramDispatchTable
from fhelium.backend.implementation import OperationImplementationRegistry
from fhelium.backend.resources import (
    BoundResource,
    ResourceBindings,
    ResourceRequirement,
)
from fhelium.compile.passes.backend import (
    ResolveBackendOperationsPass,
)
from fhelium.compile import Compilation
from fhelium.compile import Pipeline
from fhelium.ir import Program
from fhelium.ir.dialects import core, memory


def _transfer_program(*, memory_space: str = "default") -> Program:
    block = Block(arg_types=(core.MessageType(),))
    target = core.ResourceRefOp(
        memory.DeviceType(),
        symbol="target-device",
        kind=DEVICE_RESOURCE_KIND,
    )
    transfer = memory.TransferOp(
        block.args[0],
        target,
        memory_space=memory_space,
    )
    block.add_ops((target, transfer, ReturnOp(transfer.result)))
    return Program.from_function(block, (transfer.result.type,))


def _backend(target: torch.device | str) -> OperationBackend:
    target_resource = BoundResource(
        "target-device",
        DEVICE_RESOURCE_KIND,
        torch.device(target),
    )
    return OperationBackend(
        OperationImplementationRegistry((TorchMemoryTransferImplementation(),)),
        named_resources=ResourceBindings((target_resource,)),
    )


def _build(
    target: torch.device | str,
    program: Program,
):
    backend = _backend(target)
    return backend.link(Compilation(program))


def test_same_device_transfer_may_alias_the_input() -> None:
    value = torch.arange(8, dtype=torch.int64)

    result = _build("cpu", _transfer_program()).run(value)

    assert result is value


def test_pageable_host_transfer_preserves_values() -> None:
    value = torch.arange(8, dtype=torch.float64)

    result = _build("cpu", _transfer_program(memory_space="pageable_host")).run(
        value
    )

    assert isinstance(result, torch.Tensor)
    assert result.device == torch.device("cpu")
    assert not result.is_pinned()
    torch.testing.assert_close(result, value)


def test_transfer_executes_inside_rank_parameterized_scf_loop() -> None:
    block = Block(arg_types=(core.MessageType(), IndexType()))
    target = core.ResourceRefOp(
        memory.DeviceType(),
        symbol="target-device",
        kind=DEVICE_RESOURCE_KIND,
    )
    one = arith.ConstantOp.from_int_and_width(1, IndexType())
    two = arith.ConstantOp.from_int_and_width(2, IndexType())
    upper = arith.AddiOp(block.args[1], two)
    body = Block(arg_types=(IndexType(), core.MessageType()))
    transfer = memory.TransferOp(body.args[1], target)
    body.add_ops((transfer, scf.YieldOp(transfer.result)))
    loop = scf.ForOp(
        block.args[1],
        upper,
        one,
        (block.args[0],),
        body,
    )
    block.add_ops((target, one, two, upper, loop, ReturnOp(loop.res[0])))
    program = Program.from_function(block, (loop.res[0].type,))
    value = torch.arange(8, dtype=torch.int64)

    result = _build("cpu", program).run(value, 3)

    assert result is value


def test_backend_resolves_supported_operations_before_resource_linking() -> (
    None
):
    program = _transfer_program()
    backend = _backend("cpu")

    result = Pipeline((ResolveBackendOperationsPass(backend.registry),)).run(
        Compilation(program)
    )
    dispatch_table = result.workspace[ProgramDispatchTable]

    assert isinstance(dispatch_table, ProgramDispatchTable)
    assert [
        dispatch.implementation.name
        for dispatch in dispatch_table.operations.values()
    ] == ["torch-memory-transfer"]


def test_resource_linking_rejects_an_unbound_program_placeholder() -> None:
    backend = OperationBackend(
        OperationImplementationRegistry((TorchMemoryTransferImplementation(),)),
    )

    with pytest.raises(KeyError, match="target-device"):
        backend.link(Compilation(_transfer_program()))


def test_backend_resolution_rejects_an_unsupported_operation() -> None:
    backend = OperationBackend(OperationImplementationRegistry())

    with pytest.raises(ValueError, match="0 available implementations"):
        backend.link(Compilation(_transfer_program()))


def test_backend_materializes_shared_resources_once_for_the_whole_program() -> (
    None
):
    block = Block(arg_types=(core.MessageType(),))
    target = core.ResourceRefOp(
        memory.DeviceType(),
        symbol="target-device",
        kind=DEVICE_RESOURCE_KIND,
    )
    first = memory.TransferOp(block.args[0], target)
    second = memory.TransferOp(first.result, target)
    block.add_ops((target, first, second, ReturnOp(second.result)))
    program = Program.from_function(block, (second.result.type,))

    class Materializer:
        def __init__(self) -> None:
            self.calls = 0

        def materialize(
            self,
            requirements: tuple[ResourceRequirement, ...],
        ) -> ResourceBindings:
            self.calls += 1
            assert requirements == (
                ResourceRequirement("target-device", DEVICE_RESOURCE_KIND),
            )
            return ResourceBindings(
                (
                    BoundResource(
                        "target-device",
                        DEVICE_RESOURCE_KIND,
                        torch.device("cpu"),
                    ),
                )
            )

    materializer = Materializer()
    backend = OperationBackend(
        OperationImplementationRegistry((TorchMemoryTransferImplementation(),)),
        materializer=materializer,
    )
    value = torch.arange(8, dtype=torch.int64)

    result = backend.link(Compilation(program)).run(value)

    assert result is value
    assert materializer.calls == 1


@pytest.mark.gpu
def test_cpu_cuda_round_trip_uses_caller_bound_devices() -> None:
    value = torch.arange(8, dtype=torch.float64)
    cuda = _build("cuda:0", _transfer_program()).run(value)
    assert isinstance(cuda, torch.Tensor)
    restored = _build("cpu", _transfer_program()).run(cuda)

    assert isinstance(restored, torch.Tensor)
    assert cuda.device == torch.device("cuda:0")
    assert restored.device == torch.device("cpu")
    torch.testing.assert_close(restored, value)


@pytest.mark.gpu
def test_pinned_host_transfer_allocates_pinned_storage() -> None:
    value = torch.arange(8, dtype=torch.float64, device="cuda:0")

    result = _build("cpu", _transfer_program(memory_space="pinned_host")).run(
        value
    )

    assert isinstance(result, torch.Tensor)
    assert result.device == torch.device("cpu")
    assert result.is_pinned()
    torch.testing.assert_close(result, value.cpu())
