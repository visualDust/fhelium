"""End-to-end execution of rank-local structured distributed Programs."""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block, Operation

from fhelium.backend.distributed import (
    ProcessGroupExecutionResource,
    distributed_operation_contributions,
)
from fhelium.backend.execution import OperationBackend, ProgramExecutable
from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    OperationInvocation,
)
from fhelium.backend.resources import (
    BoundResource,
    ResourceBindings,
    ResourceRequirement,
)
from fhelium.compile import Compilation, CompileWorkspace, ConstantBundle
from fhelium.compile import Pipeline
from fhelium.ir import Program
from fhelium.ir.dialects import core, distributed, semantic


@dataclass(frozen=True)
class _TensorAddImplementation:
    name: str = "test-tensor-add"
    operation_types: tuple[type[Operation], ...] = (semantic.AddOp,)
    supports_in_place: bool = False

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, resources, in_place
        return (inputs[0] + inputs[1],)


def _structured_collective_program() -> Program:
    value_type = semantic.PublicType()
    block = Block(arg_types=(value_type,))
    group = core.ResourceRefOp(
        distributed.GroupType(),
        symbol="world",
        kind="process-group",
    )
    rank = distributed.RankOp(group)
    one = arith.ConstantOp.from_int_and_width(1, IndexType())
    upper = arith.AddiOp(rank.result, one)
    body = Block(arg_types=(IndexType(), value_type))
    broadcast = distributed.BroadcastOp(body.args[1], group, root=0)
    body.add_ops((broadcast, scf.YieldOp(broadcast.result)))
    loop = scf.ForOp(
        rank.result,
        upper,
        one,
        (block.args[0],),
        body,
    )
    combine = Block(arg_types=(value_type, value_type))
    add = semantic.AddOp(combine.args[0], combine.args[1])
    combine.add_ops((add, distributed.YieldOp(add.result)))
    reduced = distributed.AllReduceOp(loop.res[0], group, combine)
    block.add_ops(
        (group, rank, one, upper, loop, reduced, ReturnOp(reduced.result))
    )
    return Program.from_function(block, (value_type,))


def _material_add_program() -> Program:
    value_type = semantic.PublicType()
    block = Block(arg_types=(value_type,))
    material = core.MaterialRefOp(
        value_type,
        symbol="model/bias",
        kind="tensor",
    )
    addition = semantic.AddOp(block.args[0], material.value)
    block.add_ops((material, addition, ReturnOp(addition.result)))
    return Program.from_function(block, (value_type,))


def test_compile_constant_is_linked_without_mutating_compile_workspace() -> (
    None
):
    backend = OperationBackend(
        OperationImplementationRegistry((_TensorAddImplementation(),)),
    )
    bias = torch.tensor([2, 4], dtype=torch.int64)
    compilation = Compilation(
        _material_add_program(),
        CompileWorkspace(
            {ConstantBundle: ConstantBundle({"model/bias": bias})}
        ),
    )
    executable = backend.link(compilation)

    result = executable.run(torch.tensor([1, 3], dtype=torch.int64))

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([3, 7]))
    assert ResourceBindings not in compilation.workspace


def test_backend_material_override_replaces_the_compile_constant() -> None:
    compile_bias = torch.tensor([2, 4], dtype=torch.int64)
    backend_bias = torch.tensor([10, 20], dtype=torch.int64)
    backend = OperationBackend(
        OperationImplementationRegistry((_TensorAddImplementation(),)),
        material_overrides={"model/bias": backend_bias},
    )
    compilation = Compilation(
        _material_add_program(),
        CompileWorkspace(
            {ConstantBundle: ConstantBundle({"model/bias": compile_bias})}
        ),
    )

    result = backend.link(compilation).run(
        torch.tensor([1, 3], dtype=torch.int64)
    )

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([11, 23]))


def test_material_linking_rejects_a_missing_symbol_before_execution() -> None:
    backend = OperationBackend(
        OperationImplementationRegistry((_TensorAddImplementation(),)),
    )

    with pytest.raises(KeyError, match="model/bias"):
        backend.link(Compilation(_material_add_program()))


def test_incomplete_linking_pipeline_cannot_return_a_stale_executable() -> None:
    bias = torch.tensor([2, 4], dtype=torch.int64)
    backend = OperationBackend(
        OperationImplementationRegistry((_TensorAddImplementation(),)),
        material_overrides={"model/bias": bias},
    )
    compilation = Compilation(_material_add_program())
    compilation.workspace[ProgramExecutable] = backend.link(compilation)

    with pytest.raises(RuntimeError, match="did not produce"):
        backend.link(compilation, pipeline=Pipeline())


def _structured_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_directory: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        resource = BoundResource(
            "world",
            "process-group",
            ProcessGroupExecutionResource(),
        )
        registry = OperationImplementationRegistry(
            (_TensorAddImplementation(),)
        ).with_implementations(distributed_operation_contributions())
        backend = OperationBackend(
            registry,
            named_resources=ResourceBindings((resource,)),
        )
        executable = backend.link(Compilation(_structured_collective_program()))
        result = executable.run(torch.tensor([rank + 1], dtype=torch.int64))
        assert isinstance(result, torch.Tensor)
        Path(output_directory, f"rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "result": result.item()}),
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def test_rank_query_scf_and_collective_region_execute_end_to_end(
    tmp_path: Path,
) -> None:
    rendezvous = tmp_path / "structured-rendezvous"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_structured_worker,
            args=(rank, 2, str(rendezvous), str(outputs)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert [
        json.loads((outputs / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ] == [
        {"rank": 0, "result": 2},
        {"rank": 1, "result": 2},
    ]
