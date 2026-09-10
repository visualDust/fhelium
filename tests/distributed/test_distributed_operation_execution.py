"""Behavior coverage for synchronous distributed operation implementations."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import cast

import torch
from xdsl.dialects.func import ReturnOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.ir import Block, Operation

from fhelium import Preset
from fhelium.backend.execution import OperationBackend
from fhelium.backend.distributed import (
    TorchBroadcastImplementation,
    TorchCiphertextAddAllReduceImplementation,
    TorchGenericAllReduceImplementation,
    ProcessGroupExecutionResource,
    TorchProcessGroupQueryImplementation,
    distributed_operation_contributions,
    prepare_ciphertext_add_combine,
)
from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    OperationInvocation,
)
from fhelium.backend.resources import (
    BoundResource,
    ResourceBindings,
    ResourceRequirement,
)
from fhelium.backend.rns.operations import (
    NativeRnsLinearImplementation,
)
from fhelium.backend.rns.context import RnsContext
from fhelium.config import CkksConfig
from fhelium.compile import Compilation
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core, distributed, semantic
from fhelium.values import Ciphertext


class _TensorAddImplementation:
    name = "test-tensor-add"
    supports_in_place = False
    operation_types: tuple[type[Operation], ...] = (semantic.AddOp,)

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


def _generic_all_reduce_program() -> Program:
    value_type = semantic.PublicType()
    body = Block(arg_types=(value_type,))
    group = core.ResourceRefOp(
        distributed.GroupType(),
        symbol="world",
        kind="process-group",
    )
    combine = Block(arg_types=(value_type, value_type))
    addition = semantic.AddOp(combine.args[0], combine.args[1])
    combine.add_ops((addition, distributed.YieldOp(addition)))
    reduction = distributed.AllReduceOp(body.args[0], group, combine)
    body.add_ops((group, reduction, ReturnOp(reduction)))
    return Program.from_function(body, (value_type,))


def _invocation(
    operation_type: type,
    operand_count: int,
    result_count: int,
    **attributes: object,
) -> OperationInvocation:
    return OperationInvocation(
        operation_type,
        operand_count,
        result_count,
        attributes,
    )


def _build(backend: OperationBackend, program: Program):
    return backend.link(Compilation(program))


def _gloo_operation_worker(
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
        resource = ProcessGroupExecutionResource()
        bound = BoundResource(
            "world",
            "process-group",
            resource,
        )
        resources = (bound,)
        query = TorchProcessGroupQueryImplementation()
        represented_rank = query.execute(
            _invocation(distributed.RankOp, 0, 1),
            (),
            resources,
            in_place=False,
        )[0]
        represented_size = query.execute(
            _invocation(distributed.GroupSizeOp, 0, 1),
            (),
            resources,
            in_place=False,
        )[0]

        broadcast = TorchBroadcastImplementation().execute(
            _invocation(distributed.BroadcastOp, 1, 1, root=0),
            (torch.tensor([rank + 11], dtype=torch.int64),),
            resources,
            in_place=False,
        )[0]

        generic = TorchGenericAllReduceImplementation().execute_regions(
            _invocation(distributed.AllReduceOp, 1, 1),
            (torch.tensor([rank + 1], dtype=torch.int64),),
            resources,
            (lambda values: (values[0] + values[1],),),
            in_place=False,
        )[0]

        backend = OperationBackend(
            OperationImplementationRegistry(
                (
                    TorchGenericAllReduceImplementation(),
                    _TensorAddImplementation(),
                )
            ),
            named_resources=ResourceBindings((bound,)),
        )
        executable_generic = cast(
            torch.Tensor,
            _build(backend, _generic_all_reduce_program()).run(
                torch.tensor([rank + 1], dtype=torch.int64)
            ),
        )

        specialized = TorchCiphertextAddAllReduceImplementation(
            lambda lhs, rhs: (lhs + rhs).remainder(17)
        ).execute(
            _invocation(distributed.AllReduceAddCiphertextOp, 1, 1),
            (torch.tensor([15 if rank == 0 else 5], dtype=torch.int64),),
            resources,
            in_place=False,
        )[0]

        Path(output_directory, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": represented_rank.item(),
                    "size": represented_size.item(),
                    "broadcast": broadcast.item(),
                    "generic": generic.item(),
                    "executable_generic": executable_generic.item(),
                    "specialized": specialized.item(),
                }
            ),
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_gloo_operation_implementations(tmp_path: Path) -> None:
    rendezvous = tmp_path / "gloo-rendezvous"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_gloo_operation_worker,
            args=(rank, 2, str(rendezvous), str(outputs)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    records = [
        json.loads((outputs / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert records == [
        {
            "rank": 0,
            "size": 2,
            "broadcast": 11,
            "generic": 3,
            "executable_generic": 3,
            "specialized": 3,
        },
        {
            "rank": 1,
            "size": 2,
            "broadcast": 11,
            "generic": 3,
            "executable_generic": 3,
            "specialized": 3,
        },
    ]


def _gloo_ckks_reduction_worker(
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
        config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
        rns_context = RnsContext(config, device="cpu")
        rns_resource = BoundResource(
            "rank-local-rns",
            "rns-parameters",
            rns_context,
        )
        arithmetic_backend = OperationBackend(
            OperationImplementationRegistry((NativeRnsLinearImplementation(),)),
        )
        combine = prepare_ciphertext_add_combine(
            arithmetic_backend,
            rns_resource,
        )
        group_resource = BoundResource(
            "world",
            "process-group",
            ProcessGroupExecutionResource(),
        )
        value_type = ckks.CiphertextType().with_state(
            {
                "basis": StringAttr("Q"),
                "components": IntegerAttr(2, 64),
                "depth": IntegerAttr(0, 64),
                "scale": FloatAttr(1.0, Float64Type()),
                "prime_ids": ArrayAttr(
                    IntegerAttr(index, 64)
                    for index in range(config.num_q_primes)
                ),
                "polynomial_domain": StringAttr("coefficient"),
                "residue_representation": StringAttr("standard"),
            }
        )
        block = Block(arg_types=(value_type,))
        group = core.ResourceRefOp(
            distributed.GroupType(),
            symbol="world",
            kind="process-group",
        )
        reduction = distributed.AllReduceAddCiphertextOp(
            block.args[0],
            group,
        )
        block.add_ops((group, reduction, ReturnOp(reduction.result)))
        program = Program.from_function(block, (value_type,))
        backend = OperationBackend(
            OperationImplementationRegistry(
                distributed_operation_contributions(
                    ciphertext_add=combine,
                )
            ),
            named_resources=ResourceBindings((group_resource,)),
        )
        source = torch.full(
            (2, config.num_q_primes, config.N),
            rank + 1,
            dtype=rns_context.dtype,
        )
        source_value = Ciphertext(
            data=source,
            depth=0,
            scale=1.0,
            prime_ids=tuple(range(config.num_q_primes)),
            polynomial_domain="coefficient",
            modulus_basis="Q",
            residue_representation="standard",
        )
        result = _build(backend, program).run(source_value)
        assert isinstance(result, Ciphertext)
        Path(output_directory, f"ckks-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "all_three": bool(torch.all(result.data == 3)),
                }
            ),
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def test_ciphertext_add_collective_reuses_native_rns_backend(
    tmp_path: Path,
) -> None:
    rendezvous = tmp_path / "ckks-rendezvous"
    outputs = tmp_path / "ckks-outputs"
    outputs.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_gloo_ckks_reduction_worker,
            args=(rank, 2, str(rendezvous), str(outputs)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    assert [
        json.loads(
            (outputs / f"ckks-rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    ] == [
        {"rank": 0, "all_three": True},
        {"rank": 1, "all_three": True},
    ]
