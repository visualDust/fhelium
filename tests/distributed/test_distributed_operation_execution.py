"""Behavior coverage for synchronous distributed operation implementations."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import cast

import pytest
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
from fhelium.distributed import all_reduce_ciphertext, reduce_ciphertext
from fhelium.eager import Engine
from fhelium.backend.execution import OperationBackend
from fhelium.backend.distributed import (
    TorchBroadcastImplementation,
    TorchGenericAllReduceImplementation,
    ProcessGroupExecutionResource,
    TorchProcessGroupQueryImplementation,
    distributed_operation_contributions,
)
from fhelium.backend.lowering import LoweredCall
from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    OperationInvocation,
)
from fhelium.backend.resources import (
    BoundResource,
    ResourceBindings,
    ResourceRequirement,
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

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: object,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, resources, builder
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
        represented_rank = LoweredCall(
            query, _invocation(distributed.RankOp, 0, 1)
        )((), resources)[0]
        represented_size = LoweredCall(
            query, _invocation(distributed.GroupSizeOp, 0, 1)
        )((), resources)[0]

        broadcast = LoweredCall(
            TorchBroadcastImplementation(),
            _invocation(distributed.BroadcastOp, 1, 1, root=0),
        )((torch.tensor([rank + 11], dtype=torch.int64),), resources)[0]

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

        Path(output_directory, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": represented_rank.item(),
                    "size": represented_size.item(),
                    "broadcast": broadcast.item(),
                    "executable_generic": executable_generic.item(),
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
            "executable_generic": 3,
        },
        {
            "rank": 1,
            "size": 2,
            "broadcast": 11,
            "executable_generic": 3,
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
        config = CkksConfig.parse(Preset.slots8192_scale40_depth7)
        rns_context = RnsContext(config, device="cpu")
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
            attributes={"ckks_config": StringAttr(json.dumps(config.dumps()))},
        )
        block.add_ops((group, reduction, ReturnOp(reduction.result)))
        program = Program.from_function(block, (value_type,))
        backend = OperationBackend(
            OperationImplementationRegistry(
                distributed_operation_contributions()
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
        result = backend.link(Compilation(program)).run(source_value)
        assert isinstance(result, Ciphertext)
        backing = torch.full(
            (2, config.num_q_primes + 1, config.N), -1, dtype=source.dtype
        )
        view = backing[..., 1:, :]
        view.copy_(source)
        strided_value = source_value.with_data(view)
        all_reduce_ciphertext(strided_value, engine=Engine(config))
        assert strided_value.data is view
        assert bool(torch.all(view == 3))
        assert bool(torch.all(backing[..., 0, :] == -1))

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


def test_ciphertext_all_reduce_program_and_strided_eager_value(
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


def _compiled_reduction_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_directory: str,
) -> None:
    from fhelium import compile as fc
    from fhelium import distributed as fd

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    config = CkksConfig.catalog_suffix_chain(
        logN=8,
        scale_bits=40,
        depth=2,
        p_count=1,
        catalog_logN=14,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=761)
    secret = engine.create_secret_key(device="cpu")
    public = engine.create_public_key(secret, device="cpu")
    clear = torch.full(
        (config.num_slots,), 0.01 * (rank + 1), dtype=torch.float64
    )
    value = engine.encrypt_message(clear, public, device="cpu")
    group = cast(
        torch.distributed.ProcessGroup, torch.distributed.new_group([0, 1])
    )

    def source(current):
        doubled = engine.add(current, current)
        all_reduce_ciphertext(doubled, engine=engine)
        return doubled

    def grouped(current):
        doubled = engine.add(current, current)
        all_reduce_ciphertext(doubled, engine=engine, group=group)
        return doubled

    expected = source(value.clone())
    compiled = fc.compile(source, cache=False, graph=False)
    same = all(
        torch.equal(compiled(value).data, expected.data) for _ in range(2)
    )
    # A second callable loads the stored specialization of the subgroup
    # reduction and binds the recorded group.
    cache = Path(output_directory, f"cache-{rank}")
    stored = fc.compile(grouped, cache=cache, graph=False)
    loaded = fc.compile(grouped, cache=cache, graph=False)
    same = same and torch.equal(stored(value).data, expected.data)
    same = same and torch.equal(loaded(value).data, expected.data)
    hit = cast(list[dict[str, object]], loaded.stats()["specializations"])[0][
        "cache"
    ]
    # Shutdown releases the groups of live compiled callables, which bind the
    # new default group at their next execution.
    fd.shutdown()
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}-again",
        rank=rank,
        world_size=world_size,
    )
    try:
        same = same and torch.equal(compiled(value).data, expected.data)
        Path(output_directory, f"compiled-rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "equal": same, "cache": hit}),
            encoding="utf-8",
        )
    finally:
        fd.shutdown()


def test_compiled_ciphertext_reduction_matches_eager(tmp_path: Path) -> None:
    rendezvous = tmp_path / "compiled-rendezvous"
    outputs = tmp_path / "compiled-outputs"
    outputs.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_compiled_reduction_worker,
            args=(rank, 2, str(rendezvous), str(outputs)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    assert [
        json.loads(
            (outputs / f"compiled-rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    ] == [
        {"rank": 0, "equal": True, "cache": "hit"},
        {"rank": 1, "equal": True, "cache": "hit"},
    ]


def _modular_reduction_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_directory: str,
) -> None:
    from fhelium import compile as fc

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # 30-bit primes with int32 residues: four ranks exceed the lazy-range
        # sum bound and five ranks exceed the reduced-range bound.
        engine = Engine(
            CkksConfig.parse(Preset.slots8192_scale30_depth9), rng_seed=5
        )
        secret = engine.create_secret_key(device="cpu")
        public = engine.create_public_key(secret, device="cpu")
        clear = torch.linspace(
            -0.5, 0.5, engine.config.num_slots, dtype=torch.float64
        )
        value = engine.encrypt_message(clear * (rank + 1), public, device="cpu")

        def summed(current):
            all_reduce_ciphertext(current, engine=engine)
            return current

        def reduced(current):
            reduce_ciphertext(current, dst=1, engine=engine)
            return current

        expected = summed(value.clone())
        expected_dst = reduced(value.clone())
        all_equal = torch.equal(
            fc.compile(summed, cache=False, graph=False)(value).data,
            expected.data,
        )
        compiled_dst = fc.compile(reduced, cache=False, graph=False)(value)
        dst_equal = rank != 1 or torch.equal(
            compiled_dst.data, expected_dst.data
        )
        Path(output_directory, f"rank-{rank}.json").write_text(
            json.dumps({"all": all_equal, "dst": dst_equal}), encoding="utf-8"
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [4, 5])
def test_compiled_modular_reduction_matches_eager_beyond_sum_bounds(
    tmp_path: Path, world_size: int
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_modular_reduction_worker,
            args=(rank, world_size, str(tmp_path / "rendezvous"), str(outputs)),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
        assert process.exitcode == 0
    assert [
        json.loads((outputs / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ] == [{"all": True, "dst": True}] * world_size
