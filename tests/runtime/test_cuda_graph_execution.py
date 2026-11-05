from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest
import torch

from fhelium import Preset
from fhelium.eager import Engine
from fhelium.errors import CudaGraphInputError, ExecutionError
from fhelium.runtime import CudaGraphPool, CudaGraphProgram, pin_value_tree


def _message(seed: int, size: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            size,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )
        * 0.002
    )


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    previous_device = torch.get_default_device()
    torch.set_default_device("cuda:0")
    instance = Engine(Preset.slots8192_scale40_depth7_int64)
    yield instance
    torch.set_default_device(previous_device)
    del instance
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("shared", [False, True])
def test_cuda_graph_replays_dynamic_exact_values_at_fixed_addresses(
    engine: Engine,
    shared: bool,
) -> None:
    prototype_messages = (_message(1), _message(2))
    prototype = {
        "operands": [
            engine.encrypt_message(prototype_messages[0]),
            engine.encrypt_message(prototype_messages[1]),
        ]
    }

    def add_pair(inputs):
        left, right = inputs["operands"]
        return engine.add(left, right)

    pool = CudaGraphPool("cuda:0") if shared else None
    program = CudaGraphProgram.capture(
        add_pair,
        example_inputs=(prototype,),
        warmup=2,
        pool=pool,
    )
    output_pointer = program.output.data.data_ptr()

    first_messages = (_message(3), _message(4))
    first_inputs = {
        "operands": [
            engine.encrypt_message(first_messages[0]),
            engine.encrypt_message(first_messages[1]),
        ]
    }
    first = program.replay(first_inputs, synchronize=True)
    assert first is program.output
    assert first.data.data_ptr() == output_pointer
    torch.testing.assert_close(
        engine.decrypt_message(
            first,
            engine.secret_key.to("cpu"),
            is_real=True,
            device="cpu",
        )[:32],
        first_messages[0] + first_messages[1],
        atol=1e-5,
        rtol=0,
    )

    second_messages = (_message(5), _message(6))
    second_inputs = {
        "operands": [
            engine.encrypt_message(second_messages[0]),
            engine.encrypt_message(second_messages[1]),
        ]
    }
    owned = program.replay(
        second_inputs,
        copy_output=True,
        synchronize=True,
    )
    assert owned.data.data_ptr() != output_pointer
    torch.testing.assert_close(
        engine.decrypt_message(
            owned,
            engine.secret_key.to("cpu"),
            is_real=True,
            device="cpu",
        )[:32],
        second_messages[0] + second_messages[1],
        atol=1e-5,
        rtol=0,
    )
    assert program.stats.warmup_iterations == 2

    wrong_domain = {
        "operands": [
            engine.coefficient_domain_to_ntt_domain(
                second_inputs["operands"][0]
            ),
            second_inputs["operands"][1],
        ]
    }
    with pytest.raises(CudaGraphInputError, match="value signature"):
        program.replay(wrong_domain)

    program.close()
    with pytest.raises(ExecutionError, match="closed"):
        program.replay(second_inputs)
    if pool is not None:
        pool.close()


@pytest.mark.gpu
def test_cuda_graph_orders_pinned_transfer_before_prepared_replay(
    engine: Engine,
) -> None:
    prototype = engine.encrypt_message(_message(20))
    program = CudaGraphProgram.capture(
        lambda value: engine.add(value, value),
        example_inputs=(prototype,),
        warmup=1,
    )
    output_pointer = program.output.data.data_ptr()
    message = _message(21)
    pinned_source = pin_value_tree(engine.encrypt_message(message).cpu())
    transfer_stream = torch.cuda.Stream(device=program.device)
    copied = program.copy_inputs_from(
        pinned_source,
        stream=transfer_stream,
        non_blocking=True,
    )

    with pytest.raises(CudaGraphInputError, match="latest CopyHandle"):
        program.replay_prepared()

    result = program.replay_prepared(
        copy_handle=copied,
        synchronize=True,
    )
    assert result.data.data_ptr() == output_pointer
    torch.testing.assert_close(
        engine.decrypt_message(
            result,
            engine.secret_key.to("cpu"),
            is_real=True,
            device="cpu",
        )[:32],
        message * 2,
        atol=1e-5,
        rtol=0,
    )
    assert program.input_nbytes == prototype.nbytes

    with pytest.raises(CudaGraphInputError, match="latest CopyHandle"):
        program.replay_prepared(copy_handle=copied)
    program.close()


@pytest.mark.gpu
def test_shared_pool_preserves_outputs_across_programs_and_streams():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda:0")
    x = (
        torch.arange(64 * 64, device=device, dtype=torch.float64).reshape(
            64, 64
        )
        / 4096
    )
    x.requires_grad_(True)
    weight = torch.eye(64, device=device, dtype=torch.float64)

    def project(source, matrix):
        temporary = source @ matrix
        return {
            "result": temporary.square(),
            "view": source[:, ::2],
            "optional": None,
        }

    def reduce(source):
        return (source.sin() + source.cos()).sum(dim=1)

    with CudaGraphPool(device) as pool:
        first = CudaGraphProgram.capture(
            project, example_inputs=(x, weight), warmup=1, pool=pool
        )
        second = CudaGraphProgram.capture(
            reduce, example_inputs=(x,), warmup=1, pool=pool
        )
        for leaf in (
            first.output["result"],
            first.output["view"],
            second.output,
        ):
            assert not leaf.requires_grad
            assert leaf.grad_fn is None
        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)
        producer = torch.cuda.current_stream(device)
        changed = weight * 2
        stream_a.wait_stream(producer)
        stream_b.wait_stream(producer)
        # Replay in a different order, using two streams and changing weights.
        with torch.cuda.stream(stream_b):
            reduced = second.replay(x)
        with torch.cuda.stream(stream_a):
            projected = first.replay(x, changed)
            owned = first.replay(x, weight, copy_output=True)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(reduced, reduce(x))
        torch.testing.assert_close(owned["result"], x.square())
        torch.testing.assert_close(owned["view"], x[:, ::2])
        assert owned["optional"] is None
        assert not owned["result"].requires_grad
        assert owned["result"].grad_fn is None
        retained_reduction = reduced.clone()
        retained_projection = projected["result"].clone()
        second.replay(x * 3, synchronize=True)
        torch.testing.assert_close(projected["result"], retained_projection)
        first.replay(x, changed, synchronize=True)
        torch.testing.assert_close(owned["result"], x.square())
        torch.testing.assert_close(second.output, reduce(x * 3))
        assert not torch.equal(second.output, retained_reduction)
        first.close()
        torch.testing.assert_close(
            second.replay(x, synchronize=True), reduce(x)
        )
    with pytest.raises(ExecutionError, match="closed"):
        second.replay(x)
