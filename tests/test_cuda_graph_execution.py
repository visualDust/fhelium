from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest
import torch

from fhelium import CkksEngine, Preset
from fhelium.errors import CudaGraphInputError, ExecutionError
from fhelium.execution import CudaGraphProgram, pin_value_tree


def _message(seed: int, size: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(size, generator=generator, dtype=torch.float64) * 0.002


@pytest.fixture(scope="module")
def engine() -> Iterator[CkksEngine]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    instance = CkksEngine(
        Preset.slots8192_scale40_levels7_int64, device="cuda:0"
    )
    yield instance
    del instance
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_cuda_graph_replays_dynamic_exact_values_at_fixed_addresses(
    engine: CkksEngine,
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

    program = CudaGraphProgram.capture(
        add_pair,
        example_inputs=(prototype,),
        warmup=2,
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
        engine.decrypt_message(first, is_real=True)[:32],
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
        engine.decrypt_message(owned, is_real=True)[:32],
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


@pytest.mark.gpu
def test_cuda_graph_orders_pinned_transfer_before_prepared_replay(
    engine: CkksEngine,
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
        engine.decrypt_message(result, is_real=True)[:32],
        message * 2,
        atol=1e-5,
        rtol=0,
    )
    assert program.input_nbytes == prototype.nbytes

    with pytest.raises(CudaGraphInputError, match="latest CopyHandle"):
        program.replay_prepared(copy_handle=copied)
    program.close()
