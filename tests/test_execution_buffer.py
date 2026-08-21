from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from fhelium import CkksEngine, Plaintext, Preset
from fhelium.errors import ExecutionError, ExecutionInputError
from fhelium.execution import (
    ReusableValueBuffer,
    TensorSignature,
    ValueSignature,
    ValueTreeSignature,
    value_tree_nbytes,
)


def test_reusable_value_buffer_preserves_addresses_and_validates_before_copy() -> (
    None
):
    prototype = {
        "tensor": torch.arange(4, dtype=torch.float64),
        "values": [
            Plaintext(
                message=torch.tensor([1.0, 2.0]),
                level=0,
                scale=16.0,
            )
        ],
    }
    buffer = ReusableValueBuffer.like(prototype, device="cpu")
    initial = buffer.value
    tensor_pointer = initial["tensor"].data_ptr()
    plaintext_pointer = initial["values"][0].message.data_ptr()

    source = {
        "tensor": torch.full((4,), 7.0, dtype=torch.float64),
        "values": [
            Plaintext(
                message=torch.tensor([3.0, 4.0]),
                level=0,
                scale=16.0,
            )
        ],
    }
    copied = buffer.copy_from(source)
    assert copied.done()
    actual = buffer.value
    assert actual["tensor"].data_ptr() == tensor_pointer
    assert actual["values"][0].message.data_ptr() == plaintext_pointer
    torch.testing.assert_close(actual["tensor"], source["tensor"])
    torch.testing.assert_close(
        actual["values"][0].message,
        source["values"][0].message,
    )
    assert buffer.signature == ValueTreeSignature.from_value(prototype)
    assert buffer.nbytes == value_tree_nbytes(prototype)

    wrong_level = {
        "tensor": torch.full((4,), 99.0, dtype=torch.float64),
        "values": [
            Plaintext(
                message=torch.tensor([5.0, 6.0]),
                level=1,
                scale=16.0,
            )
        ],
    }
    with pytest.raises(ExecutionInputError, match="value signature"):
        buffer.copy_from(wrong_level)
    torch.testing.assert_close(buffer.value["tensor"], source["tensor"])

    buffer.close()
    with pytest.raises(ExecutionError, match="closed"):
        _ = buffer.value


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_copy_handle_keeps_pinned_source_alive_until_transfer_completes() -> (
    None
):
    prototype = torch.zeros(4096, dtype=torch.float64, device="cuda:0")
    buffer = ReusableValueBuffer.like(prototype)
    target_pointer = buffer.value.data_ptr()
    source = torch.arange(4096, dtype=torch.float64).pin_memory()
    source_reference = weakref.ref(source)
    transfer_stream = torch.cuda.Stream(device=buffer.device)

    copied = buffer.copy_from(
        source,
        stream=transfer_stream,
        non_blocking=True,
    )
    del source
    gc.collect()
    assert source_reference() is not None
    assert copied.bytes_copied == prototype.numel() * prototype.element_size()

    copied.wait_on()
    torch.cuda.current_stream(buffer.device).synchronize()
    copied.synchronize()
    gc.collect()
    assert source_reference() is None
    assert buffer.value.data_ptr() == target_pointer
    torch.testing.assert_close(
        buffer.value.cpu(),
        torch.arange(4096, dtype=torch.float64),
    )
    buffer.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_first_cross_stream_overwrite_waits_for_buffer_initialization() -> None:
    device = torch.device("cuda:0")
    prototype = torch.full((4096,), 1.0, device=device)
    replacement = torch.full((4096,), 7.0, device=device)
    torch.cuda.synchronize(device)
    prototype_reference = weakref.ref(prototype)
    initialization_stream = torch.cuda.Stream(device=device)
    overwrite_stream = torch.cuda.Stream(device=device)

    # Delay the asynchronous CUDA-to-CUDA clone performed by ``like``. Without
    # an initialization event, the overwrite completes on its independent
    # stream first and the delayed clone deterministically restores prototype.
    with torch.cuda.stream(initialization_stream):
        torch.cuda._sleep(100_000_000)
        buffer = ReusableValueBuffer.like(prototype)

    copied = buffer.copy_from(replacement, stream=overwrite_stream)
    assert copied.event is not None
    del prototype
    gc.collect()
    assert prototype_reference() is not None

    try:
        copied.synchronize()
        initialization_stream.synchronize()
        torch.testing.assert_close(buffer.value, replacement)
    finally:
        buffer.close()

    gc.collect()
    assert prototype_reference() is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_copy_handle_retains_submitted_leaf_after_source_tree_mutation() -> (
    None
):
    prototype = [torch.zeros(4096, dtype=torch.float64, device="cuda:0")]
    buffer = ReusableValueBuffer.like(prototype)
    source_tensor = torch.arange(4096, dtype=torch.float64).pin_memory()
    source_reference = weakref.ref(source_tensor)
    source_tree = [source_tensor]
    transfer_stream = torch.cuda.Stream(device=buffer.device)

    copied = buffer.copy_from(
        source_tree,
        stream=transfer_stream,
        non_blocking=True,
    )
    source_tree.clear()
    del source_tensor
    gc.collect()
    assert source_reference() is not None

    copied.synchronize()
    gc.collect()
    assert source_reference() is None
    torch.testing.assert_close(
        buffer.value[0].cpu(),
        torch.arange(4096, dtype=torch.float64),
    )
    buffer.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_signatures_ignore_residency_but_bind_exact_value_state() -> None:
    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        allow_sk_gen=False,
    )
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    message = torch.linspace(-0.01, 0.01, 32, dtype=torch.float64)
    cuda_value = engine.encrypt_message(message, public_key)
    cpu_value = cuda_value.cpu()

    assert TensorSignature.from_tensor(
        cuda_value.data
    ) == TensorSignature.from_tensor(cpu_value.data)
    assert ValueSignature.from_value(cuda_value) == ValueSignature.from_value(
        cpu_value
    )

    signature = ValueTreeSignature.from_value({"input": cuda_value})
    signature.validate({"input": cpu_value})
    wrong_level = replace(cpu_value, level=cpu_value.level + 1)
    with pytest.raises(ExecutionInputError, match="signature differs"):
        signature.validate({"input": wrong_level})

    del cpu_value, cuda_value, public_key, secret_key, engine
    gc.collect()
    torch.cuda.empty_cache()
