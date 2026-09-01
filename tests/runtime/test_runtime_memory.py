from __future__ import annotations

import pytest
import torch

from fhelium.runtime import MemorySnapshot


def test_memory_snapshot_reads_cross_platform_host_memory() -> None:
    first = MemorySnapshot.read("cpu")
    second = MemorySnapshot.read(torch.device("cpu"))

    assert first.device == torch.device("cpu")
    assert first.capacity_bytes > 0
    assert 0 <= first.available_bytes <= first.capacity_bytes
    assert first.torch_allocated_bytes is None
    assert first.torch_reserved_bytes is None
    assert second.observed_at_ns >= first.observed_at_ns
    assert first.as_dict()["capacity_bytes"] == first.capacity_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_memory_snapshot_reads_global_and_pytorch_cuda_counters() -> None:
    before = MemorySnapshot.read("cuda")
    allocation = torch.empty(
        1024 * 1024,
        dtype=torch.uint8,
        device=before.device,
    )
    snapshot = MemorySnapshot.read(before.device)

    assert snapshot.device.index == torch.cuda.current_device()
    assert snapshot.capacity_bytes > 0
    assert 0 <= snapshot.available_bytes <= snapshot.capacity_bytes
    assert snapshot.torch_allocated_bytes is not None
    assert snapshot.torch_allocated_bytes >= 0
    assert snapshot.torch_reserved_bytes is not None
    assert snapshot.torch_reserved_bytes >= 0
    assert before.torch_allocated_bytes is not None
    assert (
        snapshot.torch_allocated_bytes - before.torch_allocated_bytes
        >= allocation.numel() * allocation.element_size()
    )
