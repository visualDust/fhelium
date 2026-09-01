from __future__ import annotations

import pytest
import torch

from fhelium.benchmarks.v1 import collect_platform
from fhelium.runtime import (
    CpuTopology,
    CudaDeviceInfo,
    CudaTopology,
)


def _cuda_info(index: int) -> CudaDeviceInfo:
    return CudaDeviceInfo(
        device=torch.device("cuda", index),
        name=f"test-cuda-{index}",
        uuid=f"00000000-0000-0000-0000-{index:012d}",
        pci_domain_id=0,
        pci_bus_id=index,
        pci_device_id=0,
        compute_capability=(9, 0),
        total_memory_bytes=1024,
        multiprocessor_count=1,
        warp_size=32,
        l2_cache_bytes=1,
        memory_bus_width_bits=1,
        core_clock_khz=1,
        memory_clock_khz=1,
        max_threads_per_block=1024,
        max_threads_per_multiprocessor=1024,
        shared_memory_per_block_bytes=1,
        shared_memory_per_multiprocessor_bytes=1,
    )


def test_cpu_topology_observes_the_host() -> None:
    topology = CpuTopology.probe()

    assert topology.logical_processor_count is not None
    assert topology.logical_processor_count > 0
    assert topology.model


def test_cuda_topology_preserves_directional_peer_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        CudaDeviceInfo,
        "probe",
        classmethod(lambda cls, index: _cuda_info(index)),
    )
    calls: list[tuple[int, int]] = []

    def can_access(source: int, destination: int) -> bool:
        calls.append((source, destination))
        return source == 0 and destination == 1

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_access)
    topology = CudaTopology.probe()

    assert topology.peer_access == ((True, True), (False, True))
    assert calls == [(0, 1), (1, 0)]

    with pytest.raises(ValueError, match="square matrix"):
        CudaTopology(topology.devices, ((True,), (False, True)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_topology_observes_visible_devices() -> None:
    topology = CudaTopology.probe()
    assert len(topology.devices) == torch.cuda.device_count()
    current_index = torch.cuda.current_device()
    assert topology.devices[current_index].compute_capability == (
        torch.cuda.get_device_capability(current_index)
    )
    assert all(
        topology.peer_access[index][index]
        for index in range(len(topology.devices))
    )
    if len(topology.devices) >= 2:
        assert topology.peer_access[0][1] == torch.cuda.can_device_access_peer(
            0, 1
        )


def test_benchmark_platform_uses_runtime_observation_shapes() -> None:
    snapshot = collect_platform(invocation=("benchmark",), environ={})

    assert snapshot.cpu["logical_processor_count"] > 0
    assert snapshot.memory["device"] == "cpu"
    assert snapshot.memory["capacity_bytes"] > 0
    assert snapshot.memory["available_bytes"] >= 0
    if torch.cuda.is_available():
        assert len(snapshot.cuda["devices"]) == torch.cuda.device_count()
        assert len(snapshot.cuda["peer_access"]) == torch.cuda.device_count()
