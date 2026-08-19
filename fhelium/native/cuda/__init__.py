"""CUDA device and peer-topology inspection."""

from __future__ import annotations

import importlib
from typing import Any, TypedDict, cast


CudaDeviceProperties = dict[str, Any]
CudaDeviceInventory = dict[int, CudaDeviceProperties]


class CudaSystemInfo(TypedDict):
    """CUDA device inventory and native peer-topology result."""

    devices: CudaDeviceInventory
    p2p: dict[str, Any]


def _extension() -> Any:
    from fhelium.native import require_native_backend

    require_native_backend("cuda")
    try:
        return importlib.import_module("fhelium.native.cuda.cuda_info")
    except ImportError as error:
        raise ImportError(
            "FHElium CUDA inspection is unavailable because cuda_info is "
            "missing from this CUDA-enabled build"
        ) from error


def get_cuda_info(test_p2p_bandwidth: bool = True) -> CudaSystemInfo:
    """Return CUDA device and peer-topology information.

    Args:
        test_p2p_bandwidth: Measure peer-to-peer bandwidth when multiple CUDA
            devices are visible. The measurement allocates and copies test
            buffers.
    """

    return cast(
        CudaSystemInfo,
        _extension().get_cuda_system_info(testP2PBandwidth=test_p2p_bandwidth),
    )


def get_cuda_device_properties() -> CudaDeviceInventory:
    """Return CUDA device properties without peer-bandwidth measurements."""

    return cast(CudaDeviceInventory, _extension().get_cuda_device_properties())


__all__ = [
    "CudaDeviceInventory",
    "CudaDeviceProperties",
    "CudaSystemInfo",
    "get_cuda_device_properties",
    "get_cuda_info",
]
