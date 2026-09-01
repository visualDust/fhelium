"""Placement resources and registered Tensor transfer execution."""

from .operations import TorchMemoryTransferImplementation
from .resources import (
    DEVICE_RESOURCE_KIND,
)

__all__ = [
    "DEVICE_RESOURCE_KIND",
    "TorchMemoryTransferImplementation",
]
