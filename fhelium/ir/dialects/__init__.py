"""Registered dialect modules for FHElium's multi-level IR."""

from . import (
    ckks,
    core,
    distributed,
    logical,
    memory,
    ntt,
    rns,
    semantic,
    torch,
)
from .ckks import FHEliumCkks
from .core import FHElium
from .distributed import FHEliumDistributed
from .logical import FHEliumLogical
from .memory import FHEliumMemory
from .ntt import FHEliumNtt
from .rns import FHEliumRns
from .semantic import FHEliumSemantic
from .torch import Torch

REGISTERED_DIALECTS = (
    FHElium,
    FHEliumSemantic,
    FHEliumLogical,
    FHEliumCkks,
    FHEliumRns,
    FHEliumNtt,
    FHEliumMemory,
    FHEliumDistributed,
    Torch,
)
"""All first-party dialects loaded by the permissive FHElium context."""

__all__ = [
    "FHElium",
    "FHEliumCkks",
    "FHEliumDistributed",
    "FHEliumLogical",
    "FHEliumMemory",
    "FHEliumNtt",
    "FHEliumRns",
    "FHEliumSemantic",
    "REGISTERED_DIALECTS",
    "Torch",
    "ckks",
    "core",
    "distributed",
    "logical",
    "memory",
    "ntt",
    "rns",
    "semantic",
    "torch",
]
