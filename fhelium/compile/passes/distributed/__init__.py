"""Caller-selected transforms for rank-local distributed operations."""

from ._lower_specialized_collectives import LowerSpecializedCollectivesPass

__all__ = ["LowerSpecializedCollectivesPass"]
