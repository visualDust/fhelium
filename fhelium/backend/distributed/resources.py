"""Process-group resources used by distributed operation implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

PROCESS_GROUP_RESOURCE_KIND = "process-group"


@dataclass(frozen=True)
class ProcessGroupExecutionResource:
    """Expose one initialized rank-local process group to Backend operations.

    ``group=None`` names PyTorch's default process group.  Ranks represented by
    distributed IR are process-group ranks; conversion to a global rank occurs
    only when a PyTorch collective requires it.
    """

    group: torch.distributed.ProcessGroup | None = None

    def __post_init__(self) -> None:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available")
        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        if self.group is not None and not isinstance(
            self.group, torch.distributed.ProcessGroup
        ):
            raise TypeError("group must be a torch.distributed.ProcessGroup")

    @property
    def rank(self) -> int:
        """Return this process's rank within the selected group."""

        return torch.distributed.get_rank(self.group)

    @property
    def size(self) -> int:
        """Return the number of participating group ranks."""

        return torch.distributed.get_world_size(self.group)

    def global_rank(self, group_rank: int) -> int:
        """Map one group-relative rank to its process-global rank."""

        if not 0 <= group_rank < self.size:
            raise IndexError(
                f"group rank {group_rank} is outside [0, {self.size})"
            )
        if self.group is None:
            return group_rank
        return torch.distributed.get_global_rank(self.group, group_rank)


__all__ = [
    "PROCESS_GROUP_RESOURCE_KIND",
    "ProcessGroupExecutionResource",
]
