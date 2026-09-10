"""Read execution-relevant operand linkage for native implementations."""

from __future__ import annotations

import torch

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.rns.context import RnsContext


def _operand_basis(invocation: OperationInvocation, index: int = 0) -> str:
    represented = invocation.operand_bases[index]
    return "Q" if represented is None else represented


def _active_depth(
    tensor: torch.Tensor,
    context: RnsContext,
    *,
    include_p: bool,
) -> int:
    """Map a complete context Q or QP tensor to resource-table position."""

    return context.rns_layout.depth_for_active_row_count(
        tensor.size(-2),
        include_p=include_p,
    )
