"""PyTorch distributed operations and CKKS value collectives.

For ordinary ``torch.Tensor`` values, this namespace exposes selected
``torch.distributed`` functions with their native signatures, mutation rules,
ProcessGroup behavior, and Work handles. FHElium value operations use typed
names such as ``broadcast_ciphertext`` and ``all_reduce_ciphertext``.
"""

import torch.distributed as _torch_distributed
from torch.distributed import (
    Backend,
    P2POp,
    ProcessGroup,
    ReduceOp,
    Work,
    all_gather,
    all_reduce,
    all_to_all,
    all_to_all_single,
    barrier,
    batch_isend_irecv,
    broadcast,
    destroy_process_group,
    gather,
    get_backend,
    get_global_rank,
    get_group_rank,
    get_process_group_ranks,
    get_rank,
    get_world_size,
    init_process_group,
    irecv,
    is_available,
    is_initialized,
    isend,
    new_group,
    recv,
    reduce,
    reduce_scatter,
    scatter,
    send,
)

from fhelium.distributed._state import init, local_device, shutdown
from fhelium.distributed._typed_collectives import (
    all_gather_ciphertexts,
    all_gather_compressed_plaintexts,
    all_gather_plaintexts,
    all_reduce_ciphertext,
    broadcast_ciphertext,
    broadcast_compressed_plaintext,
    broadcast_key,
    broadcast_plaintext,
    gather_ciphertext_limbs,
    gather_ciphertexts,
    reduce_ciphertext,
    scatter_ciphertext_limbs,
    scatter_ciphertexts,
)

# PyTorch 2.13 added concise names for the single-output tensor collectives.
# Keep FHElium's facade stable on 2.10 through 2.12 by using the established
# names with identical signatures on those releases.
all_gather_single = getattr(
    _torch_distributed,
    "all_gather_single",
    _torch_distributed.all_gather_into_tensor,
)
reduce_scatter_single = getattr(
    _torch_distributed,
    "reduce_scatter_single",
    _torch_distributed.reduce_scatter_tensor,
)

__all__ = [
    "Backend",
    "P2POp",
    "ProcessGroup",
    "ReduceOp",
    "Work",
    "all_gather",
    "all_gather_ciphertexts",
    "all_gather_compressed_plaintexts",
    "all_gather_plaintexts",
    "all_gather_single",
    "all_reduce",
    "all_reduce_ciphertext",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "broadcast_ciphertext",
    "broadcast_compressed_plaintext",
    "broadcast_key",
    "broadcast_plaintext",
    "destroy_process_group",
    "gather",
    "gather_ciphertext_limbs",
    "gather_ciphertexts",
    "get_backend",
    "get_global_rank",
    "get_group_rank",
    "get_process_group_ranks",
    "get_rank",
    "get_world_size",
    "init",
    "init_process_group",
    "irecv",
    "is_available",
    "is_initialized",
    "isend",
    "local_device",
    "new_group",
    "recv",
    "reduce",
    "reduce_ciphertext",
    "reduce_scatter",
    "reduce_scatter_single",
    "scatter",
    "scatter_ciphertext_limbs",
    "scatter_ciphertexts",
    "send",
    "shutdown",
]
