"""Private aggregation point for FHElium's typed collectives.

Implementations live in responsibility-specific modules.  The public
``fhelium.distributed`` facade imports through this module to keep its surface
unchanged.
"""

from fhelium.distributed._ciphertext_reduction import (
    all_reduce_ciphertext,
    reduce_ciphertext,
)
from fhelium.distributed._limb_collectives import (
    gather_ciphertext_limbs,
    scatter_ciphertext_limbs,
)
from fhelium.distributed._value_collectives import (
    all_gather_ciphertexts,
    all_gather_compressed_plaintexts,
    all_gather_plaintexts,
    broadcast_ciphertext,
    broadcast_compressed_plaintext,
    broadcast_key,
    broadcast_plaintext,
    gather_ciphertexts,
    scatter_ciphertexts,
)

__all__ = [
    "all_gather_ciphertexts",
    "all_gather_compressed_plaintexts",
    "all_gather_plaintexts",
    "all_reduce_ciphertext",
    "broadcast_ciphertext",
    "broadcast_compressed_plaintext",
    "broadcast_key",
    "broadcast_plaintext",
    "gather_ciphertext_limbs",
    "gather_ciphertexts",
    "reduce_ciphertext",
    "scatter_ciphertext_limbs",
    "scatter_ciphertexts",
]
