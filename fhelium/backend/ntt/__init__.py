"""NTT numerical implementations, table preparation, and context-owned transforms.

NttContext provides policy-specific Tensor tables. NativeNttImplementation
prepares operations over supplied Tensor operands and shares native calls with
context transforms. Compile can select a schedule before preparing execution.
"""

from .context import NttContext
from .operations import NativeNttImplementation
from .tables import (
    CompactPowerOfTwoRadixTables,
    CompactRadix2Tables,
    IndexedRadix2Tables,
    NttTables,
    prepare_ntt_tables,
)

__all__ = [
    "CompactPowerOfTwoRadixTables",
    "CompactRadix2Tables",
    "IndexedRadix2Tables",
    "NativeNttImplementation",
    "NttContext",
    "NttTables",
    "prepare_ntt_tables",
]
