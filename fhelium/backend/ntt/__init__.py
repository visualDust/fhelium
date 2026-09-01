"""NTT plans, parameter tables, and configured schedule executors.

``NttContext`` composes one RNS context with a selected policy, parameter
tables, and schedule executor. ``NativeNttImplementation`` implements NTT IR
operations and delegates each transform through that context.
"""

from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ntt.executors.compact_radix2 import (
    CompactRadix2NttBackend,
)
from fhelium.backend.ntt.executors.indexed_radix2 import (
    IndexedRadix2NttBackend,
)
from fhelium.backend.ntt.executors.power_of_two_radix import (
    CompactPowerOfTwoRadixNttBackend,
)
from fhelium.backend.ntt.factory import create_ntt_backend
from fhelium.backend.ntt.interface import NttBackend
from fhelium.backend.ntt.tables import (
    CompactPowerOfTwoRadixTables,
    CompactRadix2Tables,
    IndexedRadix2Tables,
    NttTables,
    prepare_ntt_tables,
)

__all__ = [
    "CompactPowerOfTwoRadixNttBackend",
    "CompactPowerOfTwoRadixTables",
    "CompactRadix2NttBackend",
    "CompactRadix2Tables",
    "IndexedRadix2NttBackend",
    "IndexedRadix2Tables",
    "NttBackend",
    "NttContext",
    "NttTables",
    "create_ntt_backend",
    "prepare_ntt_tables",
]
