"""Transform logical operations into scheduled CKKS operations."""

from ._assign_depths import AssignCkksDepthsPass
from ._assign_scales import AssignCkksScalesPass
from ._hoist_rotations import HoistRotationsPass
from ._insert_multiply_ntt_transitions import InsertMultiplyNttTransitionsPass
from ._insert_plaintext_preparation import InsertPlaintextPreparationPass
from ._relinearization import (
    InsertRelinearizationPass,
    LateRelinearizationPass,
)
from ._rescale import InsertRescalePass, LateRescalePass
from ._lower_message_preparation import LowerMessagePlaintextPreparationPass
from ._lower_logical_to_ckks import LowerLogicalToCkksPass
from ._resolve_rotation_keys import ResolveRotationKeyOperandsPass

__all__ = [
    "AssignCkksDepthsPass",
    "AssignCkksScalesPass",
    "HoistRotationsPass",
    "InsertMultiplyNttTransitionsPass",
    "InsertPlaintextPreparationPass",
    "InsertRelinearizationPass",
    "InsertRescalePass",
    "LateRelinearizationPass",
    "LateRescalePass",
    "LowerLogicalToCkksPass",
    "LowerMessagePlaintextPreparationPass",
    "ResolveRotationKeyOperandsPass",
]
