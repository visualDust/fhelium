"""Select the scalar representation used by device-local RNS resources."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RnsExecutionFormat:
    """Describe one supported residue Tensor and Montgomery-radix pairing."""

    dtype: torch.dtype
    radix_bits: int

    @classmethod
    def select(
        cls,
        moduli: tuple[int, ...],
        dtype: torch.dtype | None = None,
    ) -> "RnsExecutionFormat":
        """Select or validate a representation for the supplied prime basis."""

        if dtype is None:
            dtype = (
                torch.int32
                if all(modulus < 1 << 30 for modulus in moduli)
                else torch.int64
            )
        try:
            result = {
                torch.int32: cls(torch.int32, 32),
                torch.int64: cls(torch.int64, 62),
            }[dtype]
        except KeyError:
            raise ValueError(
                "RNS dtype must be torch.int32 or torch.int64"
            ) from None
        limit = 1 << (30 if result.dtype == torch.int32 else 60)
        if any(modulus >= limit for modulus in moduli):
            raise ValueError(
                "A modulus exceeds the selected RNS representation's lazy range"
            )
        return result


__all__ = ["RnsExecutionFormat"]
