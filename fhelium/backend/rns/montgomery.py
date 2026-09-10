from functools import cached_property

import torch

from fhelium.config import CkksConfig
from .format import RnsExecutionFormat


class MontgomeryParameters:
    r"""Host-side constants for per-prime Montgomery arithmetic.

    Let $R=2^w$, where $w=\mathtt{buffer\_bit\_length}$. For every odd modulus $q_i$
    in depth-zero ``[Q | P]`` order, standard residue $x_i$ is stored
    in Montgomery representation as $x_iR\bmod q_i$. The native reduction

    $$
    \operatorname{REDC}_{q_i}(t)=(t+m q_i)/R,
    \qquad m=t(-q_i^{-1})\bmod R,
    $$

    therefore maps a product of two Montgomery residues back to Montgomery
    form. Native kernels accept lazy representatives in $[0,2q_i)$ and rely on
    $4q_i<R$ for signed-word overflow safety. This object stores Python
    integers only; ``RnsContext`` materializes integral row tables.
    """

    def __init__(
        self,
        ckks_config: CkksConfig,
        execution_format: RnsExecutionFormat,
    ):
        radix_bits = execution_format.radix_bits
        moduli = ckks_config.moduli
        # Montgomery reduction with R=2^w requires every modulus to be odd.
        if any((qi % 2 == 0) for qi in moduli):
            raise ValueError(
                "All qi in q must be odd for Montgomery with R=2^w."
            )
        radix = 1 << radix_bits
        maximum_modulus = 1 << (30 if execution_format.dtype == torch.int32 else 60)
        if any(qi >= maximum_modulus for qi in moduli):
            raise ValueError(
                "Every modulus must fit the configured Montgomery and lazy-residue representation"
            )

        self._radix_bits = radix_bits
        self._moduli = list(moduli)

    @property
    def moduli(self) -> list[int]:
        """Modulus order ``[Q | P]``."""
        return self._moduli

    @cached_property
    def twice_modulus(self) -> list[int]:
        r"""Return $2q_i$ for lazy-range corrections, in prime-id order."""
        return [qi << 1 for qi in self.moduli]

    @property
    def radix_bits(self) -> int:
        """Bit-length of the buffer type (30 or 62)."""
        return self._radix_bits

    @cached_property
    def half_radix_bits(self) -> int:
        r"""Return $w/2$ for the native split-word implementation."""
        return self._radix_bits // 2

    @cached_property
    def lower_bits_mask(self) -> int:
        r"""Return the lower-half mask $2^{w/2}-1$."""
        return (1 << self.half_radix_bits) - 1

    @cached_property
    def full_bits_mask(self) -> int:
        r"""Return the radix mask $R-1=2^w-1$."""
        return (1 << self._radix_bits) - 1

    @cached_property
    def R(self) -> int:
        r"""Return Montgomery radix $R=2^w$."""
        return 1 << self._radix_bits

    @cached_property
    def montgomery_r2(self) -> list[int]:
        r"""Return $R^2\bmod q_i$ used by REDC to map $x_i$ to $x_iR$."""
        return [pow(self.R, 2, qi) for qi in self.moduli]

    @cached_property
    def montgomery_r_inverse(self) -> list[int]:
        r"""Return $R^{-1}\bmod q_i$ in prime-id order."""
        return [pow(self.R, -1, qi) for qi in self.moduli]

    @cached_property
    def neg_inv_modulus(self) -> list[int]:
        r"""Return $-q_i^{-1}\bmod R$ for Montgomery reduction."""
        inv_q_mod_R = [pow(qi, -1, self.R) for qi in self.moduli]
        return [(-inv) % self.R for inv in inv_q_mod_R]

    @cached_property
    def neg_inv_modulus_lower_bits(self) -> list[int]:
        r"""Return the lower $w/2$ bits of $-q_i^{-1}\bmod R$."""
        return [ki & self.lower_bits_mask for ki in self.neg_inv_modulus]

    @cached_property
    def neg_inv_modulus_higher_bits(self) -> list[int]:
        r"""Return the higher $w/2$ bits of $-q_i^{-1}\bmod R$."""
        return [
            ki >> self.half_radix_bits for ki in self.neg_inv_modulus
        ]

    @cached_property
    def modulus_lower_bits(self) -> list[int]:
        r"""Return the lower $w/2$ bits of each $q_i$."""
        return [qi & self.lower_bits_mask for qi in self.moduli]

    @cached_property
    def modulus_higher_bits(self) -> list[int]:
        r"""Return the higher $w/2$ bits of each $q_i$."""
        return [qi >> self.half_radix_bits for qi in self.moduli]

    # -------------------------------------------------------------------------
    # Printing / representation
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MontgomeryParameters("
            f"radix_bits={self._radix_bits}, "
            f"modulus_count={len(self.moduli)})"
        )

    def __str__(self) -> str:
        q_bits_preview = [qi.bit_length() for qi in self.moduli[:8]]
        suffix = "..." if len(self.moduli) > 8 else ""
        return (
            f"MontgomeryParameters("
            f"radix_bits={self._radix_bits}, "
            f"R=2^{self._radix_bits}, "
            f"half_radix_bits={self.half_radix_bits}, "
            f"modulus_count={len(self.moduli)}, "
            f"modulus_bit_lengths={q_bits_preview}{suffix}"
            f")"
        )
