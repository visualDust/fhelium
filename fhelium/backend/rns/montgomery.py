from functools import cached_property

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
    integers only; ``RnsContext`` materializes the integral row tables with
    libfhelium.
    """

    def __init__(
        self,
        ckks_config: CkksConfig,
        execution_format: RnsExecutionFormat,
    ):
        radix_bits = execution_format.radix_bits
        moduli = ckks_config.moduli
        self._radix_bits = radix_bits
        self._moduli = list(moduli)

    @property
    def moduli(self) -> list[int]:
        """Modulus order ``[Q | P]``."""
        return self._moduli

    @property
    def radix_bits(self) -> int:
        """Montgomery radix exponent: 32 for int32 and 62 for int64 residues."""
        return self._radix_bits

    @cached_property
    def R(self) -> int:
        r"""Return Montgomery radix $R=2^w$."""
        return 1 << self._radix_bits

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
            f"modulus_count={len(self.moduli)}, "
            f"modulus_bit_lengths={q_bits_preview}{suffix}"
            f")"
        )
