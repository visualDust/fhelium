from functools import cached_property

from fhelium.config import CkksConfig


class MontgomeryParameters:
    r"""Host-side constants for per-prime Montgomery arithmetic.

    Let $R=2^w$, where $w=\mathtt{buffer\_bit\_length}$. For every odd modulus $q_i$
    in canonical level-zero ``[Q | P]`` order, standard residue $x_i$ is stored
    in Montgomery representation as $x_iR\bmod q_i$. The native reduction

    $$
    \operatorname{REDC}_{q_i}(t)=(t+m q_i)/R,
    \qquad m=t(-q_i^{-1})\bmod R,
    $$

    therefore maps a product of two Montgomery residues back to Montgomery
    form. Native kernels accept lazy representatives in $[0,2q_i)$ and rely on
    $4q_i<R$ for signed-word overflow safety. This object stores Python
    integers only; ``RnsRuntime`` materializes exact integral row tables.
    """

    def __init__(self, ckks_config: CkksConfig):
        buffer_bit_length = ckks_config.buffer_bit_length
        moduli = ckks_config.moduli
        # Montgomery reduction with R=2^w requires every modulus to be odd.
        if any((qi % 2 == 0) for qi in moduli):
            raise ValueError(
                "All qi in q must be odd for Montgomery with R=2^w."
            )
        radix = 1 << buffer_bit_length
        if any(4 * qi >= radix for qi in moduli):
            raise ValueError(
                "Every modulus must satisfy 4 * modulus < "
                "2**buffer_bit_length for signed-word lazy Montgomery "
                "arithmetic"
            )

        self._buffer_bit_length = buffer_bit_length
        self._moduli = list(moduli)

    @property
    def moduli(self) -> list[int]:
        """Canonical modulus order ``[Q | P]``."""
        return self._moduli

    @cached_property
    def twice_modulus(self) -> list[int]:
        r"""Return $2q_i$ for lazy-range corrections, in prime-id order."""
        return [qi << 1 for qi in self.moduli]

    @property
    def buffer_bit_length(self) -> int:
        """Bit-length of the buffer type (30 or 62)."""
        return self._buffer_bit_length

    @cached_property
    def half_buffer_bit_length(self) -> int:
        r"""Return $w/2$ for the native split-word implementation."""
        return self._buffer_bit_length // 2

    @cached_property
    def lower_bits_mask(self) -> int:
        r"""Return the lower-half mask $2^{w/2}-1$."""
        return (1 << self.half_buffer_bit_length) - 1

    @cached_property
    def full_bits_mask(self) -> int:
        r"""Return the radix mask $R-1=2^w-1$."""
        return (1 << self._buffer_bit_length) - 1

    @cached_property
    def R(self) -> int:
        r"""Return Montgomery radix $R=2^w$."""
        return 1 << self._buffer_bit_length

    @cached_property
    def montgomery_r2(self) -> list[int]:
        r"""Return $R^2\bmod q_i$ used by REDC to map $x_i$ to $x_iR$."""
        return [pow(self.R, 2, qi) for qi in self.moduli]

    @cached_property
    def montgomery_r_inverse(self) -> list[int]:
        r"""Return $R^{-1}\bmod q_i$ in canonical prime-id order."""
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
            ki >> self.half_buffer_bit_length for ki in self.neg_inv_modulus
        ]

    @cached_property
    def modulus_lower_bits(self) -> list[int]:
        r"""Return the lower $w/2$ bits of each $q_i$."""
        return [qi & self.lower_bits_mask for qi in self.moduli]

    @cached_property
    def modulus_higher_bits(self) -> list[int]:
        r"""Return the higher $w/2$ bits of each $q_i$."""
        return [qi >> self.half_buffer_bit_length for qi in self.moduli]

    # -------------------------------------------------------------------------
    # Printing / representation
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MontgomeryParameters("
            f"buffer_bit_length={self._buffer_bit_length}, "
            f"modulus_count={len(self.moduli)})"
        )

    def __str__(self) -> str:
        q_bits_preview = [qi.bit_length() for qi in self.moduli[:8]]
        suffix = "..." if len(self.moduli) > 8 else ""
        return (
            f"MontgomeryParameters("
            f"buffer_bit_length={self._buffer_bit_length}, "
            f"R=2^{self._buffer_bit_length}, "
            f"half_buffer_bit_length={self.half_buffer_bit_length}, "
            f"modulus_count={len(self.moduli)}, "
            f"modulus_bit_lengths={q_bits_preview}{suffix}"
            f")"
        )
