"""Map CKKS slot conventions to polynomial Galois elements."""

from __future__ import annotations
from functools import cache


@cache
def rotation_galois_element(
    ring_dimension: int, rotation_step: int, generator: int = 3
) -> int:
    r"""Map signed ``rotation_step`` to odd polynomial ``galois_element``.

    The slot convention is ``torch.roll(slots, shifts=rotation_step)``. The
    result $g$ identifies automorphism $\sigma_g:X\mapsto X^g$ modulo
    $X^N+1$. The step and element are related but have distinct
    names.
    """

    exponent = -int(rotation_step) if generator == 5 else int(rotation_step)
    return pow(
        generator,
        exponent % ring_dimension,
        2 * ring_dimension,
    )
