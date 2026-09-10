r"""Cyclic-diagonal representations of packed-slot linear maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
import torch

ArrayLike: TypeAlias = Sequence[complex] | np.ndarray | torch.Tensor


def _as_complex_numpy(values: ArrayLike, *, copy: bool = False) -> np.ndarray:
    """Convert public offline data to a CPU ``complex128`` NumPy array."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if copy:
        return np.array(values, dtype=np.complex128, copy=True)
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True)
class DiagonalLinearTransform:
    r"""An immutable cyclic-diagonal linear map over packed CKKS slots.

    The map is

    $$
    y=\sum_k d_k\mathbin{\odot}\operatorname{Rot}_k(x).
    $$

    Each stored diagonal is a CPU `complex128` NumPy vector with shape
    `[slot]`; `reference` accepts and returns the same one-dimensional shape.
    This object contains no choice of execution algorithm. The matching
    evaluator independently decides whether to use
    direct diagonals, BSGS, hoisting, distribution, or a user implementation.
    Offsets are cyclic modulo $S$; duplicate representatives such as `-1` and
    `slots - 1` are combined by `normalized_diagonals`.
    """

    diagonals: Mapping[int, ArrayLike]
    slots: int
    name: str = 'diagonal_linear_transform'

    def __post_init__(self) -> None:
        if self.slots <= 0:
            raise ValueError('slots must be positive')
        if not self.diagonals:
            raise ValueError('at least one diagonal is required')
        frozen: dict[int, np.ndarray] = {}
        for offset, diagonal in self.diagonals.items():
            if not isinstance(offset, int):
                raise TypeError(f'diagonal offset must be int, got {offset!r}')
            array = _as_complex_numpy(diagonal, copy=True)
            if array.shape != (self.slots,):
                raise ValueError(
                    f'diagonal at offset {offset} has shape {array.shape}, '
                    f'expected {(self.slots,)}'
                )
            array.setflags(write=False)
            frozen[offset] = array
        object.__setattr__(self, 'diagonals', MappingProxyType(frozen))

    def normalized_diagonals(self) -> dict[int, np.ndarray]:
        r"""Map every offset to $k\bmod S$ and combine equal rotations.

        The stored arrays remain immutable. A new mapping is returned because
        two input offsets can normalize to the same cyclic key and must then be
        added elementwise. Returned vectors retain shape `[slot]`.
        """

        normalized: dict[int, np.ndarray] = {}
        for offset, diagonal in self.diagonals.items():
            key = offset % self.slots
            array = np.asarray(diagonal)
            normalized[key] = (
                normalized[key] + array if key in normalized else array
            )
        return normalized

    def reference(self, values: ArrayLike) -> np.ndarray:
        r"""Apply $L(x)=\sum_kd_k\odot\operatorname{Rot}_k(x)$ in NumPy.

        `values` must have shape `[slot]`. The returned CPU `complex128`
        array has shape `[slot]`. This plaintext oracle does not encode, rescale,
        consume depths, or model CKKS error.
        """

        source = _as_complex_numpy(values)
        if source.shape != (self.slots,):
            raise ValueError(
                f'input has shape {source.shape}, expected {(self.slots,)}'
            )
        result = np.zeros_like(
            source,
            dtype=np.result_type(source, np.complex128),
        )
        for offset, diagonal in self.diagonals.items():
            result = result + np.asarray(diagonal) * np.roll(source, offset)
        return result
