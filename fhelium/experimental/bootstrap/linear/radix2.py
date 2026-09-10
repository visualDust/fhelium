r"""Radix-2 synthesis of CKKS coefficient/slot transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fhelium.experimental.bootstrap.linear.transform import (
    DiagonalLinearTransform,
)

TransformDirection = Literal['coeffs_to_slots', 'slots_to_coeffs']


def _balanced_layer_groups(
    total_layers: int, stage_count: int
) -> tuple[int, ...]:
    """Partition adjacent radix layers into nearly equal collapse groups.

    Earlier groups receive one extra layer when division is uneven.  The
    returned positive widths preserve order and sum to ``total_layers``.
    """

    if total_layers <= 0:
        raise ValueError('total_layers must be positive')
    if not 0 < stage_count <= total_layers:
        raise ValueError('stage_count must lie in [1, total_layers]')
    width, wider_groups = divmod(total_layers, stage_count)
    return tuple(
        width + int(index < wider_groups) for index in range(stage_count)
    )


def _cyclotomic_orbit(slots: int, generator: int) -> np.ndarray:
    r"""Enumerate $1,g,g^2,\ldots,g^{S-1}\pmod{4S}$ as `[slot]`.

    The returned CPU `uint64` tensor has shape `[slot]`. A valid full-slot
    generator must visit $S$ distinct odd exponents.
    Rejecting a shorter orbit here prevents a transform table from silently
    using a different slot ordering than the codec and rotation keys.
    """

    modulus = 4 * slots
    mask = modulus - 1
    orbit = np.empty(slots, dtype=np.uint64)
    value = 1
    for index in range(slots):
        orbit[index] = value
        value = (value * generator) & mask
    if len(set(int(value) for value in orbit)) != slots:
        raise ValueError(
            f'generator {generator} does not produce a full slot orbit'
        )
    return orbit


def _root_powers(slots: int) -> np.ndarray:
    r"""Return `[root_index]` values $\exp(2\pi i k/(4S))$ in `complex128`."""

    modulus = 4 * slots
    powers = np.exp(2j * np.pi * np.arange(modulus + 1) / modulus)
    powers[-1] = powers[0]
    return powers.astype(np.complex128)


def _forward_layer_coefficients(
    roots: np.ndarray,
    orbit: np.ndarray,
    *,
    imaginary_unit_correction: bool,
) -> list[np.ndarray]:
    """Construct three diagonals for each coefficient-to-slot butterfly.

    The list is grouped by diagonal role: center diagonals for all radix
    stages, followed by their left and right butterfly diagonals.  For a layer
    of width ``m``, the first and second half of each block receive the
    butterfly constants and orbit-indexed twiddle factors.  Arrays are indexed
    in the cyclotomic slot order supplied by ``orbit``.

    ``imaginary_unit_correction`` applies the terminal ``-i`` convention at
    width two.  It must be chosen consistently with the inverse compiler.
    """

    slots = int(orbit.size)
    dimension = int(roots.size - 1)
    log_slots = int(np.log2(slots))
    coefficients = [
        np.zeros(slots, dtype=np.complex128) for _ in range(3 * log_slots)
    ]
    for stage in range(log_slots - 1, -1, -1):
        width = 1 << (stage + 1)
        half = width >> 1
        local_order = width << 2
        phase = -1j if imaginary_unit_correction and width == 2 else 1.0
        twiddle_indices = (
            local_order - np.remainder(orbit[:half], local_order)
        ) * (dimension // local_order)
        twiddles = phase * roots[twiddle_indices.astype(np.int64)]

        center = coefficients[stage].reshape(-1, width)
        left = coefficients[stage + log_slots].reshape(-1, width)
        right = coefficients[stage + 2 * log_slots].reshape(-1, width)
        right[:, :half] = phase
        left[:, :half] = phase
        left[:, half:] = -twiddles
        center[:, half:] = twiddles
    return coefficients


def _inverse_layer_coefficients(
    roots: np.ndarray,
    orbit: np.ndarray,
    *,
    imaginary_unit_correction: bool,
) -> list[np.ndarray]:
    r"""Construct inverse slot-to-coefficient radix-2 butterfly diagonals.

    These are the algebraic inverse layers before stage collapse and external
    normalization are applied.  Layers are generated in increasing butterfly
    width, reversing the forward transform's order.  The optional terminal
    phase is $+i$, the inverse of the forward correction.
    """

    slots = int(orbit.size)
    dimension = int(roots.size - 1)
    log_slots = int(np.log2(slots))
    coefficients = [
        np.zeros(slots, dtype=np.complex128) for _ in range(3 * log_slots)
    ]
    for stage in range(log_slots):
        width = 1 << (stage + 1)
        half = width >> 1
        local_order = width << 2
        phase = 1j if imaginary_unit_correction and width == 2 else 1.0
        twiddle_indices = np.remainder(orbit[:half], local_order) * (
            dimension // local_order
        )
        twiddles = phase * roots[twiddle_indices.astype(np.int64)]

        center = coefficients[stage].reshape(-1, width)
        left = coefficients[stage + log_slots].reshape(-1, width)
        right = coefficients[stage + 2 * log_slots].reshape(-1, width)
        center[:, half:] = phase
        left[:, :half] = phase
        right[:, :half] = twiddles
        left[:, half:] = -twiddles
    return coefficients


def _radix2_layer_transform(
    *,
    slots: int,
    coefficients: Sequence[np.ndarray],
    stage: int,
    name: str,
) -> DiagonalLinearTransform:
    r"""Lower one radix butterfly to offsets $+s$, $0$, and $-s$.

    The right diagonal is rotated by $-2s$ because it is applied before
    the output-side rotation represented by the negative offset.  Offsets can
    coincide for the smallest slot counts, so contributions are accumulated
    rather than assigned.
    """

    stride = 1 << stage
    diagonals: dict[int, np.ndarray] = {}
    contributions = (
        (stride, np.asarray(coefficients[0])),
        (0, np.asarray(coefficients[1])),
        (-stride, np.roll(np.asarray(coefficients[2]), -2 * stride)),
    )
    for offset, diagonal in contributions:
        key = offset % slots
        diagonals[key] = diagonals.get(key, 0) + diagonal
    return DiagonalLinearTransform(
        diagonals=diagonals,
        slots=slots,
        name=name,
    )


def _compose_diagonal_transforms(
    first: DiagonalLinearTransform,
    second: DiagonalLinearTransform,
    *,
    name: str,
) -> DiagonalLinearTransform:
    r"""Return the cyclic-diagonal composition $B(A(x))$.

    If $A_i$ and $B_j$ are diagonals at offsets $i$ and $j$, their composed
    contribution is $B_j\odot\operatorname{Rot}_j(A_i)$ at offset $i+j$.
    This algebraic composition collapses several one-depth butterflies into a
    single diagonal stage without using sampled matrix multiplication.
    """

    if first.slots != second.slots:
        raise ValueError('cannot compose transforms with different slot counts')
    slots = first.slots
    diagonals: dict[int, np.ndarray] = {}
    for first_offset, first_diagonal in first.diagonals.items():
        for second_offset, second_diagonal in second.diagonals.items():
            offset = (first_offset + second_offset) % slots
            contribution = np.asarray(second_diagonal) * np.roll(
                np.asarray(first_diagonal), second_offset
            )
            diagonals[offset] = diagonals.get(offset, 0) + contribution
    return DiagonalLinearTransform(
        diagonals=diagonals,
        slots=slots,
        name=name,
    )


@dataclass(frozen=True)
class Radix2FourierTransformCompiler:
    r"""Synthesize CKKS basis transforms from radix-2 butterflies.

    Let $C$ be the unscaled `coeffs_to_slots` map and $T$ the unscaled
    `slots_to_coeffs` map in the engine's cyclotomic slot order. The compiler's
    convention is

    $$
    T(C(x))=Sx.
    $$

    Consequently a plaintext round trip uses forward `scale=1` and inverse
    `scale=1/S`. The supplied `scale` multiplies the numerical map; it is not a
    CKKS metadata scale and does not change the diagonal plaintext encoding
    scale selected later by the evaluator.

    `stage_count` controls only algebraic layer collapse. A smaller value
    consumes fewer CKKS depths but materializes more diagonals in each stage;
    a larger value retains sparse butterflies but spends more depths.  The
    choice of direct, BSGS, distributed, or custom execution remains
    independent.
    """

    stage_count: int
    imaginary_unit_correction: bool = False

    def __post_init__(self) -> None:
        if self.stage_count <= 0:
            raise ValueError('stage_count must be positive')

    def compile(
        self,
        *,
        slots: int,
        direction: TransformDirection,
        generator: int,
        scale: float = 1.0,
    ) -> tuple[DiagonalLinearTransform, ...]:
        r"""Compile $C$ or $T$ into ordered cyclic-diagonal stages.

        Compilation proceeds in four steps:

        1. validate the cyclotomic slot orbit and build root tables;
        2. construct one three-diagonal transform per radix-2 layer;
        3. compose adjacent layers according to ``stage_count``;
        4. fold `scale` into the first forward stage or final inverse
           stage so the normalization is applied once in the complete transform.

        `slots` is $S$ and every stored diagonal has axes `[slot]`. `generator`
        must enumerate $S$ distinct odd cyclotomic exponents modulo $4S$.
        `direction` chooses $C$ or $T$. Returns immutable CPU `complex128`
        stages in online execution order; compilation performs no encryption.
        """

        if slots <= 0 or slots & (slots - 1):
            raise ValueError('slots must be a positive power of two')
        if direction not in {'coeffs_to_slots', 'slots_to_coeffs'}:
            raise ValueError('unsupported transform direction')
        if scale == 0:
            raise ValueError('transform scale cannot be zero')
        log_slots = int(np.log2(slots))
        groups = _balanced_layer_groups(log_slots, self.stage_count)
        roots = _root_powers(slots)
        orbit = _cyclotomic_orbit(slots, generator)
        if direction == 'coeffs_to_slots':
            coefficients = _forward_layer_coefficients(
                roots,
                orbit,
                imaginary_unit_correction=self.imaginary_unit_correction,
            )
            stage_ids = list(range(log_slots - 1, -1, -1))
        else:
            coefficients = _inverse_layer_coefficients(
                roots,
                orbit,
                imaginary_unit_correction=self.imaginary_unit_correction,
            )
            stage_ids = list(range(log_slots))

        one_layer: list[DiagonalLinearTransform] = []
        for stage in stage_ids:
            one_layer.append(
                _radix2_layer_transform(
                    slots=slots,
                    coefficients=(
                        coefficients[stage],
                        coefficients[stage + log_slots],
                        coefficients[stage + 2 * log_slots],
                    ),
                    stage=stage,
                    name=f'{direction}_radix2_{stage}',
                )
            )

        compiled_stages: list[DiagonalLinearTransform] = []
        cursor = 0
        for group_index, group_size in enumerate(groups):
            composed = DiagonalLinearTransform(
                diagonals={0: np.ones(slots, dtype=np.complex128)},
                slots=slots,
                name='identity',
            )
            for layer in one_layer[cursor : cursor + group_size]:
                composed = _compose_diagonal_transforms(
                    composed,
                    layer,
                    name=f'{direction}_stage_{group_index}',
                )
            compiled_stages.append(composed)
            cursor += group_size

        scale_index = 0 if direction == 'coeffs_to_slots' else -1
        selected = compiled_stages[scale_index]
        compiled_stages[scale_index] = DiagonalLinearTransform(
            diagonals={
                offset: np.asarray(diagonal) * scale
                for offset, diagonal in selected.diagonals.items()
            },
            slots=slots,
            name=selected.name,
        )
        return tuple(compiled_stages)
