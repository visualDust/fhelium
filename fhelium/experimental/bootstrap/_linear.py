r"""Compile linear maps into stages, then execute them with a separate strategy.

The compiler/evaluator split defines the central responsibilities in this module.  A
compiler chooses a representation and an algebraic stage decomposition.  An
evaluator interprets each stage as homomorphic operations and declares the
rotation keys and CKKS levels it needs.  The built-in representation is a
cyclic-diagonal map, but custom compiler/evaluator pairs may use another
representation without changing the full-slot evaluator.

For a length-$S$ slot-coordinate vector $x$, the built-in representation is

$$
L(x)=\sum_{k\in K} d_k\mathbin{\odot}\operatorname{Rot}_k(x),
$$

where every diagonal $d_k$ has shape `[slot]`, $\odot$ is elementwise
multiplication, and `Rot_k` follows `numpy.roll(x, k)` and FHElium's public
signed slot-rotation convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import torch

from fhelium.core import Ciphertext, Plaintext, RotationKeySet

if TYPE_CHECKING:
    from fhelium.engine.ckks_engine import CkksEngine

ArrayLike: TypeAlias = Sequence[complex] | np.ndarray | torch.Tensor
TransformDirection = Literal['coeffs_to_slots', 'slots_to_coeffs']


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

        `values` must have exact shape `[slot]`. The returned CPU `complex128`
        array has shape `[slot]`. This plaintext oracle does not encode, rescale,
        consume levels, or model CKKS error.
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


@dataclass(frozen=True)
class DirectDiagonalEvaluator:
    r"""Evaluate each cyclic diagonal independently, then rescale once.

    For every nonzero offset this strategy rotates the input, multiplies it by
    the corresponding encoded diagonal, and adds the product to an accumulator.
    All products have pending scale $\Delta_{\rm in}\Delta_0$, so the sum is
    rescaled only after every diagonal has been accumulated. It is simple but can require one
    exact rotation for every nonzero diagonal.

    The input is a two-component coefficient-domain standard-RNS Q ciphertext
    with data axes `[component, *batch, limb, coefficient]`, ring extent $N$,
    and one homogeneous `prime_ids` tuple. If the leading active prime is
    $q_{\rm drop}$, the functional result has the same component and batch
    axes, level $\ell+1$, scale

    $$
    \Delta_{\rm out}=\frac{\Delta_{\rm in}\Delta_0}{q_{\rm drop}},
    $$

    and Q `prime_ids` with the leading row removed. The output remains in
    coefficient domain with standard residues; temporary diagonal plaintexts
    are NTT-domain Montgomery RNS. The result does not alias an input.
    """

    def required_levels(self, transform: Any) -> int:
        """Return the single rescale consumed by one diagonal stage."""

        del transform
        return 1

    def required_rotation_offsets(
        self,
        transform: Any,
    ) -> tuple[int, ...]:
        """Return direct non-zero diagonal offsets."""

        if not isinstance(transform, DiagonalLinearTransform):
            raise TypeError('DirectDiagonalEvaluator requires diagonal stages')
        return tuple(
            sorted(
                offset for offset in transform.normalized_diagonals() if offset
            )
        )

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        transform: Any,
        *,
        rotation_keys: RotationKeySet,
        rotate: Callable[[Ciphertext, int], Ciphertext],
        encode_diagonal: Callable[..., Plaintext],
    ) -> Ciphertext:
        r"""Apply $L(x)$ directly and consume one Q rescale level.

        The input must match the engine's slot count. For offset zero the input
        is reused directly; every other term requests one rotation through the
        supplied rotation-key strategy. Each diagonal is encoded at the input
        level,
        multiplied into its rotated ciphertext, and accumulated at pending
        scale. A single final rescale advances the output by one level.

        Raises:
            TypeError: If ``transform`` uses another stage representation.
            ValueError: If slot count or diagonal content is invalid.
        """

        if not isinstance(transform, DiagonalLinearTransform):
            raise TypeError('DirectDiagonalEvaluator requires diagonal stages')
        del rotation_keys
        if transform.slots != engine.num_slots:
            raise ValueError('linear transform has the wrong slot count')
        ciphertext.assert_state(
            polynomial_domain='coefficient',
            residue_representation='standard',
            modulus_basis='Q',
            components=2,
        )
        result: Ciphertext | None = None
        for offset, diagonal in sorted(
            transform.normalized_diagonals().items()
        ):
            rotated = ciphertext if offset == 0 else rotate(ciphertext, offset)
            plaintext = encode_diagonal(
                transform=transform,
                offset=offset,
                giant=0,
                level=ciphertext.level,
                diagonal=diagonal,
            )
            term = engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(rotated), plaintext
            )
            result = term if result is None else engine.add(result, term)
        if result is None:
            raise ValueError('linear transform has no terms')
        return engine.rescale_to_next_level(
            engine.ntt_domain_to_coefficient_domain(result)
        )


@dataclass(frozen=True)
class DiagonalBSGSEvaluator:
    r"""Evaluate the same diagonal map with a BSGS rotation schedule.

    An offset $k$ is split as $k=g+b$, where $g$ is a multiple of
    `baby_step`. Baby rotations of the input are shared across
    giant groups.  Each group's diagonals are shifted to compensate for the
    final giant rotation, its plaintext products are accumulated and rescaled,
    and then the group result is giant-rotated into place.

    Algebraically, each term is unchanged because

    $$
    \operatorname{Rot}_g\left(
      \operatorname{Rot}_b(x)\odot\operatorname{Rot}_{-g}(d_{g+b})
    \right)
    =\operatorname{Rot}_{g+b}(x)\odot d_{g+b}.
    $$

    Thus direct and BSGS evaluators implement the same map and level/scale/state
    transition; different grouping and CKKS rounding need not produce
    bit-identical residues. `hoist_baby_rotations` uses
    `engine.rotate_many_with_keys` only when exact baby keys are available.
    Compact power-of-two inventories compose rotations through the private
    key-aware evaluation helper.
    """

    baby_step: int
    hoist_baby_rotations: bool = True

    def required_levels(self, transform: Any) -> int:
        """Return the single rescale consumed by one BSGS stage."""

        del transform
        return 1

    def __post_init__(self) -> None:
        if self.baby_step <= 0:
            raise ValueError('baby_step must be positive')

    def _partition(
        self,
        transform: DiagonalLinearTransform,
    ) -> dict[int, list[tuple[int, int, np.ndarray]]]:
        r"""Map offset $k$ to execution coordinates $(g,b)$ with $k=g+b$.

        Values retain the original normalized offset because encoded-diagonal
        cache keys distinguish terms even when they share a baby rotation.
        """

        groups: dict[int, list[tuple[int, int, np.ndarray]]] = {}
        for offset, diagonal in transform.normalized_diagonals().items():
            giant = (offset // self.baby_step) * self.baby_step
            baby = offset - giant
            groups.setdefault(giant, []).append((baby, offset, diagonal))
        return groups

    def required_rotation_offsets(
        self,
        transform: Any,
    ) -> tuple[int, ...]:
        """Return the union of nonzero baby and giant rotations."""

        if not isinstance(transform, DiagonalLinearTransform):
            raise TypeError('DiagonalBSGSEvaluator requires diagonal stages')
        offsets: set[int] = set()
        for giant, terms in self._partition(transform).items():
            if giant:
                offsets.add(giant)
            offsets.update(baby for baby, _, _ in terms if baby)
        return tuple(sorted(offsets))

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        transform: Any,
        *,
        rotation_keys: RotationKeySet,
        rotate: Callable[[Ciphertext, int], Ciphertext],
        encode_diagonal: Callable[..., Plaintext],
    ) -> Ciphertext:
        r"""Execute shared baby rotations, group sums, and giant rotations.

        Each giant-group accumulator is rescaled before its giant rotation, so
        all group results have common level and actual scale

        $$
        \Delta_{\rm out}=\Delta_{\rm in}\Delta_0/q_{\rm drop}.
        $$

        The input and output tensor/state requirements are identical to
        :class:`DirectDiagonalEvaluator`; evaluation is functional.
        """

        if not isinstance(transform, DiagonalLinearTransform):
            raise TypeError('DiagonalBSGSEvaluator requires diagonal stages')
        if transform.slots != engine.num_slots:
            raise ValueError('linear transform has the wrong slot count')
        ciphertext.assert_state(
            polynomial_domain='coefficient',
            residue_representation="standard",
            modulus_basis='Q',
            components=2,
        )
        groups = self._partition(transform)
        used_babies = sorted(
            {baby for terms in groups.values() for baby, _, _ in terms}
        )
        nonzero_babies = [step for step in used_babies if step]
        if (
            self.hoist_baby_rotations
            and all(
                rotation_keys.get(step) is not None for step in nonzero_babies
            )
            and nonzero_babies
        ):
            rotations = engine.rotate_many_with_keys(
                ciphertext,
                [rotation_keys[step] for step in nonzero_babies],
                use_hoisting=True,
            )
            baby_ciphertexts = dict(zip(nonzero_babies, rotations, strict=True))
            if 0 in used_babies:
                baby_ciphertexts[0] = ciphertext
        else:
            baby_ciphertexts = {
                step: ciphertext if step == 0 else rotate(ciphertext, step)
                for step in used_babies
            }

        result: Ciphertext | None = None
        for giant, terms in sorted(groups.items()):
            inner: Ciphertext | None = None
            for baby, offset, diagonal in sorted(terms):
                plaintext = encode_diagonal(
                    transform=transform,
                    offset=offset,
                    giant=giant,
                    level=ciphertext.level,
                    diagonal=diagonal,
                )
                term = engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(
                        baby_ciphertexts[baby]
                    ),
                    plaintext,
                )
                inner = term if inner is None else engine.add(inner, term)
            if inner is None:
                continue
            inner = engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(inner)
            )
            partial = inner if giant == 0 else rotate(inner, giant)
            result = partial if result is None else engine.add(result, partial)
        if result is None:
            raise ValueError('linear transform has no terms')
        return result


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
    r"""Return the exact cyclic-diagonal composition $B(A(x))$.

    If $A_i$ and $B_j$ are diagonals at offsets $i$ and $j$, their composed
    contribution is $B_j\odot\operatorname{Rot}_j(A_i)$ at offset $i+j$.
    This algebraic composition collapses several one-level butterflies into a
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
    consumes fewer CKKS levels but materializes more diagonals in each stage;
    a larger value retains sparse butterflies but spends more levels.  The
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
        2. construct one exact three-diagonal transform per radix-2 layer;
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
