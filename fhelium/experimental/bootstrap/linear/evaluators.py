r"""Direct and baby-step/giant-step execution of diagonal linear maps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fhelium.values import Ciphertext, Plaintext, RotationKeySet
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.linear.transform import (
    DiagonalLinearTransform,
)

if TYPE_CHECKING:
    pass


class DirectDiagonalEvaluator:
    r"""Evaluate each cyclic diagonal independently, then rescale once.

    For every nonzero offset this strategy rotates the input, multiplies it by
    the corresponding encoded diagonal, and adds the product to an accumulator.
    All products have pending scale $\Delta_{\rm in}\Delta_p$, so the sum is
    rescaled only after every diagonal has been accumulated. It is simple but can require one
    direct rotation key for every nonzero diagonal.

    The input is a two-component coefficient-domain standard-RNS Q ciphertext
    with data axes `[component, *batch, limb, coefficient]`, ring extent $N$,
    and one homogeneous `prime_ids` tuple. If the removed Q group has product
    $M_d$, the functional result has the same component and batch
    axes, depth $\ell+1$, scale

    $$
    \Delta_{\rm out}=\frac{\Delta_{\rm in}\Delta_p}{M_d},
    $$

    and Q `prime_ids` with the complete group removed. The output remains in
    coefficient domain with standard residues; temporary diagonal plaintexts
    are NTT-domain Montgomery RNS. The result does not alias an input.
    """

    def required_depths(self, transform: Any) -> int:
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
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        transform: Any,
        *,
        rotation_keys: RotationKeySet,
        rotate: Callable[[Ciphertext, int], Ciphertext],
        encode_diagonal: Callable[..., Plaintext],
    ) -> Ciphertext:
        r"""Apply $L(x)$ directly and consume one Q rescale depth.

        The input must match the engine's slot count. For offset zero the input
        is reused directly; every other term requests one rotation through the
        supplied rotation-key strategy. Each diagonal is encoded at the input
        depth,
        multiplied into its rotated ciphertext, and accumulated at pending
        scale. A single final rescale advances the output by one depth.

        Raises:
            TypeError: If ``transform`` uses another stage representation.
            ValueError: If slot count or diagonal content is invalid.
        """

        engine = arithmetic.engine
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
                depth=ciphertext.depth,
                diagonal=diagonal,
            )
            term = engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(rotated), plaintext
            )
            result = term if result is None else engine.add(result, term)
        if result is None:
            raise ValueError('linear transform has no terms')
        return engine.rescale_to_next_depth(
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

    Thus direct and BSGS evaluators implement the same map and depth/scale/state
    transition; different grouping and CKKS rounding need not produce
    bit-identical residues. `hoist_baby_rotations` uses
    `engine.rotate_many_with_keys` only when direct baby-step keys are available.
    Compact power-of-two inventories compose rotations through the private
    key-aware evaluation helper. ``baby_steps_by_transform`` may override the
    fallback step for named compiled transforms, keeping a stage-specific BSGS
    schedule inspectable without changing the transform representation.
    ``aggregate_groups`` applies all giant-group plaintext rows to the shared
    baby ciphertexts in one represented RNS operation; it changes execution
    grouping but not the BSGS partition or arithmetic.
    """

    baby_step: int
    hoist_baby_rotations: bool = True
    aggregate_groups: bool = False
    baby_steps_by_transform: Mapping[str, int] | None = None

    def required_depths(self, transform: Any) -> int:
        """Return the single rescale consumed by one BSGS stage."""

        del transform
        return 1

    def __post_init__(self) -> None:
        if self.baby_step <= 0:
            raise ValueError('baby_step must be positive')
        stage_steps = dict(self.baby_steps_by_transform or {})
        if any(not name or step <= 0 for name, step in stage_steps.items()):
            raise ValueError(
                'stage-specific BSGS steps require non-empty names and '
                'positive values'
            )
        object.__setattr__(
            self,
            'baby_steps_by_transform',
            MappingProxyType(stage_steps),
        )

    def baby_step_for(self, transform: DiagonalLinearTransform) -> int:
        """Return the caller-selected BSGS step for one compiled transform."""

        assert self.baby_steps_by_transform is not None
        return self.baby_steps_by_transform.get(transform.name, self.baby_step)

    def _partition(
        self,
        transform: DiagonalLinearTransform,
    ) -> dict[int, list[tuple[int, int, np.ndarray]]]:
        r"""Map offset $k$ to execution coordinates $(g,b)$ with $k=g+b$.

        Values retain the original normalized offset because encoded-diagonal
        cache keys distinguish terms even when they share a baby rotation.
        """

        groups: dict[int, list[tuple[int, int, np.ndarray]]] = {}
        baby_step = self.baby_step_for(transform)
        for offset, diagonal in transform.normalized_diagonals().items():
            giant = (offset // baby_step) * baby_step
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
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        transform: Any,
        *,
        rotation_keys: RotationKeySet,
        rotate: Callable[[Ciphertext, int], Ciphertext],
        encode_diagonal: Callable[..., Plaintext],
    ) -> Ciphertext:
        r"""Execute shared baby rotations, group sums, and giant rotations.

        Each giant-group accumulator is rescaled before its giant rotation, so
        all group results have common depth and actual scale

        $$
        \Delta_{\rm out}=\frac{\Delta_{\rm in}\Delta_p}{M_d}.
        $$

        Here Delta_p is the selected diagonal plaintext scale and M_d is the
        dropped Q-group product.

        The input and output tensor/state requirements are identical to
        :class:`DirectDiagonalEvaluator`; evaluation is functional.
        """

        engine = arithmetic.engine
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
        ordered_groups = sorted(groups.items())

        def dense_plaintext_groups() -> list[list[Plaintext]]:
            """Prepare the caller-selected group-major diagonal matrix."""

            plaintext_groups: list[list[Plaintext | None]] = []
            prototype: Plaintext | None = None
            for giant, terms in ordered_groups:
                by_baby = {
                    baby: (offset, diagonal) for baby, offset, diagonal in terms
                }
                plaintext_group: list[Plaintext | None] = []
                for baby in used_babies:
                    entry = by_baby.get(baby)
                    if entry is None:
                        plaintext_group.append(None)
                        continue
                    offset, diagonal = entry
                    plaintext = encode_diagonal(
                        transform=transform,
                        offset=offset,
                        giant=giant,
                        depth=ciphertext.depth,
                        diagonal=diagonal,
                    )
                    if prototype is None:
                        prototype = plaintext
                    plaintext_group.append(plaintext)
                plaintext_groups.append(plaintext_group)
            if prototype is None or prototype.data is None:
                raise ValueError('linear transform has no terms')
            cache_key = (
                'bootstrap-zero-diagonal',
                prototype.device,
                prototype.depth,
                prototype.scale,
                prototype.prime_ids,
            )
            constant_cache = arithmetic.constant_cache
            zero = (
                None
                if constant_cache is None
                else constant_cache.get(cache_key)
            )
            if not isinstance(zero, Plaintext):
                zero = replace(prototype, data=torch.zeros_like(prototype.data))
                if constant_cache is not None:
                    constant_cache[cache_key] = zero
            return [
                [
                    zero if plaintext is None else plaintext
                    for plaintext in group
                ]
                for group in plaintext_groups
            ]

        direct_baby_keys = all(
            rotation_keys.get(step) is not None for step in nonzero_babies
        )
        if (
            self.aggregate_groups
            and self.hoist_baby_rotations
            and direct_baby_keys
            and nonzero_babies
        ):
            pending_groups = engine.sum_rotated_plaintext_product_groups(
                ciphertext,
                [
                    None if step == 0 else rotation_keys[step]
                    for step in used_babies
                ],
                dense_plaintext_groups(),
            ).unbind_batch()
        else:
            if (
                self.hoist_baby_rotations
                and direct_baby_keys
                and nonzero_babies
            ):
                rotations = engine.rotate_many_with_keys(
                    ciphertext,
                    [rotation_keys[step] for step in nonzero_babies],
                    use_hoisting=True,
                    output_domain="ntt",
                )
                baby_ciphertexts_ntt = dict(
                    zip(nonzero_babies, rotations, strict=True)
                )
                if 0 in used_babies:
                    baby_ciphertexts_ntt[0] = (
                        engine.coefficient_domain_to_ntt_domain(ciphertext)
                    )
            else:
                baby_ciphertexts_ntt = {
                    step: engine.coefficient_domain_to_ntt_domain(
                        ciphertext if step == 0 else rotate(ciphertext, step)
                    )
                    for step in used_babies
                }
            if self.aggregate_groups:
                pending_groups = engine.sum_plaintext_product_groups(
                    [baby_ciphertexts_ntt[baby] for baby in used_babies],
                    dense_plaintext_groups(),
                ).unbind_batch()
            else:
                pending_groups = []
                for giant, terms in ordered_groups:
                    inner_ciphertexts: list[Ciphertext] = []
                    inner_plaintexts: list[Plaintext] = []
                    for baby, offset, diagonal in sorted(terms):
                        plaintext = encode_diagonal(
                            transform=transform,
                            offset=offset,
                            giant=giant,
                            depth=ciphertext.depth,
                            diagonal=diagonal,
                        )
                        inner_ciphertexts.append(baby_ciphertexts_ntt[baby])
                        inner_plaintexts.append(plaintext)
                    pending_groups.append(
                        engine.sum_plaintext_products(
                            inner_ciphertexts,
                            inner_plaintexts,
                        )
                    )

        result: Ciphertext | None = None
        for (giant, _), inner in zip(
            ordered_groups, pending_groups, strict=True
        ):
            inner = engine.rescale_to_next_depth(
                engine.ntt_domain_to_coefficient_domain(inner)
            )
            partial = inner if giant == 0 else rotate(inner, giant)
            result = partial if result is None else engine.add(result, partial)
        if result is None:
            raise ValueError('linear transform has no terms')
        return result
