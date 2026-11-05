r"""Direct and baby-step/giant-step execution of diagonal linear maps."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch

from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    Plaintext,
    RotationKey,
)
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.linear.transform import (
    DiagonalLinearTransform,
)


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
        rotation_paths: Mapping[int, tuple[RotationKey, ...]],
        rotate: Callable[[Ciphertext, int], Ciphertext],
        encode_diagonal: Callable[..., Plaintext | CompressedPlaintext],
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
        del rotation_paths
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
    r"""Execute a diagonal map with shared BSGS rotations and live-set scheduling.

    Splitting an offset as $k=g+b$ gives

    $$
    L(x)=\sum_g\operatorname{Rot}_g\left(
      \sum_b\operatorname{Rot}_{-g}(d_{g+b})\odot\operatorname{Rot}_b(x)
    \right).
    $$

    ``rotation_schedule="group_major"`` retains shared baby rotations and
    finishes at most ``group_batch_size`` giant groups at a time. Each baby is released after its last
    group consumer. ``"baby_major"`` generates one baby at a time and updates
    the giant-group accumulators. ``"auto"`` selects baby-major when there
    are fewer giant groups than distinct babies, otherwise group-major.
    The selected traversal is available through :meth:`schedule_for`.
    Group batches share ciphertext loads in one represented weighted sum; the
    default batch size is sixteen, and one selects immediate single-group completion.

    ``product_implementation`` and ``product_chunk_size`` select the weighted
    product consumer. ``None`` retains native selection. Group-major traversal
    exposes multi-term rows; baby-major traversal honors the same request with
    one-term calls and preserves its live set. ``"auto"`` resolves by operand
    representation, device and registered Backend capabilities.

    ``giant_step_reduction="aggregate"`` streams giant key products into one
    QP accumulator, then divides once by $P M_d$. ``"independent"`` rescales
    and key-switches each giant separately before summing its Q result. These
    schedules differ in rounding and temporary storage. At depth $d$, diagonal
    scale $\Delta_p$ and Q-group product $M_d$, the coefficient/
    standard Q output has two components, depth $d+1$ and actual scale
    $\Delta_{out}=\Delta_{in}\Delta_p/M_d$. Input batch axes and ring extent
    are preserved. Accumulation mutates only invocation-owned products.

    Direct baby keys share one hybrid decomposition when
    ``hoist_baby_rotations`` is true. Composed inventories use the supplied
    rotation callable. ``baby_steps_by_transform`` selects named-stage steps.
    """

    baby_step: int
    hoist_baby_rotations: bool = True
    rotation_schedule: Literal["auto", "group_major", "baby_major"] = "auto"
    baby_steps_by_transform: Mapping[str, int] | None = None
    group_batch_size: int = 16
    product_implementation: str | None = None
    product_chunk_size: int | None = None
    giant_step_reduction: Literal["aggregate", "independent"] = "aggregate"
    modup_method: Literal["exact", "crt", "normalized_crt"] | None = None

    def required_depths(self, transform: Any) -> int:
        """Return the single rescale consumed by one BSGS stage."""
        del transform
        return 1

    def __post_init__(self) -> None:
        if self.baby_step <= 0 or self.group_batch_size <= 0:
            raise ValueError('baby_step and group_batch_size must be positive')
        if self.rotation_schedule not in {"auto", "group_major", "baby_major"}:
            raise ValueError('unknown BSGS rotation schedule')
        if self.giant_step_reduction not in {"aggregate", "independent"}:
            raise ValueError(
                "giant_step_reduction must be aggregate or independent"
            )
        if self.modup_method not in {None, "exact", "crt", "normalized_crt"}:
            raise ValueError("Unknown key-switch ModUp method")
        stage_steps = dict(self.baby_steps_by_transform or {})
        if any(not name or step <= 0 for name, step in stage_steps.items()):
            raise ValueError(
                'stage-specific BSGS steps require non-empty names and positive values'
            )
        object.__setattr__(
            self, 'baby_steps_by_transform', MappingProxyType(stage_steps)
        )

    def baby_step_for(self, transform: DiagonalLinearTransform) -> int:
        """Return the caller-selected BSGS step for one compiled transform."""
        assert self.baby_steps_by_transform is not None
        return self.baby_steps_by_transform.get(transform.name, self.baby_step)

    def _partition(
        self,
        transform: DiagonalLinearTransform,
    ) -> dict[int, list[tuple[int, int, np.ndarray]]]:
        r"""Retain the original diagonal offset under coordinates $k=g+b$."""
        groups: dict[int, list[tuple[int, int, np.ndarray]]] = {}
        baby_step = self.baby_step_for(transform)
        for offset, diagonal in transform.normalized_diagonals().items():
            giant = (offset // baby_step) * baby_step
            groups.setdefault(giant, []).append(
                (offset - giant, offset, diagonal)
            )
        return groups

    def schedule_for(self, transform: DiagonalLinearTransform) -> str:
        """Select a traversal from the stage's represented baby/group counts."""
        groups = self._partition(transform)
        return self._schedule(groups)

    def _schedule(
        self, groups: dict[int, list[tuple[int, int, np.ndarray]]]
    ) -> str:
        if self.rotation_schedule != "auto":
            return self.rotation_schedule
        babies = {baby for terms in groups.values() for baby, _, _ in terms}
        return "baby_major" if len(groups) < len(babies) else "group_major"

    def required_rotation_offsets(self, transform: Any) -> tuple[int, ...]:
        """Return the union of nonzero baby and giant rotations."""
        if not isinstance(transform, DiagonalLinearTransform):
            raise TypeError('DiagonalBSGSEvaluator requires diagonal stages')
        offsets: set[int] = set()
        for giant, terms in self._partition(transform).items():
            if giant:
                offsets.add(giant)
            offsets.update(baby for baby, _, _ in terms if baby)
        return tuple(sorted(offsets))

    def baby_rotation_offsets(
        self,
        transform: DiagonalLinearTransform,
    ) -> tuple[int, ...]:
        """Return BSGS baby steps independently of decomposition sharing."""
        return tuple(
            sorted(
                {
                    baby
                    for terms in self._partition(transform).values()
                    for baby, _, _ in terms
                    if baby
                }
            )
        )

    def _babies(
        self,
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        steps: list[int],
        rotation_paths: Mapping[int, tuple[RotationKey, ...]],
        rotate: Callable[[Ciphertext, int], Ciphertext],
    ) -> Iterator[tuple[int, Ciphertext]]:
        """Yield one NTT baby result while reusing available direct-key hoisting."""
        engine = arithmetic.engine
        nonzero = [step for step in steps if step]
        paths = [
            rotation_paths[
                RotationKey.normalize_step(
                    step, ring_dimension=engine.ring_dimension
                )
            ]
            for step in nonzero
        ]
        direct = all(len(path) == 1 for path in paths)
        shared_layout = (
            direct
            and bool(paths)
            and all(
                (path[0].prime_ids, path[0].digit_prime_ids)
                == (paths[0][0].prime_ids, paths[0][0].digit_prime_ids)
                for path in paths
            )
        )
        if self.hoist_baby_rotations and shared_layout:
            yield from zip(
                steps,
                engine.iter_rotations_with_keys(
                    ciphertext,
                    [
                        rotation_paths[
                            RotationKey.normalize_step(
                                step, ring_dimension=engine.ring_dimension
                            )
                        ][0]
                        if step
                        else None
                        for step in steps
                    ],
                    output_domain="ntt",
                    modup_method=self.modup_method or arithmetic.modup_method,
                ),
                strict=True,
            )
        else:
            for step in steps:
                yield (
                    step,
                    engine.coefficient_domain_to_ntt_domain(
                        ciphertext if step == 0 else rotate(ciphertext, step)
                    ),
                )

    def evaluate(
        self,
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        transform: Any,
        *,
        rotation_paths: Mapping[int, tuple[RotationKey, ...]],
        rotate: Callable[[Ciphertext, int], Ciphertext],
        encode_diagonal: Callable[..., Plaintext | CompressedPlaintext],
    ) -> Ciphertext:
        r"""Evaluate $L(x)$ while retaining only the chosen BSGS live set.

        Group-major uses the ordinary represented plaintext-product sum for
        a bounded block of groups. Baby-major accumulates terms in the same baby order.
        Giant groups finish in increasing offset order using the selected
        shared or independent quotient schedule. Composed rotation inventories
        retain prefix key switches; the aggregate schedule combines their final
        switches. The shared quotient divides by the complete Q depth group.
        Diagonal material preparation and retention remain caller-owned.
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
        ordered_groups = sorted(groups.items())
        used_babies = sorted(
            {baby for terms in groups.values() for baby, _, _ in terms}
        )
        products = self._group_products(
            arithmetic,
            ciphertext,
            transform,
            groups,
            ordered_groups,
            used_babies,
            rotation_paths=rotation_paths,
            rotate=rotate,
            encode_diagonal=encode_diagonal,
        )
        if self.giant_step_reduction == "aggregate":
            return engine.sum_rotations(
                self._rotation_terms(arithmetic, products, rotation_paths),
                rescale=True,
                modup_method=self.modup_method or arithmetic.modup_method,
            )
        result: Ciphertext | None = None
        for giant, inner in products:
            inner = engine.rescale_to_next_depth(
                engine.ntt_domain_to_coefficient_domain(inner)
            )
            if giant:
                inner = rotate(inner, giant)
            result = inner if result is None else engine.add(result, inner)
        if result is None:
            raise ValueError("linear transform has no terms")
        return result

    def _rotation_terms(
        self,
        arithmetic: BootstrapArithmetic,
        products: Iterator[tuple[int, Ciphertext]],
        rotation_paths: Mapping[int, tuple[RotationKey, ...]],
    ) -> Iterator[tuple[Ciphertext, RotationKey | None]]:
        """Resolve giant paths and leave the final key switch for the sum."""
        engine = arithmetic.engine
        for giant, inner in products:
            step = RotationKey.normalize_step(
                giant, ring_dimension=engine.ring_dimension
            )
            if not step:
                yield inner, None
                del inner
                continue
            path = rotation_paths[step]
            for key in path[:-1]:
                inner = engine.rotate_with_key(
                    inner,
                    key,
                    output_domain="ntt",
                    modup_method=self.modup_method or arithmetic.modup_method,
                )
            yield inner, path[-1]
            del inner

    def _group_products(
        self,
        arithmetic,
        ciphertext,
        transform,
        groups,
        ordered_groups,
        used_babies,
        *,
        rotation_paths,
        rotate,
        encode_diagonal,
    ) -> Iterator[tuple[int, Ciphertext]]:
        """Yield NTT giant-group products in the selected bounded traversal."""
        engine = arithmetic.engine

        def encode(
            giant: int, offset: int, diagonal: np.ndarray
        ) -> Plaintext | CompressedPlaintext:
            plaintext = encode_diagonal(
                transform=transform,
                offset=offset,
                giant=giant,
                depth=ciphertext.depth,
                diagonal=diagonal,
            )
            # Native grouped products require one plaintext axis per batch axis.
            if ciphertext.data.ndim > 3:
                assert plaintext.data is not None
                batch_axes = (1,) * (ciphertext.data.ndim - 3)
                fields = {
                    "data": plaintext.data.reshape(
                        *batch_axes, *plaintext.data.shape[-2:]
                    )
                }
                if (
                    isinstance(plaintext, CompressedPlaintext)
                    and plaintext.implicit_data is not None
                ):
                    fields["implicit_data"] = plaintext.implicit_data.reshape(
                        *batch_axes, plaintext.data.size(-2)
                    )
                plaintext = replace(plaintext, **fields)
            return plaintext

        if self._schedule(groups) == "group_major":
            babies = dict(
                self._babies(
                    arithmetic, ciphertext, used_babies, rotation_paths, rotate
                )
            )
            last_group = {
                baby: giant
                for giant, terms in ordered_groups
                for baby, _, _ in terms
            }
            zero: Plaintext | CompressedPlaintext | None = None
            for start in range(0, len(ordered_groups), self.group_batch_size):
                block = ordered_groups[start : start + self.group_batch_size]
                block_babies = sorted(
                    {baby for _, terms in block for baby, _, _ in terms}
                )
                rows: list[list[Plaintext | CompressedPlaintext]] = []
                for giant, terms in block:
                    entries = {
                        baby: encode(giant, offset, diagonal)
                        for baby, offset, diagonal in sorted(terms)
                    }
                    if len(entries) < len(block_babies) and zero is None:
                        prototype = next(iter(entries.values()))
                        assert prototype.data is not None
                        fields = {"data": torch.zeros_like(prototype.data)}
                        if (
                            isinstance(prototype, CompressedPlaintext)
                            and prototype.implicit_data is not None
                        ):
                            fields["implicit_data"] = torch.zeros_like(
                                prototype.implicit_data
                            )
                        zero = replace(prototype, **fields)
                    row: list[Plaintext | CompressedPlaintext] = []
                    for baby in block_babies:
                        entry = entries.get(baby, zero)
                        assert entry is not None
                        row.append(entry)
                    rows.append(row)
                block_result = engine.sum_plaintext_product_groups(
                    [babies[baby] for baby in block_babies],
                    rows,
                    implementation=self.product_implementation,
                    term_chunk_size=self.product_chunk_size,
                )
                del rows, row, entries, entry
                for baby in block_babies:
                    if last_group[baby] <= block[-1][0]:
                        del babies[baby]
                # Views share the block payload until all groups have finished.
                for index, (giant, _) in enumerate(block):
                    inner = block_result.select_batch(index)
                    yield giant, inner
                    del inner
                del block_result
        else:
            by_baby: dict[int, list[tuple[int, int, np.ndarray]]] = {}
            for giant, terms in ordered_groups:
                for baby, offset, diagonal in terms:
                    by_baby.setdefault(baby, []).append(
                        (giant, offset, diagonal)
                    )
            partials: dict[int, Ciphertext] = {}
            for baby, rotated in self._babies(
                arithmetic, ciphertext, used_babies, rotation_paths, rotate
            ):
                for giant, offset, diagonal in by_baby[baby]:
                    plaintext = encode(giant, offset, diagonal)
                    if (
                        self.product_implementation is not None
                        or self.product_chunk_size is not None
                    ):
                        term = engine.sum_plaintext_products(
                            (rotated,),
                            (plaintext,),
                            implementation=self.product_implementation,
                            term_chunk_size=self.product_chunk_size,
                        )
                    else:
                        term = engine.multiply_plaintext(rotated, plaintext)
                    partial = partials.get(giant)
                    partials[giant] = (
                        term if partial is None else engine.add(partial, term)
                    )
                    del term, partial
                del rotated
            for giant, _ in ordered_groups:
                inner = partials.pop(giant)
                yield giant, inner
                del inner
