r"""Slot-wise linear transforms in cyclic-diagonal form.

A :class:`LinearTransform` stores the diagonals $d_k$ of
$L(x)=\sum_k d_k\odot\operatorname{Rot}_k(x)$ with the ``torch.roll`` rotation
convention. :func:`radix2_dft` factors the CKKS coefficient/slot maps into such
stages. :class:`BsgsPlan` and :class:`DirectPlan` select the evaluation schedule
and report rotation steps. :func:`apply` evaluates ordered stages;
:func:`prepare` encodes their diagonals and selects keys once so repeated
:func:`apply` calls and Compile capture only read the prepared materials.

A plan, built-in or custom, provides ``required_depths(stage)``,
``required_rotation_offsets(stage)`` and ``evaluate(arithmetic, ciphertext,
stage, *, rotation_paths, rotate, encode_diagonal)``. ``rotation_paths`` maps
each required offset to its selected primitive keys, ``rotate(value, offset)``
performs that rotation, and ``encode_diagonal(transform=, offset=, giant=,
depth=, diagonal=)`` returns the NTT-domain plaintext of ``diagonal`` at the
stage's :meth:`~fhelium.algorithms.Arithmetic.plaintext_scale`, keyed by
``(transform, offset, giant)`` so preparation records it and prepared calls
reuse it. With input depth $d$, the result is at depth
$d+$ ``required_depths(stage)`` with scale ``arithmetic.target_scales`` of that
depth, two components and the input batch axes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


from fhelium.algorithms.linear._evaluators import BsgsPlan, DirectPlan
from fhelium.algorithms.linear._radix2 import Radix2FourierTransformCompiler
from fhelium.algorithms.linear._transform import LinearTransform


def radix2_dft(
    slots: int,
    *,
    direction: Literal["coeffs_to_slots", "slots_to_coeffs"],
    stages: int,
    generator: int = 5,
    scale: float = 1.0,
    scale_stage: int | None = None,
    imaginary_unit_correction: bool = False,
) -> tuple[LinearTransform, ...]:
    r"""Factor the CKKS map $C$ (coefficients to slots) or $T$ into ``stages`` maps.

    The $\log_2 S$ radix-2 butterfly layers are grouped into ``stages``
    balanced products in execution order. ``scale`` is folded into the first
    forward or last inverse stage, or into ``scale_stage`` when given, so the
    composed map equals ``scale`` times the unscaled transform with
    $T(C(a))=Sa$. Stage names are ``{direction}_stage_{i}``.
    """

    return Radix2FourierTransformCompiler(
        stage_count=stages,
        scale_stage=scale_stage,
        imaginary_unit_correction=imaginary_unit_correction,
    ).compile(
        slots=slots, direction=direction, generator=generator, scale=scale
    )


def plan(transform: LinearTransform | Sequence[LinearTransform]) -> BsgsPlan:
    r"""Return a BSGS plan with baby step $2^{\lceil\log_2\sqrt{n}\rceil}$.

    $n$ is the largest diagonal count among the stages.
    """

    stages = _stages(transform)
    count = max(len(stage.normalized_diagonals()) for stage in stages)
    return BsgsPlan(baby_step=1 << math.ceil(math.log2(math.sqrt(count))))


@dataclass
class PreparedLinearTransform:
    """Encoded diagonals and selected rotation keys owned by the caller.

    The object is produced by :func:`prepare` for one input depth, scale and
    device. Passing it to :func:`apply` reuses its materials without encoding;
    drop the object to release them.
    """

    stages: tuple[LinearTransform, ...]
    plan: Any
    output_scale: float | None
    _diagonals: dict = field(default_factory=dict, repr=False)
    _rotation_paths: dict = field(default_factory=dict, repr=False)


def _stages(transform) -> tuple[LinearTransform, ...]:
    return (
        (transform,)
        if isinstance(transform, LinearTransform)
        else tuple(transform)
    )


def _run(engine, ciphertext, stages, selected, keys, output_scale, prepared):
    from fhelium.algorithms._arithmetic import Arithmetic
    from fhelium.algorithms.linear._execution import _apply_linear_transform

    depth = ciphertext.depth + sum(
        selected.required_depths(stage) for stage in stages
    )
    scales = list(engine.config.scales)
    if output_scale is not None:
        scales[depth] = float(output_scale)
    return _apply_linear_transform(
        Arithmetic(engine, {}, _scales=tuple(scales)),
        stages,
        selected,
        ciphertext,
        evaluation_keys=keys,
        key_bindings=prepared._rotation_paths,
        diagonal_cache=prepared._diagonals,
        retain_diagonals=True,
        prepare_materials=keys is not None,
    )


def prepare(
    engine,
    like,
    transform: LinearTransform | Sequence[LinearTransform],
    plan: Any = None,
    *,
    keys,
    output_scale: float | None = None,
) -> PreparedLinearTransform:
    """Encode diagonals and select rotation keys for inputs with the state of ``like``.

    The preparation evaluates the stages once on ``like`` (any ciphertext with
    the intended depth, scale, batch shape and device); the encoding RNG
    advances once per diagonal. The result does not retain ``like``.
    """

    stages = _stages(transform)
    selected = globals()["plan"](stages) if plan is None else plan
    prepared = PreparedLinearTransform(stages, selected, output_scale)
    _run(engine, like, stages, selected, keys, output_scale, prepared)
    return prepared


def apply(
    engine,
    ciphertext,
    transform: LinearTransform | Sequence[LinearTransform] | None = None,
    plan: Any = None,
    *,
    keys=None,
    prepared: PreparedLinearTransform | None = None,
    output_scale: float | None = None,
):
    r"""Evaluate ordered stages $L_{m-1}\circ\cdots\circ L_0$ on a ciphertext.

    Each BSGS or direct stage multiplies encoded diagonals at plaintext scale
    $\Delta_p=s_{d+1}\rho_d/\Delta_{in}$ and rescales once, so stage $i$ moves
    depth $d\to d+1$ with actual scale $s_{d+1}$: the configured
    $\Delta_{d+1}$, or ``output_scale`` for the last stage. The coefficient-
    domain input keeps two components and its batch axes. Without
    ``prepared``, diagonals are encoded and keys selected from ``keys`` for
    this call; with ``prepared``, its stages, plan, output scale and materials
    are used and ``keys`` is not consulted.
    """

    if prepared is not None:
        return _run(
            engine,
            ciphertext,
            prepared.stages,
            prepared.plan,
            None,
            prepared.output_scale,
            prepared,
        )
    stages = _stages(transform)
    selected = globals()["plan"](stages) if plan is None else plan
    scratch = PreparedLinearTransform(stages, selected, output_scale)
    return _run(engine, ciphertext, stages, selected, keys, output_scale, scratch)


__all__ = [
    "BsgsPlan",
    "DirectPlan",
    "LinearTransform",
    "PreparedLinearTransform",
    "apply",
    "plan",
    "prepare",
    "radix2_dft",
]
