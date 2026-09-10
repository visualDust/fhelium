r"""Preconfigured full-slot callables using cosine double-angle reduction.

Each function returns a
:class:`fhelium.experimental.bootstrap.FullSlotBootstrap`.
The functions are conveniences rather than registered runtime objects; callers
may inspect, replace, or directly construct every component.

The versioned `logn16` names identify component profiles. Their
documented end-to-end configuration is derived from
`Preset.slots32768_scale50_depth27_int64` and
`galois_generator=5`. Construction validates transform slot counts,
structural-base/default-scale proximity, and depth; it does not certify an
encrypted input range or application error budget.
"""

from __future__ import annotations

from fhelium.eager import Engine
from fhelium.experimental.bootstrap import (
    BinaryDecompositionChebyshevEvaluator,
    ChebyshevInterpolator,
    CosineDoubleAngleReduction,
    DiagonalBSGSEvaluator,
    FullSlotBootstrap,
    Radix2FourierTransformCompiler,
)


def _bsgs_radix2_components():
    """Construct the shared radix-2 compiler and BSGS evaluator pair."""

    return (
        Radix2FourierTransformCompiler(stage_count=2),
        DiagonalBSGSEvaluator(
            baby_step=16,
            hoist_baby_rotations=True,
            aggregate_groups=True,
            baby_steps_by_transform={
                'coeffs_to_slots_stage_0': 2048,
                'slots_to_coeffs_stage_1': 4096,
            },
        ),
    )


def cosine_depth_refresh_logn16_v1(
    engine: Engine,
) -> FullSlotBootstrap:
    r"""Construct the versioned 7/44 cosine full-slot callable.

    The composition uses two collapsed radix-2 stages in each transform, BSGS
    baby step 16, a degree-44 Chebyshev cosine seed, seven double-angle
    iterations, and raw periodic-reduction input bound $B=1024$. Input
    normalization $x=r/B$ is fused into CoeffsToSlots.

    The documented deployment configuration is derived from
    `Preset.slots32768_scale50_depth27_int64` and
    uses `galois_generator=5`. The input is a full-slot ciphertext at
    ``bootstrap.input_depth = engine.max_depth - 1``. Entry preparation uses
    its actual scale; the resulting raw real and imaginary branch coordinates
    must lie within $[-1024,1024]$. The factory does not enforce the
    deployment identity or encrypted range; it invokes `FullSlotBootstrap`'s
    structural and depth validation. Applications test their own range and
    error distribution. Online execution requires the rotation,
    relinearization, and conjugation keys reported by the returned callable.

    Args:
        engine: Engine supplying slot count and Galois convention.

    Returns:
        A full-slot callable configured for the supplied Engine.
    """

    compiler, evaluator = _bsgs_radix2_components()
    return FullSlotBootstrap(
        engine,
        coeffs_to_slots_compiler=compiler,
        coeffs_to_slots_evaluator=evaluator,
        modular_reduction=CosineDoubleAngleReduction(
            input_bound=1024,
            double_angle_iterations=7,
            approximator=ChebyshevInterpolator(degree=44),
            evaluator=BinaryDecompositionChebyshevEvaluator(
                skip_near_zero=1e-15
            ),
            fuse_input_normalization=True,
            retain_ntt=True,
        ),
        slots_to_coeffs_compiler=compiler,
        slots_to_coeffs_evaluator=evaluator,
        modulus_raise_target_depth=0,
        retain_diagonals=True,
        retain_constants=True,
        batch_modular_branches=True,
    )


def cosine_depth_refresh_logn16_8_28_v1(
    engine: Engine,
) -> FullSlotBootstrap:
    r"""Construct the versioned 8/28 cosine full-slot callable.

    The transform design matches :func:`cosine_depth_refresh_logn16_v1`. Its
    modular-reduction component uses a degree-28 Chebyshev seed followed by
    eight double-angle iterations, fused normalization, and raw input bound
    $B=1024$. It has the same documented $N=2^{16}$, 50-bit, generator-5
    deployment requirements and the same caller-established encrypted range.
    Application tests determine whether its numerical behavior is suitable for
    a workload.

    Args:
        engine: Engine supplying slot count and Galois convention.

    Returns:
        A full-slot callable configured for the supplied Engine.
    """

    compiler, evaluator = _bsgs_radix2_components()
    return FullSlotBootstrap(
        engine,
        coeffs_to_slots_compiler=compiler,
        coeffs_to_slots_evaluator=evaluator,
        modular_reduction=CosineDoubleAngleReduction(
            input_bound=1024,
            double_angle_iterations=8,
            approximator=ChebyshevInterpolator(degree=28),
            evaluator=BinaryDecompositionChebyshevEvaluator(
                skip_near_zero=1e-15
            ),
            fuse_input_normalization=True,
            retain_ntt=True,
        ),
        slots_to_coeffs_compiler=compiler,
        slots_to_coeffs_evaluator=evaluator,
        modulus_raise_target_depth=0,
        retain_diagonals=True,
        retain_constants=True,
        batch_modular_branches=True,
    )


def cosine_depth_refresh_logn16_8_28_s2c3_v1(
    engine: Engine,
) -> FullSlotBootstrap:
    r"""Construct the 8/28 profile with three SlotsToCoeffs stages.

    CoeffsToSlots retains the two-stage schedule of
    :func:`cosine_depth_refresh_logn16_8_28_v1`.  SlotsToCoeffs uses three
    collapsed radix-2 stages with inspected BSGS steps 8, 256, and 4096.  The
    additional inverse-transform rescale advances the output by one depth and
    reduces the direct rotation-key inventory for the documented logN16 map.
    This factory retains the same $B=1024$ range precondition and does not
    choose a direct, composed, or mixed rotation-key inventory.
    """

    coeffs_to_slots_compiler, coeffs_to_slots_evaluator = (
        _bsgs_radix2_components()
    )
    slots_to_coeffs_compiler = Radix2FourierTransformCompiler(stage_count=3)
    slots_to_coeffs_evaluator = DiagonalBSGSEvaluator(
        baby_step=16,
        hoist_baby_rotations=True,
        aggregate_groups=True,
        baby_steps_by_transform={
            'slots_to_coeffs_stage_0': 8,
            'slots_to_coeffs_stage_1': 256,
            'slots_to_coeffs_stage_2': 4096,
        },
    )
    return FullSlotBootstrap(
        engine,
        coeffs_to_slots_compiler=coeffs_to_slots_compiler,
        coeffs_to_slots_evaluator=coeffs_to_slots_evaluator,
        modular_reduction=CosineDoubleAngleReduction(
            input_bound=1024,
            double_angle_iterations=8,
            approximator=ChebyshevInterpolator(degree=28),
            evaluator=BinaryDecompositionChebyshevEvaluator(
                skip_near_zero=1e-15
            ),
            fuse_input_normalization=True,
            retain_ntt=True,
        ),
        slots_to_coeffs_compiler=slots_to_coeffs_compiler,
        slots_to_coeffs_evaluator=slots_to_coeffs_evaluator,
        modulus_raise_target_depth=0,
        retain_diagonals=True,
        retain_constants=True,
        batch_modular_branches=True,
    )


def cosine_depth_refresh_logn16_7_32_bound512_v1(
    engine: Engine,
) -> FullSlotBootstrap:
    r"""Construct a versioned degree-32 cosine callable for raw bound $B=512$.

    The modular-reduction component uses seven double-angle iterations and a
    degree-32 Chebyshev seed.  It is intended for callers that establish both
    raw branch coordinates within $[-512,512]$; the narrower bound reduces the
    reducer's declared depth relative to the $B=1024$ profiles.
    Construction does not certify an encrypted range or application error.
    """

    compiler, evaluator = _bsgs_radix2_components()
    return FullSlotBootstrap(
        engine,
        coeffs_to_slots_compiler=compiler,
        coeffs_to_slots_evaluator=evaluator,
        modular_reduction=CosineDoubleAngleReduction(
            input_bound=512,
            double_angle_iterations=7,
            approximator=ChebyshevInterpolator(degree=32),
            evaluator=BinaryDecompositionChebyshevEvaluator(
                skip_near_zero=1e-15
            ),
            fuse_input_normalization=True,
            retain_ntt=True,
        ),
        slots_to_coeffs_compiler=compiler,
        slots_to_coeffs_evaluator=evaluator,
        modulus_raise_target_depth=0,
        retain_diagonals=True,
        retain_constants=True,
        batch_modular_branches=True,
    )



__all__ = [
    'cosine_depth_refresh_logn16_7_32_bound512_v1',
    'cosine_depth_refresh_logn16_8_28_s2c3_v1',
    'cosine_depth_refresh_logn16_8_28_v1',
    'cosine_depth_refresh_logn16_v1',
]
