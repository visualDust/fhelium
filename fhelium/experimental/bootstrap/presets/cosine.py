r"""Preconfigured full-slot callables using cosine double-angle reduction.

Each function returns an engine-bound
:class:`fhelium.experimental.bootstrap.FullSlotBootstrap`.
The functions are conveniences rather than registered runtime objects; callers
may inspect, replace, or directly construct every component.

The versioned `logn16` names identify the measured component profile. Its
documented end-to-end configuration is derived from
`Preset.slots32768_scale50_levels27_int64` with `base_prime_bits=50` and uses
`galois_generator=5`. Construction itself validates transform slot counts,
structural-base/default-scale proximity, and depth, but does not certify the
encrypted input range or an application error budget.
"""

from __future__ import annotations

from fhelium.engine.ckks_engine import CkksEngine
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
        DiagonalBSGSEvaluator(baby_step=16, hoist_baby_rotations=True),
    )


def cosine_depth_refresh_logn16_v1(
    engine: CkksEngine,
) -> FullSlotBootstrap:
    r"""Construct the versioned 7/44 cosine full-slot callable.

    The composition uses two collapsed radix-2 stages in each transform, BSGS
    baby step 16, a degree-44 Chebyshev cosine seed, seven double-angle
    iterations, and raw periodic-reduction input bound $B=1024$. Input
    normalization $x=r/B$ is fused into CoeffsToSlots.

    The validated deployment configuration is derived from
    `Preset.slots32768_scale50_levels27_int64` with `base_prime_bits=50` and is bound
    to an engine using `galois_generator=5`. It accepts a final-public-level
    full-slot input near the default scale whose raw real and imaginary branch
    coordinates are within $[-1024,1024]$. The factory does not enforce the
    deployment identity or encrypted range; it only invokes
    `FullSlotBootstrap`'s structural and depth validation. Applications must
    test their own error distribution and range. Online execution requires the
    rotation, relinearization, and conjugation keys reported by the returned
    callable.

    Args:
        engine: Engine supplying slot count and Galois convention.

    Returns:
        An engine-bound full-slot callable ready for evaluation.
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
        ),
        slots_to_coeffs_compiler=compiler,
        slots_to_coeffs_evaluator=evaluator,
        modulus_raise_target_level=0,
    )


def cosine_depth_refresh_logn16_8_28_v1(
    engine: CkksEngine,
) -> FullSlotBootstrap:
    r"""Construct the versioned 8/28 cosine full-slot callable.

    The transform design matches :func:`cosine_depth_refresh_logn16_v1`. Its
    modular-reduction component uses a degree-28 Chebyshev seed followed by
    eight double-angle iterations, fused normalization, and raw input bound
    $B=1024$. It has the same documented $N=2^{16}$, 50-bit, generator-5 deployment
    requirements and the same non-enforcement of encrypted range. Application
    tests determine whether its numerical behavior is suitable for a workload.

    Args:
        engine: Engine supplying slot count and Galois convention.

    Returns:
        An engine-bound full-slot callable ready for evaluation.
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
        ),
        slots_to_coeffs_compiler=compiler,
        slots_to_coeffs_evaluator=evaluator,
        modulus_raise_target_level=0,
    )


__all__ = [
    'cosine_depth_refresh_logn16_8_28_v1',
    'cosine_depth_refresh_logn16_v1',
]
