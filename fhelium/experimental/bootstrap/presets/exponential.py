r"""Preconfigured full-slot callables using exponential-squaring reduction.

The versioned profile is documented for a configuration derived from
`Preset.slots32768_scale50_depth27_int64` with `galois_generator=5`.
The factory composes public components; it does not
certify encrypted range or application error.
"""

from __future__ import annotations

from fhelium.eager import Engine
from fhelium.experimental.bootstrap import (
    BalancedPowerEvaluator,
    DiagonalBSGSEvaluator,
    ExponentialSquaringReduction,
    FullSlotBootstrap,
    Radix2FourierTransformCompiler,
)


def exponential_depth_refresh_logn16_d16_v1(
    engine: Engine,
) -> FullSlotBootstrap:
    r"""Construct the versioned degree-16 exponential full-slot callable.

    The composition uses two collapsed radix-2 stages in each transform, BSGS
    baby step 16, a degree-16 balanced-power exponential seed, repeated
    squaring for raw input bound $B=1024$, fused normalization $x=r/B$, and sine
    extraction by conjugation.

    The documented configuration is derived from
    `Preset.slots32768_scale50_depth27_int64` with `galois_generator=5`.
    The input occupies ``bootstrap.input_depth = engine.max_depth - 1``.
    Entry preparation uses its actual scale; both resulting raw branch
    coordinates must lie in $[-1024,1024]$. The function does not enforce the deployment identity or
    encrypted range; `FullSlotBootstrap` checks structural-scale proximity,
    transform shape, and depth. Application tests must establish numerical
    suitability and depth budget. Online execution requires rotation,
    relinearization, and conjugation keys.

    Args:
        engine: Engine supplying slot count and Galois convention.

    Returns:
        A full-slot callable configured for the supplied Engine.
    """

    compiler = Radix2FourierTransformCompiler(stage_count=2)
    evaluator = DiagonalBSGSEvaluator(
        baby_step=16,
        hoist_baby_rotations=True,
    )
    return FullSlotBootstrap(
        engine,
        coeffs_to_slots_compiler=compiler,
        coeffs_to_slots_evaluator=evaluator,
        modular_reduction=ExponentialSquaringReduction(
            input_bound=1024,
            degree=16,
            evaluator=BalancedPowerEvaluator(skip_near_zero=1e-15),
            fuse_input_normalization=True,
        ),
        slots_to_coeffs_compiler=compiler,
        slots_to_coeffs_evaluator=evaluator,
        modulus_raise_target_depth=0,
        retain_diagonals=True,
        retain_constants=True,
    )


__all__ = ['exponential_depth_refresh_logn16_d16_v1']
