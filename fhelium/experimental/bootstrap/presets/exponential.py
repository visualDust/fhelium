r"""Preconfigured full-slot callables using exponential-squaring reduction.

The versioned profile is documented for a configuration derived from
`Preset.slots32768_scale50_levels27_int64` with `base_prime_bits=50` and an engine
using `galois_generator=5`. The factory composes public components; it does not
certify encrypted range or application error.
"""

from __future__ import annotations

from fhelium.engine.ckks_engine import CkksEngine
from fhelium.experimental.bootstrap import (
    BalancedPowerEvaluator,
    DiagonalBSGSEvaluator,
    ExponentialSquaringReduction,
    FullSlotBootstrap,
    Radix2FourierTransformCompiler,
)


def exponential_depth_refresh_logn16_d16_v1(
    engine: CkksEngine,
) -> FullSlotBootstrap:
    r"""Construct the versioned degree-16 exponential full-slot callable.

    The composition uses two collapsed radix-2 stages in each transform, BSGS
    baby step 16, a degree-16 balanced-power exponential seed, repeated
    squaring for raw input bound $B=1024$, fused normalization $x=r/B$, and sine
    extraction by conjugation.

    The measured configuration is derived from
    `Preset.slots32768_scale50_levels27_int64` with `base_prime_bits=50` and is bound
    to an engine using `galois_generator=5`. It accepts a final-public-level
    input near default scale with both raw branch coordinates in
    $[-1024,1024]$. The function does not enforce the deployment identity or
    encrypted range; `FullSlotBootstrap` checks structural-scale proximity,
    transform shape, and depth. Application tests must establish numerical
    suitability and level budget. Online execution requires rotation,
    relinearization, and conjugation keys.

    Args:
        engine: Engine supplying slot count and Galois convention.

    Returns:
        An engine-bound full-slot callable ready for evaluation.
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
        modulus_raise_target_level=0,
    )


__all__ = ['exponential_depth_refresh_logn16_d16_v1']
