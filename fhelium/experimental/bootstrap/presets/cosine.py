r"""Cosine Bootstrap profiles with replaceable execution components.

Profiles select a Chebyshev seed, double-angle recurrence and Fourier stage
counts. The Engine supplies the modulus chain and default scale. The returned
FullSlotBootstrap exposes its output depth, remaining depth and refresh window.
"""

from __future__ import annotations

from typing import Any, Literal
from collections.abc import Sequence
from fhelium.experimental.bootstrap.structural import ModulusRaising

from fhelium.eager import Engine
from fhelium.experimental.bootstrap.full_slot import FullSlotBootstrap
from fhelium.experimental.bootstrap.linear import (
    DiagonalBSGSEvaluator,
    Radix2FourierTransformCompiler,
)
from fhelium.experimental.bootstrap.linear.radix2 import _balanced_layer_groups
from fhelium.experimental.bootstrap.polynomial import (
    EvenChebyshevEvaluator,
    BinaryDecompositionChebyshevEvaluator,
    BlockedChebyshevEvaluator,
    ChebyshevInterpolator,
)
from fhelium.experimental.bootstrap.reduction import CosineDoubleAngleReduction


def cosine_bootstrap(
    engine: Engine,
    *,
    profile: Literal["7/44", "8/28", "8/28-s2c3", "3/30"] = "7/44",
    message_domain: Literal["complex", "real"] = "complex",
    polynomial_evaluator: Any | None = None,
    giant_step_reduction: Literal["aggregate", "independent"] | None = None,
    coeffs_to_slots_evaluator: Any | None = None,
    slots_to_coeffs_evaluator: Any | None = None,
    product_implementation: str | None = "auto",
    product_chunk_size: int | None = 16,
    retain_diagonals: bool = True,
    compress_diagonals: bool = True,
    retain_constants: bool = True,
    batch_modular_branches: bool = True,
    modulus_raising: ModulusRaising | None = None,
    source_depth: int | None = None,
    input_depth: int | None = None,
    input_scale: float | None = None,
    target_scales: Sequence[float] | None = None,
    modup_method: Literal["exact", "crt", "normalized_crt"] = "exact",
) -> FullSlotBootstrap:
    r"""Construct a full-slot cosine Bootstrap with inspectable choices.

    ``"7/44"`` uses seven double-angle iterations, a degree-44 Chebyshev seed,
    Blocked evaluation with baby step eight, and aggregate giant-step
    quotients. ``"8/28"`` uses eight iterations and a degree-28 seed;
    ``"8/28-s2c3"`` adds a third SlotsToCoeffs stage. Both degree-28 profiles
    default to Binary evaluation and independent giant quotients. These three profiles
    have raw branch bound $B=1024$ and fold division by B into CoeffsToSlots.

    ``"3/30"`` selects raw bound B=32, four CoeffsToSlots stages, an even
    degree-30 seed evaluated through T2 and a degree-15 Blocked schedule,
    three double-angle iterations and three SlotsToCoeffs stages. The seed
    coordinate shifts by -1/64. Transform scaling and slot normalization occur
    in the last forward stage; the following coefficient product aligns scale.
    Its input range must be established for the chosen modulus-raising relation.

    ``modulus_raising`` binds the selected raising component. ``source_depth``,
    ``input_depth`` and ``input_scale`` define entry coordinates, while
    ``target_scales`` supplies coefficient-product targets for each depth.
    The example encapsulation parameters specify these values together.

    Defaults are based on the logN16, generator-five, approximately $2^{50}$
    scale composition. A custom modulus chain or scale changes the numerical
    behavior. Inspect ``remaining_depth`` and ``refresh_window`` and validate
    the decoded error for the intended parameter and message range.

    ``polynomial_evaluator`` replaces the profile's polynomial schedule.
    ``giant_step_reduction`` selects QP accumulation followed by one quotient,
    or independent per-giant rescale and key switching. These schedules differ
    in rounding and workspace. A supplied linear evaluator is used unchanged;
    product and giant-step shortcuts configure only the default evaluators.
    Construct FullSlotBootstrap directly for custom transform compilers or
    modular-reduction components.

    ``product_implementation="auto"`` selects weighted-product implementations
    from represented data and device. ``None`` selects native products, and a
    registered name requests that implementation. ``product_chunk_size=16``
    bounds fused product terms. Retaining diagonals and constants trades resident memory for
    repeated preparation; ``compress_diagonals`` losslessly compresses eligible
    encoded data. Branch batching applies to complex messages. Real messages
    require generator-five radix-2 reconstruction.
    """
    profiles: dict[
        str, tuple[int, int, int, Literal["aggregate", "independent"]]
    ] = {
        "7/44": (7, 44, 2, "aggregate"),
        "3/30": (3, 30, 3, "aggregate"),
        "8/28": (8, 28, 2, "independent"),
        "8/28-s2c3": (8, 28, 3, "independent"),
    }
    try:
        iterations, degree, inverse_stages, default_reduction = profiles[
            profile
        ]
    except KeyError:
        raise ValueError(
            f"Unknown cosine Bootstrap profile {profile!r}"
        ) from None
    if polynomial_evaluator is None:
        if profile == "3/30":
            polynomial_evaluator = EvenChebyshevEvaluator(
                BlockedChebyshevEvaluator(baby_step=8)
            )
        elif profile == "7/44":
            polynomial_evaluator = BlockedChebyshevEvaluator(baby_step=8)
        else:
            polynomial_evaluator = BinaryDecompositionChebyshevEvaluator(
                skip_near_zero=1e-15
            )
    if giant_step_reduction is None:
        giant_step_reduction = default_reduction
    common: dict[str, Any] = {
        "baby_step": 16,
        "hoist_baby_rotations": True,
        "giant_step_reduction": giant_step_reduction,
        "product_implementation": product_implementation,
        "product_chunk_size": product_chunk_size,
    }
    if coeffs_to_slots_evaluator is None:
        forward_steps = {"coeffs_to_slots_stage_0": 2048}
        forward_schedule: Literal["auto", "group_major"] = "auto"
        if profile == "3/30":
            remaining_layers = engine.config.logN - 1
            forward_steps = {}
            for index, width in enumerate(
                _balanced_layer_groups(remaining_layers, 4)
            ):
                remaining_layers -= width
                span_bits = width + int(index != 0)
                forward_steps[f"coeffs_to_slots_stage_{index}"] = 1 << (
                    remaining_layers + (span_bits + 1) // 2
                )
            forward_schedule = "group_major"
        coeffs_to_slots_evaluator = DiagonalBSGSEvaluator(
            **common,
            baby_steps_by_transform=forward_steps,
            rotation_schedule=forward_schedule,
        )
    if slots_to_coeffs_evaluator is None:
        inverse_steps = (
            {"slots_to_coeffs_stage_1": 4096}
            if inverse_stages == 2
            else {
                "slots_to_coeffs_stage_0": 8,
                "slots_to_coeffs_stage_1": 256,
                "slots_to_coeffs_stage_2": 4096,
            }
        )
        slots_to_coeffs_evaluator = DiagonalBSGSEvaluator(
            **common, baby_steps_by_transform=inverse_steps
        )
    return FullSlotBootstrap(
        engine,
        message_domain=message_domain,
        coeffs_to_slots_compiler=Radix2FourierTransformCompiler(
            stage_count=4 if profile == "3/30" else 2,
            scale_stage=-1 if profile == "3/30" else None,
        ),
        coeffs_to_slots_evaluator=coeffs_to_slots_evaluator,
        modular_reduction=CosineDoubleAngleReduction(
            input_bound=32 if profile == "3/30" else 1024,
            seed_shift=-1 / 64 if profile == "3/30" else 0.0,
            double_angle_iterations=iterations,
            approximator=ChebyshevInterpolator(degree=degree),
            evaluator=polynomial_evaluator,
            fuse_input_normalization=True,
            retain_ntt=True,
        ),
        slots_to_coeffs_compiler=Radix2FourierTransformCompiler(
            stage_count=inverse_stages
        ),
        slots_to_coeffs_evaluator=slots_to_coeffs_evaluator,
        modulus_raise_target_depth=0,
        modulus_raising=modulus_raising,
        source_depth=source_depth,
        input_depth=input_depth,
        input_scale=input_scale,
        target_scales=target_scales,
        modup_method=modup_method,
        normalization_stage=-1 if profile == "3/30" else None,
        retain_diagonals=retain_diagonals,
        compress_diagonals=compress_diagonals,
        retain_constants=retain_constants,
        batch_modular_branches=batch_modular_branches,
    )


__all__ = ["cosine_bootstrap"]
