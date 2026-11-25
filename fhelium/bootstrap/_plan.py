r"""Bootstrap plans: component choices, depth placement and key requirements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from fhelium.algorithms.linear import BsgsPlan, LinearTransform, radix2_dft
from fhelium.algorithms.linear._radix2 import _balanced_layer_groups
from fhelium.algorithms.polynomial import (
    BinaryChebyshevPlan,
    BlockedChebyshevPlan,
    ChebyshevInterpolator,
    EvenChebyshevPlan,
    RecursiveChebyshevPlan,
)
from fhelium.bootstrap._modulus_raise import Centered, Encapsulated
from fhelium.bootstrap.periodic import CosineDoubleAngle, CosineExpansion
from fhelium.utils.rotation import decompose_signed_power_of_two_rotation
from fhelium.values import EvaluationKeyRequirements, RotationKey

Profile = Literal["7/44", "8/28", "8/28-s2c3", "3/30", "3/38"]
_PROFILES: dict[str, tuple[int, int, int, str]] = {
    "7/44": (7, 44, 2, "aggregate"),
    "3/30": (3, 30, 3, "aggregate"),
    "3/38": (3, 38, 3, "aggregate"),
    "8/28": (8, 28, 2, "independent"),
    "8/28-s2c3": (8, 28, 3, "independent"),
}


@dataclass(frozen=True)
class BootstrapStage:
    """One step of a plan: its name and depth/scale placement."""

    name: str
    input_depth: int
    output_depth: int
    output_scale: float


@dataclass(frozen=True)
class BootstrapPlan:
    r"""Full-slot bootstrap composition for one CKKS configuration.

    The refresh evaluates, in order: modulus raise from ``source_depth`` to
    ``raise_target_depth``; CoeffsToSlots stages $C$; multiplication by the
    normalization factor $1/S$ ($S=N/2$; one depth, or none when folded into the
    last stage and ``normalization_factor`` is one);
    real/imaginary split by conjugation; EvalMod ``eval_mod`` on each branch;
    merge with the monomial $X^{S}$; SlotsToCoeffs stages $T$. With
    $\Delta_0=$ ``input_scale``, source basis product $q_b$ and EvalMod bound
    $B$, the transforms carry the factors
    $\alpha_C=\Delta_0/(q_bD)$ and $\alpha_T=q_b/(2\Delta_0)$ where $D=B$ when
    input normalization is fused into $C$ and $1$ otherwise. If $a$ is the
    raised coefficient vector at scale $\Delta_0$,
    $w=\Delta_0C(a)/(Sq_b)$, $r_R=2\Re w$, $r_I=2\Im w$, and
    $\rho(r)=\sin(\pi r)/\pi$, the idealized result is
    $\alpha_TT(\rho(r_R)+i\rho(r_I))$. ``message_domain="real"`` evaluates only
    $r_R$ and reconstructs $z+\bar z$ after a modified first STC stage.

    The raised value satisfies $a=m+q_bI$ with integer overflow polynomial $I$;
    EvalMod requires $|I|+|m/q_b|\le B$. ``modulus_raise_bound`` reports the
    deterministic worst case of $|I|+1$ when the raise declares a secret
    weight.

    Plaintext constants are encoded so each multiplication reaches
    ``target_scales[d+1]``; the default schedule is
    $s_L=\Delta_L$, $s_d=\sqrt{\rho_d s_{d+1}}$. The plan holds no Tensor data
    and no Engine; :func:`fhelium.bootstrap.bootstrap` evaluates it.

    ``coeffs_to_slots_plan``, ``slots_to_coeffs_plan`` and ``eval_mod`` may be
    any objects following the linear-plan protocol of
    :mod:`fhelium.algorithms.linear` and the reduction protocol of
    :mod:`fhelium.bootstrap.periodic`. The plan reads their declared
    ``required_depths`` for depth placement and their rotation offsets and
    relinearization depths for key requirements; it calls their
    ``evaluate`` methods with one :class:`~fhelium.algorithms.Arithmetic`
    on ``target_scales``. :func:`fhelium.bootstrap.prepare` runs these calls
    once to record keys and encoded materials, and the prepared object
    repeats the same calls on the recorded entries.
    """

    source_depth: int
    raise_target_depth: int
    input_scale: float
    coeffs_to_slots: tuple[LinearTransform, ...]
    coeffs_to_slots_plan: Any
    eval_mod: Any
    slots_to_coeffs: tuple[LinearTransform, ...]
    slots_to_coeffs_plan: Any
    target_scales: tuple[float, ...]
    normalization_factor: float = 1.0
    modulus_raise: Any = field(default_factory=Centered)
    message_domain: Literal["complex", "real"] = "complex"
    batch_branches: bool = True
    compress_diagonals: bool = True
    basis_conversion: Literal["exact", "crt", "normalized_crt"] = "exact"

    @property
    def cts_depth(self) -> int:
        return sum(
            self.coeffs_to_slots_plan.required_depths(stage)
            for stage in self.coeffs_to_slots
        )

    @property
    def stc_depth(self) -> int:
        return sum(
            self.slots_to_coeffs_plan.required_depths(stage)
            for stage in self.slots_to_coeffs
        )

    @property
    def normalization_depths(self) -> int:
        """Depths of the separate $1/S$ multiplication (zero when folded)."""
        return int(self.normalization_factor != 1.0)

    @property
    def output_depth(self) -> int:
        """Depth of the refreshed ciphertext."""
        return (
            self.raise_target_depth
            + self.cts_depth
            + self.normalization_depths
            + self.eval_mod.required_depths
            + self.stc_depth
        )

    @property
    def consumed_depths(self) -> int:
        """Depth transitions consumed after the modulus raise."""
        return self.output_depth - self.raise_target_depth

    @property
    def refresh_window(self) -> int:
        """Transitions available between the output and the next bootstrap input."""
        return self.source_depth - self.output_depth

    @property
    def modulus_raise_bound(self) -> float | None:
        r"""Worst-case $|I|+1$ for the declared raise secret weight $h$: $(h+1)/2+1$.

        Typical coefficients of $I$ are of order $\sqrt{h}$; the worst case is
        reported next to ``eval_mod.input_bound`` for comparison.
        """
        weight = getattr(self.modulus_raise, "intermediate_weight", None)
        return None if weight is None else (weight + 1) / 2 + 1

    def stages(self) -> tuple[BootstrapStage, ...]:
        """Return each step's depth transition and output scale."""
        result = []
        depth = self.raise_target_depth
        result.append(
            BootstrapStage(
                "modulus_raise", self.source_depth, depth, self.input_scale
            )
        )
        for name, width in (
            ("coeffs_to_slots", self.cts_depth),
            ("normalize", self.normalization_depths),
            ("eval_mod", self.eval_mod.required_depths),
            ("slots_to_coeffs", self.stc_depth),
        ):
            result.append(
                BootstrapStage(
                    name,
                    depth,
                    depth + width,
                    self.target_scales[depth + width],
                )
            )
            depth += width
        return tuple(result)

    @property
    def required_rotations(self) -> tuple[int, ...]:
        r"""Normalized signed $\operatorname{Rot}_r$ steps of both transforms."""
        steps = set()
        for stages, plan in (
            (self.coeffs_to_slots, self.coeffs_to_slots_plan),
            (self.slots_to_coeffs, self.slots_to_coeffs_plan),
        ):
            ring = 2 * stages[0].slots
            for stage in stages:
                steps.update(
                    RotationKey.normalize_step(step, ring_dimension=ring)
                    for step in plan.required_rotation_offsets(stage)
                    if step
                )
        return tuple(sorted(steps))

    def key_steps(self, strategy: str = "all_required") -> tuple[int, ...]:
        """Rotation steps provisioned by a key strategy.

        ``all_required`` stores one key per transform rotation;
        ``signed_power_of_two`` stores signed powers of two that compose them;
        ``signed_power_of_two_with_baby_steps`` adds direct BSGS baby keys.
        """
        if strategy == "all_required":
            return self.required_rotations
        if strategy not in {
            "signed_power_of_two",
            "signed_power_of_two_with_baby_steps",
        }:
            raise ValueError(f"Unknown rotation key strategy {strategy!r}")
        slots = self.coeffs_to_slots[0].slots
        steps = {
            digit
            for step in self.required_rotations
            for digit in decompose_signed_power_of_two_rotation(step, slots)
        }
        if strategy == "signed_power_of_two_with_baby_steps":
            for stages, plan in (
                (self.coeffs_to_slots, self.coeffs_to_slots_plan),
                (self.slots_to_coeffs, self.slots_to_coeffs_plan),
            ):
                babies = getattr(plan, "baby_rotation_offsets", None)
                if babies is not None:
                    for stage in stages:
                        steps.update(
                            RotationKey.normalize_step(
                                step, ring_dimension=2 * slots
                            )
                            for step in babies(stage)
                        )
        steps.discard(0)
        return tuple(sorted(steps))

    def key_requirements(
        self, strategy: str = "all_required"
    ) -> EvaluationKeyRequirements:
        """Rotation, relinearization and conjugation capabilities of the plan."""
        return EvaluationKeyRequirements(
            rotation_steps=frozenset(self.key_steps(strategy)),
            requires_relinearization=self.eval_mod.requires_relinearization,
            requires_conjugation=True,
        )


def _default_target_scales(engine) -> tuple[float, ...]:
    config = engine.config
    scales = [config.scale_at(engine.max_depth)] * (engine.max_depth + 1)
    for depth in reversed(range(engine.max_depth)):
        scales[depth] = math.sqrt(
            config.rescale_divisor(depth) * scales[depth + 1]
        )
    return tuple(scales)


_ENCODING_BITS = 61


def _held_target_scales(
    engine,
    input_scale: float,
    cts_bounds: tuple[float, ...],
    eval_mod_end: int,
    output: int,
):
    r"""Scales that keep EvalMod near its product fixed point $s=\rho$.

    EvalMod squares follow $s_d=\sqrt{\rho_ds_{d+1}}$ from
    $s_{e}=\rho_{e-1}$ at its output depth $e$. CoeffsToSlots stage $i$,
    one depth each from depth 0 with diagonal magnitude bound $M_i$, reaches
    $s_{i+1}=\min(s_c, 2^{61}s_i/(\rho_iM_i))$ where $s_c$ is the EvalMod
    input scale, so its plaintext scale $s_{i+1}\rho_i/s_i$ keeps the encoded
    integer coefficients below $2^{61}$. SlotsToCoeffs stages hold $s_e$
    until the output depth, whose scale and every later one follow the
    default schedule.
    """

    config = engine.config
    cts_end = len(cts_bounds)
    scales = list(_default_target_scales(engine))
    scales[eval_mod_end] = float(config.rescale_divisor(eval_mod_end - 1))
    for depth in range(eval_mod_end - 1, cts_end - 1, -1):
        scales[depth] = math.sqrt(
            float(config.rescale_divisor(depth)) * scales[depth + 1]
        )
    scales[0] = input_scale
    for depth, bound in enumerate(cts_bounds):
        limit = math.ldexp(scales[depth], _ENCODING_BITS) / (
            float(config.rescale_divisor(depth)) * bound
        )
        scales[depth + 1] = min(scales[cts_end], limit)
    for depth in range(eval_mod_end + 1, output):
        scales[depth] = scales[eval_mod_end]
    return tuple(scales)


def plan(
    engine,
    preset: Profile = "3/38",
    *,
    intermediate_weight: int | None = None,
    modulus_raise: Any = None,
    message_domain: Literal["complex", "real"] = "complex",
    eval_mod: Any = None,
    polynomial_plan: Any = None,
    coeffs_to_slots_plan: Any = None,
    slots_to_coeffs_plan: Any = None,
    giant_step_reduction: Literal["aggregate", "independent"] | None = None,
    product_implementation: str | None = "auto",
    product_chunk_size: int | None = 16,
    batch_branches: bool = True,
    compress_diagonals: bool = True,
    input_scale: float | None = None,
    target_scales: tuple[float, ...] | None = None,
    basis_conversion: Literal["exact", "crt", "normalized_crt"] | None = None,
) -> BootstrapPlan:
    r"""Construct the default plan of a cosine EvalMod profile for ``engine``'s chain.

    The input is at ``max_depth``, the raise targets depth 0 and the output
    depth equals the plan's consumed depth. The default ``input_scale`` is
    $\Delta_0=\Delta_L2^{e}$ with $e=\max(0,\operatorname{round}(\log_2(q_b/\Delta_L)))$
    for the maximum-depth default scale $\Delta_L$, so the entry lift places
    the message coordinate $m\Delta_0/q_b$ within a factor $\sqrt2$ of $m$,
    as on a chain whose terminal prime equals $\Delta_L$. Profiles name
    ``iterations/seed degree``: ``"7/44"`` ($B=1024$, two CTS and two
    STC stages, blocked seed evaluation), ``"8/28"`` and ``"8/28-s2c3"``
    (binary seed evaluation, independent giant-step quotients, two or three STC
    stages) and ``"3/30"`` ($B=32$, four CTS and three STC stages, even seed).
    ``"3/38"`` (default, $B=33$), which the ``-boot`` presets are built for, has four CTS stages with the normalization folded into
    the last, a degree-38 seed expanded analytically in Chebyshev form
    (:class:`~fhelium.bootstrap.periodic.CosineExpansion`) and evaluated by
    :class:`~fhelium.algorithms.polynomial.RecursiveChebyshevPlan`, three
    STC stages, ``basis_conversion="normalized_crt"``, ``input_scale`` equal to
    $\Delta_L$, and target scales that hold EvalMod near its product fixed
    point; it consumes 16 depths. Other profiles use
    ``basis_conversion="exact"``. ``intermediate_weight=h`` selects
    :class:`Encapsulated` with intermediate secret weight $h$; otherwise the
    raise is :class:`Centered`. Keyword arguments replace individual choices;
    ``dataclasses.replace`` on the result changes any field afterwards.
    """

    try:
        iterations, degree, inverse_stages, default_reduction = _PROFILES[
            preset
        ]
    except KeyError:
        raise ValueError(
            f"Unknown cosine bootstrap preset {preset!r}"
        ) from None
    config = engine.config
    slots, generator = engine.num_slots, engine.galois_generator
    if generator != 5:
        # Generator-3 slots conjugate odd orbit positions of the generator-5
        # order, which no slot-linear CoeffsToSlots map reproduces.
        raise ValueError("Full-slot bootstrapping requires galois_generator=5")
    folded = preset in {"3/30", "3/38"}
    if basis_conversion is None:
        basis_conversion = "normalized_crt" if preset == "3/38" else "exact"
    if intermediate_weight is not None:
        if modulus_raise is not None:
            raise ValueError(
                "intermediate_weight and modulus_raise are alternatives"
            )
        modulus_raise = Encapsulated(intermediate_weight=intermediate_weight)
    source_depth = engine.max_depth
    structural = math.prod(config.active_q_moduli(source_depth))
    if input_scale is None:
        terminal: float = config.scale_at(source_depth)
        input_scale = (
            terminal
            if preset == "3/38"
            else terminal
            * 2.0 ** max(0, round(math.log2(structural / terminal)))
        )
    scale = float(input_scale)
    if eval_mod is None:
        if polynomial_plan is None:
            polynomial_plan = (
                RecursiveChebyshevPlan()
                if preset == "3/38"
                else EvenChebyshevPlan(BlockedChebyshevPlan(baby_step=8))
                if preset == "3/30"
                else BlockedChebyshevPlan(baby_step=8)
                if preset == "7/44"
                else BinaryChebyshevPlan(skip_near_zero=1e-15)
            )
        bound = {"3/30": 32, "3/38": 33}.get(preset, 1024)
        eval_mod = CosineDoubleAngle(
            input_bound=bound,
            seed_shift=-1 / (2 * bound) if folded else 0.0,
            double_angle_iterations=iterations,
            approximator=CosineExpansion(degree)
            if preset == "3/38"
            else ChebyshevInterpolator(degree=degree),
            polynomial_plan=polynomial_plan,
            fuse_input_normalization=True,
            retain_ntt=True,
        )
    common: dict[str, Any] = {
        "baby_step": 16,
        "hoist_baby_rotations": True,
        "giant_step_reduction": giant_step_reduction or default_reduction,
        "product_implementation": product_implementation,
        "product_chunk_size": product_chunk_size,
    }
    if coeffs_to_slots_plan is None:
        forward_steps = {"coeffs_to_slots_stage_0": 2048}
        schedule = "auto"
        if folded:
            remaining = config.logN - 1
            forward_steps = {}
            for index, width in enumerate(_balanced_layer_groups(remaining, 4)):
                remaining -= width
                span = width + int(index != 0)
                forward_steps[f"coeffs_to_slots_stage_{index}"] = 1 << (
                    remaining + (span + 1) // 2
                )
            schedule = "group_major"
        coeffs_to_slots_plan = BsgsPlan(
            **common,
            baby_steps_by_transform=forward_steps,
            rotation_schedule=schedule,
        )
    if slots_to_coeffs_plan is None:
        inverse_steps = (
            {"slots_to_coeffs_stage_1": 4096}
            if inverse_stages == 2
            else {
                "slots_to_coeffs_stage_0": 8,
                "slots_to_coeffs_stage_1": 256,
                "slots_to_coeffs_stage_2": 4096,
            }
        )
        slots_to_coeffs_plan = BsgsPlan(
            **common, baby_steps_by_transform=inverse_steps
        )
    cts = radix2_dft(
        slots,
        direction="coeffs_to_slots",
        stages=4 if folded else 2,
        generator=generator,
        scale=scale / structural / eval_mod.fused_input_divisor,
        scale_stage=-1 if folded else None,
    )
    normalization = 1.0 / slots
    if folded:
        last = cts[-1]
        cts = (
            *cts[:-1],
            replace(
                last,
                diagonals={
                    k: np.asarray(v) / slots for k, v in last.diagonals.items()
                },
            ),
        )
        normalization = 1.0
    stc = radix2_dft(
        slots,
        direction="slots_to_coeffs",
        stages=inverse_stages,
        generator=generator,
        scale=structural / (2.0 * scale),
    )
    if message_domain == "real":
        # T(Ha) with H[0]=1/2 keeps the constant coefficient once in z+conj(z).
        mask = np.ones(slots)
        mask[0] = 0.5
        first = stc[0]
        stc = (
            replace(
                first,
                diagonals={
                    k: np.asarray(d) * np.roll(mask, k)
                    for k, d in first.diagonals.items()
                },
            ),
            *stc[1:],
        )
    result = BootstrapPlan(
        source_depth=source_depth,
        raise_target_depth=0,
        input_scale=scale,
        coeffs_to_slots=cts,
        coeffs_to_slots_plan=coeffs_to_slots_plan,
        eval_mod=eval_mod,
        slots_to_coeffs=stc,
        slots_to_coeffs_plan=slots_to_coeffs_plan,
        target_scales=(
            _default_target_scales(engine)
            if target_scales is None
            else tuple(float(x) for x in target_scales)
        ),
        normalization_factor=normalization,
        modulus_raise=Centered() if modulus_raise is None else modulus_raise,
        message_domain=message_domain,
        batch_branches=batch_branches,
        compress_diagonals=compress_diagonals,
        basis_conversion=basis_conversion,
    )
    if result.output_depth > source_depth:
        raise ValueError(
            f"Bootstrap consumes {result.consumed_depths} depths but the chain "
            f"provides {source_depth}"
        )
    if preset == "3/38" and target_scales is None:
        bounds = tuple(
            max(float(np.abs(d).max()) for d in stage.diagonals.values())
            for stage in cts
        )
        result = replace(
            result,
            target_scales=_held_target_scales(
                engine,
                scale,
                bounds,
                len(cts) + eval_mod.required_depths,
                result.output_depth,
            ),
        )
    return result


@dataclass(frozen=True)
class _Bound:
    """Engine-bound view used by key derivation and core execution."""

    engine: Any
    plan: BootstrapPlan

    def __getattr__(self, name: str) -> Any:
        plan = object.__getattribute__(self, "plan")
        alias = {
            "modulus_raise_target_depth": "raise_target_depth",
            "coeffs_to_slots_evaluator": "coeffs_to_slots_plan",
            "slots_to_coeffs_evaluator": "slots_to_coeffs_plan",
            "modular_reduction": "eval_mod",
        }.get(name, name)
        if name == "requires_relinearization":
            return plan.eval_mod.requires_relinearization
        return getattr(plan, alias)


__all__ = ["BootstrapPlan", "BootstrapStage", "plan"]
