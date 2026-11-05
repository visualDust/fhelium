r"""Full-slot CKKS bootstrap composition configured for one Engine.

The composition accepts and returns two-component,
coefficient-domain, standard-residue Q RNS values. Temporary
NTT/Montgomery values are confined to ordinary engine arithmetic. Ciphertext
payload axes are `[component, *batch, limb, coefficient]`; linear-map diagonal
payloads use `[slot]` with $S=N/2$.
"""

from __future__ import annotations

import math
from dataclasses import replace
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch

from fhelium.values import (
    Ciphertext,
    EvaluationKeyRequirements,
    EvaluationKeySet,
    RotationKey,
    SecretKey,
)
from fhelium.eager import Engine
from fhelium.experimental.bootstrap.structural import (
    CenteredModRaise,
    ModulusRaising,
    _prepare_entry,
    _rescale_to_structural_basis,
)
from fhelium.runtime.signature import ValueSignature
from fhelium.experimental.bootstrap._core import (
    _CoreAlgorithm,
    _CoreMaterials,
    PreparedBootstrapCore,
    PreparedBootstrap,
)

from fhelium.utils.rotation import (
    decompose_signed_power_of_two_rotation,
)


class FullSlotBootstrap:
    r"""Full-slot refresh callable with prepared transforms and replaceable components.

    Construction configures transform compilers/evaluators and modular reduction
    for one engine. Calling the object executes the visible full-slot algorithm with
    primitive key references selected from an application-owned inventory.

    ``compress_diagonals=True`` selects the minimum lossless contiguous
    storage of each encoded NTT diagonal. ``retain_diagonals`` keeps these materials between
    calls. ``retain_constants`` similarly keeps the scalar, entry, and
    monomial constants used by the circuit between calls. If
    ``batch_modular_branches`` is true, the real and imaginary periodic
    reductions share one dense branch-batch execution before being unpacked.

    ``source_depth`` selects the Q suffix consumed by modulus raising.
    ``input_depth`` may equal it or its preceding group; the latter entry
    multiplies by an integer and rescales into the source basis. Defaults use
    the terminal basis and its preceding group. ``modulus_raising`` selects
    centered extension or an intermediate-secret construction.

    Let $\Delta_0$ be `input_scale`, $q_b$ the structural
    Q-basis product, $S=N/2$, $C$ the unscaled CoeffsToSlots map, and $T$ the
    unscaled SlotsToCoeffs map with $T(C(a))=Sa$. Let $B$ be the modular
    reduction's `input_bound` and let $D$ be `fused_input_divisor`, equal to
    $B$ when input normalization is fused and $1$ otherwise. The constructor
    compiles numerical transform factors

    $$
    \alpha_C=\frac{\Delta_0}{q_bD},\qquad
    \alpha_T=\frac{q_b}{2\Delta_0}.
    $$

    If $a$ is the coefficient-coordinate value represented by the centered
    ModRaised ciphertext at metadata scale $\Delta_0$, define

    $$
    w=\frac{\Delta_0}{Sq_b}C(a),\qquad
    r_{\rm R}=2\operatorname{Re}(w),\qquad
    r_{\rm I}=2\operatorname{Im}(w).
    $$

    The encrypted transform followed by multiplication by $1/S$ produces $w/D$. Conjugation and monomial correction expose either raw $r_{\rm R},r_{\rm I}$ when $D=1$, or normalized $r_{\rm R}/B,r_{\rm I}/B$ when $D=B$. With

    $$
    \rho(r)=\frac{\sin(\pi r)}{\pi},
    $$

    the idealized nonlinear and inverse-transform portion is

    $$
    a_{\rm out}=\alpha_T T\left(
      \rho(r_{\rm R})+i\rho(r_{\rm I})
    \right).
    $$

    ``message_domain="complex"`` uses this two-branch map. For real-valued
    slot messages, ``message_domain="real"`` requires the generator-5 radix-2
    STC compiler and evaluates only $r_{\rm R}$. With
    $H=\operatorname{diag}(1/2,1,\ldots,1)$, it reconstructs
    $a_{\rm out}=z+\overline z$ from $z=\alpha_T T(H\rho(r_{\rm R}))$.
    The first STC stage absorbs $H$ into a new diagonal plan. The final
    conjugation/addition preserves depth and actual scale; branch batching
    applies only to the complex composition.

    Polynomial approximation, CKKS arithmetic, and key switching perturb this
    idealized map. The caller must establish the
    reducer's raw-coordinate precondition
    $|r_{\rm R}|,|r_{\rm I}|\le B$; encrypted execution cannot inspect it.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        coeffs_to_slots_compiler: Any,
        coeffs_to_slots_evaluator: Any,
        modular_reduction: Any,
        slots_to_coeffs_compiler: Any,
        slots_to_coeffs_evaluator: Any,
        modulus_raise_target_depth: int = 0,
        modulus_raising: ModulusRaising | None = None,
        source_depth: int | None = None,
        input_depth: int | None = None,
        input_scale: float | None = None,
        target_scales: Sequence[float] | None = None,
        modup_method: Literal["exact", "crt", "normalized_crt"] = "exact",
        normalization_stage: int | None = None,
        retain_diagonals: bool = False,
        compress_diagonals: bool = True,
        message_domain: Literal["complex", "real"] = "complex",
        retain_constants: bool = False,
        batch_modular_branches: bool = False,
    ) -> None:
        if modulus_raise_target_depth < 0:
            raise ValueError('modulus_raise_target_depth cannot be negative')
        if modulus_raise_target_depth > engine.max_depth:
            raise ValueError(
                'modulus_raise_target_depth is outside the Q chain'
            )
        source_depth = (
            engine.max_depth if source_depth is None else source_depth
        )
        engine.config.q_row_start(source_depth)
        input_depth = (
            max(0, source_depth - 1) if input_depth is None else input_depth
        )
        if (
            input_depth not in {source_depth, source_depth - 1}
            or input_depth < 0
        ):
            raise ValueError(
                "Bootstrap input depth must be the source depth or its preceding group"
            )
        if modulus_raise_target_depth > source_depth:
            raise ValueError(
                "Modulus raise target must precede or equal source depth"
            )
        scale = float(
            engine.config.default_scale if input_scale is None else input_scale
        )
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Bootstrap input scale must be positive and finite"
            )
        structural_base = math.prod(engine.config.active_q_moduli(source_depth))
        self.source_depth = source_depth
        self.input_depth = input_depth
        self.input_scale = scale
        self.modulus_raising = (
            CenteredModRaise() if modulus_raising is None else modulus_raising
        )
        if modup_method not in {"exact", "crt", "normalized_crt"}:
            raise ValueError("Unknown key-switch ModUp method")
        self.modup_method: Literal["exact", "crt", "normalized_crt"] = (
            modup_method
        )
        self.target_scales = (
            None
            if target_scales is None
            else tuple(float(x) for x in target_scales)
        )
        if self.target_scales is not None and (
            len(self.target_scales) != engine.max_depth + 1
            or any(not math.isfinite(x) or x <= 0 for x in self.target_scales)
        ):
            raise ValueError(
                "target_scales must give one positive finite scale per public depth"
            )

        if message_domain not in {"complex", "real"}:
            raise ValueError("message_domain must be 'complex' or 'real'")
        self.message_domain = message_domain
        self.engine = engine
        self.modular_reduction = modular_reduction
        requires_relinearization = getattr(
            modular_reduction, 'requires_relinearization', None
        )
        if type(requires_relinearization) is not bool:
            raise TypeError(
                'modular_reduction.requires_relinearization must be a bool'
            )
        self.requires_relinearization = requires_relinearization
        self.modulus_raise_target_depth = modulus_raise_target_depth
        self.retain_diagonals = retain_diagonals
        self.compress_diagonals = compress_diagonals
        self.retain_constants = retain_constants
        self.batch_modular_branches = batch_modular_branches
        self.coeffs_to_slots = coeffs_to_slots_compiler.compile(
            slots=engine.num_slots,
            direction='coeffs_to_slots',
            generator=engine.galois_generator,
            scale=(
                scale / structural_base / modular_reduction.fused_input_divisor
            ),
        )
        self.normalization_factor = 1.0 / engine.num_slots
        self.normalization_stage = normalization_stage
        if normalization_stage is not None:
            if (
                not -len(self.coeffs_to_slots)
                <= normalization_stage
                < len(self.coeffs_to_slots)
            ):
                raise ValueError(
                    "normalization_stage is outside the CoeffsToSlots plan"
                )
            stages = list(self.coeffs_to_slots)
            stage = stages[normalization_stage]
            stages[normalization_stage] = replace(
                stage,
                diagonals={
                    k: np.asarray(v) / engine.num_slots
                    for k, v in stage.diagonals.items()
                },
            )
            self.coeffs_to_slots = tuple(stages)
            self.normalization_factor = 1.0
        self.coeffs_to_slots_evaluator = coeffs_to_slots_evaluator
        self.slots_to_coeffs = slots_to_coeffs_compiler.compile(
            slots=engine.num_slots,
            direction='slots_to_coeffs',
            generator=engine.galois_generator,
            scale=structural_base / (2.0 * scale),
        )
        self.slots_to_coeffs_evaluator = slots_to_coeffs_evaluator
        if message_domain == "real":
            from fhelium.experimental.bootstrap.linear.radix2 import (
                Radix2FourierTransformCompiler,
            )

            if engine.galois_generator != 5 or not isinstance(
                slots_to_coeffs_compiler, Radix2FourierTransformCompiler
            ):
                raise ValueError(
                    "Real-slot reconstruction requires the generator-5 radix-2 SlotsToCoeffs compiler"
                )
            # For real decoded slots, T(a) + conjugate(T(a)) counts the
            # constant polynomial coefficient twice. T(Ha), H[0]=1/2,
            # preserves that coefficient while reconstructing its conjugate
            # polynomial half. Fold H into L(x)=sum_k d_k Rot_k(x).
            first = self.slots_to_coeffs[0]
            mask = np.ones(self.engine.num_slots)
            mask[0] = 0.5
            corrected = replace(
                first,
                diagonals={
                    k: np.asarray(diagonal) * np.roll(mask, k)
                    for k, diagonal in first.diagonals.items()
                },
            )
            self.slots_to_coeffs = (corrected, *self.slots_to_coeffs[1:])
        for stages in (self.coeffs_to_slots, self.slots_to_coeffs):
            if not stages or any(
                stage.slots != engine.num_slots for stage in stages
            ):
                raise ValueError('compiled transform has the wrong slot count')
        forward_depths = sum(
            coeffs_to_slots_evaluator.required_depths(stage)
            for stage in self.coeffs_to_slots
        )
        inverse_depths = sum(
            slots_to_coeffs_evaluator.required_depths(stage)
            for stage in self.slots_to_coeffs
        )
        self.output_depth = (
            modulus_raise_target_depth
            + forward_depths
            + 1
            + modular_reduction.required_depths
            + inverse_depths
        )
        if self.output_depth > engine.max_depth:
            raise ValueError(
                'bootstrap depth exceeds the modulus chain: '
                f'output={self.output_depth}, '
                f'max_depth={engine.max_depth}'
            )

        self._materials = _CoreMaterials()

    @property
    def remaining_depth(self) -> int:
        """Return Q depth transitions available after this Bootstrap."""
        return self.engine.max_depth - self.output_depth

    @property
    def refresh_window(self) -> int:
        """Return transitions from the refreshed output to the next entry depth."""
        return self.input_depth - self.output_depth

    @property
    def required_rotations(self) -> tuple[int, ...]:
        r"""Return normalized signed $\operatorname{Rot}_r$ steps for both maps."""

        return tuple(
            sorted(
                {
                    RotationKey.normalize_step(
                        step,
                        ring_dimension=self.engine.config.N,
                    )
                    for step in (
                        *(
                            step
                            for stage in self.coeffs_to_slots
                            for step in self.coeffs_to_slots_evaluator.required_rotation_offsets(
                                stage
                            )
                        ),
                        *(
                            step
                            for stage in self.slots_to_coeffs
                            for step in self.slots_to_coeffs_evaluator.required_rotation_offsets(
                                stage
                            )
                        ),
                    )
                    if step
                }
            )
        )

    def key_steps(
        self, rotation_key_strategy: str = 'all_required'
    ) -> tuple[int, ...]:
        """Return the standalone rotation basis requested by a generation strategy.

        ``signed_power_of_two`` stores the signed-power components needed to
        compose the transform rotations. ``signed_power_of_two_with_baby_steps``
        adds direct keys for every BSGS baby step. ``all_required`` stores a
        direct key for every required transform rotation.

        An existing inventory can satisfy signed-power requests with more
        direct paths. ``all_required`` still requires direct transform keys;
        ``signed_power_of_two_with_baby_steps`` requires direct baby keys.
        Baby decomposition sharing additionally requires compatible stored
        layouts and the evaluator's ``hoist_baby_rotations`` choice.

        Args:
            rotation_key_strategy: Rotation-key inventory to enumerate.

        Raises:
            ValueError: If the rotation-key strategy is not supported.
        """

        if rotation_key_strategy == 'all_required':
            return self.required_rotations
        if rotation_key_strategy not in {
            'signed_power_of_two',
            'signed_power_of_two_with_baby_steps',
        }:
            raise ValueError(
                "rotation_key_strategy must be 'signed_power_of_two', "
                "'signed_power_of_two_with_baby_steps', or 'all_required'"
            )
        steps = {
            digit
            for step in self.required_rotations
            for digit in decompose_signed_power_of_two_rotation(
                step,
                self.engine.num_slots,
            )
        }
        if rotation_key_strategy == 'signed_power_of_two_with_baby_steps':
            for stages, evaluator in (
                (self.coeffs_to_slots, self.coeffs_to_slots_evaluator),
                (self.slots_to_coeffs, self.slots_to_coeffs_evaluator),
            ):
                baby_offsets = getattr(evaluator, 'baby_rotation_offsets', None)
                if baby_offsets is not None:
                    for stage in stages:
                        steps.update(
                            RotationKey.normalize_step(
                                step, ring_dimension=self.engine.config.N
                            )
                            for step in baby_offsets(stage)
                        )
        steps.discard(0)
        return tuple(sorted(steps))

    def evaluation_key_requirements(
        self, rotation_key_strategy: str = 'all_required'
    ) -> EvaluationKeyRequirements:
        """Return the capability description of the standalone rotation basis.

        Rotation steps come from :meth:`key_steps`. Conjugation is always
        required by full-slot reconstruction; relinearization is required only
        when the selected modular reduction declares ciphertext products. The
        result contains no key tensors. Reuse from a shared inventory can
        replace composed paths with available direct keys.
        """

        return EvaluationKeyRequirements(
            rotation_steps=frozenset(self.key_steps(rotation_key_strategy)),
            requires_relinearization=self.requires_relinearization,
            requires_conjugation=True,
        )

    def create_evaluation_keys(
        self,
        secret_key: SecretKey,
        *,
        into: EvaluationKeySet | None = None,
        rotation_key_strategy: str = "all_required",
        device: torch.device | str | None = None,
    ) -> EvaluationKeySet:
        """Populate a shared inventory with evaluation keys for this composition.

        The default ``all_required`` strategy provisions direct transform keys.
        Existing paths satisfying the selected direct-key requirements are
        reused by reference. Missing rotations are generated from a
        consumer-wide layout choice accounting for shared
        steps and additional key storage. Relinearization and conjugation
        capabilities are provisioned through the engine's primitive factories.
        The returned inventory is ``into`` when supplied; other members remain
        available to their consumers. The caller supplies keys for one secret
        relation and mathematical parameter context.
        """
        from ._keys import _create_evaluation_keys

        return _create_evaluation_keys(
            self,
            secret_key,
            into=into,
            rotation_key_strategy=rotation_key_strategy,
            device=device,
        )

    @property
    def cached_diagonal_bytes(self) -> int:
        """Return encoded diagonal bytes retained by ordinary evaluations."""
        return self._materials.diagonal_bytes

    @property
    def cached_constant_bytes(self) -> int:
        """Return scalar and monomial bytes retained by ordinary evaluations."""
        return self._materials.constant_bytes

    def clear_cache(self) -> None:
        """Release this callable's ordinary caches.

        PreparedBootstrapCore objects own separate dictionaries and retain their
        materials independently. Close Graphs using ordinary evaluations first.
        """
        self._materials.clear()

    def _algorithm(self) -> _CoreAlgorithm:
        return _CoreAlgorithm(
            self.engine,
            self.coeffs_to_slots,
            self.coeffs_to_slots_evaluator,
            self.modular_reduction,
            self.slots_to_coeffs,
            self.slots_to_coeffs_evaluator,
            self.modulus_raise_target_depth,
            self.batch_modular_branches,
            self.compress_diagonals,
            self.message_domain,
            self.modulus_raising,
            self.source_depth,
            self.input_scale,
            self.normalization_factor,
            self.target_scales,
            self.modup_method,
        )

    def entry(self, ciphertext: Ciphertext) -> tuple[Ciphertext, float]:
        r"""Return source-basis coordinates and their output scale compensation.

        For entry group product M, integer k=max(1, ceil(M*Delta0/Delta_in))
        gives Delta_b=k*Delta_in/M after nearest rescale. Direct source-basis
        entry instead uses Delta_b=Delta_in without this transition. The returned payload
        is interpreted at Delta0; tau=Delta_b/Delta0 restores the original
        message coordinate in ``finish``. The input is preserved.
        """
        if ciphertext.depth != self.input_depth:
            raise ValueError(
                "Bootstrap input depth differs from its configured entry"
            )
        if self.input_depth == self.source_depth:
            self.engine.validate_ciphertext(ciphertext)
            structural_value = ciphertext
        else:
            prepared = _prepare_entry(
                self.engine, ciphertext, target_scale=self.input_scale
            )
            structural_value = _rescale_to_structural_basis(
                self.engine, prepared, source_depth=self.source_depth
            )
        ratio = structural_value.scale / self.input_scale
        coordinate = replace(structural_value, scale=self.input_scale)
        return coordinate, ratio

    def core(
        self, coordinate: Ciphertext, *, evaluation_keys: EvaluationKeySet
    ) -> Ciphertext:
        """Evaluate the normalized core with the ordinary retention controls.

        ``coordinate`` is the source-basis result of ``entry``. Use ``prepare_core``
        to bind retained materials before repeated execution or CUDA capture.
        """
        try:
            return self._algorithm().evaluate(
                coordinate,
                evaluation_keys,
                self._materials,
                retain_diagonals=self.retain_diagonals,
                retain_constants=self.retain_constants,
            )
        finally:
            self._materials.clear_key_bindings()

    @staticmethod
    def finish(core_result: Ciphertext, entry_scale_ratio: float) -> Ciphertext:
        r"""Return a fresh envelope with scale Delta_out * entry_scale_ratio.

        Payload storage is shared with ``core_result``. Its original metadata
        remains unchanged, including when it is a borrowed Graph output.
        """
        return replace(core_result, scale=core_result.scale * entry_scale_ratio)

    def prepare(
        self,
        example_input: Ciphertext,
        *,
        evaluation_keys: EvaluationKeySet,
    ) -> PreparedBootstrap:
        """Bind a complete refresh for Eager execution or callable Compile.

        Preparation follows the existing core material discovery once. Newly
        encoded materials consume the encoding RNG; later calls and Compile
        capture only read those materials. The prototype payload is not retained.
        """
        coordinate, _ = self.entry(example_input)
        return PreparedBootstrap(
            self.prepare_core(coordinate, evaluation_keys=evaluation_keys),
            ValueSignature.from_value(example_input),
        )

    def prepare_core(
        self,
        coordinate: Ciphertext,
        *,
        evaluation_keys: EvaluationKeySet,
    ) -> PreparedBootstrapCore:
        """Bind and prepare a core for the post-entry input's state and device.

        Preparation executes the core once to discover and retain its diagonal,
        scalar and structural materials. It performs real arithmetic and advances
        the encoding RNG for new materials. The temporary output and prototype
        payload are released; only the signature, selected component objects,
        key values and prepared materials are retained. Call before CUDA capture.
        """
        if coordinate.device.type == "cuda":
            with torch.cuda.device(coordinate.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "prepare_core must run before CUDA Graph capture"
                    )
        algorithm = self._algorithm()
        materials = _CoreMaterials()
        signature = ValueSignature.from_value(coordinate)
        algorithm.evaluate(coordinate, evaluation_keys, materials)
        return PreparedBootstrapCore(
            algorithm, materials, signature, coordinate.device
        )

    def __call__(
        self,
        ciphertext: Ciphertext,
        *,
        evaluation_keys: EvaluationKeySet,
    ) -> Ciphertext:
        r"""Refresh a full-slot ciphertext or dense batch at ``input_depth``.

        The input must be a context-compatible two-component Q ciphertext in
        coefficient domain with standard residues, active `prime_ids`,
        data axes `[component, *batch, limb, coefficient]`, ring extent $N$,
        the configured ``input_depth``, and sufficient input precision for the
        chosen circuit. Entry either uses the source basis directly or rescales
        its preceding complete Q group. All $S$ slots are transformed; there
        is no sparse-slot mode. ``evaluation_keys`` must provide every selected transform
        rotation and a conjugation key. The built-in reductions also require a
        relinearization key for ciphertext products; a custom slotwise
        reduction that performs no such product may omit it from the inventory.

        The state and scale recurrence is:

        1. Let $M_{input}$ be the entry group product. Multiply the input
           residues and scale by integer
           $k=\max(1,\lceil M_{input}\Delta_0/\Delta_{in}\rceil)$.
           Nearest group rescale then reaches the source Q basis with actual
           scale $\Delta_b=k\Delta_{in}/M_{input}$. Direct source-basis entry
           instead uses $\Delta_b=\Delta_{in}$ without multiplication or rescale.
        2. For the fixed transform circuit, interpret this source-basis payload
           in coordinates with scale $\Delta_0$. Its message coordinate is
           $u=(\Delta_b/\Delta_0)m$. ModRaise and the fixed CoeffsToSlots /
           SlotsToCoeffs maps refresh $u$. At the output, multiplying recorded
           scale by $\Delta_b/\Delta_0$ restores the coordinate $m$.
           No input-dependent diagonal materials are needed.
        3. Coefficient-product targets are supplied through ``target_scales``
           or selected backward from $s_D=\Delta_0$ with
           $s_d=\sqrt{M_ds_{d+1}}$.
           Polynomial basis nodes propagate their input's actual scale using
           $t_{d+1}=t_d^2/M_d$; coefficient products and accumulators connect
           those basis values to the selected output targets. Quotient scales
           are retained, rather than reset after each multiplication.
        4. Conjugation, branch addition/subtraction, and monomial multiplication
           preserve actual scale and depth. Each stage consumes its declared
           arithmetic transitions. The final result records the actual output
           scale including the compensated entry-coordinate factor.

        The functional result does not alias the input. It is a two-component
        coefficient-domain standard-RNS Q ciphertext with unchanged batch
        axes, depth `output_depth`, and
        the Engine's Q basis at `output_depth`. The method does not enforce
        an application error bound or the reducer's encrypted input range.
        """

        coordinate, ratio = self.entry(ciphertext)
        result = self.core(coordinate, evaluation_keys=evaluation_keys)
        return self.finish(result, ratio)
