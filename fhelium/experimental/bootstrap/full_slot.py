r"""Full-slot CKKS bootstrap composition configured for one Engine.

The composition accepts and returns two-component,
coefficient-domain, standard-residue Q RNS values. Temporary
NTT/Montgomery values are confined to ordinary engine arithmetic. Ciphertext
payload axes are `[component, *batch, limb, coefficient]`; linear-map diagonal
payloads use `[slot]` with $S=N/2$.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from fhelium.values import (
    Ciphertext,
    EvaluationKeyRequirements,
    EvaluationKeySet,
    KeySwitchKey,
    Plaintext,
    RotationKey,
    RotationKeySet,
    SecretKey,
)
from fhelium.eager import Engine
from fhelium.experimental.bootstrap.structural import (
    _modulus_raise,
    _prepare_entry,
    _rescale_to_structural_basis,
)
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.linear.execution import (
    _apply_linear_transform,
    _apply_modraised_linear,
)
from fhelium.utils.rotation import (
    decompose_signed_power_of_two_rotation,
)


class FullSlotBootstrap:
    r"""Full-slot refresh callable with prepared transforms and replaceable components.

    Construction configures transform compilers/evaluators and modular reduction
    for one engine. Calling the object executes the visible full-slot algorithm with
    one validated evaluator-only key inventory.

    ``retain_diagonals`` keeps operation-ready transform plaintexts between
    calls. ``retain_constants`` similarly keeps the scalar, entry, and
    monomial constants used by the circuit between calls. If
    ``batch_modular_branches`` is true, the real and imaginary periodic
    reductions share one dense branch-batch execution before being unpacked.

    ``input_depth`` selects the penultimate Q group. The entry rescale reaches
    the ordinary terminal basis, which this composition uses for centered
    ModRaise. It reserves no private CKKS depth.

    Let $\Delta_0$ be `engine.config.default_scale`, $q_b$ the structural
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

    The encrypted transform and explicit $1/S$ multiplication produce $w/D$.
    Conjugation and monomial correction expose either raw $r_{\rm R},r_{\rm I}$
    when $D=1$, or normalized $r_{\rm R}/B,r_{\rm I}/B$ when $D=B$. With

    $$
    \rho(r)=\frac{\sin(\pi r)}{\pi},
    $$

    the idealized nonlinear and inverse-transform portion is

    $$
    a_{\rm out}=\alpha_T T\left(
      \rho(r_{\rm R})+i\rho(r_{\rm I})
    \right).
    $$

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
        retain_diagonals: bool = False,
        retain_constants: bool = False,
        batch_modular_branches: bool = False,
    ) -> None:
        if modulus_raise_target_depth < 0:
            raise ValueError('modulus_raise_target_depth cannot be negative')
        if modulus_raise_target_depth > engine.max_depth:
            raise ValueError(
                'modulus_raise_target_depth is outside the Q chain'
            )
        scale = float(engine.config.default_scale)
        structural_base = math.prod(engine.config.q_depth_groups[-1])
        ratio = structural_base / scale
        if not 0.5 <= ratio <= 2.0:
            raise ValueError(
                'Bootstrap structural basis must be close to the CKKS scale; '
                f'got q_base/Delta={ratio:.6g}'
            )

        self.engine = engine
        self.input_depth = engine.max_depth - 1
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
        self.coeffs_to_slots_evaluator = coeffs_to_slots_evaluator
        self.slots_to_coeffs = slots_to_coeffs_compiler.compile(
            slots=engine.num_slots,
            direction='slots_to_coeffs',
            generator=engine.galois_generator,
            scale=structural_base / (2.0 * scale),
        )
        self.slots_to_coeffs_evaluator = slots_to_coeffs_evaluator
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

        self._diagonal_cache: dict[tuple[torch.device, int, int, int, int, float], Plaintext] = {}
        self._rotation_cache: dict[int, tuple[int, ...]] = {}
        self._rotation_inventory_steps: tuple[int, ...] | None = None
        self._modraise_cache: dict[object, object] = {}
        self._constant_cache: dict[object, object] = {}

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

    def key_steps(self, strategy: str = 'direct') -> tuple[int, ...]:
        """Return the rotation-key inventory for one composition strategy.

        ``direct`` returns every logical transform step as a direct key.
        ``power_of_two`` returns the deduplicated signed-power steps whose
        compositions cover those transforms. The latter therefore describes
        actual inventory entries, not the original transform offsets.

        Raises:
            ValueError: If ``strategy`` is not ``direct`` or ``power_of_two``.
        """

        if strategy == 'direct':
            return self.required_rotations
        if strategy != 'power_of_two':
            raise ValueError("strategy must be 'direct' or 'power_of_two'")
        return tuple(
            sorted(
                {
                    digit
                    for step in self.required_rotations
                    for digit in decompose_signed_power_of_two_rotation(
                        step,
                        self.engine.num_slots,
                    )
                }
            )
        )

    def evaluation_key_requirements(
        self, rotation_strategy: str = 'power_of_two'
    ) -> EvaluationKeyRequirements:
        """Return all evaluator capabilities for one rotation strategy.

        Rotation steps come from :meth:`key_steps`. Conjugation is always
        required by full-slot reconstruction; relinearization is required only
        when the selected modular reduction declares ciphertext products. The
        result contains no key tensors or key-generation policy.
        """

        return EvaluationKeyRequirements(
            rotation_steps=frozenset(self.key_steps(rotation_strategy)),
            requires_relinearization=self.requires_relinearization,
            requires_conjugation=True,
        )

    def _requirements_for_inventory(
        self, evaluation_keys: EvaluationKeySet
    ) -> EvaluationKeyRequirements:
        """Resolve direct-versus-composed rotations from one actual inventory."""

        steps: set[int] = set()
        for required in self.required_rotations:
            if required in evaluation_keys.rotations:
                steps.add(required)
            else:
                steps.update(
                    decompose_signed_power_of_two_rotation(
                        required, self.engine.num_slots
                    )
                )
        return EvaluationKeyRequirements(
            rotation_steps=frozenset(steps),
            requires_relinearization=self.requires_relinearization,
            requires_conjugation=True,
        )

    def create_rotation_keys(
        self,
        secret_key: SecretKey,
        *,
        rotation_strategy: str = 'power_of_two',
    ) -> RotationKeySet:
        """Generate only the selected bootstrap rotation-key inventory.

        ``secret_key`` is consumed by primitive engine key generation and is
        not stored in the returned set. Relinearization and conjugation keys are
        intentionally not created here; applications construct those separate
        capabilities and assemble an :class:`EvaluationKeySet`.
        """

        rotations = RotationKeySet()
        for step in self.key_steps(rotation_strategy):
            rotations.add(self.engine.create_rotation_key(step, secret_key))
        return rotations

    @property
    def cached_diagonal_bytes(self) -> int:
        """Return encoded diagonal tensor bytes retained by this evaluator."""

        total = 0
        for plaintext in self._diagonal_cache.values():
            data = plaintext.data
            if data is not None:
                total += data.numel() * data.element_size()
        return total

    @property
    def cached_constant_bytes(self) -> int:
        """Return tensor bytes retained for scalar and structural constants."""

        total = 0
        for value in self._constant_cache.values():
            data = value.data if isinstance(value, Plaintext) else value
            if isinstance(data, torch.Tensor):
                total += data.numel() * data.element_size()
        return total

    def clear_cache(self) -> None:
        """Release prepared constants, encoded diagonals, and arithmetic tables."""

        self._diagonal_cache.clear()
        self._rotation_cache.clear()
        self._modraise_cache.clear()
        self._constant_cache.clear()

    def _evaluate_linear_transform(
        self,
        arithmetic: BootstrapArithmetic,
        rotation_keys: RotationKeySet,
        stages: tuple[Any, ...],
        evaluator: Any,
        ciphertext: Ciphertext,
    ) -> Ciphertext:
        """Evaluate one compiled transform under the selected cache policy."""

        try:
            return _apply_linear_transform(
                arithmetic,
                stages,
                evaluator,
                ciphertext,
                rotation_keys=rotation_keys,
                diagonal_cache=self._diagonal_cache,
                rotation_cache=self._rotation_cache,
                retain_diagonals=self.retain_diagonals,
            )
        finally:
            if not self.retain_diagonals:
                self._diagonal_cache.clear()

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
        ``input_depth = engine.max_depth - 1``, and sufficient input precision
        for the chosen circuit. Entry rescale reaches the ordinary terminal
        Q group at ``engine.max_depth``. All $S$ slots are transformed; there
        is no sparse-slot mode. ``evaluation_keys`` must provide every selected transform
        rotation and a conjugation key. The built-in reductions also require a
        relinearization key for ciphertext products; a custom slotwise
        reduction that performs no such product may omit it from the inventory.

        The state and scale recurrence is:

        1. Let $M_{input}$ be the entry group product. Multiply the input
           residues and scale by integer
           $k=\max(1,\lceil M_{input}\Delta_0/\Delta_{in}\rceil)$.
           Nearest group rescale then reaches the terminal Q basis with actual
           scale $\Delta_b=k\Delta_{in}/M_{input}$.
        2. For the fixed transform circuit, interpret this terminal payload
           in coordinates with scale $\Delta_0$. Its message coordinate is
           $u=(\Delta_b/\Delta_0)m$. ModRaise and the fixed CoeffsToSlots /
           SlotsToCoeffs maps refresh $u$. At the output, multiplying recorded
           scale by $\Delta_b/\Delta_0$ restores the coordinate $m$.
           No input-dependent diagonal materials are needed.
        3. The arithmetic owner selects coefficient-product targets backward
           from $s_D=\Delta_0$ with $s_d=\sqrt{M_ds_{d+1}}$.
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

        requirements = self._requirements_for_inventory(evaluation_keys)
        evaluation_keys.require(requirements)
        rotation_keys = evaluation_keys.rotations
        conjugation_key = evaluation_keys.conjugation
        assert conjugation_key is not None
        relinearization_key = evaluation_keys.relinearization
        selected_keys: list[KeySwitchKey] = [
            rotation_keys[step] for step in requirements.rotation_steps
        ]
        selected_keys.append(conjugation_key)
        if requirements.requires_relinearization:
            assert relinearization_key is not None
            selected_keys.append(relinearization_key)
        for key in selected_keys:
            self.engine.validate_key_switch_key(key)
        inventory_steps = tuple(sorted(rotation_keys))
        if inventory_steps != self._rotation_inventory_steps:
            self._rotation_cache.clear()
            self._rotation_inventory_steps = inventory_steps

        constant_cache = self._constant_cache if self.retain_constants else {}
        ops = BootstrapArithmetic(self.engine, constant_cache)
        prepared = _prepare_entry(self.engine, ciphertext)
        structural_value = _rescale_to_structural_basis(self.engine, prepared)
        entry_scale_ratio = structural_value.scale / self.engine.config.default_scale
        coordinate = structural_value.with_data(structural_value.data)
        coordinate.scale = self.engine.config.default_scale
        raised = _modulus_raise(
            self.engine,
            self._modraise_cache,
            coordinate,
            target_depth=self.modulus_raise_target_depth,
        )
        try:
            transformed = _apply_modraised_linear(
                ops,
                raised,
                self.coeffs_to_slots,
                self.coeffs_to_slots_evaluator,
                rotation_keys=rotation_keys,
                diagonal_cache=self._diagonal_cache,
                rotation_cache=self._rotation_cache,
                retain_diagonals=self.retain_diagonals,
            )
        finally:
            if not self.retain_diagonals:
                self._diagonal_cache.clear()
        transformed = ops.multiply_scalar(
            transformed, 1.0 / self.engine.num_slots
        )

        conjugated = self.engine.conjugate(transformed, conjugation_key)
        real = self.engine.add(transformed, conjugated)
        imaginary = self.engine.subtract(transformed, conjugated)
        imaginary = ops.multiply_by_monomial(
            imaginary, 3 * self.engine.num_slots
        )
        if self.batch_modular_branches:
            branches = real.with_data(
                torch.stack((real.data, imaginary.data), dim=1)
            )
            reduced_branches = self.modular_reduction.evaluate(
                ops,
                branches,
                relinearization_key=relinearization_key,
                conjugation_key=conjugation_key,
            )
            real = reduced_branches.with_data(reduced_branches.data[:, 0])
            imaginary = reduced_branches.with_data(reduced_branches.data[:, 1])
        else:
            real = self.modular_reduction.evaluate(
                ops,
                real,
                relinearization_key=relinearization_key,
                conjugation_key=conjugation_key,
            )
            imaginary = self.modular_reduction.evaluate(
                ops,
                imaginary,
                relinearization_key=relinearization_key,
                conjugation_key=conjugation_key,
            )
        imaginary = ops.multiply_by_monomial(imaginary, self.engine.num_slots)
        real, imaginary = ops.align_depths(real, imaginary)
        result = self._evaluate_linear_transform(
            ops,
            rotation_keys,
            self.slots_to_coeffs,
            self.slots_to_coeffs_evaluator,
            self.engine.add(real, imaginary),
        )
        result.scale *= entry_scale_ratio
        return result
