r"""Engine-bound callable full-slot CKKS bootstrap composition.

The composition accepts and returns two-component,
coefficient-domain, standard-residue Q RNS values. Temporary
NTT/Montgomery values are confined to ordinary engine arithmetic. Ciphertext
payload axes are `[component, *batch, limb, coefficient]`; linear-map diagonal
payloads use `[slot]` with $S=N/2$.
"""

from __future__ import annotations

from typing import Any

from fhelium.core import (
    Ciphertext,
    EvaluationKeyRequirements,
    EvaluationKeySet,
    KeySwitchKey,
    Plaintext,
    RotationKey,
    RotationKeySet,
    SecretKey,
)
from fhelium.engine.ckks_engine import CkksEngine
from fhelium.experimental.bootstrap._modraise import (
    _modulus_raise,
    _prepare_entry,
    _rescale_to_structural_base,
)
from fhelium.experimental.bootstrap._ops import (
    _align_levels,
    _apply_linear_transform,
    _apply_modraised_linear,
    _multiply_by_monomial,
    _multiply_scalar,
)
from fhelium.core.rotation import (
    decompose_signed_power_of_two_rotation,
)


class FullSlotBootstrap:
    r"""Compiled full-slot refresh callable with replaceable components.

    Construction binds transform compilers/evaluators and modular reduction to
    one engine. Calling the object executes the visible full-slot algorithm with
    one validated evaluator-only key inventory.

    Let $\Delta_0$ be `engine.config.default_scale`, $q_b$ the one-prime
    structural Q base, $S=N/2$, $C$ the unscaled CoeffsToSlots map, and $T$ the
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

    Polynomial approximation, CKKS arithmetic, key switching, and scale
    reinterpretation perturb this idealized map. The caller must establish the
    reducer's raw-coordinate precondition
    $|r_{\rm R}|,|r_{\rm I}|\le B$; encrypted execution cannot inspect it.
    """

    def __init__(
        self,
        engine: CkksEngine,
        *,
        coeffs_to_slots_compiler: Any,
        coeffs_to_slots_evaluator: Any,
        modular_reduction: Any,
        slots_to_coeffs_compiler: Any,
        slots_to_coeffs_evaluator: Any,
        modulus_raise_target_level: int = 0,
        retain_diagonals: bool = False,
    ) -> None:
        if modulus_raise_target_level < 0:
            raise ValueError('modulus_raise_target_level cannot be negative')
        if modulus_raise_target_level >= engine.public_level_count:
            raise ValueError(
                'modulus_raise_target_level is outside the Q chain'
            )
        scale = float(engine.config.default_scale)
        structural_base = engine.config.q_moduli[-1]
        ratio = structural_base / scale
        if not 0.5 <= ratio <= 2.0:
            raise ValueError(
                'Bootstrap structural base must be close to the CKKS scale; '
                f'got q_base/Delta={ratio:.6g}'
            )

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
        self.modulus_raise_target_level = modulus_raise_target_level
        self.retain_diagonals = retain_diagonals
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
        forward_levels = sum(
            coeffs_to_slots_evaluator.required_levels(stage)
            for stage in self.coeffs_to_slots
        )
        inverse_levels = sum(
            slots_to_coeffs_evaluator.required_levels(stage)
            for stage in self.slots_to_coeffs
        )
        self.output_level = (
            modulus_raise_target_level
            + forward_levels
            + 1
            + modular_reduction.required_levels
            + inverse_levels
        )
        if self.output_level >= engine.public_level_count:
            raise ValueError(
                'bootstrap depth exceeds the modulus chain: '
                f'output={self.output_level}, '
                f'final={engine.final_public_level}'
            )

        self._diagonal_cache: dict[tuple[int, int, int, int], Plaintext] = {}
        self._rotation_cache: dict[int, tuple[int, ...]] = {}
        self._rotation_inventory_steps: tuple[int, ...] | None = None
        self._modraise_cache: dict[object, object] = {}

    @property
    def required_rotations(self) -> tuple[int, ...]:
        r"""Return canonical signed $\operatorname{Rot}_r$ steps for both maps."""

        return tuple(
            sorted(
                {
                    RotationKey.canonical_step(
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

    def clear_cache(self) -> None:
        """Release encoded diagonals and non-semantic arithmetic tables."""

        self._diagonal_cache.clear()
        self._rotation_cache.clear()
        self._modraise_cache.clear()

    def _evaluate_linear_transform(
        self,
        rotation_keys: RotationKeySet,
        stages: tuple[Any, ...],
        evaluator: Any,
        ciphertext: Ciphertext,
    ) -> Ciphertext:
        """Evaluate one compiled transform under the selected cache policy."""

        try:
            return _apply_linear_transform(
                self.engine,
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
        r"""Refresh a final-public-level full-slot ciphertext or dense batch.

        The input must be a context-compatible two-component Q ciphertext in
        coefficient domain with standard residues, active `prime_ids`,
        data axes `[component, *batch, limb, coefficient]`, ring extent $N$,
        final public level $L-1$, and actual scale near either $\Delta_0$ or
        $\Delta_0^2$. All $S$ slots are transformed; there is no sparse-slot
        mode. ``evaluation_keys`` must provide every selected transform
        rotation and a conjugation key. The built-in reductions also require a
        relinearization key for ciphertext products; a custom slotwise
        reduction that performs no such product may omit it from the inventory.

        The state and scale recurrence is:

        1. An ordinary-scale input is multiplied by encoded $1$ at scale
           $\Delta_0$, producing pending scale
           $\Delta_{\rm pre}=\Delta_{\rm in}\Delta_0$ without consuming a
           level. A pending-scale input passes through with
           $\Delta_{\rm pre}=\Delta_{\rm in}$.
        2. The private nearest structural transition drops the last public
           scale prime $q_{L-1}$, first producing actual scale
           $\Delta_{\rm pre}/q_{L-1}$ at internal level $L$ over `[q_b]`.
           The unchanged residues are then explicitly reinterpreted at
           $\Delta_0$.
        3. Centered ModRaise extends each component from `[q_b]` to the target
           Q `prime_ids`. It preserves coefficient domain, standard residues,
           two components, batch axes, represented centered integers, and
           metadata scale $\Delta_0$; it performs no rescale.
        4. For each CoeffsToSlots stage $j$, diagonal multiplication and core
           rescale update

           $$
           \ell_{j+1}=\ell_j+1,\qquad
           \Delta_{j+1}=\Delta_j\Delta_0/q_j,
           $$

           and remove leading Q row $q_j$. The explicit $1/S$ scalar step
           consumes one more level and reinterprets its result at $\Delta_0$.
        5. Conjugation, branch addition/subtraction, and multiplication by
           $X^{3S}$ preserve level, scale $\Delta_0$, domain, basis, rows, and
           component count. Each built-in periodic reduction advances by its
           `required_levels` and returns at scale $\Delta_0$. Multiplication of
           the imaginary result by $X^S$ and branch recombination preserve that
           state.
        6. If SlotsToCoeffs starts at level $\ell_T$ with scale $\Delta_0$, its
           $m_T$ stages follow the same core-rescale recurrence. Therefore the
           final actual scale is

           $$
           \Delta_{\rm out}=\Delta_0
           \prod_{j=0}^{m_T-1}\frac{\Delta_0}{q_{\ell_T+j}},
           $$

           not an implicit reset to $\Delta_0$.

        The functional result does not alias the input. It is a two-component
        coefficient-domain standard-RNS Q ciphertext with unchanged batch
        axes, level `output_level`, and
        `engine.rns_layout.prime_ids(output_level)`. The method does not enforce
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

        prepared = _prepare_entry(self.engine, ciphertext)
        structural_value = _rescale_to_structural_base(self.engine, prepared)
        raised = _modulus_raise(
            self.engine,
            self._modraise_cache,
            structural_value,
            target_level=self.modulus_raise_target_level,
        )
        try:
            transformed = _apply_modraised_linear(
                self.engine,
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
        transformed = _multiply_scalar(
            self.engine,
            transformed,
            1.0 / self.engine.num_slots,
        )

        conjugated = self.engine.conjugate(transformed, conjugation_key)
        real = self.engine.add(transformed, conjugated)
        imaginary = self.engine.subtract(transformed, conjugated)
        imaginary = _multiply_by_monomial(
            self.engine,
            imaginary,
            3 * self.engine.num_slots,
        )
        real = self.modular_reduction.evaluate(
            self.engine,
            real,
            relinearization_key=relinearization_key,
            conjugation_key=conjugation_key,
        )
        imaginary = self.modular_reduction.evaluate(
            self.engine,
            imaginary,
            relinearization_key=relinearization_key,
            conjugation_key=conjugation_key,
        )
        imaginary = _multiply_by_monomial(
            self.engine,
            imaginary,
            self.engine.num_slots,
        )
        real, imaginary = _align_levels(
            self.engine,
            real,
            imaginary,
        )
        return self._evaluate_linear_transform(
            rotation_keys,
            self.slots_to_coeffs,
            self.slots_to_coeffs_evaluator,
            self.engine.add(real, imaginary),
        )
