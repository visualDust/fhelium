"""CKKS rescaling and the bootstrap structural-base transition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from fhelium.core import Ciphertext
from fhelium.core.scale import coerce_scale
from fhelium.engine.rns.montgomery import MontgomeryParameters
from fhelium.engine.rns.runtime import RnsRuntime
from fhelium.errors import MaximumLevelError
from fhelium.native.wrapper import ckks_ops


@dataclass(frozen=True)
class _ScalePrimeDrop:
    """Precomputed metadata shared by copy and in-place rescale kernels.

    The inverse vector has ``[remaining_limb]`` layout in Montgomery scalar
    form; ``remaining_parameters`` has ``[parameter, remaining_limb]`` layout.
    Both use engine integral dtype/device and align exactly with the surviving
    ``prime_ids``. The record owns references to engine table views and callers
    must not mutate them.
    """

    next_level: int
    dropped_q_inverse_montgomery_by_remaining_row: torch.Tensor
    remaining_parameters: torch.Tensor
    half_dropped_prime: int
    output_scale: float


class CkksRescaler:
    r"""Execute one-prime CKKS rescale transitions for an engine.

    Each transition divides by the leading active Q prime
    $q_{\mathrm{drop}}$, rounds the quotient, and removes that residue row.
    The class serves two kinds of level transition:

    * :meth:`rescale_to_next_level` and :meth:`rescale_to_next_level_` operate only on
      public CKKS levels;
    * :meth:`_rescale_final_public_level_to_structural_base` performs the one
      additional private transition needed immediately before bootstrap
      modulus raising.
    """

    def __init__(
        self,
        *,
        engine_id: str,
        device: torch.device,
        public_level_count: int,
        rns_runtime: RnsRuntime,
        montgomery_parameters: MontgomeryParameters,
        dropped_q_inverses_montgomery_by_level: list[torch.Tensor],
        assert_engine_ciphertext: Callable[[Ciphertext], None],
        ciphertext_from_components: Callable[..., Ciphertext],
        rescale_to_next_output_scale: Callable[..., float],
    ) -> None:
        self.engine_id = engine_id
        self.device = device
        self.public_level_count = public_level_count
        self.rns_runtime = rns_runtime
        self.montgomery_parameters = montgomery_parameters
        self.dropped_q_inverses_montgomery_by_level = (
            dropped_q_inverses_montgomery_by_level
        )
        self._assert_engine_ciphertext = assert_engine_ciphertext
        self._ciphertext_from_components = ciphertext_from_components
        self._rescale_to_next_output_scale = rescale_to_next_output_scale

    def __str__(self) -> str:
        return (
            f"CkksRescaler(engine_id={self.engine_id}, "
            f"public_levels={self.public_level_count}, device={self.device})"
        )

    __repr__ = __str__

    def rescale_to_next_level(
        self,
        ct: Ciphertext,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        r"""Drop the current scale prime at an ordinary public CKKS level.

        Rescaling divides by the leading active Q prime and removes that RNS
        row. For each component,

        $$
        c'=\operatorname{Round}\!\left(
          \frac{c}{q_{\mathrm{drop}}}
        \right)\pmod{B_{\ell+1}},\qquad
        \Delta(c')=\frac{\Delta(c)}{q_{\mathrm{drop}}}.
        $$

        Q input produces $Q_{\ell+1}$; QP input produces
        $Q_{\ell+1}P$ and retains all P rows.

        Public rescaling stops before the one-prime structural basis. This
        method returns new ciphertext storage; :meth:`rescale_to_next_level_`
        mutates its ciphertext argument.

        Args:
            ct: Coefficient-domain, standard-residue ciphertext over the
                engine's full active Q or QP layout.
            rounding: ``"nearest"`` for nearest-integer division or
                ``"floor"`` for the least-nonnegative-residue quotient.
        Returns:
            A new coefficient-domain standard ciphertext at ``ct.level + 1``
            with unchanged component count, batch shape, Q/QP basis, engine
            integral dtype/device, exact ``prime_ids=ct.prime_ids[1:]``, and
            canonical residues in $[0,q_i)$. ``ct`` is unchanged and output
            storage is independent.

        Raises:
            MaximumLevelError: If no further public rescale level exists.
            InvalidScaleError: If the output scale is invalid.
            ValueError: If the ciphertext state is incompatible with rescaling.
        """

        self._require_public_rescale_level(ct)
        self._validate_rounding(rounding)
        return self._drop_leading_scale_prime(
            ct,
            rounding=rounding,
        )

    def rescale_to_next_level_(
        self,
        ct: Ciphertext,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        r"""In-place form of :meth:`rescale_to_next_level`.

        Native kernels update the remaining RNS rows through views into
        ``ct.data``.  This method then narrows the dense tensor by one row and
        updates ``level``, ``scale``, and ``prime_ids``.  The returned object is
        ``ct`` itself; aliases must therefore be treated as mutated.

        The quotient and actual-scale equations are identical to
        :meth:`rescale_to_next_level`. Aliases observe updated surviving rows,
        narrowed ``ct.data``, level, scale, and exact ``prime_ids``. The
        narrowed tensor remains a view of the original allocation.
        """

        self._require_public_rescale_level(ct)
        self._validate_rounding(rounding)
        return self._drop_leading_scale_prime_(
            ct,
            rounding=rounding,
        )

    @staticmethod
    def _validate_rounding(rounding: str) -> None:
        if rounding not in {"nearest", "floor"}:
            raise ValueError(
                "rounding must be either 'nearest' or 'floor'; "
                f"got {rounding!r}"
            )

    def _require_public_rescale_level(self, ct: Ciphertext) -> None:
        """Validate that ``ct`` has a following public CKKS level."""

        if ct.level >= self.public_level_count - 1:
            raise MaximumLevelError(
                level=ct.level,
                maximum_level=self.public_level_count - 1,
            )

    def _rescale_final_public_level_to_structural_base(
        self,
        ct: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        r"""Enter the one-prime structural basis required by ModRaise.

        The method applies the out-of-place divide-round-drop formula used by
        :meth:`rescale_to_next_level` in the bootstrap-only transition after the final
        public level. The active basis is conceptually
        ``[q_last_scale, q_structural_base]``; dropping its leading row leaves
        only ``q_structural_base``.

        The result is the internal bootstrap representation consumed by
        centered modulus raising.

        $$
        c'=\operatorname{Round}\!\left(
          \frac{c}{q_{\mathrm{last}}}
        \right)\bmod q_{\mathrm{structural}},\qquad
        \Delta(c')=\frac{\Delta(c)}{q_{\mathrm{last}}}.
        $$

        Args:
            ct: Final-public-level coefficient-domain, standard-residue
                ciphertext.
            rounding: ``"nearest"`` or ``"floor"`` quotient selection.

        Returns:
            A new ciphertext at internal level ``public_level_count`` over the
            single structural base prime.

        Raises:
            InvalidScaleError: If the output scale is invalid.
            ValueError: If ``ct`` is not at the final public level or has an
                incompatible arithmetic state.
        """

        if ct.level != self.public_level_count - 1:
            raise ValueError(
                "structural-base rescale requires the final public level"
            )
        self._validate_rounding(rounding)
        return self._drop_leading_scale_prime(
            ct,
            rounding=rounding,
        )

    def _prepare_scale_prime_drop(self, ct: Ciphertext) -> _ScalePrimeDrop:
        """Validate one divide-round-drop step and collect native metadata.

        The returned object contains only level- and modulus-dependent values.
        Ciphertext component views are intentionally obtained by the copy or
        in-place caller so mutation is visible at the call site. Its inverse
        vector is ``[remaining_limb]`` and parameter table is
        ``[parameter, remaining_limb]`` in engine integral dtype/device,
        aligned with ``ct.prime_ids[1:]``. It aliases engine-owned tables.
        """

        ct.assert_state(
            polynomial_domain="coefficient", residue_representation="standard"
        )
        self._assert_engine_ciphertext(ct)
        next_level = ct.level + 1
        if next_level > self.public_level_count:
            raise MaximumLevelError(
                level=ct.level,
                maximum_level=self.public_level_count,
            )
        remaining_prime_ids = ct.prime_ids[1:]
        remaining_parameters = self.rns_runtime.rns_parameters_for_prime_ids(
            remaining_prime_ids
        )
        dropped_prime_id = ct.prime_ids[0]
        dropped_prime = self.montgomery_parameters.moduli[dropped_prime_id]
        output_scale = (
            self._rescale_to_next_output_scale(ct.scale, level=ct.level)
            if ct.level < self.public_level_count - 1
            else coerce_scale(
                ct.scale / float(dropped_prime),
                value_name="rescale result",
            )
        )
        return _ScalePrimeDrop(
            next_level=next_level,
            dropped_q_inverse_montgomery_by_remaining_row=(
                self.dropped_q_inverses_montgomery_by_level[ct.level][
                    : len(remaining_prime_ids)
                ]
            ),
            remaining_parameters=remaining_parameters,
            half_dropped_prime=(dropped_prime // 2),
            output_scale=output_scale,
        )

    @staticmethod
    def _rescale_component(
        remaining: torch.Tensor,
        dropped: torch.Tensor,
        step: _ScalePrimeDrop,
        *,
        rounding: Literal["nearest", "floor"],
    ) -> torch.Tensor:
        r"""Return one rescaled component without mutating source storage.

        ``remaining`` is engine-integral
        ``[*batch, remaining_limb, coefficient]`` in coefficient-domain
        standard form; ``dropped`` is the matching
        ``[*batch, coefficient]`` leading-Q row. Limb $i$ maps exactly to the
        surviving ``prime_ids[i]``. For each coefficient, compute the selected
        quotient of division by $q_{\mathrm{drop}}$ modulo every remaining
        prime. The native kernel may collapse only a zero-copy batch prefix.
        Output has the ``remaining`` shape/dtype/device, canonical $[0,q_i)$
        residues, and independent storage.
        """

        if rounding == "nearest":
            return ckks_ops.rescale_drop_leading_prime_nearest(
                remaining,
                step.dropped_q_inverse_montgomery_by_remaining_row,
                dropped,
                step.remaining_parameters,
                step.half_dropped_prime,
            )
        return ckks_ops.rescale_drop_leading_prime_truncate(
            remaining,
            step.dropped_q_inverse_montgomery_by_remaining_row,
            dropped,
            step.remaining_parameters,
        )

    @staticmethod
    def _rescale_component_(
        remaining: torch.Tensor,
        dropped: torch.Tensor,
        step: _ScalePrimeDrop,
        *,
        rounding: Literal["nearest", "floor"],
    ) -> None:
        r"""Rescale one component's surviving RNS-row view in place.

        Tensor axes, dtype/device, exact row mapping, quotient law, and
        canonical $[0,q_i)$ output match :meth:`_rescale_component`. The
        ``remaining`` view is mutated; ``dropped`` and table tensors are read
        only. Aliases of surviving rows observe the update.
        """

        if rounding == "nearest":
            ckks_ops.rescale_drop_leading_prime_nearest_(
                remaining,
                step.dropped_q_inverse_montgomery_by_remaining_row,
                dropped,
                step.remaining_parameters,
                step.half_dropped_prime,
            )
            return
        ckks_ops.rescale_drop_leading_prime_truncate_(
            remaining,
            step.dropped_q_inverse_montgomery_by_remaining_row,
            dropped,
            step.remaining_parameters,
        )

    def _drop_leading_scale_prime(
        self,
        ct: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"],
    ) -> Ciphertext:
        r"""Return the out-of-place RNS divide-round-drop result.

        For each component, the native kernel subtracts the dropped residue
        from every remaining residue, multiplies by the dropped prime's modular
        inverse, and applies the selected rounding correction.  A new
        ciphertext is then constructed from those remaining rows. Component
        input layout ``[*batch, limb, coefficient]`` becomes
        ``[*batch, limb-1, coefficient]`` with engine integral dtype/device,
        canonical standard residues, and exact ``prime_ids[1:]``. Output
        storage is independent.
        """

        step = self._prepare_scale_prime_drop(ct)
        components = [
            self._rescale_component(
                component[..., 1:, :],
                component[..., 0, :],
                step,
                rounding=rounding,
            )
            for component in (
                ct.component(index) for index in range(ct.component_count)
            )
        ]
        return self._ciphertext_from_components(
            components,
            level=step.next_level,
            scale=step.output_scale,
            polynomial_domain=ct.polynomial_domain,
            modulus_basis=ct.modulus_basis,
            residue_representation=ct.residue_representation,
            prime_ids=ct.prime_ids[1:],
        )

    def _drop_leading_scale_prime_(
        self,
        ct: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"],
    ) -> Ciphertext:
        """Apply the same prime drop and narrow ``ct`` storage in place.

        Surviving component views are overwritten with canonical quotient
        residues, then ``ct.data`` is narrowed by one limb without allocation.
        Aliases observe row updates and metadata mutation.
        """

        step = self._prepare_scale_prime_drop(ct)
        for component in (
            ct.component(index) for index in range(ct.component_count)
        ):
            self._rescale_component_(
                component[..., 1:, :],
                component[..., 0, :],
                step,
                rounding=rounding,
            )
        ct.data = ct.data[..., 1:, :]
        ct.level = step.next_level
        ct.scale = step.output_scale
        ct.prime_ids = ct.prime_ids[1:]
        return ct
