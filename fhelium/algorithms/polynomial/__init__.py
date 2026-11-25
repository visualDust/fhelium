r"""Polynomials, evaluation plans and homomorphic polynomial evaluation.

A :class:`Polynomial` stores ascending power or Chebyshev coefficients over an
interval. A plan object (for example :class:`BlockedChebyshevPlan`) selects the
evaluation schedule and reports ``required_depths(p)`` and
``operation_inventory(p)``. :func:`evaluate` executes a plan on a ciphertext.

A plan, built-in or custom, provides:

- ``required_depths(p)``: the depths $L$ consumed by every execution path;
- ``evaluate(arithmetic, ciphertext, p, *, relinearization_key=None)``:
  return $p(x)$ for an input at depth $d$ at depth $d+L$, with two
  components and the scale reached through ``arithmetic``
  (:class:`~fhelium.algorithms.Arithmetic`). ``relinearization_key`` is a
  key, an ``EvaluationKeySet`` or a depth-to-key mapping and is passed
  unchanged to ``arithmetic.relinearize`` or
  ``arithmetic.multiply_relinearize_rescale``;
- optionally ``relinearization_depths(p, *, input_depth)``: the depths of its
  ciphertext products, which key provisioning uses; without it, keys covering
  every depth are provisioned;
- optionally ``operation_inventory(p)``.

The input coordinate is the polynomial's normalized coordinate $x$.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fhelium.algorithms.polynomial._approximation import (
    ChebyshevInterpolator,
    Polynomial,
)
from fhelium.algorithms.polynomial._blocked import BlockedChebyshevPlan
from fhelium.algorithms.polynomial._chebyshev import BinaryChebyshevPlan
from fhelium.algorithms.polynomial._even import EvenChebyshevPlan
from fhelium.algorithms.polynomial._recursive import RecursiveChebyshevPlan
from fhelium.algorithms.polynomial._power import (
    BalancedPowerPlan,
    HornerPlan,
    PatersonStockmeyerPlan,
)


def chebyshev_fit(
    function: Callable[[np.ndarray], np.ndarray],
    degree: int,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    name: str = "polynomial",
) -> Polynomial:
    r"""Interpolate ``function`` at the $\text{degree}+1$ first-kind Chebyshev nodes.

    The returned coefficients are in the Chebyshev basis of the normalized
    coordinate $x=(2t-a-b)/(b-a)$ for ``domain=(a, b)``; ``max_error`` records
    the sampled maximum deviation on the interval.
    """

    return ChebyshevInterpolator(degree=degree).approximate(
        function, domain=domain, name=name
    )


def plan(p: Polynomial):
    """Return the default plan: blocked evaluation for Chebyshev, balanced powers otherwise."""

    return (
        BlockedChebyshevPlan()
        if p.basis == "chebyshev"
        else BalancedPowerPlan()
    )


def evaluate(
    engine,
    ciphertext,
    p: Polynomial,
    plan=None,
    *,
    keys=None,
    output_scale: float | None = None,
):
    r"""Evaluate $p(x)$ slot-wise on a ciphertext at depth $d$.

    The result is at depth $d+L$ with $L=$ ``plan.required_depths(p)`` and
    actual scale ``output_scale`` (default $\Delta_{d+L}$ of the configuration).
    Intermediate basis values follow $s_{k+1}=s_k^2/\rho_k$ from the input's
    actual scale; coefficient products select plaintext scales that reach the
    configured scale of their output depth. ``keys`` supplies the
    relinearization key, either directly or through an ``EvaluationKeySet``;
    degree-one polynomials need none. The input polynomial domain is retained.
    """

    from fhelium.algorithms._arithmetic import Arithmetic

    selected = globals()["plan"](p) if plan is None else plan
    depth = ciphertext.depth + selected.required_depths(p)
    scales = list(engine.config.scales)
    if output_scale is not None:
        scales[depth] = float(output_scale)
    arithmetic = Arithmetic(engine, {}, _scales=tuple(scales))
    return selected.evaluate(
        arithmetic, ciphertext, p, relinearization_key=keys
    )


__all__ = [
    "BalancedPowerPlan",
    "BinaryChebyshevPlan",
    "BlockedChebyshevPlan",
    "ChebyshevInterpolator",
    "EvenChebyshevPlan",
    "HornerPlan",
    "PatersonStockmeyerPlan",
    "Polynomial",
    "RecursiveChebyshevPlan",
    "chebyshev_fit",
    "evaluate",
    "plan",
]
