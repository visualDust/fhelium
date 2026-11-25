r"""CKKS full-slot bootstrapping composed from ordinary Eager operations.

:func:`plan` returns a :class:`BootstrapPlan` holding every component choice
(modulus raise, CoeffsToSlots, EvalMod, SlotsToCoeffs) and its depth placement.
:func:`create_evaluation_keys` provisions the keys the plan requires,
:func:`bootstrap` refreshes a ciphertext, and :func:`prepare` returns a
caller-owned :class:`PreparedBootstrap` for repeated execution and capture.
The step functions (:func:`modulus_raise`, :func:`coeffs_to_slots`,
:func:`split_real_imag`, :func:`eval_mod`, :func:`merge_real_imag`,
:func:`slots_to_coeffs`) evaluate one step each for custom compositions.
"""

from __future__ import annotations

import torch

from fhelium.bootstrap import periodic
from fhelium.bootstrap._core import (
    PreparedBootstrap,
    _CoreAlgorithm,
    _CoreMaterials,
    _entry,
)
from fhelium.bootstrap._modulus_raise import Centered, Encapsulated
from fhelium.bootstrap._plan import BootstrapPlan, BootstrapStage, _Bound, plan
from fhelium.runtime.signature import ValueSignature


def create_evaluation_keys(
    engine,
    plan: BootstrapPlan,
    secret_key,
    *,
    into=None,
    rotation_key_strategy: str = "all_required",
    device=None,
):
    """Add the keys required by ``plan`` to an inventory and return it.

    Rotation keys follow ``rotation_key_strategy`` (see
    :meth:`BootstrapPlan.key_steps`); keys already present in ``into`` with a
    compatible layout are reused. Relinearization and conjugation keys are
    added when missing. An :class:`Encapsulated` raise also adds its low and
    return key-switch keys, sampling the intermediate secret internally.
    """

    from fhelium.bootstrap._keys import _create_evaluation_keys

    inventory = _create_evaluation_keys(
        _Bound(engine, plan),
        secret_key,
        into=into,
        rotation_key_strategy=rotation_key_strategy,
        device=device,
    )
    if isinstance(plan.modulus_raise, Encapsulated):
        from fhelium.bootstrap._encapsulation import _add_encapsulation_keys

        _add_encapsulation_keys(
            engine, plan, secret_key, inventory, device=device
        )
    return inventory


def prepare(engine, plan: BootstrapPlan, like, *, keys) -> PreparedBootstrap:
    """Encode the plan's constants and diagonals and select its keys once.

    ``like`` fixes the input signature (depth, scale, batch shape, device);
    preparation refreshes it once and discards the output. Call before CUDA
    Graph capture.
    """

    if like.device.type == "cuda":
        with torch.cuda.device(like.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare a bootstrap before CUDA Graph capture")
    algorithm = _CoreAlgorithm(engine, plan)
    materials = _CoreMaterials()
    signature = ValueSignature.from_value(like)
    coordinate, _ = _entry(engine, plan, like)
    algorithm.evaluate(coordinate, keys, materials)
    del coordinate
    return PreparedBootstrap(algorithm, materials, signature, like.device)


def bootstrap(engine, ciphertext, plan: BootstrapPlan | None = None, *, keys=None, prepared=None):
    r"""Refresh a coefficient-domain ciphertext at depth $\le$ ``plan.source_depth``.

    An input below the source depth first reaches it with
    ``mod_switch_to_depth``. The source-basis value $(c,\Delta_b)$ is evaluated
    in coordinates of scale $\Delta_0=$ ``plan.input_scale``; the output at
    ``plan.output_depth`` records actual scale $s_{out}\Delta_b/\Delta_0$ so
    that it represents the input message. With ``prepared``, its materials are
    used and ``plan``/``keys`` are not consulted; otherwise constants and
    diagonals are encoded and keys selected from ``keys`` for this call.
    """

    if prepared is not None:
        return prepared(ciphertext)
    if plan is None:
        raise TypeError("bootstrap requires a plan or prepared materials")
    algorithm = _CoreAlgorithm(engine, plan)
    coordinate, ratio = _entry(engine, plan, ciphertext)
    output = algorithm.evaluate(coordinate, keys, _CoreMaterials())
    return engine.reinterpret_at_scale(output, output.scale * ratio)


def _step(engine, plan):
    algorithm = _CoreAlgorithm(engine, plan)
    materials = _CoreMaterials()
    return algorithm, materials, algorithm.arithmetic(materials)


def modulus_raise(engine, ciphertext, plan: BootstrapPlan, *, keys=None):
    """Raise a source-basis ciphertext to ``plan.raise_target_depth``."""
    return plan.modulus_raise._raise(
        engine,
        ciphertext,
        plan=plan,
        target_depth=plan.raise_target_depth,
        keys=keys,
    )


def coeffs_to_slots(engine, ciphertext, plan: BootstrapPlan, *, keys):
    r"""Apply $\alpha_C C$ and the $1/S$ normalization to a raised ciphertext."""
    algorithm, materials, ops = _step(engine, plan)
    return algorithm.coeffs_to_slots(ops, ciphertext, keys, materials, True)


def split_real_imag(engine, ciphertext, plan: BootstrapPlan, *, keys):
    r"""Return $(z+\bar z, X^{3S}(z-\bar z))$; the second item is ``None`` for real messages.

    Multiplication by $X^{3S}$ maps the imaginary branch $2i\Im z$ to the real
    slot vector $2\Im z$. Depth and scale are preserved.
    """
    algorithm, materials, ops = _step(engine, plan)
    return algorithm.split(ops, ciphertext, keys, materials, True)


def eval_mod(engine, real, imaginary, plan: BootstrapPlan, *, keys):
    """Apply ``plan.eval_mod`` to one or both branches (batched when ``plan.batch_branches``)."""
    algorithm, materials, ops = _step(engine, plan)
    return algorithm.eval_mod(ops, real, imaginary, keys, materials, True)


def merge_real_imag(engine, real, imaginary, plan: BootstrapPlan):
    r"""Return $r+X^{S}i$ after aligning the branch depths."""
    algorithm, _, ops = _step(engine, plan)
    return algorithm.merge(ops, real, imaginary)


def slots_to_coeffs(engine, ciphertext, plan: BootstrapPlan, *, keys):
    r"""Apply $\alpha_T T$ (and the real-message reconstruction $z+\bar z$)."""
    algorithm, materials, ops = _step(engine, plan)
    return algorithm.slots_to_coeffs(ops, ciphertext, keys, materials, True)


__all__ = [
    "BootstrapPlan",
    "BootstrapStage",
    "Centered",
    "Encapsulated",
    "PreparedBootstrap",
    "bootstrap",
    "coeffs_to_slots",
    "create_evaluation_keys",
    "eval_mod",
    "merge_real_imag",
    "modulus_raise",
    "periodic",
    "plan",
    "prepare",
    "slots_to_coeffs",
    "split_real_imag",
]
