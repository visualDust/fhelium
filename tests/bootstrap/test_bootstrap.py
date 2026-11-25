"""Full-slot bootstrap depth placement and message recovery on a small ring."""

from __future__ import annotations

import math

import numpy as np
import torch

import fhelium as fh
from fhelium import bootstrap
from fhelium.algorithms import polynomial
from fhelium.eager import Engine


def _small_engine() -> Engine:
    config = fh.CkksConfig.catalog_suffix_chain(
        logN=12,
        scale_bits=40,
        depth=20,
        p_count=4,
        galois_generator=5,
        enforce_security_budget=False,
    )
    return Engine(config, allow_automatic_key_generation=False)


def _message(engine: Engine) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    real, imag = torch.rand(2, engine.num_slots, generator=generator, dtype=torch.float64)
    return ((2 * real - 1) + 1j * (2 * imag - 1)) / 16


def test_encapsulated_bootstrap_refreshes_message() -> None:
    engine = _small_engine()
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    plan = bootstrap.plan(engine, intermediate_weight=16)
    keys = bootstrap.create_evaluation_keys(engine, plan, secret_key)
    values = _message(engine)
    ciphertext = engine.encrypt_message(values, public_key, depth=plan.source_depth)

    refreshed = bootstrap.bootstrap(engine, ciphertext, plan, keys=keys)
    prepared = bootstrap.prepare(engine, plan, ciphertext, keys=keys)
    try:
        repeated = prepared(ciphertext)
    finally:
        prepared.close()

    assert [stage.output_depth for stage in plan.stages()] == [0, 4, 4, 13, 16]
    for value in (refreshed, repeated):
        assert value.depth == plan.output_depth == 16
        error = (engine.decrypt_message(value, secret_key) - values).abs().max()
        assert error < 2e-4


class _SeedAndDoubleAngle:
    r"""EvalMod from an interpolated cosine seed and $R$ double-angle steps.

    The seed approximates $\pi^{-1/2^R}\cos(\pi Bx/2^R-\pi/2^{R+1})$ and each
    step maps $z\mapsto 2z^2-\pi^{-1/2^{R-j}}$, giving $\sin(\pi Bx)/\pi$.
    """

    input_bound = 33
    fused_input_divisor = 33.0
    requires_relinearization = True
    iterations = 3

    def __init__(self) -> None:
        bound, steps = self.input_bound, 2**self.iterations
        self.seed = polynomial.chebyshev_fit(
            lambda x: math.pi ** (-1 / steps)
            * np.cos(math.pi * bound * x / steps - math.pi / (2 * steps)),
            38,
        )
        self.seed_plan = polynomial.BlockedChebyshevPlan(baby_step=8)

    @property
    def required_depths(self) -> int:
        return self.seed_plan.required_depths(self.seed) + self.iterations

    def _constants(self):
        return [
            math.pi ** (-1 / 2 ** (self.iterations - j))
            for j in range(1, self.iterations + 1)
        ]

    def reference(self, x):
        z = self.seed.evaluate_plaintext(x)
        for constant in self._constants():
            z = 2 * z * z - constant
        return z

    def evaluate(self, arithmetic, ciphertext, *, relinearization_key):
        z = self.seed_plan.evaluate(
            arithmetic, ciphertext, self.seed, relinearization_key=relinearization_key
        )
        for constant in self._constants():
            z = arithmetic.multiply_relinearize_rescale(
                z, z, relinearization_key=relinearization_key
            )
            z = arithmetic.add_scalar(arithmetic.engine.add(z, z), -constant)
        return z


def test_custom_eval_mod_replaces_builtin_component() -> None:
    engine = _small_engine()
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    eval_mod = _SeedAndDoubleAngle()
    grid = np.linspace(-1, 1, 4097)
    assert np.abs(eval_mod.reference(grid) - np.sin(math.pi * 33 * grid) / math.pi).max() < 1e-6
    plan = bootstrap.plan(engine, intermediate_weight=16, eval_mod=eval_mod)
    keys = bootstrap.create_evaluation_keys(engine, plan, secret_key)
    values = _message(engine)
    ciphertext = engine.encrypt_message(values, public_key, depth=plan.source_depth)

    refreshed = bootstrap.bootstrap(engine, ciphertext, plan, keys=keys)

    eval_mod_stage = next(s for s in plan.stages() if s.name == "eval_mod")
    assert eval_mod_stage.output_depth - eval_mod_stage.input_depth == eval_mod.required_depths
    assert refreshed.depth == plan.output_depth
    error = (engine.decrypt_message(refreshed, secret_key) - values).abs().max()
    assert error < 2e-4
