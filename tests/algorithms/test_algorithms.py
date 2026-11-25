"""Encrypted polynomial and linear-transform evaluation against plaintext references."""

from __future__ import annotations

import numpy as np
import torch

import fhelium as fh
from fhelium.algorithms import linear, polynomial
from fhelium.eager import Engine
from fhelium.values import EvaluationKeySet


def _engine(depth: int) -> Engine:
    config = fh.CkksConfig.catalog_suffix_chain(
        logN=12,
        scale_bits=40,
        depth=depth,
        p_count=2,
        galois_generator=5,
        enforce_security_budget=False,
    )
    return Engine(config, allow_automatic_key_generation=False)


def _message(engine: Engine, amplitude: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    real, imag = torch.rand(2, engine.num_slots, generator=generator, dtype=torch.float64)
    return amplitude * ((2 * real - 1) + 1j * (2 * imag - 1))


def test_polynomial_evaluate_matches_plaintext_reference() -> None:
    engine = _engine(8)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    keys = EvaluationKeySet([engine.create_relinearization_key(secret_key)])
    p = polynomial.chebyshev_fit(np.tanh, 15)
    values = _message(engine, 1.0).real
    ciphertext = engine.encrypt_message(values, public_key, depth=1)

    result = polynomial.evaluate(engine, ciphertext, p, keys=keys)

    assert result.depth == 1 + polynomial.plan(p).required_depths(p)
    decoded = engine.decrypt_message(result, secret_key, is_real=True)
    expected = p.evaluate_plaintext(values.numpy())
    assert np.abs(decoded.numpy() - expected).max() < 1e-5


def test_linear_apply_matches_plaintext_reference() -> None:
    engine = _engine(4)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    stages = linear.radix2_dft(
        engine.num_slots, direction="coeffs_to_slots", stages=2, scale=0.5
    )
    selected = linear.plan(stages)
    steps = {
        step for stage in stages for step in selected.required_rotation_offsets(stage)
    }
    keys = EvaluationKeySet(
        engine.create_rotation_key(step, secret_key) for step in sorted(steps)
    )
    values = _message(engine, 0.5)
    ciphertext = engine.encrypt_message(values, public_key, depth=0)

    result = linear.apply(engine, ciphertext, stages, selected, keys=keys)

    expected = values.numpy()
    for stage in stages:
        expected = stage.reference(expected)
    assert result.depth == len(stages)
    decoded = engine.decrypt_message(result, secret_key)
    assert np.abs(decoded.numpy() - expected).max() < 1e-5
