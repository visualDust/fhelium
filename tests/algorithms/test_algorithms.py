"""Encrypted polynomial and linear-transform evaluation against plaintext references."""

from __future__ import annotations

import numpy as np
import torch

import fhelium as fh
from fhelium import compile as fc
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
    real, imag = torch.rand(
        2, engine.num_slots, generator=generator, dtype=torch.float64
    )
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
        step
        for stage in stages
        for step in selected.required_rotation_offsets(stage)
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


def test_prepared_algorithms_inside_compile_match_eager() -> None:
    engine = _engine(8)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    stages = linear.radix2_dft(
        engine.num_slots, direction="coeffs_to_slots", stages=2, scale=0.01
    )
    selected = linear.plan(stages)
    steps = {
        step
        for stage in stages
        for step in selected.required_rotation_offsets(stage)
    }
    keys = EvaluationKeySet(
        [engine.create_rotation_key(step, secret_key) for step in sorted(steps)]
        + [engine.create_relinearization_key(secret_key)]
    )
    # The transform keeps hidden slots inside the fitted interval [-1, 1].
    p = polynomial.chebyshev_fit(np.tanh, 7)
    ciphertext = engine.encrypt_message(
        _message(engine, 0.5), public_key, depth=0
    )
    prepared = linear.prepare(engine, ciphertext, stages, selected, keys=keys)

    def layer(value):
        hidden = linear.apply(engine, value, prepared=prepared)
        return polynomial.evaluate(engine, hidden, p, keys=keys)

    eager = layer(ciphertext)
    compiled = fc.compile(layer, cache=False)(ciphertext)

    assert (compiled.depth, compiled.scale) == (eager.depth, eager.scale)
    difference = engine.decrypt_message(
        compiled, secret_key
    ) - engine.decrypt_message(eager, secret_key)
    assert difference.abs().max() < 1e-6


def test_non_subset_chain_inside_compile_matches_eager() -> None:
    # The built-in preset's depth-2 basis is not a subset of the depth-1 basis,
    # so the move and the polynomial's rescales lower to level reduction.
    engine = Engine(fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7))
    assert not engine.config.is_subbasis(1, 2)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    keys = EvaluationKeySet([engine.create_relinearization_key(secret_key)])
    ciphertext = engine.encrypt_message(
        _message(engine, 0.5).real, public_key, depth=1
    )

    def move(value):
        return engine.mod_switch_to_depth(value, 2)

    for source in (
        ciphertext,
        engine.coefficient_domain_to_ntt_domain(ciphertext),
    ):
        eager = move(source)
        compiled = fc.compile(move, cache=False)(source)
        assert (compiled.depth, compiled.scale) == (eager.depth, eager.scale)
        assert torch.equal(compiled.data, eager.data)

    p = polynomial.chebyshev_fit(np.tanh, 7)

    def activation(value):
        return polynomial.evaluate(engine, value, p, keys=keys)

    eager = activation(ciphertext)
    compiled = fc.compile(activation, cache=False)(ciphertext)
    assert (compiled.depth, compiled.scale) == (eager.depth, eager.scale)
    difference = engine.decrypt_message(
        compiled, secret_key
    ) - engine.decrypt_message(eager, secret_key)
    assert difference.abs().max() < 1e-6


def test_unprepared_algorithms_inside_compile_match_eager() -> None:
    engine = _engine(6)
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    slots = engine.num_slots
    # One constant (compressible) and one dense diagonal.
    transform = linear.LinearTransform(
        {0: np.full(slots, 0.5 + 0j), 1: np.linspace(-0.2, 0.2, slots) + 0j},
        slots,
    )
    keys = EvaluationKeySet(
        [
            engine.create_rotation_key(step, secret_key)
            for step in linear.plan(transform).required_rotation_offsets(
                transform
            )
        ]
        + [engine.create_relinearization_key(secret_key)]
    )
    p = polynomial.chebyshev_fit(np.tanh, 7)
    ciphertext = engine.encrypt_message(_message(engine, 0.5), public_key)

    def layer(value):
        hidden = linear.apply(engine, value, transform, keys=keys)
        return polynomial.evaluate(
            engine, hidden, p, polynomial.plan(p), keys=keys
        )

    eager = layer(ciphertext)
    compiled = fc.compile(layer, cache=False)(ciphertext)

    assert (compiled.depth, compiled.scale) == (eager.depth, eager.scale)
    difference = engine.decrypt_message(
        compiled, secret_key
    ) - engine.decrypt_message(eager, secret_key)
    assert difference.abs().max() < 1e-6
