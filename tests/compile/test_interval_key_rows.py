"""Compiled key switching on interval bases whose rows are not the trailing key rows."""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch

import fhelium as fh
from fhelium import compile as fc
from fhelium.eager import Engine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("triton") is None,
    reason="generated key products require CUDA and Triton",
)


def _prefix_chain() -> fh.CkksConfig:
    """Drop Q primes from the end, so depth > 0 rows precede unused key rows."""
    suffix = fh.CkksConfig.catalog_suffix_chain(
        logN=12, scale_bits=40, depth=4, p_count=2, enforce_security_budget=False
    ).dumps()
    suffix["q_moduli"] = list(reversed(suffix["q_moduli"]))
    suffix["q_bases"] = [list(range(len(basis))) for basis in suffix["q_bases"]]
    return fh.CkksConfig.parse(suffix)


def test_compiled_rotation_matches_eager_after_rescale() -> None:
    config = _prefix_chain()
    device = torch.device("cuda")
    engine = Engine(config, rng_seed=3, rng_nonce=0)
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    key = engine.create_rotation_key(1, secret_key, device=device)
    message = torch.linspace(-0.5, 0.5, config.num_slots, dtype=torch.float64)

    def rotate(value):
        return engine.rotate_with_key(value, key)

    compiled = fc.compile(rotate, graph=False)
    for depth in (0, 2):
        value = engine.encrypt_message(message, public_key, depth=depth, device=device)
        assert torch.equal(compiled(value).data, rotate(value).data)
