"""Key switches with equal structure share one lowered function body."""

import pytest
import torch
from xdsl.dialects.func import CallOp

from fhelium import Preset, compile as fc
from fhelium.eager import Engine


def test_rotations_with_distinct_steps_call_one_body():
    # Lowering is selected where a fusion implementation covers key switching.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda:0"
    engine = Engine(Preset.slots8192_scale40_depth7, rng_seed=211)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    keys = [
        engine.create_rotation_key(step, secret, device=device)
        for step in (1, -3, 7)
    ]
    message = torch.linspace(
        -0.02, 0.03, engine.config.num_slots, device=device, dtype=torch.float64
    )
    x = engine.encrypt_message(message, public, device=device)

    def source(x):
        for key in keys:
            x = engine.negate(engine.rotate_with_key(x, key))
        return x

    compiled = fc.compile(source, cache=False, graph=False)
    moduli = torch.tensor(
        engine.config.q_moduli, dtype=torch.int64, device=device
    ).view(-1, 1)
    actual = compiled(x)
    expected = source(x)
    assert torch.equal(actual.data % moduli, expected.data % moduli)
    torch.testing.assert_close(
        engine.decrypt_message(actual, secret, is_real=True),
        torch.roll(-message, 1 - 3 + 7),
        rtol=0,
        atol=1e-5,
    )
    program = compiled.specializations[0].compilation.program
    calls = [op for op in program.single_block().ops if isinstance(op, CallOp)]
    assert len(program.functions) == 2
    assert len(calls) >= 2
    assert len({call.callee for call in calls}) == 1
