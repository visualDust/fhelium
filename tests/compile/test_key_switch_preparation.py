"""Prepare whole and lowered key switching from mathematical expressions."""

import pytest
import torch

from fhelium import Preset, compile as fc
from fhelium.backend import OperationBackend
from fhelium.backend.ckks.materialization import CkksDeviceResources
from fhelium.eager import Engine
from fhelium.ir.dialects import ckks


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_expression_rotation_prepares_whole_and_lowered_routes(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=163)
    config = engine.config
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    key = engine.create_rotation_key(1, secret, device=device)
    message = torch.linspace(
        -0.02, 0.03, config.num_slots, dtype=torch.float64, device=device
    )
    encrypted = engine.encrypt_message(message, public, device=device)

    def expression(x):
        return (torch.roll(x, 1, dims=-1) + x) * x + x * x

    backend = OperationBackend()
    resources = CkksDeviceResources(config=config, device=device)
    for lowered in (False, True):
        captured = fc.capture_eager(
            expression,
            arguments={"x": encrypted},
            workspace=fc.CompileWorkspace({type(config): config}),
        )
        recipe = fc.default_lower_and_fuse_pipeline(backend)
        index = next(
            i
            for i, p in enumerate(recipe.passes)
            if p.name == "select-execution-lowerings"
        )
        prefix = recipe.passes[:index]
        route = (
            (fc.LowerCkksToRnsNttPass(),)
            if lowered
            else (
                fc.AssignImplementationsPass(
                    {ckks.RotateOp.name: "native-rotate-streaming"}
                ),
                recipe.passes[index],
            )
        )
        compilation = fc.Pipeline(
            (*prefix, *route, *recipe.passes[index + 1 :])
        ).run(captured)
        # Missing keys stay unresolved rather than triggering automatic creation.
        missing = fc.prepare_material_bindings(compilation, resources=resources)
        assert missing
        assert all(
            compilation.program.material_descriptions[name]["kind"]
            == "RotationKey"
            for name in missing
        )
        assert (
            fc.prepare_material_bindings(
                compilation, resources=resources, keys=(key,)
            )
            == ()
        )
        executable = backend.link(compilation)
        for _ in range(2):
            result = executable.run(encrypted)
            mixed = engine.add(
                engine.rotate_with_key(encrypted, key), encrypted
            )
            x_ntt = engine.coefficient_domain_to_ntt_domain(encrypted)
            mixed_ntt = engine.coefficient_domain_to_ntt_domain(mixed)
            reference = engine.add(
                engine.multiply(mixed_ntt, x_ntt), engine.multiply(x_ntt, x_ntt)
            )
            moduli = torch.tensor(
                config.q_moduli, dtype=torch.int64, device=device
            ).view(-1, 1)
            assert torch.equal(result.data % moduli, reference.data % moduli)
            assert (result.depth, result.scale, result.prime_ids) == (
                reference.depth,
                reference.scale,
                reference.prime_ids,
            )
            encrypted = engine.negate(encrypted)
            key.data.copy_(
                engine.create_rotation_key(1, secret, device=device).data
            )
        original = dict(compilation.material_bindings)
        assert (
            fc.prepare_material_bindings(
                compilation, resources=resources, keys=(key,)
            )
            == ()
        )
        assert all(
            compilation.material_bindings[symbol] is tensor
            for symbol, tensor in original.items()
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_frontends_share_one_pipeline_and_request_local_materials(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=191)
    config = engine.config
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    keys = [
        engine.create_rotation_key(i, secret, device=device) for i in (1, 2, 3)
    ]
    message = torch.linspace(
        -0.02, 0.03, config.num_slots, device=device, dtype=torch.float64
    )
    x = engine.encrypt_message(message, public, device=device)
    y = engine.add(x, x)

    def methods(x, y):
        outputs = []
        for key in keys:
            left = engine.negate(engine.rotate_with_key(x, key))
            right = engine.negate(engine.rotate_with_key(y, key))
            outputs.append(engine.add(left, right))
        return tuple(outputs)

    def operators(x, y):
        outputs = []
        for step in (1, 2, 3):
            left = -torch.roll(x, step, dims=-1)
            right = -torch.roll(y, step, dims=-1)
            outputs.append(left + right)
        return tuple(outputs)

    backend = OperationBackend()
    recipe = fc.default_lower_and_fuse_pipeline(
        backend,
        resources=CkksDeviceResources(config=config, device=device),
        keys=keys,
    )
    # The same Pipeline instance can prepare unrelated callable specializations.
    method_call = fc.compile(
        methods,
        backend=backend,
        pipeline=recipe,
        workspace=fc.CompileWorkspace({type(config): config}),
    )
    operator_call = fc.compile(
        operators,
        backend=backend,
        pipeline=recipe,
        workspace=fc.CompileWorkspace({type(config): config}),
    )
    for _ in range(2):
        actual = operator_call(x, y)
        expected = method_call(x, y)
        moduli = torch.tensor(
            config.q_moduli, dtype=torch.int64, device=device
        ).view(-1, 1)
        for a, b in zip(actual, expected, strict=True):
            assert torch.equal(a.data % moduli, b.data % moduli)
        x = engine.negate(x)
    left = method_call.specializations[0].compilation
    right = operator_call.specializations[0].compilation
    assert left.material_bindings is not right.material_bindings
    for compilation in (left, right):
        assert (
            sum(
                isinstance(op, ckks.RotateManyOp)
                for op in compilation.program.walk()
            )
            == 2
        )
        assert not any(
            isinstance(op, ckks.RotateOp) for op in compilation.program.walk()
        )
        assert compilation.material_bindings
