"""Prepare whole and lowered key switching from mathematical expressions."""

import pytest
import torch

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.backend import OperationBackend
from fhelium.backend.ckks.materialization import CkksDeviceResources
from fhelium.eager import Engine
from fhelium.ir.dialects import ckks


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("domain", ["coefficient", "ntt"])
def test_compound_relinearization_consumes_a_complete_group(device, domain):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    preset = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=(preset.q_moduli[:2], preset.q_moduli[2:4]),
        p_moduli=preset.p_moduli,
        default_scale=preset.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=271)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    key = engine.create_relinearization_key(secret, device=device)
    encrypted = engine.encrypt_message(
        torch.linspace(-0.02, 0.03, 16, device=device), public, device=device
    )
    ntt = engine.coefficient_domain_to_ntt_domain(encrypted)
    product = engine.multiply(ntt, ntt)
    original = product.data.clone()
    reference = engine.rescale_to_next_depth(
        engine.relinearize(
            product, key, output_domain=domain, moddown_rounding="floor"
        ),
        rounding="nearest",
    )

    def compound(value):
        return engine.relinearize(
            value,
            key,
            rescale=True,
            rescale_rounding="nearest",
            moddown_rounding="floor",
            output_domain=domain,
        )

    moduli = torch.tensor(
        config.q_moduli[2:], dtype=torch.int64, device=device
    ).view(-1, 1)
    for operation in (compound, fc.compile(compound)):
        result = operation(product)
        assert torch.equal(result.data % moduli, reference.data % moduli)
        assert (result.depth, result.scale, result.prime_ids) == (
            1,
            product.scale / config.rescale_divisor(0),
            product.prime_ids[2:],
        )
        assert result.component_count == 2
        assert result.polynomial_domain == reference.polynomial_domain
        assert result.residue_representation == reference.residue_representation
    assert torch.equal(product.data, original)


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
        compilation = fc.Pipeline(prefix).run(captured)
        # Unknown keys retain unknown digit supports until actual values are supplied.
        missing = fc.prepare_material_bindings(compilation, resources=resources)
        assert missing
        assert all(
            compilation.program.material_descriptions[name]["kind"]
            == "RotationKey"
            for name in missing
        )
        assert (
            fc.prepare_material_bindings(
                compilation,
                resources=resources,
                keys={name: key for name in missing},
            )
            == ()
        )
        compilation = fc.Pipeline((*route, *recipe.passes[index + 1 :])).run(
            compilation
        )
        assert (
            fc.prepare_material_bindings(compilation, resources=resources) == ()
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("modup_method", ["exact", "crt", "normalized_crt"])
def test_lowered_ntt_switch_keeps_live_coefficient_input_and_key(
    device, modup_method
):
    from fhelium import CkksConfig

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    base = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig.parse(
        base.dumps(), logN=8, enforce_security_budget=False
    )
    engine = Engine(config, rng_seed=223)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    key = engine.create_key_switch_key(secret, secret, device=device)
    message = torch.linspace(
        -0.02, 0.03, config.num_slots, dtype=torch.float64, device=device
    )
    value = engine.encrypt_message(
        torch.stack((message, -message)), public, device=device
    )

    def source(x, key, rounding):
        return engine.switch_key(
            x,
            key,
            output_domain="ntt",
            moddown_rounding=rounding,
            modup_method=modup_method,
        )

    backend = OperationBackend()
    pipeline = fc.default_lower_and_fuse_pipeline(backend).before(
        "select-execution-lowerings", fc.LowerCkksToRnsNttPass()
    )
    compiled = fc.compile(source, backend=backend, pipeline=pipeline)
    for rounding in ("nearest", "floor"):
        for _ in range(2):
            expected = source(value, key, rounding)
            actual = compiled(value, key, rounding)
            moduli = torch.tensor(
                config.q_moduli, dtype=actual.data.dtype, device=device
            )[:, None]
            assert torch.equal(actual.data % moduli, expected.data % moduli)
            assert (
                actual.depth,
                actual.scale,
                actual.prime_ids,
                actual.polynomial_domain,
                actual.residue_representation,
            ) == (
                expected.depth,
                expected.scale,
                expected.prime_ids,
                expected.polynomial_domain,
                expected.residue_representation,
            )
            value = engine.negate(value)
            key.data.copy_(
                engine.create_key_switch_key(secret, secret, device=device).data
            )
