"""Prepare whole and lowered key switching from mathematical expressions."""

import pytest
import torch

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.backend import OperationBackend, OperationImplementationRegistry
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
        # Supply the Tensor data for the declared key and table inputs.
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
    recipe = fc.default_lower_and_fuse_pipeline(backend)
    resources = CkksDeviceResources(config=config, device=device)
    # The same Pipeline instance can prepare unrelated callable specializations.
    method_call = fc.compile(
        methods,
        backend=backend,
        resources=resources,
        keys=keys,
        pipeline=recipe,
        workspace=fc.CompileWorkspace({type(config): config}),
    )
    operator_call = fc.compile(
        operators,
        backend=backend,
        resources=resources,
        keys=keys,
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("route", ["whole", "lowered", "selected"])
@pytest.mark.parametrize("custom_layout", [False, True])
def test_material_free_program_rebinds_independent_keys(
    tmp_path, route, custom_layout, device
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import ir
    from fhelium.serialization import load_compilation, save_compilation
    from fhelium.values import EvaluationKeySet

    preset = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=tuple((q,) for q in preset.q_moduli[:4]),
        p_moduli=preset.p_moduli[:1],
        default_scale=preset.default_scale,
        enforce_security_budget=False,
    )

    def expression(x):
        mixed = x + torch.roll(x, 3, dims=-1)
        return mixed * mixed

    backend = OperationBackend()
    if route == "selected":
        backend = OperationBackend(
            OperationImplementationRegistry(
                tuple(
                    implementation
                    for implementation in backend.registry.implementations
                    if not any(
                        operation.name.startswith("fhelium_ckks.")
                        for operation in implementation.operation_types
                    )
                )
            )
        )
    captured = fc.capture(
        expression,
        inputs={
            "x": fc.encrypted(
                slots=16,
                polynomial_domain="coefficient",
                residue_representation="standard",
            )
        },
        workspace=fc.CompileWorkspace(
            {
                CkksConfig: config,
                OperationImplementationRegistry: backend.registry,
            }
        ),
    )
    compilation = fc.Pipeline(
        (
            fc.LowerSemanticToLogicalPass(),
            fc.InsertMultiplyNttTransitionsPass(),
            fc.LowerLogicalToCkksPass(),
            fc.ResolveRotationKeyOperandsPass(),
            fc.InsertRelinearizationPass(),
            fc.InsertRescalePass(),
            fc.AssignCkksDepthsPass(entry_depth=1),
            fc.AssignCkksScalesPass(entry_scale=config.default_scale),
        )
    ).run(captured)
    layouts = {
        symbol: {
            "prime_ids": (1, 2, 3, 0, 4),
            "digit_prime_ids": ((1,), (2, 3)),
        }
        for symbol, description in compilation.program.material_descriptions.items()
        if custom_layout
        and description.get("kind") in {"RotationKey", "RelinearizationKey"}
    }
    compilation.workspace["key_layouts"] = layouts
    if route == "selected":
        del compilation.workspace[OperationImplementationRegistry]
    lowering = (
        fc.SelectExecutionLoweringsPass(backend.registry)
        if route == "selected"
        else fc.LowerCkksToRnsNttPass(
            preserve=frozenset()
            if route == "lowered"
            else frozenset({ckks.RotateOp.name, ckks.RelinearizeOp.name})
        )
    )
    compilation = fc.Pipeline(
        (
            fc.AssignNttImplementationPass(
                "native-ntt", ntt_backend="radix2_indexed"
            ),
            lowering,
            fc.SelectNttImplementationsPass(backend.registry),
        )
    ).run(compilation)
    assert compilation.material_bindings == {}
    requirements = ir.analyze_evaluation_key_requirements(compilation.program)
    assert requirements.rotation_steps == frozenset({3})
    assert requirements.requires_relinearization
    path = tmp_path / "unbound.safetensors"
    save_compilation(compilation, path)
    original = path.read_bytes()

    clear = torch.linspace(
        -0.03, 0.04, 16, dtype=torch.float64, device=device
    ).repeat(config.num_slots // 16)
    expected = (clear + torch.roll(clear, 3, dims=-1)).square()
    executions = []
    for seed, named in ((301, True), (302, False)):
        # No Engine, keys, parameters or NTT tables existed during compilation.
        engine = Engine(config, rng_seed=seed, rng_nonce=0)
        secret = engine.create_secret_key(device=device)
        public = engine.create_public_key(secret, device=device)
        options = (
            {
                "source_depth": 1,
                "p_prime_ids": (0, 4),
                "digit_prime_ids": ((1,), (2, 3)),
            }
            if custom_layout
            else {}
        )
        rotation = engine.create_rotation_key(
            3, secret, device=device, **options
        )
        relin = engine.create_relinearization_key(
            secret, device=device, **options
        )
        bound = load_compilation(path)
        assert bound.material_bindings == {}
        keys = (
            {
                symbol: rotation
                if description.get("kind") == "RotationKey"
                else relin
                for symbol, description in bound.program.material_descriptions.items()
                if description.get("kind")
                in {"RotationKey", "RelinearizationKey"}
            }
            if named
            else EvaluationKeySet((rotation, relin))
        )
        assert (
            fc.prepare_material_bindings(
                bound,
                resources=CkksDeviceResources(
                    config=config, device=device, ntt_backend="radix2_indexed"
                ),
                keys=keys,
            )
            == ()
        )
        encrypted = engine.encrypt_message(
            clear, public, depth=1, scale=config.default_scale, device=device
        )
        executable = backend.link(bound)
        result = executable.run(encrypted)
        torch.testing.assert_close(
            engine.decrypt_message(result, secret).real,
            expected,
            atol=5e-6,
            rtol=5e-6,
        )
        assert result.depth == 2
        assert result.component_count == 2
        assert result.prime_ids == (2, 3)
        assert result.scale == config.default_scale**2 / config.rescale_divisor(
            1
        )
        executions.append((executable, encrypted, result.data.clone()))
    for executable, encrypted, expected_data in executions:
        assert torch.equal(executable.run(encrypted).data, expected_data)
    assert path.read_bytes() == original
    assert compilation.material_bindings == {}

    rejected = load_compilation(path)
    symbol = next(
        symbol
        for symbol, description in rejected.program.material_descriptions.items()
        if description.get("kind") == "RotationKey"
    )
    wrong_options = {} if custom_layout else {"source_depth": 1}
    incompatible = engine.create_rotation_key(
        3, secret, device=device, **wrong_options
    )
    with pytest.raises(ValueError, match="conflicts with declared"):
        fc.prepare_material_bindings(rejected, keys={symbol: incompatible})
    assert rejected.material_bindings == {}
