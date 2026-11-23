"""Captured mathematical templates specialize current values and materials."""

import json
from unittest.mock import patch

import pytest
import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.backend import OperationBackend, OperationImplementationRegistry
from fhelium.eager import Engine
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core
from fhelium.compile._template import StructuralTemplate, build_template


@pytest.fixture(scope="module")
def fixture():
    preset = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=(
            preset.q_moduli[:2],
            (preset.q_moduli[2],),
            (preset.q_moduli[3],),
        ),
        p_moduli=preset.p_moduli[:1],
        default_scale=preset.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=349)
    secret = engine.create_secret_key(device="cpu")
    public = engine.create_public_key(secret, device="cpu")
    clear = torch.linspace(-0.02, 0.03, config.num_slots, dtype=torch.float64)
    value = engine.encrypt_message(clear, public, device="cpu")
    keys = [
        engine.create_rotation_key(step, secret, device="cpu")
        for step in (1, 2)
    ]
    for key in keys:
        engine.set_rotation_key(key)
    return engine, value, keys


def capture(function, arguments, backend=None):
    workspace = fc.CompileWorkspace({"structural_template_recording": True})
    if backend is not None:
        workspace[OperationImplementationRegistry] = backend.registry
    source = fc.capture_eager(
        function, arguments=arguments, workspace=workspace
    )
    template = build_template(source)
    assert template is not None
    # The same representation is usable without the original Python function.
    restored = StructuralTemplate(
        Program.parse(template.program.to_text()),
        json.loads(json.dumps(template.metadata)),
    )
    return source, restored


def execute(compilation, engine, backend, *inputs):
    optimized = fc.default_lower_and_fuse_pipeline(backend).run(compilation)
    resources = engine._dispatcher_for(torch.device("cpu")).device_resources
    assert fc.prepare_material_bindings(optimized, resources=resources) == ()
    return backend.link(optimized).run(*inputs)


def assert_same(actual, expected, config):
    moduli = torch.tensor(
        [config.moduli[i] for i in expected.prime_ids], dtype=torch.int64
    ).view(-1, 1)
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


def test_weighted_product_reuses_structure_across_depth_and_actual_scale(
    fixture,
):
    engine, original, _ = fixture
    backend = OperationBackend()

    def weighted(lhs, rhs, weight, negative=False):
        a = engine.coefficient_domain_to_ntt_domain(lhs)
        b = engine.coefficient_domain_to_ntt_domain(rhs)
        product = engine.multiply(a, b)
        weighted = engine.multiply_plaintext(product, weight)
        doubled = engine.subtract(weighted, engine.negate(weighted))
        if negative:
            doubled = engine.negate(doubled)
        return engine.ntt_domain_to_coefficient_domain(doubled)

    def arguments(depth, factor):
        value = engine.reinterpret_at_scale(
            engine.mod_switch_to_depth(original, depth), original.scale * factor
        )
        weight = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                torch.full((engine.num_slots,), 0.5), depth=depth, device="cpu"
            )
        )
        return dict(lhs=value, rhs=value, weight=weight, negative=True)

    source, template = capture(weighted, arguments(0, 1), backend)
    original_text = source.program.to_text()
    for depth, factor in ((0, 1.25), (1, 1.5)):
        current = arguments(depth, factor)
        with patch(
            "fhelium.compile.frontend._eager_capture._EngineCalls._emit",
            side_effect=AssertionError("adapter replay"),
        ):
            instance = template.instantiate(current, workspace=source.workspace)
        assert instance is not None
        result = execute(
            instance,
            engine,
            backend,
            current["lhs"],
            current["rhs"],
            current["weight"],
        )
        assert_same(result, weighted(**current), engine.config)
    assert source.program.to_text() == original_text
    assert template.instantiate({**arguments(0, 1), "negative": False}) is None


@pytest.mark.parametrize("with_keys", (False, True))
def test_matrix_materials_follow_current_depth_and_fixed_key_layout(
    fixture, with_keys
):
    engine, original, keys = fixture
    backend = OperationBackend()
    diagonals = []

    def set_diagonals(depth):
        diagonals[:] = [
            engine.prepare_compressed_plaintext(
                torch.tensor([0.1 + i / 20, 0.2]), depth=depth, device="cpu"
            )
            for i in range(4)
        ]

    def matrix(value):
        rotated = (
            engine.rotate_many_with_keys(value, keys[:1], use_hoisting=True)
            if with_keys
            else engine.rotate_many_by_steps(value, (1,), use_hoisting=True)
        )
        babies = [
            engine.coefficient_domain_to_ntt_domain(value),
            engine.coefficient_domain_to_ntt_domain(rotated[0]),
        ]
        groups = []
        for group in range(2):
            partial = engine.multiply_plaintext(babies[0], diagonals[2 * group])
            partial = engine.add(
                partial,
                engine.multiply_plaintext(babies[1], diagonals[2 * group + 1]),
            )
            partial = engine.ntt_domain_to_coefficient_domain(partial)
            if group:
                partial = engine.rotate_with_key(partial, keys[1])
            groups.append(partial)
        return engine.rescale_to_next_depth(engine.add(groups[0], groups[1]))

    set_diagonals(0)
    source, template = capture(matrix, {"value": original}, backend)
    rotation = next(
        op for op in source.program.walk() if isinstance(op, ckks.RotateManyOp)
    )
    for result in rotation.results:
        state = result.type.state.data
        assert tuple(item.value.data for item in state["shape"]) == tuple(
            original.data.shape
        )
        assert state["requires_grad"].value.data == 0
    data_owners = {
        id(value.data): ("diagonal", i) for i, value in enumerate(diagonals)
    }
    data_owners.update({id(key.data): ("key", i) for i, key in enumerate(keys)})
    locations = {
        symbol: data_owners[id(tensor)]
        for symbol, tensor in source.material_bindings.items()
        if id(tensor) in data_owners
    }
    for depth in (0, 1):
        set_diagonals(depth)
        current = engine.mod_switch_to_depth(original, depth)
        leaves = {
            symbol: (diagonals[index] if kind == "diagonal" else keys[index])
            for symbol, (kind, index) in locations.items()
        }
        with patch(
            "fhelium.compile.frontend._eager_capture._EngineCalls._emit",
            side_effect=AssertionError("adapter replay"),
        ):
            instance = template.instantiate(
                {"value": current},
                material_bindings={
                    symbol: value.data for symbol, value in leaves.items()
                },
                leaf_states=leaves,
                workspace=source.workspace,
            )
        assert instance is not None
        result = execute(instance, engine, backend, current)
        assert_same(result, matrix(current), engine.config)
    symbol = next(
        symbol for symbol, (kind, _) in locations.items() if kind == "key"
    )
    bad_type = next(
        op.value.type
        for op in template.program.walk()
        if isinstance(op, core.MaterialRefOp) and op.symbol.data == symbol
    ).with_state({"prime_ids": ArrayAttr((IntegerAttr(99, 64),))})
    assert (
        template.instantiate(
            {"value": current},
            material_bindings={
                symbol: value.data for symbol, value in leaves.items()
            },
            leaf_states={**leaves, symbol: bad_type},
            workspace=source.workspace,
        )
        is None
    )


def test_metadata_guards_and_unsupported_tensor_calls_preserve_fallback(
    fixture,
):
    engine, original, _ = fixture

    def guarded(value):
        result = engine.negate(value)
        for _ in range(result.depth + 1):
            result = engine.negate(result)
        return result

    _, template = capture(guarded, {"value": original})
    assert template.metadata["guards"]
    assert (
        template.instantiate({"value": engine.mod_switch_to_depth(original, 1)})
        is None
    )

    def scale_guard(value):
        result = engine.negate(value)
        return engine.negate(result) if result.scale > 1 else result

    _, scale_template = capture(scale_guard, {"value": original})
    assert (
        scale_template.instantiate(
            {
                "value": engine.reinterpret_at_scale(
                    original, original.scale * 1.25
                )
            }
        )
        is None
    )
    unsupported = fc.capture_eager(
        lambda x: x.replace(scale=x.scale),
        arguments={"x": original},
        workspace=fc.CompileWorkspace({"structural_template_recording": True}),
    )
    assert build_template(unsupported) is None


def test_compressed_output_descriptor_uses_current_leaf_state(fixture):
    engine, _, _ = fixture
    first = engine.prepare_compressed_plaintext(
        torch.tensor([0.1, 0.2]),
        depth=0,
        polynomial_domain="coefficient",
        device="cpu",
    )
    current = engine.prepare_compressed_plaintext(
        torch.tensor([0.2, 0.3]),
        depth=1,
        scale=first.scale * 1.5,
        polynomial_domain="coefficient",
        device="cpu",
    )
    _, template = capture(lambda value: value, {"value": first})
    instance = template.instantiate({"value": current})
    assert instance is not None
    descriptor = json.loads(
        instance.program.module.attributes["fhelium.output_structure"].data
    )
    assert descriptor["fields"]["depth"] == current.depth
    assert descriptor["fields"]["scale"] == current.scale
    assert descriptor["fields"]["prime_ids"] == list(current.prime_ids)


def test_mixed_tensor_and_semantic_rules_reuse_current_metadata(fixture):
    engine, original, _ = fixture
    backend = OperationBackend()

    def mixed(value):
        return -(value + value), torch.sin(
            value.data[..., :8].to(torch.float64)
        )

    source, template = capture(mixed, {"value": original}, backend)
    current = engine.mod_switch_to_depth(original, 1)
    instance = template.instantiate(
        {"value": current}, workspace=source.workspace
    )
    assert instance is not None
    assert (
        instance.workspace["structural_template_stage"] == "captured-semantic"
    )
    result = execute(instance, engine, backend, current)
    assert_same(
        result[0], engine.negate(engine.add(current, current)), engine.config
    )
    torch.testing.assert_close(
        result[1],
        torch.sin(current.data[..., :8].to(torch.float64)),
        rtol=0,
        atol=0,
    )
    assert result[1].shape[-1] == 8


def test_callable_uses_memory_and_disk_templates_for_new_depths(
    fixture, tmp_path
):
    from fhelium.compile import _structural_cache

    engine, original, _ = fixture

    def doubled(value):
        transformed = engine.coefficient_domain_to_ntt_domain(value)
        return engine.ntt_domain_to_coefficient_domain(
            engine.add(transformed, transformed)
        )

    compiled = fc.compile(doubled, cache=tmp_path)
    try:
        assert_same(compiled(original), doubled(original), engine.config)
        next_value = engine.mod_switch_to_depth(original, 1)
        with patch.object(
            compiled, "_capture", side_effect=AssertionError("recapture")
        ):
            assert_same(
                compiled(next_value), doubled(next_value), engine.config
            )
        record = compiled.stats()["specializations"][-1]
        assert record["structure"] == "hit"
        assert record["structure_layer"] == "memory"
        assert record["selection_plans"]["selection"]["hits"] > 0
        assert record["selection_plans"]["fusion"]["hits"] > 0
        assert_same(compiled(original), doubled(original), engine.config)
        assert compiled.stats()["resident_cache_hits"] > 0
    finally:
        compiled.clear()

    with _structural_cache._MEMORY_LOCK:
        _structural_cache._MEMORY.clear()
    reloaded = fc.compile(doubled, cache=tmp_path)
    try:
        next_value = engine.mod_switch_to_depth(original, 2)
        with patch.object(
            reloaded, "_capture", side_effect=AssertionError("recapture")
        ):
            assert_same(
                reloaded(next_value), doubled(next_value), engine.config
            )
        record = reloaded.stats()["specializations"][-1]
        assert record["structure"] == "hit"
        assert record["structure_layer"] == "disk"
    finally:
        reloaded.clear()
