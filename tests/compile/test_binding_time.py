"""Plaintext preparation replays current weights while retaining ciphertext code."""

import torch

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.eager import Engine
from fhelium.ir import Program
from fhelium.values import Plaintext


def _fixture():
    preset = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=tuple((prime,) for prime in preset.q_moduli[:3]),
        p_moduli=preset.p_moduli[:1],
        default_scale=preset.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=761)
    secret = engine.create_secret_key(device="cpu")
    public = engine.create_public_key(secret, device="cpu")
    clear = torch.linspace(-0.02, 0.03, config.num_slots, dtype=torch.float64)
    value = engine.encrypt_message(clear, public, device="cpu")
    return engine, secret, clear, value


def test_plaintext_recipe_rebinds_weights_and_roundtrips(tmp_path):
    engine, secret, clear, value = _fixture()

    def source(current, weight):
        transformed = engine.coefficient_domain_to_ntt_domain(current)
        parts = []
        for _ in range(2):
            packed = torch.zeros(
                weight.numel(), dtype=weight.dtype, device=weight.device
            )
            packed[:] = weight.unsqueeze(0).repeat(1, 1).reshape(-1)
            plain = engine.prepare_compressed_plaintext(
                packed,
                depth=current.depth,
                scale=current.scale,
                device=current.device,
            )
            parts.append(engine.multiply_plaintext(transformed, plain))
        return engine.ntt_domain_to_coefficient_domain(
            engine.add(parts[0], parts[1])
        )

    compiled = fc.compile(source, cache=tmp_path, graph=False)
    weights = torch.tensor([0.125, 0.25, 0.375, 0.5], dtype=torch.float64)
    first = compiled.prepare(value, weights)
    plan = first.compilation.workspace["plaintext_preparation"]
    roundtrip = Program.parse(plan.program.to_text())
    assert roundtrip.to_text() == plan.program.to_text()
    dispatch = first.executable.dispatch_table
    code = first.executable._call.__code__
    weights.add_(0.125)
    for weight in (weights, weights + 0.125):
        result = compiled(value, weight)
        actual = engine.decrypt_message(result, secret)
        expected = 2 * clear * weight.repeat(clear.numel() // weight.numel())
        torch.testing.assert_close(actual.real, expected, rtol=0, atol=1e-8)
        current = compiled.specializations[0]
        assert current.executable.dispatch_table is dispatch
        assert current.executable._call.__code__ is code
    weights.add_(0.25)
    actual = engine.decrypt_message(compiled(value, weights), secret)
    torch.testing.assert_close(
        actual.real,
        2 * clear * weights.repeat(clear.numel() // weights.numel()),
        rtol=0,
        atol=1e-8,
    )
    stats = compiled.stats()["specializations"][0]["plaintext_preparation"]
    assert stats["binding_generation"] == 4
    assert (
        stats["derived_material_bytes_before_merge"]
        > stats["derived_material_bytes"]
    )
    assert stats["material_rebind_seconds"] > 0
    refreshed = compiled.refresh_materials(value, weights)
    assert refreshed.executable.dispatch_table is dispatch
    assert (
        refreshed.compilation.workspace["plaintext_preparation"].generation == 5
    )
    independent = fc.compile(
        source,
        cache=False,
        graph=False,
        workspace=fc.CompileWorkspace({"merge_plaintext_encodings": False}),
    )
    independent.prepare(value, weights)
    independent_stats = independent.stats()["specializations"][0][
        "plaintext_preparation"
    ]
    assert (
        independent_stats["derived_material_bytes_before_merge"]
        == independent_stats["derived_material_bytes"]
    )
    # The existing source-eligibility policy deliberately excludes local writes.
    assert compiled.stats()["specializations"][0]["cache"] == "ineligible"
    # Such sources are found by their captured Program, and new weights still bind.
    assert compiled.stats()["specializations"][0]["trace"] == "stored"
    traced = fc.compile(source, cache=tmp_path, graph=False)
    changed = weights - 0.5
    actual = engine.decrypt_message(traced(value, changed), secret)
    torch.testing.assert_close(
        actual.real,
        2 * clear * changed.repeat(clear.numel() // changed.numel()),
        rtol=0,
        atol=1e-8,
    )
    assert traced.stats()["specializations"][0]["trace"] == "hit"

    def cacheable(current, weight):
        plain = engine.prepare_compressed_plaintext(
            weight,
            depth=current.depth,
            scale=current.scale,
            device=current.device,
        )
        transformed = engine.coefficient_domain_to_ntt_domain(current)
        return engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(transformed, plain)
        )

    for expected_cache in ("stored", "hit"):
        restored = fc.compile(cacheable, cache=tmp_path, graph=False)
        actual = engine.decrypt_message(restored(value, weights), secret)
        torch.testing.assert_close(
            actual.real,
            clear * weights.repeat(clear.numel() // weights.numel()),
            rtol=0,
            atol=1e-8,
        )
        assert restored.stats()["specializations"][0]["cache"] == expected_cache

    def selected_views(current, weight):
        plain = engine.prepare_compressed_plaintext(
            weight.reshape(2, -1),
            depth=current.depth,
            scale=current.scale,
            device=current.device,
        )
        transformed = engine.coefficient_domain_to_ntt_domain(current)
        first = engine.multiply_plaintext(transformed, plain.select_batch(0))
        second = engine.multiply_plaintext(transformed, plain.select_batch(1))
        return engine.ntt_domain_to_coefficient_domain(
            engine.add(first, second)
        )

    for expected_cache in ("stored", "hit"):
        selected = fc.compile(selected_views, cache=tmp_path, graph=False)
        selected.prepare(value, weights)
        prepared = selected.specializations[0].compilation.workspace[
            "plaintext_preparation"
        ]
        assert len(prepared.exports) == 1 and len(prepared.views) == 2
        for weight in (weights, weights + 0.125):
            actual = engine.decrypt_message(selected(value, weight), secret)
            expected = clear * weight.reshape(2, -1).sum(0).repeat(
                clear.numel() // 2
            )
            torch.testing.assert_close(actual.real, expected, rtol=0, atol=1e-8)
            assert (
                len(
                    {
                        t.untyped_storage()._cdata
                        for t in prepared.bindings.values()
                    }
                )
                == 1
            )
        assert selected.stats()["specializations"][0]["cache"] == expected_cache

    peer = compiled.with_backend(compiled.specializations[0].backend)
    changed = weights + 0.125
    for wrapper in (peer, compiled):
        actual = engine.decrypt_message(wrapper(value, changed), secret)
        torch.testing.assert_close(
            actual.real,
            2 * clear * changed.repeat(clear.numel() // changed.numel()),
            rtol=0,
            atol=1e-8,
        )
    assert (
        peer.specializations[0].compilation.workspace["plaintext_preparation"]
        is not compiled.specializations[0].compilation.workspace[
            "plaintext_preparation"
        ]
    )

    bound_weight = weights.clone()

    def bound(current):
        return cacheable(current, bound_weight)

    initial = fc.compile(bound, cache=tmp_path, graph=False)
    initial.prepare(value)
    loaded = fc.compile(bound, cache=tmp_path, graph=False)
    loaded.prepare(value)
    assert loaded.stats()["specializations"][0]["cache"] == "hit"
    bound_weight = bound_weight + 0.25
    actual = engine.decrypt_message(loaded(value), secret)
    torch.testing.assert_close(
        actual.real,
        clear * bound_weight.repeat(clear.numel() // bound_weight.numel()),
        rtol=0,
        atol=1e-8,
    )


def test_preparation_view_keeps_layout_across_later_in_place_change():
    engine, secret, _, value = _fixture()

    def source(current, weight):
        grid = weight.reshape(2, 2).clone()
        row = grid[0]
        grid.transpose_(0, 1)
        plain = engine.prepare_compressed_plaintext(
            row.repeat(weight.numel() // 2),
            depth=current.depth,
            scale=current.scale,
            device=current.device,
        )
        return engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(current), plain
            )
        )

    compiled = fc.compile(source, cache=False, graph=False)
    weights = torch.tensor([0.125, 0.25, 0.375, 0.5], dtype=torch.float64)
    compiled.prepare(value, weights)
    for weight in (weights, weights + 0.5):
        torch.testing.assert_close(
            engine.decrypt_message(compiled(value, weight), secret).real,
            engine.decrypt_message(source(value, weight), secret).real,
            rtol=0,
            atol=1e-12,
        )


def test_plaintext_memoization_and_per_call_material_lifetime():
    engine, secret, clear, value = _fixture()
    memo = {}

    def source(current, weight):
        key = (engine, current.dtype, current.device, current.prime_ids)
        if key not in memo:
            memo[key] = torch.tensor(
                [engine.config.moduli[i] for i in current.prime_ids],
                dtype=current.dtype,
                device=current.device,
            )
        current = current.with_data(
            torch.remainder(current.data, memo[key].view(-1, 1))
        )
        message = Plaintext(
            message=weight, depth=current.depth, scale=current.scale
        )
        plain = engine.prepare_plaintext_for_addition(message)
        return engine.add_plaintext(current, plain)

    compiled = fc.compile(
        source,
        cache=False,
        graph=False,
        workspace=fc.CompileWorkspace(
            {
                "plaintext_preparation_mode": "per_call",
                "merge_plaintext_encodings": False,
            }
        ),
    )
    for offset in (0.125, 0.25):
        weight = torch.full_like(clear, offset)
        actual = engine.decrypt_message(compiled(value, weight), secret)
        torch.testing.assert_close(
            actual.real, clear + weight, rtol=0, atol=1e-8
        )
        assert (
            compiled.stats()["specializations"][0]["plaintext_preparation"][
                "resident_material_bytes"
            ]
            == 0
        )
    assert len(memo) == 1
    assert next(iter(memo))[0] is engine


def test_preparation_and_host_rebinding_release_payloads_without_gc(
    monkeypatch,
):
    import gc
    import weakref
    from types import SimpleNamespace

    from fhelium.backend.execution import ProgramExecutable
    from fhelium.compile._binding_time import PreparationPlan
    from fhelium.compile.frontend._preparation_capture import _Effects

    engine, _, _, value = _fixture()
    intermediates = []
    derived = []
    record = _Effects.record
    initialize = PreparationPlan.initialize

    def observe(self, tensor, digest, virtual, **kwargs):
        if virtual[0] == "aten":
            intermediates.append(weakref.ref(tensor))
        return record(self, tensor, digest, virtual, **kwargs)

    def initialized(self, arguments, bindings):
        derived.extend(weakref.ref(tensor) for tensor in bindings.values())
        return initialize(self, arguments, bindings)

    monkeypatch.setattr(_Effects, "record", observe)
    monkeypatch.setattr(PreparationPlan, "initialize", initialized)

    def source(current, weight):
        packed = torch.sin(weight * 0.5)
        plain = engine.prepare_compressed_plaintext(
            packed,
            depth=current.depth,
            scale=current.scale,
            device=current.device,
        )
        return engine.multiply_plaintext(
            engine.coefficient_domain_to_ntt_domain(current), plain
        )

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        compiled = fc.compile(
            source,
            graph=False,
            cache=False,
            workspace=fc.CompileWorkspace(
                {"plaintext_preparation_mode": "per_call"}
            ),
        )
        weights = torch.tensor([0.125, 0.25, 0.375, 0.5], dtype=torch.float64)
        result = compiled(value, weights)
        assert result.depth == value.depth
        assert intermediates and all(
            reference() is None for reference in intermediates
        )
        assert derived and all(reference() is None for reference in derived)

        class Host:
            pass

        class Material:
            pass

        operation = Material()
        operation.symbol = SimpleNamespace(data="weight")
        original, replacement = torch.ones(4), torch.zeros(4)
        first, second = weakref.ref(original), weakref.ref(replacement)
        namespace = {"material0": original}
        exec(
            "def __fhelium_codegen_execute():\n    return material0\n",
            namespace,
        )
        host = Host()
        host._call = namespace.pop("__fhelium_codegen_execute")
        host.bound_materials = {operation: original}
        host._material_globals = {"material0": "weight"}
        rebound = ProgramExecutable.rebind_materials(
            host, {"weight": replacement}
        )
        del host, original, namespace
        assert first() is None
        assert rebound._call() is replacement
        released = ProgramExecutable.rebind_materials(rebound, {"weight": None})
        del rebound, replacement
        assert second() is None
        assert released._call() is None
    finally:
        if was_enabled:
            gc.enable()
