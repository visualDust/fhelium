"""Behavior of persistent caching, callable statistics, and Torch/CUDA invocation context."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from fhelium import CkksConfig, Preset
from fhelium import compile as fc
from fhelium.eager import Engine


def _fixture(device):
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7)
    engine = Engine(config, rng_seed=7, rng_nonce=0)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    rotation = engine.create_rotation_key(1, secret, device=device)
    engine.set_rotation_key(rotation)
    weight = torch.linspace(0.1, 0.2, 16, dtype=torch.float64, device=device)
    plain = engine.prepare_compressed_plaintext(weight, depth=0, device=device)
    x = engine.encrypt_message(
        torch.linspace(-0.02, 0.03, 16, device=device), public, device=device
    )

    def source(value):
        rotated = engine.rotate_with_key(value, rotation)
        return engine.rescale_to_next_depth(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(
                    engine.add(value, rotated)
                ),
                plain,
            )
        )

    return engine, secret, x, source


def test_persistent_cache_hit_relinks_current_materials(tmp_path):
    engine, secret, x, source = _fixture("cpu")
    compiled = fc.compile(source, cache=tmp_path)
    expected = compiled(x)
    stored = compiled.stats()["specializations"][0]
    assert (stored["cache"], stored["artifact"]) == ("stored", "stored")
    again = fc.compile(source, cache=tmp_path)
    actual = again(x)
    assert (
        again.specializations[0].source.program.to_text()
        == compiled.specializations[0].source.program.to_text()
    )
    record = again.stats()["specializations"][0]
    assert (record["cache"], record["artifact"]) == ("hit", "hit")
    assert "capture_seconds" not in record and record["cache_load_seconds"] > 0
    torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
    assert (actual.depth, actual.scale, actual.prime_ids) == (
        expected.depth,
        expected.scale,
        expected.prime_ids,
    )
    # A different captured constant changes the source identity.
    scale = 2.0

    def scaled(value):
        return engine.multiply_constant(source(value), scale, rescale=False)

    other = fc.compile(scaled, cache=tmp_path)
    other(x)
    assert other.stats()["specializations"][0]["cache"] == "stored"


def test_cache_entry_survives_a_new_process(tmp_path):
    import os

    import fhelium

    fhelium_file = fhelium.__file__
    script = textwrap.dedent(
        f"""
        import json, os, sys
        # Import the package under test and this module ahead of any editable finder.
        sys.meta_path = [f for f in sys.meta_path if "editable" not in type(f).__module__]
        sys.path.insert(0, {os.path.dirname(os.path.dirname(fhelium_file))!r})
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from test_callable_cache import _fixture
        from fhelium import compile as fc
        engine, secret, x, source = _fixture("cpu")
        compiled = fc.compile(source, cache={str(tmp_path)!r})
        out = compiled(x)
        record = compiled.stats()["specializations"][0]
        print(json.dumps([record["cache"], record["artifact"], float(engine.decrypt_message(out, secret)[0].real)]))
        """
    )
    results = [
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        .stdout.strip()
        .splitlines()[-1]
        for _ in range(2)
    ]
    import json

    first, second = (json.loads(line) for line in results)
    assert first[:2] == ["stored", "stored"] and second[:2] == ["hit", "hit"]
    assert first[2] == pytest.approx(second[2], abs=1e-9)


def test_unpacked_arguments_keep_the_source_cacheable(tmp_path):
    import fhelium as fh

    engine, secret, x, source = _fixture("cpu")

    def stacked(value):
        babies = [engine.add(value, value)]
        pair = (value, babies[0])
        batch = fh.Ciphertext.stack_batch([engine.add(*pair), *babies])
        return engine.sum_ciphertext_batch(batch)

    expected = stacked(x)
    compiled = fc.compile(stacked, cache=tmp_path)
    compiled(x)
    assert compiled.stats()["specializations"][0]["cache"] == "stored"
    again = fc.compile(stacked, cache=tmp_path)
    actual = again(x)
    assert again.stats()["specializations"][0]["cache"] == "hit"
    torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)


def test_torch_modes_select_specializations():
    def source(value):
        return value * (2 if torch.is_grad_enabled() else 3)

    compiled = fc.compile(source, cache=False, graph=False)
    value = torch.ones(2)
    with torch.no_grad():
        torch.testing.assert_close(compiled(value), value * 3)
    torch.testing.assert_close(compiled(value), value * 2)
    assert len(compiled.specializations) == 2
    compiled.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("relink", [False, True])
def test_prepared_materials_are_ready_on_the_consumer_stream(
    monkeypatch, relink
):
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.device(0), torch.cuda.stream(stream):
        weight = torch.full((32,), 3.0, device="cuda:0")
        value = torch.ones_like(weight)

        def source(x):
            return x * weight

        compiled = fc.compile(source, cache=False, graph=False)

        def prepare_materials(prepared, arguments, stages):
            # Keep a device update pending after host preparation returns.
            torch.cuda._sleep(40_000_000)
            weight.fill_(7)

        monkeypatch.setattr(compiled, "_warm", prepare_materials)
        compiled.prepare(value)
        if relink:
            from fhelium.backend import OperationBackend

            active = compiled.with_backend(OperationBackend())
            active.prepare(value)
        else:
            active = compiled
        actual = active(value)
        compiled.clear()
        active.clear()
        torch.testing.assert_close(actual.cpu(), torch.full((32,), 7.0))


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Two CUDA devices required"
)
def test_capture_uses_ambient_cuda_device():
    def source(value):
        return value * (torch.cuda.current_device() + 1)

    compiled = fc.compile(source, cache=False, graph=False)
    with torch.cuda.device(1):
        value = torch.ones(2, device="cuda:1")
        torch.testing.assert_close(compiled(value), value * 2)
    compiled.clear()


def test_callable_keys_are_bound_into_a_stored_artifact(tmp_path):
    engine, secret, x, _ = _fixture("cpu")
    second = engine.create_rotation_key(2, secret, device="cpu")
    first = engine.create_rotation_key(1, secret, device="cpu")

    def source(value):
        return engine.rotate_by_step(engine.rotate_with_key(value, first), 2)

    expected = fc.compile(source, keys=[second], cache=False)(x)
    for status in ("stored", "hit"):
        compiled = fc.compile(source, keys=[second], cache=tmp_path)
        actual = compiled(x)
        record = compiled.stats()["specializations"][0]
        assert record["artifact"] == status, record
        torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)


class _Model:
    def __init__(self, engine, rotation, weight):
        self.engine, self.rotation = engine, rotation
        self.plain = engine.prepare_compressed_plaintext(
            weight, depth=0, device="cpu"
        )
        self.unrelated = object()

    @fc.compile
    def forward(self, value):
        return self.engine.rescale_to_next_depth(
            self.engine.multiply_plaintext(self.mixed(value), self.plain)
        )

    def __call__(self, value):
        return self.engine.multiply_plaintext(self.mixed(value), self.plain)

    def mixed(self, value):
        rotated = self.engine.rotate_with_key(value, self.rotation)
        return self.engine.coefficient_domain_to_ntt_domain(
            self.engine.add(value, rotated)
        )


def _models():
    engine, secret, x, _ = _fixture("cpu")
    rotation = engine._keys.rotation_keys[1]
    weights = (
        torch.linspace(0.1, 0.2, 16, dtype=torch.float64),
        torch.linspace(-0.3, 0.1, 16, dtype=torch.float64),
    )
    return engine, secret, x, [_Model(engine, rotation, w) for w in weights]


def test_compiled_methods_specialize_per_instance(tmp_path):
    import json
    import os

    import fhelium

    fhelium_file = fhelium.__file__
    # Two processes compile the method and the callable objects of two instances with different weights.
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.meta_path = [f for f in sys.meta_path if "editable" not in type(f).__module__]
        sys.path.insert(0, {os.path.dirname(os.path.dirname(fhelium_file))!r})
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from test_callable_cache import _Model, _models
        from fhelium import compile as fc
        engine, secret, x, models = _models()
        assert type(_Model.forward) is fc.CompiledCallable
        report = []
        for model in models:
            out = model.forward(x)
            assert model.forward is model.forward
            record = model.forward.stats()["specializations"][0]
            expected = model.forward.reference(x)
            report.append([record["cache"], "capture_seconds" in record, bool((out.data == expected.data).all())])
        for model in models:
            compiled = fc.compile(model)
            out = compiled(x)
            record = compiled.stats()["specializations"][0]
            report.append([record["cache"], "capture_seconds" in record, bool((out.data == model(x).data).all())])
        print(json.dumps(report))
        """
    )
    first, second = (
        json.loads(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "FHELIUM_CACHE_DIR": str(tmp_path),
                },
            )
            .stdout.strip()
            .splitlines()[-1]
        )
        for _ in range(2)
    )
    # Instances with equal attribute layouts share one entry and bind their own weights.
    # The second pair compiles the instances as callable objects.
    assert first == [["stored", True, True], ["hit", False, True]] * 2
    assert second == [["hit", False, True]] * 4
