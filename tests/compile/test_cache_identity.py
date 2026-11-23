"""Persistent cache identity and current-material relocation on CPU."""

from __future__ import annotations

import json
import math
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from fhelium.compile._cache import (
    CacheEntry,
    CacheIneligible,
    CompileCache,
    build_manifest,
    entry_key,
    relocate_materials,
    relocate_structural_materials,
    runtime_cacheable,
    source_identity,
    structural_entry_key,
)
from fhelium.compile._specialization import CallSignature, describe_call
from fhelium.ir import Program
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    Plaintext,
    RotationKey,
)


def _weighted(weights):
    def apply(value):
        return value * weights[0] + weights[1]

    return apply


def _identity(function, names=None):
    return source_identity(function, material_names=names)


def test_tensor_values_relocate_but_aliases_and_static_values_change_identity():
    weight = torch.ones(3)
    current = torch.full((3,), 7.0)
    first = _weighted([weight, weight])
    second = _weighted([current, current])
    old, new = _identity(first), _identity(second)
    assert old.key == new.key
    assert old.key != _identity(_weighted([current, current.clone()])).key
    assert _identity(_weighted([1, 2])).key != _identity(_weighted([1, 3])).key
    weights = [weight, weight]
    changing = _weighted(weights)
    prior = _identity(changing)
    weights[1] = weight.clone()
    assert prior.key != _identity(changing).key
    manifest = build_manifest(old, {"weight": weight}, {})
    assert manifest is not None
    rebound, providers = relocate_materials(
        second, json.loads(json.dumps(manifest))
    )
    assert rebound["weight"] is current and providers == []
    with pytest.raises(CacheIneligible, match="aliases"):
        relocate_materials(_weighted([current, current.clone()]), manifest)


def test_scalar_identity_branches_bypass_but_singletons_are_supported():
    def make(left, right):
        def source(value):
            return value + (1 if left is right else 2)

        return source

    shared = float("1.25")
    same = make(shared, shared)
    distinct = make(float("1.25"), float("1.25"))
    assert same(0) == 1 and distinct(0) == 2
    for source in (same, distinct):
        with pytest.raises(CacheIneligible, match="Identity comparisons"):
            _identity(source)

    def singleton(value, option=None):
        return value + (1 if option is None else 2)

    assert _identity(singleton).key


def test_named_material_paths_preserve_special_characters():
    def source(value):
        return value + 1

    label = 'weight]#key.[nested]"\\日本語'
    first, second = torch.ones(2), torch.zeros(2)
    identity = _identity(source, {label: first})
    manifest = build_manifest(identity, {"material": first}, {})
    assert manifest is not None
    bindings, _ = relocate_materials(
        source, json.loads(json.dumps(manifest)), material_names={label: second}
    )
    assert bindings["material"] is second


def _module_function(path, text, namespace=None):
    path.write_text(text)
    module = types.ModuleType(path.stem)
    module.__dict__.update(namespace or {})
    exec(compile(text, str(path), "exec"), module.__dict__)
    return module


def test_module_helper_code_defaults_and_current_source_file_are_dependencies(
    tmp_path,
):
    helper = _module_function(
        tmp_path / "helper.py",
        "def calculate(value, factor=2):\n    return value * factor\n",
    )
    caller = _module_function(
        tmp_path / "caller.py",
        "def source(value):\n    return helper.calculate(value)\n",
        {"helper": helper},
    )
    first = _identity(caller.source).key
    helper.calculate.__defaults__ = (3,)
    assert first != _identity(caller.source).key
    helper.calculate.__defaults__ = (2,)
    original = helper.calculate.__code__
    replacement = _module_function(
        tmp_path / "replacement.py",
        "def calculate(value, factor=2):\n    return value + factor\n",
    )
    helper.calculate.__code__ = replacement.calculate.__code__
    assert first != _identity(caller.source).key
    helper.calculate.__code__ = original
    assert first == _identity(caller.source).key
    (tmp_path / "helper.py").write_text(
        "def calculate(value, factor=2):\n    return value * factor\n# changed source\n"
    )
    assert first != _identity(caller.source).key


@pytest.mark.parametrize(
    "body,namespace",
    [
        ("return value + time.time()", {"time": __import__("time")}),
        ("return value + os.getpid()", {"os": __import__("os")}),
        ("return value + random.random()", {"random": __import__("random")}),
        ("return value + native()", {"native": __import__("time").time}),
        ("global counter\n    counter = 1\n    return value", {"counter": 0}),
        (
            "return value + object_.scale",
            {"object_": types.SimpleNamespace(scale=2)},
        ),
        ("return value + getattr(math, 'pi')", {"math": math}),
        ("return value + torch.rand(())", {"torch": torch}),
    ],
)
def test_external_state_dynamic_objects_and_mutations_bypass(
    tmp_path, body, namespace
):
    module = _module_function(
        tmp_path / "source.py",
        "def source(value):\n    " + body + "\n",
        namespace,
    )
    with pytest.raises(CacheIneligible):
        _identity(module.source)


def test_cycles_and_tensor_subclasses_bypass():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(CacheIneligible, match="cycles"):
        _identity(_weighted(cyclic))

    class SpecialTensor(torch.Tensor):
        pass

    special = torch.ones(2).as_subclass(SpecialTensor)
    with pytest.raises(CacheIneligible):
        _identity(_weighted([special, 1]))
    assert runtime_cacheable({"value": torch.ones(2), "flag": True})
    assert not runtime_cacheable({"value": special})
    assert not runtime_cacheable({"value": types.SimpleNamespace()})


def test_supported_torch_math_and_context_identity():
    def source(value):
        return torch.sin(value) * math.sqrt(2.0)

    identity = _identity(source)
    signature = CallSignature(())
    assert entry_key(identity, signature) != entry_key(
        identity, replace(signature, context=(False,))
    )


def test_engine_loop_with_closure_plaintexts_and_global_keys(tmp_path):
    from fhelium import CkksConfig, Preset
    from fhelium.eager import Engine

    engine = Engine(CkksConfig.parse(Preset.slots8192_scale40_depth7_int64))
    key = RotationKey(
        torch.zeros((1, 2, 2, 8), dtype=torch.int64),
        (0, 1),
        ((0,),),
        rotation_step=1,
    )
    plain = CompressedPlaintext(
        torch.zeros((2, 2), dtype=torch.int64),
        8,
        "cyclic",
        0,
        16.0,
        "ntt",
        "Q",
        "montgomery",
        (0, 1),
    )
    code = '''def make_function(diagonals):
    def matrix_multiply(value):
        rotated = engine.rotate_many_by_steps(value, baby_steps, use_hoisting=True)
        babies = [engine.coefficient_domain_to_ntt_domain(value)]
        for baby in rotated:
            babies.append(engine.coefficient_domain_to_ntt_domain(baby))
        groups = []
        for group in range(size // baby_step):
            partial = engine.multiply_plaintext(babies[0], diagonals[group * baby_step])
            for baby in range(1, baby_step):
                term = engine.multiply_plaintext(babies[baby], diagonals[group * baby_step + baby])
                partial = engine.add(partial, term)
            partial = engine.ntt_domain_to_coefficient_domain(partial)
            if group:
                partial = engine.rotate_with_key(partial, rotation_keys[group * baby_step])
            groups.append(partial)
        total = groups[0]
        for partial in groups[1:]:
            total = engine.add(total, partial)
        return engine.rescale_to_next_depth(total)
    return matrix_multiply
'''
    module = _module_function(
        tmp_path / "projection.py",
        code,
        {
            "engine": engine,
            "baby_steps": (1,),
            "baby_step": 2,
            "size": 4,
            "rotation_keys": {2: key},
        },
    )
    source = module.make_function([plain] * 4)
    identity = _identity(source)
    manifest = build_manifest(
        identity, {"plaintext": plain.data, "rotation": key.data}, {}
    )
    assert manifest is not None
    rebound, _ = relocate_materials(source, json.loads(json.dumps(manifest)))
    assert (
        rebound["plaintext"] is plain.data and rebound["rotation"] is key.data
    )
    module.size = 8
    assert identity.key != _identity(source).key


def test_atomic_entry_keeps_program_and_manifest_together_and_corruption_misses(
    tmp_path,
):
    cache = CompileCache(tmp_path)
    entries = [
        CacheEntry(Program.empty(), {"ordinal": i}, i, ()) for i in range(8)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda entry: cache.store("ab123", entry), entries))
    result = cache.load("ab123")
    assert result is not None and result.program is not None
    assert result.manifest["ordinal"] == result.output_structure
    files = list(tmp_path.rglob("*.*"))
    assert len(files) == 1 and files[0].suffix == ".json"
    payload = json.loads(files[0].read_text())
    payload["program"] = "invalid IR"
    files[0].write_text(json.dumps(payload))
    assert cache.load("ab123") is None
    files[0].write_text("{")
    assert cache.load("ab123") is None


def _ckks_leaf(kind, rows, depth, scale, *, batch=(), dtype=torch.int64):
    primes = tuple(range(depth, depth + rows))
    if kind is Ciphertext:
        return Ciphertext(
            torch.zeros((2, *batch, rows, 8), dtype=dtype), depth, scale, primes
        )
    if kind is Plaintext:
        return Plaintext(
            None,
            depth,
            scale,
            torch.zeros((*batch, rows, 8), dtype=dtype),
            "rns",
            "coefficient",
            "Q",
            "standard",
            primes,
        )
    return CompressedPlaintext(
        torch.zeros((*batch, rows, 2), dtype=dtype),
        8,
        "strided_sparse",
        depth,
        scale,
        "ntt",
        "Q",
        "montgomery",
        primes,
        torch.zeros((*batch, rows), dtype=dtype),
    )


@pytest.mark.parametrize("kind", [Ciphertext, Plaintext, CompressedPlaintext])
def test_structural_ckks_identity_and_relocation_use_current_owner(kind):
    first = _ckks_leaf(kind, 4, 0, 16.0, batch=(3,))
    current = _ckks_leaf(kind, 2, 2, 32.0, batch=(3,))
    old_source, new_source = (
        _weighted([first, first]),
        _weighted([current, current]),
    )
    old = source_identity(old_source, material_names=None, structural=True)
    new = source_identity(new_source, material_names=None, structural=True)
    assert old.key == new.key
    assert _identity(old_source).key != _identity(new_source).key
    assert structural_entry_key(
        old, describe_call({"value": first})
    ) == structural_entry_key(new, describe_call({"value": current}))
    bindings = {"main": first.data}
    if kind is CompressedPlaintext:
        bindings["implicit"] = first.implicit_data
    manifest = build_manifest(old, bindings, {}, structural=True)
    assert manifest is not None
    rebound, providers, states = relocate_structural_materials(
        new_source, json.loads(json.dumps(manifest))
    )
    assert rebound["main"] is current.data and states["main"] is current
    assert providers == []
    if kind is CompressedPlaintext:
        assert rebound["implicit"] is current.implicit_data
        assert states["implicit"] is current.implicit_data
    changed_batch = _ckks_leaf(kind, 2, 2, 32.0, batch=(4,))
    changed_dtype = _ckks_leaf(kind, 2, 2, 32.0, batch=(3,), dtype=torch.int32)
    for other in (changed_batch, changed_dtype):
        other_identity = source_identity(
            _weighted([other, other]), material_names=None, structural=True
        )
        assert new.key != other_identity.key
        assert structural_entry_key(
            new, describe_call({"value": current})
        ) != structural_entry_key(new, describe_call({"value": other}))
        with pytest.raises(CacheIneligible):
            relocate_structural_materials(_weighted([other, other]), manifest)


def test_structural_keys_scalars_and_raw_tensor_aliases_remain_fixed():
    first = RotationKey(
        torch.zeros((1, 2, 3, 8), dtype=torch.int64),
        (0, 1, 2),
        ((0,),),
        rotation_step=1,
    )
    current = replace(first, data=torch.ones_like(first.data))
    different_layout = replace(current, digit_prime_ids=((0, 1),))

    def identity(value):
        return source_identity(
            _weighted([value, 2]), material_names=None, structural=True
        )

    old, new = identity(first), identity(current)
    assert old.key == new.key and new.key != identity(different_layout).key
    assert structural_entry_key(
        new, describe_call({"key": current})
    ) != structural_entry_key(new, describe_call({"key": different_layout}))
    assert structural_entry_key(
        new, describe_call({"scalar": 2})
    ) != structural_entry_key(new, describe_call({"scalar": 3}))
    manifest = build_manifest(old, {"key": first.data}, {}, structural=True)
    assert manifest is not None
    rebound, _, states = relocate_structural_materials(
        _weighted([current, 2]), manifest
    )
    assert rebound["key"] is current.data and states["key"] is current
    with pytest.raises(CacheIneligible):
        relocate_structural_materials(
            _weighted([different_layout, 2]), manifest
        )
    before = _ckks_leaf(CompressedPlaintext, 4, 0, 16.0)
    after = _ckks_leaf(CompressedPlaintext, 2, 2, 32.0)
    # The raw reference follows the wrapper, exercising already-assigned alias tokens.
    raw_before = source_identity(
        _weighted([before, before.data]), material_names=None, structural=True
    )
    raw_after = source_identity(
        _weighted([after, after.data]), material_names=None, structural=True
    )
    assert raw_before.key != raw_after.key
    assert structural_entry_key(
        new, describe_call({"raw": before.data})
    ) != structural_entry_key(new, describe_call({"raw": after.data}))


def test_structural_nonstandard_layout_is_a_candidate_miss_not_a_rejection():
    first = _ckks_leaf(Plaintext, 4, 0, 16.0)
    current = _ckks_leaf(Plaintext, 2, 2, 32.0)
    first = replace(first, data=torch.zeros((8, 4), dtype=torch.int64).t())
    current = replace(current, data=torch.zeros((8, 2), dtype=torch.int64).t())
    old_source, new_source = (
        _weighted([first, first]),
        _weighted([current, current]),
    )
    old = source_identity(old_source, material_names=None, structural=True)
    new = source_identity(new_source, material_names=None, structural=True)
    assert old.key != new.key
    manifest = build_manifest(old, {"plain": first.data}, {}, structural=True)
    assert manifest is not None
    same_state = replace(first, data=first.data.clone())
    bindings, _, states = relocate_structural_materials(
        _weighted([same_state, same_state]), manifest
    )
    assert (
        bindings["plain"] is same_state.data and states["plain"] is same_state
    )
    with pytest.raises(CacheIneligible):
        relocate_structural_materials(new_source, manifest)
