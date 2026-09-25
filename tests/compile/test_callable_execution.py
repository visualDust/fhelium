"""Repeated callable execution preserves Python and CKKS behavior."""

import collections

import pytest
import torch

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.eager import Engine


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_reuses_live_inputs_and_preserves_argument_binding(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    weight = torch.linspace(0.5, 1.0, 32, dtype=torch.float64, device=device)

    def source(x, /, offset=1, *, negative=False):
        value = (x + torch.roll(x, offset, dims=-1)) * weight
        return {"value": -value if negative else value, "pair": (value, value)}

    compiled = fc.compile(source)
    x = torch.linspace(-0.2, 0.3, 32, dtype=torch.float64, device=device)
    for negative in (False, True, False):
        actual = compiled(x, negative=negative)
        expected = source(x, negative=negative)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual["pair"][0] is actual["pair"][1]
        weight.add_(0.1)
    assert len(compiled.specializations) == 2
    with pytest.raises(TypeError):
        compiled(x=x)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_plaintext_arguments_and_static_variants(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    engine = Engine(config, rng_seed=89)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    message = torch.linspace(
        -0.02, 0.03, config.num_slots, dtype=torch.float64, device=device
    )
    encrypted = engine.encrypt_message(message, public, device=device)
    plaintext = engine.prepare_plaintext_for_addition(
        engine.encode(message, device=device)
    )

    def source(x, p, negative=False):
        value = engine.add_plaintext(x, p)
        return engine.negate(value) if negative else value

    compiled = fc.compile(source)
    for negative in (False, True, False):
        actual = compiled(encrypted, plaintext, negative)
        expected = source(encrypted, plaintext, negative)
        torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
        assert (actual.depth, actual.scale, actual.prime_ids) == (
            expected.depth,
            expected.scale,
            expected.prime_ids,
        )
    assert len(compiled.specializations) == 2


@pytest.mark.parametrize("on_miss", ["compile", "error"])
def test_clear_recaptures_changed_constants(on_miss):
    factor = 2

    def source(x):
        return x * factor

    compiled = fc.compile(source, on_miss=on_miss)
    x = torch.tensor([1.0])
    compiled.prepare(x)
    torch.testing.assert_close(compiled(x), source(x), rtol=0, atol=0)

    compiled.clear()
    factor = 3
    assert compiled.specializations == ()
    if on_miss == "error":
        with pytest.raises(fc.SpecializationMiss):
            compiled(x)
        compiled.prepare(x)
    torch.testing.assert_close(compiled(x), source(x), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_key_switch_regions_keep_live_key_and_ciphertext_data(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    engine = Engine(
        CkksConfig.parse(Preset.slots8192_scale40_depth7_int64), rng_seed=97
    )
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    keys = [
        engine.create_rotation_key(1, secret, device=device) for _ in range(2)
    ]
    message = torch.linspace(
        -0.02, 0.03, 32, dtype=torch.float64, device=device
    )
    value = engine.encrypt_message(message, public, device=device)

    def source(value, key):
        rotated = engine.rotate_with_key(value, key)
        return engine.add(rotated, value)

    compiled = fc.compile(source)
    for key in keys:
        expected = source(value, key)
        actual = compiled(value, key)
        torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
        assert (actual.depth, actual.scale, actual.prime_ids) == (
            expected.depth,
            expected.scale,
            expected.prime_ids,
        )
        value = engine.negate(value)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_batch_views_reduction_and_supplied_hoisted_keys(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import Ciphertext, Plaintext

    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=103)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret)
    keys = [engine.create_rotation_key(step, secret) for step in (1, 2)]
    x = engine.encrypt_message(
        torch.linspace(-0.02, 0.03, 32), public, device=device
    )
    p = engine.prepare_plaintext_for_multiplication(
        engine.encode(0.125, device=device)
    )

    def source(x, p):
        rotated = engine.rotate_many_with_keys(x, keys, output_domain="ntt")
        original = engine.coefficient_domain_to_ntt_domain(x)
        batch = Ciphertext.stack_batch(
            [original, *rotated, original, rotated[0]]
        )
        weights = Plaintext.stack_batch([p] * 5)
        product = engine.multiply_plaintext(batch, weights)
        return engine.sum_ciphertext_batch(product), batch.slice_batch(
            1, 3
        ).select_batch(0)

    compiled = fc.compile(source)
    for _ in range(2):
        actual, expected = compiled(x, p), source(x, p)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a.data, e.data, rtol=0, atol=0)
            assert (a.batch_shape, a.prime_ids, a.scale) == (
                e.batch_shape,
                e.prime_ids,
                e.scale,
            )
        x = engine.negate(x)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_fuses_ciphertext_products_after_batch_storage_views(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import Ciphertext
    from fhelium.ir.dialects import fusion, rns

    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=107)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret)
    x = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(
            torch.linspace(-0.02, 0.03, 32), public, device=device
        )
    )

    def source(x):
        batch = Ciphertext.stack_batch([x] * 5).slice_batch(1, 4)
        product = engine.multiply(batch, batch)
        return product, engine.sum_ciphertext_batch(product)

    compiled = fc.compile(source)
    for _ in range(2):
        actual, expected = compiled(x), source(x)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a.data, e.data, rtol=0, atol=0)
            assert (a.component_count, a.batch_shape, a.prime_ids, a.scale) == (
                e.component_count,
                e.batch_shape,
                e.prime_ids,
                e.scale,
            )
        x = engine.add(x, x)

    if device.startswith("cuda"):
        # Lowering must actually fuse the component convolution, not leave
        # separate products and a materialized component pack behind.
        program = compiled.specializations[0].compilation.program
        assert any(
            isinstance(op, fusion.FusedOp)
            and any(
                isinstance(nested, rns.PackThreeComponentsOp)
                for nested in op.body.block.ops
            )
            for op in program.single_block("main").ops
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_multiplies_live_batched_compressed_plaintexts(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import Ciphertext, CompressedPlaintext

    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=109)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret)
    message = torch.linspace(
        -0.02, 0.03, 32, dtype=torch.float64, device=device
    )
    x = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(message, public, device=device)
    )
    batch = Ciphertext.stack_batch([x] * 3)
    weights = []
    for factor in (1, 2):
        dense = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                (factor * message).repeat(engine.num_slots // 32), device=device
            )
        )
        compact = CompressedPlaintext.from_plaintext(
            dense, unique_count=64, compression_layout="contiguous"
        )
        weights.append(CompressedPlaintext.stack_batch([compact] * 3))
    fixed = weights[0].clone()

    def source(x, p):
        return engine.multiply_plaintext(x, p), engine.multiply_plaintext(
            x, fixed
        )

    compiled = fc.compile(source)
    for weight in weights:
        fixed.data.copy_(weight.data)
        expected = engine.multiply_plaintext(batch, weight.to_plaintext())
        for actual in compiled(batch, weight):
            torch.testing.assert_close(
                actual.data, expected.data, rtol=0, atol=0
            )
            assert (actual.depth, actual.scale, actual.prime_ids) == (
                expected.depth,
                expected.scale,
                expected.prime_ids,
            )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_compressed_preparation_and_sparse_storage_capture(device, tmp_path):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import fhelium as fh

    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=113)
    period = torch.linspace(-0.02, 0.03, 32, dtype=torch.float64, device=device)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret)
    x = engine.encrypt_message(period, public, device=device)

    def prepare(message):
        return engine.prepare_compressed_plaintext(message, device=device)

    compact = fc.compile(prepare)(period)
    assert isinstance(compact, fh.CompressedPlaintext)
    assert compact.is_rns and compact.representation == "rns"
    assert not compact.is_slots and compact.unique_count == 64
    product = engine.multiply_plaintext(
        engine.coefficient_domain_to_ntt_domain(x), compact
    )
    actual = engine.decrypt_message(
        engine.ntt_domain_to_coefficient_domain(product), secret, is_real=True
    )[:32]
    torch.testing.assert_close(actual, period.square(), rtol=0, atol=1e-5)

    dense_add = engine.prepare_plaintext_for_addition(
        engine.encode(period.repeat(engine.num_slots // 32), device=device)
    )
    sparse = fh.CompressedPlaintext.from_plaintext(
        dense_add, unique_count=64, compression_layout="strided_sparse"
    )
    fixed = sparse.clone()

    def add(value, p):
        return (
            engine.add_plaintext(value, p),
            engine.add_plaintext(value, fixed),
            p.clone(),
        )

    compiled = fc.compile(add)
    for implicit in (0, 1):
        sparse.implicit_data.fill_(implicit)
        fixed.implicit_data.copy_(sparse.implicit_data)
        a, b, returned = compiled(x, sparse)
        expected = engine.add_plaintext(x, sparse.to_plaintext())
        torch.testing.assert_close(a.data, expected.data, rtol=0, atol=0)
        torch.testing.assert_close(b.data, expected.data, rtol=0, atol=0)
        torch.testing.assert_close(
            returned.decompress_data(), sparse.decompress_data(), rtol=0, atol=0
        )
    standard = engine.montgomery_residues_to_standard_residues(sparse)
    restored = engine.standard_residues_to_montgomery_residues(standard)
    moduli = torch.tensor(
        [engine.config.moduli[i] for i in sparse.prime_ids], device=device
    ).view(-1, 1)
    torch.testing.assert_close(
        restored.decompress_data() % moduli,
        sparse.decompress_data() % moduli,
        rtol=0,
        atol=0,
    )
    path = tmp_path / "compressed.safetensors"
    fh.save_value(standard, path)
    loaded = fh.load_value(
        path, device=device, expected_type=fh.CompressedPlaintext
    )
    assert loaded.residue_representation == "standard"
    torch.testing.assert_close(
        loaded.decompress_data(), standard.decompress_data(), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_compressed_grouped_products_and_ntt_addition(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import fhelium as fh

    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=193)
    message = torch.linspace(
        -0.01, 0.02, 128, dtype=torch.float64, device=device
    )
    key = engine.create_public_key(engine.create_secret_key(device=device))
    x = engine.encrypt_message(message, key, device=device)
    x = engine.coefficient_domain_to_ntt_domain(x)
    compact = engine.prepare_compressed_plaintext(message, device=device)
    dense = compact.to_plaintext()

    def grouped(value, p):
        return engine.sum_plaintext_product_groups(
            [value, value], [[p, p], [p, p]]
        )

    compiled = fc.compile(grouped)
    expected = grouped(x, dense)
    actual = compiled(x, compact)
    moduli = torch.tensor(
        [engine.config.moduli[i] for i in x.prime_ids], device=device
    ).view(-1, 1)
    torch.testing.assert_close(
        actual.data % moduli, expected.data % moduli, rtol=0, atol=0
    )
    if device.startswith("cuda"):
        assert any(
            "triton" in dispatch.implementation.name
            for dispatch in compiled.specializations[
                0
            ].executable.dispatch_table.operations.values()
        )

    target = x.clone()
    alias = target.data
    engine.add_plaintext_(target, compact)
    assert target.data is alias
    expected = engine.add_plaintext(x, dense)
    torch.testing.assert_close(
        target.data % moduli, expected.data % moduli, rtol=0, atol=0
    )
    zero = engine.zero_plaintext_like(compact)
    assert isinstance(zero, fh.CompressedPlaintext)
    assert torch.count_nonzero(zero.data) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rotation_hoisting_shares_interleaved_inputs_and_keeps_live_keys(
    device,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium.ir.dialects import ckks

    engine = Engine(Preset.slots8192_scale40_depth7_int64, rng_seed=149)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    keys = [
        engine.create_rotation_key(step, secret, device=device)
        for step in (1, 2, 3)
    ]
    message = torch.linspace(
        -0.02, 0.03, engine.config.num_slots, device=device, dtype=torch.float64
    )
    x = engine.encrypt_message(message, public, device=device)
    y = engine.add(x, x)

    def source(x, y):
        results = []
        for key in keys:
            left = engine.negate(engine.rotate_with_key(x, key))
            right = engine.negate(engine.rotate_with_key(y, key))
            results.append(engine.add(left, right))
        return tuple(results)

    compiled = fc.compile(source)
    moduli = torch.tensor(
        engine.config.q_moduli, dtype=torch.int64, device=device
    ).view(-1, 1)
    for _ in range(2):
        actual = compiled(x, y)
        left = engine.rotate_many_with_keys(x, keys, use_hoisting=True)
        right = engine.rotate_many_with_keys(y, keys, use_hoisting=True)
        independent = source(x, y)
        for got, a, b, separate in zip(
            actual, left, right, independent, strict=True
        ):
            expected = engine.add(engine.negate(a), engine.negate(b))
            assert torch.equal(got.data % moduli, expected.data % moduli)
            assert (got.depth, got.scale, got.prime_ids) == (
                expected.depth,
                expected.scale,
                expected.prime_ids,
            )
            # Shared digit preparation and independent key switching can have
            # different encryption noise while encoding the same rotation.
            torch.testing.assert_close(
                engine.decrypt_message(got, secret, is_real=True),
                engine.decrypt_message(separate, secret, is_real=True),
                rtol=0,
                atol=1e-5,
            )
        x = engine.negate(x)
        for step, key in zip((1, 2, 3), keys, strict=True):
            key.data.copy_(
                engine.create_rotation_key(step, secret, device=device).data
            )
    operations = tuple(compiled.specializations[0].compilation.program.walk())
    assert sum(isinstance(op, ckks.RotateManyOp) for op in operations) == 2
    assert not any(isinstance(op, ckks.RotateOp) for op in operations)
    assert len(compiled.specializations) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_batch_reduction_uses_one_reduction(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    engine = Engine(config, rng_seed=97)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    messages = torch.linspace(
        -0.02, 0.03, config.num_slots, dtype=torch.float64, device=device
    )
    batch = engine.encrypt_message(
        torch.stack((messages, -messages, messages, -messages, messages, -messages))
    )
    before = batch.data.clone()

    def source(x):
        return engine.sum_ciphertext_batch(x)

    compiled = fc.compile(source)
    actual = compiled(batch)
    expected = source(batch)
    torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
    assert (actual.depth, actual.scale, actual.prime_ids) == (
        expected.depth,
        expected.scale,
        expected.prime_ids,
    )
    assert actual.batch_shape == expected.batch_shape == torch.Size([])
    assert torch.equal(batch.data, before)
    # The compiled program must not rebuild the pairwise fold it replaces.
    names = collections.Counter(
        operation.name
        for operation in compiled.specializations[0].compilation.program.walk()
    )
    assert names["fhelium_rns.sum_standard_batch"] == 1
    assert names["fhelium_rns.add_standard"] == 0
