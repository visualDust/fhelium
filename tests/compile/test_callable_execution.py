"""Repeated callable execution preserves Python and CKKS behavior."""

import collections

from typing import Any

import pytest
import torch

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.eager import Engine
from xdsl.utils.exceptions import VerifyException


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
        torch.stack(
            (messages, -messages, messages, -messages, messages, -messages)
        ),
        public,
        device=device,
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


@pytest.mark.parametrize(
    "policy", ["radix2_compact_group8_smem8", "radix4_compact"]
)
def test_callable_ntt_chain_preserves_fanout_and_live_inputs(policy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    preset = {
        "radix8_compact": Preset.slots16384_scale40_depth16_int64,
        "radix16_compact": Preset.slots32768_scale40_depth34_int64,
    }.get(policy, Preset.slots8192_scale40_depth7_int64)
    engine = Engine(
        preset,
        ntt_backend=policy,
        rng_seed=137,
        rng_nonce=7,
    )
    secret = engine.create_secret_key(device="cuda:0")
    public = engine.create_public_key(secret, device="cuda:0")
    x = engine.encrypt_message(
        torch.linspace(-0.02, 0.03, 32, dtype=torch.float64, device="cuda:0"),
        public,
        device="cuda:0",
    )
    weight = engine.prepare_plaintext_for_multiplication(
        engine.encode(0.25, device="cuda:0")
    )

    def source(x, weight):
        first = engine.coefficient_domain_to_ntt_domain(x)
        product = engine.multiply_plaintext(first, weight)
        side = engine.rescale_to_next_depth(x)
        coefficient = engine.ntt_domain_to_coefficient_domain(product)
        doubled = engine.add(coefficient, coefficient)
        second = engine.coefficient_domain_to_ntt_domain(doubled)
        result = engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(second, weight)
        )
        return first, result, side

    compiled = fc.compile(source)
    for _ in range(2):
        actual, expected = compiled(x, weight), source(x, weight)
        for got, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(got.data, reference.data, rtol=0, atol=0)
            assert (
                got.depth,
                got.scale,
                got.prime_ids,
                got.component_count,
                got.polynomial_domain,
                got.residue_representation,
            ) == (
                reference.depth,
                reference.scale,
                reference.prime_ids,
                reference.component_count,
                reference.polynomial_domain,
                reference.residue_representation,
            )
        x = engine.negate(x)
        weight.data.copy_(
            engine.prepare_plaintext_for_multiplication(
                engine.encode(0.5, device="cuda:0")
            ).data
        )


@pytest.mark.parametrize(
    "policy",
    [
        "radix2_compact_group8_smem8",
        "radix4_compact",
        "radix8_compact",
        "radix16_compact",
    ],
)
def test_callable_ntt_join_preserves_branch_outputs(policy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    preset = {
        "radix8_compact": Preset.slots16384_scale40_depth16_int64,
        "radix16_compact": Preset.slots32768_scale40_depth34_int64,
    }.get(policy, Preset.slots8192_scale40_depth7_int64)
    from fhelium import Ciphertext

    engine = Engine(preset, ntt_backend=policy)
    data = torch.stack(
        [
            torch.randint(
                0,
                modulus,
                (2, engine.config.N),
                device="cuda:0",
                dtype=engine.dtype,
            )
            for modulus in engine.config.q_moduli
        ],
        dim=-2,
    )
    x = Ciphertext(
        data,
        depth=0,
        scale=engine.config.default_scale,
        prime_ids=tuple(range(engine.config.num_q_primes)),
    )
    y = engine.add(x, x)
    weight = engine.prepare_plaintext_for_multiplication(
        engine.encode(0.25, device="cuda:0")
    )

    def source(x, y):
        first = engine.coefficient_domain_to_ntt_domain(x)
        left = engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(first, weight)
        )
        second = engine.coefficient_domain_to_ntt_domain(y)
        right = engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(second, weight)
        )
        joined = engine.coefficient_domain_to_ntt_domain(
            engine.add(left, right)
        )
        result = engine.ntt_domain_to_coefficient_domain(joined)
        return first, left, right, result

    compiled = fc.compile(source)
    actual, expected = compiled(x, y), source(x, y)
    for got, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.data, reference.data, rtol=0, atol=0)
        assert (got.depth, got.scale, got.prime_ids, got.polynomial_domain) == (
            reference.depth,
            reference.scale,
            reference.prime_ids,
            reference.polynomial_domain,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_preserves_key_switch_rounding(device):
    """Capture and execution retain each requested ModDown quotient rule."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    engine = Engine(config, rng_seed=211)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    rotation = engine.create_rotation_key(1, secret, device=device)
    conjugation = engine.create_conjugation_key(secret, device=device)
    relin = engine.create_relinearization_key(secret, device=device)
    x = engine.encrypt_message(
        torch.linspace(-0.02, 0.03, 32, device=device), public, device=device
    )
    x_ntt = engine.coefficient_domain_to_ntt_domain(x)
    triplet = engine.multiply(x_ntt, x_ntt)

    def source(value, product, rounding="nearest"):
        return (
            engine.rotate_with_key(value, rotation, moddown_rounding=rounding),
            engine.conjugate(
                value,
                conjugation,
                output_domain="ntt",
                moddown_rounding=rounding,
            ),
            engine.relinearize(product, relin, moddown_rounding=rounding),
        )

    reference = engine.decrypt_message(x, secret)
    mathematical_outputs = (
        torch.roll(reference, 1),
        reference.conj(),
        reference.square(),
    )
    compiled = fc.compile(source)
    for rounding in ("nearest", "floor"):
        expected = source(x, triplet, rounding)
        actual = compiled(x, triplet, rounding)
        for a, e, mathematical in zip(
            actual, expected, mathematical_outputs, strict=True
        ):
            moduli = torch.tensor(
                [config.moduli[i] for i in a.prime_ids],
                device=a.device,
                dtype=a.data.dtype,
            )[:, None]
            # NTT implementations may return different representatives in [0, 2q).
            assert bool(((a.data >= 0) & (a.data < 2 * moduli)).all())
            assert bool(((e.data >= 0) & (e.data < 2 * moduli)).all())
            torch.testing.assert_close(
                a.data % moduli, e.data % moduli, rtol=0, atol=0
            )
            torch.testing.assert_close(
                engine.decrypt_message(a, secret),
                mathematical,
                rtol=0,
                atol=1e-6,
            )
            assert (a.polynomial_domain, a.residue_representation) == (
                e.polynomial_domain,
                e.residue_representation,
            )
            assert (a.depth, a.scale, a.prime_ids) == (
                e.depth,
                e.scale,
                e.prime_ids,
            )
    default = engine.rotate_with_key(x, rotation)
    explicit = engine.rotate_with_key(x, rotation, moddown_rounding="nearest")
    torch.testing.assert_close(default.data, explicit.data, rtol=0, atol=0)
    assert len(compiled.specializations) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_weighted_sum_keeps_requested_backend_and_sparse_operands(
    device,
):
    from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE, Program
    from fhelium.ir.dialects import rns
    from fhelium.runtime.signature import ValueSignature
    from fhelium.values import Ciphertext, CompressedPlaintext

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    base = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig.parse(
        base.dumps(),
        logN=8,
        q_depth_groups=base.q_depth_groups[:2],
        enforce_security_budget=False,
    )
    engine = Engine(config)
    state: dict[str, Any] = dict(
        depth=0,
        scale=8.0,
        prime_ids=(0, 1),
        polynomial_domain="ntt",
        residue_representation="montgomery",
        modulus_basis="Q",
    )
    x = Ciphertext(
        data=torch.arange(1024, dtype=torch.int64, device=device).reshape(
            2, 2, 256
        ),
        **state,
    )
    data = torch.arange(16, dtype=torch.int64, device=device).reshape(2, 8) + 17
    p = CompressedPlaintext(
        data=data, ring_dimension=256, compression_layout="contiguous", **state
    )
    q = CompressedPlaintext(
        data=data + 1,
        implicit_data=torch.tensor([3, 5], dtype=torch.int64, device=device),
        ring_dimension=256,
        compression_layout="strided_sparse",
        **state,
    )

    def source(x, p, q):
        return engine.sum_plaintext_product_groups(
            (x, x), ((p, q), (q, p)), implementation="auto", term_chunk_size=1
        )

    compiled = fc.compile(source)
    actual = compiled(x, p, q)
    expected = source(x, p, q)
    assert torch.equal(actual.data, expected.data)
    assert ValueSignature.from_value(actual) == ValueSignature.from_value(
        expected
    )
    program = compiled.specializations[0].source.program
    operations = [
        op
        for op in program.single_block().ops
        if isinstance(op, rns.MontgomeryWeightedSumsOp)
    ]
    assert len(operations) == 1
    assert operations[0].attributes[
        EXECUTION_IMPLEMENTATION_ATTRIBUTE
    ].data == engine.plaintext_product_implementation(
        device, compressed=True, implementation="auto"
    )
    Program.parse(str(program)).verify_structure()
    q.implicit_data.add_(2)
    assert torch.equal(compiled(x, p, q).data, source(x, p, q).data)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aggregate_rotation_sum_preserves_decoded_values_and_live_keys(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium.backend import OperationBackend

    standard = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=standard.q_depth_groups,
        p_moduli=standard.p_moduli,
        default_scale=standard.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=137, allow_automatic_key_generation=False)
    secret = engine.create_secret_key(device=device)
    public = engine.create_public_key(secret, device=device)
    keys = [
        engine.create_rotation_key(step, secret, device=device)
        for step in (1, -3)
    ]
    message = torch.linspace(
        -0.02, 0.03, config.num_slots, dtype=torch.float64, device=device
    )
    encrypted = engine.encrypt_message(message, public, device=device)
    identity = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.ones_like(message),
            scale=float(config.q_depth_groups[0][0]),
            device=device,
        )
    )
    high = engine.ntt_domain_to_coefficient_domain(
        engine.multiply_plaintext(
            engine.coefficient_domain_to_ntt_domain(encrypted), identity
        )
    )

    def source(x, key1, key2):
        return engine.rescale_to_next_depth(
            engine.add(
                engine.add(
                    engine.rotate_with_key(x, key1),
                    engine.rotate_with_key(x, key2),
                ),
                x,
            )
        )

    backend = OperationBackend()
    resources = engine._dispatcher_for(device).device_resources
    pipeline = fc.default_lower_and_fuse_pipeline(backend).before(
        "rotation-hoisting", fc.AggregateRotationSumsPass()
    )
    compiled = fc.compile(
        source, backend=backend, pipeline=pipeline, resources=resources
    )

    def aggregate(x, key1, key2):
        return engine.sum_rotations(
            ((x, key1), (x, key2), (x, None)), rescale=True
        )

    direct = fc.compile(aggregate, backend=backend)
    expected = message + torch.roll(message, 1) + torch.roll(message, -3)
    for sign in (1, -1):
        value = high if sign == 1 else engine.negate(high)
        result = compiled(value, *keys)
        independent = source(value, *keys)
        eager = aggregate(value, *keys)
        captured = direct(value, *keys)
        torch.testing.assert_close(result.data, eager.data, rtol=0, atol=0)
        torch.testing.assert_close(captured.data, eager.data, rtol=0, atol=0)
        streamed = list(
            engine.iter_rotations_with_keys(
                value, [None, *keys], output_domain="ntt"
            )
        )
        hoisted = engine.rotate_many_with_keys(value, keys, output_domain="ntt")
        for actual, reference in zip(streamed[1:], hoisted, strict=True):
            torch.testing.assert_close(
                actual.data, reference.data, rtol=0, atol=0
            )
        assert (result.depth, result.scale, result.prime_ids) == (
            independent.depth,
            independent.scale,
            independent.prime_ids,
        )
        torch.testing.assert_close(
            engine.decrypt_message(result, secret).real,
            expected * sign,
            rtol=0,
            atol=1e-5,
        )
        replacement = engine.create_rotation_key(1, secret, device=device)
        keys[0].data.copy_(replacement.data)

    positive = high.clone()
    negative = engine.negate(positive)
    expected_sum = engine.sum_rotations(((positive, None), (negative, None)))
    buffer = positive.clone()

    def reused_buffer():
        yield buffer, None
        buffer.data.copy_(negative.data)
        yield buffer, None

    torch.testing.assert_close(
        engine.sum_rotations(reused_buffer()).data,
        expected_sum.data,
        rtol=0,
        atol=0,
    )

    def identities(x):
        return tuple(engine.iter_rotations_with_keys(x, (None, None)))

    for outputs in (identities(positive), fc.compile(identities)(positive)):
        pointers = [
            positive.data.untyped_storage().data_ptr(),
            *(value.data.untyped_storage().data_ptr() for value in outputs),
        ]
        assert len(set(pointers)) == 3

    def identity_sum(x):
        return engine.sum_rotations(((x, None), (x, None)))

    logical = fc.default_lower_and_fuse_pipeline(backend).before(
        "select-execution-lowerings", fc.LowerCkksToRnsNttPass()
    )

    def rotations_only(x, key1, key2):
        return engine.sum_rotations(((x, key1), (x, key2)), rescale=True)

    for value in (positive, engine.coefficient_domain_to_ntt_domain(positive)):
        for function in (aggregate, rotations_only):
            actual = fc.compile(
                function,
                backend=backend,
                pipeline=logical,
                resources=engine.rns_context(device),
            )(value, *keys)
            expected_value = function(value, *keys)
            torch.testing.assert_close(
                actual.data, expected_value.data, rtol=0, atol=0
            )
            assert (actual.depth, actual.scale, actual.prime_ids) == (
                expected_value.depth,
                expected_value.scale,
                expected_value.prime_ids,
            )

    observed = fc.compile(
        identity_sum,
        backend=backend,
        pipeline=logical,
        resources=engine.rns_context(device),
    )(positive)
    torch.testing.assert_close(
        observed.data, identity_sum(positive).data, rtol=0, atol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_modulus_raise_uses_live_encapsulation_keys(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CkksConfig(
        logN=8,
        q_depth_groups=(
            (254279681, 253493249),
            (253100033, 249561089),
            (205651969, 204865537),
        ),
        p_moduli=(257949697, 258605057),
        default_scale=2.0**40,
        galois_generator=5,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=331)
    secret = engine.create_secret_key(hamming_weight=128, device=device)
    middle = engine.create_secret_key(hamming_weight=32, device=device)
    auxiliary = dict(
        low_p_moduli=(268042241,),
        return_prime=256376833,
        return_p_moduli=(265420801, 268042241),
    )
    low, back = engine.create_encapsulation_keys(secret, middle, **auxiliary)
    public = engine.create_public_key(secret)
    message = torch.linspace(
        -0.01, 0.02, config.num_slots, dtype=torch.float64, device=device
    )
    value = engine.encrypt_message(
        message, public, depth=config.max_depth, device=device
    )

    def source(x, first, second):
        return engine.modulus_raise(x), engine.encapsulated_modulus_raise(
            x, first, second, **auxiliary
        )

    compiled = fc.compile(source)
    original = value.data.clone()
    for changed in (False, True):
        if changed:
            fresh_low, fresh_back = engine.create_encapsulation_keys(
                secret, middle, **auxiliary
            )
            low.data.copy_(fresh_low.data)
            back.data.copy_(fresh_back.data)
        actual, expected = compiled(value, low, back), source(value, low, back)
        for a, e in zip(actual, expected, strict=True):
            moduli = torch.tensor(
                config.q_moduli, dtype=torch.int64, device=device
            ).view(-1, 1)
            torch.testing.assert_close(
                a.data.to(torch.int64) % moduli,
                e.data.to(torch.int64) % moduli,
                rtol=0,
                atol=0,
            )
            assert (a.depth, a.scale, a.prime_ids) == (
                0,
                value.scale,
                tuple(range(config.num_q_primes)),
            )
        torch.testing.assert_close(value.data, original, rtol=0, atol=0)
    assert len(compiled.specializations) == 1

    def centered(x, first, second):
        return engine.modulus_raise(x)

    def encapsulated(x, first, second):
        return engine.encapsulated_modulus_raise(x, first, second, **auxiliary)

    invalid = engine.coefficient_domain_to_ntt_domain(value)
    for function in (centered, encapsulated):
        for execute in (function, fc.compile(function)):
            with pytest.raises(
                (ValueError, VerifyException), match="coefficient"
            ):
                execute(invalid, low, back)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_objects_inline_methods_and_bind_only_used_materials(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from collections.abc import Mapping

    class WrappedKeys(Mapping):
        def __init__(self, weight, unused):
            self.table = {1: weight, 2: unused}

        def __getitem__(self, key):
            return self.table[key % 8]

        def get(self, key, default=None):
            return self[key] * 2

        def __iter__(self):
            return iter(self.table)

        def __len__(self):
            return len(self.table)

    class Affine:
        def __init__(self):
            self.weight = torch.arange(8, dtype=torch.float64, device=device)
            self.unused = torch.ones(3, device=device)
            self.weights = WrappedKeys(self.weight, self.unused)
            self.offset = 2
            self._capture = False

        @property
        def doubled(self):
            return self.weights.get(9)

        def transform(self, x):
            assert type(self) is Affine
            return x * self.doubled + self.offset + (1 if self._capture else 2)

        def __call__(self, x):
            return self.transform(torch.roll(x, 1))

    source = Affine()
    x = torch.ones(8, dtype=torch.float64, device=device)
    compiled = fc.compile(source)
    torch.testing.assert_close(compiled(x), source(x), rtol=0, atol=0)
    bindings = compiled.specializations[
        0
    ].compilation.material_bindings.values()
    assert all(t is not source.unused for t in bindings)
    source.weight.add_(1)
    torch.testing.assert_close(compiled(x), source(x), rtol=0, atol=0)

    @fc.compile
    def outer(x):
        return source(x) + compiled(x)

    torch.testing.assert_close(outer(x), source(x) * 2, rtol=0, atol=0)
    torch.testing.assert_close(
        fc.compile(source.transform)(x), source.transform(x), rtol=0, atol=0
    )


def test_capture_rejects_mutation_of_bound_object():
    class Stateful:
        count = 0

        @property
        def offset(self):
            self.count += 1
            return 1

        def __call__(self, x):
            return x + self.offset

    source = Stateful()
    with pytest.raises(fc.CaptureError, match="mutate bound object"):
        fc.compile(source)(torch.ones(1))
    assert source.count == 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_callable_value_replacement_preserves_derived_metadata(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from dataclasses import replace
    from fhelium.values import Ciphertext, CompressedPlaintext

    value = Ciphertext(
        torch.ones((2, 1, 8), dtype=torch.int64, device=device),
        depth=0,
        scale=1.0,
        prime_ids=(0,),
    )
    plaintext = CompressedPlaintext(
        torch.ones((1, 2), dtype=torch.int64, device=device),
        ring_dimension=8,
        compression_layout="strided_sparse",
        depth=0,
        scale=1.0,
        polynomial_domain="coefficient",
        modulus_basis="Q",
        residue_representation="standard",
        prime_ids=(0,),
        implicit_data=torch.ones(1, dtype=torch.int64, device=device),
    )

    def source(x, p):
        result = x.with_data(torch.cat((x.data, x.data[:1]), dim=0)[..., :4])
        updated = replace(p, implicit_data=p.implicit_data + 1)
        return (
            result,
            result.data * result.component_count * result.ring_dimension,
            updated,
        )

    expected = source(value, plaintext)
    actual = fc.compile(source)(value, plaintext)
    assert actual[0].component_count == expected[0].component_count == 3
    assert actual[0].ring_dimension == expected[0].ring_dimension == 4
    torch.testing.assert_close(actual[0].data, expected[0].data, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[2].implicit_data, expected[2].implicit_data, rtol=0, atol=0
    )
    assert actual[2].data.data_ptr() == plaintext.data.data_ptr()


def test_repeated_returned_arithmetic_has_independent_tensor_storage():
    def source(value):
        return value * value, value * value

    value = torch.tensor([2.0, 3.0], dtype=torch.float64)
    first, second = fc.compile(source)(value)

    torch.testing.assert_close(first, value * value, rtol=0, atol=0)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert (
        first.untyped_storage().data_ptr()
        != second.untyped_storage().data_ptr()
    )
    first.add_(1)
    torch.testing.assert_close(second, value * value, rtol=0, atol=0)
