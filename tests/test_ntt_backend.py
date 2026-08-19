from __future__ import annotations

import gc

import pytest
import torch

from fhelium import CkksEngine
from fhelium.config import CkksConfig, Preset
from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    compatible_ntt_backends,
    resolve_ntt_backend_policy,
)
from fhelium.engine.ntt.plans import (
    CompactPowerOfTwoRadixNttPlan,
    IndexedRadix2NttPlan,
)
from fhelium.engine.rns.montgomery import MontgomeryParameters
from fhelium.engine.ntt.tables import (
    CompactPowerOfTwoRadixTables,
    CompactRadix2Tables,
)
from fhelium.native.wrapper import ntt_diagnostic_ops, ntt_ops

INDEXED_BACKEND = "radix2_indexed"
RADIX2_PRODUCTION_BACKENDS = (
    "radix2_compact_group4_smem8",
    "radix2_compact_group8_smem8",
    "radix2_compact_group16_smem8",
)
FIXED_RADIX_CASES = (
    (Preset.slots8192_scale40_levels7_int64, "radix4_compact"),
    (Preset.slots16384_scale40_levels16_int64, "radix8_compact"),
    (Preset.slots32768_scale40_levels34_int64, "radix16_compact"),
)
EXACT_BACKEND_CASES = (
    (
        Preset.slots8192_scale40_levels7_int64,
        (*RADIX2_PRODUCTION_BACKENDS, "radix4_compact"),
    ),
    (Preset.slots16384_scale40_levels16_int64, ("radix8_compact",)),
    (Preset.slots32768_scale40_levels34_int64, ("radix16_compact",)),
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _cpu_indexed_runtime(
    *, buffer_bit_length: int, scale_bits: int
) -> tuple[
    CkksConfig,
    tuple[IndexedRadix2NttPlan, torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    config = CkksConfig.parse(
        Preset.slots8192_scale30_levels9_int64,
        buffer_bit_length=buffer_bit_length,
        scale_bits=scale_bits,
        enforce_security_budget=False,
    )
    montgomery = MontgomeryParameters(config)
    plan = IndexedRadix2NttPlan(config, device="cpu")
    parameters = torch.tensor(
        [
            montgomery.twice_modulus,
            montgomery.modulus_lower_bits,
            montgomery.modulus_higher_bits,
            montgomery.neg_inv_modulus_lower_bits,
            montgomery.neg_inv_modulus_higher_bits,
            montgomery.montgomery_r2,
            [0] * len(config.moduli),
            [
                (normalizer * montgomery.R) % modulus
                for normalizer, modulus in zip(
                    config.inverse_ntt_scale, config.moduli, strict=True
                )
            ],
        ],
        dtype=config.torch_dtype,
    )

    def to_montgomery(table: torch.Tensor) -> torch.Tensor:
        rows = [
            torch.tensor(
                [
                    (int(value) * montgomery.R) % modulus
                    for value in row.reshape(-1)
                ],
                dtype=config.torch_dtype,
            ).reshape_as(row)
            for row, modulus in zip(table, config.moduli, strict=True)
        ]
        return torch.stack(rows)

    return (
        config,
        (
            plan,
            to_montgomery(plan.forward_twiddles),
            to_montgomery(plan.inverse_twiddles),
        ),
        parameters,
    )


@pytest.mark.parametrize(
    ("buffer_bit_length", "scale_bits"), [(30, 25), (62, 30)]
)
def test_cpu_indexed_ntt_uses_the_same_exact_schema_and_representation(
    buffer_bit_length: int, scale_bits: int
) -> None:
    config, tables, parameters = _cpu_indexed_runtime(
        buffer_bit_length=buffer_bit_length, scale_bits=scale_bits
    )
    plan, forward_twiddles, inverse_twiddles = tables
    generator = torch.Generator().manual_seed(20260811 + buffer_bit_length)
    active_limb_count = 3
    standard = (
        torch.stack(
            [
                torch.randint(
                    0,
                    modulus,
                    (2, config.N),
                    dtype=config.torch_dtype,
                    generator=generator,
                )
                for modulus in config.moduli[:active_limb_count]
            ]
        )
        .transpose(0, 1)
        .contiguous()
    )
    active_forward_twiddles = forward_twiddles[:active_limb_count]
    active_inverse_twiddles = inverse_twiddles[:active_limb_count]
    active_parameters = parameters[:, :active_limb_count]
    table_snapshots = tuple(
        table.clone()
        for table in (
            plan.forward_indices,
            plan.inverse_indices,
            active_forward_twiddles,
            active_inverse_twiddles,
            active_parameters,
        )
    )

    transformed = ntt_ops.forward_ntt_to_montgomery_indexed(
        standard,
        plan.forward_indices[0],
        plan.forward_indices[1],
        active_forward_twiddles,
        active_parameters,
    )
    assert transformed.data_ptr() != standard.data_ptr()

    transformed_inplace = standard.clone()
    ntt_ops.forward_ntt_to_montgomery_indexed_(
        transformed_inplace,
        plan.forward_indices[0],
        plan.forward_indices[1],
        active_forward_twiddles,
        active_parameters,
    )
    active_moduli = torch.tensor(
        config.moduli[:active_limb_count], dtype=config.torch_dtype
    ).view(1, -1, 1)
    assert torch.equal(
        torch.remainder(transformed_inplace - transformed, active_moduli),
        torch.zeros_like(transformed),
    )

    restored = transformed.clone()
    original_storage = restored.data_ptr()
    ntt_ops.inverse_ntt_to_standard_indexed_(
        restored,
        plan.inverse_indices[0],
        plan.inverse_indices[1],
        active_inverse_twiddles,
        active_parameters,
    )
    assert restored.data_ptr() == original_storage
    assert torch.equal(restored, standard)

    lazy = transformed.clone()
    ntt_ops.inverse_ntt_to_standard_lazy_indexed_(
        lazy,
        plan.inverse_indices[0],
        plan.inverse_indices[1],
        active_inverse_twiddles,
        active_parameters,
    )
    assert torch.equal(torch.remainder(lazy, active_moduli), standard)

    centered = transformed.clone()
    ntt_ops.inverse_ntt_to_centered_indexed_(
        centered,
        plan.inverse_indices[0],
        plan.inverse_indices[1],
        active_inverse_twiddles,
        active_parameters,
    )
    expected_centered = torch.where(
        standard <= (active_moduli >> 1), standard, standard - active_moduli
    )
    assert torch.equal(centered, expected_centered)

    montgomery_coefficients = transformed.clone()
    ntt_ops.inverse_ntt_montgomery_indexed_(
        montgomery_coefficients,
        plan.inverse_indices[0],
        plan.inverse_indices[1],
        active_inverse_twiddles,
        active_parameters,
    )
    ntt_ops.forward_ntt_montgomery_indexed_(
        montgomery_coefficients,
        plan.forward_indices[0],
        plan.forward_indices[1],
        active_forward_twiddles,
        active_parameters,
    )
    assert torch.equal(
        torch.remainder(montgomery_coefficients - transformed, active_moduli),
        torch.zeros_like(transformed),
    )

    for table, snapshot in zip(
        (
            plan.forward_indices,
            plan.inverse_indices,
            active_forward_twiddles,
            active_inverse_twiddles,
            active_parameters,
        ),
        table_snapshots,
        strict=True,
    ):
        assert torch.equal(table, snapshot)


def test_cpu_indexed_ntt_rejects_unsafe_schedules_and_storage() -> None:
    config, tables, parameters = _cpu_indexed_runtime(
        buffer_bit_length=30, scale_bits=25
    )
    plan, forward_twiddles, _ = tables
    standard = torch.stack(
        [
            torch.arange(config.N, dtype=config.torch_dtype) % modulus
            for modulus in config.moduli
        ]
    )

    with pytest.raises(RuntimeError, match=r"log2\(N\) stages"):
        ntt_ops.forward_ntt_to_montgomery_indexed(
            standard,
            plan.forward_indices[0][:-1],
            plan.forward_indices[1][:-1],
            forward_twiddles[:, :-1],
            parameters,
        )

    out_of_range = plan.forward_indices[0].clone()
    out_of_range[0, 0] = config.N
    with pytest.raises(RuntimeError, match="out-of-range coefficient index"):
        ntt_ops.forward_ntt_to_montgomery_indexed(
            standard,
            out_of_range,
            plan.forward_indices[1],
            forward_twiddles,
            parameters,
        )

    duplicate = plan.forward_indices[0].clone()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(RuntimeError, match="partition every coefficient"):
        ntt_ops.forward_ntt_to_montgomery_indexed(
            standard,
            duplicate,
            plan.forward_indices[1],
            forward_twiddles,
            parameters,
        )

    overlapping = standard[:1].expand(2, -1, -1)
    with pytest.raises(RuntimeError, match="more than one element"):
        ntt_ops.forward_ntt_to_montgomery_indexed_(
            overlapping,
            plan.forward_indices[0],
            plan.forward_indices[1],
            forward_twiddles,
            parameters,
        )


def _standard_rows(engine: CkksEngine) -> torch.Tensor:
    coefficients = torch.arange(
        engine.config.N,
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    rows = []
    for row_id, modulus in enumerate(engine.config.moduli):
        row = torch.remainder(coefficients + 97 * row_id, modulus)
        row[1::2] = modulus - 1 - row[1::2]
        rows.append(row)
    return torch.stack(rows).contiguous()


@pytest.mark.parametrize("log_n", [14, 15, 16, 17])
def test_fixed_radix_compatibility_requires_integral_stage_counts(
    log_n: int,
) -> None:
    compatible = compatible_ntt_backends(log_n)
    for _, backend in FIXED_RADIX_CASES:
        policy = resolve_ntt_backend_policy(backend)
        assert isinstance(policy, CompactFixedRadixPolicy)
        stage_width = policy.radix.bit_length() - 1
        assert (backend in compatible) is (log_n % stage_width == 0)

    assert INDEXED_BACKEND in compatible
    assert all(backend in compatible for backend in RADIX2_PRODUCTION_BACKENDS)

    with pytest.raises(ValueError, match="must be positive"):
        compatible_ntt_backends(0)


@pytest.mark.parametrize(("preset", "backend"), FIXED_RADIX_CASES)
def test_fixed_radix_root_powers_are_primitive_and_invertible(
    preset: Preset,
    backend: str,
) -> None:
    config = CkksConfig.parse(preset)
    policy = resolve_ntt_backend_policy(backend)
    assert isinstance(policy, CompactFixedRadixPolicy)
    plan = CompactPowerOfTwoRadixNttPlan(config, policy)

    for row, modulus in enumerate(config.moduli):
        forward_root = int(plan.forward_radix_root_powers[row, 1])
        inverse_root = int(plan.inverse_radix_root_powers[row, 1])
        assert pow(forward_root, policy.radix, modulus) == 1
        assert pow(forward_root, policy.radix // 2, modulus) == modulus - 1
        assert (forward_root * inverse_root) % modulus == 1


@pytest.mark.gpu
@pytest.mark.parametrize(("preset", "backends"), EXACT_BACKEND_CASES)
def test_production_backend_families_match_reference_exactly(
    preset: Preset,
    backends: tuple[str, ...],
) -> None:
    _require_cuda()
    reference = CkksEngine(
        preset,
        device="cuda:0",
        ntt_backend=INDEXED_BACKEND,
        allow_sk_gen=False,
    )
    standard = _standard_rows(reference)
    coefficient_montgomery = standard.clone()
    reference.rns_runtime.to_montgomery_(coefficient_montgomery, include_p=True)

    expected_ntt = standard.clone()
    reference.rns_runtime.forward_to_montgomery_(expected_ntt, include_p=True)
    expected_centered = torch.where(
        standard > reference.rns_runtime.moduli[:, None] // 2,
        standard - reference.rns_runtime.moduli[:, None],
        standard,
    )
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    for backend in backends:
        engine = CkksEngine(
            preset,
            device="cuda:0",
            ntt_backend=backend,
            allow_sk_gen=False,
        )

        from_standard = standard.clone()
        engine.rns_runtime.forward_to_montgomery_(
            from_standard,
            include_p=True,
        )
        assert (
            torch.count_nonzero(
                torch.remainder(
                    from_standard - expected_ntt,
                    engine.rns_runtime.moduli[:, None],
                )
            ).item()
            == 0
        )

        from_montgomery = coefficient_montgomery.clone()
        engine.rns_runtime.forward_montgomery_(
            from_montgomery,
            include_p=True,
        )
        assert (
            torch.count_nonzero(
                torch.remainder(
                    from_montgomery - expected_ntt,
                    engine.rns_runtime.moduli[:, None],
                )
            ).item()
            == 0
        )

        to_standard = expected_ntt.clone()
        engine.rns_runtime.inverse_to_standard_(to_standard, include_p=True)
        assert torch.equal(to_standard, standard)

        to_standard_lazy = expected_ntt.clone()
        engine.rns_runtime.inverse_to_standard_lazy_(
            to_standard_lazy,
            include_p=True,
        )
        assert torch.equal(
            torch.remainder(
                to_standard_lazy,
                engine.rns_runtime.moduli[:, None],
            ),
            standard,
        )

        to_montgomery = expected_ntt.clone()
        engine.rns_runtime.inverse_montgomery_(
            to_montgomery,
            include_p=True,
        )
        assert (
            torch.count_nonzero(
                torch.remainder(
                    to_montgomery - coefficient_montgomery,
                    engine.rns_runtime.moduli[:, None],
                )
            ).item()
            == 0
        )

        to_centered = expected_ntt.clone()
        engine.rns_runtime.inverse_to_centered_(to_centered, include_p=True)
        assert torch.equal(to_centered, expected_centered)

        del engine
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("backend", "family"),
    [
        ("radix2_compact_group8_smem8", "compact_radix2"),
        ("radix4_compact", "fixed_radix"),
    ],
)
def test_native_ntt_schema_families_reject_invalid_twiddle_shapes(
    backend: str,
    family: str,
) -> None:
    _require_cuda()
    engine = CkksEngine(
        Preset.slots8192_scale40_levels7_int64,
        device="cuda:0",
        ntt_backend=backend,
        allow_sk_gen=False,
    )
    tables = engine.rns_runtime.ntt_tables
    operand = _standard_rows(engine)
    engine.rns_runtime.to_montgomery_(operand, include_p=True)

    if family == "compact_radix2":
        assert isinstance(tables, CompactRadix2Tables)
        with pytest.raises(RuntimeError, match="Compact twiddles"):
            ntt_ops.forward_ntt_montgomery_compact_grouped_smem_(
                operand,
                tables.forward_twiddles[:, :-1],
                engine.rns_runtime.rns_parameter_tensor,
                3,
            )
    else:
        assert isinstance(tables, CompactPowerOfTwoRadixTables)
        with pytest.raises(RuntimeError, match="outer twiddles"):
            ntt_ops.forward_ntt_montgomery_power_of_two_radix_compact_(
                operand,
                tables.forward_outer_twiddles[:, :-1],
                tables.forward_radix_root_powers,
                engine.rns_runtime.rns_parameter_tensor,
            )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("preset", "backend", "shared_memory_limits"),
    [
        (
            Preset.slots8192_scale40_levels7_int64,
            "radix4_compact",
            (1, 2, 4, 6, 8),
        ),
        (
            Preset.slots16384_scale40_levels16_int64,
            "radix8_compact",
            (2, 3, 5, 6, 8),
        ),
        (
            Preset.slots32768_scale40_levels34_int64,
            "radix16_compact",
            (3, 4, 7, 8),
        ),
    ],
)
def test_partial_shared_tail_bit_budgets_match_global_digits_exactly(
    preset: Preset,
    backend: str,
    shared_memory_limits: tuple[int, ...],
) -> None:
    _require_cuda()
    engine = CkksEngine(
        preset,
        device="cuda:0",
        ntt_backend=backend,
        allow_sk_gen=False,
    )
    tables = engine.rns_runtime.ntt_tables
    assert isinstance(tables, CompactPowerOfTwoRadixTables)
    operand = _standard_rows(engine)
    engine.rns_runtime.to_montgomery_(operand, include_p=True)
    operand = operand[:1].contiguous()
    parameters = engine.rns_runtime.rns_parameter_tensor[:, :1]
    forward_args = (
        tables.forward_outer_twiddles[:1],
        tables.forward_radix_root_powers[:1],
        parameters,
    )
    inverse_args = (
        tables.inverse_outer_twiddles[:1],
        tables.inverse_radix_root_powers[:1],
        parameters,
    )

    global_forward = operand.clone()
    ntt_diagnostic_ops.forward_ntt_montgomery_power_of_two_radix_compact_override_(
        global_forward,
        *forward_args,
        0,
    )
    global_inverse = global_forward.clone()
    ntt_diagnostic_ops.inverse_ntt_montgomery_power_of_two_radix_compact_override_(
        global_inverse,
        *inverse_args,
        0,
    )

    for shared_memory_log_n in shared_memory_limits:
        candidate_forward = operand.clone()
        ntt_diagnostic_ops.forward_ntt_montgomery_power_of_two_radix_compact_override_(
            candidate_forward,
            *forward_args,
            shared_memory_log_n,
        )
        assert torch.equal(candidate_forward, global_forward)

        candidate_inverse = candidate_forward.clone()
        ntt_diagnostic_ops.inverse_ntt_montgomery_power_of_two_radix_compact_override_(
            candidate_inverse,
            *inverse_args,
            shared_memory_log_n,
        )
        assert torch.equal(candidate_inverse, global_inverse)
