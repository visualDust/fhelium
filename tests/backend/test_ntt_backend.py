from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch

from fhelium.backend.assembly import create_builtin_operation_registry
from fhelium.backend.rns.format import RnsExecutionFormat
from fhelium.config import CkksConfig, Preset
from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    compatible_ntt_backends,
    resolve_ntt_backend_policy,
)
from fhelium.ir.dialects import ntt as ntt_dialect
from fhelium.backend.ntt.tables import (
    CompactPowerOfTwoRadixTables,
    CompactRadix2Tables,
)
from fhelium.eager import Engine
from fhelium.native.wrapper import ntt_diagnostic_ops, ntt_ops, rns_ops

INDEXED_BACKEND = "radix2_indexed"
RADIX2_PRODUCTION_BACKENDS = ("radix2_compact", "radix2_compact_shoup32")
FIXED_RADIX_CASES = (
    (Preset.slots8192_scale40_depth7, "radix4_compact"),
    (Preset.slots16384_scale40_depth16, "radix8_compact"),
    (Preset.slots32768_scale40_depth34, "radix16_compact"),
)
EXACT_BACKEND_CASES = (
    (
        Preset.slots8192_scale40_depth7,
        (*RADIX2_PRODUCTION_BACKENDS, "radix4_compact"),
    ),
    (Preset.slots16384_scale40_depth16, ("radix8_compact",)),
    (Preset.slots32768_scale40_depth34, ("radix16_compact",)),
)


def test_ntt_ir_operation_names_describe_both_endpoint_states() -> None:
    assert (
        ntt_dialect.CoefficientStandardToNttMontgomeryOp.name
        == "fhelium_ntt.coefficient_standard_to_ntt_montgomery"
    )
    assert (
        ntt_dialect.CoefficientMontgomeryToNttMontgomeryOp.name
        == "fhelium_ntt.coefficient_montgomery_to_ntt_montgomery"
    )
    assert (
        ntt_dialect.NttMontgomeryToCoefficientStandardOp.name
        == "fhelium_ntt.ntt_montgomery_to_coefficient_standard"
    )


def test_native_ntt_registration_uses_selected_executor_identity() -> None:
    registry = create_builtin_operation_registry(
        ntt_backend_name=INDEXED_BACKEND,
    )

    for operation_type in (
        ntt_dialect.CoefficientStandardToNttMontgomeryOp,
        ntt_dialect.CoefficientMontgomeryToNttMontgomeryOp,
        ntt_dialect.NttMontgomeryToCoefficientStandardOp,
        ntt_dialect.NttMontgomeryToCoefficientMontgomeryOp,
    ):
        assert registry.available(operation_type) == (INDEXED_BACKEND,)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _cpu_indexed_runtime(
    preset: Preset,
) -> tuple[
    CkksConfig, tuple[SimpleNamespace, torch.Tensor, torch.Tensor], torch.Tensor
]:
    config = CkksConfig.parse(preset)
    execution_format = RnsExecutionFormat.select(config.moduli)
    dtype = execution_format.dtype
    radix = 1 << execution_format.radix_bits
    forward_indices, forward_twiddles, inverse_indices, inverse_twiddles = (
        ntt_ops.indexed_radix2_plan(list(config.moduli), config.logN, dtype)
    )
    for indices in (forward_indices, inverse_indices):
        ntt_ops.validate_indexed_schedule(indices[0], indices[1], config.N)
    plan = SimpleNamespace(
        forward_indices=forward_indices, inverse_indices=inverse_indices
    )
    parameters = rns_ops.parameter_table(
        list(config.moduli), config.logN, config.scale_at(0), dtype
    )

    def to_montgomery(table: torch.Tensor) -> torch.Tensor:
        rows = [
            torch.tensor(
                [(int(value) * radix) % modulus for value in row.reshape(-1)],
                dtype=dtype,
            ).reshape_as(row)
            for row, modulus in zip(table, config.moduli, strict=True)
        ]
        return torch.stack(rows)

    return (
        config,
        (
            plan,
            to_montgomery(forward_twiddles),
            to_montgomery(inverse_twiddles),
        ),
        parameters,
    )


@pytest.mark.parametrize(
    "preset",
    [
        Preset.slots8192_scale30_depth9,
        Preset.slots8192_scale30_depth9,
    ],
)
def test_cpu_indexed_ntt_uses_the_same_exact_schema_and_representation(
    preset: Preset,
) -> None:
    config, tables, parameters = _cpu_indexed_runtime(preset)
    dtype = RnsExecutionFormat.select(config.moduli).dtype
    plan, forward_twiddles, inverse_twiddles = tables
    generator = torch.Generator().manual_seed(
        20260811 + config.total_modulus_bits
    )
    active_limb_count = 3
    standard = (
        torch.stack(
            [
                torch.randint(
                    0,
                    modulus,
                    (2, config.N),
                    dtype=dtype,
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
        config.moduli[:active_limb_count], dtype=dtype
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


def test_indexed_ntt_preparation_rejects_invalid_schedules_and_execution_checks_storage() -> (
    None
):
    config, tables, parameters = _cpu_indexed_runtime(
        Preset.slots8192_scale30_depth9
    )
    dtype = RnsExecutionFormat.select(config.moduli).dtype
    plan, forward_twiddles, _ = tables
    standard = torch.stack(
        [
            torch.arange(config.N, dtype=dtype) % modulus
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
        ntt_ops.validate_indexed_schedule(
            out_of_range, plan.forward_indices[1], config.N
        )

    duplicate = plan.forward_indices[0].clone()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(RuntimeError, match="partition every coefficient"):
        ntt_ops.validate_indexed_schedule(
            duplicate, plan.forward_indices[1], config.N
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


def _cuda_runtime(preset: Preset, ntt_backend: str) -> SimpleNamespace:
    engine = Engine(
        preset, ntt_backend=ntt_backend, allow_automatic_key_generation=False
    )
    return SimpleNamespace(
        config=engine.config,
        device=torch.device("cuda:0"),
        rns=engine.rns_context("cuda:0"),
        ntt=engine._ntt_context_for("cuda:0"),
    )


def _standard_rows(engine: SimpleNamespace) -> torch.Tensor:
    coefficients = torch.arange(
        engine.config.N,
        dtype=engine.rns.dtype,
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
    _, _, forward_roots, inverse_roots = ntt_ops.power_of_two_radix_twiddles(
        list(config.moduli),
        config.logN,
        policy.radix,
        RnsExecutionFormat.select(config.moduli).dtype,
    )

    for row, modulus in enumerate(config.moduli):
        forward_root = int(forward_roots[row, 1])
        inverse_root = int(inverse_roots[row, 1])
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
    reference = _cuda_runtime(preset, INDEXED_BACKEND)
    standard = _standard_rows(reference)
    coefficient_montgomery = standard.clone()
    reference.rns.to_montgomery_(coefficient_montgomery, include_p=True)

    expected_ntt = standard.clone()
    reference.ntt.forward_to_montgomery_(expected_ntt, include_p=True)
    expected_centered = torch.where(
        standard > reference.rns.moduli[:, None] // 2,
        standard - reference.rns.moduli[:, None],
        standard,
    )
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    for backend in backends:
        engine = _cuda_runtime(preset, backend)

        from_standard = standard.clone()
        engine.ntt.forward_to_montgomery_(
            from_standard,
            include_p=True,
        )
        assert (
            torch.count_nonzero(
                torch.remainder(
                    from_standard - expected_ntt,
                    engine.rns.moduli[:, None],
                )
            ).item()
            == 0
        )

        from_montgomery = coefficient_montgomery.clone()
        engine.ntt.forward_montgomery_(
            from_montgomery,
            include_p=True,
        )
        assert (
            torch.count_nonzero(
                torch.remainder(
                    from_montgomery - expected_ntt,
                    engine.rns.moduli[:, None],
                )
            ).item()
            == 0
        )

        to_standard = expected_ntt.clone()
        engine.ntt.inverse_to_standard_(to_standard, include_p=True)
        assert torch.equal(to_standard, standard)

        to_standard_lazy = expected_ntt.clone()
        engine.ntt.inverse_to_standard_lazy_(
            to_standard_lazy,
            include_p=True,
        )
        assert torch.equal(
            torch.remainder(
                to_standard_lazy,
                engine.rns.moduli[:, None],
            ),
            standard,
        )

        to_montgomery = expected_ntt.clone()
        engine.ntt.inverse_montgomery_(
            to_montgomery,
            include_p=True,
        )
        assert (
            torch.count_nonzero(
                torch.remainder(
                    to_montgomery - coefficient_montgomery,
                    engine.rns.moduli[:, None],
                )
            ).item()
            == 0
        )

        to_centered = expected_ntt.clone()
        engine.ntt.inverse_to_centered_(to_centered, include_p=True)
        assert torch.equal(to_centered, expected_centered)

        del engine
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("backend", "family"),
    [
        ("radix2_compact", "compact_radix2"),
        ("radix4_compact", "fixed_radix"),
    ],
)
def test_native_ntt_schema_families_reject_invalid_twiddle_shapes(
    backend: str,
    family: str,
) -> None:
    _require_cuda()
    engine = _cuda_runtime(Preset.slots8192_scale40_depth7, backend)
    tables = engine.ntt.ntt_tables
    operand = _standard_rows(engine)
    engine.rns.to_montgomery_(operand, include_p=True)

    if family == "compact_radix2":
        assert isinstance(tables, CompactRadix2Tables)
        with pytest.raises(RuntimeError, match="Compact twiddles"):
            ntt_ops.forward_ntt_montgomery_compact_(
                operand,
                tables.forward_twiddles[:, :-1],
                engine.rns.rns_parameter_tensor,
            )
    else:
        assert isinstance(tables, CompactPowerOfTwoRadixTables)
        with pytest.raises(RuntimeError, match="outer twiddles"):
            ntt_ops.forward_ntt_montgomery_power_of_two_radix_compact_(
                operand,
                tables.forward_outer_twiddles[:, :-1],
                tables.forward_radix_root_powers,
                engine.rns.rns_parameter_tensor,
            )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("preset", "backend", "shared_memory_limits"),
    [
        (
            Preset.slots8192_scale40_depth7,
            "radix4_compact",
            (1, 2, 4, 6, 8),
        ),
        (
            Preset.slots16384_scale40_depth16,
            "radix8_compact",
            (2, 3, 5, 6, 8),
        ),
        (
            Preset.slots32768_scale40_depth34,
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
    engine = _cuda_runtime(preset, backend)
    tables = engine.ntt.ntt_tables
    assert isinstance(tables, CompactPowerOfTwoRadixTables)
    operand = _standard_rows(engine)
    engine.rns.to_montgomery_(operand, include_p=True)
    operand = operand[:1].contiguous()
    parameters = engine.rns.rns_parameter_tensor[:, :1]
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
