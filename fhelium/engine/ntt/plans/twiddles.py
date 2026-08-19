"""Canonical negacyclic NTT root and compact-twiddle construction."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def _primitive_2nth_root(modulus: int, ring_dimension: int) -> int:
    transform_order = 2 * ring_dimension
    exponent = (modulus - 1) // transform_order
    for candidate in range(2, ring_dimension):
        root = pow(candidate, exponent, modulus)
        if pow(root, ring_dimension, modulus) != 1:
            return root
    raise RuntimeError(
        "Failed to find a primitive root of order "
        f"{transform_order} modulo {modulus}"
    )


def _power_series(root: int, length: int, modulus: int) -> list[int]:
    powers = [1]
    for _ in range(length - 1):
        powers.append((powers[-1] * root) % modulus)
    return powers


def _bit_reverse(value: int, bit_count: int) -> int:
    return int(f"{value:0{bit_count}b}"[::-1], 2)


def _bit_reversed_indices(log_ring_dimension: int) -> torch.Tensor:
    ring_dimension = 1 << log_ring_dimension
    return torch.tensor(
        [
            _bit_reverse(index, log_ring_dimension)
            for index in range(ring_dimension)
        ],
        dtype=torch.long,
    )


def build_compact_twiddles(
    moduli: Iterable[int],
    log_ring_dimension: int,
    dtype: torch.dtype,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build canonical bit-reversed forward and inverse rows per prime.

    These ``[prime, coefficient]`` rows are the mathematical source shared by
    compact kernels and by indexed-plan expansion. They do not
    encode a grouped execution policy.

    For row ``i`` with primitive $2N$-th root $\psi_i$, column ``k`` stores
    $\psi_i^{\operatorname{br}(k)}\bmod q_i$ for forward execution and its
    inverse for inverse execution. Rows follow the input ``moduli`` order
    exactly. Outputs use ``dtype`` on ``device`` in standard representation;
    callers convert the tables to Montgomery form before CUDA NTT.
    """

    ring_dimension = 1 << log_ring_dimension
    bit_reversed = _bit_reversed_indices(log_ring_dimension).to(device=device)
    forward_rows: list[torch.Tensor] = []
    inverse_rows: list[torch.Tensor] = []
    for modulus in moduli:
        forward_root = _primitive_2nth_root(modulus, ring_dimension)
        inverse_root = pow(forward_root, -1, modulus)
        forward_powers = torch.tensor(
            _power_series(forward_root, ring_dimension, modulus),
            dtype=dtype,
            device=device,
        )
        inverse_powers = torch.tensor(
            _power_series(inverse_root, ring_dimension, modulus),
            dtype=dtype,
            device=device,
        )
        forward_rows.append(forward_powers.index_select(0, bit_reversed))
        inverse_rows.append(inverse_powers.index_select(0, bit_reversed))
    return torch.stack(forward_rows), torch.stack(inverse_rows)


def expand_stage_twiddles(
    compact_twiddles: torch.Tensor,
    twiddle_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather standard twiddles into ``[prime, stage, butterfly]``.

    The functional result follows the prime rows, dtype, and device of
    ``compact_twiddles`` and does not alias either input.
    """

    if twiddle_indices.ndim != 2:
        raise ValueError("twiddle_indices must have shape [stage, butterfly]")
    stage_count, butterfly_count = twiddle_indices.shape
    flat_indices = twiddle_indices.reshape(-1).to(
        dtype=torch.long,
        device=compact_twiddles.device,
    )
    return compact_twiddles.index_select(1, flat_indices).reshape(
        compact_twiddles.size(0), stage_count, butterfly_count
    )


def _power_of_two_radix_outer_exponents(
    log_ring_dimension: int,
    radix_bits: int,
    *,
    inverse_execution: bool,
) -> torch.Tensor:
    r"""Return packed powers for every radix digit's outer twist.

    For a radix $R_d$ digit, each group stores $\beta^1$ through
    $\beta^{R_d-1}$. The digit-local cyclic NTT is then evaluated with a fixed
    primitive $R_d$-th root. Across the strict fixed-radix transform this
    packing contains exactly ``N - 1`` values.
    """

    ring_dimension = 1 << log_ring_dimension
    transform_order = 2 * ring_dimension
    if log_ring_dimension % radix_bits != 0:
        raise ValueError("Strict radix width must divide log_ring_dimension")
    exponents: list[int] = []
    for stage_start in range(0, log_ring_dimension, radix_bits):
        radix = 1 << radix_bits
        forward_stage_start = (
            log_ring_dimension - stage_start - radix_bits
            if inverse_execution
            else stage_start
        )
        group_count = 1 << forward_stage_start
        leaf_stage = forward_stage_start + radix_bits - 1
        for group in range(group_count):
            compact_index = (1 << leaf_stage) + group * (radix // 2)
            beta_exponent = _bit_reverse(compact_index, log_ring_dimension)
            exponents.extend(
                (beta_exponent * lane) % transform_order
                for lane in range(1, radix)
            )
    if len(exponents) != ring_dimension - 1:
        raise AssertionError(
            "A complete power-of-two radix outer-twiddle packing must contain N - 1 "
            f"values, got {len(exponents)} for N={ring_dimension}"
        )
    return torch.tensor(exponents, dtype=torch.long)


def build_power_of_two_radix_twiddles(
    moduli: Iterable[int],
    log_ring_dimension: int,
    radix: int,
    dtype: torch.dtype,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build genuine power-of-two radix outer twists and cyclic-root powers.

    The returned direction-specific outer tables have shape ``[prime, N-1]``.
    Each digit is represented as a twisted cyclic radix-4/8/16 transform,
    rather than as a list of radix-2 stage twiddles. Root-power tables have
    shape ``[prime, radix]`` and contain powers of the fixed
    primitive radix root used by the digit-local butterfly.

    Rows map exactly to the input ``moduli`` order. Every returned tensor uses
    integral ``dtype`` on ``device`` and standard residues; the runtime later
    converts all four tables to Montgomery representation in place.
    """

    if radix not in (4, 8, 16):
        raise ValueError("radix must be 4, 8, or 16")
    radix_bits = {4: 2, 8: 3, 16: 4}[radix]
    if log_ring_dimension % radix_bits != 0:
        raise ValueError("Strict radix width must divide log_ring_dimension")

    ring_dimension = 1 << log_ring_dimension
    transform_order = 2 * ring_dimension
    forward_exponents = _power_of_two_radix_outer_exponents(
        log_ring_dimension,
        radix_bits,
        inverse_execution=False,
    )
    inverse_exponents = _power_of_two_radix_outer_exponents(
        log_ring_dimension,
        radix_bits,
        inverse_execution=True,
    )
    radix_exponents = torch.arange(radix, dtype=torch.long) * (
        transform_order // radix
    )

    forward_outer_rows: list[torch.Tensor] = []
    inverse_outer_rows: list[torch.Tensor] = []
    forward_root_rows: list[torch.Tensor] = []
    inverse_root_rows: list[torch.Tensor] = []
    for modulus in moduli:
        forward_root = _primitive_2nth_root(modulus, ring_dimension)
        inverse_root = pow(forward_root, -1, modulus)
        forward_powers = torch.tensor(
            _power_series(forward_root, transform_order, modulus),
            dtype=dtype,
        )
        inverse_powers = torch.tensor(
            _power_series(inverse_root, transform_order, modulus),
            dtype=dtype,
        )
        forward_outer_rows.append(
            forward_powers.index_select(0, forward_exponents)
        )
        inverse_outer_rows.append(
            inverse_powers.index_select(0, inverse_exponents)
        )
        forward_root_rows.append(
            forward_powers.index_select(0, radix_exponents)
        )
        inverse_root_rows.append(
            inverse_powers.index_select(0, radix_exponents)
        )

    return (
        torch.stack(forward_outer_rows).to(device=device),
        torch.stack(inverse_outer_rows).to(device=device),
        torch.stack(forward_root_rows).to(device=device),
        torch.stack(inverse_root_rows).to(device=device),
    )
