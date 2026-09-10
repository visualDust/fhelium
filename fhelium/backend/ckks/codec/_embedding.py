"""CKKS slot and coefficient embeddings for fixed-context codecs.

The functions in this module implement the generator-3 and generator-5 slot
orders, the conjugate-symmetric negacyclic FFT embedding, and stochastic
coefficient quantization. They operate on tensors and store no runtime state.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import torch

from fhelium.rng import Csprng

_FFT_NORM = "forward"


def make_slot_tensor(
    message: Sequence[object] | torch.Tensor | complex | float | int,
    num_slots: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a dense ``[*batch, slot]`` tensor with a full slot axis."""

    if type(num_slots) is not int:
        raise TypeError("num_slots must be an integer")
    if num_slots <= 0 or num_slots & (num_slots - 1):
        raise ValueError("num_slots must be a positive power of two")
    if isinstance(message, (int, float, complex)):
        value = torch.full((num_slots,), message, device=device)
    elif isinstance(message, torch.Tensor):
        value = message
    else:
        value = torch.as_tensor(message, device=device)
    if value.ndim == 0:
        value = torch.full(
            (num_slots,),
            value.item(),
            dtype=value.dtype,
            device=device,
        )
    if value.size(-1) > num_slots:
        raise ValueError(
            f"Message slot extent {value.size(-1)} exceeds {num_slots}"
        )
    if value.size(-1) < num_slots:
        value = torch.nn.functional.pad(
            value,
            (0, num_slots - value.size(-1)),
            "constant",
            0,
        )
    return value


@cache
def _forward_generator_positions(
    ring_dimension: int,
    generator: int,
) -> tuple[int, ...]:
    if ring_dimension <= 0 or ring_dimension & (ring_dimension - 1):
        raise ValueError("ring_dimension must be a positive power of two")
    if generator not in {3, 5}:
        raise ValueError("generator must be 3 or 5")
    modulus = 2 * ring_dimension
    value = 1
    positions = []
    for _ in range(ring_dimension // 2):
        positions.append((value - 1) // 2)
        value = (value * generator) % modulus
    return tuple(positions)


def _circular_shift_permutation(N: int, shift: int = 1) -> torch.Tensor:
    half = torch.arange(N // 2, dtype=torch.int64)
    return torch.cat(
        (
            torch.roll(half, shifts=shift),
            torch.roll(half, shifts=-shift) + N // 2,
        )
    )


def _generator_permutation(N: int, generator: int) -> torch.Tensor:
    modulus = 2 * N
    values = torch.arange(modulus, dtype=torch.int64)
    return (generator * values) % modulus


def _fold_permutation(permutation: torch.Tensor) -> torch.Tensor:
    return (permutation[1::2] - 1) // 2


def _cycles(permutation: torch.Tensor) -> list[list[int]]:
    remaining = {
        index: value for index, value in enumerate(permutation.tolist())
    }
    cycles: list[list[int]] = []
    while remaining:
        first = next(iter(remaining))
        current = remaining[first]
        following = remaining[current]
        cycle = []
        while True:
            cycle.append(current)
            del remaining[current]
            current = following
            if following not in remaining:
                break
            following = remaining[following]
        cycles.append(cycle)
    return cycles


def _conjugate_permutation(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    first_cycles = _cycles(first)
    second_cycles = _cycles(second)
    if [len(cycle) for cycle in first_cycles] != [
        len(cycle) for cycle in second_cycles
    ]:
        raise ValueError("Permutation cycle structures differ")
    first_expanded = torch.tensor(
        [value for cycle in first_cycles for value in cycle],
        dtype=torch.int64,
    )
    second_expanded = torch.tensor(
        [value for cycle in second_cycles for value in cycle],
        dtype=torch.int64,
    )
    result = torch.zeros_like(first)
    result[second_expanded] = first_expanded
    return result


@cache
def _permutations(
    ring_dimension: int,
    device: str,
    generator: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if generator == 5:
        pre = torch.tensor(
            _forward_generator_positions(ring_dimension, generator),
            dtype=torch.int64,
        )
        post = torch.empty(ring_dimension, dtype=torch.int64)
        user = torch.arange(ring_dimension // 2, dtype=torch.int64)
        post[pre] = user
        post[ring_dimension - 1 - pre] = ring_dimension - 1 - user
        return pre.to(device), post.to(device)
    if generator != 3:
        raise ValueError("generator must be 3 or 5")
    circular = _circular_shift_permutation(ring_dimension)
    generated = _generator_permutation(ring_dimension, generator)
    folded = _fold_permutation(generated)
    post = _conjugate_permutation(circular, folded)
    pre = torch.argsort(post)[: ring_dimension // 2]
    return pre.to(device), post.to(device)


@cache
def _twister(ring_dimension: int, device: str) -> torch.Tensor:
    indices = torch.arange(ring_dimension, device=device, dtype=torch.float64)
    return torch.exp(-1j * torch.pi * indices / ring_dimension)


@cache
def _skewer(ring_dimension: int, device: str) -> torch.Tensor:
    indices = torch.arange(ring_dimension, device=device, dtype=torch.float64)
    return torch.exp(1j * torch.pi * indices / ring_dimension)


def inverse_embed_slots(
    message: torch.Tensor,
    *,
    device: torch.device,
    generator: int,
) -> torch.Tensor:
    """Map ``[*batch, slot]`` messages to unscaled real coefficients."""

    ring_dimension = message.size(-1) * 2
    pre, _ = _permutations(ring_dimension, str(device), generator)
    local = message.to(device)
    permuted = torch.zeros(
        (*local.shape[:-1], ring_dimension),
        dtype=local.dtype,
        device=device,
    )
    permuted[..., pre] = local
    conjugate_symmetric = permuted + permuted.conj().flip(-1)
    transformed = torch.fft.fft(conjugate_symmetric, norm=_FFT_NORM)
    return (transformed * _twister(ring_dimension, str(device))).real


def embed_coefficients(
    coefficients: torch.Tensor,
    *,
    generator: int,
) -> torch.Tensor:
    """Map real coefficients to the selected complex CKKS slot order."""

    ring_dimension = coefficients.size(-1)
    _, post = _permutations(
        ring_dimension,
        str(coefficients.device),
        generator,
    )
    recovered = torch.fft.ifft(
        coefficients * _skewer(ring_dimension, str(coefficients.device)),
        norm=_FFT_NORM,
    )
    result = torch.zeros_like(recovered)
    result[..., post] = recovered
    return result


def encode_slots(
    message: torch.Tensor,
    *,
    rng: Csprng,
    scale: float,
    device: torch.device,
    generator: int,
) -> torch.Tensor:
    """Return stochastic-rounded integer coefficients for ordered slots."""

    coefficients = inverse_embed_slots(
        message,
        device=device,
        generator=generator,
    )
    return rng.randround(coefficients * float(scale), dtype=torch.int64)


def decode_slots(
    coefficients: torch.Tensor,
    *,
    scale: float,
    generator: int,
) -> torch.Tensor:
    """Return complex slots from integer or bounded approximate coefficients."""

    return embed_coefficients(coefficients, generator=generator) / float(scale)
