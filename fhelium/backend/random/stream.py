r"""Per-device ChaCha20 random streams for CKKS sampling.

A stream is an int64 seed Tensor of shape ``[5]`` on its device: four 64-bit
ChaCha20 key words followed by the 64-bit nonce. Every sampling call reads the
key stream selected by the current nonce and then increments the nonce in place
on the device, so consecutive calls, including replays of captured device work,
read disjoint streams. The samplers are the ``fhelium_random_ops`` native
operators; their sample-to-word assignment is documented with the C++ header
``fhelium/random.h``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch

from fhelium.native.wrapper import random_ops

_UINT64_MASK = (1 << 64) - 1


def _signed64(value: int) -> int:
    value &= _UINT64_MASK
    return value - (1 << 64) if value >= 1 << 63 else value


def _derived_key(seed: int, stream_id: int) -> list[int]:
    """Expand an integer seed and stream identifier into four key words."""
    digest = hashlib.sha256(
        b"fhelium.random\0"
        + int(seed).to_bytes(32, "little", signed=True)
        + int(stream_id).to_bytes(8, "little")
    ).digest()
    return [
        _signed64(int.from_bytes(digest[8 * i : 8 * i + 8], "little"))
        for i in range(4)
    ]


def advance_(seed: torch.Tensor, count: int = 1) -> None:
    """Advance the nonce of ``seed`` in place by ``count``."""
    seed[4:].add_(count)


def stochastic_round_(
    values: torch.Tensor, seed: torch.Tensor, *, word_stride: int = 1
) -> torch.Tensor:
    r"""Round float64 ``values`` randomly and advance the stream of ``seed``.

    Value $i$ of the flattened array rounds up with probability equal to its
    fractional magnitude, quantized to $2^{-32}$, using stream word
    $i \cdot$ ``word_stride``. A compact array holding every
    ``word_stride``-th coefficient of a full array therefore uses the words of
    the corresponding full-array coefficients. Returns int64 values.
    """
    rounded = random_ops.stochastic_round(
        seed, values.to(torch.float64), word_stride
    )
    advance_(seed)
    return rounded


class RandomStream:
    r"""ChaCha20 sampling stream owned by one device.

    Args:
        device: Device that holds the seed and receives samples.
        sigma: Discrete-Gaussian parameter of the error distribution.
        seed: Optional integer from which the key is derived together with
            ``stream_id``. Without it, the key comes from the operating
            system's cryptographic random source.
        nonce: Initial nonce. The caller owns nonce uniqueness across streams
            that share a derived key.
        stream_id: Identifier separating streams derived from one seed, such
            as the streams of different devices.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        sigma: float,
        seed: int | None = None,
        nonce: int | None = None,
        stream_id: int = 0,
    ) -> None:
        self.device = torch.device(device)
        words = (
            random_ops.generate_seed()
            if seed is None
            else torch.tensor(_derived_key(seed, stream_id) + [0])
        )
        words[4] = _signed64(0 if nonce is None else int(nonce))
        self.seed = words.to(self.device)
        self.sigma = float(sigma)
        self.cdt_table = random_ops.gaussian_cdt_table(self.sigma).to(
            self.device
        )

    def uniform_residues(
        self, shape: Sequence[int], rns_params: torch.Tensor
    ) -> torch.Tensor:
        """Sample residues uniformly modulo every prime of ``rns_params``.

        ``shape`` is ``[*batch, coefficient]``; the result has shape
        ``[*batch, limb, coefficient]``.
        """
        out = random_ops.sample_uniform_residues(
            self.seed, list(shape), rns_params
        )
        advance_(self.seed)
        return out

    def discrete_gaussian(
        self,
        shape: Sequence[int],
        rns_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample the error distribution as centered int64 values or residues."""
        out = random_ops.sample_discrete_gaussian(
            self.seed, list(shape), self.cdt_table, rns_params
        )
        advance_(self.seed)
        return out

    def add_discrete_gaussian_(
        self, residues: torch.Tensor, rns_params: torch.Tensor
    ) -> None:
        r"""Add error samples to ``residues`` in place.

        Entries of ``[*batch, limb, coefficient]`` residues in $[0, 2q_j)$
        become $(x + e) \bmod q_j$ in $[0, q_j)$, with one sample $e$ per
        coefficient shared by every limb.
        """
        random_ops.add_discrete_gaussian_(
            residues, self.seed, self.cdt_table, rns_params
        )
        advance_(self.seed)

    def ternary(
        self,
        shape: Sequence[int],
        nonzero_probability: float,
        rns_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Sample independent coefficients with $\Pr[\pm 1] = \rho / 2$."""
        out = random_ops.sample_ternary(
            self.seed, list(shape), nonzero_probability, rns_params
        )
        advance_(self.seed)
        return out

    def fixed_weight_ternary(
        self,
        shape: Sequence[int],
        weight: int,
        rns_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ternary rows with exactly ``weight`` nonzero coefficients."""
        out = random_ops.sample_fixed_weight_ternary(
            self.seed, list(shape), weight, rns_params
        )
        advance_(self.seed)
        return out

    def stochastic_round(
        self, values: torch.Tensor, *, word_stride: int = 1
    ) -> torch.Tensor:
        """Round ``values`` randomly with expectation equal to each value."""
        return stochastic_round_(values, self.seed, word_stride=word_stride)


__all__ = ["RandomStream", "advance_", "stochastic_round_"]
