"""Integer input construction and analytic oracles for complete RNS/NTT calls."""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig


def analytic_ntt_rows(config: CkksConfig, radix: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return $Rz_k$ and $Rz_k^2$ for the bit-reversed odd roots of each Q prime.

    With $z_k=\psi^{2\operatorname{br}(k)+1}$, these rows evaluate the
    monomials $X$ and $X^2$. Host integer multiplication constructs the oracle
    without invoking an NTT or a Montgomery kernel.
    """
    n = 1 << config.logN
    reverse = [int(f"{k:0{config.logN}b}"[::-1], 2) for k in range(n)]
    linear, square = [], []
    for q in config.q_moduli:
        exponent = (q - 1) // (2 * n)
        generator = 2
        while True:
            root = pow(generator, exponent, q)
            if pow(root, n, q) == q - 1:
                break
            generator += 1
        step = root * root % q
        odd_roots = []
        value = root
        for _ in range(n):
            odd_roots.append(value)
            value = value * step % q
        ordered = [odd_roots[k] for k in reverse]
        linear.append(torch.tensor([z * radix % q for z in ordered], dtype=torch.int64))
        square.append(torch.tensor([z * z * radix % q for z in ordered], dtype=torch.int64))
    return torch.stack(linear), torch.stack(square)


def ntt_operands(
    moduli: tuple[int, ...],
    radix: int,
    linear: torch.Tensor,
    square: torch.Tensor,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Construct $a_b(X)=(b+1)+X-X^2$ and its analytic Montgomery NTT."""
    q = torch.tensor(moduli, dtype=torch.int64).view(1, -1, 1)
    members = torch.arange(1, batch + 1, dtype=torch.int64).view(-1, 1, 1)
    coefficient = torch.zeros((batch, len(moduli), linear.shape[-1]), dtype=torch.int64)
    coefficient[:, :, 0] = members[:, :, 0]
    coefficient[:, :, 1] = 1
    coefficient[:, :, 2] = q[0, :, 0] - 1
    r = torch.tensor([radix % modulus for modulus in moduli], dtype=torch.int64).view(1, -1, 1)
    transformed = (members * r + linear[None] - square[None]).remainder(q)
    return coefficient, transformed


def rns_operands(
    moduli: tuple[int, ...], n: int, batch: int, radix: int, *, multiply: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build dense full-width residue inputs with an integer closed-form result.

    The fixed scale-50 suite primes and coefficients below $N+31B$ keep every
    oracle product inside signed int64. Montgomery multiplication uses stored
    coordinates $a=q-k$ and $b=tR\bmod q$, hence output $-kt\bmod q$.
    """
    q = torch.tensor(moduli, dtype=torch.int64).view(1, -1, 1)
    k = torch.arange(n, dtype=torch.int64).view(1, 1, -1)
    k = k + 31 * torch.arange(batch, dtype=torch.int64).view(-1, 1, 1) + 1
    lhs = q - k
    if multiply:
        t = k.remainder(17) + 1
        r = torch.tensor([radix % modulus for modulus in moduli], dtype=torch.int64).view(1, -1, 1)
        rhs = (t * r).remainder(q)
        expected = (-k * t).remainder(q)
    else:
        rhs = (3 * k + 1).expand_as(lhs).contiguous()
        expected = (2 * k + 1).expand_as(lhs).contiguous()
    return lhs, rhs, expected