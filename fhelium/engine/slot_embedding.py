"""Low-level CKKS slot/polynomial encoding helpers."""

from collections.abc import Sequence
from functools import cache

import torch

from fhelium.engine.galois import forward_slot_generator_positions
from fhelium.rng import Csprng

# CKKS slot encoding is part of the serialized mathematical representation,
# not a selectable execution policy. Keep one context-wide FFT normalization
# convention rather than allowing engines with the same context_id to attach
# different amplitudes to the same encoded polynomial.
_SLOT_FFT_NORM = "forward"


# ---------------------------------------------------------------
# Encoding and decoding.
# ---------------------------------------------------------------


def make_slot_tensor(
    m: Sequence | torch.Tensor | complex,
    num_slots: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Materialize the canonical ``[*batch, slot]`` message layout.

    A scalar is repeated across all $S$ slots. A tensor-like message with final
    extent at most $S$ preserves every leading batch axis and is padded with
    zeros on the right of the slot axis. Existing tensor dtype is preserved;
    Python/non-tensor inputs are inferred by PyTorch. ``device`` places scalar
    and non-tensor inputs, while an existing non-scalar tensor retains its
    current device until the embedding operation moves it.

    The function is functional for scalar/non-tensor inputs and for inputs
    requiring padding. A full-length tensor may be returned directly and can
    therefore alias the input.
    """

    if type(num_slots) is not int:
        raise TypeError("num_slots must be an integer")
    if num_slots <= 0 or num_slots & (num_slots - 1):
        raise ValueError("num_slots must be a positive power of two")
    if isinstance(m, (int, float, complex)):  # single value, repeat it
        m = torch.full((num_slots,), m, device=device)
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m, device=device)

    if isinstance(m, torch.Tensor):
        if m.ndim == 0:
            m = torch.full((num_slots,), m.item(), dtype=m.dtype, device=device)
        if m.size(-1) > num_slots:
            raise ValueError(
                f"Size of last dim {m.size(-1)} exceeds the number of slots "
                f"{num_slots}."
            )

        # Preserve every leading batch dimension and pad only the slot axis.
        if m.shape[-1] < num_slots:
            pad_size = num_slots - m.shape[-1]
            m = torch.nn.functional.pad(m, (0, pad_size), "constant", 0)

    return m


def encode_slots(
    m: torch.Tensor,
    rng: Csprng,
    scale: float = float(2**40),
    device: str | torch.device = "cpu",
    galois_generator: int = 3,
) -> torch.Tensor:
    r"""Encode canonical slots as stochastic-rounded integer coefficients.

    For $m\in\mathbb{C}^S$ in the slot order selected by
    ``galois_generator``, compute

    $$
    p_i=\operatorname{SRound}\!\left(
      \Delta\,\mathcal{E}^{-1}_g(m)_i
    \right),
    \qquad
    \mathbb{E}[\operatorname{SRound}(x)]=x.
    $$

    ``m`` has layout ``[*batch, slot]`` with final extent $S=N/2$. The result
    has layout ``[*batch, coefficient]`` with final extent $N$, uses the
    integral dtype returned by ``rng.randround``, and resides on ``device``.
    It is a coefficient-domain polynomial, not RNS data. The operation is
    functional and the output does not alias ``m``.
    """

    coefficients = inverse_embed_slots(
        m,
        device=device,
        galois_generator=galois_generator,
    )
    return rng.randround(coefficients * float(scale))


def inverse_embed_slots(
    m: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    galois_generator: int = 3,
) -> torch.Tensor:
    r"""Apply the unscaled inverse canonical embedding
    $\mathcal{E}^{-1}_g$.

    ``m`` is ``[*batch, slot]`` in FHElium's canonical generator-3 order or the
    conventional OpenFHE generator-5 order, with $S=N/2$. The implementation
    permutes slots into a conjugate-symmetric length-$N$ vector $z$, then uses

    $$
    a_k=\operatorname{Re}\!\left(
      e^{-\pi i k/N}\,\operatorname{FFT}_{\mathrm{forward}}(z)_k
    \right).
    $$

    PyTorch ``norm="forward"`` contributes the $1/N$ forward normalization;
    this is paired with the inverse transform in :func:`embed_coefficients`.
    The result is a binary64 real ``[*batch, coefficient]`` tensor on
    ``device``. Leading batch axes are preserved. It has not been scaled,
    quantized, or reduced into RNS, and it does not alias ``m``.
    """

    N = m.size(-1) * 2

    pre_perm, post_perm = prepost_perms(N, device, galois_generator)

    mm = m.to(device)
    mm = pre_permute(mm, pre_perm)
    twister = generate_twister(N, device)
    return m2poly(mm, twister)


def decode_slots(
    m: torch.Tensor,
    scale: float = float(2**40),
    galois_generator: int = 3,
) -> torch.Tensor:
    r"""Decode coefficient values with their actual CKKS scale.

    For ``m`` with layout ``[*batch, coefficient]`` and final extent $N$,

    $$
    m_{\mathrm{slots}}=\mathcal{E}_g(m)/\Delta.
    $$

    The result is complex binary64 ``[*batch, canonical_slot]`` with final
    extent $N$ on the input device; callers select the first $S=N/2$ semantic
    slots. ``m`` may contain integer coefficients or bounded binary64
    approximate decrypt coefficients. The operation is functional.
    """

    return embed_coefficients(
        m,
        galois_generator=galois_generator,
    ) / float(scale)


def embed_coefficients(
    m: torch.Tensor,
    *,
    galois_generator: int = 3,
) -> torch.Tensor:
    r"""Apply the unscaled canonical embedding $\mathcal{E}_g$.

    For a real ``[*batch, coefficient]`` tensor $a$ with final extent $N$,
    form

    $$
    z=\operatorname{IFFT}_{\mathrm{forward}}\!\left(
      a_k e^{\pi i k/N}
    \right)
    $$

    and permute $z$ into the selected canonical slot order. PyTorch
    ``norm="forward"`` makes the inverse transform unnormalized so this is the
    inverse of :func:`inverse_embed_slots`. The result is complex binary64,
    preserves all leading batch axes, remains on the input device, and does
    not alias ``m``.
    """

    N = m.size(-1)
    device = str(m.device)

    pre_perm, post_perm = prepost_perms(N, device, galois_generator)

    skewer = generate_skewer(N, device)

    return post_permute(poly2m(m, skewer), post_perm)


# ---------------------------------------------------------------
# Permutation.
# ---------------------------------------------------------------


def circular_shift_permutation(N: int, shift: int = 1) -> torch.Tensor:
    half = torch.arange(N // 2, dtype=torch.int64)
    left = torch.roll(half, shifts=shift)
    right = torch.roll(half, shifts=-shift) + N // 2
    return torch.cat([left, right])


def canon_permutation(
    N: int,
    k: int = 1,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Permutes the coefficients of the lattice basis that yields correctly the permutation
    of the decoded message.

    The canonical permutation is defined as mu_p(n) = pn mod M where p is coprime with M,
    where p=2*k+1.
    """
    M = 2 * N
    p = int(2 * k + 1)  # Make sure p is an integer.
    n = torch.arange(M, dtype=torch.int64)  # n starts from 0.
    pn = p * n % M
    if verbose:
        print(f"Canonical permutation for p={p} is\n{pn}")
    return pn


def fold_permutation(
    N: int,
    p: torch.Tensor,
    verbose: bool = False,
) -> torch.Tensor:
    """
    In application to crypto, we fold the FFT at Nyquist.

    Inverse FFT results in selection of alternating elements.
    Folding should correct the indices of the permutation according to the
    folding rule.

    For example, 1->0, 3->1, 5->2, and so on.
    """
    fold_p = (p[1::2] - 1) // 2
    if verbose:
        print(f"Folding\n{p}\nresulted in\n{fold_p}.")
    return fold_p


def conjugate_permutation(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """
    Conjugate permutations p and q by stacking p on top of q.

    Permutations p and q must share the same cycle structures.
    """
    # Calculate cycles.
    pc = permutation_cycles(p)
    qc = permutation_cycles(q)

    # Check if the cycle structures match.
    cs1 = [len(c) for c in pc]
    cs2 = [len(c) for c in qc]
    if cs1 != cs2:
        raise ValueError(
            "Cycle structures of permutations must match for a conjugate to exist"
        )

    # Expand cycles.
    pe = torch.tensor([i for c in pc for i in c], dtype=torch.int64)
    qe = torch.tensor([i for c in qc for i in c], dtype=torch.int64)

    # Move slots.
    r = torch.zeros_like(p)
    r[qe] = pe

    # Return.
    return r


def permutation_cycles(perm: torch.Tensor) -> list[list[int]]:
    """
    Transform a plain permutation into a composition of cycles.
    """
    pi = {i: value for i, value in enumerate(perm.tolist())}
    cycles = []
    while pi:
        elem0 = next(iter(pi))  # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break
        cycles.append(cycle)
    return cycles


def inverse_permutation(
    p: torch.Tensor,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Calculates the inverse permutation.
    """
    ip = torch.argsort(p)
    if verbose:
        print(f"The inverse of permutation\n{p}\nis\n{ip}.")
    return ip


# ---------------------------------------------------------------
# Negacyclic fft.
# ---------------------------------------------------------------


def expand2conjugate(m: torch.Tensor) -> torch.Tensor:
    return torch.concat([m, torch.flip(torch.conj(m), dims=(-1,))], dim=-1)


@cache
def generate_twister(
    N: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    expr = (
        -1j * torch.pi * torch.arange(N, device=device, dtype=torch.float64) / N
    )
    return torch.exp(expr)


@cache
def generate_skewer(
    N: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    expr = (
        1j * torch.pi * torch.arange(N, device=device, dtype=torch.float64) / N
    )
    skew = torch.exp(expr)
    return skew


def m2poly(m: torch.Tensor, twister: torch.Tensor) -> torch.Tensor:
    """
    m is the message and this function turns the message into
    polynomial coefficients.
    The message must be expanded mirrored in conjugacy.
    """

    # Run fft and multiply twister.
    ffted = torch.fft.fft(m, norm=_SLOT_FFT_NORM)

    # Twist.
    twisted = ffted * twister

    # Return the real part.
    return twisted.real


def poly2m(poly: torch.Tensor, skewer: torch.Tensor) -> torch.Tensor:
    """
    poly is the polynomial coefficients and this function turns the coefficients
    into a plain message.
    """

    # Multiply skewer.
    t = poly * skewer

    # Recover.
    recovered = torch.fft.ifft(t, norm=_SLOT_FFT_NORM)

    # Return the real part.
    return recovered


# ---------------------------------------------------------------
# Utilities.
# ---------------------------------------------------------------


@cache
def prepost_perms(
    N: int,
    device: str | torch.device = "cpu",
    galois_generator: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    circ_shift = circular_shift_permutation(N)
    canon_perm = canon_permutation(N, k=(galois_generator - 1) // 2)
    fold_perm = fold_permutation(N, canon_perm)
    if galois_generator == 5:
        # Match the conventional generator-5 slot order used by OpenFHE rather
        # than conjugating it to FHElium's generator-3 order.
        pre_perm = torch.tensor(
            forward_slot_generator_positions(N, 5),
            dtype=torch.int64,
        )
        post_perm = torch.empty(N, dtype=torch.int64)
        user = torch.arange(N // 2, dtype=torch.int64)
        post_perm[pre_perm] = user
        post_perm[N - 1 - pre_perm] = N - 1 - user
        return pre_perm.to(device), post_perm.to(device)
    post_perm = conjugate_permutation(circ_shift, fold_perm)
    pre_perm = inverse_permutation(post_perm)[: N // 2]

    return pre_perm.to(device), post_perm.to(device)


def pre_permute(m: torch.Tensor, pre_perm: torch.Tensor) -> torch.Tensor:
    """
    Input m must be a torch tensor.
    """
    N = m.size(-1)
    permed_m = torch.zeros(
        (*m.shape[:-1], N * 2), dtype=m.dtype, device=m.device
    )
    permed_m[..., pre_perm] = m
    conj_permed_m = permed_m + permed_m.conj().flip(-1)
    return conj_permed_m


def post_permute(m: torch.Tensor, post_perm: torch.Tensor) -> torch.Tensor:
    """
    Input m must be a torch tensor.
    """
    permed_m = torch.zeros_like(m)
    permed_m[..., post_perm] = m
    return permed_m
