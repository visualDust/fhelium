"""Bootstrap entry coordinates and composable modulus-raising components."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Literal

import torch

from fhelium.eager import Engine
from fhelium.values import Ciphertext, KeySwitchKey


class ModulusRaising(Protocol):
    """Map source Q coordinates to coefficient/standard target Q ciphertexts."""

    def __call__(
        self, engine: Engine, value: Ciphertext, *, target_depth: int
    ) -> Ciphertext: ...


@dataclass(frozen=True)
class CenteredModRaise:
    r"""Extend each component's centered CRT representative into target Q.

    The lifted decryption phase contains the source modulus times an integer
    polynomial. The following Fourier and periodic reduction handle that term.
    """

    def __call__(
        self, engine: Engine, value: Ciphertext, *, target_depth: int = 0
    ) -> Ciphertext:
        return engine.modulus_raise(value, target_depth)


@dataclass(frozen=True)
class ScaledModRaise:
    r"""Raise the modulus and multiply its internal scale by an integer.

    For the wrapped result (c, Delta), integer payload multiplication followed
    by the corresponding scale transition produces (k*c, k*Delta), preserving
    its represented message.
    The following transform schedule consumes its existing Q groups; this
    component adds no modulus rows and performs no rescale. Existing decoded
    error is unchanged; the larger scale reduces the normalized contribution
    of subsequent key-switch errors. The multiplier
    is a mathematical scale choice shared by Eager and captured execution.
    """

    inner: ModulusRaising
    multiplier: int

    def __post_init__(self) -> None:
        if (
            type(self.multiplier) is not int
            or not 0 < self.multiplier < (1 << 63)
            or int(float(self.multiplier)) != self.multiplier
        ):
            raise ValueError(
                "Scale multiplier must be a positive exactly representable integer below 2^63"
            )

    def __call__(
        self, engine: Engine, value: Ciphertext, *, target_depth: int = 0
    ) -> Ciphertext:
        raised = self.inner(engine, value, target_depth=target_depth)
        product = engine.multiply_integer_scalar(raised, self.multiplier)
        return engine.reinterpret_at_scale(
            product, raised.scale * self.multiplier
        )


@dataclass(frozen=True)
class EncapsulatedModRaise:
    r"""Bind low and return hints for an intermediate secret relation.

    ``low_key`` encrypts P_low*E_j*s under t. ``return_key`` has one digit
    encrypting guard*P_return*t under s. Both are ordinary KeySwitchKey values;
    the component retains their live tensors, not either secret. Execution
    switches to t, centered-raises, then returns to s before Fourier transforms.
    Secret distributions, modulus choices and reducer range are caller choices.
    """

    low_key: KeySwitchKey
    return_key: KeySwitchKey
    low_p_moduli: tuple[int, ...]
    return_prime: int
    return_p_moduli: tuple[int, ...]
    moddown_rounding: Literal["nearest", "floor"] = "nearest"

    def __call__(
        self, engine: Engine, value: Ciphertext, *, target_depth: int = 0
    ) -> Ciphertext:
        return engine.encapsulated_modulus_raise(
            value,
            self.low_key,
            self.return_key,
            low_p_moduli=self.low_p_moduli,
            return_prime=self.return_prime,
            return_p_moduli=self.return_p_moduli,
            target_depth=target_depth,
            moddown_rounding=self.moddown_rounding,
        )


def reference_centered_basis_extend(
    residues: torch.Tensor,
    *,
    source_moduli: Sequence[int],
    target_moduli: Sequence[int],
    centered: bool = True,
) -> torch.Tensor:
    """Reconstruct source integers and reduce them into a target RNS basis.

    This reference applies CRT with Python integers to each coefficient for comparison with the native mixed-radix path. When ``centered`` is true, representatives greater than half the source modulus product are interpreted as negative before reduction into each target modulus.

    Args:
        residues: Tensor shaped ``[..., source_row, coefficient]``.
        source_moduli: Modulus corresponding to each source row.
        target_moduli: Moduli of the desired output rows.
        centered: Choose signed half-interval representatives instead of
            least nonnegative representatives.

    Returns:
        Tensor on the original device with shape
        ``[..., target_row, coefficient]`` and the same integer dtype.
    """

    if residues.ndim < 2:
        raise ValueError(
            "residues must have [..., source row, coefficient] shape"
        )
    if residues.size(-2) != len(source_moduli):
        raise ValueError("source row count does not match source_moduli")
    source_moduli = tuple(int(modulus) for modulus in source_moduli)
    target_moduli = tuple(int(modulus) for modulus in target_moduli)
    source_product = math.prod(source_moduli)
    crt_terms = []
    for modulus in source_moduli:
        partial = source_product // modulus
        crt_terms.append(partial * pow(partial, -1, modulus))

    source = (
        residues.detach()
        .cpu()
        .reshape(-1, len(source_moduli), residues.size(-1))
    )
    output = torch.empty(
        (source.size(0), len(target_moduli), source.size(-1)),
        dtype=residues.dtype,
    )
    for batch in range(source.size(0)):
        for coefficient in range(source.size(-1)):
            value = (
                sum(
                    int(source[batch, row, coefficient])
                    % source_moduli[row]
                    * crt_terms[row]
                    for row in range(len(source_moduli))
                )
                % source_product
            )
            if centered and value > source_product // 2:
                value -= source_product
            for row, modulus in enumerate(target_moduli):
                output[batch, row, coefficient] = value % modulus
    return output.reshape(
        *residues.shape[:-2], len(target_moduli), residues.size(-1)
    ).to(residues.device)


def _prepare_entry(
    engine: Engine,
    ciphertext: Ciphertext,
    *,
    target_scale: float,
) -> Ciphertext:
    r"""Lift the entry scale by an integer before terminal rescale.

    Choose $k=\max(1,\lceil M_d\Delta_0/\Delta_{in}\rceil)$ and replace
    $(c,\Delta_{in})$ with $(kc,k\Delta_{in})$. The integer multiplication
    preserves the represented message without quantizing an encoded identity.
    When $k=1$ the input is returned unchanged. The quotient scale after the
    following rescale is retained for the full-slot coordinate conversion.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        components=2,
    )
    engine.validate_ciphertext(ciphertext)
    multiplier = max(
        1,
        math.ceil(
            engine.config.rescale_divisor(ciphertext.depth)
            * target_scale
            / ciphertext.scale
        ),
    )
    if multiplier == 1:
        return ciphertext
    result = engine.multiply_integer_scalar(ciphertext, multiplier)
    return engine.reinterpret_at_scale(result, ciphertext.scale * multiplier)


def _rescale_to_structural_basis(
    engine: Engine,
    ciphertext: Ciphertext,
    *,
    source_depth: int,
) -> Ciphertext:
    r"""Rescale the prepared Bootstrap input into the terminal Q basis.

    The input occupies the penultimate depth. Nearest group rescale preserves
    the represented message and divides the pending scale by the group
    product. The terminal basis is then used for centered ModRaise.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        components=2,
    )
    engine.validate_ciphertext(ciphertext)
    expected_depth = source_depth - 1
    if ciphertext.depth != expected_depth:
        raise ValueError(
            'bootstrap entry rescale requires penultimate depth '
            f'{expected_depth}, got {ciphertext.depth}'
        )
    structural_value = engine.rescale_to_next_depth(
        ciphertext,
        rounding='nearest',
    )
    return structural_value
