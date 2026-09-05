"""Registered CKKS decryption and bounded coefficient reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ntt.resources import NTT_RESOURCE_KIND
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.resources import RNS_RESOURCE_KIND
from fhelium.values import ModulusBasis, PolynomialDomain, SecretKey
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import rns_ops

from ._resources import (
    DECRYPT_RECONSTRUCTION_RESOURCE_KIND,
    SECRET_KEY_RESOURCE_KIND,
    DecryptReconstructionResource,
)


def _decrypt_tensor_to_coefficient_standard_rns(
    ciphertext_data: torch.Tensor,
    secret_key_data: torch.Tensor,
    *,
    level: int,
    includes_p: bool,
    secret_key_basis: ModulusBasis,
    input_domain: PolynomialDomain,
    rns_context: RnsContext,
    ntt_context: NttContext,
) -> torch.Tensor:
    r"""Evaluate $\sum_j c_js^j$ and return coefficient standard RNS.

    NTT inputs form the complete phase before one inverse transform.
    Coefficient inputs transform each nonconstant component for multiplication
    by its NTT/Montgomery secret-key power and return that product before the
    coefficient-domain sum.
    """

    secret_data = secret_key_data[rns_context.level_row_starts[level] :]
    if not includes_p and secret_key_basis == "QP":
        secret_data = secret_data[: -rns_context.config.num_p_primes]
    if ciphertext_data.size(0) not in (2, 3):
        raise ValueError("Decryption requires a CT2 or CT3 payload")
    if input_domain not in ("coefficient", "ntt"):
        raise ValueError(
            "Decryption input_domain must be 'coefficient' or 'ntt'"
        )

    phase = ciphertext_data[0].clone()
    secret_power = secret_data
    for component in range(1, ciphertext_data.size(0)):
        value = ciphertext_data[component]
        if input_domain == "coefficient":
            value = value.clone()
            ntt_context.forward_to_montgomery_(
                value,
                include_p=includes_p,
            )
        product = rns_context.montgomery_mul(
            value,
            secret_power,
            include_p=includes_p,
        )
        if input_domain == "coefficient":
            ntt_context.inverse_to_standard_(
                product,
                include_p=includes_p,
            )
            phase = rns_context.add_standard(
                phase,
                product,
                include_p=includes_p,
            )
        else:
            phase = rns_context.add_lazy(
                phase,
                product,
                include_p=includes_p,
            )
        if component + 1 < ciphertext_data.size(0):
            secret_power = rns_context.montgomery_mul(
                secret_power,
                secret_data,
                include_p=includes_p,
            )
    if input_domain == "ntt":
        ntt_context.inverse_to_standard_(phase, include_p=includes_p)
    return phase


def _mixed_radix_is_above_half(
    digits: torch.Tensor,
    moduli: tuple[int, ...],
) -> torch.Tensor:
    """Compare mixed-radix digits with half the represented modulus."""

    half = 1
    for modulus in moduli:
        half *= modulus
    half //= 2
    half_digits = []
    for modulus in moduli:
        half, digit = divmod(half, modulus)
        half_digits.append(digit)
    equal = torch.ones_like(digits[..., 0, :], dtype=torch.bool)
    above = torch.zeros_like(equal)
    for row in range(len(moduli) - 1, -1, -1):
        above |= equal & (digits[..., row, :] > half_digits[row])
        equal &= digits[..., row, :] == half_digits[row]
    return above


def reconstruct_tail_q_coefficients_tensor(
    plaintext_rns: torch.Tensor,
    *,
    level: int,
    includes_p: bool,
    rns_context: RnsContext,
    reconstruction: DecryptReconstructionResource,
) -> torch.Tensor:
    """Reconstruct the centered trailing-Q class on the rns_context device."""

    q_prime_ids = rns_context.rns_layout.prime_ids(level)
    source_prime_ids = tuple(q_prime_ids[-2:])
    del includes_p
    source = plaintext_rns.narrow(
        -2,
        len(q_prime_ids) - len(source_prime_ids),
        len(source_prime_ids),
    ).clone()
    start = source_prime_ids[0]
    stop = source_prime_ids[-1] + 1
    source_params = rns_context.rns_parameter_tensor[:, start:stop]
    rns_ops.reduce_to_standard_(source, source_params)
    source_moduli = tuple(
        int(rns_context.montgomery_parameters.moduli[index])
        for index in source_prime_ids
    )
    if len(source_prime_ids) == 1:
        modulus = source_moduli[0]
        centered = source.squeeze(-2)
        return torch.where(
            centered > modulus // 2,
            centered - modulus,
            centered,
        ).to(torch.float64)

    if source_prime_ids != reconstruction.source_prime_ids:
        raise ValueError(
            "Decryption reconstruction table differs from active trailing Q"
        )
    normalizers = reconstruction.normalizers
    propagation = reconstruction.propagation
    lo, hi, neg_lo, neg_hi = source_params[1:5]

    def decompose(residues: torch.Tensor) -> torch.Tensor:
        digits = rns_ops.mixed_radix_decompose(
            residues,
            normalizers,
            propagation,
            lo,
            hi,
            neg_lo,
            neg_hi,
        )
        rns_ops.reduce_to_standard_(digits, source_params)
        return digits

    positive_digits = decompose(source)
    negative = _mixed_radix_is_above_half(positive_digits, source_moduli)
    moduli_tensor = torch.tensor(
        source_moduli,
        dtype=source.dtype,
        device=source.device,
    ).view(*([1] * (source.ndim - 2)), -1, 1)
    negated_source = torch.where(source == 0, source, moduli_tensor - source)
    negative_digits = decompose(negated_source)

    def reconstruct(digits: torch.Tensor) -> torch.Tensor:
        value = digits[..., -1, :].to(torch.float64)
        for row in range(len(source_moduli) - 2, -1, -1):
            value = value * source_moduli[row] + digits[..., row, :]
        return value

    positive_value = reconstruct(positive_digits)
    negative_value = reconstruct(negative_digits)
    return torch.where(negative, -negative_value, positive_value)


def decrypt_tensor(
    ciphertext_data: torch.Tensor,
    secret_key_data: torch.Tensor,
    *,
    level: int,
    ciphertext_basis: ModulusBasis,
    secret_key_basis: ModulusBasis,
    input_domain: PolynomialDomain,
    rns_context: RnsContext,
    ntt_context: NttContext,
    reconstruction: DecryptReconstructionResource,
) -> torch.Tensor:
    """Decrypt ciphertext and secret-key Tensor payloads to coefficients."""

    includes_p = ciphertext_basis == "QP"
    plaintext_rns = _decrypt_tensor_to_coefficient_standard_rns(
        ciphertext_data,
        secret_key_data,
        level=level,
        includes_p=includes_p,
        secret_key_basis=secret_key_basis,
        input_domain=input_domain,
        rns_context=rns_context,
        ntt_context=ntt_context,
    )
    return reconstruct_tail_q_coefficients_tensor(
        plaintext_rns,
        level=level,
        includes_p=includes_p,
        rns_context=rns_context,
        reconstruction=reconstruction,
    )


@dataclass(frozen=True)
class NativeDecryptImplementation:
    """Execute decryption with call-bound RNS and secret-key resources."""

    name: str = "native-ckks-decrypt"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (ckks.DecryptOp,)

    def resource_requirements(
        self,
        invocation: OperationInvocation,
    ) -> tuple[ResourceRequirement, ...]:
        return (
            ResourceRequirement("active-rns-parameters", RNS_RESOURCE_KIND),
            ResourceRequirement("active-ntt-plan", NTT_RESOURCE_KIND),
            ResourceRequirement(
                "ckks-decrypt-reconstruction",
                DECRYPT_RECONSTRUCTION_RESOURCE_KIND,
            ),
            ResourceRequirement(
                cast(str, invocation.attributes["key_symbol"]),
                SECRET_KEY_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        rns_context = cast(RnsContext, resources[0].value)
        ntt_context = cast(NttContext, resources[1].value)
        if ntt_context.rns_context is not rns_context:
            raise ValueError("Decryption resources bind different contexts")
        reconstruction = cast(
            DecryptReconstructionResource,
            resources[2].value,
        )
        key = cast(SecretKey, resources[3].value)
        return (
            decrypt_tensor(
                inputs[0],
                key.data,
                level=int(cast(int, invocation.attributes["level"])),
                ciphertext_basis=cast(
                    ModulusBasis,
                    invocation.attributes["modulus_basis"],
                ),
                secret_key_basis=key.modulus_basis,
                input_domain=cast(
                    PolynomialDomain,
                    invocation.attributes["input_domain"],
                ),
                rns_context=rns_context,
                ntt_context=ntt_context,
                reconstruction=reconstruction,
            ),
        )


__all__ = [
    "NativeDecryptImplementation",
    "decrypt_tensor",
    "reconstruct_tail_q_coefficients_tensor",
]
