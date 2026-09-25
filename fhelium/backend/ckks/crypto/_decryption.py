"""Registered CKKS decryption and centered Q-basis reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import rns_ops


def _decrypt_tensor_to_coefficient_standard_rns(
    ciphertext_data: torch.Tensor,
    secret_key_data: torch.Tensor,
    *,
    parameters: torch.Tensor,
    forward: tuple[torch.Tensor, ...],
    inverse: tuple[torch.Tensor, ...],
    key_row_start: int,
    input_domain: str,
    ntt_backend: str,
) -> torch.Tensor:
    """Evaluate the CT2 or CT3 phase using supplied arithmetic tables."""
    from fhelium.backend.ntt.operations import execute_transition

    secret_data = secret_key_data[
        key_row_start : key_row_start + parameters.size(1)
    ]
    if ciphertext_data.size(0) not in (2, 3):
        raise ValueError("Decryption requires a CT2 or CT3 payload")
    if input_domain not in ("coefficient", "ntt"):
        raise ValueError("Decryption requires a coefficient or NTT input")
    phase = ciphertext_data[0].clone()
    secret_power = secret_data
    for component in range(1, ciphertext_data.size(0)):
        value = ciphertext_data[component]
        if input_domain == "coefficient":
            value = execute_transition(
                "forward_to_montgomery_",
                (value, *forward),
                ntt_backend,
                in_place=False,
            )
        product = rns_ops.montgomery_mul(value, secret_power, parameters)
        if input_domain == "coefficient":
            execute_transition(
                "inverse_to_standard_",
                (product, *inverse),
                ntt_backend,
                in_place=True,
            )
            phase = rns_ops.add_standard(phase, product, parameters)
        else:
            phase = rns_ops.add_lazy(phase, product, parameters)
        if component + 1 < ciphertext_data.size(0):
            secret_power = rns_ops.montgomery_mul(
                secret_power, secret_data, parameters
            )
    if input_domain == "ntt":
        execute_transition(
            "inverse_to_standard_",
            (phase, *inverse),
            ntt_backend,
            in_place=True,
        )
    return phase


def _mixed_radix_is_above_half(
    digits: torch.Tensor, half_digits: torch.Tensor
) -> torch.Tensor:
    """Compare mixed-radix digits with the supplied half-product digits."""
    equal = torch.ones_like(digits[..., 0, :], dtype=torch.bool)
    above = torch.zeros_like(equal)
    for row in range(half_digits.numel() - 1, -1, -1):
        above |= equal & (digits[..., row, :] > half_digits[row])
        equal &= digits[..., row, :] == half_digits[row]
    return above


def reconstruct_q_coefficients_tensor(
    plaintext_rns: torch.Tensor,
    source_params: torch.Tensor,
    normalizers: torch.Tensor,
    propagation: torch.Tensor,
    half_digits: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the centered class modulo the complete supplied Q product."""
    count = source_params.size(1)
    source = plaintext_rns[..., :count, :].clone()
    rns_ops.reduce_to_standard_(source, source_params)
    source_moduli = source_params[0] // 2
    if count == 1:
        modulus = source_moduli[0]
        centered = source.squeeze(-2)
        return torch.where(
            centered > modulus // 2, centered - modulus, centered
        ).to(torch.float64)
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
    negative = _mixed_radix_is_above_half(positive_digits, half_digits)
    moduli_tensor = source_moduli.view(*([1] * (source.ndim - 2)), -1, 1)
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
    ciphertext: torch.Tensor,
    secret_key: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    attributes: dict[str, object],
) -> torch.Tensor:
    """Decrypt Tensor payloads and reconstruct their centered Q coefficients."""
    forward = tuple(
        value for name, value in tensors.items() if name.startswith("forward_")
    )
    inverse = tuple(
        value for name, value in tensors.items() if name.startswith("inverse_")
    )
    phase = _decrypt_tensor_to_coefficient_standard_rns(
        ciphertext,
        secret_key,
        parameters=tensors["parameters"],
        forward=forward,
        inverse=inverse,
        key_row_start=int(cast(int, attributes["key_row_start"])),
        input_domain=str(attributes["input_domain"]),
        ntt_backend=str(attributes["ntt_backend"]),
    )
    return reconstruct_q_coefficients_tensor(
        phase,
        tensors["reconstruction_parameters"],
        tensors["normalizers"],
        tensors["propagation"],
        tensors["half_digits"],
    )


@dataclass(frozen=True)
class NativeDecryptImplementation:
    """Execute decryption with Tensor key, transform, and reconstruction operands."""

    name: str = "native-ckks-decrypt"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (ckks.DecryptOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources, in_place
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                inputs[2:],
                strict=True,
            )
        )
        return (
            decrypt_tensor(
                inputs[0], inputs[1], tensors, dict(invocation.attributes)
            ),
        )


__all__ = [
    "NativeDecryptImplementation",
    "decrypt_tensor",
    "reconstruct_q_coefficients_tensor",
]
