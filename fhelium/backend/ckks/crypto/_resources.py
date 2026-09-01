"""Concrete random-stream and key resources for CKKS cryptography."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhelium.backend.rns.context import RnsContext

RANDOM_STREAM_RESOURCE_KIND = "ckks-random-stream"
DECRYPT_RECONSTRUCTION_RESOURCE_KIND = "ckks-decrypt-reconstruction"
PUBLIC_KEY_RESOURCE_KIND = "ckks-public-key"
SECRET_KEY_RESOURCE_KIND = "ckks-secret-key"


@dataclass(frozen=True)
class DecryptReconstructionResource:
    """Hold device tables for bounded trailing-Q reconstruction."""

    source_prime_ids: tuple[int, ...]
    normalizers: torch.Tensor
    propagation: torch.Tensor
    device: torch.device

    @classmethod
    def create(cls, context: RnsContext) -> DecryptReconstructionResource:
        """Build reconstruction tables for one context-local RNS context."""

        source_prime_ids = tuple(context.rns_layout.prime_ids(0)[-2:])
        source_moduli = [
            int(context.montgomery_parameters.moduli[index])
            for index in source_prime_ids
        ]
        prefix_products = []
        product = source_moduli[0]
        for index in range(len(source_moduli) - 1):
            if index:
                product *= source_moduli[index]
            prefix_products.append(product)
        normalizers = []
        propagation = torch.zeros(
            (len(source_moduli) - 1, len(source_moduli)),
            dtype=context.config.torch_dtype,
            device=context.device,
        )
        for component_index, prefix in enumerate(prefix_products):
            next_modulus = source_moduli[component_index + 1]
            normalizers.append(
                (
                    pow(prefix, -1, next_modulus)
                    * context.montgomery_parameters.R
                )
                % next_modulus
            )
            for target_index in range(
                component_index + 2,
                len(source_moduli),
            ):
                propagation[component_index, target_index] = (
                    prefix * context.montgomery_parameters.R
                ) % source_moduli[target_index]
        return cls(
            source_prime_ids,
            torch.tensor(
                normalizers,
                dtype=context.config.torch_dtype,
                device=context.device,
            ),
            propagation,
            context.device,
        )


__all__ = [
    "DECRYPT_RECONSTRUCTION_RESOURCE_KIND",
    "PUBLIC_KEY_RESOURCE_KIND",
    "RANDOM_STREAM_RESOURCE_KIND",
    "SECRET_KEY_RESOURCE_KIND",
    "DecryptReconstructionResource",
]
