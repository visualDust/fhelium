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
    """Hold device tables for centered reconstruction of the complete active Q basis."""

    source_prime_ids_by_depth: tuple[tuple[int, ...], ...]
    normalizers_by_depth: tuple[torch.Tensor, ...]
    propagation_by_depth: tuple[torch.Tensor, ...]
    device: torch.device

    @classmethod
    def create(cls, context: RnsContext) -> DecryptReconstructionResource:
        """Build reconstruction tables for one context-local RNS context."""

        source_prime_ids_by_depth = []
        normalizers_by_depth = []
        propagation_by_depth = []
        for depth in range(context.config.max_depth + 1):
            q_prime_ids = context.rns_layout.prime_ids(depth)
            source_prime_ids = tuple(q_prime_ids)
            source_prime_ids_by_depth.append(source_prime_ids)
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
                dtype=context.dtype,
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
            normalizers_by_depth.append(
                torch.tensor(
                    normalizers,
                    dtype=context.dtype,
                    device=context.device,
                )
            )
            propagation_by_depth.append(propagation)
        return cls(
            tuple(source_prime_ids_by_depth),
            tuple(normalizers_by_depth),
            tuple(propagation_by_depth),
            context.device,
        )


__all__ = [
    "DECRYPT_RECONSTRUCTION_RESOURCE_KIND",
    "PUBLIC_KEY_RESOURCE_KIND",
    "RANDOM_STREAM_RESOURCE_KIND",
    "SECRET_KEY_RESOURCE_KIND",
    "DecryptReconstructionResource",
]
