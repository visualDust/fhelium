"""Registered CKKS public-key encryption over Tensor payloads."""

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
from fhelium.values import ModulusBasis, PublicKey
from fhelium.values import PolynomialDomain
from fhelium.ir.dialects import ckks
from fhelium.rng import Csprng

from ._resources import (
    PUBLIC_KEY_RESOURCE_KIND,
    RANDOM_STREAM_RESOURCE_KIND,
)


def _encrypt_rns_plaintext_tensor(
    plaintext_rns: torch.Tensor,
    *,
    public_key_data: torch.Tensor,
    public_key_basis: ModulusBasis,
    depth: int,
    rns_context: RnsContext,
    ntt_context: NttContext,
    rng: Csprng,
    output_domain: PolynomialDomain = "coefficient",
) -> torch.Tensor:
    """Encrypt standard RNS plaintext rows in the selected output domain."""

    include_p = public_key_basis == "QP"
    expected_prime_ids = rns_context.rns_layout.prime_ids(
        depth,
        include_p=include_p,
    )
    if plaintext_rns.ndim < 2 or plaintext_rns.size(-2) != len(
        expected_prime_ids
    ):
        raise ValueError(
            "Encryption plaintext must have active "
            "[*batch, limb, coeff] layout: "
            f"shape={tuple(plaintext_rns.shape)}, "
            f"prime_ids={expected_prime_ids}"
        )

    batch_shape = plaintext_rns.shape[:-2]
    batch_size = batch_shape.numel()
    e0e1 = rng.discrete_gaussian(repeats=2 * batch_size)[0].view(
        2, *batch_shape, rns_context.config.N
    )
    e0_tiled = rns_context.lift_centered_coefficients(
        e0e1[0], depth, include_p=include_p
    )
    e1_tiled = rns_context.lift_centered_coefficients(
        e0e1[1], depth, include_p=include_p
    )

    pte0 = rns_context.add_lazy(
        plaintext_rns,
        e0_tiled,
        include_p=include_p,
    )

    basis = rns_context.basis_parameters(depth, include_p=include_p)
    pk0 = public_key_data[0, basis.parameter_row_start : basis.parameter_row_stop]
    pk1 = public_key_data[1, basis.parameter_row_start : basis.parameter_row_stop]
    v = rng.randint(amax=2, shift=0, repeats=batch_size)[0].view(
        *batch_shape, rns_context.config.N
    )
    v = rns_context.lift_centered_coefficients(
        v,
        depth,
        include_p=include_p,
    )
    ntt_context.forward_to_montgomery_(v, include_p=include_p)
    products = torch.stack(
        (
            rns_context.montgomery_mul(v, pk0, include_p=include_p),
            rns_context.montgomery_mul(v, pk1, include_p=include_p),
        )
    )
    if output_domain == "ntt":
        added = torch.stack((pte0, e1_tiled))
        ntt_context.forward_to_montgomery_(added, include_p=include_p)
        return rns_context.add_lazy(products, added, include_p=include_p)
    if output_domain != "coefficient":
        raise ValueError("Encryption output_domain is unsupported")
    ntt_context.inverse_to_standard_lazy_(products, include_p=include_p)

    ct0 = rns_context.add_standard(products[0], pte0, include_p=include_p)
    ct1 = rns_context.add_standard(products[1], e1_tiled, include_p=include_p)
    return torch.stack((ct0, ct1), dim=0)


def encrypt_tensor(
    coefficients: torch.Tensor,
    public_key_data: torch.Tensor,
    *,
    public_key_basis: ModulusBasis,
    depth: int,
    rns_context: RnsContext,
    ntt_context: NttContext,
    rng: Csprng,
    output_domain: PolynomialDomain = "coefficient",
) -> torch.Tensor:
    """Encrypt integer coefficients with call-bound rns_context resources."""

    include_p = public_key_basis == "QP"
    plaintext_rns = rns_context.lift_integer_coefficients_exact(
        coefficients,
        depth,
        include_p=include_p,
        max_abs=int(torch.max(torch.abs(coefficients)).item()),
    )
    return _encrypt_rns_plaintext_tensor(
        plaintext_rns,
        public_key_data=public_key_data,
        public_key_basis=public_key_basis,
        depth=depth,
        rns_context=rns_context,
        ntt_context=ntt_context,
        rng=rng,
        output_domain=output_domain,
    )


@dataclass(frozen=True)
class NativeEncryptImplementation:
    """Execute public-key encryption with call-bound RNS, RNG, and key resources."""

    name: str = "native-ckks-encrypt"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (ckks.EncryptOp,)

    def resource_requirements(
        self,
        invocation: OperationInvocation,
    ) -> tuple[ResourceRequirement, ...]:
        return (
            ResourceRequirement("active-rns-parameters", RNS_RESOURCE_KIND),
            ResourceRequirement("active-ntt-plan", NTT_RESOURCE_KIND),
            ResourceRequirement(
                "ckks-random-stream", RANDOM_STREAM_RESOURCE_KIND
            ),
            ResourceRequirement(
                cast(str, invocation.attributes["key_symbol"]),
                PUBLIC_KEY_RESOURCE_KIND,
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
            raise ValueError("Encryption resources bind different contexts")
        rng = cast(Csprng, resources[2].value)
        key = cast(PublicKey, resources[3].value)
        return (
            encrypt_tensor(
                inputs[0],
                key.data,
                public_key_basis=key.modulus_basis,
                depth=int(cast(int, invocation.attributes["depth"])),
                rns_context=rns_context,
                ntt_context=ntt_context,
                rng=rng,
                output_domain=cast(
                    PolynomialDomain,
                    invocation.attributes.get("output_domain", "coefficient"),
                ),
            ),
        )


__all__ = ["NativeEncryptImplementation", "encrypt_tensor"]
