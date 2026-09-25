"""Public-key CKKS encryption over supplied keys and arithmetic tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.ntt.operations import execute_transition
from fhelium.backend.rns.context import lift_integer_coefficients_exact
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import rns_ops
from fhelium.rng import Csprng
from ._resources import (
    RANDOM_STREAM_RESOURCE_KIND,
    RANDOM_STREAM_RESOURCE_SYMBOL,
)


def encrypt_tensor(
    coefficients: torch.Tensor,
    public_key: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    attributes: dict[str, object],
    rng: Csprng,
) -> torch.Tensor:
    """Sample encryption randomness and evaluate the two ciphertext components."""
    parameters = tensors["parameters"]
    forward = tuple(
        value for name, value in tensors.items() if name.startswith("forward_")
    )
    inverse = tuple(
        value for name, value in tensors.items() if name.startswith("inverse_")
    )
    policy = str(attributes["ntt_backend"])
    plaintext = lift_integer_coefficients_exact(
        coefficients,
        parameters[0],
        min_modulus=int(cast(int, attributes["min_modulus"])),
    )
    batch_shape = plaintext.shape[:-2]
    batch_size = batch_shape.numel()
    errors = rng.discrete_gaussian(repeats=2 * batch_size)[0].view(
        2, *batch_shape, plaintext.size(-1)
    )
    e0 = rns_ops.lift_centered_coefficients(errors[0], parameters[0])
    e1 = rns_ops.lift_centered_coefficients(errors[1], parameters[0])
    pte0 = rns_ops.add_lazy(plaintext, e0, parameters)
    start = int(cast(int, attributes["key_row_start"]))
    pk0, pk1 = public_key[:, start : start + parameters.size(1)]
    binary = rng.randint(amax=2, shift=0, repeats=batch_size)[0].view(
        *batch_shape, plaintext.size(-1)
    )
    v = rns_ops.lift_centered_coefficients(binary, parameters[0])
    execute_transition(
        "forward_to_montgomery_", (v, *forward), policy, in_place=True
    )
    products = torch.stack(
        (
            rns_ops.montgomery_mul(v, pk0, parameters),
            rns_ops.montgomery_mul(v, pk1, parameters),
        )
    )
    output_domain = str(attributes.get("output_domain", "coefficient"))
    if output_domain == "ntt":
        added = torch.stack((pte0, e1))
        execute_transition(
            "forward_to_montgomery_", (added, *forward), policy, in_place=True
        )
        return rns_ops.add_lazy(products, added, parameters)
    if output_domain != "coefficient":
        raise ValueError("Encryption output_domain is unsupported")
    execute_transition(
        "inverse_to_standard_lazy_", (products, *inverse), policy, in_place=True
    )
    return torch.stack(
        (
            rns_ops.add_standard(products[0], pte0, parameters),
            rns_ops.add_standard(products[1], e1, parameters),
        )
    )


@dataclass(frozen=True)
class NativeEncryptImplementation:
    """Execute encryption with data operands and a bound random stream."""

    name: str = "native-ckks-encrypt"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (ckks.EncryptOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return (
            ResourceRequirement(
                RANDOM_STREAM_RESOURCE_SYMBOL, RANDOM_STREAM_RESOURCE_KIND
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
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                inputs[2:],
                strict=True,
            )
        )
        return (
            encrypt_tensor(
                inputs[0],
                inputs[1],
                tensors,
                dict(invocation.attributes),
                cast(Csprng, resources[0].value),
            ),
        )


__all__ = ["NativeEncryptImplementation", "encrypt_tensor"]
