"""Registered CKKS decryption and centered Q-basis reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend._native_operands import named_tables
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops
from fhelium.backend.lowering import ExecBuilder


@dataclass(frozen=True)
class NativeDecryptImplementation:
    r"""Decrypt CT2 or CT3 payloads to centered Q coefficients.

    The phase $\sum_k c_k s^k$ is evaluated over the active Q rows in the
    input domain, inverted to coefficient standard residues, and
    reconstructed as the centered representative modulo the product of the
    supplied Q primes with mixed-radix digits.
    """

    name: str = "native-ckks-decrypt"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (ckks.DecryptOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        attributes = invocation.attributes
        return (
            ckks_ops.decrypt(
                inputs[0],
                inputs[1],
                named_tables(
                    "decryption", attributes["parameter_names"], inputs[2:]
                ),
                str(attributes["ntt_backend"]),
                int(cast(int, attributes["key_row_start"])),
                str(attributes["input_domain"]),
            ),
        )


__all__ = ["NativeDecryptImplementation"]
