"""Public-key CKKS encryption over supplied keys and arithmetic tables."""

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
from fhelium.backend.random.stream import RandomStream
from fhelium.backend.lowering import ExecBuilder
from ._resources import (
    RANDOM_STREAM_RESOURCE_KIND,
    RANDOM_STREAM_RESOURCE_SYMBOL,
)


@dataclass(frozen=True)
class NativeEncryptImplementation:
    r"""Encrypt integer plaintext coefficients under a public key.

    For public key $(k_0, k_1)$, plaintext $m$, ternary mask $v$ with
    $\Pr[\pm 1] = 1/4$, and discrete-Gaussian errors $e_0, e_1$ drawn from the
    bound random stream in the order $e_0, v, e_1$, the result is
    $(v k_0 + m + e_0,\; v k_1 + e_1)$ over the rows of ``parameters``. The
    mask follows the ZO(0.5) law of Cheon, Kim, Kim, and Song, "Homomorphic
    Encryption for Arithmetic of Approximate Numbers", Section 3.4. With the
    public-key error $e_{pk} = k_0 + k_1 s$, the encryption error is
    $v e_{pk} + e_0 + e_1 s$. The stream advances by three nonces.
    """

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

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        attributes = invocation.attributes
        stream = cast(RandomStream, resources[0].value)
        seed = builder.state(stream, "seed")
        ciphertext = ckks_ops.encrypt(
            inputs[0],
            inputs[1],
            named_tables("ring", attributes["parameter_names"], inputs[2:]),
            seed,
            builder.state(stream, "cdt_table"),
            str(attributes["ntt_backend"]),
            int(cast(int, attributes["key_row_start"])),
            str(attributes.get("output_domain", "coefficient")),
        )
        return (ciphertext,)


__all__ = ["NativeEncryptImplementation"]
