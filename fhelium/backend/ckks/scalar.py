"""Native CKKS scalar arithmetic over parameter and random-state Tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns.context import RnsContext
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops, rns_ops
from fhelium.backend.lowering import ExecBuilder


def scalar_operands(
    context: RnsContext,
    prime_ids: tuple[int, ...],
    scalar: int | float,
    scalar_scale: float | None,
    rounding_state: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Supply row parameters or live rounding state without sampling noise."""
    parameters = context.rns_parameters_for_prime_ids(prime_ids)
    if scalar_scale is None:
        table = context.integer_scalar_parameters(int(scalar))
        start = prime_ids[0]
        rows = (
            table[start : start + len(prime_ids)]
            if prime_ids == tuple(range(start, start + len(prime_ids)))
            else table[list(prime_ids)]
        )
        return (parameters, rows), {}
    if rounding_state is None:
        raise ValueError(
            "Real-scalar preparation requires a rounding-state Tensor"
        )
    return (parameters, rounding_state), {}


@dataclass(frozen=True)
class NativeScalarArithmeticImplementation:
    r"""Add or multiply CKKS ciphertexts by scalars.

    Integer scalars multiply every component by their row residues. A real
    scalar $a$ at scale $\Delta$ is rounded stochastically to
    $\lfloor a\Delta \rceil$ with the rounding state, which then advances;
    addition adds it to the constant coefficient of component zero, and
    multiplication multiplies every component by it.
    """

    name: str = "native-ckks-scalar-arithmetic"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        ckks.AddScalarOp,
        ckks.MultiplyScalarOp,
        ckks.MultiplyIntegerScalarOp,
    )

    @staticmethod
    def tensor_requirements(operation, config):
        from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

        if operation.parameters:
            return {}, {}
        ids = operation.ciphertext.type.state.data.get("prime_ids")
        if not isinstance(ids, ArrayAttr) or not all(
            isinstance(item, IntegerAttr) for item in ids
        ):
            return None
        prime_ids = tuple(int(item.value.data) for item in ids)
        scalar = operation.scalar.value.data
        descriptions = {
            "rns_parameters": {
                "kind": "rns_parameters",
                "prime_ids": list(prime_ids),
            }
        }
        if isinstance(operation, ckks.MultiplyIntegerScalarOp):
            descriptions["integer_scalar"] = {
                "kind": "integer_scalar",
                "prime_ids": list(prime_ids),
                "scalar": scalar,
            }
            return {}, descriptions
        descriptions["rounding_state"] = {"kind": "rounding_state"}
        return {}, descriptions

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
        ciphertext, parameters, auxiliary = inputs
        if invocation.operation_type is ckks.MultiplyIntegerScalarOp:
            return (
                rns_ops.montgomery_mul_row_scalars_standard(
                    ciphertext, auxiliary, parameters
                ),
            )
        output = ckks_ops.scalar_arithmetic(
            ciphertext,
            parameters,
            auxiliary,
            # A HostScalar attribute stays a runtime input of the fragment.
            cast(float, invocation.attributes["scalar"]),
            cast(float, invocation.attributes["scalar_scale"]),
            invocation.operation_type is ckks.AddScalarOp,
        )
        return (output,)


__all__ = ["NativeScalarArithmeticImplementation"]
