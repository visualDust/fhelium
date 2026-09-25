"""Native implementations of logical RNS operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops, rns_ops


class _TensorOperandImplementation:
    """Receive numerical data as ordinary Tensor operands."""

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()


@dataclass(frozen=True)
class NativeRnsLinearImplementation(_TensorOperandImplementation):
    """Execute standard-range RNS add, subtract, and negate operations."""

    name: str = "native-rns-linear"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.AddStandardOp,
        rns.SubtractStandardOp,
        rns.NegateStandardOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        lhs, parameters = values[0], values[-1]
        if invocation.operation_type is rns.NegateStandardOp:
            result = lhs if in_place else lhs.clone()
            moduli = parameters[0] // 2
            result.neg_().remainder_(
                moduli.view(*([1] * (result.ndim - 2)), -1, 1)
            )
            return (result,)
        rhs = values[1]
        operation = (
            rns_ops.add_standard
            if invocation.operation_type is rns.AddStandardOp
            else rns_ops.sub_standard
        )
        mutate = (
            rns_ops.add_standard_
            if invocation.operation_type is rns.AddStandardOp
            else rns_ops.sub_standard_
        )
        try:
            lhs.view(-1, lhs.size(-2), lhs.size(-1))
            rhs.view(-1, rhs.size(-2), rhs.size(-1))
        except RuntimeError:
            if in_place:
                for index in range(lhs.size(0)):
                    mutate(lhs[index], rhs[index], parameters)
                return (lhs,)
            return (
                torch.stack(
                    tuple(
                        operation(a, b, parameters)
                        for a, b in zip(lhs, rhs, strict=True)
                    )
                ),
            )
        if in_place:
            mutate(lhs, rhs, parameters)
            return (lhs,)
        return (operation(lhs, rhs, parameters),)


@dataclass(frozen=True)
class NativeBatchSumImplementation(_TensorOperandImplementation):
    """Reduce one leading batch axis with standard-residue modular addition."""

    name: str = "native-batch-sum"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.SumStandardBatchOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources, in_place
        source, parameters = values
        dim = cast(int, invocation.attributes["dim"])
        return (rns_ops.sum_standard_batch(source, dim, parameters),)


@dataclass(frozen=True)
class NativeRnsTransitionImplementation(_TensorOperandImplementation):
    """Execute residue conversion, depth restriction, and scale metadata ops."""

    name: str = "native-rns-transition"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.StandardToMontgomeryOp,
        rns.MontgomeryToStandardOp,
        rns.RestrictDepthOp,
        rns.ReinterpretScaleOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        source = values[0]
        if invocation.operation_type in {
            rns.StandardToMontgomeryOp,
            rns.MontgomeryToStandardOp,
        }:
            parameters = values[1]
            data = source if in_place else source.clone()
            if invocation.operation_type is rns.StandardToMontgomeryOp:
                rns_ops.to_montgomery_(data, parameters)
            else:
                rns_ops.from_montgomery_(data, parameters)
            return (data,)
        if invocation.operation_type is rns.RestrictDepthOp:
            expected_prime_ids = invocation.result_prime_ids[0]
            if expected_prime_ids is None:
                raise ValueError(
                    "Depth restriction requires a physical result row count"
                )
            result_rows = len(expected_prime_ids)
            source_rows = source.size(-2)
            if not 0 < result_rows <= source_rows:
                raise ValueError(
                    "Depth restriction result rows must be within the source "
                    "Tensor extent"
                )
            selected = source.narrow(
                -2,
                source_rows - result_rows,
                result_rows,
            )
            data = selected if in_place else selected.clone()
            return (data,)
        return (source if in_place else source.clone(),)


@dataclass(frozen=True)
class NativePlaintextArithmeticImplementation(_TensorOperandImplementation):
    """Execute prepared plaintext addition and multiplication."""

    name: str = "native-plaintext-arithmetic"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.AddPlaintextOp,
        rns.MultiplyPlaintextOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        ciphertext, plaintext, parameters = values
        if invocation.operation_type is rns.AddPlaintextOp:
            if (
                invocation.attributes.get("polynomial_domain", "coefficient")
                == "ntt"
            ):
                component = rns_ops.add_lazy(
                    ciphertext[0], plaintext, parameters
                )
                if in_place:
                    ciphertext[0].copy_(component)
                    return (ciphertext,)
            elif in_place:
                ckks_ops.add_prepared_plaintext_component_(
                    ciphertext[0], plaintext, parameters
                )
                return (ciphertext,)
            else:
                component = ckks_ops.add_prepared_plaintext_component(
                    ciphertext[0], plaintext, parameters
                )
            return (torch.cat((component.unsqueeze(0), ciphertext[1:]), dim=0),)
        if plaintext.shape[:-2].numel() == 1:
            return (rns_ops.montgomery_mul(ciphertext, plaintext, parameters),)
        return (
            torch.stack(
                tuple(
                    rns_ops.montgomery_mul(component, plaintext, parameters)
                    for component in ciphertext
                )
            ),
        )


@dataclass(frozen=True)
class NativeMontgomeryWeightedSumImplementation(_TensorOperandImplementation):
    r"""Compute one or more $\sum_t c_t p_t$ results without term products."""

    name: str = "native-montgomery-weighted-sum"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.MontgomeryWeightedSumOp,
        rns.MontgomeryWeightedSumsOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources, in_place
        count = cast(int, invocation.attributes["term_count"])
        ciphertexts, plaintexts, parameters = (
            values[:count],
            values[count:-1],
            values[-1],
        )
        if invocation.operation_type is rns.MontgomeryWeightedSumsOp:
            return (
                rns_ops.montgomery_weighted_sums(
                    list(ciphertexts),
                    list(plaintexts),
                    cast(int, invocation.attributes["group_count"]),
                    parameters,
                ),
            )
        return (
            rns_ops.montgomery_weighted_sum(
                list(ciphertexts), list(plaintexts), parameters
            ),
        )


@dataclass(frozen=True)
class NativeMontgomeryMultiplyImplementation(_TensorOperandImplementation):
    """Multiply two logical NTT/Montgomery polynomial bundles."""

    name: str = "native-montgomery-multiply"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.MontgomeryMultiplyOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, resources, in_place
        lhs, rhs, parameters = values
        return (rns_ops.montgomery_mul(lhs, rhs, parameters),)


@dataclass(frozen=True)
class NativeRnsStructureImplementation(_TensorOperandImplementation):
    """Extract and assemble logical RNS component bundles."""

    name: str = "native-rns-structure"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.ExtractComponentOp,
        rns.PackTwoComponentsOp,
        rns.PackThreeComponentsOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        if resources:
            raise ValueError("RNS structural operations accept no resources")
        if invocation.operation_type is rns.ExtractComponentOp:
            if len(values) != 1:
                raise ValueError("RNS component extraction requires one value")
            source = values[0]
            component = int(invocation.attributes["component"])  # type: ignore[arg-type]
            if component >= source.size(0):
                raise IndexError(
                    f"RNS component {component} is outside "
                    f"[0, {source.size(0)})"
                )
            return (source[component],)
        expected_count = (
            3 if invocation.operation_type is rns.PackThreeComponentsOp else 2
        )
        if len(values) != expected_count:
            raise ValueError(
                f"RNS component packing requires {expected_count} values"
            )
        return (torch.stack(values, dim=0),)


@dataclass(frozen=True)
class NativeMontgomeryAccumulateImplementation(_TensorOperandImplementation):
    """Accumulate equal-layout Montgomery bundles modulo twice each prime."""

    name: str = "native-montgomery-accumulate"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.AddMontgomeryLazyOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, resources, in_place
        lhs, rhs, parameters = values
        return (rns_ops.add_lazy(lhs, rhs, parameters),)


__all__ = [
    "NativeMontgomeryAccumulateImplementation",
    "NativeMontgomeryMultiplyImplementation",
    "NativeMontgomeryWeightedSumImplementation",
    "NativePlaintextArithmeticImplementation",
    "NativeRnsLinearImplementation",
    "NativeRnsStructureImplementation",
    "NativeRnsTransitionImplementation",
]
