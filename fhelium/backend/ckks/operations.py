"""Native direct implementations of whole CKKS tensor operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.ckks.rotation._hoisted import _RotationHoistExecutor
from fhelium.backend.ckks.crypto._galois import (
    coefficient_galois_gather_indices,
)
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ntt.resources import NTT_RESOURCE_KIND
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns._operand_state import _active_level, _operand_basis
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.resources import RNS_RESOURCE_KIND
from fhelium.values import KeySwitchKey, RelinearizationKey, RotationKey
from fhelium.ir.dialects import ckks, rns
from fhelium.native.wrapper import ckks_ops, rns_ops

from .resources import (
    KEY_SWITCH_PLAN_RESOURCE_KIND,
    RELINEARIZATION_KEY_RESOURCE_KIND,
    KeySwitchExecutionResource,
)


def _key_switch_contexts(
    resources: tuple[BoundResource, ...],
    *,
    offset: int = 0,
) -> tuple[RnsContext, NttContext, KeySwitchExecutionResource]:
    """Resolve the RNS, NTT, and key-switch contexts for a direct call."""

    rns_context = cast(RnsContext, resources[offset].value)
    ntt_context = cast(NttContext, resources[offset + 1].value)
    plan = cast(KeySwitchExecutionResource, resources[offset + 2].value)
    if (
        ntt_context.rns_context is not rns_context
        or plan.rns_context is not rns_context
        or plan.ntt_context is not ntt_context
    ):
        raise ValueError("Key-switch resources bind different contexts")
    return rns_context, ntt_context, plan


def _hybrid_modup_digit(
    source: torch.Tensor,
    *,
    rns_context: RnsContext,
    level: int,
    digit_index: int,
) -> torch.Tensor:
    """Extend one coefficient-domain Q digit into active QP rows."""

    digit_spec = rns_context.rns_layout.digit_specs(level)[digit_index]
    source_rows = digit_spec.component_row_ids
    mixed = source[..., source_rows[0] : source_rows[-1] + 1, :]
    digit_width = len(source_rows)
    row_parameters = rns_context.row_parameters(digit_spec.prime_ids)
    if digit_width > 1:
        if digit_width > 8:
            raise ValueError(
                "native streaming key switching supports digit widths "
                "through eight"
            )
        normalizers = row_parameters.mixed_radix_normalizers
        propagation = row_parameters.mixed_radix_propagation_coefficients
        if normalizers is None or propagation is None:
            raise RuntimeError(
                "Streaming relinearization is missing mixed-radix tables"
            )
        modulus_lo, modulus_hi, neg_inv_lo, neg_inv_hi = (
            row_parameters.montgomery_reduction_parameters
        )
        mixed = rns_ops.mixed_radix_decompose(
            mixed,
            normalizers.contiguous(),
            propagation.contiguous(),
            modulus_lo.contiguous(),
            modulus_hi.contiguous(),
            neg_inv_lo.contiguous(),
            neg_inv_hi.contiguous(),
        )
    active_basis = rns_context.basis_parameters(level, include_p=True)
    basis_extension = row_parameters.basis_extension_coefficients
    if basis_extension is None:
        basis_extension = torch.empty(
            0,
            0,
            dtype=source.dtype,
            device=source.device,
        )
    start = active_basis.parameter_row_start
    stop = start + len(active_basis.prime_ids)
    return rns_ops.mixed_radix_basis_extend_to_montgomery(
        mixed,
        basis_extension[:, start:stop],
        active_basis.native_parameters,
        len(active_basis.prime_ids),
    )


def _stream_key_switch_corrections(
    source: torch.Tensor,
    *,
    rns_context: RnsContext,
    ntt_context: NttContext,
    plan: KeySwitchExecutionResource,
    key: KeySwitchKey,
    level: int,
) -> torch.Tensor:
    """Stream one coefficient-domain Q component through hybrid key switching."""

    qp_parameters = rns_context.basis_parameters(level, include_p=True)
    accumulator: torch.Tensor | None = None
    for digit_index, digit_spec in enumerate(
        rns_context.rns_layout.digit_specs(level)
    ):
        lifted = _hybrid_modup_digit(
            source,
            rns_context=rns_context,
            level=level,
            digit_index=digit_index,
        )
        ntt_context.forward_montgomery_(
            lifted,
            parameter_row_start=qp_parameters.parameter_row_start,
        )
        if accumulator is None:
            accumulator = torch.zeros(
                (2, *lifted.shape),
                dtype=lifted.dtype,
                device=lifted.device,
            )
        ckks_ops.keyswitch_accumulate_digit_products_(
            accumulator[0],
            accumulator[1],
            lifted,
            key.digit(digit_spec.key_digit_index),
            qp_parameters.native_parameters,
            qp_parameters.parameter_row_start,
        )
    if accumulator is None:
        raise ValueError("Key switching requires at least one RNS digit")

    for component in accumulator.unbind(0):
        ntt_context.inverse_to_standard_(
            component,
            parameter_row_start=qp_parameters.parameter_row_start,
        )
    p_count = rns_context.config.num_p_primes
    inverses = plan.moddown_tables[level]
    return ckks_ops.keyswitch_moddown_qp_to_q(
        accumulator[..., :-p_count, :],
        accumulator[..., -p_count:, :],
        inverses,
        qp_parameters.native_parameters,
    )


@dataclass(frozen=True)
class NativeCiphertextMultiplyImplementation:
    """Execute whole two-component CKKS convolution in one native call."""

    supports_in_place: bool = False
    name: str = "native-ct2-convolution"
    operation_types: tuple[type[Operation], ...] = (ckks.MultiplyOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
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
        lhs, rhs = inputs
        resource = cast(RnsContext, resources[0].value)
        parameters = resource.rns_parameters_for(
            lhs,
            include_p=_operand_basis(invocation) == "QP",
        )
        return (
            ckks_ops.multiply_two_component_ntt_montgomery(
                lhs,
                rhs,
                parameters,
            ),
        )


@dataclass(frozen=True)
class NativeRelinearizeImplementation:
    """Relinearize CT3 through one streaming hybrid key-switch schedule."""

    supports_in_place: bool = False
    name: str = "native-relinearize-streaming"
    operation_types: tuple[type[Operation], ...] = (ckks.RelinearizeOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-ntt-plan",
                NTT_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-key-switch-plan",
                KEY_SWITCH_PLAN_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "relinearization-key",
                RELINEARIZATION_KEY_RESOURCE_KIND,
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
        del invocation
        source = inputs[0]
        if source.size(0) != 3:
            raise ValueError(
                "Relinearization requires ciphertext components=3, "
                f"got {source.size(0)}"
            )
        rns_context, ntt_context, plan = _key_switch_contexts(resources)
        key = cast(RelinearizationKey, resources[3].value)

        level = _active_level(source, rns_context, include_p=False)
        q_parameters = rns_context.basis_parameters(level, include_p=False)
        coefficient = source.clone()
        for component in coefficient.unbind(0):
            ntt_context.inverse_to_standard_(
                component,
                parameter_row_start=q_parameters.parameter_row_start,
            )
        switched_component = coefficient[2]

        corrections = _stream_key_switch_corrections(
            switched_component,
            rns_context=rns_context,
            ntt_context=ntt_context,
            plan=plan,
            key=key,
            level=level,
        )
        return (
            rns_context.add_standard(
                coefficient[:2],
                corrections,
                include_p=False,
            ),
        )


@dataclass(frozen=True)
class NativeRotateImplementation:
    """Rotate CT2 through one automorphism and streaming hybrid key switch."""

    supports_in_place: bool = False
    name: str = "native-rotate-streaming"
    operation_types: tuple[type[Operation], ...] = (ckks.RotateOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-ntt-plan",
                NTT_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-key-switch-plan",
                KEY_SWITCH_PLAN_RESOURCE_KIND,
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
        del invocation
        source = inputs[0]
        if source.size(0) != 2:
            raise ValueError(
                "Rotation requires ciphertext components=2, "
                f"got {source.size(0)}"
            )
        key = cast(RotationKey, resources[0].value)
        rns_context, ntt_context, plan = _key_switch_contexts(
            resources,
            offset=1,
        )
        shift = key.rotation_step
        exponent = -shift if plan.galois_generator == 5 else shift
        galois_element = pow(
            plan.galois_generator,
            exponent % rns_context.config.N,
            2 * rns_context.config.N,
        )
        level = _active_level(source, rns_context, include_p=False)
        source_indices, source_sign = coefficient_galois_gather_indices(
            rns_context.config.N,
            galois_element,
            source.device,
        )
        rotated = ckks_ops.apply_coefficient_galois_automorphism(
            source,
            source_indices,
            source_sign,
            rns_context.twice_modulus_for_basis(level, include_p=False),
        )
        corrections = _stream_key_switch_corrections(
            rotated[1],
            rns_context=rns_context,
            ntt_context=ntt_context,
            plan=plan,
            key=key,
            level=level,
        )
        result0 = rns_context.add_standard(
            rotated[0],
            corrections[0],
            include_p=False,
        )
        return (torch.stack((result0, corrections[1]), dim=0),)


@dataclass(frozen=True)
class NativeCompressedPlaintextImplementation:
    """Execute compressed plaintext arithmetic on Tensor payloads."""

    supports_in_place: bool = False
    name: str = "native-compressed-plaintext"
    operation_types: tuple[type[Operation], ...] = (
        ckks.AddCompressedPlaintextOp,
        ckks.MultiplyCompressedPlaintextOp,
    )

    def resource_requirements(
        self,
        invocation: OperationInvocation,
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
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
        resource = cast(RnsContext, resources[0].value)
        ciphertext, compressed = inputs[:2]
        layout = cast(str, invocation.attributes["compression_layout"])
        include_p = _operand_basis(invocation) == "QP"
        parameters = resource.rns_parameters_for(
            ciphertext,
            include_p=include_p,
        )
        if invocation.operation_type is ckks.AddCompressedPlaintextOp:
            component0 = ciphertext[0]
            if layout == "cyclic" and len(inputs) == 2:
                output0 = ckks_ops.add_cyclic_compressed_plaintext_component(
                    component0,
                    compressed,
                    parameters,
                )
            elif layout == "contiguous" and len(inputs) == 2:
                output0 = (
                    ckks_ops.add_contiguous_compressed_plaintext_component(
                        component0,
                        compressed,
                        parameters,
                    )
                )
            elif layout == "strided_sparse" and len(inputs) == 3:
                output0 = ckks_ops.add_strided_plaintext_component(
                    component0,
                    compressed,
                    inputs[2],
                    parameters,
                )
            else:
                raise ValueError(
                    "Compressed addition inputs differ from its layout"
                )
            output = ciphertext.clone()
            output[0].copy_(output0)
            return (output,)
        if len(inputs) != 2 or layout not in {"cyclic", "contiguous"}:
            raise ValueError(
                "Compressed multiplication requires cyclic or contiguous data"
            )
        if compressed.shape[:-2].numel() == 1:
            if layout == "cyclic":
                return (
                    resource.montgomery_mul_cyclic_compressed(
                        ciphertext,
                        compressed,
                        include_p=include_p,
                    ),
                )
            return (
                resource.montgomery_mul_contiguous_compressed(
                    ciphertext,
                    compressed,
                    include_p=include_p,
                ),
            )
        products = []
        for component_index in range(ciphertext.size(0)):
            component = ciphertext[component_index]
            if layout == "cyclic":
                product = resource.montgomery_mul_cyclic_compressed(
                    component,
                    compressed,
                    include_p=include_p,
                )
            else:
                product = resource.montgomery_mul_contiguous_compressed(
                    component,
                    compressed,
                    include_p=include_p,
                )
            products.append(product)
        return (torch.stack(products, dim=0),)


@dataclass(frozen=True)
class NativeKeySwitchDigitProductImplementation:
    """Multiply one NTT QP digit by an evaluation-key digit."""

    name: str = "native-key-switch-digit-product"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.KeySwitchDigitProductOp,
    )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        if len(values) != 1:
            raise ValueError("Key-switch digit product requires one digit")
        digit = values[0]
        key = cast(KeySwitchKey, resources[0].value)
        rns_context = cast(RnsContext, resources[1].value)
        key_digit_index = int(invocation.attributes["key_digit_index"])  # type: ignore[arg-type]
        if key_digit_index >= key.digit_count:
            raise IndexError(
                "Key-switch key digit index is outside key storage"
            )
        accumulator0 = torch.zeros_like(digit)
        accumulator1 = torch.zeros_like(digit)
        level = _active_level(digit, rns_context, include_p=True)
        active = rns_context.basis_parameters(level, include_p=True)
        ckks_ops.keyswitch_accumulate_digit_products_(
            accumulator0,
            accumulator1,
            digit,
            key.digit(key_digit_index),
            active.native_parameters,
            active.parameter_row_start,
        )
        return (torch.stack((accumulator0, accumulator1), dim=0),)


@dataclass(frozen=True)
class NativeKeySwitchModDownImplementation:
    """Divide a coefficient QP key-switch accumulator by P into active Q."""

    name: str = "native-key-switch-moddown"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.ModDownQpToQOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation
        del in_place
        if len(values) != 1:
            raise ValueError("Key-switch ModDown requires one accumulator")
        source = values[0]
        if source.size(0) != 2:
            raise ValueError(
                "Key-switch ModDown requires two correction components"
            )
        rns_context = cast(RnsContext, resources[0].value)
        plan = cast(KeySwitchExecutionResource, resources[1].value)
        level = _active_level(source, rns_context, include_p=True)
        p_count = rns_context.config.num_p_primes
        inverses = plan.moddown_tables[level]
        parameters = rns_context.basis_parameters(
            level, include_p=True
        ).native_parameters
        corrections = tuple(
            ckks_ops.keyswitch_moddown_qp_to_q(
                source[component, ..., :-p_count, :],
                source[component, ..., -p_count:, :],
                inverses,
                parameters,
            )
            for component in range(2)
        )
        return (torch.stack(corrections, dim=0),)


@dataclass(frozen=True)
class NativeHoistedRotateManyImplementation:
    """Execute one scheduled rotation group without changing its membership."""

    supports_in_place: bool = False
    name: str = "native-rotate-many-hoisted"
    operation_types: tuple[type[Operation], ...] = (ckks.RotateManyOp,)
    _executor_entry: (
        tuple[
            KeySwitchExecutionResource,
            RnsContext,
            NttContext,
            _RotationHoistExecutor,
        ]
        | None
    ) = field(default=None, init=False, repr=False, compare=False)

    def _executor(
        self,
        plan: KeySwitchExecutionResource,
        rns_context: RnsContext,
        ntt_context: NttContext,
    ) -> _RotationHoistExecutor:
        current = self._executor_entry
        if (
            current is not None
            and current[0] is plan
            and current[1] is rns_context
            and current[2] is ntt_context
        ):
            return current[3]
        executor = _RotationHoistExecutor(
            config=rns_context.config,
            rns_context=rns_context,
            ntt_context=ntt_context,
            moddown_p_drop_inverses_montgomery_by_level=plan.moddown_tables,
            galois_generator=plan.galois_generator,
        )
        object.__setattr__(
            self,
            "_executor_entry",
            (plan, rns_context, ntt_context, executor),
        )
        return executor

    def resource_requirements(
        self,
        invocation: OperationInvocation,
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-ntt-plan",
                NTT_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-key-switch-plan",
                KEY_SWITCH_PLAN_RESOURCE_KIND,
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
        key_count = invocation.result_count
        if len(inputs) != 1 or len(resources) != key_count + 3:
            raise ValueError("Rotate-many input or resource count differs")
        source = inputs[0]
        if source.size(0) != 2:
            raise ValueError(
                "operand 0 requires ciphertext components=2, "
                f"got {source.size(0)}"
            )
        rns_context, ntt_context, plan = _key_switch_contexts(
            resources,
            offset=key_count,
        )
        level = _active_level(source, rns_context, include_p=False)
        executor = self._executor(plan, rns_context, ntt_context)
        prepared = executor.prepare(source[1], level)
        outputs: list[torch.Tensor] = []
        for key_binding in resources[:key_count]:
            key = cast(RotationKey, key_binding.value)
            rotated_c0 = executor.rotate_component(
                source[0],
                level=level,
                key=key,
            )
            output0, output1 = executor.apply(
                rotated_c0,
                prepared,
                key,
            )
            outputs.append(torch.stack((output0, output1), dim=0))
        return tuple(outputs)


__all__ = [
    "NativeCiphertextMultiplyImplementation",
    "NativeHoistedRotateManyImplementation",
    "NativeKeySwitchDigitProductImplementation",
    "NativeKeySwitchModDownImplementation",
    "NativeRelinearizeImplementation",
    "NativeRotateImplementation",
]
