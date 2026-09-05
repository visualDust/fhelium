"""Assemble FHElium's built-in operation implementations."""

from __future__ import annotations

from fhelium.backend.implementation import OperationImplementationRegistry
from fhelium.backend.memory import TorchMemoryTransferImplementation
from fhelium.backend.ckks import (
    NativeDecodeImplementation,
    NativeDecryptImplementation,
    NativeEncodeImplementation,
    NativeEncryptImplementation,
    NativeIntegerCoefficientsToRnsImplementation,
    NativeScalarArithmeticImplementation,
)

from .ckks.operations import (
    NativeCiphertextMultiplyImplementation,
    NativeCompressedPlaintextImplementation,
    NativeHoistedRotateManyImplementation,
    NativeKeySwitchDigitProductImplementation,
    NativeKeySwitchImplementation,
    NativeKeySwitchModDownImplementation,
    NativeRelinearizeImplementation,
    NativeRotateImplementation,
)
from .ckks.rescale import NativeRescaleImplementation
from .ntt.operations import NativeNttImplementation
from .rns.operations import (
    NativeCoefficientAutomorphismImplementation,
    NativeHybridModUpImplementation,
    NativeMontgomeryAccumulateImplementation,
    NativeMontgomeryMultiplyImplementation,
    NativePlaintextArithmeticImplementation,
    NativeRnsLinearImplementation,
    NativeRnsStructureImplementation,
    NativeRnsTransitionImplementation,
)


def create_builtin_operation_registry(
    *,
    ntt_backend_name: str | None = None,
) -> OperationImplementationRegistry:
    """Construct built-in implementations for one device resource bundle."""

    ntt_implementation = (
        NativeNttImplementation(
            ntt_backend_name,
            required_backend_name=ntt_backend_name,
        )
        if ntt_backend_name is not None
        else NativeNttImplementation(
            "supplied-ntt-plan",
            required_backend_name=None,
        )
    )
    return OperationImplementationRegistry(
        (
            TorchMemoryTransferImplementation(),
            NativeRnsLinearImplementation(),
            NativeRnsTransitionImplementation(),
            NativePlaintextArithmeticImplementation(),
            NativeMontgomeryMultiplyImplementation(),
            NativeRnsStructureImplementation(),
            NativeHybridModUpImplementation(),
            NativeKeySwitchDigitProductImplementation(),
            NativeMontgomeryAccumulateImplementation(),
            NativeKeySwitchModDownImplementation(),
            NativeCoefficientAutomorphismImplementation(),
            ntt_implementation,
            NativeRescaleImplementation(),
            NativeCiphertextMultiplyImplementation(),
            NativeRelinearizeImplementation(),
            NativeKeySwitchImplementation(),
            NativeRotateImplementation(),
            NativeCompressedPlaintextImplementation(),
            NativeHoistedRotateManyImplementation(),
            NativeEncodeImplementation(),
            NativeDecodeImplementation(),
            NativeIntegerCoefficientsToRnsImplementation(),
            NativeScalarArithmeticImplementation(),
            NativeEncryptImplementation(),
            NativeDecryptImplementation(),
        )
    )


__all__ = ["create_builtin_operation_registry"]
