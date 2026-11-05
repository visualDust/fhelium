"""Assemble FHElium's built-in operation implementations."""

from __future__ import annotations

from importlib.util import find_spec

from fhelium.backend.implementation import OperationImplementationRegistry
from .torch import TorchCallImplementation
from .ckks.codec._periodic import NativePrepareCompressedPlaintextImplementation
from fhelium.backend.memory import TorchMemoryTransferImplementation
from fhelium.backend.ckks import (
    NativeDecodeImplementation,
    NativeDecryptImplementation,
    NativeEncodeImplementation,
    NativeEncryptImplementation,
    NativeIntegerCoefficientsToRnsImplementation,
    NativeScalarArithmeticImplementation,
)

from .ckks.arithmetic import (
    NativeCiphertextMultiplyImplementation,
    NativeCompressedPlaintextImplementation,
)
from .ckks.modulus_raise import NativeModulusRaiseImplementation
from .ckks.key_switch import (
    NativeKeySwitchImplementation,
    NativeRelinearizeImplementation,
)
from .ckks.rotation.operations import (
    NativeRotateImplementation,
    NativeRotationSumImplementation,
    NativeHoistedRotateManyImplementation,
    NativeGroupedRotationWeightedSumImplementation,
)
from .rns.basis_extension import NativeCenteredBasisExtensionImplementation
from .rns.crt import NativeCrtBasisExtensionImplementation
from .rns.modup import NativeHybridModUpImplementation
from .rns.moddown import NativeModDownImplementation
from .rns.key_product import NativeKeySwitchDigitProductImplementation
from .rns.automorphism import NativeCoefficientAutomorphismImplementation
from .rns.rescale import NativeRescaleImplementation
from .ntt.operations import NativeNttImplementation
from .ntt.automorphism import NativeNttAutomorphismImplementation
from .rns.operations import (
    NativeBatchSumImplementation,
    NativeMontgomeryAccumulateImplementation,
    NativeMontgomeryMultiplyImplementation,
    NativeMontgomeryWeightedSumImplementation,
    NativePlaintextArithmeticImplementation,
    NativeRnsLinearImplementation,
    NativeRnsStructureImplementation,
    NativeRnsTransitionImplementation,
)


def create_builtin_operation_registry(
    *,
    ntt_backend_name: str | None = None,
) -> OperationImplementationRegistry:
    """Assemble built-in implementations without binding numerical resources."""

    ntt_implementation = (
        NativeNttImplementation(
            ntt_backend_name,
            required_backend_name=ntt_backend_name,
        )
        if ntt_backend_name is not None
        else NativeNttImplementation(
            "native-ntt",
            required_backend_name=None,
        )
    )
    generated = ()
    weighted = ()
    if find_spec("triton") is not None:
        from .triton import (
            TritonFusionImplementation,
            TritonTensorFusionImplementation,
        )

        from .triton._weighted_sum import TritonWeightedSumImplementation

        weighted = (TritonWeightedSumImplementation(),)
        generated = (
            TritonTensorFusionImplementation(),
            TritonFusionImplementation(),
            TritonFusionImplementation(
                name="triton-ntt-fused", include_ntt=True
            ),
        )
    return OperationImplementationRegistry(
        (
            *generated,
            TorchCallImplementation(),
            TorchMemoryTransferImplementation(),
            NativeRnsLinearImplementation(),
            NativeBatchSumImplementation(),
            NativeRnsTransitionImplementation(),
            NativePlaintextArithmeticImplementation(),
            NativeMontgomeryMultiplyImplementation(),
            NativeMontgomeryWeightedSumImplementation(),
            *weighted,
            NativeRnsStructureImplementation(),
            NativeCenteredBasisExtensionImplementation(),
            NativeCrtBasisExtensionImplementation(),
            NativeHybridModUpImplementation(),
            NativeKeySwitchDigitProductImplementation(),
            NativeMontgomeryAccumulateImplementation(),
            NativeModDownImplementation(),
            NativeCoefficientAutomorphismImplementation(),
            ntt_implementation,
            NativeNttAutomorphismImplementation(),
            NativeRescaleImplementation(),
            NativeCiphertextMultiplyImplementation(),
            NativeRelinearizeImplementation(),
            NativeModulusRaiseImplementation(),
            NativeKeySwitchImplementation(),
            NativeRotateImplementation(),
            NativeRotationSumImplementation(),
            NativeCompressedPlaintextImplementation(),
            NativeHoistedRotateManyImplementation(),
            NativeGroupedRotationWeightedSumImplementation(),
            NativeEncodeImplementation(),
            NativePrepareCompressedPlaintextImplementation(),
            NativeDecodeImplementation(),
            NativeIntegerCoefficientsToRnsImplementation(),
            NativeScalarArithmeticImplementation(),
            NativeEncryptImplementation(),
            NativeDecryptImplementation(),
        )
    )


__all__ = ["create_builtin_operation_registry"]
