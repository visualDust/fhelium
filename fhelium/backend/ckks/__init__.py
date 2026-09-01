"""Provide CKKS Tensor algorithms, implementations, and execution resources.

Execution owners bind placement-specific arithmetic, random-stream, and key
resources. Implementations do not own an Engine or a concrete device.
"""

from .codec import (
    CKKS_CONFIG_RESOURCE_KIND,
    CKKS_CONFIG_RESOURCE_SYMBOL,
    RANDOM_STREAM_RESOURCE_KIND,
    RANDOM_STREAM_RESOURCE_SYMBOL,
    NativeDecodeImplementation,
    NativeEncodeImplementation,
    NativeIntegerCoefficientsToRnsImplementation,
)
from .crypto import (
    DECRYPT_RECONSTRUCTION_RESOURCE_KIND,
    PUBLIC_KEY_RESOURCE_KIND,
    SECRET_KEY_RESOURCE_KIND,
    CkksKeyGenerator,
    NativeDecryptImplementation,
    DecryptReconstructionResource,
    NativeEncryptImplementation,
    KeyGenerationResource,
)
from .materialization import (
    CkksDeviceResources,
)
from .scalar import NativeScalarArithmeticImplementation

__all__ = [
    "DECRYPT_RECONSTRUCTION_RESOURCE_KIND",
    "CKKS_CONFIG_RESOURCE_KIND",
    "CKKS_CONFIG_RESOURCE_SYMBOL",
    "RANDOM_STREAM_RESOURCE_KIND",
    "RANDOM_STREAM_RESOURCE_SYMBOL",
    "CkksDeviceResources",
    "CkksKeyGenerator",
    "NativeDecodeImplementation",
    "NativeDecryptImplementation",
    "DecryptReconstructionResource",
    "NativeEncodeImplementation",
    "NativeEncryptImplementation",
    "NativeIntegerCoefficientsToRnsImplementation",
    "NativeScalarArithmeticImplementation",
    "KeyGenerationResource",
    "PUBLIC_KEY_RESOURCE_KIND",
    "SECRET_KEY_RESOURCE_KIND",
]
