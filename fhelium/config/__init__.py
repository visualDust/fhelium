"""CKKS parameter, NTT-policy, and security-assessment interfaces.

Public modules are named for the configured subject: ``ckks`` owns the CKKS
parameter model, ``ntt`` owns NTT execution policies, and ``security`` assesses
complete modulus sets against the published table built into FHElium. Private
resource mechanisms use a leading underscore, as in ``_prime_catalog``.
"""

from .ckks import CkksConfig, Preset  # noqa: I001
from .ntt import (
    DEFAULT_CPU_NTT_BACKEND,
    DEFAULT_NTT_BACKEND,
    SUPPORTED_NTT_BACKENDS,
    compatible_ntt_backends,
)
from .security import (
    SecurityAssessment,
    assess_config_security,
    assess_security,
)


__all__ = [
    "DEFAULT_CPU_NTT_BACKEND",
    "DEFAULT_NTT_BACKEND",
    "SUPPORTED_NTT_BACKENDS",
    "CkksConfig",
    "Preset",
    "SecurityAssessment",
    "assess_config_security",
    "assess_security",
    "compatible_ntt_backends",
]
