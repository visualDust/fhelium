"""Public CKKS configuration, values, keys, engine, and file operations."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fhelium.config import (
    DEFAULT_CPU_NTT_BACKEND,
    DEFAULT_NTT_BACKEND,
    SUPPORTED_NTT_BACKENDS,
    CkksConfig,
    Preset,
    compatible_ntt_backends,
)
from fhelium.core import (
    COMPRESSED_PLAINTEXT_FORMAT_VERSION,
    Ciphertext,
    CkksContextSpec,
    CompressedPlaintext,
    CompressedPlaintextLayout,
    ConjugationKey,
    EvaluationKeyRequirements,
    EvaluationKeySet,
    KeySwitchKey,
    ModulusBasis,
    Plaintext,
    PlaintextRepresentation,
    PolynomialDomain,
    PublicKey,
    RelinearizationKey,
    ResidueRepresentation,
    RotationKey,
    RotationKeySet,
    SecretKey,
    TensorResident,
)
from fhelium.engine import CkksEngine
from fhelium.serialization import (
    ValueFileMetadata,
    inspect_value,
    load_value,
    save_value,
)

from . import errors

try:
    __version__ = version("fhelium")
except PackageNotFoundError:
    # Permit pure-Python/static inspection from an uninstalled source checkout.
    # Installed packages must carry dist-info; only a repository-local
    # pyproject is accepted as the fallback source of version truth.
    _source_pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not _source_pyproject.is_file():
        raise
    with _source_pyproject.open("rb") as _stream:
        __version__ = tomllib.load(_stream)["project"]["version"]

__all__ = [
    "DEFAULT_CPU_NTT_BACKEND",
    "COMPRESSED_PLAINTEXT_FORMAT_VERSION",
    "DEFAULT_NTT_BACKEND",
    "SUPPORTED_NTT_BACKENDS",
    "Ciphertext",
    "CkksConfig",
    "CkksContextSpec",
    "CkksEngine",
    "CompressedPlaintext",
    "CompressedPlaintextLayout",
    "ConjugationKey",
    "EvaluationKeyRequirements",
    "EvaluationKeySet",
    "KeySwitchKey",
    "ModulusBasis",
    "Plaintext",
    "PlaintextRepresentation",
    "PolynomialDomain",
    "Preset",
    "PublicKey",
    "RelinearizationKey",
    "ResidueRepresentation",
    "RotationKey",
    "RotationKeySet",
    "SecretKey",
    "TensorResident",
    "ValueFileMetadata",
    "compatible_ntt_backends",
    "errors",
    "inspect_value",
    "load_value",
    "save_value",
]
