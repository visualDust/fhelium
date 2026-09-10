"""CKKS value and key validation for one parameter set.

`CkksValidator` checks live tensor-backed values against one modulus layout
and integral dtype. Concrete execution bindings
check placement when a value is passed to a CPU or CUDA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from fhelium.config import CkksConfig
from fhelium.values import (
    Ciphertext,
    ConjugationKey,
    KeySwitchKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)


class RnsLayout(Protocol):
    """Provide active prime rows and key-switch digit count."""

    @property
    def key_digit_count(self) -> int: ...

    def prime_ids(
        self,
        depth: int,
        *,
        include_p: bool = False,
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class CkksValidator:
    """Validate CKKS values and keys for one CKKS parameter set."""

    config: CkksConfig
    rns_layout: RnsLayout
    rns_dtype: torch.dtype

    def _validate_depth(self, depth: object) -> int:
        if type(depth) is not int:
            raise TypeError("depth must be an integer")
        if not 0 <= depth <= self.config.max_depth:
            raise ValueError(
                f"depth must be in [0, {self.config.max_depth}]"
            )
        return depth

    def _validate_tensor(
        self,
        tensor: torch.Tensor,
        *,
        value_name: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        if tensor.layout != torch.strided:
            raise TypeError(f"{value_name} must use dense strided storage")
        expected_dtype = self.rns_dtype if dtype is None else dtype
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"{value_name} dtype {tensor.dtype} differs from {expected_dtype}"
            )
        if tensor.size(-1) != self.config.N:
            raise ValueError(
                f"{value_name} final extent {tensor.size(-1)} differs from "
                f"ring dimension {self.config.N}"
            )

    def _validate_prime_ids(
        self,
        *,
        depth: int,
        modulus_basis: object,
        prime_ids: tuple[int, ...],
        value_name: str,
    ) -> None:
        if modulus_basis not in {"Q", "QP"}:
            raise ValueError(f"{value_name} has unsupported modulus basis")
        expected = self.rns_layout.prime_ids(
            depth,
            include_p=modulus_basis == "QP",
        )
        if prime_ids != expected:
            raise ValueError(
                f"{value_name} prime_ids {prime_ids} differ from {expected}"
            )

    def validate_ciphertext(self, value: Ciphertext) -> None:
        """Validate a dense ciphertext against the configured runtime."""

        if not isinstance(value, Ciphertext):
            raise TypeError(f"Expected Ciphertext, got {type(value).__name__}")
        depth = self._validate_depth(value.depth)
        self._validate_tensor(value.data, value_name="Ciphertext")
        self._validate_prime_ids(
            depth=depth,
            modulus_basis=value.modulus_basis,
            prime_ids=value.prime_ids,
            value_name="Ciphertext",
        )

    def validate_plaintext(self, value: Plaintext) -> None:
        """Validate slots, coefficient, or RNS plaintext storage."""

        if not isinstance(value, Plaintext):
            raise TypeError(f"Expected Plaintext, got {type(value).__name__}")
        self._validate_depth(value.depth)
        if value.is_slots:
            if value.message is None:
                raise ValueError("Slots Plaintext has no message tensor")
            return
        if value.data is None:
            raise ValueError("Encoded Plaintext has no data tensor")
        expected_dtype = (
            torch.float64 if value.is_approximate_coefficients else None
        )
        self._validate_tensor(
            value.data,
            value_name="Plaintext",
            dtype=expected_dtype,
        )
        if value.is_rns:
            self._validate_prime_ids(
                depth=value.depth,
                modulus_basis=value.modulus_basis,
                prime_ids=value.prime_ids,
                value_name="Plaintext",
            )

    def _validate_key_storage(
        self,
        key: SecretKey | PublicKey | KeySwitchKey,
        *,
        expected_type: type[object],
        required_basis: str | None = None,
    ) -> None:
        if type(key) is not expected_type:
            raise TypeError(
                f"Expected {expected_type.__name__}, got {type(key).__name__}"
            )
        self._validate_tensor(key.data, value_name=type(key).__name__)
        if key.polynomial_domain != "ntt":
            raise ValueError(f"{type(key).__name__} must use NTT domain")
        if key.residue_representation != "montgomery":
            raise ValueError(
                f"{type(key).__name__} must use Montgomery residues"
            )
        if required_basis is not None and key.modulus_basis != required_basis:
            raise ValueError(
                f"{type(key).__name__} requires {required_basis} basis"
            )
        self._validate_prime_ids(
            depth=0,
            modulus_basis=key.modulus_basis,
            prime_ids=key.prime_ids,
            value_name=type(key).__name__,
        )
        if isinstance(key, KeySwitchKey):
            if key.digit_count != self.rns_layout.key_digit_count:
                raise ValueError(
                    f"{type(key).__name__} digit count {key.digit_count} "
                    f"differs from {self.rns_layout.key_digit_count}"
                )
            expected_shape = (
                self.rns_layout.key_digit_count,
                2,
                len(self.rns_layout.prime_ids(0, include_p=True)),
                self.config.N,
            )
            if tuple(key.data.shape) != expected_shape:
                raise ValueError(
                    f"{type(key).__name__} shape {tuple(key.data.shape)} "
                    f"differs from {expected_shape}"
                )
        elif isinstance(key, PublicKey):
            expected_shape = (2, len(key.prime_ids), self.config.N)
            if tuple(key.data.shape) != expected_shape:
                raise ValueError(
                    f"PublicKey shape {tuple(key.data.shape)} differs from "
                    f"{expected_shape}"
                )
        elif isinstance(key, SecretKey):
            expected_shape = (len(key.prime_ids), self.config.N)
            if tuple(key.data.shape) != expected_shape:
                raise ValueError(
                    f"SecretKey shape {tuple(key.data.shape)} differs from "
                    f"{expected_shape}"
                )

    def validate_secret_key(self, key: SecretKey) -> None:
        """Validate a depth-zero secret key."""

        self._validate_key_storage(key, expected_type=SecretKey)

    def validate_public_key(self, key: PublicKey) -> None:
        """Validate a depth-zero public encryption key."""

        self._validate_key_storage(key, expected_type=PublicKey)

    def validate_key_switch_key(self, key: KeySwitchKey) -> None:
        """Validate QP key-switch material and its concrete role metadata."""

        if type(key) not in {
            KeySwitchKey,
            RelinearizationKey,
            RotationKey,
            ConjugationKey,
        }:
            raise TypeError(f"Expected KeySwitchKey, got {type(key).__name__}")
        self._validate_key_storage(
            key,
            expected_type=type(key),
            required_basis="QP",
        )
        if isinstance(key, RotationKey):
            normalized = RotationKey.normalize_step(
                key.rotation_step,
                ring_dimension=self.config.N,
            )
            if key.rotation_step != normalized:
                raise ValueError("RotationKey step is not normalized")


__all__ = ["CkksValidator", "RnsLayout"]
