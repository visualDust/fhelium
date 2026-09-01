"""Typed key ownership for one eager CKKS runtime.

`KeyInventory` stores validated secret, public, relinearization, conjugation,
and normalized rotation keys. Replacing the secret key clears every key whose
mathematical relation was derived from the previous secret.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fhelium.values import (
    ConjugationKey,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from fhelium.eager._validation import CkksValidator


@dataclass
class KeyInventory:
    """Own validated keys and their dependency invalidation rules."""

    validator: CkksValidator
    _secret_key: SecretKey | None = field(default=None, init=False, repr=False)
    _public_key: PublicKey | None = field(default=None, init=False, repr=False)
    _relinearization_key: RelinearizationKey | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _conjugation_key: ConjugationKey | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _rotation_keys: dict[int, RotationKey] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.validator, CkksValidator):
            raise TypeError("validator must be a CkksValidator")

    def clear_evaluation_keys(self) -> None:
        """Remove relinearization, conjugation, and rotation keys."""

        self._relinearization_key = None
        self._conjugation_key = None
        self._rotation_keys.clear()

    def clear_all(self) -> None:
        """Remove every installed key."""

        self._secret_key = None
        self._public_key = None
        self.clear_evaluation_keys()

    def set_secret_key(self, key: SecretKey | None) -> None:
        """Install a secret key and invalidate keys derived from its predecessor."""

        if key is not None:
            self.validator.validate_secret_key(key)
        if key is self._secret_key:
            return
        self._secret_key = key
        self._public_key = None
        self.clear_evaluation_keys()

    def require_secret_key(self) -> SecretKey:
        """Return the installed secret key or raise when none is installed."""

        if self._secret_key is None:
            raise KeyError("No secret key is installed")
        return self._secret_key

    @property
    def secret_key(self) -> SecretKey | None:
        """Return the installed secret key, if any."""

        return self._secret_key

    def set_public_key(self, key: PublicKey | None) -> None:
        """Install or remove the public encryption key."""

        if key is not None:
            self.validator.validate_public_key(key)
        self._public_key = key

    def require_public_key(self) -> PublicKey:
        """Return the installed public key or raise when none is installed."""

        if self._public_key is None:
            raise KeyError("No public key is installed")
        return self._public_key

    @property
    def public_key(self) -> PublicKey | None:
        """Return the installed public key, if any."""

        return self._public_key

    def set_relinearization_key(
        self,
        key: RelinearizationKey | None,
    ) -> None:
        """Install or remove relinearization material."""

        if key is not None:
            self.validator.validate_key_switch_key(key)
        self._relinearization_key = key

    def require_relinearization_key(self) -> RelinearizationKey:
        """Return installed relinearization material."""

        if self._relinearization_key is None:
            raise KeyError("No relinearization key is installed")
        return self._relinearization_key

    @property
    def relinearization_key(self) -> RelinearizationKey | None:
        """Return installed relinearization material, if any."""

        return self._relinearization_key

    def set_conjugation_key(self, key: ConjugationKey | None) -> None:
        """Install or remove conjugation material."""

        if key is not None:
            self.validator.validate_key_switch_key(key)
        self._conjugation_key = key

    def require_conjugation_key(self) -> ConjugationKey:
        """Return installed conjugation material."""

        if self._conjugation_key is None:
            raise KeyError("No conjugation key is installed")
        return self._conjugation_key

    @property
    def conjugation_key(self) -> ConjugationKey | None:
        """Return installed conjugation material, if any."""

        return self._conjugation_key

    def set_rotation_key(self, key: RotationKey | None, *, step: int) -> None:
        """Install or remove the normalized rotation key for `step`."""

        normalized = RotationKey.normalize_step(
            step,
            ring_dimension=self.validator.config.N,
        )
        if key is None:
            self._rotation_keys.pop(normalized, None)
            return
        self.validator.validate_key_switch_key(key)
        if key.rotation_step != normalized:
            raise ValueError(
                f"RotationKey step {key.rotation_step} differs from {normalized}"
            )
        self._rotation_keys[normalized] = key

    def rotation_key(self, step: int) -> RotationKey | None:
        """Return the key for a normalized signed rotation step, if installed."""

        normalized = RotationKey.normalize_step(
            step,
            ring_dimension=self.validator.config.N,
        )
        return self._rotation_keys.get(normalized)

    def require_rotation_key(self, step: int) -> RotationKey:
        """Return the key for `step` or raise when none is installed."""

        key = self.rotation_key(step)
        if key is None:
            raise KeyError(f"No rotation key is installed for step {step}")
        return key

    @property
    def rotation_keys(self) -> dict[int, RotationKey]:
        """Return a copy of the normalized rotation-key map."""

        return dict(self._rotation_keys)


__all__ = ["KeyInventory"]
