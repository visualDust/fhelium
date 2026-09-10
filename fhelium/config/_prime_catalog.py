"""Load the versioned CKKS prime resources packaged with FHElium.

The installed tables supply ordered Q-prime candidates and a separate
sequence for the special primes in P. Entries are keyed
by target prime width and ring dimension.

Loading validates the resource format/version metadata, serialized key schema,
one-dimensional ``int64`` tensors, nonempty unique sequences, and the required
``q = 1 mod 2N`` NTT congruence. The decoded tables are sorted, wrapped in
read-only mappings, and cached once per process. :class:`PrimeCatalog` returns
list copies so parameter selection cannot mutate the shared installed data.

The corresponding offline generator is ``scripts/generate_prime_catalog.py``;
runtime code only consumes its reviewed, packaged outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType

import torch
from safetensors import safe_open

from fhelium.errors import PrimeCatalogResourceError

_RESOURCE_PACKAGE = "fhelium.config.resources"
_CATALOG_VERSION = "1"
_SCALING_RESOURCE = "scaling_primes_v1.safetensors"
_SPECIAL_RESOURCE = "special_primes_v1.safetensors"

PrimeKey = tuple[int, int]
PrimeTable = Mapping[PrimeKey, tuple[int, ...]]


def _decode_key(raw: str, *, resource_name: str) -> PrimeKey:
    """Decode one ``sb=<bits>;N=<degree>`` resource key."""

    try:
        fields = dict(part.split("=", 1) for part in raw.split(";"))
        if set(fields) != {"sb", "N"}:
            raise ValueError
        bits = int(fields["sb"])
        degree = int(fields["N"])
    except (TypeError, ValueError) as exc:
        raise PrimeCatalogResourceError(
            resource_name=resource_name,
            detail=f"invalid parameter key {raw!r}",
        ) from exc

    if bits <= 0 or degree <= 0 or degree & (degree - 1):
        raise PrimeCatalogResourceError(
            resource_name=resource_name,
            detail=f"invalid parameter key {raw!r}",
        )
    return bits, degree


def _validate_primes(
    key: PrimeKey,
    primes: tuple[int, ...],
    *,
    resource_name: str,
) -> None:
    """Validate one nonempty, unique sequence of NTT-compatible primes."""

    if not primes:
        raise PrimeCatalogResourceError(
            resource_name=resource_name,
            detail=f"empty prime sequence for {key!r}",
        )
    if len(set(primes)) != len(primes):
        raise PrimeCatalogResourceError(
            resource_name=resource_name,
            detail=f"duplicate primes for {key!r}",
        )

    _, degree = key
    modulus = 2 * degree
    if any(prime <= 2 or (prime - 1) % modulus for prime in primes):
        raise PrimeCatalogResourceError(
            resource_name=resource_name,
            detail=f"invalid NTT prime sequence for {key!r}",
        )


def _load_table(
    resource_name: str,
    *,
    expected_format: str,
) -> PrimeTable:
    """Load one packaged safetensors table with the required schema and version."""

    decoded: dict[PrimeKey, tuple[int, ...]] = {}
    try:
        resource = resources.files(_RESOURCE_PACKAGE).joinpath(resource_name)
        if not resource.is_file():
            raise PrimeCatalogResourceError(
                resource_name=resource_name,
                detail=(
                    "resource is missing; reinstall FHElium from a complete "
                    "distribution"
                ),
            )
        with resources.as_file(resource) as path:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("format") != expected_format:
                    raise PrimeCatalogResourceError(
                        resource_name=resource_name,
                        detail=(
                            f"expected format {expected_format!r}, got "
                            f"{metadata.get('format')!r}"
                        ),
                    )
                if metadata.get("version") != _CATALOG_VERSION:
                    raise PrimeCatalogResourceError(
                        resource_name=resource_name,
                        detail=(
                            f"unsupported catalog version "
                            f"{metadata.get('version')!r}"
                        ),
                    )

                for raw_key in handle.keys():
                    key = _decode_key(raw_key, resource_name=resource_name)
                    tensor = handle.get_tensor(raw_key)
                    if tensor.dtype != torch.int64 or tensor.ndim != 1:
                        raise PrimeCatalogResourceError(
                            resource_name=resource_name,
                            detail=(
                                f"prime sequence {raw_key!r} must be a "
                                "one-dimensional int64 tensor"
                            ),
                        )
                    primes = tuple(int(value) for value in tensor.tolist())
                    _validate_primes(
                        key,
                        primes,
                        resource_name=resource_name,
                    )
                    if key in decoded:
                        raise PrimeCatalogResourceError(
                            resource_name=resource_name,
                            detail=f"duplicate decoded key {key!r}",
                        )
                    decoded[key] = primes
    except PrimeCatalogResourceError:
        raise
    except Exception as exc:
        raise PrimeCatalogResourceError(
            resource_name=resource_name,
            detail="resource could not be loaded",
        ) from exc

    if not decoded:
        raise PrimeCatalogResourceError(
            resource_name=resource_name,
            detail="catalog is empty",
        )
    return MappingProxyType(dict(sorted(decoded.items())))


@dataclass(frozen=True, slots=True)
class PrimeCatalog:
    """Process-local view of immutable packaged CKKS prime tables."""

    _scaling: PrimeTable
    _special: PrimeTable

    def scaling_primes(self, prime_bits: int, degree: int) -> list[int]:
        """Return a mutable copy of one ordered scaling-prime sequence."""

        return list(self._scaling[int(prime_bits), int(degree)])

    def special_primes(self, prime_bits: int, degree: int) -> list[int]:
        """Return a mutable copy of one ordered special-prime sequence."""

        return list(self._special[int(prime_bits), int(degree)])


    @property
    def scaling_keys(self) -> tuple[PrimeKey, ...]:
        """Return supported scaling-prime catalog keys."""

        return tuple(self._scaling)

    @property
    def special_keys(self) -> tuple[PrimeKey, ...]:
        """Return supported terminal/special-prime catalog keys."""

        return tuple(self._special)


@lru_cache(maxsize=1)
def get_prime_catalog() -> PrimeCatalog:
    """Load and validate the installed catalog once per process."""

    return PrimeCatalog(
        _scaling=_load_table(
            _SCALING_RESOURCE,
            expected_format="ckks-scaling-primes",
        ),
        _special=_load_table(
            _SPECIAL_RESOURCE,
            expected_format="ckks-special-primes",
        ),
    )
