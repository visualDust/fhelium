#!/usr/bin/env python3
"""Generate the versioned CKKS prime resources shipped with FHElium.

This offline development/release tool enumerates the maintained scale widths,
message-prime widths, and power-of-two ring dimensions. It searches the
required residue class ``q = 1 mod 2N``, applies deterministic 64-bit
Miller--Rabin primality testing, preserves the historical alternating
scale-prime ordering, and validates every completed sequence.

The output is ``scale_primes_v1.safetensors`` for public Q scale rows and
``message_primes_v1.safetensors`` for the structural Q base and key-switch P
rows. Resource keys have the schema ``sb=<bits>;N=<ring_dimension>``. Writes
use canonical safetensors headers and atomic replacement so equal logical
catalogs have equal file hashes and readers never observe a partial resource.

Run this script only when reviewing a catalog revision. Existing
version-1 resources are protected unless ``--force`` is supplied. Runtime
loading and schema validation belong to :mod:`fhelium.config._prime_catalog`;
package import and engine construction do not invoke this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

_MILLER_RABIN_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
_SCALE_BITS = tuple(range(20, 55, 5))
_LOG_DEGREES = tuple(range(12, 18))
_MESSAGE_BITS = (28, 60)
_CATALOG_VERSION = "1"

PrimeKey = tuple[int, int]
PrimeTable = dict[PrimeKey, list[int]]


def is_prime_64(number: int) -> bool:
    """Deterministically test primality for an unsigned 64-bit integer."""

    if number >= 1 << 64:
        raise ValueError("is_prime_64 only accepts unsigned 64-bit integers")
    if number < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if number == prime:
            return True
        if number % prime == 0:
            return False

    odd_part = number - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1

    for base in _MILLER_RABIN_BASES_64:
        if base % number == 0:
            continue
        witness = pow(base, odd_part, number)
        if witness in (1, number - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = pow(witness, 2, number)
            if witness == number - 1:
                break
        else:
            return False
    return True


def is_ntt_prime(prime: int, degree: int) -> bool:
    """Return whether ``prime`` supports a negacyclic size-``degree`` NTT."""

    return (prime - 1) % (2 * degree) == 0 and is_prime_64(prime)


def align_ntt_candidate(start: int, degree: int, *, upward: bool) -> int:
    """Return the nearest candidate congruent to one modulo ``2 * degree``."""

    modulus = 2 * degree
    remainder = (int(start) - 1) % modulus
    if upward and remainder:
        return int(start) + modulus - remainder
    return int(start) - remainder


def find_next_prime(start: int, degree: int, *, upward: bool) -> int:
    """Search only the residue class that can support a size-``degree`` NTT."""

    modulus = 2 * degree
    step = modulus if upward else -modulus
    candidate = align_ntt_candidate(start, degree, upward=upward)
    while candidate > 2:
        if is_prime_64(candidate):
            return candidate
        candidate += step
    raise ValueError(
        f"No positive NTT prime found from {start} "
        f"while searching {'upward' if upward else 'downward'}."
    )


def generate_alternating_prime_sequence(
    *,
    scale_bits: int,
    degree: int,
    count: int,
) -> list[int]:
    """Generate the historical balanced scale-prime sequence deterministically."""

    scale = 1 << scale_bits
    upward_start = scale + 1
    downward_start = scale - 1

    nearest_up = find_next_prime(upward_start, degree, upward=True)
    nearest_down = find_next_prime(downward_start, degree, upward=False)
    up_error = nearest_up - scale
    down_error = scale - nearest_down

    # Preserve FHElium's existing sequence rule: begin opposite the nearest
    # candidate, then alternate while balancing cumulative scale deviation.
    upward = not (up_error < down_error)
    cumulative_scale = 1.0
    output: list[int] = []

    while len(output) < count:
        start = upward_start if upward else downward_start
        prime = find_next_prime(start, degree, upward=upward)
        current_deviation = scale / prime
        cumulative_scale = cumulative_scale**2 * current_deviation**2

        if upward:
            upward_start = prime + 2
            searched = int((cumulative_scale * scale) // 2 * 2 - 1)
            downward_start = min(downward_start, searched)
        else:
            downward_start = prime - 2
            searched = int((cumulative_scale * scale) // 2 * 2 + 1)
            upward_start = max(upward_start, searched)

        upward = not upward
        output.append(prime)

    return output


def generate_scale_sequence(
    scale_bits: int,
    degree: int,
    requested_count: int,
) -> list[int] | None:
    """Generate up to ``requested_count`` scale primes for one parameter key.

    Failed long searches are retried with half as many primes. ``None`` means
    that no sequence of at least two primes could be generated.
    """

    attempt = requested_count
    while attempt >= 2:
        try:
            return generate_alternating_prime_sequence(
                scale_bits=scale_bits,
                degree=degree,
                count=attempt,
            )
        except (OverflowError, ValueError, ZeroDivisionError):
            attempt //= 2
    return None


def generate_scale_catalog() -> PrimeTable:
    """Generate every supported scale-width and ring-dimension sequence."""

    inputs: list[tuple[int, int, int]] = []
    for log_degree in _LOG_DEGREES:
        degree = 1 << log_degree
        count = 64 if log_degree < 16 else 128
        inputs.extend((bits, degree, count) for bits in _SCALE_BITS)

    catalog: PrimeTable = {}
    for completed, (scale_bits, degree, count) in enumerate(inputs, start=1):
        primes = generate_scale_sequence(scale_bits, degree, count)
        if primes:
            catalog[scale_bits, degree] = primes
        if completed % 5 == 0 or completed == len(inputs):
            print(f"Generated scale-prime candidates {completed}/{len(inputs)}")

    failures = sorted(
        (bits, degree)
        for bits, degree, _ in inputs
        if (bits, degree) not in catalog
    )
    print(
        f"Generated {len(catalog)}/{len(inputs)} scale-prime sequences; "
        f"unsupported={failures}"
    )
    return catalog


def generate_message_catalog(*, count: int = 11) -> PrimeTable:
    """Generate descending structural-Q/P prime sequences for every ring."""

    catalog: PrimeTable = {}
    for message_bits in _MESSAGE_BITS:
        for log_degree in _LOG_DEGREES:
            degree = 1 << log_degree
            candidate = align_ntt_candidate(
                (1 << message_bits) - 1,
                degree,
                upward=False,
            )
            primes: list[int] = []
            while len(primes) < count:
                if is_prime_64(candidate):
                    primes.append(candidate)
                candidate -= 2 * degree
            catalog[message_bits, degree] = primes
        print(f"Generated message-prime candidates for {message_bits} bits")
    return catalog


def validate_catalog(catalog: PrimeTable) -> None:
    """Validate catalog keys, uniqueness, primality, and NTT congruence."""

    for (bits, degree), primes in catalog.items():
        if bits <= 0 or degree <= 0 or degree & (degree - 1):
            raise ValueError(f"Invalid catalog key {(bits, degree)!r}.")
        if not primes or len(primes) != len(set(primes)):
            raise ValueError(f"Invalid prime sequence for {(bits, degree)!r}.")
        for prime in primes:
            if not is_ntt_prime(prime, degree):
                raise ValueError(
                    f"Invalid NTT prime {prime} for {(bits, degree)!r}."
                )


def encode_catalog(catalog: PrimeTable) -> dict[str, torch.Tensor]:
    """Encode a prime table as canonical safetensors names and int64 vectors."""

    return {
        f"sb={bits};N={degree}": torch.tensor(primes, dtype=torch.int64)
        for (bits, degree), primes in sorted(catalog.items())
    }


def canonicalize_safetensors_header(path: Path) -> None:
    """Normalize JSON map ordering so equal catalogs have equal file hashes."""

    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValueError(f"Invalid safetensors output {path}.")
    header_length = struct.unpack("<Q", payload[:8])[0]
    header_end = 8 + header_length
    header = json.loads(payload[8:header_end].decode("utf-8"))

    metadata = header.pop("__metadata__", {})
    canonical = {
        "__metadata__": dict(sorted(metadata.items())),
        **{key: header[key] for key in sorted(header)},
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload[header_end:]
    )


def atomic_save(
    path: Path,
    catalog: PrimeTable,
    *,
    format_name: str,
) -> None:
    """Write one validated catalog resource without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            encode_catalog(catalog),
            temporary,
            metadata={"format": format_name, "version": _CATALOG_VERSION},
        )
        canonicalize_safetensors_header(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a generated resource."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse the output directory and replacement authorization."""

    default_output = (
        Path(__file__).resolve().parents[1] / "fhelium" / "config" / "resources"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing version-1 catalog resources",
    )
    return parser.parse_args()


def main() -> None:
    """Generate, validate, and atomically install both version-1 catalogs."""

    args = parse_args()
    scale_path = args.output_dir / "scale_primes_v1.safetensors"
    message_path = args.output_dir / "message_primes_v1.safetensors"
    existing = [path for path in (scale_path, message_path) if path.exists()]
    if existing and not args.force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing catalog resources: {formatted}. "
            "Pass --force after reviewing the generator change."
        )

    scale_catalog = generate_scale_catalog()
    message_catalog = generate_message_catalog()
    validate_catalog(scale_catalog)
    validate_catalog(message_catalog)

    atomic_save(
        scale_path,
        scale_catalog,
        format_name="ckks-scale-primes",
    )
    atomic_save(
        message_path,
        message_catalog,
        format_name="ckks-message-primes",
    )

    for path in (scale_path, message_path):
        print(f"Wrote {path} sha256={file_digest(path)}")


if __name__ == "__main__":
    main()
