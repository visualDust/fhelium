"""Exact typed value envelopes, validation, and reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self, cast

import torch

from fhelium.core import (
    COMPRESSED_PLAINTEXT_FORMAT_VERSION,
    Ciphertext,
    CompressedPlaintext,
    ConjugationKey,
    KeySwitchKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
    TensorResident,
)
from fhelium.core.state import (
    CompressedPlaintextLayout,
    ModulusBasis,
    PlaintextRepresentation,
    PolynomialDomain,
    ResidueRepresentation,
)

VALUE_SCHEMA_VERSION = 2

_KEY_TYPES: dict[str, type] = {
    key_type.__name__: key_type
    for key_type in (
        SecretKey,
        PublicKey,
        KeySwitchKey,
        RotationKey,
        RelinearizationKey,
        ConjugationKey,
    )
}


@dataclass(frozen=True)
class ValueEnvelope:
    """An exact value description with tensors but no path or store policy.

    The envelope is the shared representation for application-owned storage and
    memory managers. Its tensors may be moved or persisted by the caller, then passed
    to :meth:`to_value` to reconstruct the exact FHElium value type.
    """

    schema_version: int
    value_type: str
    context_id: str | None
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]

    @classmethod
    def from_value(cls, value: TensorResident) -> Self:
        """Describe one exact live value without choosing storage policy."""

        return cast(Self, _envelope_from_value(value, cls))

    def to_value(self) -> TensorResident:
        """Reconstruct the exact concrete FHElium value type."""

        return _value_from_envelope(self)

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.tensors.values()
        )


def _envelope_from_value(
    value: TensorResident,
    envelope_type: type[ValueEnvelope],
) -> ValueEnvelope:
    """Build the path-independent representation of one exact live value."""

    if isinstance(value, Plaintext):
        tensors = {}
        if value.message is not None:
            tensors["message"] = value.message
        if value.data is not None:
            tensors["data"] = value.data
        return envelope_type(
            schema_version=VALUE_SCHEMA_VERSION,
            value_type="Plaintext",
            context_id=value.context_id,
            metadata={
                "level": value.level,
                "scale": value.scale,
                "representation": value.representation,
                "polynomial_domain": value.polynomial_domain,
                "modulus_basis": value.modulus_basis,
                "residue_representation": value.residue_representation,
                "prime_ids": list(value.prime_ids),
                "has_message": value.message is not None,
                "has_data": value.data is not None,
            },
            tensors=tensors,
        )

    if isinstance(value, CompressedPlaintext):
        return envelope_type(
            schema_version=VALUE_SCHEMA_VERSION,
            value_type="CompressedPlaintext",
            context_id=value.context_id,
            metadata={
                "ring_dimension": value.ring_dimension,
                "compression_layout": value.compression_layout,
                "compression_format_version": (
                    value.compression_format_version
                ),
                "level": value.level,
                "scale": value.scale,
                "polynomial_domain": value.polynomial_domain,
                "modulus_basis": value.modulus_basis,
                "residue_representation": value.residue_representation,
                "prime_ids": list(value.prime_ids),
                "has_implicit_data": value.implicit_data is not None,
            },
            tensors={
                "data": value.data,
                **(
                    {}
                    if value.implicit_data is None
                    else {"implicit_data": value.implicit_data}
                ),
            },
        )

    if isinstance(value, Ciphertext):
        return envelope_type(
            schema_version=VALUE_SCHEMA_VERSION,
            value_type="Ciphertext",
            context_id=value.context_id,
            metadata={
                "level": value.level,
                "scale": value.scale,
                "prime_ids": list(value.prime_ids),
                "polynomial_domain": value.polynomial_domain,
                "modulus_basis": value.modulus_basis,
                "residue_representation": value.residue_representation,
            },
            tensors={"data": value.data},
        )

    key_type = type(value)
    if (
        key_type.__name__ in _KEY_TYPES
        and _KEY_TYPES[key_type.__name__] is key_type
    ):
        key_value = cast(SecretKey | PublicKey | KeySwitchKey, value)
        return envelope_type(
            schema_version=VALUE_SCHEMA_VERSION,
            value_type=key_type.__name__,
            context_id=key_value.context_id,
            metadata={
                "prime_ids": list(key_value.prime_ids),
                "polynomial_domain": key_value.polynomial_domain,
                "modulus_basis": key_value.modulus_basis,
                "residue_representation": key_value.residue_representation,
                **(
                    {"rotation_step": value.rotation_step}
                    if isinstance(value, RotationKey)
                    else {}
                ),
            },
            tensors={"data": key_value.data},
        )

    supported = [
        "Plaintext",
        "CompressedPlaintext",
        "Ciphertext",
        *_KEY_TYPES,
    ]
    raise TypeError(
        "Exact value serialization only supports FHElium values "
        f"{supported}; got {type(value).__name__}"
    )


def _value_from_envelope(envelope: ValueEnvelope) -> TensorResident:
    """Reconstruct one exact live value from a validated tensor envelope."""

    if not isinstance(envelope, ValueEnvelope):
        raise TypeError(
            "Value reconstruction requires a ValueEnvelope; got "
            f"{type(envelope).__name__}"
        )
    validate_value_description(
        schema_version=envelope.schema_version,
        value_type=envelope.value_type,
        context_id=envelope.context_id,
        metadata=envelope.metadata,
        tensor_names=set(envelope.tensors),
    )
    value_type = envelope.value_type
    metadata = envelope.metadata
    tensors = envelope.tensors
    context_id = envelope.context_id

    if value_type == "Plaintext":
        _require_metadata_fields(
            metadata,
            value_type,
            {
                "level",
                "scale",
                "representation",
                "polynomial_domain",
                "modulus_basis",
                "residue_representation",
                "prime_ids",
                "has_message",
                "has_data",
            },
        )
        message = tensors.get("message")
        data = tensors.get("data")
        has_message = _metadata_bool(metadata, "has_message", value_type)
        has_data = _metadata_bool(metadata, "has_data", value_type)
        expected_tensor_names = {
            name
            for name, present in (
                ("message", has_message),
                ("data", has_data),
            )
            if present
        }
        _require_tensor_names(tensors, value_type, expected_tensor_names)
        if (message is not None) != has_message:
            raise ValueError(
                "Plaintext envelope message payload is inconsistent"
            )
        if (data is not None) != has_data:
            raise ValueError("Plaintext envelope data payload is inconsistent")
        representation = _metadata_string(
            metadata, "representation", value_type
        )
        if representation not in (
            "slots",
            "integer_coefficients",
            "approximate_coefficients",
            "rns",
        ):
            raise ValueError(
                "Plaintext envelope metadata has an unsupported representation: "
                f"{representation!r}"
            )
        polynomial_domain = _metadata_optional_string(
            metadata, "polynomial_domain", value_type
        )
        if polynomial_domain not in (None, "coefficient", "ntt"):
            raise ValueError(
                "Plaintext envelope metadata has an unsupported domain: "
                f"{polynomial_domain!r}"
            )
        modulus_basis = _metadata_optional_string(
            metadata, "modulus_basis", value_type
        )
        return Plaintext(
            message=message,
            level=_metadata_integer(metadata, "level", value_type),
            scale=_metadata_scale(metadata, value_type),
            data=data,
            context_id=context_id,
            representation=cast(PlaintextRepresentation, representation),
            polynomial_domain=cast(PolynomialDomain | None, polynomial_domain),
            modulus_basis=cast(ModulusBasis | None, modulus_basis),
            residue_representation=_metadata_optional_residue_representation(
                metadata, value_type
            ),
            prime_ids=_metadata_prime_ids(metadata, value_type),
        )

    if value_type == "Ciphertext":
        _require_metadata_fields(
            metadata,
            value_type,
            {
                "level",
                "scale",
                "prime_ids",
                "polynomial_domain",
                "modulus_basis",
                "residue_representation",
            },
        )
        _require_tensor_names(tensors, value_type, {"data"})
        return Ciphertext(
            data=_required_tensor(tensors, "data", value_type),
            level=_metadata_integer(metadata, "level", value_type),
            scale=_metadata_scale(metadata, value_type),
            context_id=_required_context(context_id, value_type),
            prime_ids=_metadata_prime_ids(metadata, value_type),
            polynomial_domain=cast(
                PolynomialDomain,
                _metadata_string(metadata, "polynomial_domain", value_type),
            ),
            modulus_basis=cast(
                ModulusBasis,
                _metadata_string(metadata, "modulus_basis", value_type),
            ),
            residue_representation=_metadata_residue_representation(
                metadata, value_type
            ),
        )

    if value_type == "CompressedPlaintext":
        _require_metadata_fields(
            metadata,
            value_type,
            {
                "ring_dimension",
                "compression_layout",
                "compression_format_version",
                "level",
                "scale",
                "polynomial_domain",
                "modulus_basis",
                "residue_representation",
                "prime_ids",
                "has_implicit_data",
            },
        )
        has_implicit_data = _metadata_bool(
            metadata, "has_implicit_data", value_type
        )
        expected_tensor_names = {"data"}
        if has_implicit_data:
            expected_tensor_names.add("implicit_data")
        _require_tensor_names(tensors, value_type, expected_tensor_names)
        compression_layout = _metadata_string(
            metadata, "compression_layout", value_type
        )
        if compression_layout not in (
            "cyclic",
            "contiguous",
            "strided_sparse",
        ):
            raise ValueError(
                "CompressedPlaintext envelope metadata has an unsupported "
                f"compression_layout: {compression_layout!r}"
            )
        return CompressedPlaintext(
            data=_required_tensor(tensors, "data", value_type),
            ring_dimension=_metadata_integer(
                metadata, "ring_dimension", value_type
            ),
            compression_layout=cast(
                CompressedPlaintextLayout, compression_layout
            ),
            level=_metadata_integer(metadata, "level", value_type),
            scale=_metadata_scale(metadata, value_type),
            context_id=_required_context(context_id, value_type),
            polynomial_domain=cast(
                PolynomialDomain,
                _metadata_string(metadata, "polynomial_domain", value_type),
            ),
            modulus_basis=cast(
                ModulusBasis,
                _metadata_string(metadata, "modulus_basis", value_type),
            ),
            residue_representation=_metadata_residue_representation(
                metadata, value_type
            ),
            prime_ids=_metadata_prime_ids(metadata, value_type),
            implicit_data=tensors.get("implicit_data"),
            compression_format_version=_metadata_integer(
                metadata,
                "compression_format_version",
                value_type,
            ),
        )

    try:
        key_type = _KEY_TYPES[value_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported value envelope type: {value_type!r}"
        ) from error
    required_key_metadata = {
        "prime_ids",
        "polynomial_domain",
        "modulus_basis",
        "residue_representation",
    }
    if key_type is RotationKey:
        required_key_metadata.add("rotation_step")
    _require_metadata_fields(
        metadata,
        value_type,
        required_key_metadata,
    )
    _require_tensor_names(tensors, value_type, {"data"})
    key_arguments: dict[str, Any] = dict(
        data=_required_tensor(tensors, "data", value_type),
        context_id=_required_context(context_id, value_type),
        prime_ids=_metadata_prime_ids(metadata, value_type),
        polynomial_domain=_metadata_string(
            metadata, "polynomial_domain", value_type
        ),
        modulus_basis=_metadata_string(metadata, "modulus_basis", value_type),
        residue_representation=_metadata_residue_representation(
            metadata, value_type
        ),
    )
    if key_type is RotationKey:
        key_arguments["rotation_step"] = _metadata_integer(
            metadata, "rotation_step", value_type
        )
    return key_type(**key_arguments)


def supported_value_types() -> tuple[str, ...]:
    return ("Plaintext", "CompressedPlaintext", "Ciphertext", *_KEY_TYPES)


def validate_value_description(
    *,
    schema_version: object,
    value_type: object,
    context_id: object,
    metadata: object,
    tensor_names: set[str],
    tensor_metadata: dict[str, Any] | None = None,
) -> None:
    """Validate an exact tensor-free value schema description.

    Persistence inspection and envelope materialization share this validator,
    so stale metadata and tensor-name mismatches fail before payload loading.
    """

    if (
        type(schema_version) is not int
        or schema_version != VALUE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported value envelope schema version: {schema_version!r}"
        )
    if not isinstance(value_type, str):
        raise ValueError("Value envelope type must be a string")
    if not isinstance(metadata, dict):
        raise ValueError(f"{value_type} envelope metadata must be an object")
    if any(not isinstance(name, str) or not name for name in tensor_names):
        raise ValueError(
            "Value envelope tensor names must be non-empty strings"
        )
    typed_metadata = cast(dict[str, Any], metadata)

    if value_type == "Plaintext":
        fields = {
            "level",
            "scale",
            "representation",
            "polynomial_domain",
            "modulus_basis",
            "residue_representation",
            "prime_ids",
            "has_message",
            "has_data",
        }
        _require_metadata_fields(typed_metadata, value_type, fields)
        if _metadata_integer(typed_metadata, "level", value_type) < 0:
            raise ValueError("Plaintext envelope level must be non-negative")
        _metadata_scale(typed_metadata, value_type)
        has_message = _metadata_bool(typed_metadata, "has_message", value_type)
        has_data = _metadata_bool(typed_metadata, "has_data", value_type)
        if has_message == has_data:
            raise ValueError(
                "Plaintext envelope must describe exactly one payload form"
            )
        expected_names = {
            name
            for name, present in (("message", has_message), ("data", has_data))
            if present
        }
        _require_tensor_names_by_name(tensor_names, value_type, expected_names)
        representation = _metadata_string(
            typed_metadata, "representation", value_type
        )
        if representation not in (
            "slots",
            "integer_coefficients",
            "approximate_coefficients",
            "rns",
        ):
            raise ValueError(
                f"Unsupported Plaintext representation: {representation!r}"
            )
        if (representation == "slots") != has_message:
            raise ValueError(
                "Plaintext representation does not match its payload form"
            )
        polynomial_domain = _metadata_optional_string(
            typed_metadata, "polynomial_domain", value_type
        )
        modulus_basis = _metadata_optional_string(
            typed_metadata, "modulus_basis", value_type
        )
        residue = _metadata_optional_residue_representation(
            typed_metadata, value_type
        )
        prime_ids = _metadata_prime_ids(typed_metadata, value_type)
        if representation == "slots":
            if (
                any(
                    item is not None
                    for item in (polynomial_domain, modulus_basis, residue)
                )
                or prime_ids
            ):
                raise ValueError("Slots Plaintext cannot declare RNS metadata")
            if context_id is not None and (
                not isinstance(context_id, str) or not context_id
            ):
                raise ValueError(
                    "Plaintext context_id must be a non-empty string or null"
                )
        else:
            _required_context(cast(str | None, context_id), value_type)
            if polynomial_domain != "coefficient" and representation != "rns":
                raise ValueError(
                    "Coefficient Plaintext requires coefficient domain"
                )
            if representation == "rns":
                _validate_rns_metadata(
                    value_type,
                    polynomial_domain,
                    modulus_basis,
                    residue,
                    prime_ids,
                )
            elif modulus_basis is not None or residue is not None or prime_ids:
                raise ValueError(
                    "Coefficient Plaintext cannot declare RNS metadata"
                )
        return

    if value_type == "Ciphertext":
        fields = {
            "level",
            "scale",
            "prime_ids",
            "polynomial_domain",
            "modulus_basis",
            "residue_representation",
        }
        _require_metadata_fields(typed_metadata, value_type, fields)
        if _metadata_integer(typed_metadata, "level", value_type) < 0:
            raise ValueError("Ciphertext envelope level must be non-negative")
        _metadata_scale(typed_metadata, value_type)
        _validate_required_rns_metadata(typed_metadata, value_type)
        if (
            typed_metadata["polynomial_domain"] == "coefficient"
            and typed_metadata["residue_representation"] != "standard"
        ):
            raise ValueError(
                "Coefficient-domain Ciphertext requires standard residues"
            )
        _required_context(cast(str | None, context_id), value_type)
        _require_tensor_names_by_name(tensor_names, value_type, {"data"})
        return

    if value_type == "CompressedPlaintext":
        fields = {
            "ring_dimension",
            "compression_layout",
            "compression_format_version",
            "level",
            "scale",
            "polynomial_domain",
            "modulus_basis",
            "residue_representation",
            "prime_ids",
            "has_implicit_data",
        }
        _require_metadata_fields(typed_metadata, value_type, fields)
        for name in ("ring_dimension", "compression_format_version", "level"):
            if _metadata_integer(typed_metadata, name, value_type) < 0:
                raise ValueError(
                    f"CompressedPlaintext {name} must be non-negative"
                )
        ring_dimension = cast(int, typed_metadata["ring_dimension"])
        if ring_dimension <= 0 or ring_dimension & (ring_dimension - 1):
            raise ValueError(
                "CompressedPlaintext ring_dimension must be a positive power of two"
            )
        if (
            typed_metadata["compression_format_version"]
            != COMPRESSED_PLAINTEXT_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported CompressedPlaintext compression format version"
            )
        _metadata_scale(typed_metadata, value_type)
        layout = _metadata_string(
            typed_metadata, "compression_layout", value_type
        )
        if layout not in ("cyclic", "contiguous", "strided_sparse"):
            raise ValueError(f"Unsupported compression_layout: {layout!r}")
        has_implicit = _metadata_bool(
            typed_metadata, "has_implicit_data", value_type
        )
        expected_names = {"data"}
        if has_implicit:
            expected_names.add("implicit_data")
        _require_tensor_names_by_name(tensor_names, value_type, expected_names)
        if (layout == "strided_sparse") != has_implicit:
            raise ValueError(
                "CompressedPlaintext layout and implicit_data metadata disagree"
            )
        _validate_required_rns_metadata(typed_metadata, value_type)
        if typed_metadata["residue_representation"] != "montgomery":
            raise ValueError("CompressedPlaintext requires Montgomery residues")
        _required_context(cast(str | None, context_id), value_type)
        if tensor_metadata is not None:
            _validate_compressed_tensor_metadata(
                tensor_metadata,
                typed_metadata,
                has_implicit=has_implicit,
            )
        return

    try:
        key_type = _KEY_TYPES[value_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported value envelope type: {value_type!r}"
        ) from error
    fields = {
        "prime_ids",
        "polynomial_domain",
        "modulus_basis",
        "residue_representation",
    }
    if key_type is RotationKey:
        fields.add("rotation_step")
    _require_metadata_fields(typed_metadata, value_type, fields)
    if key_type is RotationKey:
        _metadata_integer(typed_metadata, "rotation_step", value_type)
    _validate_required_rns_metadata(typed_metadata, value_type)
    _required_context(cast(str | None, context_id), value_type)
    _require_tensor_names_by_name(tensor_names, value_type, {"data"})


def _validate_required_rns_metadata(
    metadata: dict[str, Any], value_type: str
) -> None:
    _validate_rns_metadata(
        value_type,
        _metadata_string(metadata, "polynomial_domain", value_type),
        _metadata_string(metadata, "modulus_basis", value_type),
        _metadata_residue_representation(metadata, value_type),
        _metadata_prime_ids(metadata, value_type),
    )


def _validate_rns_metadata(
    value_type: str,
    polynomial_domain: str | None,
    modulus_basis: str | None,
    residue_representation: str | None,
    prime_ids: tuple[int, ...],
) -> None:
    if polynomial_domain not in ("coefficient", "ntt"):
        raise ValueError(f"Unsupported {value_type} polynomial_domain")
    if modulus_basis not in ("Q", "QP"):
        raise ValueError(f"Unsupported {value_type} modulus_basis")
    if residue_representation not in ("standard", "montgomery"):
        raise ValueError(f"Unsupported {value_type} residue_representation")
    if polynomial_domain == "ntt" and residue_representation != "montgomery":
        raise ValueError(
            f"NTT-domain {value_type} requires Montgomery residues"
        )
    if not prime_ids:
        raise ValueError(f"{value_type} prime_ids cannot be empty")
    if any(a >= b for a, b in zip(prime_ids, prime_ids[1:])):
        raise ValueError(f"{value_type} prime_ids must be strictly increasing")


def _require_tensor_names_by_name(
    names: set[str], value_type: str, expected: set[str]
) -> None:
    if names != expected:
        raise ValueError(
            f"{value_type} envelope tensor names do not match its metadata: "
            f"expected={sorted(expected)}, actual={sorted(names)}"
        )


def _validate_compressed_tensor_metadata(
    tensor_metadata: dict[str, Any],
    metadata: dict[str, Any],
    *,
    has_implicit: bool,
) -> None:
    """Validate compressed logical shapes using manifest metadata only."""

    data_item = tensor_metadata.get("data")
    if not isinstance(data_item, dict):
        raise ValueError("CompressedPlaintext data tensor metadata is missing")
    shape = data_item.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        raise ValueError(
            "CompressedPlaintext data shape must describe limb and unique axes"
        )
    limb_count, unique_count = shape[-2:]
    if limb_count != len(_metadata_prime_ids(metadata, "CompressedPlaintext")):
        raise ValueError(
            "CompressedPlaintext logical limb count does not match prime_ids"
        )
    ring_dimension = cast(int, metadata["ring_dimension"])
    if (
        type(unique_count) is not int
        or unique_count <= 0
        or unique_count & (unique_count - 1)
        or unique_count >= ring_dimension
        or ring_dimension % unique_count
    ):
        raise ValueError(
            "CompressedPlaintext logical unique axis is incompatible with ring_dimension"
        )
    if has_implicit:
        implicit_item = tensor_metadata.get("implicit_data")
        if not isinstance(implicit_item, dict):
            raise ValueError(
                "CompressedPlaintext implicit tensor metadata is missing"
            )
        if implicit_item.get("shape") != shape[:-1]:
            raise ValueError(
                "CompressedPlaintext implicit logical shape is incompatible"
            )
        if implicit_item.get("dtype") != data_item.get("dtype"):
            raise ValueError(
                "CompressedPlaintext implicit logical dtype is incompatible"
            )


def _required_tensor(
    tensors: dict[str, torch.Tensor], name: str, value_type: str
) -> torch.Tensor:
    try:
        return tensors[name]
    except KeyError as error:
        raise ValueError(
            f"{value_type} envelope is missing tensor payload {name!r}"
        ) from error


def _required_context(context_id: str | None, value_type: str) -> str:
    if not isinstance(context_id, str) or not context_id:
        raise ValueError(
            f"{value_type} envelope requires a non-empty context_id"
        )
    return context_id


def _require_metadata_fields(
    metadata: dict[str, Any],
    value_type: str,
    required: set[str],
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError(f"{value_type} envelope metadata must be an object")
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            f"{value_type} envelope metadata is missing fields: "
            f"{sorted(missing)}"
        )
    unexpected = set(metadata).difference(required)
    if unexpected:
        raise ValueError(
            f"{value_type} envelope metadata has unexpected fields: "
            f"{sorted(unexpected)}"
        )


def _require_tensor_names(
    tensors: dict[str, torch.Tensor],
    value_type: str,
    expected: set[str],
) -> None:
    if set(tensors) != expected:
        raise ValueError(
            f"{value_type} envelope tensor names do not match its metadata: "
            f"expected={sorted(expected)}, actual={sorted(tensors)}"
        )


def _metadata_integer(
    metadata: dict[str, Any], name: str, value_type: str
) -> int:
    value = metadata[name]
    if type(value) is not int:
        raise ValueError(
            f"{value_type} envelope metadata {name!r} must be an integer"
        )
    return value


def _metadata_scale(metadata: dict[str, Any], value_type: str) -> float:
    value = metadata["scale"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{value_type} envelope metadata 'scale' must be numeric"
        )
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"{value_type} envelope metadata 'scale' must be finite and positive"
        )
    return scale


def _metadata_bool(
    metadata: dict[str, Any], name: str, value_type: str
) -> bool:
    value = metadata[name]
    if type(value) is not bool:
        raise ValueError(
            f"{value_type} envelope metadata {name!r} must be a boolean"
        )
    return value


def _metadata_string(
    metadata: dict[str, Any], name: str, value_type: str
) -> str:
    value = metadata[name]
    if not isinstance(value, str):
        raise ValueError(
            f"{value_type} envelope metadata {name!r} must be a string"
        )
    return value


def _metadata_optional_string(
    metadata: dict[str, Any], name: str, value_type: str
) -> str | None:
    value = metadata[name]
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"{value_type} envelope metadata {name!r} must be a string or null"
        )
    return value


def _metadata_residue_representation(
    metadata: dict[str, Any],
    value_type: str,
) -> ResidueRepresentation:
    value = _metadata_string(metadata, "residue_representation", value_type)
    if value not in ("standard", "montgomery"):
        raise ValueError(
            f"{value_type} envelope has unsupported residue representation: "
            f"{value!r}"
        )
    return cast(ResidueRepresentation, value)


def _metadata_optional_residue_representation(
    metadata: dict[str, Any],
    value_type: str,
) -> ResidueRepresentation | None:
    value = _metadata_optional_string(
        metadata,
        "residue_representation",
        value_type,
    )
    if value not in (None, "standard", "montgomery"):
        raise ValueError(
            f"{value_type} envelope has unsupported residue representation: "
            f"{value!r}"
        )
    return cast(ResidueRepresentation | None, value)


def _metadata_prime_ids(
    metadata: dict[str, Any], value_type: str
) -> tuple[int, ...]:
    values = metadata["prime_ids"]
    if not isinstance(values, list) or any(
        type(item) is not int or item < 0 for item in values
    ):
        raise ValueError(
            f"{value_type} envelope metadata 'prime_ids' must be an integer list"
        )
    return tuple(values)
