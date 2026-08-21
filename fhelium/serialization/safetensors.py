"""Single-file safetensors persistence for FHElium values."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fhelium.core import SecretKey, TensorResident
from fhelium.serialization.value import (
    VALUE_SCHEMA_VERSION,
    ValueEnvelope,
    supported_value_types,
    validate_value_description,
)

FILE_FORMAT = "fhelium-value"
FILE_SCHEMA_VERSION = 1
_FORMAT_METADATA_KEY = "fhelium.format"
_SCHEMA_METADATA_KEY = "fhelium.schema_version"
_MANIFEST_METADATA_KEY = "fhelium.manifest"

T = TypeVar("T", bound=TensorResident)


@dataclass(frozen=True)
class ValueFileMetadata:
    """Validated metadata inspectable without materializing tensor payloads."""

    file_schema_version: int
    value_schema_version: int
    value_type: str
    context_id: str | None
    nbytes: int
    tensor_metadata: dict[str, dict[str, Any]]
    value_metadata: dict[str, Any]


def save_value(
    value: TensorResident,
    path: str | os.PathLike[str],
    *,
    allow_secret: bool = False,
    overwrite: bool = False,
) -> ValueFileMetadata:
    """Atomically save one value to the caller-selected file path.

    This function provides a versioned file representation, not a namespace,
    cache, encryption-at-rest policy, or storage manager. Secret-key material
    requires explicit opt-in and remains unencrypted unless the caller wraps
    this API in an appropriate security layer.
    """

    envelope = ValueEnvelope.from_value(value)
    if isinstance(value, SecretKey) and not allow_secret:
        raise PermissionError(
            "SecretKey persistence is disabled by default; pass "
            "allow_secret=True only when the selected path has an appropriate "
            "at-rest security policy."
        )
    destination = Path(path).expanduser()
    parent = destination.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"Value file parent directory does not exist: {parent}"
        )
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    if destination.exists():
        if destination.is_dir():
            raise IsADirectoryError(destination)
        if not overwrite:
            raise FileExistsError(destination)

    logical_tensors, payload_tensors, tensor_metadata = _pack_tensors(
        envelope.tensors
    )
    nbytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in logical_tensors.values()
    )
    manifest = {
        "file_schema_version": FILE_SCHEMA_VERSION,
        "value_schema_version": envelope.schema_version,
        "value_type": envelope.value_type,
        "context_id": envelope.context_id,
        "nbytes": nbytes,
        "tensor_metadata": tensor_metadata,
        "value_metadata": envelope.metadata,
    }
    metadata = _metadata_from_manifest(manifest)
    safetensors_metadata = {
        _FORMAT_METADATA_KEY: FILE_FORMAT,
        _SCHEMA_METADATA_KEY: str(FILE_SCHEMA_VERSION),
        _MANIFEST_METADATA_KEY: json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            payload_tensors,
            str(temporary),
            metadata=safetensors_metadata,
        )
        os.chmod(temporary, 0o600)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return metadata


def inspect_value(
    path: str | os.PathLike[str],
) -> ValueFileMetadata:
    """Inspect one value file without materializing its tensor payloads."""

    source = _require_value_file(path)
    try:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = _metadata_from_safetensors(handle.metadata())
            _validate_payload_names(set(handle.keys()), metadata)
            _validate_payload_headers(handle, metadata)
            return metadata
    except Exception as error:
        if isinstance(error, (TypeError, ValueError, OSError)):
            raise
        raise ValueError(f"Invalid FHElium value file: {source}") from error


def load_value(
    path: str | os.PathLike[str],
    *,
    device: torch.device | str = "cpu",
    expected_type: type[T] | None = None,
    expected_context_id: str | None = None,
) -> T:
    """Load one value from a caller-selected value-file path.

    This is a file-codec operation: the caller owns path naming, replacement,
    and lifecycle. ``ArtifactStore.get`` is the separate repository operation
    for logical names, generations, checksums, and catalog transactions.
    Materialization defaults to CPU unless ``device`` selects another target.
    """

    source = _require_value_file(path)
    target = torch.device(device)
    try:
        with safe_open(
            str(source),
            framework="pt",
            device=str(target),
        ) as handle:
            metadata = _metadata_from_safetensors(handle.metadata())
            if (
                expected_context_id is not None
                and metadata.context_id != expected_context_id
            ):
                raise ValueError(
                    "Value file context mismatch: expected "
                    f"{expected_context_id!r}, got {metadata.context_id!r}"
                )
            if (
                expected_type is not None
                and metadata.value_type != expected_type.__name__
            ):
                raise TypeError(
                    f"Value file has type {metadata.value_type}, "
                    f"expected {expected_type.__name__}"
                )
            payload_names = set(handle.keys())
            _validate_payload_names(payload_names, metadata)
            _validate_payload_headers(handle, metadata)
            physical = {name: handle.get_tensor(name) for name in payload_names}
    except Exception as error:
        if isinstance(error, (TypeError, ValueError, OSError)):
            raise
        raise ValueError(f"Invalid FHElium value file: {source}") from error

    logical = _unpack_tensors(
        physical,
        metadata.tensor_metadata,
        source_name=str(source),
    )
    actual_nbytes = sum(
        tensor.numel() * tensor.element_size() for tensor in logical.values()
    )
    if actual_nbytes != metadata.nbytes:
        raise ValueError(
            "Value file byte count mismatch: expected "
            f"{metadata.nbytes}, got {actual_nbytes}"
        )
    envelope = ValueEnvelope(
        schema_version=metadata.value_schema_version,
        value_type=metadata.value_type,
        context_id=metadata.context_id,
        metadata=metadata.value_metadata,
        tensors=logical,
    )
    value = envelope.to_value()
    if expected_type is not None and type(value) is not expected_type:
        raise TypeError(
            f"Value file has type {type(value).__name__}, "
            f"expected {expected_type.__name__}"
        )
    return cast(T, value)


def _require_value_file(path: str | os.PathLike[str]) -> Path:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"No FHElium value file at {source}")
    return source


def _metadata_from_safetensors(
    raw_metadata: dict[str, str] | None,
) -> ValueFileMetadata:
    if raw_metadata is None:
        raise ValueError("Value file is missing safetensors metadata")
    if raw_metadata.get(_FORMAT_METADATA_KEY) != FILE_FORMAT:
        raise ValueError(
            "Unsupported value file format: "
            f"{raw_metadata.get(_FORMAT_METADATA_KEY)!r}"
        )
    if raw_metadata.get(_SCHEMA_METADATA_KEY) != str(FILE_SCHEMA_VERSION):
        raise ValueError(
            "Unsupported value file schema version: "
            f"{raw_metadata.get(_SCHEMA_METADATA_KEY)!r}"
        )
    serialized_manifest = raw_metadata.get(_MANIFEST_METADATA_KEY)
    if serialized_manifest is None:
        raise ValueError("Value file is missing its embedded manifest")
    try:
        manifest = json.loads(serialized_manifest)
    except json.JSONDecodeError as error:
        raise ValueError("Value file manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("Value file manifest must be a JSON object")
    return _metadata_from_manifest(manifest)


def _metadata_from_manifest(manifest: dict[str, Any]) -> ValueFileMetadata:
    required = {
        "file_schema_version",
        "value_schema_version",
        "value_type",
        "context_id",
        "nbytes",
        "tensor_metadata",
        "value_metadata",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(
            f"Value file manifest is missing fields: {sorted(missing)}"
        )
    unexpected = set(manifest).difference(required)
    if unexpected:
        raise ValueError(
            f"Value file manifest has unexpected fields: {sorted(unexpected)}"
        )
    if (
        type(manifest["file_schema_version"]) is not int
        or manifest["file_schema_version"] != FILE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported value file schema version: "
            f"{manifest['file_schema_version']!r}"
        )
    if (
        type(manifest["value_schema_version"]) is not int
        or manifest["value_schema_version"] != VALUE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported value envelope schema version: "
            f"{manifest['value_schema_version']!r}"
        )
    value_type = manifest["value_type"]
    if (
        not isinstance(value_type, str)
        or value_type not in supported_value_types()
    ):
        raise ValueError(f"Unsupported value file type: {value_type!r}")
    context_id = manifest["context_id"]
    if context_id is not None and not isinstance(context_id, str):
        raise ValueError("Value file context_id must be a string or null")
    nbytes = manifest["nbytes"]
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("Value file nbytes must be a non-negative integer")
    tensor_metadata = manifest["tensor_metadata"]
    value_metadata = manifest["value_metadata"]
    if not isinstance(tensor_metadata, dict) or not isinstance(
        value_metadata, dict
    ):
        raise ValueError("Value file metadata fields must be JSON objects")
    _validate_tensor_metadata(tensor_metadata)
    validate_value_description(
        schema_version=manifest["value_schema_version"],
        value_type=value_type,
        context_id=context_id,
        metadata=value_metadata,
        tensor_names=set(tensor_metadata),
        tensor_metadata=tensor_metadata,
    )
    return ValueFileMetadata(
        file_schema_version=FILE_SCHEMA_VERSION,
        value_schema_version=VALUE_SCHEMA_VERSION,
        value_type=value_type,
        context_id=context_id,
        nbytes=nbytes,
        tensor_metadata=tensor_metadata,
        value_metadata=value_metadata,
    )


def _pack_tensors(
    tensors: dict[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, dict[str, Any]],
]:
    logical = {}
    physical = {}
    metadata = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Value tensor names must be non-empty strings")
        if tensor.layout != torch.strided:
            raise TypeError(
                f"Value tensor {name!r} must use dense strided storage"
            )
        snapshot = tensor.detach().to("cpu").contiguous()
        logical[name] = snapshot
        item = {
            "shape": list(snapshot.shape),
            "dtype": str(snapshot.dtype),
        }
        if snapshot.is_complex():
            real_name = f"{name}.real"
            imag_name = f"{name}.imag"
            physical[real_name] = snapshot.real.contiguous()
            physical[imag_name] = snapshot.imag.contiguous()
            item.update(
                {
                    "encoding": "complex_split",
                    "payloads": [real_name, imag_name],
                }
            )
        else:
            physical[name] = snapshot
            item.update({"encoding": "direct", "payloads": [name]})
        metadata[name] = item
    if not physical:
        raise ValueError(
            "A value file must contain at least one tensor payload"
        )
    return logical, physical, metadata


def _unpack_tensors(
    physical: dict[str, torch.Tensor],
    metadata: dict[str, dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, torch.Tensor]:
    logical = {}
    expected_payloads = {
        payload for item in metadata.values() for payload in item["payloads"]
    }
    unexpected_payloads = set(physical).symmetric_difference(expected_payloads)
    if unexpected_payloads:
        raise ValueError(
            f"Value file {source_name!r} payload set does not match its "
            f"manifest: {sorted(unexpected_payloads)}"
        )
    for name, item in metadata.items():
        encoding = item["encoding"]
        payloads = item["payloads"]
        try:
            if encoding == "direct":
                tensor = physical[payloads[0]]
            elif encoding == "complex_split":
                real = physical[payloads[0]]
                imag = physical[payloads[1]]
                tensor = torch.complex(real, imag)
            else:  # validated before payload materialization
                raise ValueError(
                    f"Unsupported tensor encoding {encoding!r} in "
                    f"value file {source_name!r}"
                )
        except KeyError as error:
            raise ValueError(
                f"Value file {source_name!r} is missing tensor payload "
                f"{error.args[0]!r}"
            ) from error
        expected_shape = tuple(int(i) for i in item["shape"])
        expected_dtype = str(item["dtype"])
        if (
            tuple(tensor.shape) != expected_shape
            or str(tensor.dtype) != expected_dtype
        ):
            raise ValueError(
                f"Value file {source_name!r} tensor {name!r} metadata "
                "mismatch: expected "
                f"shape={expected_shape}, dtype={expected_dtype}; got "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
        logical[name] = tensor
    return logical


def _validate_payload_names(
    payload_names: set[str],
    metadata: ValueFileMetadata,
) -> None:
    expected = {
        payload
        for item in metadata.tensor_metadata.values()
        for payload in item["payloads"]
    }
    if payload_names != expected:
        raise ValueError(
            "Value file payload set does not match its manifest: "
            f"expected={sorted(expected)}, actual={sorted(payload_names)}"
        )


_SAFETENSORS_DTYPE_BY_TORCH = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.uint16": "U16",
    "torch.uint32": "U32",
    "torch.uint64": "U64",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
    "torch.complex64": "F32",
    "torch.complex128": "F64",
}


def _validate_payload_headers(
    handle: Any,
    metadata: ValueFileMetadata,
) -> None:
    """Compare manifest claims with safetensors headers without loading data."""

    for logical_name, item in metadata.tensor_metadata.items():
        expected_shape = tuple(item["shape"])
        logical_dtype = item["dtype"]
        try:
            expected_dtype = _SAFETENSORS_DTYPE_BY_TORCH[logical_dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported logical tensor dtype {logical_dtype!r} for "
                f"{logical_name!r}"
            ) from error
        for payload_name in item["payloads"]:
            payload_slice = handle.get_slice(payload_name)
            actual_shape = tuple(payload_slice.get_shape())
            actual_dtype = payload_slice.get_dtype()
            if actual_shape != expected_shape or actual_dtype != expected_dtype:
                raise ValueError(
                    "Value file payload header does not match logical tensor "
                    f"metadata for {logical_name!r}/{payload_name!r}: "
                    f"expected shape={expected_shape}, dtype={expected_dtype}; "
                    f"actual shape={actual_shape}, dtype={actual_dtype}"
                )


def _validate_tensor_metadata(metadata: dict[str, Any]) -> None:
    if not metadata:
        raise ValueError("Value file tensor_metadata cannot be empty")
    all_payload_names = []
    for name, item in metadata.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Value tensor metadata names must be non-empty strings"
            )
        if not isinstance(item, dict):
            raise ValueError(
                f"Value tensor metadata for {name!r} must be an object"
            )
        required = {"shape", "dtype", "encoding", "payloads"}
        missing = required.difference(item)
        if missing:
            raise ValueError(
                f"Value tensor metadata for {name!r} is missing fields: "
                f"{sorted(missing)}"
            )
        unexpected = set(item).difference(required)
        if unexpected:
            raise ValueError(
                f"Value tensor metadata for {name!r} has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        shape = item["shape"]
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ValueError(
                f"Value tensor metadata shape for {name!r} must be a "
                "non-negative integer list"
            )
        dtype = item["dtype"]
        if not isinstance(dtype, str):
            raise ValueError(
                f"Value tensor metadata dtype for {name!r} must be a string"
            )
        if dtype not in _SAFETENSORS_DTYPE_BY_TORCH:
            raise ValueError(
                f"Unsupported logical tensor dtype {dtype!r} for {name!r}"
            )
        encoding = item["encoding"]
        if encoding not in ("direct", "complex_split"):
            raise ValueError(
                f"Unsupported tensor encoding {encoding!r} for {name!r}"
            )
        is_complex = dtype in ("torch.complex64", "torch.complex128")
        if is_complex != (encoding == "complex_split"):
            raise ValueError(
                f"Value tensor metadata for {name!r} must use "
                f"{'complex_split' if is_complex else 'direct'} encoding "
                f"for dtype {dtype}"
            )
        expected_payload_count = 2 if encoding == "complex_split" else 1
        payloads = item["payloads"]
        if (
            not isinstance(payloads, list)
            or len(payloads) != expected_payload_count
            or any(
                not isinstance(payload, str) or not payload
                for payload in payloads
            )
            or len(set(payloads)) != len(payloads)
        ):
            raise ValueError(
                f"Value tensor payload list for {name!r} is invalid"
            )
        all_payload_names.extend(payloads)
    if len(set(all_payload_names)) != len(all_payload_names):
        raise ValueError("Value tensor payload names must be globally unique")
