"""Single-file Program persistence with optional, storage-preserving bindings."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Collection, Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open

from .safetensors import (
    ValueFileMetadata,
    _require_value_file,
    _write_safetensors,
)

if TYPE_CHECKING:
    from fhelium.compile import Compilation

COMPILATION_FILE_FORMAT = "fhelium-compilation"
COMPILATION_SCHEMA_VERSION = 1


def save_compilation(
    compilation: Compilation,
    path: str | os.PathLike[str],
    *,
    include_materials: bool | Collection[str] = False,
    overwrite: bool = False,
) -> ValueFileMetadata:
    """Save a Program and selected bound Tensors without serializing Python state.

    False omits all data; True includes every current binding; a collection
    selects bound symbols. Missing Program bindings may remain external.
    The payload retains strided views, shared storage, repeated Tensor objects,
    and conjugate/negative view bits. Only bytes covered by selected views are
    copied; holes in their shared storage are zero-filled. Original addresses,
    devices, autograd history, workspaces, reports, and executables are not saved.

    Data is unencrypted. Inclusion is opt-in and does not detect sensitive
    contents of arbitrary Tensors. Callers own synchronization with live writers.
    """
    from fhelium.compile import Compilation

    if not isinstance(compilation, Compilation):
        raise TypeError("save_compilation requires a Compilation")
    if include_materials is True:
        bindings = dict(compilation.material_bindings)
    elif include_materials is False:
        bindings = {}
    else:
        bindings = {
            name: compilation.material_bindings[name]
            for name in include_materials
        }
    payloads, tensors, symbols = _pack_bindings(bindings)
    metadata = ValueFileMetadata(
        file_schema_version=COMPILATION_SCHEMA_VERSION,
        value_schema_version=COMPILATION_SCHEMA_VERSION,
        value_type="Compilation",
        nbytes=sum(t.numel() * t.element_size() for t in bindings.values()),
        tensor_metadata=tensors,
        value_metadata={
            "program": compilation.program.to_text(),
            "bindings": symbols,
            "storages": {
                name: value.numel() for name, value in payloads.items()
            },
        },
    )
    _write_safetensors(
        payloads,
        {
            "fhelium.format": COMPILATION_FILE_FORMAT,
            "fhelium.schema_version": str(COMPILATION_SCHEMA_VERSION),
            "fhelium.manifest": json.dumps(
                asdict(metadata), sort_keys=True, allow_nan=False
            ),
        },
        path,
        overwrite=overwrite,
    )
    return metadata


def inspect_compilation(path: str | os.PathLike[str]) -> ValueFileMetadata:
    """Inspect Program and binding metadata without loading numerical payloads."""
    source = _require_value_file(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        return _metadata(handle)


def load_compilation(
    path: str | os.PathLike[str],
    *,
    device: torch.device | str = "cpu",
) -> Compilation:
    """Restore a Program and saved bindings; leave all other symbols unbound.

    Loading does not prepare resources, run passes, compile kernels, or execute
    the Program. Every stored byte storage is copied to independent writable
    memory on ``device`` once, so views continue to share storage after transfer.
    """
    from fhelium.compile import Compilation
    from fhelium.ir import Program

    source = _require_value_file(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = _metadata(handle)
        storages = {
            name: handle.get_tensor(name).to(device=device, copy=True)
            for name in metadata.value_metadata["storages"]
        }
    tensors: dict[str, torch.Tensor] = {}
    for name, item in metadata.tensor_metadata.items():
        value = torch.empty(0, dtype=_dtype(item["dtype"]), device=device).set_(
            storages[item["storage"]].untyped_storage(),
            item["offset"],
            tuple(item["shape"]),
            tuple(item["stride"]),
        )
        if item["conjugate"]:
            value = value.conj()
        if item["negative"]:
            value = torch.ops.aten._neg_view.default(value)
        value.requires_grad_(item["requires_grad"])
        tensors[name] = value
    return Compilation(
        Program.parse(
            metadata.value_metadata["program"], source_name=str(source)
        ),
        material_bindings={
            symbol: tensors[name]
            for symbol, name in metadata.value_metadata["bindings"].items()
        },
    )


def _pack_bindings(bindings: Mapping[str, torch.Tensor]):
    groups: dict[int, list[torch.Tensor]] = {}
    object_names: dict[int, str] = {}
    symbols: dict[str, str] = {}
    for symbol, value in bindings.items():
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(
                "Compilation persistence supports strided, non-quantized Tensors"
            )
        identity = id(value)
        if identity not in object_names:
            object_names[identity] = f"tensor{len(object_names)}"
            groups.setdefault(value.untyped_storage()._cdata, []).append(value)
        symbols[symbol] = object_names[identity]
    payloads: dict[str, torch.Tensor] = {}
    tensors: dict[str, dict[str, Any]] = {}
    for index, values in enumerate(groups.values()):
        storage_name = f"storage{index}"
        alignment = math.lcm(*(value.element_size() for value in values))
        origin = min(
            int(value.storage_offset()) * value.element_size()
            for value in values
        )
        origin -= origin % alignment
        end = max(
            (
                int(value.storage_offset()) * value.element_size()
                + _span(value.shape, value.stride(), value.element_size())
                for value in values
                if value.numel()
            ),
            default=origin,
        )
        target = torch.zeros(end - origin, dtype=torch.uint8, device="cpu")
        source = torch.empty(
            0, dtype=torch.uint8, device=values[0].device
        ).set_(
            values[0].untyped_storage(),
            0,
            (values[0].untyped_storage().nbytes(),),
            (1,),
        )
        for value in values:
            offset = int(value.storage_offset()) * value.element_size()
            size = value.element_size()
            _copy_view_bytes(value, source, target, offset, origin)
            tensors[object_names[id(value)]] = {
                "storage": storage_name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "offset": (offset - origin) // size,
                "conjugate": value.is_conj(),
                "negative": value.is_neg(),
                "requires_grad": value.requires_grad,
            }
        payloads[storage_name] = target
    return payloads, tensors, symbols


def _copy_view_bytes(value, source, target, offset: int, origin: int) -> None:
    if not value.numel():
        return
    dense = True
    expected = 1
    for stride, dimension in sorted(
        (stride, dimension)
        for dimension, stride in zip(value.shape, value.stride(), strict=True)
        if dimension > 1
    ):
        dense &= stride == expected
        expected *= dimension
    if dense:
        count = value.numel() * value.element_size()
        target[offset - origin : offset - origin + count].copy_(
            source[offset : offset + count]
        )
        return
    # Bounded indices also support overlapping and zero-stride views.
    width = value.element_size()
    for start in range(0, value.numel(), 65536):
        linear = torch.arange(
            start,
            min(start + 65536, value.numel()),
            device="cpu",
            dtype=torch.int64,
        )
        positions = torch.zeros_like(linear)
        for dimension, stride in zip(
            reversed(value.shape), reversed(value.stride()), strict=True
        ):
            positions += (linear % dimension) * stride
            linear = linear // dimension
        positions = (
            positions[:, None] * width
            + torch.arange(width, device="cpu", dtype=torch.int64)[None, :]
            + offset
        ).flatten()
        selected = source.index_select(0, positions.to(source.device)).cpu()
        target.index_copy_(0, positions - origin, selected)


def _span(shape, stride, width: int) -> int:
    return (
        0
        if 0 in shape
        else (
            1
            + sum(
                (dimension - 1) * step
                for dimension, step in zip(shape, stride, strict=True)
            )
        )
        * width
    )


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype) or str(dtype) != name:
        raise ValueError(f"Unsupported serialized Tensor dtype: {name!r}")
    return dtype


def _metadata(handle) -> ValueFileMetadata:
    header = handle.metadata() or {}
    if header.get("fhelium.format") != COMPILATION_FILE_FORMAT or header.get(
        "fhelium.schema_version"
    ) != str(COMPILATION_SCHEMA_VERSION):
        raise ValueError(
            "Unsupported Compilation file format or schema version"
        )
    metadata = ValueFileMetadata(**json.loads(header["fhelium.manifest"]))
    if (
        metadata.value_type != "Compilation"
        or metadata.file_schema_version != COMPILATION_SCHEMA_VERSION
        or metadata.value_schema_version != COMPILATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Compilation manifest version differs from the file header"
        )
    storages = metadata.value_metadata["storages"]
    if set(storages) != set(handle.keys()):
        raise ValueError(
            "Compilation byte storage names differ from the manifest"
        )
    for name, size in storages.items():
        view = handle.get_slice(name)
        if (
            type(size) is not int
            or size < 0
            or view.get_shape() != [size]
            or view.get_dtype() != "U8"
        ):
            raise ValueError(
                "Compilation byte storage extent differs from its payload"
            )
    sizes: dict[str, int] = {}
    for name, item in metadata.tensor_metadata.items():
        shape, stride, offset = item["shape"], item["stride"], item["offset"]
        width = torch.empty(
            (), dtype=_dtype(item["dtype"]), device="meta"
        ).element_size()
        if len(shape) != len(stride) or any(
            type(n) is not int or n < 0 for n in (*shape, *stride, offset)
        ):
            raise ValueError("Invalid serialized strided Tensor view")
        span = _span(shape, stride, width)
        if item["storage"] not in storages or (
            span and offset * width + span > storages[item["storage"]]
        ):
            raise ValueError("Serialized Tensor view exceeds its byte storage")
        if any(
            type(item[field]) is not bool
            for field in ("conjugate", "negative", "requires_grad")
        ):
            raise ValueError("Invalid serialized Tensor view flags")
        sizes[name] = math.prod(shape) * width
    bindings = metadata.value_metadata["bindings"]
    if any(
        not isinstance(symbol, str) or name not in sizes
        for symbol, name in bindings.items()
    ):
        raise ValueError("Compilation bindings refer to missing Tensor records")
    if metadata.nbytes != sum(sizes[name] for name in bindings.values()):
        raise ValueError(
            "Compilation logical byte count differs from its bindings"
        )
    if not isinstance(metadata.value_metadata["program"], str):
        raise ValueError("Compilation requires textual Program IR")
    return metadata


__all__ = ["inspect_compilation", "load_compilation", "save_compilation"]
