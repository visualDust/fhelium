"""Structural signatures for reusable FHElium execution values.

Signatures describe copy-compatible tensor topology and FHElium value
state. They deliberately exclude tensor residency: a CPU value and a CUDA value
may have the same signature and can therefore be copied through a
:class:`~fhelium.execution.ReusableValueBuffer` whose target device is tracked
separately.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, NoReturn

import torch

from fhelium.core import TensorResident
from fhelium.errors import ExecutionInputError
from fhelium.serialization import ValueEnvelope

TreeKind = Literal["tensor", "value", "list", "tuple", "dict"]


@dataclass(frozen=True)
class TensorSignature:
    """Device-independent tensor topology accepted by one execution buffer.

    ``device`` is intentionally absent. Shape, stride, dtype, layout, and
    ``requires_grad`` determine whether payload can be copied into fixed target
    storage; the target residency belongs to the buffer or program that owns
    that storage.

    Args:
        shape: Tensor dimensions.
        stride: Element strides required by fixed target storage.
        dtype: Tensor scalar dtype.
        layout: PyTorch tensor layout, normally ``torch.strided``.
        requires_grad: Autograd flag expected by the reusable payload.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    layout: torch.layout
    requires_grad: bool

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorSignature:
        """Build a copy-compatibility signature for ``tensor``.

        Args:
            tensor: Representative tensor whose device-independent topology is
                captured.

        Returns:
            Signature excluding the tensor's current device.
        """

        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )


@dataclass(frozen=True)
class ValueSignature:
    """FHElium value state plus device-independent tensor topology.

    A value signature fixes cryptographic metadata such as context, level,
    scale, plaintext representation, polynomial domain, modulus basis, residue
    representation, prime identities, schema version, and key identity.
    A value signature is storage-independent metadata. Model, user, request,
    and cache associations remain external.

    Args:
        type_name: Registered serialized-value class name.
        schema_version: Value-serialization schema version.
        context_id: Cryptographic context identity, if the value has one.
        metadata: Frozen, deterministically ordered non-tensor state.
        tensors: Ordered tensor names and their device-independent signatures.
    """

    type_name: str
    schema_version: int
    context_id: str | None
    metadata: tuple[tuple[str, object], ...]
    tensors: tuple[tuple[str, TensorSignature], ...]

    @classmethod
    def from_value(cls, value: TensorResident) -> ValueSignature:
        """Describe one serializable FHElium value.

        Args:
            value: Resident value whose type, metadata, and tensor
                topology are captured.

        Returns:
            Device-independent value signature.
        """

        return _value_signature_from_envelope(ValueEnvelope.from_value(value))


def _value_signature_from_envelope(envelope: ValueEnvelope) -> ValueSignature:
    """Compile value state and tensor topology from one existing envelope."""

    return ValueSignature(
        type_name=envelope.value_type,
        schema_version=envelope.schema_version,
        context_id=envelope.context_id,
        metadata=tuple(
            (name, _freeze_metadata(item))
            for name, item in sorted(envelope.metadata.items())
        ),
        tensors=tuple(
            (name, TensorSignature.from_tensor(tensor))
            for name, tensor in sorted(envelope.tensors.items())
        ),
    )


@dataclass(frozen=True)
class ValueTreeSignature:
    """Structure of tensors and FHElium values in a reusable execution payload.

    Supported leaves are :class:`torch.Tensor` and serializable FHElium
    :class:`~fhelium.core.TensorResident` values. Lists, tuples, and
    dictionaries may nest those leaves. Arbitrary Python scalars and control
    objects are intentionally unsupported; bind them statically in a callable
    or keep them in the application control plane.

    ``ValueTreeSignature`` composes :class:`TensorSignature` and
    :class:`ValueSignature` rather than replacing them: tensor leaves need only
    tensor topology, while FHElium-value leaves additionally require
    cryptographic metadata.

    Args:
        kind: Node kind: tensor, FHElium value, list, tuple, or dictionary.
        leaf: Tensor/value signature for a leaf node; ``None`` for containers.
        children: Ordered signatures nested by a container node.
        keys: Dictionary keys in the same order as ``children``; empty for all
            other node kinds.
    """

    kind: TreeKind
    leaf: TensorSignature | ValueSignature | None = None
    children: tuple[ValueTreeSignature, ...] = ()
    keys: tuple[object, ...] = ()

    @classmethod
    def from_value(cls, value: object) -> ValueTreeSignature:
        """Describe a supported tensor/value tree.

        Args:
            value: Representative tensor/value tree to describe.

        Returns:
            Recursive device-independent structure and state signature.

        Raises:
            TypeError: If a leaf is not a tensor or serializable FHElium
                value, or if a container is not a list, tuple, or dictionary.
        """

        return _build_value_tree_signature(value)

    def validate(self, value: object, *, path: str = "value") -> None:
        """Require ``value`` to have this structure and state.

        Tensor devices may differ because signatures describe transfer
        compatibility, not residency. Validation completes for the full tree
        before :class:`~fhelium.execution.ReusableValueBuffer` copies any
        payload.

        Args:
            value: Candidate tree to compare with this signature.
            path: Root label used in mismatch diagnostics.

        Raises:
            ExecutionInputError: If structure, tensor topology, or value
                metadata differs.
        """

        _collect_matching_tensors(value, self, [], path=path)


def _freeze_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            (key, _freeze_metadata(item)) for key, item in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "Unsupported serialized metadata in execution signature: "
        f"{type(value).__name__}"
    )


def _build_value_tree_signature(
    value: object,
    tensors: list[torch.Tensor] | None = None,
    value_envelopes: list[ValueEnvelope] | None = None,
) -> ValueTreeSignature:
    """Compile one tree, optionally retaining its already-inspected leaves."""

    if isinstance(value, TensorResident):
        envelope = ValueEnvelope.from_value(value)
        if not envelope.tensors:
            raise ValueError(
                f"Dynamic {type(value).__name__} value has no resident tensors"
            )
        leaf = _value_signature_from_envelope(envelope)
        if tensors is not None:
            tensors.extend(envelope.tensors[name] for name, _ in leaf.tensors)
        if value_envelopes is not None:
            value_envelopes.append(envelope)
        return ValueTreeSignature(kind="value", leaf=leaf)
    if isinstance(value, torch.Tensor):
        if tensors is not None:
            tensors.append(value)
        return ValueTreeSignature(
            kind="tensor", leaf=TensorSignature.from_tensor(value)
        )
    if isinstance(value, list):
        return ValueTreeSignature(
            kind="list",
            children=tuple(
                _build_value_tree_signature(item, tensors, value_envelopes)
                for item in value
            ),
        )
    if isinstance(value, tuple):
        return ValueTreeSignature(
            kind="tuple",
            children=tuple(
                _build_value_tree_signature(item, tensors, value_envelopes)
                for item in value
            ),
        )
    if isinstance(value, dict):
        keys = tuple(value)
        return ValueTreeSignature(
            kind="dict",
            children=tuple(
                _build_value_tree_signature(
                    value[key], tensors, value_envelopes
                )
                for key in keys
            ),
            keys=keys,
        )
    raise TypeError(
        "Execution value trees must contain tensors, serializable "
        "TensorResident values, or nested list/tuple/dict containers; "
        f"got {type(value).__name__}"
    )


def _collect_matching_tensors(
    value: object,
    signature: ValueTreeSignature,
    tensors: list[torch.Tensor],
    *,
    path: str,
    expected_envelopes: Iterator[ValueEnvelope] | None = None,
) -> None:
    if signature.kind == "value":
        if not isinstance(value, TensorResident):
            _structure_mismatch(path, signature, value)
        envelope = ValueEnvelope.from_value(value)
        expected = signature.leaf
        if not isinstance(expected, ValueSignature):
            raise RuntimeError("Value tree has an invalid value signature")
        if expected_envelopes is None:
            matches = _value_signature_from_envelope(envelope) == expected
        else:
            try:
                expected_envelope = next(expected_envelopes)
            except StopIteration as error:
                raise RuntimeError(
                    "Compiled execution envelopes do not match the value tree"
                ) from error
            matches = _envelope_matches_signature(
                envelope,
                expected_envelope,
                expected,
            )
        if not matches:
            actual = _value_signature_from_envelope(envelope)
            raise ExecutionInputError(
                f"Execution {path} value signature differs from the buffer: "
                f"expected={expected}, actual={actual}"
            )
        tensors.extend(envelope.tensors[name] for name, _ in expected.tensors)
        return
    if signature.kind == "tensor":
        if not isinstance(value, torch.Tensor):
            _structure_mismatch(path, signature, value)
        expected = signature.leaf
        if not isinstance(expected, TensorSignature):
            raise RuntimeError("Value tree has an invalid tensor signature")
        if not _tensor_matches_signature(value, expected):
            actual = TensorSignature.from_tensor(value)
            raise ExecutionInputError(
                f"Execution {path} tensor signature differs from the buffer: "
                f"expected={expected}, actual={actual}"
            )
        tensors.append(value)
        return
    if signature.kind in {"list", "tuple"}:
        expected_type = list if signature.kind == "list" else tuple
        if not isinstance(value, expected_type):
            _structure_mismatch(path, signature, value)
        if len(value) != len(signature.children):
            raise ExecutionInputError(
                f"Execution {path} length differs from the buffer: "
                f"expected={len(signature.children)}, actual={len(value)}"
            )
        for index, (item, child) in enumerate(
            zip(value, signature.children, strict=True)
        ):
            _collect_matching_tensors(
                item,
                child,
                tensors,
                path=f"{path}[{index}]",
                expected_envelopes=expected_envelopes,
            )
        return
    if signature.kind == "dict":
        if not isinstance(value, dict):
            _structure_mismatch(path, signature, value)
        if len(value) != len(signature.keys) or set(value) != set(
            signature.keys
        ):
            raise ExecutionInputError(
                f"Execution {path} dictionary keys differ from the buffer: "
                f"expected={signature.keys}, actual={tuple(value)}"
            )
        for key, child in zip(signature.keys, signature.children, strict=True):
            _collect_matching_tensors(
                value[key],
                child,
                tensors,
                path=f"{path}[{key!r}]",
                expected_envelopes=expected_envelopes,
            )
        return
    raise RuntimeError(f"Unknown execution value-tree kind {signature.kind!r}")


def _envelope_matches_signature(
    envelope: ValueEnvelope,
    expected_envelope: ValueEnvelope,
    expected_signature: ValueSignature,
) -> bool:
    """Compare fresh value state with one private immutable buffer template."""

    if (
        envelope.value_type != expected_envelope.value_type
        or envelope.schema_version != expected_envelope.schema_version
        or envelope.context_id != expected_envelope.context_id
        or envelope.metadata != expected_envelope.metadata
        or len(envelope.tensors) != len(expected_signature.tensors)
    ):
        return False
    for name, tensor_signature in expected_signature.tensors:
        tensor = envelope.tensors.get(name)
        if tensor is None or not _tensor_matches_signature(
            tensor, tensor_signature
        ):
            return False
    return True


def _tensor_matches_signature(
    tensor: torch.Tensor,
    expected: TensorSignature,
) -> bool:
    return (
        tuple(tensor.shape) == expected.shape
        and tuple(tensor.stride()) == expected.stride
        and tensor.dtype == expected.dtype
        and tensor.layout == expected.layout
        and tensor.requires_grad == expected.requires_grad
    )


def _structure_mismatch(
    path: str,
    signature: ValueTreeSignature,
    value: object,
) -> NoReturn:
    raise ExecutionInputError(
        f"Execution {path} structure differs from the buffer: "
        f"expected={signature.kind}, actual={type(value).__name__}"
    )


__all__ = [
    "TensorSignature",
    "ValueSignature",
    "ValueTreeSignature",
]
