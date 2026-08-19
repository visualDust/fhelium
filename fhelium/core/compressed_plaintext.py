"""Exact operation-ready CKKS plaintexts with repeated encoded values."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhelium.core.plaintext import Plaintext
from fhelium.core.scale import coerce_scale
from fhelium.core.state import (
    CompressedPlaintextLayout,
    ModulusBasis,
    PolynomialDomain,
    ResidueRepresentation,
)
from fhelium.core.tensor_resident import TensorResident
from fhelium.core.validation import (
    validate_context_id,
    validate_integral_tensor,
    validate_nonnegative_level,
    validate_prime_ids,
)

COMPRESSED_PLAINTEXT_FORMAT_VERSION = 1

_SUPPORTED_COMPRESSION_LAYOUTS: tuple[CompressedPlaintextLayout, ...] = (
    "cyclic",
    "contiguous",
    "strided_sparse",
)


@dataclass(eq=False)
class CompressedPlaintext(TensorResident):
    r"""An exact operation-ready RNS plaintext with compressed storage.

    ``data`` is a dense integral tensor with layout
    ``[*batch, limb, unique_index]`` rather than the dense
    ``[*batch, limb, coefficient_or_ntt_index]`` layout used by
    :class:`Plaintext`. Limb row $i$ is modulo the prime identified by
    ``prime_ids[i]`` in $Q_\ell$ or $Q_\ell P$; ``polynomial_domain``
    determines whether the expanded last
    axis indexes coefficients or NTT evaluations. Operation-ready compressed
    values always use Montgomery residues. ``compression_layout`` defines the
    exact, lossless expansion of each compact row:

    - ``"cyclic"`` expands ``[a, b]`` as ``[a, b, a, b, ...]``;
    - ``"contiguous"`` expands ``[a, b]`` as
      ``[a, ..., a, b, ..., b]``.
    - ``"strided_sparse"``: compact values occupy positions separated by
      $N/U$, where $N$ is ``ring_dimension`` and $U$ is ``unique_count``; all
      other positions use the stored per-batch/per-limb ``implicit_data``
      value.

    These modes describe the encoded polynomial/NTT tensor axis, not the
    user-visible CKKS slot order. CKKS encoding permutes slots, and coefficient
    rounding can destroy repetition that exists only in semantic slot space.
    Construct this type from a dense operation-ready plaintext with
    :meth:`from_plaintext`; that conversion verifies exact representability.

    ``implicit_data`` is absent except for ``"strided_sparse"``, where it has
    layout ``[*batch, limb]`` and the same integral dtype and device as
    ``data``. Direct construction retains supplied storage. :meth:`clone` and
    decompression allocate independent storage; batch selection and unbinding
    return storage-sharing views. All batch entries share level, actual scale
    $\Delta(p)$, domain, basis, residue form, and exact ``prime_ids``. The value
    has no engine, cache, placement, or persistence policy.
    """

    data: torch.Tensor
    ring_dimension: int
    compression_layout: CompressedPlaintextLayout
    level: int
    scale: float
    context_id: str
    polynomial_domain: PolynomialDomain
    modulus_basis: ModulusBasis
    residue_representation: ResidueRepresentation
    prime_ids: tuple[int, ...]
    implicit_data: torch.Tensor | None = None
    compression_format_version: int = COMPRESSED_PLAINTEXT_FORMAT_VERSION

    def __post_init__(self) -> None:
        self.scale = coerce_scale(
            self.scale,
            value_name="CompressedPlaintext",
        )
        self.level = validate_nonnegative_level(
            self.level, value_name="CompressedPlaintext"
        )
        self.context_id = validate_context_id(
            self.context_id, value_name="CompressedPlaintext"
        )
        self.data = validate_integral_tensor(
            self.data, value_name="CompressedPlaintext"
        )
        self.prime_ids = validate_prime_ids(
            self.prime_ids, value_name="CompressedPlaintext"
        )
        if self.compression_layout not in _SUPPORTED_COMPRESSION_LAYOUTS:
            raise ValueError(
                "Unsupported CompressedPlaintext compression_layout: "
                f"{self.compression_layout!r}"
            )
        if (
            type(self.compression_format_version) is not int
            or self.compression_format_version
            != COMPRESSED_PLAINTEXT_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported CompressedPlaintext compression format version: "
                f"{self.compression_format_version}; expected "
                f"{COMPRESSED_PLAINTEXT_FORMAT_VERSION}"
            )
        if self.polynomial_domain not in ("coefficient", "ntt"):
            raise ValueError(
                "CompressedPlaintext polynomial_domain must be 'coefficient' "
                "or 'ntt': "
                f"{self.polynomial_domain!r}"
            )
        if self.modulus_basis not in ("Q", "QP"):
            raise ValueError(
                "CompressedPlaintext modulus_basis must be 'Q' or 'QP': "
                f"{self.modulus_basis!r}"
            )
        if self.residue_representation != "montgomery":
            raise ValueError(
                "Operation-ready CompressedPlaintext data must use "
                "Montgomery form"
            )
        if self.data.ndim < 2:
            raise ValueError(
                "CompressedPlaintext data must have layout "
                "[*batch, limb, unique], got shape "
                f"{tuple(self.data.shape)}"
            )
        if self.data.size(-2) == 0 or self.data.size(-1) == 0:
            raise ValueError(
                "CompressedPlaintext limb and encoded axes cannot be empty"
            )
        if self.data.size(-2) != len(self.prime_ids):
            raise ValueError(
                "CompressedPlaintext limb count does not match prime_ids: "
                f"limbs={self.data.size(-2)}, prime_ids={self.prime_ids}"
            )
        if any(extent == 0 for extent in self.data.shape[:-2]):
            raise ValueError(
                "CompressedPlaintext batch dimensions must be nonzero"
            )
        if type(self.ring_dimension) is not int:
            raise TypeError(
                "CompressedPlaintext ring_dimension must be an integer"
            )
        if self.ring_dimension <= 0 or (
            self.ring_dimension & (self.ring_dimension - 1)
        ):
            raise ValueError(
                "CompressedPlaintext ring_dimension must be a positive power "
                f"of two: {self.ring_dimension}"
            )
        unique_count = self.unique_count
        if unique_count <= 0 or unique_count & (unique_count - 1):
            raise ValueError(
                "CompressedPlaintext unique count must be a positive power of "
                f"two: {unique_count}"
            )
        if unique_count >= self.ring_dimension:
            raise ValueError(
                "CompressedPlaintext must reduce the encoded last axis: "
                f"unique={unique_count}, ring_dimension={self.ring_dimension}"
            )
        if self.ring_dimension % unique_count:
            raise ValueError(
                "CompressedPlaintext unique count must divide ring_dimension: "
                f"{unique_count} does not divide {self.ring_dimension}"
            )
        if self.compression_layout == "strided_sparse":
            if self.polynomial_domain != "coefficient":
                raise ValueError(
                    "strided_sparse CompressedPlaintext requires "
                    "polynomial_domain='coefficient'"
                )
            if self.implicit_data is None:
                raise ValueError(
                    "strided_sparse CompressedPlaintext requires implicit_data"
                )
            validate_integral_tensor(
                self.implicit_data,
                value_name="CompressedPlaintext implicit_data",
            )
            if self.implicit_data.shape != self.data.shape[:-1]:
                raise ValueError(
                    "CompressedPlaintext implicit_data must have layout "
                    "[*batch, limb]: "
                    f"expected={tuple(self.data.shape[:-1])}, "
                    f"actual={tuple(self.implicit_data.shape)}"
                )
            if self.implicit_data.dtype != self.data.dtype:
                raise ValueError(
                    "CompressedPlaintext data and implicit_data dtypes differ"
                )
            if self.implicit_data.device != self.data.device:
                raise ValueError(
                    "CompressedPlaintext data and implicit_data devices differ"
                )
        elif self.implicit_data is not None:
            raise ValueError(
                "Only strided_sparse CompressedPlaintext may carry "
                "implicit_data"
            )

    @property
    def unique_count(self) -> int:
        """Number of physically stored values per RNS row."""

        return self.data.size(-1)

    @property
    def repeat_count(self) -> int:
        """Number of dense positions represented by each stored extent."""

        return self.ring_dimension // self.unique_count

    @property
    def batch_shape(self) -> torch.Size:
        """Logical homogeneous batch dimensions."""

        return self.data.shape[:-2]

    @property
    def batch_size(self) -> int:
        """Flattened logical batch size; one for an unbatched value."""

        return self.batch_shape.numel()

    @property
    def is_batched(self) -> bool:
        """Whether this value has at least one logical batch dimension."""

        return bool(self.batch_shape)

    @classmethod
    def from_plaintext(
        cls,
        plaintext: Plaintext,
        *,
        unique_count: int,
        compression_layout: CompressedPlaintextLayout,
    ) -> CompressedPlaintext:
        """Losslessly compress one operation-ready dense RNS plaintext.

        The encoded last axis is checked bit-for-bit. The compact tensor is
        cloned so it does not retain the dense input's backing storage. Level,
        actual scale, domain, basis, residue form, dtype, device, and exact
        ``prime_ids`` are preserved.
        """

        if not plaintext.is_rns or plaintext.data is None:
            raise ValueError(
                "CompressedPlaintext.from_plaintext requires an "
                "operation-ready representation='rns' Plaintext"
            )
        if compression_layout not in _SUPPORTED_COMPRESSION_LAYOUTS:
            raise ValueError(
                f"Unsupported compression_layout: {compression_layout!r}"
            )
        ring_dimension = plaintext.data.size(-1)
        if unique_count <= 0 or unique_count >= ring_dimension:
            raise ValueError(
                "unique_count must be positive and smaller than the encoded "
                f"last axis: {unique_count} vs {ring_dimension}"
            )
        if unique_count & (unique_count - 1):
            raise ValueError(
                f"unique_count must be a power of two: {unique_count}"
            )
        if ring_dimension % unique_count:
            raise ValueError(
                "unique_count must divide the encoded last axis: "
                f"{unique_count} does not divide {ring_dimension}"
            )

        repeat_count = ring_dimension // unique_count
        data = plaintext.data
        implicit_data = None
        if compression_layout == "cyclic":
            groups = data.reshape(*data.shape[:-1], repeat_count, unique_count)
            compact = groups[..., 0, :]
            expected = compact.unsqueeze(-2).expand_as(groups)
            representable = torch.equal(groups, expected)
        elif compression_layout == "contiguous":
            groups = data.reshape(*data.shape[:-1], unique_count, repeat_count)
            compact = groups[..., :, 0]
            expected = compact.unsqueeze(-1).expand_as(groups)
            representable = torch.equal(groups, expected)
        else:
            groups = data.reshape(*data.shape[:-1], unique_count, repeat_count)
            compact = groups[..., :, 0]
            implicit_data = groups[..., 0, 1]
            expected = (
                implicit_data.unsqueeze(-1)
                .unsqueeze(-1)
                .expand_as(groups[..., :, 1:])
            )
            representable = torch.equal(groups[..., :, 1:], expected)
        if not representable:
            raise ValueError(
                "Plaintext encoded data is not exactly representable by the "
                f"requested {compression_layout!r} compression layout with "
                f"unique_count={unique_count}"
            )
        if plaintext.context_id is None:
            raise ValueError("Encoded Plaintext requires a context_id")
        if (
            plaintext.polynomial_domain is None
            or plaintext.modulus_basis is None
            or plaintext.residue_representation is None
        ):
            raise ValueError("Encoded RNS Plaintext has incomplete state")
        return cls(
            data=compact.clone(),
            ring_dimension=ring_dimension,
            compression_layout=compression_layout,
            level=plaintext.level,
            scale=plaintext.scale,
            context_id=plaintext.context_id,
            polynomial_domain=plaintext.polynomial_domain,
            modulus_basis=plaintext.modulus_basis,
            residue_representation=plaintext.residue_representation,
            prime_ids=plaintext.prime_ids,
            implicit_data=(
                None if implicit_data is None else implicit_data.clone()
            ),
            compression_format_version=COMPRESSED_PLAINTEXT_FORMAT_VERSION,
        )

    def decompress_data(self) -> torch.Tensor:
        """Materialize the exact dense RNS encoded tensor.

        The output layout is
        ``[*batch, limb, coefficient_or_ntt_index]`` with last extent
        ``ring_dimension``. It preserves dtype, device, domain, basis, residue
        form, and limb-to-``prime_ids`` mapping and does not alias compact
        storage.
        """

        repeats = [1] * self.data.ndim
        repeats[-1] = self.repeat_count
        if self.compression_layout == "cyclic":
            return self.data.repeat(*repeats)
        if self.compression_layout == "contiguous":
            return self.data.repeat_interleave(self.repeat_count, dim=-1)
        if self.implicit_data is None:
            raise RuntimeError(
                "strided_sparse CompressedPlaintext has no implicit_data"
            )
        dense = (
            self.implicit_data.unsqueeze(-1)
            .expand(*self.data.shape[:-1], self.ring_dimension)
            .clone()
        )
        dense[..., :: self.repeat_count] = self.data
        return dense

    def to_plaintext(self) -> Plaintext:
        """Materialize the exact equivalent dense RNS :class:`Plaintext`.

        "Standard" here means the ordinary dense value type; the returned
        residue representation remains exactly ``self.residue_representation``.
        """

        return Plaintext(
            message=None,
            level=self.level,
            scale=self.scale,
            data=self.decompress_data(),
            context_id=self.context_id,
            representation="rns",
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
            prime_ids=self.prime_ids,
        )

    def clone(self) -> CompressedPlaintext:
        return self._with_resident_tensors(
            tuple(tensor.clone() for tensor in self._resident_tensors)
        )

    def with_data(self, data: torch.Tensor) -> CompressedPlaintext:
        """Return the same exact metadata around replacement tensor storage."""

        tensors = (
            (data,)
            if self.implicit_data is None
            else (data, self.implicit_data)
        )
        return self._with_resident_tensors(tensors)

    def with_storage(
        self,
        data: torch.Tensor,
        implicit_data: torch.Tensor | None,
    ) -> CompressedPlaintext:
        """Return the same metadata around complete replacement storage."""

        tensors = (data,) if implicit_data is None else (data, implicit_data)
        return self._with_resident_tensors(tensors)

    @classmethod
    def stack_batch(
        cls,
        values: tuple[CompressedPlaintext, ...] | list[CompressedPlaintext],
    ) -> CompressedPlaintext:
        """Allocate and copy compatible values into one new batch axis."""

        if not values:
            raise ValueError(
                "CompressedPlaintext.stack_batch requires at least one value"
            )
        first = values[0]
        metadata = (
            "ring_dimension",
            "compression_layout",
            "level",
            "scale",
            "context_id",
            "polynomial_domain",
            "modulus_basis",
            "residue_representation",
            "prime_ids",
            "compression_format_version",
        )
        tensors = [first.data]
        implicit_tensors = (
            [] if first.implicit_data is None else [first.implicit_data]
        )
        for index, value in enumerate(values[1:], start=1):
            mismatches = [
                name
                for name in metadata
                if getattr(value, name) != getattr(first, name)
            ]
            if value.data.shape != first.data.shape:
                mismatches.append("data.shape")
            if value.data.dtype != first.data.dtype:
                mismatches.append("data.dtype")
            if value.data.device != first.data.device:
                mismatches.append("data.device")
            if (value.implicit_data is None) != (first.implicit_data is None):
                mismatches.append("implicit_data")
            elif (
                value.implicit_data is not None
                and first.implicit_data is not None
            ):
                if value.implicit_data.shape != first.implicit_data.shape:
                    mismatches.append("implicit_data.shape")
                if value.implicit_data.dtype != first.implicit_data.dtype:
                    mismatches.append("implicit_data.dtype")
                if value.implicit_data.device != first.implicit_data.device:
                    mismatches.append("implicit_data.device")
            if mismatches:
                raise ValueError(
                    "CompressedPlaintext.stack_batch received incompatible "
                    f"value at index {index}: mismatches={mismatches}"
                )
            tensors.append(value.data)
            if value.implicit_data is not None:
                implicit_tensors.append(value.implicit_data)
        stacked = [torch.stack(tensors, dim=0)]
        if implicit_tensors:
            stacked.append(torch.stack(implicit_tensors, dim=0))
        return first._with_resident_tensors(tuple(stacked))

    def select_batch(self, index: int, *, dim: int = 0) -> CompressedPlaintext:
        """Return a storage-sharing view selected from one batch axis."""

        if not self.is_batched:
            raise ValueError(
                "Cannot select a batch item from an unbatched value"
            )
        logical_dim = dim if dim >= 0 else dim + len(self.batch_shape)
        if not 0 <= logical_dim < len(self.batch_shape):
            raise IndexError(
                f"Batch dimension {dim} is outside shape "
                f"{tuple(self.batch_shape)}"
            )
        tensors = [self.data.select(logical_dim, index)]
        if self.implicit_data is not None:
            tensors.append(self.implicit_data.select(logical_dim, index))
        return self._with_resident_tensors(tuple(tensors))

    def unbind_batch(self, *, dim: int = 0) -> tuple[CompressedPlaintext, ...]:
        """Return storage-sharing views along one logical batch axis."""

        if not self.is_batched:
            raise ValueError("Cannot unbind an unbatched CompressedPlaintext")
        logical_dim = dim if dim >= 0 else dim + len(self.batch_shape)
        if not 0 <= logical_dim < len(self.batch_shape):
            raise IndexError(
                f"Batch dimension {dim} is outside shape "
                f"{tuple(self.batch_shape)}"
            )
        data_parts = self.data.unbind(logical_dim)
        implicit_parts = (
            (None,) * len(data_parts)
            if self.implicit_data is None
            else self.implicit_data.unbind(logical_dim)
        )
        return tuple(
            self._with_resident_tensors(
                (data,) if implicit is None else (data, implicit)
            )
            for data, implicit in zip(data_parts, implicit_parts, strict=True)
        )

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        if self.implicit_data is None:
            return (self.data,)
        return (self.data, self.implicit_data)

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> CompressedPlaintext:
        expected_count = 1 if self.implicit_data is None else 2
        if len(tensors) != expected_count:
            raise ValueError(
                "CompressedPlaintext resident tensor count differs from its "
                f"compression layout: expected={expected_count}, "
                f"actual={len(tensors)}"
            )
        return CompressedPlaintext(
            data=tensors[0],
            ring_dimension=self.ring_dimension,
            compression_layout=self.compression_layout,
            level=self.level,
            scale=self.scale,
            context_id=self.context_id,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
            prime_ids=self.prime_ids,
            implicit_data=None if len(tensors) == 1 else tensors[1],
            compression_format_version=self.compression_format_version,
        )

    def __str__(self) -> str:
        return (
            "CompressedPlaintext("
            f"level={self.level}, scale={self.scale}, "
            f"polynomial_domain={self.polynomial_domain!r}, modulus_basis={self.modulus_basis!r}, "
            f"residue_representation={self.residue_representation}, prime_ids={self.prime_ids}, "
            f"ring_dimension={self.ring_dimension}, "
            f"compression_layout={self.compression_layout!r}, "
            f"compression_format_version={self.compression_format_version}, "
            f"repeat_count={self.repeat_count}, "
            f"batch_shape={tuple(self.batch_shape)}, "
            f"data_shape={tuple(self.data.shape)})"
        )

    __repr__ = __str__
