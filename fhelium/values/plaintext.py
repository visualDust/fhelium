"""Depth-specific CKKS plaintext values."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Self

import torch

from fhelium.values._scale import coerce_scale
from fhelium.values.state import (
    ModulusBasis,
    PlaintextRepresentation,
    PolynomialDomain,
    ResidueRepresentation,
)
from fhelium.values.tensor_resident import TensorResident
from fhelium.values._validation import (
    validate_integral_tensor,
    validate_nonnegative_depth,
    validate_prime_ids,
)


@dataclass(eq=False)
class Plaintext(TensorResident):
    r"""One homogeneous CKKS plaintext or dense batch at one state tuple.

    The state fields describe tensor layout and arithmetic form independently:

    - ``representation="slots"`` stores a scalar (repeated to all slots during
      encoding) or ``[*batch, slot]`` real/complex semantic messages. Encoding
      has not occurred; ``data`` and all RNS-state metadata are absent.
    - ``representation="integer_coefficients"`` stores an integral
      ``[*batch, coefficient]`` tensor for $p(X)\in R$, before RNS reduction.
    - ``representation="approximate_coefficients"`` stores the bounded
      binary64 ``[*batch, coefficient]`` tail-Q reconstruction produced by
      decryption. It is valid only for decoding and cannot be encrypted or
      reduced back to RNS; it is not a full-$Q_\ell$ CRT inverse.
    - ``representation="rns"`` stores a dense integral
      ``[*batch, limb, coefficient_or_ntt_index]`` tensor. Limb row $i$ is
      modulo the parameter prime ``prime_ids[i]``. The last axis indexes
      coefficients of $R=\mathbb{Z}[X]/(X^N+1)$ in ``"coefficient"`` domain
      or NTT evaluations in ``"ntt"`` domain. ``modulus_basis`` selects
      $Q_\ell$ or $Q_\ell P$, and ``residue_representation`` distinguishes
      standard from Montgomery residues.

    Tensor payloads retain their input dtype, device, and storage at direct
    construction; encoded payloads require dense strided storage and the dtype
    constraints above. Engine operations additionally require the engine's
    configured integral dtype, device, ring dimension, and matching prime-row
    parameters. Row-local operations can use a sliced RNS interval; operations
    requiring the complete active basis must receive every required row.
    Construction does not clone an input tensor. :meth:`clone`
    allocates independent storage, while batch selection and unbinding return
    storage-sharing views.

    A program that needs the same semantic message in multiple arithmetic states
    constructs separate values. The object owns no engine, cache, placement, or
    persistence reference. ``scale`` is the positive finite actual scale
    $\Delta(v)$; ``depth`` identifies $Q_\ell$ but never substitutes for
    ``prime_ids``.
    """

    message: torch.Tensor | None
    depth: int
    scale: float
    data: torch.Tensor | None = None
    representation: PlaintextRepresentation = "slots"
    polynomial_domain: PolynomialDomain | None = None
    modulus_basis: ModulusBasis | None = None
    residue_representation: ResidueRepresentation | None = None
    prime_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.scale = coerce_scale(self.scale, value_name="Plaintext")
        self.depth = validate_nonnegative_depth(
            self.depth, value_name="Plaintext"
        )
        if self.representation not in (
            "slots",
            "integer_coefficients",
            "approximate_coefficients",
            "rns",
        ):
            raise ValueError(
                f"Unsupported Plaintext representation: {self.representation!r}"
            )

        if self.representation == "slots":
            if self.message is None or self.data is not None:
                raise ValueError(
                    "Slots Plaintext requires message and no encoded data"
                )
            self.prime_ids = tuple(self.prime_ids)
            if (
                self.polynomial_domain is not None
                or self.modulus_basis is not None
                or self.residue_representation is not None
                or self.prime_ids
            ):
                raise ValueError(
                    "Slots Plaintext cannot declare polynomial or RNS state"
                )
            if self.message.numel() == 0:
                raise ValueError("Plaintext slots cannot be empty")
            return

        if self.data is None or self.message is not None:
            raise ValueError(
                "Encoded Plaintext requires data and no slots message"
            )
        if self.representation == "approximate_coefficients":
            if not isinstance(self.data, torch.Tensor):
                raise TypeError("Encoded Plaintext data must be a torch.Tensor")
            if (
                self.data.layout != torch.strided
                or self.data.dtype != torch.float64
            ):
                raise TypeError(
                    "approximate_coefficients Plaintext requires dense strided float64 data"
                )
        else:
            validate_integral_tensor(
                self.data, value_name=self.representation + " Plaintext"
            )
        rns = self.representation == "rns"
        if self.data.ndim < (2 if rns else 1):
            layout = "[*batch, limb, coeff]" if rns else "[*batch, coeff]"
            raise ValueError(
                f"Plaintext data must have layout {layout}, got shape {tuple(self.data.shape)}"
            )
        if self.data.numel() == 0:
            raise ValueError("Plaintext data cannot be empty")

        if not rns:
            self.prime_ids = tuple(self.prime_ids)
            if self.polynomial_domain != "coefficient":
                raise ValueError(
                    "Coefficient Plaintext data requires polynomial_domain='coefficient'"
                )
            if (
                self.modulus_basis is not None
                or self.residue_representation is not None
                or self.prime_ids
            ):
                raise ValueError(
                    "Coefficient Plaintext data cannot declare an RNS modulus_basis, residue_representation, or prime_ids"
                )
            return

        if self.polynomial_domain not in ("coefficient", "ntt"):
            raise ValueError(
                f"RNS Plaintext polynomial_domain must be 'coefficient' or 'ntt': {self.polynomial_domain!r}"
            )
        if self.modulus_basis not in ("Q", "QP"):
            raise ValueError(
                f"RNS Plaintext modulus_basis must be 'Q' or 'QP': {self.modulus_basis!r}"
            )
        if self.residue_representation not in ("standard", "montgomery"):
            raise ValueError(
                "RNS Plaintext residue_representation must be 'standard' or 'montgomery'"
            )
        if (
            self.polynomial_domain == "ntt"
            and self.residue_representation != "montgomery"
        ):
            raise ValueError(
                "NTT-domain RNS Plaintext must use Montgomery residues"
            )
        self.prime_ids = validate_prime_ids(
            self.prime_ids, value_name="RNS Plaintext"
        )
        if self.data.size(-2) != len(self.prime_ids):
            raise ValueError(
                "RNS Plaintext limb count does not match prime_ids: "
                f"limbs={self.data.size(-2)}, prime_ids={self.prime_ids}"
            )

    @property
    def is_slots(self) -> bool:
        return self.representation == "slots"

    @property
    def is_integer_coefficients(self) -> bool:
        return self.representation == "integer_coefficients"

    @property
    def is_approximate_coefficients(self) -> bool:
        return self.representation == "approximate_coefficients"

    @property
    def is_rns(self) -> bool:
        return self.representation == "rns"

    @property
    def limb_count(self) -> int:
        """Number of represented RNS rows; zero for non-RNS representations."""

        return len(self.prime_ids)

    @property
    def batch_shape(self) -> torch.Size:
        """Logical homogeneous batch dimensions for the active form."""

        tensor = self.message if self.message is not None else self.data
        if tensor is None or tensor.ndim == 0:
            return torch.Size()
        if self.representation in (
            "slots",
            "integer_coefficients",
            "approximate_coefficients",
        ):
            return tensor.shape[:-1]
        return tensor.shape[:-2]

    @property
    def batch_size(self) -> int:
        """Flattened logical batch size; one for an unbatched value."""

        return self.batch_shape.numel()

    @property
    def is_batched(self) -> bool:
        """Whether this value has at least one logical batch dimension."""

        return bool(self.batch_shape)

    def clone(self) -> Plaintext:
        """Return a metadata-equivalent value with independent tensor storage."""

        return self._with_resident_tensors(
            tuple(tensor.clone() for tensor in self._resident_tensors)
        )

    @classmethod
    def _from_fields(
        cls,
        *,
        message: torch.Tensor | None,
        depth: int,
        scale: float,
        data: torch.Tensor | None,
        representation: PlaintextRepresentation,
        polynomial_domain: PolynomialDomain | None,
        modulus_basis: ModulusBasis | None,
        residue_representation: ResidueRepresentation | None,
        prime_ids: tuple[int, ...],
    ) -> Self:
        """Assemble fields that an internal caller has already prepared."""

        result: Self = object.__new__(cls)
        result.message = message
        result.depth = depth
        result.scale = scale
        result.data = data
        result.representation = representation
        result.polynomial_domain = polynomial_domain
        result.modulus_basis = modulus_basis
        result.residue_representation = residue_representation
        result.prime_ids = prime_ids
        return result

    def slice_limbs(self, start: int, stop: int) -> Plaintext:
        """Return a storage-sharing RNS row interval and its prime IDs.

        ``[start, stop)`` indexes stored limb positions, not global prime IDs.
        All batch axes, depth, scale, and representation state are preserved;
        the result represents part of the same basis, not a rescaled value.
        Slots and non-RNS coefficient representations have no limb axis.
        """

        if not self.is_rns or self.data is None:
            raise ValueError(
                "Plaintext limb slicing requires RNS representation"
            )
        if not 0 <= start < stop <= self.limb_count:
            raise ValueError(
                "Plaintext limb slice must satisfy "
                f"0 <= start < stop <= {self.limb_count}; "
                f"got start={start}, stop={stop}"
            )
        result = self._with_resident_tensors((self.data[..., start:stop, :],))
        result.prime_ids = self.prime_ids[start:stop]
        return result

    @classmethod
    def stack_batch(
        cls, values: tuple[Plaintext, ...] | list[Plaintext]
    ) -> Plaintext:
        """Allocate and copy compatible plaintexts into one new batch axis.

        Scalar slots plaintexts are rejected because stacking them would
        change their repeat-to-all-slots meaning; materialize slot vectors
        first. Inputs must have identical representation, state,
        ``prime_ids``, shape, dtype, and device. The result does not alias an
        input.
        """

        if not values:
            raise ValueError(
                "Plaintext.stack_batch requires at least one value"
            )
        first = values[0]
        if (
            first.representation == "slots"
            and first._resident_tensors[0].ndim == 0
        ):
            raise ValueError(
                "Scalar slot Plaintexts cannot be batch-stacked without "
                "changing their repeat-to-all-slots semantics; materialize "
                "a slot vector for each value first"
            )
        metadata = (
            "depth",
            "scale",
            "representation",
            "polynomial_domain",
            "modulus_basis",
            "residue_representation",
            "prime_ids",
        )
        first_tensor = first._resident_tensors[0]
        tensors = [first_tensor]
        for index, value in enumerate(values[1:], start=1):
            mismatches = [
                name
                for name in metadata
                if getattr(value, name) != getattr(first, name)
            ]
            tensor = value._resident_tensors[0]
            if tensor.shape != first_tensor.shape:
                mismatches.append("tensor.shape")
            if tensor.dtype != first_tensor.dtype:
                mismatches.append("tensor.dtype")
            if tensor.device != first_tensor.device:
                mismatches.append("tensor.device")
            if mismatches:
                raise ValueError(
                    "Plaintext.stack_batch received incompatible value at "
                    f"index {index}: mismatches={mismatches}"
                )
            tensors.append(tensor)
        stacked = torch.stack(tensors, dim=0)
        return first._with_resident_tensors((stacked,))

    def slice_batch(self, start: int, stop: int, *, dim: int = 0) -> Plaintext:
        """Return a storage-sharing interval along one logical batch axis.

        ``[start, stop)`` must be a nonempty interval within the selected axis.
        ``dim`` indexes ``batch_shape`` and accepts negative dimensions. The
        axis is retained even for a one-item interval. The active slots, coefficient, or RNS representation is preserved.
        Depth, scale, prime IDs, and representation state are unchanged.
        """

        if not self.is_batched:
            raise ValueError("Cannot slice a batch axis of an unbatched value")
        logical_dim = dim if dim >= 0 else dim + len(self.batch_shape)
        if not 0 <= logical_dim < len(self.batch_shape):
            raise IndexError(
                f"Batch dimension {dim} is outside shape {tuple(self.batch_shape)}"
            )
        if not 0 <= start < stop <= self.batch_shape[logical_dim]:
            raise ValueError(
                "Plaintext batch slice must satisfy "
                f"0 <= start < stop <= {self.batch_shape[logical_dim]}; "
                f"got start={start}, stop={stop}"
            )
        return self._with_resident_tensors(
            tuple(
                tensor.narrow(logical_dim, start, stop - start)
                for tensor in self._resident_tensors
            )
        )

    def select_batch(self, index: int, *, dim: int = 0) -> Plaintext:
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
        tensor = self._resident_tensors[0].select(logical_dim, index)
        return self._with_resident_tensors((tensor,))

    def unbind_batch(self, *, dim: int = 0) -> tuple[Plaintext, ...]:
        """Return storage-sharing views along one logical batch axis."""

        if not self.is_batched:
            raise ValueError("Cannot unbind an unbatched Plaintext")
        logical_dim = dim if dim >= 0 else dim + len(self.batch_shape)
        if not 0 <= logical_dim < len(self.batch_shape):
            raise IndexError(
                f"Batch dimension {dim} is outside shape "
                f"{tuple(self.batch_shape)}"
            )
        return tuple(
            self._with_resident_tensors((tensor,))
            for tensor in self._resident_tensors[0].unbind(logical_dim)
        )

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor for tensor in (self.message, self.data) if tensor is not None
        )

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> Plaintext:
        result = copy(self)
        if self.message is not None:
            result.message = tensors[0]
        else:
            result.data = tensors[0]
        return result

    def __str__(self) -> str:
        message_shape = (
            None if self.message is None else tuple(self.message.shape)
        )
        data_shape = None if self.data is None else tuple(self.data.shape)
        return (
            "Plaintext("
            f"depth={self.depth}, scale={self.scale}, "
            f"representation={self.representation!r}, "
            f"polynomial_domain={self.polynomial_domain!r}, modulus_basis={self.modulus_basis!r}, "
            f"residue_representation={self.residue_representation}, prime_ids={self.prime_ids}, "
            f"batch_shape={tuple(self.batch_shape)}, "
            f"message_shape={message_shape}, data_shape={data_shape})"
        )

    __repr__ = __str__
