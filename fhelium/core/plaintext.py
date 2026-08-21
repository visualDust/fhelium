"""Level-specific CKKS plaintext values."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhelium.core.scale import coerce_scale
from fhelium.core.state import (
    ModulusBasis,
    PlaintextRepresentation,
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
    construction; validation requires dense strided storage and the dtype
    constraints above. Engine operations additionally require the engine's
    configured integral dtype, device, ring dimension, and complete ordered
    ``prime_ids``. Construction does not clone an input tensor. :meth:`clone`
    allocates independent storage, while batch selection and unbinding return
    storage-sharing views.

    A program that needs the same semantic message in multiple arithmetic states
    constructs separate values. The object owns no engine, cache, placement, or
    persistence reference. ``scale`` is the positive finite actual scale
    $\Delta(v)$; ``level`` identifies $Q_\ell$ but never substitutes for
    ``prime_ids``.
    """

    message: torch.Tensor | None
    level: int
    scale: float
    data: torch.Tensor | None = None
    context_id: str | None = None
    representation: PlaintextRepresentation = "slots"
    polynomial_domain: PolynomialDomain | None = None
    modulus_basis: ModulusBasis | None = None
    residue_representation: ResidueRepresentation | None = None
    prime_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.scale = coerce_scale(self.scale, value_name="Plaintext")
        self.level = validate_nonnegative_level(
            self.level, value_name="Plaintext"
        )
        if (self.message is None) == (self.data is None):
            raise ValueError(
                "Plaintext must own exactly one representation: slots message "
                "or encoded data"
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

        if self.data is None:
            assert self.message is not None
            if self.context_id is not None:
                self.context_id = validate_context_id(
                    self.context_id, value_name="slots Plaintext"
                )
            if self.representation != "slots":
                raise ValueError(
                    "A Plaintext without encoded data must use representation='slots'"
                )
            if (
                self.polynomial_domain is not None
                or self.modulus_basis is not None
                or self.residue_representation is not None
            ):
                raise ValueError(
                    "A slots-only Plaintext cannot declare polynomial_domain, "
                    "modulus_basis, or residue_representation"
                )
            self.prime_ids = validate_prime_ids(
                self.prime_ids,
                value_name="slots Plaintext",
                allow_empty=True,
            )
            if self.prime_ids:
                raise ValueError(
                    "A slots-only Plaintext cannot declare an RNS layout"
                )
            if self.message.ndim > 0 and self.message.size(-1) == 0:
                raise ValueError("Plaintext slot axis cannot be empty")
            if self.message.ndim > 0 and any(
                extent == 0 for extent in self.message.shape[:-1]
            ):
                raise ValueError("Plaintext batch dimensions must be nonzero")
            return

        if self.representation == "slots":
            raise ValueError(
                "A Plaintext with encoded data cannot use representation='slots'"
            )
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("Encoded Plaintext data must be a torch.Tensor")
        if self.data.layout != torch.strided:
            raise TypeError(
                "Encoded Plaintext data must use dense strided storage"
            )
        self.context_id = validate_context_id(
            self.context_id, value_name="encoded Plaintext"
        )
        if self.representation in (
            "integer_coefficients",
            "approximate_coefficients",
        ):
            if self.data.ndim < 1:
                raise ValueError(
                    "Coefficient Plaintext data must have layout "
                    "[*batch, coeff], "
                    f"got shape {tuple(self.data.shape)}"
                )
            if self.data.size(-1) == 0:
                raise ValueError("Plaintext coefficient axis cannot be empty")
            if self.polynomial_domain != "coefficient":
                raise ValueError(
                    "Coefficient Plaintext data requires "
                    "polynomial_domain='coefficient', got "
                    f"{self.polynomial_domain!r}"
                )
            if (
                self.modulus_basis is not None
                or self.residue_representation is not None
                or self.prime_ids
            ):
                raise ValueError(
                    "Coefficient Plaintext data cannot declare an RNS "
                    "modulus_basis, residue_representation, or prime_ids"
                )
            self.prime_ids = validate_prime_ids(
                self.prime_ids,
                value_name="coefficient Plaintext",
                allow_empty=True,
            )
            if self.representation == "integer_coefficients":
                validate_integral_tensor(
                    self.data,
                    value_name="integer_coefficients Plaintext",
                )
            else:
                if self.data.layout != torch.strided:
                    raise TypeError(
                        "approximate_coefficients Plaintext data must use "
                        "dense strided storage"
                    )
                if self.data.dtype != torch.float64:
                    raise TypeError(
                        "approximate_coefficients Plaintext data must use float64"
                    )
                if not bool(torch.all(torch.isfinite(self.data)).item()):
                    raise ValueError(
                        "approximate_coefficients Plaintext data must be finite"
                    )
            if any(extent == 0 for extent in self.data.shape[:-1]):
                raise ValueError("Plaintext batch dimensions must be nonzero")
            return

        if self.data.ndim < 2:
            raise ValueError(
                "RNS Plaintext data must have layout [*batch, limb, coeff], "
                f"got shape {tuple(self.data.shape)}"
            )
        if self.data.size(-2) == 0 or self.data.size(-1) == 0:
            raise ValueError(
                "RNS Plaintext limb and coefficient axes cannot be empty"
            )
        if (
            self.data.dtype == torch.bool
            or self.data.is_floating_point()
            or self.data.is_complex()
        ):
            raise TypeError(
                "RNS Plaintext data must use an integral scalar dtype"
            )
        if self.polynomial_domain not in ("coefficient", "ntt"):
            raise ValueError(
                "RNS Plaintext polynomial_domain must be 'coefficient' or "
                f"'ntt': {self.polynomial_domain!r}"
            )
        if self.modulus_basis not in ("Q", "QP"):
            raise ValueError(
                "RNS Plaintext modulus_basis must be 'Q' or 'QP': "
                f"{self.modulus_basis!r}"
            )
        if self.residue_representation not in ("standard", "montgomery"):
            raise ValueError(
                "RNS Plaintext residue_representation must be 'standard' or "
                f"'montgomery': {self.residue_representation!r}"
            )
        if (
            self.polynomial_domain == "ntt"
            and self.residue_representation != "montgomery"
        ):
            raise ValueError(
                "NTT-domain RNS Plaintext must use Montgomery residues"
            )
        self.prime_ids = validate_prime_ids(
            self.prime_ids,
            value_name="RNS Plaintext",
        )
        if self.data.size(-2) != len(self.prime_ids):
            raise ValueError(
                "RNS Plaintext limb count does not match prime_ids: "
                f"limbs={self.data.size(-2)}, prime_ids={self.prime_ids}"
            )
        if any(extent == 0 for extent in self.data.shape[:-2]):
            raise ValueError("Plaintext batch dimensions must be nonzero")

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

        return Plaintext(
            message=None if self.message is None else self.message.clone(),
            level=self.level,
            scale=self.scale,
            data=None if self.data is None else self.data.clone(),
            context_id=self.context_id,
            representation=self.representation,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
            prime_ids=self.prime_ids,
        )

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
            "level",
            "scale",
            "context_id",
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
        iterator = iter(tensors)
        message = next(iterator) if self.message is not None else None
        data = next(iterator) if self.data is not None else None
        return Plaintext(
            message=message,
            level=self.level,
            scale=self.scale,
            data=data,
            context_id=self.context_id,
            representation=self.representation,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
            prime_ids=self.prime_ids,
        )

    def __str__(self) -> str:
        message_shape = (
            None if self.message is None else tuple(self.message.shape)
        )
        data_shape = None if self.data is None else tuple(self.data.shape)
        return (
            "Plaintext("
            f"level={self.level}, scale={self.scale}, "
            f"representation={self.representation!r}, "
            f"polynomial_domain={self.polynomial_domain!r}, modulus_basis={self.modulus_basis!r}, "
            f"residue_representation={self.residue_representation}, prime_ids={self.prime_ids}, "
            f"batch_shape={tuple(self.batch_shape)}, "
            f"message_shape={message_shape}, data_shape={data_shape})"
        )

    __repr__ = __str__
