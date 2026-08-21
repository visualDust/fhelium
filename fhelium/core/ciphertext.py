"""Tensor-backed CKKS ciphertext value type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from fhelium.core.scale import coerce_scale
from fhelium.core.state import (
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


@dataclass
class Ciphertext(TensorResident):
    r"""One homogeneous process-local CKKS ciphertext tensor or dense batch.

    ``data`` is a dense integral tensor with layout
    ``[component, *batch, limb, coefficient_or_ntt_index]``. The component
    extent is two or three for

    $$
    u(X)=\sum_{j=0}^{d-1}c_j(X)s(X)^j.
    $$

    Limb row $i$ is modulo the parameter prime ``prime_ids[i]``. The
    final extent indexes coefficients in
    $R=\mathbb{Z}[X]/(X^N+1)$ or NTT evaluations according to
    ``polynomial_domain``. ``modulus_basis`` selects $Q_\ell$ or $Q_\ell P$;
    valid operation states pair coefficient domain with standard residues and
    NTT domain with Montgomery residues. ``scale`` is the positive finite
    actual scale $\Delta(c)$.

    Every member of ``batch_shape`` shares the same level, scale, component
    count, domain, basis, residue form, context, and ordered ``prime_ids``.
    Direct construction retains the input dtype, device, and storage; engine
    operations additionally require the engine's configured integral dtype,
    device, ring dimension, and expected row interval. :meth:`clone` owns new
    storage, while limb slices and batch selections are views. Methods ending
    in ``_`` mutate this object and are visible through aliases.

    Distribution and communication are deliberately not encoded in this
    value: an SPMD program decides what each rank stores and which collectives
    it executes.
    """

    data: torch.Tensor
    level: int
    scale: float
    context_id: str
    prime_ids: tuple[int, ...]
    polynomial_domain: PolynomialDomain = "coefficient"
    modulus_basis: ModulusBasis = "Q"
    residue_representation: ResidueRepresentation = "standard"

    def __post_init__(self) -> None:
        self.scale = coerce_scale(self.scale, value_name="Ciphertext")
        self.level = validate_nonnegative_level(
            self.level, value_name="Ciphertext"
        )
        self.context_id = validate_context_id(
            self.context_id, value_name="Ciphertext"
        )
        self.data = validate_integral_tensor(self.data, value_name="Ciphertext")
        self.prime_ids = validate_prime_ids(
            self.prime_ids, value_name="Ciphertext"
        )
        if self.data.ndim < 3:
            raise ValueError(
                "Ciphertext data must have layout "
                "[component, *batch, limb, coeff], "
                f"got shape {tuple(self.data.shape)}"
            )
        if self.data.size(0) not in (2, 3):
            raise ValueError(
                "Ciphertext component axis must have size 2 or 3, "
                f"got {self.data.size(0)}"
            )
        if self.data.size(-2) == 0 or self.data.size(-1) == 0:
            raise ValueError(
                "Ciphertext limb and coefficient axes cannot be empty"
            )
        if self.data.size(-2) != len(self.prime_ids):
            raise ValueError(
                "Ciphertext limb count does not match prime_ids: "
                f"limbs={self.data.size(-2)}, prime_ids={self.prime_ids}"
            )
        if any(extent == 0 for extent in self.data.shape[1:-2]):
            raise ValueError("Ciphertext batch dimensions must be nonzero")
        if self.polynomial_domain not in ("coefficient", "ntt"):
            raise ValueError(
                "Unsupported ciphertext polynomial_domain: "
                f"{self.polynomial_domain!r}"
            )
        if self.modulus_basis not in ("Q", "QP"):
            raise ValueError(
                f"Unsupported ciphertext modulus_basis: {self.modulus_basis!r}"
            )
        if self.residue_representation not in ("standard", "montgomery"):
            raise ValueError(
                "Unsupported ciphertext residue representation: "
                f"{self.residue_representation!r}"
            )
        if (
            self.polynomial_domain == "coefficient"
            and self.residue_representation != "standard"
        ):
            raise ValueError(
                "Coefficient-domain Ciphertext cannot be Montgomery-only"
            )
        if (
            self.polynomial_domain == "ntt"
            and self.residue_representation != "montgomery"
        ):
            raise ValueError(
                "NTT-domain Ciphertext must use Montgomery representation"
            )

    @property
    def component_count(self) -> int:
        return self.data.size(0)

    @property
    def limb_count(self) -> int:
        return self.data.size(-2)

    @property
    def ring_dimension(self) -> int:
        return self.data.size(-1)

    @property
    def batch_shape(self) -> torch.Size:
        """Logical homogeneous batch dimensions, excluding CKKS axes."""

        return self.data.shape[1:-2]

    @property
    def batch_size(self) -> int:
        """Flattened logical batch size; one for an unbatched value."""

        return self.batch_shape.numel()

    @property
    def is_batched(self) -> bool:
        """Whether this value has at least one logical batch dimension."""

        return bool(self.batch_shape)

    def component(self, index: int) -> torch.Tensor:
        """Return the storage-sharing ``[*batch, limb, index]`` component view."""

        if not 0 <= index < self.component_count:
            raise IndexError(
                f"Ciphertext component {index} is outside "
                f"[0, {self.component_count})"
            )
        return self.data[index]

    @property
    def c0(self) -> torch.Tensor:
        return self.component(0)

    @property
    def c1(self) -> torch.Tensor:
        return self.component(1)

    @property
    def c2(self) -> torch.Tensor:
        if self.component_count != 3:
            raise ValueError("A two-component Ciphertext has no c2")
        return self.component(2)

    @property
    def is_ntt_domain(self) -> bool:
        return self.polynomial_domain == "ntt"

    @property
    def is_coefficient_domain(self) -> bool:
        return self.polynomial_domain == "coefficient"

    @property
    def includes_p(self) -> bool:
        return self.modulus_basis == "QP"

    def assert_state(
        self,
        *,
        polynomial_domain: PolynomialDomain | None = None,
        residue_representation: ResidueRepresentation | None = None,
        modulus_basis: ModulusBasis | None = None,
        components: int | None = None,
    ) -> Ciphertext:
        expected: dict[str, tuple[Any, Any]] = {
            "polynomial_domain": (polynomial_domain, self.polynomial_domain),
            "residue_representation": (
                residue_representation,
                self.residue_representation,
            ),
            "modulus_basis": (modulus_basis, self.modulus_basis),
            "components": (components, self.component_count),
        }
        for name, (wanted, actual) in expected.items():
            if wanted is not None and wanted != actual:
                raise ValueError(
                    f"Invalid Ciphertext {name}: expected {wanted!r}, "
                    f"got {actual!r}"
                )
        return self

    def clone(self) -> Ciphertext:
        """Return a metadata-equivalent ciphertext with independent storage."""

        return self.with_data(self.data.clone())

    def with_data(self, data: torch.Tensor) -> Ciphertext:
        """Construct the same semantic layout around replacement storage.

        The payload is not cloned; the result aliases ``data`` exactly.
        """

        return Ciphertext(
            data=data,
            level=self.level,
            scale=self.scale,
            context_id=self.context_id,
            prime_ids=self.prime_ids,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
        )

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.data,)

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> Ciphertext:
        return self.with_data(tensors[0])

    def slice_limbs(self, start: int, stop: int) -> Ciphertext:
        """Return a storage-sharing view over ``[start:stop]`` RNS limbs.

        This is a local tensor operation, not a placement decision.  In-place
        arithmetic on the returned value also modifies the corresponding rows
        of this ciphertext.
        """

        if not 0 <= start < stop <= self.limb_count:
            raise ValueError(
                "Ciphertext limb slice must satisfy "
                f"0 <= start < stop <= {self.limb_count}; "
                f"got start={start}, stop={stop}"
            )
        return Ciphertext(
            data=self.data[..., start:stop, :],
            level=self.level,
            scale=self.scale,
            context_id=self.context_id,
            prime_ids=self.prime_ids[start:stop],
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
        )

    @classmethod
    def stack_batch(
        cls, values: tuple[Ciphertext, ...] | list[Ciphertext]
    ) -> Ciphertext:
        """Allocate and copy compatible values into one new batch dimension.

        Stacking separately allocated values cannot be zero-copy. This named
        constructor makes that cost visible rather than hiding a
        :func:`torch.stack` inside an engine operation. The new logical batch
        axis is inserted before any batch axes already owned by each value.

        ``context_id`` does not identify an encryption key. The caller must
        ensure every value has the same effective key lineage, applying an
        key switch first when necessary.
        """

        if not values:
            raise ValueError(
                "Ciphertext.stack_batch requires at least one value"
            )
        first = values[0]
        metadata = (
            "level",
            "scale",
            "context_id",
            "prime_ids",
            "polynomial_domain",
            "modulus_basis",
            "residue_representation",
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
            if mismatches:
                raise ValueError(
                    "Ciphertext.stack_batch received incompatible value at "
                    f"index {index}: mismatches={mismatches}"
                )
        return first.with_data(
            torch.stack(tuple(value.data for value in values), dim=1)
        )

    def select_batch(self, index: int, *, dim: int = 0) -> Ciphertext:
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
        return self.with_data(self.data.select(logical_dim + 1, index))

    def unbind_batch(self, *, dim: int = 0) -> tuple[Ciphertext, ...]:
        """Return storage-sharing views along one logical batch axis."""

        if not self.is_batched:
            raise ValueError("Cannot unbind an unbatched Ciphertext")
        logical_dim = dim if dim >= 0 else dim + len(self.batch_shape)
        if not 0 <= logical_dim < len(self.batch_shape):
            raise IndexError(
                f"Batch dimension {dim} is outside shape "
                f"{tuple(self.batch_shape)}"
            )
        return tuple(
            self.with_data(data) for data in self.data.unbind(logical_dim + 1)
        )

    def replace_(self, other: Ciphertext) -> Ciphertext:
        """Replace this value without changing its Python object identity.

        The result aliases ``other.data``; prior aliases of ``self.data`` keep
        the old allocation. All observable CKKS state fields are replaced.
        """

        self.data = other.data
        self.level = other.level
        self.scale = other.scale
        self.context_id = other.context_id
        self.prime_ids = other.prime_ids
        self.polynomial_domain = other.polynomial_domain
        self.modulus_basis = other.modulus_basis
        self.residue_representation = other.residue_representation
        return self

    def __str__(self) -> str:
        return (
            "Ciphertext("
            f"level={self.level}, scale={self.scale}, polynomial_domain={self.polynomial_domain!r}, "
            f"modulus_basis={self.modulus_basis!r}, components={self.component_count}, "
            f"batch_shape={tuple(self.batch_shape)}, "
            f"prime_ids={self.prime_ids}, shape={tuple(self.data.shape)})"
        )

    __repr__ = __str__
