"""Tensor-backed CKKS key value types."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field

import torch

from fhelium.core.state import (
    ModulusBasis,
    PolynomialDomain,
    ResidueRepresentation,
)
from fhelium.core.tensor_resident import TensorResident
from fhelium.core.validation import (
    validate_context_id,
    validate_integral_tensor,
    validate_prime_ids,
)


def _validate_key_state(
    data: torch.Tensor,
    *,
    expected_ndim: int,
    prime_ids: tuple[int, ...],
    limb_axis: int,
    what: str,
    polynomial_domain: PolynomialDomain,
    modulus_basis: ModulusBasis,
    residue_representation: ResidueRepresentation,
) -> tuple[int, ...]:
    validate_integral_tensor(data, value_name=what)
    prime_ids = validate_prime_ids(prime_ids, value_name=what)
    if data.ndim != expected_ndim:
        raise ValueError(
            f"{what} data must have {expected_ndim} dimensions, "
            f"got shape {tuple(data.shape)}"
        )
    if data.size(limb_axis) != len(prime_ids):
        raise ValueError(
            f"{what} limb count does not match prime_ids: "
            f"limbs={data.size(limb_axis)}, prime_ids={prime_ids}"
        )
    if data.size(limb_axis) == 0 or data.size(-1) == 0:
        raise ValueError(f"{what} limb and coefficient axes cannot be empty")
    if polynomial_domain not in ("coefficient", "ntt"):
        raise ValueError(
            f"Unsupported {what} polynomial_domain: {polynomial_domain!r}"
        )
    if modulus_basis not in ("Q", "QP"):
        raise ValueError(f"Unsupported {what} modulus_basis: {modulus_basis!r}")
    if residue_representation not in ("standard", "montgomery"):
        raise ValueError(
            f"Unsupported {what} residue representation: "
            f"{residue_representation!r}"
        )
    if polynomial_domain == "ntt" and residue_representation != "montgomery":
        raise ValueError(
            f"NTT-domain {what} must use Montgomery representation"
        )
    return prime_ids


@dataclass
class SecretKey(TensorResident):
    r"""RNS storage for the secret polynomial $s(X)$.

    ``data`` is a dense integral ``[limb, coefficient_or_ntt_index]`` tensor;
    row $i$ is modulo ``prime_ids[i]``. Engine-generated keys are level-zero
    Q or QP values in NTT domain and Montgomery form on the engine device,
    with final extent $N$. Direct construction retains the supplied tensor
    storage. :meth:`clone` allocates independent storage; residency views may
    alias their source. The key carries context and representation metadata but
    no device owner or persistence policy.
    """

    data: torch.Tensor
    context_id: str
    prime_ids: tuple[int, ...]
    polynomial_domain: PolynomialDomain = "ntt"
    modulus_basis: ModulusBasis = "QP"
    residue_representation: ResidueRepresentation = "montgomery"

    def __post_init__(self) -> None:
        self.context_id = validate_context_id(
            self.context_id, value_name="SecretKey"
        )
        self.prime_ids = _validate_key_state(
            self.data,
            expected_ndim=2,
            prime_ids=self.prime_ids,
            limb_axis=0,
            what="SecretKey",
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
        )

    def clone(self) -> SecretKey:
        """Return the same key value in independent tensor storage."""

        return SecretKey(
            self.data.clone(),
            self.context_id,
            self.prime_ids,
            self.polynomial_domain,
            self.modulus_basis,
            self.residue_representation,
        )

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.data,)

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> SecretKey:
        return SecretKey(
            tensors[0],
            self.context_id,
            self.prime_ids,
            self.polynomial_domain,
            self.modulus_basis,
            self.residue_representation,
        )


@dataclass
class PublicKey(TensorResident):
    r"""Public encryption key for one destination secret polynomial.

    ``data`` is a dense integral
    ``[key_component=2, limb, coefficient_or_ntt_index]`` tensor. In each RNS
    row the generated components satisfy

    $$
    k_0(X)+k_1(X)s(X)=e(X)\pmod{B_0},
    $$

    where $B_0$ is Q or QP according to ``modulus_basis``. Row $i$ is modulo
    ``prime_ids[i]``; generated keys are level-zero NTT-domain Montgomery
    residues on one device. The object does not record a symbolic key-lineage
    identifier, so callers must keep the key paired with its destination
    secret key. Direct construction and component access retain/share storage;
    :meth:`clone` allocates independent storage.
    """

    data: torch.Tensor
    context_id: str
    prime_ids: tuple[int, ...]
    polynomial_domain: PolynomialDomain = "ntt"
    modulus_basis: ModulusBasis = "Q"
    residue_representation: ResidueRepresentation = "montgomery"

    def __post_init__(self) -> None:
        self.context_id = validate_context_id(
            self.context_id, value_name="PublicKey"
        )
        self.prime_ids = _validate_key_state(
            self.data,
            expected_ndim=3,
            prime_ids=self.prime_ids,
            limb_axis=1,
            what="PublicKey",
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
        )
        if self.data.size(0) != 2:
            raise ValueError(
                "PublicKey key-component axis must have size 2, "
                f"got {self.data.size(0)}"
            )

    @property
    def k0(self) -> torch.Tensor:
        return self.data[0]

    @property
    def k1(self) -> torch.Tensor:
        return self.data[1]

    def component(self, component_id: int) -> torch.Tensor:
        """Return a storage-sharing ``[limb, ntt_index]`` component view."""

        if not 0 <= component_id < 2:
            raise IndexError(
                f"PublicKey component must be 0 or 1: {component_id}"
            )
        return self.data[component_id]

    def clone(self) -> PublicKey:
        return PublicKey(
            self.data.clone(),
            self.context_id,
            self.prime_ids,
            self.polynomial_domain,
            self.modulus_basis,
            self.residue_representation,
        )

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.data,)

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> PublicKey:
        return PublicKey(
            tensors[0],
            self.context_id,
            self.prime_ids,
            self.polynomial_domain,
            self.modulus_basis,
            self.residue_representation,
        )


@dataclass
class KeySwitchKey(TensorResident):
    r"""Hybrid-RNS material for one source-to-destination key relation.

    A key generated from $s_{\mathrm{src}}(X)$ and
    $s_{\mathrm{dst}}(X)$ converts a source phase

    $$
    c_0(X)+c_1(X)s_{\mathrm{src}}(X)
    $$

    to an equivalent destination phase

    $$
    c'_0(X)+c'_1(X)s_{\mathrm{dst}}(X),
    $$

    up to configured key-switch error. The object does not store symbolic
    source/destination identifiers; the caller must preserve that direction.

    ``data`` is a dense integral
    ``[key_digit, key_component=2, limb, coefficient_or_ntt_index]`` tensor.
    ``key_digit`` is stable key-storage identity, not the local active
    ``digit_index`` used at a later level. Limb row $i$ is modulo
    ``prime_ids[i]``. Generated keys use the complete level-zero QP basis,
    NTT domain, Montgomery form, the engine integral dtype, and one engine
    device. Direct construction and digit/component access retain/share
    storage; :meth:`clone` allocates independent storage.
    """

    data: torch.Tensor
    context_id: str
    prime_ids: tuple[int, ...]
    polynomial_domain: PolynomialDomain = "ntt"
    modulus_basis: ModulusBasis = "QP"
    residue_representation: ResidueRepresentation = "montgomery"

    def __post_init__(self) -> None:
        self.context_id = validate_context_id(
            self.context_id, value_name=self.__class__.__name__
        )
        self.prime_ids = _validate_key_state(
            self.data,
            expected_ndim=4,
            prime_ids=self.prime_ids,
            limb_axis=2,
            what=self.__class__.__name__,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
        )
        if self.data.size(1) != 2:
            raise ValueError(
                f"{self.__class__.__name__} key-component axis must have size 2, "
                f"got {self.data.size(1)}"
            )

    @property
    def digit_count(self) -> int:
        return self.data.size(0)

    def digit(self, key_digit_index: int) -> torch.Tensor:
        """Return a storage-sharing ``[key_component, limb, index]`` view."""

        return self.data[key_digit_index]

    def digit_component(
        self, key_digit_index: int, component_id: int
    ) -> torch.Tensor:
        """Return a storage-sharing ``[limb, index]`` key-component view."""

        return self.data[key_digit_index, component_id]

    def clone(self):
        return self.__class__(
            self.data.clone(),
            self.context_id,
            self.prime_ids,
            self.polynomial_domain,
            self.modulus_basis,
            self.residue_representation,
        )

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.data,)

    def _with_resident_tensors(self, tensors: tuple[torch.Tensor, ...]):
        return self.__class__(
            tensors[0],
            self.context_id,
            self.prime_ids,
            self.polynomial_domain,
            self.modulus_basis,
            self.residue_representation,
        )


@dataclass(kw_only=True)
class RotationKey(KeySwitchKey):
    r"""Key-switch material from $\sigma_g(s(X))$ back to $s(X)$.

    ``rotation_step`` is the canonical signed user-visible displacement $r$,
    not the Galois element $g$. Applying the matching automorphism and this key
    produces slots $m'_j=m_{(j-r)\bmod S}$, matching
    ``torch.roll(m, shifts=r)``.
    """

    rotation_step: int

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rotation_step = self.canonical_step(
            self.rotation_step,
            ring_dimension=self.data.size(-1),
        )

    @staticmethod
    def canonical_step(step: int, *, ring_dimension: int) -> int:
        r"""Map a step modulo $S=N/2$ to ``[-S/2, S/2)``."""

        if type(step) is not int:
            raise TypeError("RotationKey step must be an integer")
        if type(ring_dimension) is not int:
            raise TypeError("RotationKey ring_dimension must be an integer")
        if ring_dimension <= 0 or ring_dimension % 2:
            raise ValueError(
                "RotationKey ring_dimension must be a positive even integer"
            )
        num_slots = ring_dimension // 2
        return (step + num_slots // 2) % num_slots - num_slots // 2

    def clone(self) -> RotationKey:
        return RotationKey(
            data=self.data.clone(),
            context_id=self.context_id,
            prime_ids=self.prime_ids,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
            rotation_step=self.rotation_step,
        )

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> RotationKey:
        return RotationKey(
            data=tensors[0],
            context_id=self.context_id,
            prime_ids=self.prime_ids,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
            rotation_step=self.rotation_step,
        )


class RelinearizationKey(KeySwitchKey):
    r"""Key-switch material from source key $s(X)^2$ to $s(X)$.

    It replaces the $d_2(X)s(X)^2$ term of a three-component product with two
    corrections, preserving level and actual scale up to key-switch error.
    """


class ConjugationKey(KeySwitchKey):
    r"""Key-switch material from $\sigma_{-1}(s(X))$ back to $s(X)$.

    After the conjugation automorphism it restores the original key relation,
    producing semantic slots $m'_j=\overline{m_j}$ up to CKKS approximation.
    """


@dataclass
class RotationKeySet(MutableMapping[int, RotationKey]):
    """Mapping from canonical signed rotation steps to matching local keys."""

    table: dict[int, RotationKey] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = tuple(self.table.items())
        self.table = {}
        for rotation_step, key in initial:
            self[rotation_step] = key

    def _canonical_lookup_step(self, rotation_step: int) -> int:
        if not self.table:
            return int(rotation_step)
        prototype = next(iter(self.table.values()))
        return RotationKey.canonical_step(
            rotation_step,
            ring_dimension=prototype.data.size(-1),
        )

    def __getitem__(self, rotation_step: int) -> RotationKey:
        return self.table[self._canonical_lookup_step(rotation_step)]

    def __setitem__(self, rotation_step: int, key: RotationKey) -> None:
        if not isinstance(key, RotationKey):
            raise TypeError(
                "RotationKeySet values must be RotationKey objects, got "
                f"{type(key).__name__}"
            )
        canonical_step = RotationKey.canonical_step(
            rotation_step,
            ring_dimension=key.data.size(-1),
        )
        if canonical_step != key.rotation_step:
            raise ValueError(
                "Rotation key step does not match mapping key: "
                f"mapping={canonical_step}, key={key.rotation_step}"
            )
        if self.table:
            prototype = next(iter(self.table.values()))
            if prototype.data.size(-1) != key.data.size(-1):
                raise ValueError(
                    "RotationKeySet cannot mix keys with different ring dimensions"
                )
        self.table[canonical_step] = key

    def __delitem__(self, rotation_step: int) -> None:
        del self.table[self._canonical_lookup_step(rotation_step)]

    def __iter__(self) -> Iterator[int]:
        return iter(self.table)

    def __len__(self) -> int:
        return len(self.table)

    def add(self, key: RotationKey) -> RotationKeySet:
        """Install a key under its self-described canonical rotation step."""

        if not isinstance(key, RotationKey):
            raise TypeError(
                "RotationKeySet values must be RotationKey objects, got "
                f"{type(key).__name__}"
            )
        self[key.rotation_step] = key
        return self


@dataclass(frozen=True)
class EvaluationKeyRequirements:
    """Evaluation-key roles and rotations required by one evaluator.

    This is a value-independent capability description. It contains no key
    tensors, device placement, generation policy, or secret-key material.
    Consumers derive requirements; applications decide how to generate, load,
    distribute, and retain matching keys.
    """

    rotation_steps: frozenset[int] = frozenset()
    requires_relinearization: bool = False
    requires_conjugation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.rotation_steps, frozenset):
            object.__setattr__(
                self, "rotation_steps", frozenset(self.rotation_steps)
            )
        for step in self.rotation_steps:
            if type(step) is not int:
                raise TypeError(
                    "Evaluation key rotation steps must be integers"
                )
        for name in ("requires_relinearization", "requires_conjugation"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if 0 in self.rotation_steps:
            object.__setattr__(
                self,
                "rotation_steps",
                frozenset(step for step in self.rotation_steps if step != 0),
            )


@dataclass
class EvaluationKeySet:
    """Validated inventory of evaluator-only CKKS key capabilities.

    The inventory contains rotation, relinearization, and conjugation keys. It
    deliberately excludes :class:`PublicKey` and :class:`SecretKey`: encryption,
    decryption, and key generation are separate capabilities from
    public evaluation. The set is not a :class:`TensorResident` and does not
    move or serialize its members as one large value, so residency policies may
    continue to manage individual keys and rotation windows independently.

    Construction and :meth:`validate` reject mixed structural key states. Key
    tensors remain ordinary primitive values and may still be passed directly
    to low-level engine operations.
    """

    rotations: RotationKeySet = field(default_factory=RotationKeySet)
    relinearization: RelinearizationKey | None = None
    conjugation: ConjugationKey | None = None

    def __post_init__(self) -> None:
        self._validate_member_types()
        self.validate()

    def _validate_member_types(self) -> None:
        """Reject capability-role substitutions before structural checks."""

        if not isinstance(self.rotations, RotationKeySet):
            raise TypeError(
                "EvaluationKeySet rotations must be a RotationKeySet"
            )
        if self.relinearization is not None and not isinstance(
            self.relinearization, RelinearizationKey
        ):
            raise TypeError(
                "EvaluationKeySet relinearization must be a "
                "RelinearizationKey or None"
            )
        if self.conjugation is not None and not isinstance(
            self.conjugation, ConjugationKey
        ):
            raise TypeError(
                "EvaluationKeySet conjugation must be a ConjugationKey or None"
            )

    def _keys(self) -> tuple[KeySwitchKey, ...]:
        optional = tuple(
            key
            for key in (self.relinearization, self.conjugation)
            if key is not None
        )
        return (*self.rotations.table.values(), *optional)

    @staticmethod
    def _structural_identity(key: KeySwitchKey) -> tuple[object, ...]:
        return (
            key.context_id,
            key.prime_ids,
            key.polynomial_domain,
            key.modulus_basis,
            key.residue_representation,
            key.data.size(-1),
            key.digit_count,
            key.data.dtype,
            key.data.device,
        )

    def validate(self) -> EvaluationKeySet:
        """Validate all current members and return this inventory.

        ``RotationKeySet`` is intentionally mutable for application-owned key
        planning, so consumers call this method again before use. The
        check re-establishes capability-role types, rotation-step mapping, and
        one shared context/prime/domain/basis/residue/ring/digit/dtype/device
        structure. It does not select or validate an evaluator engine.
        """

        self._validate_member_types()
        for step, key in self.rotations.table.items():
            if not isinstance(key, RotationKey):
                raise TypeError(
                    "EvaluationKeySet rotation values must be RotationKey "
                    f"objects, got {type(key).__name__}"
                )
            if type(step) is not int or step != key.rotation_step:
                raise ValueError(
                    "EvaluationKeySet rotation mapping does not match its key: "
                    f"mapping={step!r}, key={key.rotation_step!r}"
                )
        keys = self._keys()
        if not keys:
            return self
        expected = self._structural_identity(keys[0])
        for key in keys[1:]:
            actual = self._structural_identity(key)
            if actual != expected:
                raise ValueError(
                    "EvaluationKeySet cannot mix incompatible key states: "
                    f"expected={expected}, actual={actual}"
                )
        return self

    def require(
        self, requirements: EvaluationKeyRequirements
    ) -> EvaluationKeySet:
        """Validate and require every evaluator capability in ``requirements``.

        Required rotations are canonicalized with the inventory's ring
        dimension when at least one rotation key is present. The method checks
        capability presence only; a consuming engine must separately preflight
        context, device, and native key compatibility. It returns this mutable
        inventory for fluent validation.

        Raises:
            TypeError: If ``requirements`` or a current capability role has an
                invalid type.
            ValueError: If members are structurally inconsistent or a required
                rotation/relinearization/conjugation capability is absent.
        """

        if not isinstance(requirements, EvaluationKeyRequirements):
            raise TypeError("requirements must be an EvaluationKeyRequirements")
        self.validate()
        if requirements.rotation_steps:
            if self.rotations:
                prototype = next(iter(self.rotations.values()))
                required_steps = {
                    RotationKey.canonical_step(
                        step, ring_dimension=prototype.data.size(-1)
                    )
                    for step in requirements.rotation_steps
                }
            else:
                required_steps = set(requirements.rotation_steps)
            missing = sorted(required_steps - set(self.rotations))
            if missing:
                raise ValueError(
                    f"EvaluationKeySet is missing rotation steps: {missing}"
                )
        if (
            requirements.requires_relinearization
            and self.relinearization is None
        ):
            raise ValueError(
                "EvaluationKeySet is missing a relinearization key"
            )
        if requirements.requires_conjugation and self.conjugation is None:
            raise ValueError("EvaluationKeySet is missing a conjugation key")
        return self
