"""Tensor-backed CKKS key value types."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping
from copy import copy
from dataclasses import dataclass, field

from typing import Literal, Self, cast

import torch

from fhelium.values.state import (
    ModulusBasis,
    PolynomialDomain,
    ResidueRepresentation,
)
from fhelium.values.tensor_resident import TensorResident
from fhelium.values._validation import (
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
    require_increasing: bool = True,
) -> tuple[int, ...]:
    validate_integral_tensor(data, value_name=what)
    prime_ids = validate_prime_ids(
        prime_ids, value_name=what, require_increasing=require_increasing
    )
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
    row $i$ is modulo ``prime_ids[i]``. Engine-generated keys are depth-zero
    Q or QP values in NTT domain and Montgomery form on the engine device,
    with final extent $N$. Direct construction retains the supplied tensor
    storage. :meth:`clone` allocates independent storage; residency views may
    alias their source. The key carries representation metadata but no device
    owner or persistence policy.
    """

    data: torch.Tensor
    prime_ids: tuple[int, ...]
    polynomial_domain: PolynomialDomain = "ntt"
    modulus_basis: ModulusBasis = "QP"
    residue_representation: ResidueRepresentation = "montgomery"

    def __post_init__(self) -> None:
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

        return self._with_resident_tensors((self.data.clone(),))

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.data,)

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> SecretKey:
        result = copy(self)
        result.data = tensors[0]
        return result


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
    ``prime_ids[i]``; generated keys are depth-zero NTT-domain Montgomery
    residues on one device. The object does not record a symbolic key-lineage
    identifier, so callers must keep the key paired with its destination
    secret key. Direct construction and component access retain/share storage;
    :meth:`clone` allocates independent storage.
    """

    data: torch.Tensor
    prime_ids: tuple[int, ...]
    polynomial_domain: PolynomialDomain = "ntt"
    modulus_basis: ModulusBasis = "Q"
    residue_representation: ResidueRepresentation = "montgomery"

    def __post_init__(self) -> None:
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
        return self._with_resident_tensors((self.data.clone(),))

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.data,)

    def _with_resident_tensors(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> PublicKey:
        result = copy(self)
        result.data = tensors[0]
        return result


SUPPORTED_SEED_ROUNDS = (12,)
"""ChaCha round counts of key seeds this version can expand."""


@dataclass(frozen=True)
class SeedSpec:
    r"""Generator of the uniform component of a seeded key-switching key.

    ``prf`` names the stream cipher (``"chacha"``), ``rounds`` its round count
    and ``moduli`` the prime $q_r$ of every stored row $r$ in key row order.
    Stored row $r$ of digit $d$ is sampled by segments of $L=8$ coefficients
    when $q_r<2^{32}$ and $L=4$ otherwise. Segment $s$ reads candidates $w$ of
    $W=32$ or $64$ bits in order (each word, or each little-endian word pair)
    from the ChaCha blocks with key $(w_0,\ldots,w_3)$, nonce $\mathrm{id}$
    and 64-bit block counter $(r+2^{16}d)\cdot 2^{32}+s+2^{24}t$ for attempts
    $t=0,1,\ldots$. A candidate is accepted when
    $w q_r \bmod 2^W \ge 2^W \bmod q_r$ and yields
    $\lfloor w q_r/2^W\rfloor$ (Lemire's rejection method), so every value
    is uniform on $[0,q_r)$; residues $Ls,\ldots,Ls+L-1$ are the first $L$
    accepted values. The values are the stored NTT-domain Montgomery residues
    of $k_{d,1}$ without conversion. Rows and digits are below $2^{16}$ and
    segments below $2^{24}$, so distinct digits, rows, segments and attempts
    of one key read distinct blocks.
    """

    moduli: tuple[int, ...]
    prf: Literal["chacha"] = "chacha"
    rounds: int = 12

    def __post_init__(self) -> None:
        if self.prf != "chacha" or self.rounds not in SUPPORTED_SEED_ROUNDS:
            raise ValueError(
                f"Unsupported key seed generator {self.prf!r} with "
                f"{self.rounds} rounds; supported: chacha with "
                f"{SUPPORTED_SEED_ROUNDS} rounds"
            )
        object.__setattr__(self, "moduli", tuple(int(q) for q in self.moduli))


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
    ``digit_index`` used at a later depth. Limb row $i$ is modulo
    ``prime_ids[i]``. ``digit_prime_ids[d]`` is the Q support encrypted by stored
    plane d; consecutive planes cover consecutive stored Q rows. Storage lists
    the supported Q rows first, followed by the selected auxiliary rows; global prime IDs need not increase across that Q/P join.
    On row i its phase is $P s_{src} + e_d$ when i belongs to
    that support, and $e_d$ otherwise. Consuming at lower Q intersects each
    stored support with active Q and retains the original plane index; the
    active Q rows form one interval of the stored Q rows and are followed in
    the key-switch product by all auxiliary rows.
    Generated keys use the selected source-depth QP basis,
    NTT domain, Montgomery form, the engine integral dtype, and one engine
    device. Direct construction and digit/component access retain/share
    storage; :meth:`clone` allocates independent storage.

    Keys generated by FHElium carry a seed from which $k_{d,1}$ is derived.
    ``seed`` holds int64 rows ``[key_digit, 6]``
    $(w_0,\ldots,w_3,\mathrm{id},d)$: the 256-bit ChaCha key, a 64-bit key
    identifier and the stored digit index; ``seed_spec`` (:class:`SeedSpec`)
    names the generator and the stored-row moduli. Such a key is *expanded*
    when ``data`` holds both components and *compact* when ``data`` has shape
    ``[key_digit, 1, limb, index]`` and holds only $k_{d,0}$. Both states
    describe the same key: :meth:`expand` returns the key with $k_{d,1}$
    generated from the seed and :meth:`compact` returns it without $k_{d,1}$;
    both leave the original key unchanged. Imported keys without a seed are always
    expanded (``seed=None``, ``seed_spec=None``).

    Key-switching kernels read $k_{d,1}$ from an expanded key and generate it
    from the seed of a compact key while forming the digit products, so both
    states produce bitwise equal results. A compact key halves key memory and
    adds the generation work to every key switch: on an RTX PRO 6000
    Blackwell Max-Q (300 W) a 13-tier bootstrap runs about 6% slower and
    Eager key switches 13-16% slower; on an RTX A6000 the bootstrap latency
    is unchanged. Engine-generated keys are expanded unless the Engine is
    created with ``expand_keys=False``; :func:`fhelium.load_value` returns a
    key with a seed in compact state, and
    :meth:`fhelium.eager.Engine.install_evaluation_key` expands it according
    to the Engine's ``expand_keys``. A compact key passed directly to an
    operation runs on the generating path; call :meth:`expand` first when
    latency matters more than memory.

    The security of a seeded key rests on RLWE with $k_{d,1}$ replaced by
    ChaCha12 output, i.e. on ChaCha12 being a pseudorandom function, and on
    the quality of the Engine random stream from which key generation draws
    each key's $(w_0,\ldots,w_3,\mathrm{id})$.
    """

    data: torch.Tensor
    prime_ids: tuple[int, ...]
    digit_prime_ids: tuple[tuple[int, ...], ...]
    polynomial_domain: PolynomialDomain = "ntt"
    modulus_basis: ModulusBasis = "QP"
    residue_representation: ResidueRepresentation = "montgomery"
    seed: torch.Tensor | None = None
    seed_spec: SeedSpec | None = None

    def __post_init__(self) -> None:
        self.prime_ids = _validate_key_state(
            self.data,
            expected_ndim=4,
            prime_ids=self.prime_ids,
            limb_axis=2,
            what=self.__class__.__name__,
            polynomial_domain=self.polynomial_domain,
            modulus_basis=self.modulus_basis,
            residue_representation=self.residue_representation,
            require_increasing=False,
        )
        components = self.data.size(1)
        if components not in (1, 2) or (components == 1 and self.seed is None):
            raise ValueError(
                f"{self.__class__.__name__} key-component axis must have size "
                f"2, or 1 for a compact key with a seed, got {components}"
            )
        self.digit_prime_ids = tuple(
            tuple(rows) for rows in self.digit_prime_ids
        )
        flattened = validate_prime_ids(
            (i for rows in self.digit_prime_ids for i in rows),
            value_name="Key digit supports",
        )
        if len(self.digit_prime_ids) != self.data.size(0) or any(
            not rows for rows in self.digit_prime_ids
        ):
            raise ValueError(
                "Key digit supports must describe every stored plane"
            )
        if flattened != self.prime_ids[: len(flattened)]:
            raise ValueError(
                "Key digit supports must form the stored Q prefix before auxiliary P rows"
            )
        if (self.seed is None) != (self.seed_spec is None):
            raise ValueError("A key seed requires its SeedSpec and vice versa")
        if self.seed is not None:
            if (
                self.seed.dtype != torch.int64
                or tuple(self.seed.shape) != (self.data.size(0), 6)
                or self.seed.device != self.data.device
            ):
                raise ValueError(
                    f"{self.__class__.__name__} seed must be int64 "
                    f"[{self.data.size(0)}, 6] on the key device"
                )
            if len(cast(SeedSpec, self.seed_spec).moduli) != len(self.prime_ids):
                raise ValueError("SeedSpec moduli must list every stored row")

    @property
    def digit_count(self) -> int:
        return self.data.size(0)

    @property
    def is_expanded(self) -> bool:
        """Whether ``data`` holds the uniform component $k_{d,1}$."""

        return self.data.size(1) == 2

    @property
    def generated_seed(self) -> torch.Tensor | None:
        """Seed rows from which execution generates $k_{d,1}$, for a compact key."""

        return None if self.is_expanded else self.seed

    @property
    def generated_rounds(self) -> int:
        """ChaCha rounds of a compact key's generated $k_{d,1}$; 0 when expanded."""

        return 0 if self.is_expanded else cast(SeedSpec, self.seed_spec).rounds

    def expand(self) -> Self:
        """Return this key with $k_{d,1}$ generated from the seed on its device.

        The result shares the seed rows and holds $k_{d,0}$ and $k_{d,1}$ in
        new storage; this key is unchanged, so programs and CUDA Graphs bound
        to it remain valid. Use ``key = key.expand()``. An expanded key is
        returned as is.
        """
        if self.is_expanded:
            return self
        result = copy(self)
        result.data = self._native("expand_key_switch_key")
        return result

    def _native(self, name: str) -> torch.Tensor:
        from fhelium.native.wrapper import ckks_ops

        spec = cast(SeedSpec, self.seed_spec)
        return getattr(ckks_ops, name)(
            self.data, self.seed, list(spec.moduli), spec.prf, spec.rounds
        )

    def compact(self) -> Self:
        """Return this key holding $k_{d,0}$ and the seed only.

        The result shares the seed rows and copies $k_{d,0}$ into new storage
        of half the size; this key is unchanged, so programs and CUDA Graphs
        bound to it remain valid. The memory of $k_{d,1}$ is released once no
        reference to the expanded key remains. Use ``key = key.compact()``.
        A compact key is returned as is.
        """

        if self.seed is None:
            raise ValueError(
                f"{self.__class__.__name__} has no seed; an imported dense key "
                "cannot be compacted"
            )
        if not self.is_expanded:
            return self
        result = copy(self)
        result.data = self._native("compact_key_switch_key")
        return result

    def digit(self, key_digit_index: int) -> torch.Tensor:
        """Return a storage-sharing ``[key_component, limb, index]`` view."""

        return self.data[key_digit_index]

    def digit_component(
        self, key_digit_index: int, component_id: int
    ) -> torch.Tensor:
        """Return a storage-sharing ``[limb, index]`` key-component view."""

        return self.data[key_digit_index, component_id]

    def clone(self) -> Self:
        return self._with_resident_tensors(
            tuple(tensor.clone() for tensor in self._resident_tensors)
        )

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.data,) if self.seed is None else (self.data, self.seed)

    def _with_resident_tensors(self, tensors: tuple[torch.Tensor, ...]) -> Self:
        result = copy(self)
        result.data = tensors[0]
        if self.seed is not None:
            result.seed = tensors[1]
        return result


@dataclass(kw_only=True)
class RotationKey(KeySwitchKey):
    r"""Key-switch material from $\sigma_g(s(X))$ back to $s(X)$.

    ``rotation_step`` is the normalized signed user-visible displacement $r$,
    not the Galois element $g$. Applying the matching automorphism and this key
    produces slots $m'_j=m_{(j-r)\bmod S}$, matching
    ``torch.roll(m, shifts=r)``.
    """

    rotation_step: int

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rotation_step = self.normalize_step(
            self.rotation_step,
            ring_dimension=self.data.size(-1),
        )

    @staticmethod
    def normalize_step(step: int, *, ring_dimension: int) -> int:
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


class RelinearizationKey(KeySwitchKey):
    r"""Key-switch material from source key $s(X)^2$ to $s(X)$.

    It replaces the $d_2(X)s(X)^2$ term of a three-component product with two
    corrections, preserving depth and actual scale up to key-switch error.
    """


class ConjugationKey(KeySwitchKey):
    r"""Key-switch material from $\sigma_{-1}(s(X))$ back to $s(X)$.

    After the conjugation automorphism it restores the original key relation,
    producing semantic slots $m'_j=\overline{m_j}$ up to CKKS approximation.
    """


@dataclass(kw_only=True)
class EncapsulationKey(KeySwitchKey):
    r"""Key-switch material between the main secret $s$ and an intermediate secret $t$.

    ``direction="to_intermediate"`` switches $s$ to $t$ on the source-depth Q
    rows with auxiliary rows $P_{low}$; its digits encrypt $P_{low}E_js$ under
    $t$. ``direction="to_source"`` has one digit on [guard, full Q] with rows
    $P_{ret}$ and encrypts $g P_{ret} t$ under $s$, where $g$ is the guard
    prime. ``prime_ids`` index the auxiliary basis of that relation rather than
    the main configuration. Encapsulated modulus raising consumes one key of
    each direction.
    """

    direction: Literal["to_intermediate", "to_source"]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.direction not in {"to_intermediate", "to_source"}:
            raise ValueError(
                "EncapsulationKey direction must be to_intermediate or to_source"
            )


@dataclass
class RotationKeySet(MutableMapping[int, RotationKey]):
    """Mapping from normalized signed rotation steps to matching local keys."""

    table: dict[int, RotationKey] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = tuple(self.table.items())
        self.table = {}
        for rotation_step, key in initial:
            self[rotation_step] = key

    def _normalized_lookup_step(self, rotation_step: int) -> int:
        if not self.table:
            return int(rotation_step)
        prototype = next(iter(self.table.values()))
        return RotationKey.normalize_step(
            rotation_step,
            ring_dimension=prototype.data.size(-1),
        )

    def __getitem__(self, rotation_step: int) -> RotationKey:
        return self.table[self._normalized_lookup_step(rotation_step)]

    def __setitem__(self, rotation_step: int, key: RotationKey) -> None:
        if not isinstance(key, RotationKey):
            raise TypeError(
                "RotationKeySet values must be RotationKey objects, got "
                f"{type(key).__name__}"
            )
        normalized_step = RotationKey.normalize_step(
            rotation_step,
            ring_dimension=key.data.size(-1),
        )
        if normalized_step != key.rotation_step:
            raise ValueError(
                "Rotation key step does not match mapping key: "
                f"mapping={normalized_step}, key={key.rotation_step}"
            )
        if self.table:
            prototype = next(iter(self.table.values()))
            if prototype.data.size(-1) != key.data.size(-1):
                raise ValueError(
                    "RotationKeySet cannot mix keys with different ring dimensions"
                )
        self.table[normalized_step] = key

    def __delitem__(self, rotation_step: int) -> None:
        del self.table[self._normalized_lookup_step(rotation_step)]

    def __iter__(self) -> Iterator[int]:
        return iter(self.table)

    def __len__(self) -> int:
        return len(self.table)

    def add(self, key: RotationKey) -> RotationKeySet:
        """Install a key under its self-described normalized rotation step."""

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


_EvaluationKey = (
    RotationKey | RelinearizationKey | ConjugationKey | EncapsulationKey
)


class EvaluationKeySet:
    """Application-owned references to evaluation keys with different layouts.

    Entries are ordinary rotation, relinearization, conjugation and
    encapsulation key values.
    Adding an existing object retains one reference; distinct objects remain
    distinct even when they share storage or represented metadata. The inventory
    never generates, transfers or copies keys. Callers keep every entry paired
    with its intended secret relation and modulus parameters; prime identifiers
    and structural selection do not establish those cryptographic identities.

    Selection uses active Q/P rows, ring dimension, dtype and device supplied
    by the consumer. With all facts known, rotation groups prefer fewer
    primitive rotations, then fewer active-digit applications. Single-role
    selection prefers fewer active digits. Both then prefer smaller stored
    payloads and insertion order. Missing facts permit a unique candidate only. A selected RotationKeySet is
    a step-to-key mapping, while this inventory may retain several candidates
    for the same step. Selection returns references without changing entries.
    """

    def __init__(self, keys: Iterable[_EvaluationKey] = ()) -> None:
        self._entries: list[_EvaluationKey] = []
        self.update(keys)

    def __iter__(self) -> Iterator[_EvaluationKey]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, key: _EvaluationKey) -> EvaluationKeySet:
        """Retain one reference to a supported evaluation-key object."""
        if not isinstance(
            key,
            (RotationKey, RelinearizationKey, ConjugationKey, EncapsulationKey),
        ):
            raise TypeError(
                "EvaluationKeySet accepts rotation, relinearization, conjugation and encapsulation keys"
            )
        if all(existing is not key for existing in self._entries):
            self._entries.append(key)
        return self

    def update(self, keys: Iterable[_EvaluationKey]) -> EvaluationKeySet:
        """Add supplied objects without replacing or moving existing keys."""
        for key in keys:
            self.add(key)
        return self

    def remove(self, key: _EvaluationKey) -> None:
        """Remove this object reference, raising KeyError when it is absent."""
        for index, existing in enumerate(self._entries):
            if existing is key:
                del self._entries[index]
                return
        raise KeyError(
            "Evaluation-key object is not retained in this inventory"
        )

    def clear(self) -> None:
        """Drop inventory references while preserving references held by consumers."""
        self._entries.clear()

    @property
    def rotation_keys(self) -> tuple[RotationKey, ...]:
        return tuple(
            key for key in self._entries if isinstance(key, RotationKey)
        )

    @property
    def relinearization_keys(self) -> tuple[RelinearizationKey, ...]:
        return tuple(
            key for key in self._entries if isinstance(key, RelinearizationKey)
        )

    @property
    def conjugation_keys(self) -> tuple[ConjugationKey, ...]:
        return tuple(
            key for key in self._entries if isinstance(key, ConjugationKey)
        )

    @property
    def encapsulation_keys(self) -> tuple[EncapsulationKey, ...]:
        return tuple(
            key for key in self._entries if isinstance(key, EncapsulationKey)
        )

    @staticmethod
    def _layout(key: KeySwitchKey) -> tuple[object, ...]:
        return (
            key.prime_ids,
            key.digit_prime_ids,
            key.data.size(-1),
            key.data.dtype,
            key.device,
            key.modulus_basis,
            key.polynomial_domain,
            key.residue_representation,
        )

    def _matching(
        self,
        key_type: type,
        *,
        q_prime_ids: tuple[int, ...] | None,
        p_prime_ids: tuple[int, ...] | None,
        ring_dimension: int | None,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
        key_prime_ids: tuple[int, ...] | None,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None,
    ) -> Iterator[_EvaluationKey]:
        selected_device = None if device is None else torch.device(device)
        for key in self._entries:
            if not isinstance(key, key_type):
                continue
            if (
                key.modulus_basis,
                key.polynomial_domain,
                key.residue_representation,
            ) != ("QP", "ntt", "montgomery"):
                continue
            if (
                ring_dimension is not None
                and key.data.size(-1) != ring_dimension
            ):
                continue
            if dtype is not None and key.data.dtype != dtype:
                continue
            if selected_device is not None and key.device != selected_device:
                continue
            if key_prime_ids is not None and key.prime_ids != key_prime_ids:
                continue
            if (
                digit_prime_ids is not None
                and key.digit_prime_ids != digit_prime_ids
            ):
                continue
            represented_q = tuple(
                i for rows in key.digit_prime_ids for i in rows
            )
            key_p = key.prime_ids[len(represented_q) :]
            if p_prime_ids is not None and key_p != p_prime_ids:
                continue
            if q_prime_ids is not None:
                if (
                    not q_prime_ids
                    or tuple(i for i in represented_q if i in q_prime_ids)
                    != q_prime_ids
                ):
                    continue
                start = key.prime_ids.index(q_prime_ids[0])
                if (
                    key.prime_ids[start : start + len(q_prime_ids)]
                    != q_prime_ids
                ):
                    continue
            yield key

    @staticmethod
    def _rank(
        key: KeySwitchKey, q_prime_ids: tuple[int, ...]
    ) -> tuple[int, int]:
        active_digits = sum(
            any(i in q_prime_ids for i in rows) for rows in key.digit_prime_ids
        )
        return active_digits, key.nbytes

    def select_rotations(
        self,
        steps: Iterable[int],
        *,
        allow_composed: bool = False,
        q_prime_ids: tuple[int, ...] | None = None,
        p_prime_ids: tuple[int, ...] | None = None,
        ring_dimension: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        key_prime_ids: tuple[int, ...] | None = None,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
    ) -> RotationKeySet | None:
        """Select a complete shared-layout rotation group, or leave it unresolved.

        With ``allow_composed``, each requested step uses a direct key when
        available in that layout group, otherwise its signed-power-of-two path.
        The returned mapping contains the primitive keys for those paths.
        Group comparison counts each path invocation, including repeated use
        of the same primitive key; payload size counts each selected key once.
        Existing key operands may constrain ``key_prime_ids`` and
        ``digit_prime_ids``. Zero steps require no key. Active Q intersections
        retain stored plane identities; selection never repartitions key data.
        Omitted p_prime_ids leaves auxiliary rows unconstrained; each candidate
        supplies its own known P. Unknown source geometry remains unresolved.
        """
        from fhelium.utils.rotation import (
            decompose_signed_power_of_two_rotation,
        )

        requested = tuple(steps)
        if ring_dimension is not None:
            requested = tuple(
                RotationKey.normalize_step(step, ring_dimension=ring_dimension)
                for step in requested
            )
        if not any(requested):
            return RotationKeySet()
        groups: dict[tuple[object, ...], dict[int, list[RotationKey]]] = {}
        for key in self._matching(
            RotationKey,
            q_prime_ids=q_prime_ids,
            p_prime_ids=p_prime_ids,
            ring_dimension=ring_dimension,
            dtype=dtype,
            device=device,
            key_prime_ids=key_prime_ids,
            digit_prime_ids=digit_prime_ids,
        ):
            key = cast(RotationKey, key)
            group = groups.setdefault(self._layout(key), {})
            group.setdefault(key.rotation_step, []).append(key)
        complete: list[tuple[RotationKeySet, bool, tuple[int, int, int]]] = []
        for group in groups.values():
            prototype = next(iter(group.values()))[0]
            required = tuple(
                RotationKey.normalize_step(
                    step, ring_dimension=prototype.data.size(-1)
                )
                for step in requested
            )
            primitive_steps = []
            for step in required:
                if not step:
                    continue
                if step in group:
                    primitive_steps.append(step)
                elif allow_composed:
                    primitive_steps.extend(
                        decompose_signed_power_of_two_rotation(
                            step, prototype.data.size(-1) // 2
                        )
                    )
                else:
                    primitive_steps.append(step)
            invocations = tuple(primitive_steps)
            primitive_steps = tuple(dict.fromkeys(invocations))
            if not all(step in group for step in primitive_steps):
                continue
            complete.append(
                (
                    RotationKeySet(
                        {step: group[step][0] for step in primitive_steps}
                    ),
                    all(len(group[step]) == 1 for step in primitive_steps),
                    (
                        len(invocations),
                        sum(
                            self._rank(group[step][0], q_prime_ids or ())[0]
                            for step in invocations
                        ),
                        sum(group[step][0].nbytes for step in primitive_steps),
                    ),
                )
            )
        known = all(
            value is not None
            for value in (
                q_prime_ids,
                ring_dimension,
                dtype,
                device,
            )
        )
        if not complete or (
            not known and (len(complete) != 1 or not complete[0][1])
        ):
            return None
        if not known:
            return complete[0][0]
        return min(complete, key=lambda choice: choice[2])[0]

    def _select_one(self, key_type: type, **facts) -> _EvaluationKey | None:
        candidates = tuple(self._matching(key_type, **facts))
        if not candidates:
            return None
        if any(
            facts[name] is None
            for name in (
                "q_prime_ids",
                "ring_dimension",
                "dtype",
                "device",
            )
        ):
            return candidates[0] if len(candidates) == 1 else None
        return min(
            candidates, key=lambda key: self._rank(key, facts["q_prime_ids"])
        )

    def select_relinearization(
        self,
        *,
        q_prime_ids: tuple[int, ...] | None = None,
        p_prime_ids: tuple[int, ...] | None = None,
        ring_dimension: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        key_prime_ids: tuple[int, ...] | None = None,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
    ) -> RelinearizationKey | None:
        """Select relinearization material, with an optional auxiliary-row constraint."""
        key = self._select_one(
            RelinearizationKey,
            q_prime_ids=q_prime_ids,
            p_prime_ids=p_prime_ids,
            ring_dimension=ring_dimension,
            dtype=dtype,
            device=device,
            key_prime_ids=key_prime_ids,
            digit_prime_ids=digit_prime_ids,
        )
        return key if isinstance(key, RelinearizationKey) else None

    def select_conjugation(
        self,
        *,
        q_prime_ids: tuple[int, ...] | None = None,
        p_prime_ids: tuple[int, ...] | None = None,
        ring_dimension: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        key_prime_ids: tuple[int, ...] | None = None,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
    ) -> ConjugationKey | None:
        """Select conjugation material, with an optional auxiliary-row constraint."""
        key = self._select_one(
            ConjugationKey,
            q_prime_ids=q_prime_ids,
            p_prime_ids=p_prime_ids,
            ring_dimension=ring_dimension,
            dtype=dtype,
            device=device,
            key_prime_ids=key_prime_ids,
            digit_prime_ids=digit_prime_ids,
        )
        return key if isinstance(key, ConjugationKey) else None
