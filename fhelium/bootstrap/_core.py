r"""Normalized Bootstrap execution and bound prepared-core materials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from fhelium.eager import Engine
from fhelium.runtime.signature import ValueSignature
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    EvaluationKeySet,
    RotationKey,
    RelinearizationKey,
    ConjugationKey,
    Plaintext,
)
from fhelium.algorithms._arithmetic import Arithmetic
from fhelium.algorithms.linear._execution import (
    _apply_linear_transform,
)


def _material_bytes(value: object) -> int:
    if isinstance(value, CompressedPlaintext):
        tensors = value._resident_tensors
    else:
        data = value.data if isinstance(value, Plaintext) else value
        tensors = (data,) if isinstance(data, torch.Tensor) else ()
    return sum(t.numel() * t.element_size() for t in tensors)


@dataclass
class _CoreMaterials:
    """Hold encoded materials and selected primitive key references for one owner."""

    diagonals: dict[
        tuple[torch.device, int, int, int, int, float],
        Plaintext | CompressedPlaintext,
    ] = field(default_factory=dict)
    constants: dict[object, object] = field(default_factory=dict)
    rotation_paths: dict[
        tuple[int, int, int], dict[int, tuple[RotationKey, ...]]
    ] = field(default_factory=dict)
    relinearization_keys: dict[int, RelinearizationKey] = field(
        default_factory=dict
    )
    conjugation_keys: dict[int, ConjugationKey] = field(default_factory=dict)
    explicit_relinearization_keys: dict[int, RelinearizationKey] = field(
        default_factory=dict
    )
    raise_keys: object = None

    def clear_key_bindings(self) -> None:
        self.raise_keys = None
        self.rotation_paths.clear()
        self.relinearization_keys.clear()
        self.conjugation_keys.clear()
        self.explicit_relinearization_keys.clear()

    def selected_keys(self) -> EvaluationKeySet:
        values = [
            key
            for stage in self.rotation_paths.values()
            for path in stage.values()
            for key in path
        ]
        return EvaluationKeySet(
            (
                *values,
                *self.relinearization_keys.values(),
                *self.conjugation_keys.values(),
                *self.explicit_relinearization_keys.values(),
            )
        )

    @property
    def diagonal_bytes(self) -> int:
        return sum(_material_bytes(value) for value in self.diagonals.values())

    @property
    def constant_bytes(self) -> int:
        return sum(_material_bytes(value) for value in self.constants.values())

    def clear(self) -> None:
        self.diagonals.clear()
        self.constants.clear()
        self.clear_key_bindings()


@dataclass(frozen=True)
class _CoreAlgorithm:
    """Bind one plan to an Engine and evaluate its steps in a fixed order."""

    engine: Engine
    plan: Any

    def arithmetic(
        self, materials: _CoreMaterials, *, prepare_materials: bool = True
    ) -> Arithmetic:
        return Arithmetic(
            self.engine,
            materials.constants,
            _prepare_materials=prepare_materials,
            _scales=self.plan.target_scales,
            basis_conversion=self.plan.basis_conversion,
            _relinearization_bindings=materials.relinearization_keys,
            _explicit_relinearization_keys=materials.explicit_relinearization_keys,
        )

    def raise_modulus(self, coordinate: Ciphertext, keys) -> Ciphertext:
        return self.plan.modulus_raise._raise(
            self.engine,
            coordinate,
            plan=self.plan,
            target_depth=self.plan.raise_target_depth,
            keys=keys,
        )

    def _linear(self, ops, stages, plan, value, keys, materials, prepare):
        return _apply_linear_transform(
            ops,
            stages,
            plan,
            value,
            evaluation_keys=keys,
            key_bindings=materials.rotation_paths,
            diagonal_cache=materials.diagonals,
            retain_diagonals=True,
            compress_diagonals=self.plan.compress_diagonals,
            prepare_materials=prepare,
        )

    def coeffs_to_slots(self, ops, value, keys, materials, prepare):
        transformed = self._linear(
            ops,
            self.plan.coeffs_to_slots,
            self.plan.coeffs_to_slots_plan,
            value,
            keys,
            materials,
            prepare,
        )
        if not self.plan.normalization_depths:
            return transformed
        return ops.multiply_scalar(transformed, self.plan.normalization_factor)

    def _conjugation_key(self, value, keys, materials, prepare):
        if not prepare:
            return materials.conjugation_keys[value.depth]
        assert keys is not None
        key = keys.select_conjugation(
            q_prime_ids=value.prime_ids,
            p_prime_ids=None,
            ring_dimension=self.engine.config.N,
            dtype=value.data.dtype,
            device=value.device,
        )
        if key is None:
            raise ValueError(
                f"Missing conjugation key covering depth {value.depth}"
            )
        materials.conjugation_keys[value.depth] = key
        return key

    def split(self, ops, value, keys, materials, prepare):
        """Return $(z+\\bar z, X^{3S}(z-\\bar z))$, or the real part alone."""
        key = self._conjugation_key(value, keys, materials, prepare)
        conjugated = self.engine.conjugate(
            value, key, basis_conversion=ops.basis_conversion
        )
        real = self.engine.add(value, conjugated)
        if self.plan.message_domain == "real":
            return real, None
        imaginary = self.engine.subtract(value, conjugated)
        del value, conjugated
        return real, ops.multiply_by_monomial(
            imaginary, 3 * self.engine.num_slots
        )

    def eval_mod(self, ops, real, imaginary, keys, materials, prepare):
        relinearization = keys if prepare else materials.relinearization_keys
        reduce = self.plan.eval_mod.evaluate
        if imaginary is None:
            return reduce(ops, real, relinearization_key=relinearization), None
        if self.plan.batch_branches:
            branches = real.with_data(
                torch.stack((real.data, imaginary.data), dim=1)
            )
            del real, imaginary
            reduced = reduce(ops, branches, relinearization_key=relinearization)
            del branches
            return (
                reduced.with_data(reduced.data[:, 0]),
                reduced.with_data(reduced.data[:, 1]),
            )
        return (
            reduce(ops, real, relinearization_key=relinearization),
            reduce(ops, imaginary, relinearization_key=relinearization),
        )

    def merge(self, ops, real, imaginary):
        if imaginary is None:
            return real
        imaginary = ops.multiply_by_monomial(imaginary, self.engine.num_slots)
        real, imaginary = ops.align_depths(real, imaginary)
        return self.engine.add(real, imaginary)

    def slots_to_coeffs(self, ops, value, keys, materials, prepare):
        result = self._linear(
            ops,
            self.plan.slots_to_coeffs,
            self.plan.slots_to_coeffs_plan,
            value,
            keys,
            materials,
            prepare,
        )
        if self.plan.message_domain == "real":
            # Reconstruct the real-valued slot polynomial after T(H rho(r)).
            key = self._conjugation_key(result, keys, materials, prepare)
            result = self.engine.add(
                result,
                self.engine.conjugate(result, key, basis_conversion=ops.basis_conversion),
            )
        return result

    def evaluate(
        self,
        coordinate: Ciphertext,
        evaluation_keys: EvaluationKeySet | None,
        materials: _CoreMaterials,
        *,
        prepare_materials: bool = True,
    ) -> Ciphertext:
        r"""Refresh source-basis coordinates at the plan's input scale.

        The result's scale precedes entry compensation.
        """
        plan = self.plan
        if (
            coordinate.scale != plan.input_scale
            or coordinate.depth != plan.source_depth
        ):
            raise ValueError(
                "Bootstrap core input must use the plan's source depth and input scale"
            )
        keys, prepare = evaluation_keys, prepare_materials
        ops = self.arithmetic(materials, prepare_materials=prepare)
        if prepare:
            materials.raise_keys = self.plan.modulus_raise._select_keys(
                self.engine, self.plan, keys, coordinate.device
            )
        raised = self.raise_modulus(coordinate, materials.raise_keys)
        del coordinate
        value = self.coeffs_to_slots(ops, raised, keys, materials, prepare)
        del raised
        real, imaginary = self.split(ops, value, keys, materials, prepare)
        del value
        real, imaginary = self.eval_mod(
            ops, real, imaginary, keys, materials, prepare
        )
        value = self.merge(ops, real, imaginary)
        del real, imaginary
        return self.slots_to_coeffs(ops, value, keys, materials, prepare)


def _entry(engine: Engine, plan, value: Ciphertext) -> tuple[Ciphertext, float]:
    """Move an input to the source basis and reinterpret it at the input scale.

    An earlier depth reaches ``source_depth`` through ``mod_switch_to_depth``.
    An input at actual scale $\\Delta_b$ is then multiplied by the integer
    $k=\\max(1,\\operatorname{round}(\\Delta_0/\\Delta_b))$, which keeps
    depth and enlarges the message coordinate $km/q_b$ relative to EvalMod
    error. The returned ratio $k\\Delta_b/\\Delta_0$ restores the message
    after the refresh.
    """
    if value.depth > plan.source_depth:
        raise ValueError(
            f"Bootstrap input depth {value.depth} exceeds the plan source depth "
            f"{plan.source_depth}"
        )
    if value.depth != plan.source_depth:
        value = engine.mod_switch_to_depth(value, plan.source_depth)
    lift = max(1, round(plan.input_scale / value.scale))
    if lift > 1:
        value = engine.multiply_integer_scalar(value, lift)
    ratio = value.scale * lift / plan.input_scale
    return engine.reinterpret_at_scale(value, plan.input_scale), ratio


class PreparedBootstrap:
    """Caller-owned encoded constants, diagonals and selected keys for one plan.

    :func:`fhelium.bootstrap.prepare` creates it for one input signature and
    device. Calling it refreshes an input without encoding or key selection, so
    Eager execution, ``fhelium.compile`` capture and CUDA Graph replay read only
    the prepared materials. Drop the object, or call :meth:`close` after
    closing Graphs that use it, to release them.
    """

    def __init__(
        self,
        algorithm: _CoreAlgorithm,
        materials: _CoreMaterials,
        input_signature: ValueSignature,
        device: torch.device,
    ) -> None:
        self._algorithm: _CoreAlgorithm | None = algorithm
        self._materials: _CoreMaterials | None = materials
        self.input_signature = input_signature
        self.device = device

    @property
    def plan(self):
        assert self._algorithm is not None
        return self._algorithm.plan

    @property
    def evaluation_keys(self) -> EvaluationKeySet:
        """Reference inventory of the selected evaluation keys."""
        return (
            EvaluationKeySet()
            if self._materials is None
            else self._materials.selected_keys()
        )

    @property
    def diagonal_bytes(self) -> int:
        """Bytes of the encoded transform diagonals."""
        return 0 if self._materials is None else self._materials.diagonal_bytes

    @property
    def constant_bytes(self) -> int:
        """Bytes of the encoded scalar and monomial constants."""
        return 0 if self._materials is None else self._materials.constant_bytes

    def __call__(self, value: Ciphertext) -> Ciphertext:
        algorithm, materials = self._algorithm, self._materials
        if algorithm is None or materials is None:
            raise RuntimeError("Prepared bootstrap is closed")
        expected = self.input_signature
        data = dict(expected.tensors)["data"]
        if (
            any(getattr(value, key) != item for key, item in expected.metadata)
            or tuple(value.data.shape) != data.shape
            or value.data.dtype != data.dtype
            or value.device != self.device
        ):
            raise ValueError(
                "Bootstrap input differs from its prepared signature or device"
            )
        engine = algorithm.engine
        coordinate, ratio = _entry(engine, algorithm.plan, value)
        output = algorithm.evaluate(
            coordinate, None, materials, prepare_materials=False
        )
        return engine.reinterpret_at_scale(output, output.scale * ratio)

    def close(self) -> None:
        """Release materials after every Graph using this object is closed."""
        self._materials = None
        self._algorithm = None
