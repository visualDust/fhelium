r"""Normalized Bootstrap execution and bound prepared-core materials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.structural import ModulusRaising
from fhelium.experimental.bootstrap.linear.execution import (
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

    def clear_key_bindings(self) -> None:
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
    """Bind the selected component objects and normalized-core configuration."""

    engine: Engine
    coeffs_to_slots: tuple[Any, ...]
    coeffs_to_slots_evaluator: Any
    modular_reduction: Any
    slots_to_coeffs: tuple[Any, ...]
    slots_to_coeffs_evaluator: Any
    modulus_raise_target_depth: int
    batch_modular_branches: bool
    compress_diagonals: bool
    message_domain: str
    modulus_raising: ModulusRaising
    source_depth: int
    input_scale: float
    normalization_factor: float
    target_scales: tuple[float, ...] | None
    modup_method: Literal["exact", "crt", "normalized_crt"]

    def evaluate(
        self,
        coordinate: Ciphertext,
        evaluation_keys: EvaluationKeySet | None,
        materials: _CoreMaterials,
        *,
        retain_diagonals: bool = True,
        retain_constants: bool = True,
        prepare_materials: bool = True,
    ) -> Ciphertext:
        r"""Refresh source-basis coordinates at the configured input scale.

        ModRaise, CTS, periodic reduction and STC preserve the existing order of
        arithmetic and rounding. The result's scale precedes entry compensation.
        """
        if (
            coordinate.scale != self.input_scale
            or coordinate.depth != self.source_depth
        ):
            raise ValueError(
                "Bootstrap core input must use its configured coordinate depth and scale"
            )
        ops = BootstrapArithmetic(
            self.engine,
            materials.constants if retain_constants else {},
            _prepare_materials=prepare_materials,
            _scales=self.target_scales,
            modup_method=self.modup_method,
            _relinearization_bindings=materials.relinearization_keys,
            _explicit_relinearization_keys=materials.explicit_relinearization_keys,
        )
        raised = self.modulus_raising(
            self.engine,
            coordinate,
            target_depth=self.modulus_raise_target_depth,
        )
        try:
            transformed = _apply_linear_transform(
                ops,
                self.coeffs_to_slots,
                self.coeffs_to_slots_evaluator,
                raised,
                evaluation_keys=evaluation_keys,
                key_bindings=materials.rotation_paths,
                diagonal_cache=materials.diagonals,
                retain_diagonals=retain_diagonals,
                compress_diagonals=self.compress_diagonals,
                prepare_materials=prepare_materials,
            )
        finally:
            if not retain_diagonals:
                materials.diagonals.clear()
        del coordinate, raised
        transformed = ops.multiply_scalar(
            transformed, self.normalization_factor
        )

        if prepare_materials:
            assert evaluation_keys is not None
            conjugation_key = evaluation_keys.select_conjugation(
                q_prime_ids=transformed.prime_ids,
                p_prime_ids=None,
                ring_dimension=self.engine.config.N,
                dtype=transformed.data.dtype,
                device=transformed.device,
            )
            if conjugation_key is None:
                raise ValueError(
                    f"Missing conjugation key covering depth {transformed.depth}"
                )
            materials.conjugation_keys[transformed.depth] = conjugation_key
            relinearization_key = evaluation_keys
        else:
            conjugation_key = materials.conjugation_keys[transformed.depth]
            relinearization_key = materials.relinearization_keys
        conjugated = self.engine.conjugate(
            transformed, conjugation_key, modup_method=ops.modup_method
        )
        real = self.engine.add(transformed, conjugated)
        if self.message_domain == "real":
            del transformed, conjugated
            result = self.modular_reduction.evaluate(
                ops,
                real,
                relinearization_key=relinearization_key,
                conjugation_key=conjugation_key,
            )
            del real
        else:
            imaginary = self.engine.subtract(transformed, conjugated)
            del transformed, conjugated
            imaginary = ops.multiply_by_monomial(
                imaginary, 3 * self.engine.num_slots
            )
            if self.batch_modular_branches:
                branches = real.with_data(
                    torch.stack((real.data, imaginary.data), dim=1)
                )
                del real, imaginary
                reduced_branches = self.modular_reduction.evaluate(
                    ops,
                    branches,
                    relinearization_key=relinearization_key,
                    conjugation_key=conjugation_key,
                )
                del branches
                real = reduced_branches.with_data(reduced_branches.data[:, 0])
                imaginary = reduced_branches.with_data(
                    reduced_branches.data[:, 1]
                )
                del reduced_branches
            else:
                real = self.modular_reduction.evaluate(
                    ops,
                    real,
                    relinearization_key=relinearization_key,
                    conjugation_key=conjugation_key,
                )
                imaginary = self.modular_reduction.evaluate(
                    ops,
                    imaginary,
                    relinearization_key=relinearization_key,
                    conjugation_key=conjugation_key,
                )
            imaginary = ops.multiply_by_monomial(
                imaginary, self.engine.num_slots
            )
            real, imaginary = ops.align_depths(real, imaginary)
            result = self.engine.add(real, imaginary)
            del real, imaginary
        try:
            result = _apply_linear_transform(
                ops,
                self.slots_to_coeffs,
                self.slots_to_coeffs_evaluator,
                result,
                evaluation_keys=evaluation_keys,
                key_bindings=materials.rotation_paths,
                diagonal_cache=materials.diagonals,
                retain_diagonals=retain_diagonals,
                compress_diagonals=self.compress_diagonals,
                prepare_materials=prepare_materials,
            )
        finally:
            if not retain_diagonals:
                materials.diagonals.clear()
        if self.message_domain == "real":
            # Reconstruct the real-valued slot polynomial after T(H rho(r)).
            # Conjugation/addition preserve depth and actual scale.
            if prepare_materials:
                assert evaluation_keys is not None
                final_conjugation_key = evaluation_keys.select_conjugation(
                    q_prime_ids=result.prime_ids,
                    p_prime_ids=None,
                    ring_dimension=self.engine.config.N,
                    dtype=result.data.dtype,
                    device=result.device,
                )
                if final_conjugation_key is None:
                    raise ValueError(
                        f"Missing conjugation key covering depth {result.depth}"
                    )
                materials.conjugation_keys[result.depth] = final_conjugation_key
            else:
                final_conjugation_key = materials.conjugation_keys[result.depth]
            result = self.engine.add(
                result,
                self.engine.conjugate(
                    result, final_conjugation_key, modup_method=ops.modup_method
                ),
            )
        return result


class PreparedBootstrapCore:
    """Callable owning one core's component bindings, keys and prepared materials.

    ``FullSlotBootstrap.prepare_core`` constructs this object for a post-entry
    coordinate. Its signature and device fix value state, shape, dtype and
    layout without retaining the prototype payload. Storage strides may differ:
    encoded materials are shared by strided inputs and packed Graph buffers. Component objects and evaluation-key
    values are selected during preparation and retained in ordinary lookup mappings.
    Appending or removing keys in the caller's inventory leaves these selections unchanged.
    Configure components before preparation and prepare again after changing
    them. Different prepared cores have independent material dictionaries.

    A captured Graph retains its callable and therefore these materials.
    Close every Graph using this object before calling ``close``. Closing
    releases the material and key references and ends host-call use.
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
    def evaluation_keys(self) -> EvaluationKeySet:
        """Return a reference inventory of this core's selected evaluation keys."""
        return (
            EvaluationKeySet()
            if self._materials is None
            else self._materials.selected_keys()
        )

    @property
    def cached_diagonal_bytes(self) -> int:
        """Return logical bytes in the owned encoded diagonal materials."""
        return 0 if self._materials is None else self._materials.diagonal_bytes

    @property
    def cached_constant_bytes(self) -> int:
        """Return logical bytes in the owned scalar and monomial materials."""
        return 0 if self._materials is None else self._materials.constant_bytes

    def __call__(self, coordinate: Ciphertext) -> Ciphertext:
        r"""Refresh an input with the prepared signature and device.

        Evaluation consumes the bound material tables. A missing entry raises
        rather than encoding a new rounding sample during execution or capture.
        Output remains in normalized coordinates for ``bootstrap.finish``.
        """
        algorithm, materials = self._algorithm, self._materials
        if algorithm is None or materials is None:
            raise RuntimeError("Prepared Bootstrap core is closed")
        expected = self.input_signature
        data_signature = dict(expected.tensors)['data']
        metadata_matches = all(
            getattr(coordinate, name) == value
            for name, value in expected.metadata
        )
        data = coordinate.data
        if (
            coordinate.device != self.device
            or not metadata_matches
            or tuple(data.shape) != data_signature.shape
            or data.dtype != data_signature.dtype
            or data.layout != data_signature.layout
            or bool(data.requires_grad) != data_signature.requires_grad
        ):
            raise ValueError(
                'Bootstrap core input differs from its prepared signature or device'
            )
        return algorithm.evaluate(
            coordinate, None, materials, prepare_materials=False
        )

    def close(self) -> None:
        """Release bindings after all Graphs using this core have been closed."""
        self._materials = None
        self._algorithm = None


class PreparedBootstrap:
    """Complete entry/core/finish refresh with prepared public materials and keys.

    The instance is an ordinary callable for Eager or Compile. Its core owns the
    encoded materials; neither the preparation input nor an output is retained.
    Close captured Graphs before closing this owner. Existing linked executables
    retain their own bound Tensor references.
    """

    def __init__(
        self, core: PreparedBootstrapCore, input_signature: ValueSignature
    ):
        self.core = core
        self.input_signature = input_signature

    @property
    def evaluation_keys(self) -> EvaluationKeySet:
        """Return this prepared execution's selected primitive key references."""
        return self.core.evaluation_keys

    @property
    def cached_diagonal_bytes(self) -> int:
        return self.core.cached_diagonal_bytes

    @property
    def cached_constant_bytes(self) -> int:
        return self.core.cached_constant_bytes

    def __call__(self, value: Ciphertext) -> Ciphertext:
        from fhelium.experimental.bootstrap.structural import (
            _prepare_entry,
            _rescale_to_structural_basis,
        )

        expected = self.input_signature
        signature = dict(expected.tensors)['data']
        if (
            any(getattr(value, key) != item for key, item in expected.metadata)
            or tuple(value.data.shape) != signature.shape
            or value.data.dtype != signature.dtype
            or value.device != self.core.device
        ):
            raise ValueError(
                'Bootstrap input differs from its prepared signature or device'
            )
        algorithm = self.core._algorithm
        if algorithm is None:
            raise RuntimeError('Prepared Bootstrap is closed')
        engine = algorithm.engine
        if value.depth != algorithm.source_depth:
            value = _prepare_entry(
                engine, value, target_scale=algorithm.input_scale
            )
            value = _rescale_to_structural_basis(
                engine, value, source_depth=algorithm.source_depth
            )
        ratio = value.scale / algorithm.input_scale
        coordinate = engine.reinterpret_at_scale(value, algorithm.input_scale)
        output = self.core(coordinate)
        return engine.reinterpret_at_scale(output, output.scale * ratio)

    def close(self) -> None:
        """Release the prepared owner after its captured Graphs are closed."""
        self.core.close()
