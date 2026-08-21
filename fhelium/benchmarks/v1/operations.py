"""Benchmark CKKS public-call latency across every active Q-chain depth."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

from fhelium import Ciphertext, CkksConfig, CkksEngine, Plaintext, Preset
from fhelium.benchmarks.model import (
    BenchmarkCheck,
    BenchmarkDefinition,
    BenchmarkMetric,
    BenchmarkProfile,
    BenchmarkResult,
    BenchmarkTimedBoundary,
    ProgressCallback,
)
from fhelium.benchmarks.registry import register_benchmark
from fhelium.benchmarks.v1.model import BenchmarkExecution
from fhelium.benchmarks.timing import (
    measure,
    synchronize,
)

BENCHMARK_NAME = "ckks-depth-aware-single-operations"
FIXED_PRESET = Preset.slots8192_scale40_levels7_int64
FIXED_NTT_BACKEND = "radix2_indexed"
DEFAULT_SEED = 20260807
DEFAULT_NONCE = 0
ROTATION_STEP = 1
RESCALE_ROUNDING: Literal["nearest", "floor"] = "nearest"

CANONICAL_OPERATIONS = (
    "encrypt",
    "decrypt",
    "coefficient_domain_to_ntt_domain",
    "ntt_domain_to_coefficient_domain",
    "add",
    "add_plaintext",
    "multiply_plaintext",
    "multiply",
    "relinearize",
    "rotate_with_key",
    "rescale_to_next_level",
    "mod_switch_to_next_level",
)
_NEXT_LEVEL_OPERATIONS = frozenset(
    {"rescale_to_next_level", "mod_switch_to_next_level"}
)

# These are the existing public-operation validation limits. They are not
# inferred from benchmark output and must not be widened to make a sweep pass.
_CORRECTNESS_ATOL = {
    "encrypt": 1e-5,
    "decrypt": 1e-5,
    "coefficient_domain_to_ntt_domain": 2e-5,
    "ntt_domain_to_coefficient_domain": 2e-5,
    "add": 2e-5,
    "add_plaintext": 2e-5,
    "multiply_plaintext": 3e-5,
    "multiply": 3e-5,
    "relinearize": 3e-5,
    "rotate_with_key": 2e-5,
    "rescale_to_next_level": 3e-5,
    "mod_switch_to_next_level": 2e-5,
}

_PARAMETER_SELECTION_RATIONALE = (
    "The maintained slots8192-scale40-levels7-int64 preset fixes logN=14, "
    "8,192 "
    "complex slots, seven public levels, one key-switch P prime, and an "
    "enforced 128-bit security-table budget. The indexed radix-2 backend is "
    "the one fixed implementation shared by CPU and CUDA."
)

_TIMED_BOUNDARY = BenchmarkTimedBoundary(
    id="depth-aware-single-public-ckks-call",
    description=(
        "One functional public CkksEngine call at one entry level, with "
        "synchronized device completion."
    ),
    includes=(
        "one named functional public CkksEngine call",
        "output allocation and every representation conversion specified by that call",
    ),
    excludes=(
        "engine and native-table construction",
        "key generation or lookup",
        "cleartext input construction",
        "encoding and operation-ready plaintext preparation",
        "setup encryption and domain conversion",
        "pending-product construction for relinearization or rescale",
        "correctness decryption, decoding, and state validation",
        "cleanup",
    ),
    synchronization=(
        "Synchronize the selected engine device before each sample and after "
        "the public call; CPU calls complete synchronously. Pre-sample "
        "synchronization is outside elapsed time and post-call completion is "
        "inside it."
    ),
)


@dataclass(frozen=True)
class _KeyInventory:
    secret: Any
    public: Any
    relinearization: Any
    rotation: Any

    def bytes_by_role(self) -> dict[str, int]:
        return {
            "decryption_key": _value_bytes(self.secret),
            "public_key": _value_bytes(self.public),
            "relinearization_key": _value_bytes(self.relinearization),
            "rotation_key_step_1": _value_bytes(self.rotation),
        }


@dataclass(frozen=True)
class _OperationFixture:
    operation: str
    call: Callable[[], Ciphertext | Plaintext]
    inputs: tuple[tuple[str, Ciphertext | Plaintext], ...]
    expected_slots: torch.Tensor
    predicted_exit_state: Mapping[str, Any]
    required_key_roles: tuple[str, ...]
    workload_multiplicative_depth_before_entry: int
    multiplicative_depth_added_by_call: int
    public_chain_transitions_consumed_by_call: int
    canonicalize_for_oracle: Callable[[Ciphertext | Plaintext], torch.Tensor]

    @property
    def input_states(self) -> list[dict[str, Any]]:
        return [
            {"role": role, "state": _value_state(value)}
            for role, value in self.inputs
        ]

    @property
    def logical_live_input_bytes(self) -> int:
        return sum(_value_bytes(value) for _, value in self.inputs)


@dataclass
class _LevelBundle:
    engine: CkksEngine
    level: int
    x: torch.Tensor
    y: torch.Tensor
    keys: _KeyInventory
    encoded_x: Plaintext
    ct_x: Ciphertext
    ct_y: Ciphertext
    ct_x_ntt: Ciphertext
    ct_y_ntt: Ciphertext
    add_plaintext: Plaintext
    multiply_plaintext: Plaintext
    triplet: Ciphertext
    rescale_pending: Ciphertext | None
    rescale_identity: Plaintext | None

    @classmethod
    def create(
        cls,
        engine: CkksEngine,
        *,
        level: int,
        x: torch.Tensor,
        y: torch.Tensor,
        keys: _KeyInventory,
    ) -> _LevelBundle:
        scale = engine.config.default_scale
        encoded_x = engine.encode(x, level=level, scale=scale)
        ct_x = engine.encrypt(encoded_x, keys.public)
        encoded_y = engine.encode(y, level=level, scale=scale)
        ct_y = engine.encrypt(encoded_y, keys.public)
        ct_x_ntt = engine.coefficient_domain_to_ntt_domain(ct_x)
        ct_y_ntt = engine.coefficient_domain_to_ntt_domain(ct_y)
        add_plaintext = engine.prepare_plaintext_for_addition(
            engine.encode(y, level=level, scale=scale)
        )
        multiply_plaintext = engine.prepare_plaintext_for_multiplication(
            engine.encode(y, level=level, scale=scale)
        )
        triplet = engine.multiply(ct_x_ntt, ct_y_ntt)

        rescale_pending: Ciphertext | None = None
        rescale_identity: Plaintext | None = None
        if level < engine.final_public_level:
            drop_prime = engine.rescale_to_next_drop_prime(level=level)
            identity = torch.ones(engine.num_slots, dtype=torch.float64)
            rescale_identity = engine.prepare_plaintext_for_multiplication(
                engine.encode(
                    identity,
                    level=level,
                    scale=float(drop_prime),
                )
            )
            rescale_pending = engine.ntt_domain_to_coefficient_domain(
                engine.multiply_plaintext(ct_x_ntt, rescale_identity)
            )
            predicted_scale = engine.rescale_to_next_output_scale(
                rescale_pending.scale,
                level=level,
            )
            if predicted_scale != scale:
                raise AssertionError(
                    "The fixed rescale fixture must return exactly to the "
                    f"default scale: predicted={predicted_scale!r}, "
                    f"default={scale!r}"
                )

        synchronize(engine.device)
        return cls(
            engine=engine,
            level=level,
            x=x,
            y=y,
            keys=keys,
            encoded_x=encoded_x,
            ct_x=ct_x,
            ct_y=ct_y,
            ct_x_ntt=ct_x_ntt,
            ct_y_ntt=ct_y_ntt,
            add_plaintext=add_plaintext,
            multiply_plaintext=multiply_plaintext,
            triplet=triplet,
            rescale_pending=rescale_pending,
            rescale_identity=rescale_identity,
        )

    @property
    def resident_value_bytes(self) -> int:
        values: list[Ciphertext | Plaintext] = [
            self.encoded_x,
            self.ct_x,
            self.ct_y,
            self.ct_x_ntt,
            self.ct_y_ntt,
            self.add_plaintext,
            self.multiply_plaintext,
            self.triplet,
        ]
        if self.rescale_pending is not None:
            values.append(self.rescale_pending)
        if self.rescale_identity is not None:
            values.append(self.rescale_identity)
        return sum(_value_bytes(value) for value in values)

    def fixture(self, operation: str) -> _OperationFixture:
        engine = self.engine
        level = self.level
        decode_ciphertext = lambda value: _decode_ciphertext(engine, value)
        default_scale = engine.config.default_scale

        if operation == "encrypt":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.encrypt(self.encoded_x, self.keys.public),
                inputs=(("plaintext", self.encoded_x),),
                expected_slots=self.x,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=("public_key",),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "decrypt":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.decrypt(self.ct_x, self.keys.secret),
                inputs=(("ciphertext", self.ct_x),),
                expected_slots=self.x,
                predicted_exit_state=_decrypted_plaintext_exit_state(
                    engine, level=level, scale=default_scale
                ),
                required_key_roles=("decryption_key",),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=lambda value: engine.decode(
                    _require_plaintext(value)
                ),
            )
        if operation == "coefficient_domain_to_ntt_domain":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.coefficient_domain_to_ntt_domain(self.ct_x),
                inputs=(("ciphertext", self.ct_x),),
                expected_slots=self.x,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale,
                    components=2,
                    polynomial_domain="ntt",
                    residue_representation="montgomery",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=lambda value: _decode_ciphertext(
                    engine,
                    engine.ntt_domain_to_coefficient_domain(
                        _require_ciphertext(value)
                    ),
                ),
            )
        if operation == "ntt_domain_to_coefficient_domain":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.ntt_domain_to_coefficient_domain(
                    self.ct_x_ntt
                ),
                inputs=(("ciphertext", self.ct_x_ntt),),
                expected_slots=self.x,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "add":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.add(self.ct_x, self.ct_y),
                inputs=(
                    ("lhs_ciphertext", self.ct_x),
                    ("rhs_ciphertext", self.ct_y),
                ),
                expected_slots=self.x + self.y,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "add_plaintext":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.add_plaintext(
                    self.ct_x, self.add_plaintext
                ),
                inputs=(
                    ("ciphertext", self.ct_x),
                    ("plaintext", self.add_plaintext),
                ),
                expected_slots=self.x + self.y,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "multiply_plaintext":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.multiply_plaintext(
                    self.ct_x_ntt, self.multiply_plaintext
                ),
                inputs=(
                    ("ciphertext", self.ct_x_ntt),
                    ("plaintext", self.multiply_plaintext),
                ),
                expected_slots=self.x * self.y,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale * default_scale,
                    components=2,
                    polynomial_domain="ntt",
                    residue_representation="montgomery",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=1,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=lambda value: _decode_ciphertext(
                    engine,
                    engine.ntt_domain_to_coefficient_domain(
                        _require_ciphertext(value)
                    ),
                ),
            )
        if operation == "multiply":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.multiply(self.ct_x_ntt, self.ct_y_ntt),
                inputs=(
                    ("lhs_ciphertext", self.ct_x_ntt),
                    ("rhs_ciphertext", self.ct_y_ntt),
                ),
                expected_slots=self.x * self.y,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale * default_scale,
                    components=3,
                    polynomial_domain="ntt",
                    residue_representation="montgomery",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=1,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "relinearize":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.relinearize(
                    self.triplet, self.keys.relinearization
                ),
                inputs=(("triplet_ciphertext", self.triplet),),
                expected_slots=self.x * self.y,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale * default_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=("relinearization_key",),
                workload_multiplicative_depth_before_entry=1,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "rotate_with_key":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.rotate_with_key(
                    self.ct_x, self.keys.rotation
                ),
                inputs=(("ciphertext", self.ct_x),),
                expected_slots=torch.roll(self.x, shifts=ROTATION_STEP),
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level,
                    scale=default_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=("rotation_key_step_1",),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=0,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "rescale_to_next_level":
            if self.rescale_pending is None:
                raise ValueError("rescale is not applicable at the final level")
            rescale_pending = self.rescale_pending
            output_scale = engine.rescale_to_next_output_scale(
                rescale_pending.scale,
                level=level,
            )
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.rescale_to_next_level(
                    rescale_pending,
                    rounding=RESCALE_ROUNDING,
                ),
                inputs=(("pending_ciphertext", rescale_pending),),
                expected_slots=self.x,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level + 1,
                    scale=output_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=1,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=1,
                canonicalize_for_oracle=decode_ciphertext,
            )
        if operation == "mod_switch_to_next_level":
            return _OperationFixture(
                operation=operation,
                call=lambda: engine.mod_switch_to_next_level(self.ct_x),
                inputs=(("ciphertext", self.ct_x),),
                expected_slots=self.x,
                predicted_exit_state=_ciphertext_exit_state(
                    engine,
                    level=level + 1,
                    scale=default_scale,
                    components=2,
                    polynomial_domain="coefficient",
                    residue_representation="standard",
                ),
                required_key_roles=(),
                workload_multiplicative_depth_before_entry=0,
                multiplicative_depth_added_by_call=0,
                public_chain_transitions_consumed_by_call=1,
                canonicalize_for_oracle=decode_ciphertext,
            )
        raise KeyError(f"Unsupported operation {operation!r}")


def _require_ciphertext(value: Ciphertext | Plaintext) -> Ciphertext:
    if not isinstance(value, Ciphertext):
        raise TypeError(f"Expected Ciphertext, got {type(value).__name__}")
    return value


def _require_plaintext(value: Ciphertext | Plaintext) -> Plaintext:
    if not isinstance(value, Plaintext):
        raise TypeError(f"Expected Plaintext, got {type(value).__name__}")
    return value


def _decode_ciphertext(engine: CkksEngine, value: Ciphertext) -> torch.Tensor:
    return torch.as_tensor(engine.decode(engine.decrypt(value)))


def _value_bytes(value: Any) -> int:
    data = getattr(value, "data", None)
    if isinstance(data, torch.Tensor):
        return int(data.nbytes)
    message = getattr(value, "message", None)
    if isinstance(message, torch.Tensor):
        return int(message.nbytes)
    return 0


def _value_state(value: Ciphertext | Plaintext) -> dict[str, Any]:
    data = value.data
    if isinstance(value, Ciphertext):
        return {
            "value_type": "ciphertext",
            "level": value.level,
            "scale": value.scale,
            "representation": "rns",
            "polynomial_domain": value.polynomial_domain,
            "modulus_basis": value.modulus_basis,
            "residue_representation": value.residue_representation,
            "component_count": value.component_count,
            "prime_ids": list(value.prime_ids),
            "limb_count": value.limb_count,
            "batch_shape": list(value.batch_shape),
            "ring_dimension": value.ring_dimension,
            "dtype": str(value.data.dtype),
            "device": str(value.data.device),
        }
    tensor = data if data is not None else value.message
    return {
        "value_type": "plaintext",
        "level": value.level,
        "scale": value.scale,
        "representation": value.representation,
        "polynomial_domain": value.polynomial_domain,
        "modulus_basis": value.modulus_basis,
        "residue_representation": value.residue_representation,
        "component_count": None,
        "prime_ids": list(value.prime_ids),
        "limb_count": (
            int(data.size(-2)) if value.is_rns and data is not None else None
        ),
        "batch_shape": list(value.batch_shape),
        "ring_dimension": (int(data.size(-1)) if data is not None else None),
        "dtype": str(tensor.dtype) if tensor is not None else None,
        "device": str(tensor.device) if tensor is not None else None,
    }


def _ciphertext_exit_state(
    engine: CkksEngine,
    *,
    level: int,
    scale: float,
    components: int,
    polynomial_domain: str,
    residue_representation: str,
) -> dict[str, Any]:
    prime_ids = engine.rns_layout.prime_ids(level)
    return {
        "value_type": "ciphertext",
        "level": level,
        "scale": float(scale),
        "representation": "rns",
        "polynomial_domain": polynomial_domain,
        "modulus_basis": "Q",
        "residue_representation": residue_representation,
        "component_count": components,
        "prime_ids": list(prime_ids),
        "limb_count": len(prime_ids),
        "batch_shape": [],
        "ring_dimension": engine.config.N,
        "dtype": str(engine.config.torch_dtype),
        "device": str(engine.device),
    }


def _decrypted_plaintext_exit_state(
    engine: CkksEngine, *, level: int, scale: float
) -> dict[str, Any]:
    return {
        "value_type": "plaintext",
        "level": level,
        "scale": float(scale),
        "representation": "approximate_coefficients",
        "polynomial_domain": "coefficient",
        "modulus_basis": None,
        "residue_representation": None,
        "component_count": None,
        "prime_ids": [],
        "limb_count": None,
        "batch_shape": [],
        "ring_dimension": engine.config.N,
        "dtype": str(torch.float64),
        "device": str(engine.device),
    }


def _fixed_config() -> CkksConfig:
    return CkksConfig.parse(FIXED_PRESET)


def _active_q_state(config: CkksConfig, level: int) -> dict[str, Any]:
    if type(level) is not int:
        raise TypeError("level must be an integer")
    if not 0 <= level < config.num_scale_primes:
        raise ValueError(
            f"level must be in [0, {config.num_scale_primes}), got {level}"
        )
    active = config.q_moduli[level:]
    return {
        "entry_level": level,
        "entry_chain_depth": level,
        "active_q_prime_ids": list(range(level, config.num_q_primes)),
        "active_q_count": len(active),
        "active_scale_prime_count": config.num_scale_primes - level,
        "active_q_product_bits": (math.prod(active) - 1).bit_length(),
        "available_public_transitions": config.num_scale_primes - 1 - level,
    }


def _expanded_ckks_plan(config: CkksConfig) -> dict[str, Any]:
    q_rows = [
        {
            "prime_id": index,
            "role": (
                "scale_q_prime"
                if index < config.num_scale_primes
                else "structural_base_q_prime"
            ),
            "modulus_decimal": str(modulus),
            "modulus_bits": modulus.bit_length(),
        }
        for index, modulus in enumerate(config.q_moduli)
    ]
    p_rows = [
        {
            "prime_id": config.num_q_primes + index,
            "role": "key_switch_p_prime",
            "modulus_decimal": str(modulus),
            "modulus_bits": modulus.bit_length(),
        }
        for index, modulus in enumerate(config.p_moduli)
    ]
    return {
        "preset": FIXED_PRESET.value,
        "logN": config.logN,
        "ring_dimension": config.N,
        "complex_slot_count": config.N // 2,
        "buffer_bit_length": config.buffer_bit_length,
        "torch_dtype": str(config.torch_dtype),
        "scale_bits": config.scale_bits,
        "default_scale": config.default_scale,
        "num_scale_primes": config.num_scale_primes,
        "public_level_count": config.num_scale_primes,
        "final_public_level": config.num_scale_primes - 1,
        "num_q_primes": config.num_q_primes,
        "num_p_primes": config.num_p_primes,
        "q_rows": q_rows,
        "p_rows": p_rows,
        "level_states": [
            _active_q_state(config, level)
            for level in range(config.num_scale_primes)
        ],
        "q0_product_bits": (math.prod(config.q_moduli) - 1).bit_length(),
        "p_product_bits": (math.prod(config.p_moduli) - 1).bit_length(),
        "complete_qp_product_bits": config.total_modulus_bits,
        "maximum_security_budget_bits": config.maximum_modulus_bits,
        "security_bits": config.security_bits,
        "sigma": config.sigma,
        "security_budget_enforced": config.enforce_security_budget,
        "ntt_backend": FIXED_NTT_BACKEND,
        "ciphertext_modulus_basis": "Q",
        "evaluation_key_modulus_basis": "QP",
    }


def _entry_state_axis(
    config: CkksConfig, levels: Sequence[int]
) -> dict[str, Any]:
    return {
        "axis": "public_source_level",
        "entry_chain_depth_definition": (
            "The number of leading scale-Q rows absent relative to Q_0; for "
            "a public FHElium value it equals entry_level and does not assert "
            "that the setup executed that many multiplications."
        ),
        "multiplicative_history_definition": (
            "Workload multiplicative depth before entry and depth added by "
            "the timed call are reported separately from public level."
        ),
        "construction": "direct_at_entry_level",
        "levels": [_active_q_state(config, level) for level in levels],
    }


def _benchmark_context(
    config: CkksConfig, levels: Sequence[int]
) -> dict[str, Any]:
    return {
        "ckks_plan": _expanded_ckks_plan(config),
        "parameter_selection": {
            "rationale": _PARAMETER_SELECTION_RATIONALE,
            "security_budget_status": "within_exact_builtin_budget",
            "complete_qp_product_bits": config.total_modulus_bits,
            "maximum_security_budget_bits": config.maximum_modulus_bits,
        },
        "entry_state": _entry_state_axis(config, levels),
    }


def _not_applicable_case(
    config: CkksConfig, operation: str, level: int
) -> dict[str, Any]:
    if operation not in _NEXT_LEVEL_OPERATIONS:
        raise ValueError(f"{operation!r} is not a terminal-only operation")
    if level != config.num_scale_primes - 1:
        raise ValueError("not-applicable rows are reserved for the final level")
    return {
        "operation": operation,
        **_active_q_state(config, level),
        "status": "not_applicable",
        "reason_code": "no_following_public_level",
        "reason": (
            f"{operation} requires a source below final_public_level={level}; "
            "the terminal exception path is not timed."
        ),
        "predicted_exit_state": None,
        "observed_exit_state": None,
    }


def _deterministic_messages(
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(num_slots, dtype=torch.float64)
    x = torch.complex(
        0.012 * torch.sin(index * 0.013) + 0.006 * torch.cos(index * 0.007),
        0.009 * torch.cos(index * 0.011) - 0.004 * torch.sin(index * 0.005),
    )
    y = torch.complex(
        0.010 * torch.cos(index * 0.017) - 0.003 * torch.sin(index * 0.009),
        0.007 * torch.sin(index * 0.015) + 0.002 * torch.cos(index * 0.003),
    )
    return x, y


def _validate_fixed_parameters(
    parameters: Mapping[str, Any], config: CkksConfig
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    expected_scalars = {
        "preset": FIXED_PRESET.value,
        "logN": config.logN,
        "scale_bits": config.scale_bits,
        "num_scale_primes": config.num_scale_primes,
        "num_p_primes": config.num_p_primes,
        "ntt_backend": FIXED_NTT_BACKEND,
        "ciphertext_modulus_basis": "Q",
        "rotation_step": ROTATION_STEP,
        "rescale_rounding": RESCALE_ROUNDING,
    }
    for name, expected in expected_scalars.items():
        actual = parameters.get(name)
        if actual != expected:
            raise ValueError(
                f"{name} is fixed at {expected!r} for Benchmark v1; "
                f"got {actual!r}"
            )

    configured_levels = parameters.get("levels")
    if not isinstance(configured_levels, Sequence) or isinstance(
        configured_levels, (str, bytes)
    ):
        raise ValueError("levels must be a JSON array of public levels")
    levels = tuple(configured_levels)
    if not levels:
        raise ValueError("levels must contain at least one public level")
    if any(type(level) is not int for level in levels):
        raise TypeError("levels must contain only integers")
    if len(set(levels)) != len(levels):
        raise ValueError("levels must not contain duplicates")
    for level in levels:
        _active_q_state(config, level)

    configured_operations = parameters.get("operations")
    if not isinstance(configured_operations, Sequence) or isinstance(
        configured_operations, (str, bytes)
    ):
        raise ValueError("operations must be a JSON array")
    operations = tuple(configured_operations)
    if not operations:
        raise ValueError("operations must contain at least one operation")
    if any(type(operation) is not str for operation in operations):
        raise TypeError("operations must contain only strings")
    if len(set(operations)) != len(operations):
        raise ValueError("operations must not contain duplicates")
    unsupported = tuple(
        operation
        for operation in operations
        if operation not in CANONICAL_OPERATIONS
    )
    if unsupported:
        raise ValueError(
            f"Unsupported operations {unsupported!r}; "
            f"choices: {CANONICAL_OPERATIONS!r}"
        )
    if (
        tuple(
            operation
            for operation in CANONICAL_OPERATIONS
            if operation in operations
        )
        != operations
    ):
        raise ValueError("operations must retain canonical workload order")
    return levels, operations


def _create_engine(
    parameters: Mapping[str, Any],
    config: CkksConfig,
    execution: BenchmarkExecution,
) -> CkksEngine:
    seed = parameters.get("seed", DEFAULT_SEED)
    nonce = parameters.get("nonce", DEFAULT_NONCE)
    if type(seed) is not int or type(nonce) is not int:
        raise TypeError("seed and nonce must be integers")
    return CkksEngine(
        config,
        device=execution.device,
        ntt_backend=FIXED_NTT_BACKEND,
        rng_seed=seed,
        rng_nonce=nonce,
    )


def _materialize_keys(engine: CkksEngine) -> _KeyInventory:
    secret = engine.secret_key
    public = engine.public_key
    relinearization = engine.relinearization_key
    rotation = engine.rotation_key(ROTATION_STEP)
    synchronize(engine.device)
    return _KeyInventory(
        secret=secret,
        public=public,
        relinearization=relinearization,
        rotation=rotation,
    )


def _semantic_error_stats(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | int]:
    actual = torch.as_tensor(actual).resolve_conj().resolve_neg().cpu()
    expected = torch.as_tensor(expected).resolve_conj().resolve_neg().cpu()
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Correctness output shape {tuple(actual.shape)} does not match "
            f"expected {tuple(expected.shape)}"
        )
    finite = torch.isfinite(actual)
    nonfinite_count = int((~finite).sum().item())
    if nonfinite_count:
        return {
            "max_abs_error": float(torch.finfo(torch.float64).max),
            "rms_error": float(torch.finfo(torch.float64).max),
            "relative_l2_error": float(torch.finfo(torch.float64).max),
            "nonfinite_count": nonfinite_count,
        }
    error = torch.abs(actual - expected)
    expected_norm = torch.linalg.vector_norm(expected)
    relative = torch.linalg.vector_norm(error) / expected_norm
    return {
        "max_abs_error": float(torch.max(error).item()),
        "rms_error": float(torch.sqrt(torch.mean(error.square())).item()),
        "relative_l2_error": float(relative.item()),
        "nonfinite_count": 0,
    }


def _state_mismatches(
    predicted: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    keys = set(predicted) | set(observed)
    return {
        key: {"predicted": predicted.get(key), "observed": observed.get(key)}
        for key in sorted(keys)
        if predicted.get(key) != observed.get(key)
    }


def _metric_dimensions(
    *, operation: str, level_state: Mapping[str, Any], category: str
) -> dict[str, Any]:
    return {
        "category": category,
        "operation": operation,
        "entry_level": int(level_state["entry_level"]),
        "entry_chain_depth": int(level_state["entry_chain_depth"]),
        "active_q_count": int(level_state["active_q_count"]),
        "active_q_product_bits": int(level_state["active_q_product_bits"]),
    }


def _run_depth_aware_operations(
    profile: BenchmarkProfile,
    progress: ProgressCallback,
    *,
    execution: BenchmarkExecution,
) -> BenchmarkResult:
    parameters = profile.parameters
    config = _fixed_config()
    levels, operations = _validate_fixed_parameters(parameters, config)
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    include_raw_samples = bool(parameters.get("include_raw_samples", True))
    if warmup < 0 or runs <= 0:
        raise ValueError(
            "warmup must be non-negative and runs must be positive"
        )

    progress("Creating the fixed cross-backend engine and materializing keys")
    engine = _create_engine(parameters, config, execution)
    keys = _materialize_keys(engine)
    key_bytes = keys.bytes_by_role()
    resident_key_bytes = sum(key_bytes.values())
    x, y = _deterministic_messages(engine.num_slots)

    rows: list[dict[str, Any]] = []
    metrics: list[BenchmarkMetric] = []
    checks: list[BenchmarkCheck] = []
    evidence: list[dict[str, Any]] = []
    not_applicable_cases: list[dict[str, Any]] = []

    for level in levels:
        progress(
            f"Constructing operation-ready fixtures at entry level {level}"
        )
        bundle = _LevelBundle.create(
            engine,
            level=level,
            x=x,
            y=y,
            keys=keys,
        )
        level_state = _active_q_state(config, level)
        for operation in operations:
            if (
                operation in _NEXT_LEVEL_OPERATIONS
                and level == engine.final_public_level
            ):
                not_applicable = _not_applicable_case(config, operation, level)
                not_applicable_cases.append(not_applicable)
                rows.append(not_applicable)
                continue

            progress(f"Measuring {operation} at entry level {level}")
            fixture = bundle.fixture(operation)
            for _ in range(warmup):
                fixture.call()
            synchronize(engine.device)
            gc.collect()
            timing = measure(
                fixture.call,
                warmup=0,
                runs=runs,
                device=engine.device,
                include_samples=include_raw_samples,
            )

            # Correctness and all canonicalization are deliberately outside
            # the timed samples.
            correctness_output = fixture.call()
            synchronize(engine.device)
            observed_exit_state = _value_state(correctness_output)
            mismatches = _state_mismatches(
                fixture.predicted_exit_state, observed_exit_state
            )
            decoded = fixture.canonicalize_for_oracle(correctness_output)
            synchronize(engine.device)
            error = _semantic_error_stats(decoded, fixture.expected_slots)
            atol = _CORRECTNESS_ATOL[operation]
            semantic_passed = (
                error["nonfinite_count"] == 0 and error["max_abs_error"] <= atol
            )
            dimensions = _metric_dimensions(
                operation=operation,
                level_state=level_state,
                category="latency",
            )
            required_key_bytes = sum(
                key_bytes[role] for role in fixture.required_key_roles
            )
            output_bytes = _value_bytes(correctness_output)
            samples = tuple(timing.get("samples_ms", ()))
            inverse_serial_rate = 1000.0 / timing["mean_ms"]
            packed_slot_rate = engine.num_slots * inverse_serial_rate

            row = {
                "operation": operation,
                **level_state,
                "status": "measured",
                "entry_state": fixture.input_states,
                "workload_multiplicative_depth_before_entry": (
                    fixture.workload_multiplicative_depth_before_entry
                ),
                "multiplicative_depth_added_by_call": (
                    fixture.multiplicative_depth_added_by_call
                ),
                "workload_multiplicative_depth_at_exit": (
                    fixture.workload_multiplicative_depth_before_entry
                    + fixture.multiplicative_depth_added_by_call
                ),
                "public_chain_transitions_consumed_by_call": (
                    fixture.public_chain_transitions_consumed_by_call
                ),
                "predicted_exit_state": dict(fixture.predicted_exit_state),
                "observed_exit_state": observed_exit_state,
                "mean_ms": timing["mean_ms"],
                "median_ms": timing["median_ms"],
                "min_ms": timing["min_ms"],
                "max_ms": timing["max_ms"],
                "std_ms": timing["std_ms"],
                "inverse_synchronized_serial_operations_per_second": (
                    inverse_serial_rate
                ),
                "inverse_synchronized_serial_slots_per_second": (
                    packed_slot_rate
                ),
                "logical_live_input_bytes": fixture.logical_live_input_bytes,
                "output_payload_bytes": output_bytes,
                "resident_level_fixture_bytes": bundle.resident_value_bytes,
                "required_key_roles": list(fixture.required_key_roles),
                "required_key_bytes": required_key_bytes,
                "resident_all_evaluation_key_bytes": resident_key_bytes,
                "correctness_atol": atol,
                **error,
                "state_mismatch_count": len(mismatches),
            }
            rows.append(row)
            if include_raw_samples:
                evidence.append(
                    {
                        "kind": "raw_timing_samples",
                        "operation": operation,
                        "entry_level": level,
                        "unit": "ms",
                        "samples": list(samples),
                    }
                )

            metrics.extend(
                (
                    BenchmarkMetric(
                        name="depth-aware-ckks-operation-latency",
                        value=timing["median_ms"],
                        unit="ms",
                        statistic="median",
                        direction="lower",
                        dimensions=dimensions,
                        samples=samples,
                    ),
                    BenchmarkMetric(
                        name="depth-aware-ckks-inverse-serial-rate",
                        value=inverse_serial_rate,
                        unit="operations/s",
                        statistic="inverse_mean_synchronized_latency",
                        direction="higher",
                        dimensions={**dimensions, "category": "throughput"},
                    ),
                    BenchmarkMetric(
                        name="depth-aware-ckks-packed-slot-rate",
                        value=packed_slot_rate,
                        unit="slots/s",
                        statistic="slot_count_times_inverse_mean_latency",
                        direction="higher",
                        dimensions={**dimensions, "category": "throughput"},
                    ),
                )
            )
            checks.extend(
                (
                    BenchmarkCheck(
                        name=f"{operation}-level-{level}-exit-state",
                        passed=not mismatches,
                        oracle=(
                            "Observed exit state satisfies the operation "
                            "postconditions "
                            "for level, actual scale, rows, components, domain, "
                            "basis, representation, shape, dtype, and device."
                        ),
                        metric="state_mismatch_count",
                        observed=len(mismatches),
                        comparison="==",
                        limit=0,
                        unit="fields",
                        details={
                            "operation": operation,
                            "entry_level": level,
                            "predicted_exit_state": dict(
                                fixture.predicted_exit_state
                            ),
                            "observed_exit_state": observed_exit_state,
                            "mismatches": mismatches,
                        },
                    ),
                    BenchmarkCheck(
                        name=f"{operation}-level-{level}-semantic-error",
                        passed=semantic_passed,
                        oracle=(
                            "All decoded slots are compared with the fixed "
                            "cleartext operation after untimed canonicalization."
                        ),
                        metric="max_abs_error",
                        observed=error["max_abs_error"],
                        comparison="<=",
                        limit=atol,
                        unit="absolute",
                        details={
                            "operation": operation,
                            "entry_level": level,
                            "rms_error": error["rms_error"],
                            "relative_l2_error": error["relative_l2_error"],
                            "nonfinite_count": error["nonfinite_count"],
                        },
                    ),
                )
            )
            del correctness_output, decoded, fixture
            gc.collect()

        del bundle
        gc.collect()
        synchronize(engine.device)

    effective_parameters = dict(parameters)
    benchmark_context = _benchmark_context(config, levels)
    result = BenchmarkResult(
        benchmark=BENCHMARK_NAME,
        profile=profile.name,
        workload_id=BENCHMARK_NAME,
        effective_parameters=effective_parameters,
        timed_boundary=_TIMED_BOUNDARY,
        metrics=metrics,
        correctness=checks,
        rows=rows,
        scalars={
            "measured_case_count": len(checks) // 2,
            "not_applicable_case_count": len(not_applicable_cases),
            "public_level_count": engine.public_level_count,
            "final_public_level": engine.final_public_level,
            "resident_all_evaluation_key_bytes": resident_key_bytes,
        },
        metadata={
            "workload_id": BENCHMARK_NAME,
            "benchmark_context": benchmark_context,
            "timed_boundary": _TIMED_BOUNDARY.to_dict(),
            "resolved_parameters": {
                **effective_parameters,
                "device": str(engine.device),
                "ntt_backend": engine.ntt_backend_name,
                "default_scale": engine.config.default_scale,
                "num_slots": engine.num_slots,
            },
            "operation_order": list(operations),
            "level_order": list(levels),
            "not_applicable_cases": not_applicable_cases,
            "key_bytes_by_role": key_bytes,
        },
        notes=[
            "Entry chain depth is the number of absent leading scale-Q rows and is not multiplicative history.",
            "Raw multiplication primitives remain legal at the terminal public level, but their outputs have no following public rescale transition.",
            "Inverse synchronized serial rates are derived from mean latency and are not saturation-throughput measurements.",
            "Correctness tolerances are inherited from existing public operation families and are never inferred from benchmark results.",
        ],
        evidence=evidence,
    )
    del keys, engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _profile_parameters(
    *, levels: Sequence[int], warmup: int, runs: int
) -> dict[str, Any]:
    config = _fixed_config()
    return {
        "preset": FIXED_PRESET.value,
        "logN": config.logN,
        "scale_bits": config.scale_bits,
        "num_scale_primes": config.num_scale_primes,
        "num_p_primes": config.num_p_primes,
        "ntt_backend": FIXED_NTT_BACKEND,
        "levels": list(levels),
        "operations": list(CANONICAL_OPERATIONS),
        "ciphertext_modulus_basis": "Q",
        "rotation_step": ROTATION_STEP,
        "rescale_rounding": RESCALE_ROUNDING,
        "seed": DEFAULT_SEED,
        "nonce": DEFAULT_NONCE,
        "warmup": warmup,
        "runs": runs,
        "include_raw_samples": True,
    }


DEFINITION = register_benchmark(
    BenchmarkDefinition(
        name=BENCHMARK_NAME,
        title="Depth-aware CKKS single operations",
        category="local device",
        description=(
            "Measures twelve functional public CKKS calls on one fixed "
            "cross-backend CKKS plan "
            "while sweeping active-Q entry levels. Setup and correctness "
            "canonicalization are excluded from synchronized call timing."
        ),
        profiles=(
            BenchmarkProfile(
                name="quick",
                description=(
                    "Representative entry levels 0, 3, 5, and 6 with short "
                    "timing loops."
                ),
                parameters=_profile_parameters(
                    levels=(0, 3, 5, 6), warmup=1, runs=3
                ),
            ),
            BenchmarkProfile(
                name="core",
                description=(
                    "All seven public entry levels with the complete twelve-call "
                    "measurement definition and raw timing samples."
                ),
                parameters=_profile_parameters(
                    levels=tuple(range(7)), warmup=1, runs=3
                ),
            ),
        ),
        runner=cast(Any, _run_depth_aware_operations),
        workload_id=BENCHMARK_NAME,
    )
)
