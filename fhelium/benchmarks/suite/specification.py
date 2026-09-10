"""Workload and sampling definitions for the local-device benchmark suite.

The suite covers the product of sampled input depths, leading batch sizes,
and task sizes. The same cells and arithmetic plans apply on CPU and CUDA.
A run owns the complete cell inventory, including cells that cannot execute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from fhelium.config import CkksConfig, Preset

from .polynomial import METHODS, POLYNOMIAL, definition as polynomial_definition

CONFIGURATIONS = (
    Preset.slots32768_scale50_depth27_int64,
)
BATCHES = (1, 2, 4, 8, 16)
MATRIX_SIZES = (16, 32, 64)
OPERATIONS = (
    "rotate", "rotate_many", "rotate_many_independent",
    "relinearize", "rescale_ct2", "multiply_plaintext",
    "multiply_ciphertexts", "square", "add", "sum_plaintext_products",
    "multiply_coefficient", "advance_depth",
)
MATRIX_WORKLOADS = ("matrix_ptct", "matrix_ctct")
NTT_PLANS = ("radix2_indexed", "radix2_compact_group8_smem8")
PRIMITIVE_WORKLOADS = (
    "ntt_forward", "ntt_inverse", "key_switch", "rns_add", "rns_multiply",
)
WARMUPS = 5
SAMPLES = 20
ABSOLUTE_ERROR_LIMIT = 1e-5
SEED = 20260907


@dataclass(frozen=True)
class SuiteCell:
    """One mathematical task and execution plan at a concrete input state."""

    config: str
    workload: str
    depth: int
    batch: int
    size: int | None
    plan: str

    @property
    def id(self) -> str:
        return f"{self.config}/{self.workload}/d{self.depth}/b{self.batch}/n{self.size or 0}/{self.plan}"


def configurations() -> dict[str, CkksConfig]:
    return {preset.value: CkksConfig.parse(preset) for preset in CONFIGURATIONS}


def entry_depths(config: CkksConfig) -> tuple[int, ...]:
    """Sample every fourth nonterminal input basis."""
    return tuple(range(0, config.max_depth, 4))


def cells() -> tuple[SuiteCell, ...]:
    inventory = []
    for name, config in configurations().items():
        # Products and rescaling use the nonterminal input bases.
        for op in OPERATIONS:
            for depth in entry_depths(config):
                for batch in BATCHES:
                    inventory.append(SuiteCell(
                        name, op, depth, batch, None,
                        "shared" if op == "rotate_many" else "direct",
                    ))
        for workload in MATRIX_WORKLOADS:
            for size in MATRIX_SIZES:
                for depth in entry_depths(config):
                    for batch in BATCHES:
                        for plan in ("independent", "shared"):
                            inventory.append(SuiteCell(
                                name, workload, depth, batch, size, plan,
                            ))
        for plan, evaluator in METHODS.items():
            required = evaluator.required_depths(POLYNOMIAL)
            for depth in entry_depths(config):
                if depth + required > config.max_depth:
                    continue
                for batch in BATCHES:
                    inventory.append(SuiteCell(
                        name, "polynomial", depth, batch, None, plan,
                    ))
        for workload in PRIMITIVE_WORKLOADS:
            plans = NTT_PLANS if workload.startswith("ntt_") else ("direct",)
            for plan in plans:
                for depth in entry_depths(config):
                    for batch in BATCHES:
                        inventory.append(SuiteCell(
                            name, workload, depth, batch, None, plan,
                        ))
    return tuple(inventory)


def specification() -> dict[str, object]:
    """Return the mathematical and measurement identity hashed by every run."""
    return {
        "suite": "fhelium-local",
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "absolute_error_limit": ABSOLUTE_ERROR_LIMIT,
        "seed": SEED,
        "timing": "synchronized-wall-ms",
        "execution": "eager",
        "configurations": {
            name: config.dumps() for name, config in configurations().items()
        },
        "cells": [{"id": cell.id, **asdict(cell)} for cell in cells()],
        "polynomial": polynomial_definition(),
        "matrix": {
            "equation": "C[b] = A[b] @ B[b]",
            "packing": "two periodic copies per column; packed columns grouped over ciphertext batch axes",
            "baby_step": "largest power of two not exceeding sqrt(matrix_size)",
            "diagonals": "retained NTT plaintext or ciphertext; per-group rescale; ciphertext products relinearized once per group",
        },
        "excludes": [
            "context and key construction",
            "encoding and encryption",
            "diagonal preparation",
            "decryption",
            "input transfers",
        ],
        "operations": {
            "rotation_step": 1,
            "rotation_many_steps": list(range(1, 8)),
            "input": "NTT/Montgomery except both seven-rotation plans, which start in coefficient/standard form",
            "rescale_ct2": "prepared relinearized square, CT2 NTT input; multiplication and relinearization excluded",
            "multiply_ciphertexts": "two independently encrypted messages; CT2 x CT2 -> CT3",
            "square": "same encrypted operand on both sides; CT2 x CT2 -> CT3",
            "sum_plaintext_products": "eight prepared CT2/PT pairs summed in NTT; no rescale",
            "multiply_coefficient": "encoded scalar -0.3 times CT2, followed by one rescale",
            "advance_depth": "multiply by encoded one and rescale to the next arithmetic target scale",
            "add": "two CT2 NTT inputs at the same depth and scale",
        },
        "primitives": {
            "layout": "one polynomial per task; [batch, active Q prime, coefficient or NTT index]",
            "timing": "functional complete operation including output allocation; inverse NTT includes input-to-output copy",
            "ntt_forward": "coefficient/standard -> bit-reversed NTT/Montgomery; complete normalized inverse measured separately",
            "ntt_inverse": "bit-reversed NTT/Montgomery -> coefficient/standard",
            "ntt_plans": {
                "radix2_indexed": "sixteen sequential radix-2 stages at logN=16; CUDA launch per stage; CPU stage barriers",
                "radix2_compact_group8_smem8": "compact twiddles; three radix-2 stages per global kernel and compiled shared-memory tail; CUDA only",
            },
            "ntt_oracle": "a_b(X)=(b+1)+X-X^2; independent modular evaluation at psi^(2*bit_reverse(k)+1); inverse input is this analytic transform, not a forward-under-test result",
            "rns_add": "standard residues a=q-k, b=3k+1; expected (a+b) mod q",
            "rns_multiply": "Montgomery stored residues a=q-k, b=(tR) mod q; expected (-kt) mod q; k varies over batch/coefficient, t=1+(k mod 17)",
            "rns_oracle": "integer formulas with least-nonnegative residue equality over every output coefficient",
            "key_switch": "independent source/destination secrets; complete CT2 coefficient/standard Q -> CT2 NTT/Montgomery Q; hybrid ModUp, NTT/key accumulation and ModDown included; decode under destination secret",
        },
    }


def specification_hash() -> str:
    encoded = json.dumps(
        specification(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
