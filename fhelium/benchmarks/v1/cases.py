"""The five fixed cross-backend case positions of FHElium Benchmark v1."""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

from fhelium.benchmarks.model import BenchmarkDefinition, BenchmarkProfile

from . import matrix, ntt, operations, polynomial
from ._validation import freeze_json
from .model import BENCHMARK_VERSION, BenchmarkCase, BenchmarkSpecification

# Regenerated from the resolved specification after every intentional v1 change.
BENCHMARK_MANIFEST_SHA256 = (
    "5b9ce22abf59cb5b37dcc59062e2f856df40584470082fd0a4c6f08ee9b81c4b"
)


def _case(
    *,
    id: str,
    title: str,
    category: str,
    benchmark: str,
    comparison: dict[str, object],
) -> BenchmarkCase:
    case = BenchmarkCase(
        id=id,
        title=title,
        category=category,
        benchmark=benchmark,
        workload_id=benchmark,
        profile="core",
        parameters={},
        requirements={"native_backends_any": ["cpu", "cuda"]},
        comparison=comparison,
    )
    for name in ("parameters", "requirements", "comparison"):
        object.__setattr__(
            case,
            name,
            freeze_json(getattr(case, name), path=f"cases.{id}.{name}"),
        )
    return case


CASES = (
    _case(
        id="ckks-depth-aware-single-operations",
        title="Depth-aware CKKS single operations",
        category="Single operations",
        benchmark="ckks-depth-aware-single-operations",
        comparison={
            "axis": ["operation", "entry_level"],
            "portal_highlight": {
                "label": "CT × CT multiply · level 0",
                "operation": "multiply",
                "entry_level": 0,
                "metric": "depth-aware-ckks-operation-latency",
            },
        },
    ),
    _case(
        id="indexed-ntt-operations",
        title="Indexed radix-2 NTT operations",
        category="NTT operations",
        benchmark="indexed-ntt-operations",
        comparison={
            "axis": ["entry_level", "modulus_basis", "operation"],
            "portal_highlight": {
                "label": "Forward NTT · level 0 Q",
                "metric": "indexed-ntt-latency",
                "entry_level": 0,
                "modulus_basis": "Q",
                "operation": "forward_ntt",
            },
        },
    ),
    _case(
        id="dense-matmul-ptct",
        title="Plaintext × ciphertext dense matrix multiplication",
        category="Matrix multiplication",
        benchmark="dense-matrix-multiplication-ptct",
        comparison={
            "portal_highlight": {
                "label": "PT × CT 16×16",
                "metric": "dense-matrix-multiplication-latency",
                "phase": "end-to-end",
            }
        },
    ),
    _case(
        id="dense-matmul-ctct",
        title="Ciphertext × ciphertext dense matrix multiplication",
        category="Matrix multiplication",
        benchmark="dense-matrix-multiplication-ctct",
        comparison={
            "portal_highlight": {
                "label": "CT × CT 16×16",
                "metric": "dense-matrix-multiplication-latency",
                "phase": "end-to-end",
            }
        },
    ),
    _case(
        id="polynomial-method-matrix",
        title="Affine and degree-four polynomial methods",
        category="Polynomial evaluation",
        benchmark="polynomial-evaluation",
        comparison={
            "axis": ["case_id", "method_id"],
            "portal_highlight": {
                "label": "Degree 4 · PS k=2",
                "case_id": "dense-power-d4",
                "method_id": "dense-power-d4-paterson-stockmeyer-k2",
                "metric": "polynomial-evaluation-latency",
            },
        },
    ),
)

SPECIFICATION = BenchmarkSpecification(
    benchmark_version=BENCHMARK_VERSION,
    title="FHElium Benchmark v1",
    description=(
        "Measures one immutable five-case CKKS specification on either CPU or CUDA: depth-aware public operations, indexed radix-2 NTT operations, fixed 16x16 plaintext/ciphertext and ciphertext/ciphertext matrix products, and affine/degree-four polynomial methods. Every case uses the same parameters on both execution backends; no composite score is defined."
    ),
    cases=CASES,
)


def _fixed_definition(definition: BenchmarkDefinition) -> BenchmarkDefinition:
    profiles: list[BenchmarkProfile] = []
    for profile in definition.profiles:
        fixed_profile = BenchmarkProfile(
            profile.name, profile.description, profile.parameters
        )
        object.__setattr__(
            fixed_profile,
            "parameters",
            freeze_json(
                fixed_profile.parameters,
                path=f"definitions.{definition.name}.{profile.name}",
            ),
        )
        profiles.append(fixed_profile)
    fixed = BenchmarkDefinition(
        name=definition.name,
        title=definition.title,
        description=definition.description,
        profiles=tuple(profiles),
        runner=definition.runner,
        workload_id=definition.workload_id,
        category=definition.category,
        requirements=definition.requirements,
    )
    object.__setattr__(
        fixed,
        "requirements",
        freeze_json(
            fixed.requirements,
            path=f"definitions.{definition.name}.requirements",
        ),
    )
    return fixed


_DEFINITION_MAP = {
    definition.name: _fixed_definition(definition)
    for definition in (
        cast(BenchmarkDefinition, operations.DEFINITION),
        cast(BenchmarkDefinition, ntt.DEFINITION),
        cast(BenchmarkDefinition, matrix.PTCT_DEFINITION),
        cast(BenchmarkDefinition, matrix.CTCT_DEFINITION),
        cast(BenchmarkDefinition, polynomial.DEFINITION),
    )
}
DEFINITIONS = MappingProxyType(_DEFINITION_MAP)


class FixedDefinitions:
    def get(self, name: str) -> BenchmarkDefinition:
        return DEFINITIONS[name]


FIXED_DEFINITIONS = FixedDefinitions()

__all__ = [
    "BENCHMARK_MANIFEST_SHA256",
    "BENCHMARK_VERSION",
    "CASES",
    "DEFINITIONS",
    "FIXED_DEFINITIONS",
    "SPECIFICATION",
]
