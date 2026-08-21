"""Deterministic resolution and hashing of the Benchmark v1 manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fhelium.benchmarks.model import BenchmarkDefinition, BenchmarkProfile

from ._validation import freeze_json, normalize_json, strict_object
from .model import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkSpecification,
    CaseStatus,
)


class DefinitionRegistry(Protocol):
    """Minimum fixed-definition interface required by v1 resolution."""

    def get(self, name: str) -> BenchmarkDefinition: ...


@dataclass(frozen=True)
class ResolvedCase:
    """A Benchmark v1 case bound to its definition and parameters."""

    case: BenchmarkCase
    definition: BenchmarkDefinition
    profile: BenchmarkProfile

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.profile.parameters

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "id": self.case.id,
            "title": self.case.title,
            "category": self.case.category,
            "benchmark": self.case.benchmark,
            "workload_id": self.case.workload_id,
            "profile": self.profile.name,
            "parameters": normalize_json(
                self.profile.parameters,
                path=f"cases.{self.case.id}.parameters",
            ),
            "requirements": normalize_json(self.case.requirements),
            "comparison": normalize_json(self.case.comparison),
            "unavailable_reason": self.case.unavailable_reason,
        }


@dataclass(frozen=True)
class ResolvedManifest:
    """Canonical Benchmark v1 description and its SHA-256 identity."""

    specification: BenchmarkSpecification
    cases: tuple[ResolvedCase, ...]
    sha256: str

    def manifest_dict(self) -> dict[str, Any]:
        """Return the payload covered by :attr:`sha256`."""

        return {
            "benchmark_version": self.specification.benchmark_version,
            "title": self.specification.title,
            "description": self.specification.description,
            "cases": [case.to_manifest_dict() for case in self.cases],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.manifest_dict()
        payload["manifest_sha256"] = self.sha256
        return payload

    def canonical_bytes(self) -> bytes:
        """Serialize the covered manifest in canonical UTF-8 JSON."""

        return _canonical_bytes(self.manifest_dict())


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = normalize_json(payload, path="manifest")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def resolve_benchmark(
    specification: BenchmarkSpecification,
    definitions: DefinitionRegistry,
) -> ResolvedManifest:
    """Bind the fixed Benchmark v1 cases to their leaf definitions."""

    if not isinstance(specification, BenchmarkSpecification):
        raise TypeError("specification must be a BenchmarkSpecification")

    resolved: list[ResolvedCase] = []
    for case in specification.cases:
        definition = definitions.get(case.benchmark)
        if not isinstance(definition, BenchmarkDefinition):
            raise TypeError(
                f"Definition {case.benchmark!r} is not a BenchmarkDefinition"
            )
        if definition.name != case.benchmark:
            raise ValueError(
                f"case {case.id!r} benchmark does not match definition name"
            )
        if definition.workload_id != case.workload_id:
            raise ValueError(
                f"definition {definition.name!r} workload_id "
                f"{definition.workload_id!r} does not match case {case.id!r} "
                f"workload_id {case.workload_id!r}"
            )
        selected = definition.profile(case.profile)
        effective = strict_object(
            selected.parameters,
            f"leaf_profiles.{definition.name}.{selected.name}.parameters",
        )
        effective.update(case.parameters)
        effective = strict_object(
            effective, f"cases.{case.id}.effective_parameters"
        )
        effective_profile = selected.with_overrides(effective)
        object.__setattr__(
            effective_profile,
            "parameters",
            freeze_json(
                effective_profile.parameters,
                path=f"cases.{case.id}.effective_parameters",
            ),
        )
        resolved.append(
            ResolvedCase(
                case=case,
                definition=definition,
                profile=effective_profile,
            )
        )

    case_tuple = tuple(resolved)
    body = {
        "benchmark_version": specification.benchmark_version,
        "title": specification.title,
        "description": specification.description,
        "cases": [case.to_manifest_dict() for case in case_tuple],
    }
    digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return ResolvedManifest(
        specification=specification,
        cases=case_tuple,
        sha256=digest,
    )


def validate_report_specification(report: BenchmarkReport) -> None:
    """Require the fixed manifest and five case identities of v1."""

    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a BenchmarkReport")
    from .cases import (
        BENCHMARK_MANIFEST_SHA256,
        FIXED_DEFINITIONS,
        SPECIFICATION,
    )

    expected = resolve_benchmark(SPECIFICATION, FIXED_DEFINITIONS)
    if expected.sha256 != BENCHMARK_MANIFEST_SHA256:
        raise RuntimeError(
            "package-owned Benchmark v1 manifest differs from its pinned "
            "identity"
        )
    specification_path = Path(__file__).with_name("specification.json")
    with specification_path.open("r", encoding="utf-8") as stream:
        packaged_specification = json.load(stream)
    if packaged_specification != expected.to_dict():
        raise RuntimeError(
            "Benchmark v1 runtime specification differs from specification.json"
        )
    if report.manifest_sha256 != BENCHMARK_MANIFEST_SHA256:
        raise ValueError(
            "report manifest_sha256 differs from the fixed Benchmark v1 "
            "specification"
        )
    if len(report.cases) != len(expected.cases):
        raise ValueError("report must contain all five Benchmark v1 cases")
    for index, (record, resolved) in enumerate(
        zip(report.cases, expected.cases, strict=True)
    ):
        case = resolved.case
        exact_scalars = {
            "id": case.id,
            "title": case.title,
            "category": case.category,
            "benchmark": case.benchmark,
            "workload_id": case.workload_id,
            "profile": resolved.profile.name,
        }
        for field_name, expected_scalar in exact_scalars.items():
            if getattr(record, field_name) != expected_scalar:
                raise ValueError(
                    f"report.cases[{index}].{field_name} differs from the fixed "
                    "Benchmark v1 specification"
                )
        exact_objects = {
            "parameters": resolved.parameters,
            "requirements": case.requirements,
            "comparison": case.comparison,
        }
        for field_name, expected_object in exact_objects.items():
            if normalize_json(getattr(record, field_name)) != normalize_json(
                expected_object
            ):
                raise ValueError(
                    f"report.cases[{index}].{field_name} differs from the fixed "
                    "Benchmark v1 specification"
                )
        if record.result is not None:
            result = record.result
            result_identity = {
                "benchmark": resolved.definition.name,
                "workload_id": case.workload_id,
                "profile": resolved.profile.name,
            }
            for field_name, expected_scalar in result_identity.items():
                if getattr(result, field_name) != expected_scalar:
                    raise ValueError(
                        f"report.cases[{index}].result.{field_name} differs "
                        "from the fixed Benchmark v1 specification"
                    )
            if normalize_json(result.effective_parameters) != normalize_json(
                resolved.parameters
            ):
                raise ValueError(
                    f"report.cases[{index}].result.effective_parameters "
                    "differs from the fixed Benchmark v1 specification"
                )
        if record.status is CaseStatus.MEASURED:
            assert record.result is not None
            if not record.result.metrics:
                raise ValueError(
                    f"report.cases[{index}] measured result has no metrics"
                )
            if not record.result.correctness_passed:
                raise ValueError(
                    f"report.cases[{index}] measured result did not satisfy "
                    "non-empty validation criteria"
                )
