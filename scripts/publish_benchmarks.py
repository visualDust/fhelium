#!/usr/bin/env python3
"""Project completed suite reports into the documentation's run catalog.

Raw samples and failure diagnostics stay in the caller's evidence directory.
The catalog contains qualified measurement summaries and their task identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def project(report: dict, selection: dict | None = None, *, partial: bool = False) -> dict:
    specification = report["specification"]
    digest = hashlib.sha256(
        json.dumps(
            specification, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if report["specification_hash"] != digest:
        raise ValueError("Suite specification hash does not match the report")
    expected = {cell["id"]: cell for cell in specification["cells"]}
    actual = {cell["id"]: cell for cell in report["results"]}
    if (
        report["coverage"] != ("partial" if partial else "full")
        or report["status"] != "completed"
        or len(actual) != len(report["results"])
        or (not actual.keys() <= expected.keys() if partial else actual.keys() != expected.keys())
    ):
        raise ValueError(
            "Only complete suite inventories can enter the catalog"
        )
    rows = []
    for identifier, result in actual.items():
        if any(
            result[key] != value for key, value in expected[identifier].items()
        ):
            raise ValueError(
                f"Result identity differs from suite cell {identifier}"
            )
        row = {**expected[identifier], "status": result["status"]}
        if result["status"] == "passed":
            values = result["measurement"]
            if (
                not math.isfinite(values["max_absolute_error"])
                or values["max_absolute_error"]
                > specification["absolute_error_limit"]
            ):
                raise ValueError(f"Unqualified measurement {identifier}")
            if len(values["samples_ms"]) != specification["samples"]:
                raise ValueError(f"Incomplete sampling for {identifier}")
            if result["workload"].startswith(("ntt_", "rns_")) and (
                values.get("qualification") != "exact-residues"
                or values["max_absolute_error"] != 0
            ):
                raise ValueError(f"Unqualified residue measurement {identifier}")
            timing = [values[k] for k in ("q25_ms", "median_ms", "q75_ms")]
            if any(
                not math.isfinite(v) or v <= 0 for v in timing
            ) or timing != sorted(timing):
                raise ValueError(f"Invalid latency summary for {identifier}")
            row["measurement"] = {
                k: v for k, v in values.items() if k != "samples_ms"
            }
        elif result["status"] == "unsupported":
            row["reason"] = result["reason"]
        elif result["status"] != "capacity":
            raise ValueError(f"Unfinished or failed cell {identifier}")
        rows.append(row)
    comparison = specification if selection is None else selection
    if partial:
        comparison = {**comparison, "cells": [cell for cell in comparison["cells"] if cell["id"] in actual]}
    selected_cells = {cell["id"]: cell for cell in comparison["cells"]}
    if not selected_cells.keys() <= expected.keys():
        raise ValueError("The run does not cover the selected comparison cells")
    for key, value in comparison.items():
        if key not in ("cells", "configurations") and specification.get(key) != value:
            raise ValueError(f"Measurement definition differs: {key}")
    for identifier, cell in selected_cells.items():
        if expected[identifier] != cell:
            raise ValueError(f"Comparison task differs: {identifier}")
    for name, config in comparison["configurations"].items():
        if specification["configurations"].get(name) != config:
            raise ValueError(f"Comparison configuration differs: {name}")
    rows = [row for row in rows if row["id"] in selected_cells]
    comparison_digest = hashlib.sha256(json.dumps(
        comparison, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    configs = []
    for name, config in comparison["configurations"].items():
        groups = config["q_depth_groups"]
        states = []
        for depth in range(len(groups)):
            active = [int(prime) for group in groups[depth:] for prime in group]
            q = math.prod(active)
            states.append(
                {
                    "depth": depth,
                    "q_rows": len(active),
                    "q_bits": q.bit_length(),
                    "qp_bits": (
                        q * math.prod(int(p) for p in config["p_moduli"])
                    ).bit_length(),
                }
            )
        configs.append(
            {
                "id": name,
                "ring_degree": 2 ** config["logN"],
                "max_depth": len(groups) - 1,
                "default_scale": config["default_scale"],
                "q_depth_groups": [[str(p) for p in group] for group in groups],
                "p_moduli": [str(p) for p in config["p_moduli"]],
                "states": states,
            }
        )

    return {
        "id": report["id"],
        "coverage": report["coverage"],
        "started_at": report["started_at"],
        "finished_at": report["finished_at"],
        "specification_hash": digest,
        "comparison_hash": comparison_digest,
        "recorded_case_count": len(actual),
        "source": report["source"],
        "platform": report["platform"],
        "configurations": configs,
        "sampling": {
            k: specification[k]
            for k in (
                "warmups",
                "samples",
                "absolute_error_limit",
                "timing",
                "execution",
            )
        },
        "results": rows,
        **({"polynomial": specification["polynomial"]} if "polynomial" in specification else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pages", type=Path, help="Write each run page beside the catalog projection.")
    parser.add_argument("--selection", type=Path, help="Comparison specification covered by every input run.")
    parser.add_argument("--supplement", nargs=2, action="append", default=[], metavar=("RUN_ID", "REPORT"), help="Attach separately dated supplemental measurements to an existing run page.")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text()) if args.selection else None
    runs = [project(json.loads(path.read_text()), selection) for path in args.reports]
    if len({run["comparison_hash"] for run in runs}) != 1:
        raise ValueError("Compared runs must cover the same measurement definition")
    if len({run["id"] for run in runs}) != len(runs):
        raise ValueError("Duplicate run IDs")
    for parent_id, path in args.supplement:
        parent = next((run for run in runs if run["id"] == parent_id), None)
        if parent is None:
            raise ValueError(f"Unknown supplement parent {parent_id}")
        supplement = project(json.loads(Path(path).read_text()), partial=True)
        if supplement["platform"]["name"] != parent["platform"]["name"]:
            raise ValueError("Supplement hardware differs from its parent")
        by_config = {config["id"]: config for config in parent["configurations"]}
        if any(by_config.get(config["id"]) != config for config in supplement["configurations"]):
            raise ValueError("Supplement mathematical configuration differs")
        parent.setdefault("additional_runs", []).append(supplement)
    if args.pages is not None:
        args.pages.mkdir(parents=True, exist_ok=True)
        for run in runs:
            identifier = run["id"]
            if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", identifier):
                raise ValueError("Invalid run ID for a page filename")
            title = json.dumps(run["platform"]["name"] + " — Benchmark run")
            page = (
                f"---\ntitle: {title}\nlayout: page\npageClass: benchmark-portal-page\n"
                "sidebar: false\naside: false\n---\n\n"
                "<script setup>\n"
                "import WorkloadBenchmarks from '../../.vitepress/theme/benchmarks/workloads/WorkloadBenchmarks.vue'\n"
                "</script>\n\n"
                f'<WorkloadBenchmarks record-id="{identifier}" />\n'
            )
            (args.pages / f"{identifier}.md").write_text(page)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(runs, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Projected {len(runs)} complete suite runs to {args.output}")


if __name__ == "__main__":
    main()
