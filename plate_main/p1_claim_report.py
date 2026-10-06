#!/usr/bin/env python3
"""Audit whether the complete P1 evidence package is ready for a claim.

This report is deliberately conservative.  It never turns a pilot dataset or
an uncalibrated validation run into a universal FSDT/HF accuracy statement.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence
from p1_design import (PILOT_PARAMETER_NAMES, PROPOSAL_PARAMETER_NAMES,
                       file_sha256, read_design_table, write_json)
from p1_preflight import HF_EXTRACTIONS
import numpy as np

DESIGN_SCOPES = {
    PILOT_PARAMETER_NAMES: "pilot-five-variable",
    PROPOSAL_PARAMETER_NAMES: "full-proposal",
}


def _parameter_names(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        fields = tuple(next(csv.reader(stream)))
    names = tuple(field for field in fields if field != "run_id")
    if names not in DESIGN_SCOPES:
        raise ValueError(f"Unsupported design parameter names: {names}")
    return names
def _hf_uses_export(
    root: Path, run_ids: set[int], expected_extraction: str,
) -> bool:
    for run_id in run_ids:
        path = root / f"run_{run_id:04d}.npz"
        if not path.is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            metadata.get("run_id") != run_id
            or metadata.get("extraction") != expected_extraction
        ):
            return False
    return bool(run_ids)


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"run_id", "role"}.issubset(rows[0]):
        raise ValueError("HF plan must contain run_id and role")
    return rows


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--lf-root", type=Path, required=True)
    parser.add_argument("--hf-plan", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True,
                        help="HF bundles used by the declared plan")
    parser.add_argument("--targets", type=Path, required=True,
                        help="Correction revision directory containing manifest.json")
    parser.add_argument("--holdout", type=Path, required=True,
                        help="holdout_metrics.json from p1_holdout.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-lf", type=int, default=10_000)
    parser.add_argument("--max-lf", type=int, default=20_000)
    parser.add_argument("--min-hf", type=int, default=50)
    parser.add_argument("--hf-extraction", choices=HF_EXTRACTIONS,
                        default="comsol-interp",
                        help="HF provenance required in every bundle")
    parser.add_argument("--max-hf", type=int, default=100)
    args = parser.parse_args(argv)

    design_parameters = _parameter_names(args.design)
    design_rows = read_design_table(args.design, design_parameters)
    design_ids = {run for run, _ in design_rows}
    design_manifest_path = args.design.with_suffix(".manifest.json")
    design_manifest = (
        _read_json(design_manifest_path)
        if design_manifest_path.exists() else {}
    )
    lf_paths = sorted(args.lf_root.glob("run_*.npz"))
    lf_ids = {int(path.stem.rsplit("_", 1)[1]) for path in lf_paths}
    plan_rows = _read_plan(args.hf_plan)
    plan_ids = {int(row["run_id"]) for row in plan_rows}
    roles = {row["role"].strip().lower() for row in plan_rows}
    role_counts = {
        role: sum(row["role"].strip().lower() == role for row in plan_rows)
        for role in ("train", "calibration", "test")
    }
    target_manifest_path = args.targets / "manifest.json"
    target_training_path = args.targets / "training.npz"
    holdout_report = _read_json(args.holdout)
    target_manifest = (
        _read_json(target_manifest_path)
        if target_manifest_path.exists() else {}
    )
    target_parameters = ()
    if target_training_path.exists():
        with np.load(target_training_path, allow_pickle=False) as training:
            target_parameters = tuple(
                str(value) for value in training["parameter_names"]
            )

    expected_scope = DESIGN_SCOPES[design_parameters]
    checks = {
        "design_is_explicit_and_nonempty": bool(design_rows),
        "design_uses_supported_parameter_contract": (
            design_parameters in DESIGN_SCOPES
        ),
        "design_manifest_scope_matches_contract": (
            design_manifest.get("scope") == expected_scope
        ),
        "lf_count_in_declared_range": args.min_lf <= len(lf_ids) <= args.max_lf,
        "lf_ids_match_design": lf_ids.issubset(design_ids),
        "hf_count_in_declared_range": args.min_hf <= len(plan_ids) <= args.max_hf,
        "hf_uses_requested_extraction": _hf_uses_export(
            args.hf_root, plan_ids, args.hf_extraction
        ),
        "hf_roles_present": roles == {"train", "calibration", "test"},
        "hf_role_counts_nonzero": all(count > 0 for count in role_counts.values()),
        "hf_ids_match_design": plan_ids.issubset(design_ids),
        "target_revision_exists": (
            target_manifest_path.exists() and target_training_path.exists()
        ),
        "target_parameter_contract_matches_design": (
            len(target_parameters) == len(design_parameters)
            and set(target_parameters) == set(design_parameters)
        ),
        "target_run_filter_matches_plan": (
            set(target_manifest.get("requested_run_ids", [])) == plan_ids
        ),
        "strict_holdout_report_exists": bool(holdout_report),
        "strict_holdout_has_calibration": bool(holdout_report.get("calibration")),
        "strict_holdout_has_test_rows": bool(
            holdout_report.get("accepted_rows", {}).get("test", 0)
        ),
        "scope_is_declared_contract": target_manifest.get("scope") == expected_scope,
    }
    ready = all(checks.values())
    report = {
        "schema_version": 1,
        "artifact": "p1_claim_report",
        "status": "ready_for_scientific_review" if ready else "not_ready",
        "hf_extraction": args.hf_extraction,
        "parameter_scope": expected_scope,
        "checks": checks,
        "counts": {
            "design": len(design_rows),
            "lf": len(lf_ids),
            "hf": len(plan_ids),
            "hf_roles": role_counts,
            "accepted_targets": target_manifest.get("accepted"),
            "quarantined_targets": target_manifest.get("quarantined"),
        },
        "sha256": {
            "design": file_sha256(args.design),
            "design_manifest": (
                file_sha256(design_manifest_path)
                if design_manifest_path.exists() else None
            ),
            "hf_plan": file_sha256(args.hf_plan),
            "target_manifest": (
                file_sha256(target_manifest_path)
                if target_manifest_path.exists() else None
            ),
            "holdout": file_sha256(args.holdout),
        },
        "claim_guard": (
            "Even when ready, report held-out empirical distributions and uncertainty "
            "coverage; do not claim a universal four-percent error or "
            "simulator-independent accuracy."
        ),
        "next_failures": [name for name, passed in checks.items() if not passed],
    }
    write_json(args.output, report, overwrite=True)
    print(json.dumps(report, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
