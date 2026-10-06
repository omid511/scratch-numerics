#!/usr/bin/env python3
"""Validate the versioned P1 artifacts before expensive or scientific use.

The validator is intentionally strict for the declared P1 claim contract:
the supported parameter schema, LF/HF counts, role split, target parameter
names, and selected HF provenance must all agree. It does not run either simulator.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from p1_design import (
    PILOT_PARAMETER_NAMES,
    PROPOSAL_PARAMETER_NAMES,
    file_sha256,
    read_design_table,
    write_json,
)
from p1_pairing import load_run

HF_EXTRACTIONS = ("comsol-interp", "abaqus-export")
DESIGN_CONTRACTS = {
    PILOT_PARAMETER_NAMES: "pilot-five-variable",
    PROPOSAL_PARAMETER_NAMES: "full-proposal",
}


def _parameter_names(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        fields = tuple(next(csv.reader(stream)))
    names = tuple(field for field in fields if field not in ("run_id", "run"))
    if names not in DESIGN_CONTRACTS:
        raise ValueError(f"Unsupported P1 design parameter contract: {names}")
    return names


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"run_id", "role"}.issubset(rows[0]):
        raise ValueError("HF plan must contain run_id and role")
    seen: set[int] = set()
    for row in rows:
        run_id = int(row["run_id"])
        role = row["role"].strip().lower()
        if run_id < 1 or run_id in seen or role not in {"train", "calibration", "test"}:
            raise ValueError(f"Invalid or duplicate HF plan row: {row}")
        seen.add(run_id)
    return rows


def _bundle_ids(root: Path) -> set[int]:
    result = set()
    for path in root.glob("run_*.npz"):
        try:
            run_id = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid run bundle name: {path}") from error
        if run_id < 1 or run_id in result:
            raise ValueError(f"Duplicate or nonpositive run bundle ID: {path}")
        result.add(run_id)
    return result


def _same_parameters(actual: dict[str, object], expected: dict[str, float]) -> bool:
    return set(actual) == set(expected) and all(
        np.isclose(float(actual[name]), expected[name], rtol=0, atol=1e-12)
        for name in expected
    )


def _validate_lf_bundle(
    path: Path, run_id: int, parameters: dict[str, float], modes: int | None = None,
) -> None:
    bundle = load_run(path, "lf")
    metadata = bundle["meta"]
    if metadata["run_id"] != run_id or not _same_parameters(metadata["parameters"], parameters):
        raise ValueError("run ID or parameter provenance mismatch")
    configured_modes = int(metadata.get("modes", len(bundle["f"])))
    if modes is not None and configured_modes != modes:
        raise ValueError(f"expected {modes} modes, received {configured_modes}")
    if len(bundle["f"]) != configured_modes:
        raise ValueError("frequency count disagrees with LF metadata")
def _validate_hf_bundle(
    path: Path, run_id: int, parameters: dict[str, float], expected_extraction: str,
) -> None:
    bundle = load_run(path, "hf")
    metadata = bundle["meta"]
    if metadata.get("extraction") != expected_extraction:
        raise ValueError(
            f"HF extraction is {metadata.get('extraction')!r}; "
            f"expected {expected_extraction!r}"
        )
    if metadata["run_id"] != run_id or not _same_parameters(metadata["parameters"], parameters):
        raise ValueError("run ID or parameter provenance mismatch")
    if bundle.get("transverse_fraction") is None:
        raise ValueError("HF transverse-energy provenance is missing")


def _design_checks(
    design: Path, manifest: Path | None,
) -> tuple[list[tuple[int, dict[str, float]]], dict[str, object]]:
    names = _parameter_names(design)
    rows = read_design_table(design, names)
    manifest_path = manifest or design.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"Design manifest not found: {manifest_path}")
    metadata = _read_json(manifest_path)
    expected_scope = DESIGN_CONTRACTS[names]
    if metadata.get("scope") != expected_scope:
        raise ValueError(
            f"Design manifest scope is {metadata.get('scope')!r}; "
            f"expected {expected_scope!r}"
        )
    if tuple(metadata.get("parameter_names", ())) != names:
        raise ValueError("Design manifest parameter_names do not match the design")
    if metadata.get("count") != len(rows):
        raise ValueError("Design manifest count does not match design rows")
    if metadata.get("design_sha256") != file_sha256(design):
        raise ValueError("Design manifest hash does not match design table")
    return rows, metadata


def _run_checks(
    checks: dict[str, bool], errors: dict[str, str], name: str, function: Callable[[], None]
) -> None:
    try:
        function()
    except Exception as error:  # noqa: BLE001 - report every failed gate together
        checks[name] = False
        errors[name] = str(error)
    else:
        checks[name] = True


def validate(
    stage: str, design: Path, design_manifest: Path | None, lf_root: Path | None,
    hf_plan: Path | None, hf_root: Path | None, targets: Path | None,
    holdout: Path | None, min_lf: int, max_lf: int, min_hf: int, max_hf: int,
    hf_extraction: str = "comsol-interp",
) -> dict[str, object]:
    if hf_extraction not in HF_EXTRACTIONS:
        raise ValueError(f"Unsupported HF extraction: {hf_extraction!r}")
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}
    rows: list[tuple[int, dict[str, float]]] = []
    design_metadata: dict[str, object] = {}

    def check_design() -> None:
        nonlocal rows, design_metadata
        rows, design_metadata = _design_checks(design, design_manifest)

    _run_checks(checks, errors, "design_contract", check_design)
    design_by_id = dict(rows)

    if stage in {"lf", "hf", "final"}:
        if lf_root is None:
            checks["lf_root_present"] = False
            errors["lf_root_present"] = "--lf-root is required for this stage"
        else:
            def check_lf() -> None:
                ids = _bundle_ids(lf_root)
                if not min_lf <= len(ids) <= max_lf:
                    raise ValueError(f"LF count {len(ids)} is outside {min_lf}..{max_lf}")
                design_ids = set(design_by_id)
                if ids != design_ids:
                    raise ValueError(
                        f"LF IDs differ from design: missing={sorted(design_ids - ids)[:10]}, "
                        f"extra={sorted(ids - design_ids)[:10]}"
                    )
                for run_id in sorted(ids):
                    _validate_lf_bundle(lf_root / f"run_{run_id:04d}.npz", run_id, design_by_id[run_id])
            _run_checks(checks, errors, "lf_bundles", check_lf)

    plan_rows: list[dict[str, str]] = []
    plan_ids: set[int] = set()
    if stage in {"hf", "final"}:
        if hf_plan is None:
            checks["hf_plan_present"] = False
            errors["hf_plan_present"] = "--hf-plan is required for this stage"
        else:
            def check_plan() -> None:
                nonlocal plan_rows, plan_ids
                plan_rows = _read_plan(hf_plan)
                plan_ids = {int(row["run_id"]) for row in plan_rows}
                if not min_hf <= len(plan_ids) <= max_hf:
                    raise ValueError(f"HF count {len(plan_ids)} is outside {min_hf}..{max_hf}")
                if not plan_ids.issubset(set(design_by_id)):
                    raise ValueError("HF plan contains a run absent from the design")
                roles = {row["role"].strip().lower() for row in plan_rows}
                if roles != {"train", "calibration", "test"}:
                    raise ValueError(f"HF roles are {sorted(roles)}, not train/calibration/test")
            _run_checks(checks, errors, "hf_plan", check_plan)

        if hf_root is None:
            checks["hf_root_present"] = False
            errors["hf_root_present"] = "--hf-root is required for this stage"
        else:
            def check_hf() -> None:
                if not plan_ids:
                    raise ValueError("HF plan must pass before HF bundles can be checked")
                for run_id in sorted(plan_ids):
                    _validate_hf_bundle(
                        hf_root / f"run_{run_id:04d}.npz",
                        run_id,
                        design_by_id[run_id],
                        hf_extraction,
                    )
            _run_checks(checks, errors, "hf_bundles", check_hf)

    if stage == "final":
        if targets is None or holdout is None:
            checks["final_artifacts_present"] = False
            errors["final_artifacts_present"] = "--targets and --holdout are required for final stage"
        else:
            def check_final() -> None:
                manifest_path = targets / "manifest.json"
                training_path = targets / "training.npz"
                if not manifest_path.is_file() or not training_path.is_file():
                    raise ValueError("Target revision is missing manifest.json or training.npz")
                target_manifest = _read_json(manifest_path)
                expected_scope = design_metadata.get("scope")
                expected_names = tuple(design_metadata.get("parameter_names", ()))
                if target_manifest.get("scope") != expected_scope:
                    raise ValueError("Target revision scope does not match the design")
                if set(target_manifest.get("requested_run_ids", [])) != plan_ids:
                    raise ValueError("Target run filter does not match the HF plan")
                with np.load(training_path, allow_pickle=False) as archive:
                    names = tuple(str(value) for value in archive["parameter_names"])
                if len(names) != len(expected_names) or set(names) != set(expected_names):
                    raise ValueError("Target parameter_names do not match the design")
                report = _read_json(holdout)
                if not report.get("calibration") or not report.get("accepted_rows", {}).get("test", 0):
                    raise ValueError("Holdout report lacks calibration or accepted test rows")
            _run_checks(checks, errors, "final_artifacts", check_final)

    lf_count = None
    if lf_root and lf_root.is_dir():
        try:
            lf_count = len(_bundle_ids(lf_root))
        except ValueError:
            lf_count = None
    return {
        "schema_version": 1,
        "artifact": "p1_preflight",
        "stage": stage,
        "hf_extraction": hf_extraction,
        "status": "pass" if checks and all(checks.values()) else "fail",
        "checks": checks,
        "errors": errors,
        "counts": {
            "design": len(rows),
            "lf": lf_count,
            "hf": len(plan_ids) if plan_ids else None,
            "hf_roles": {
                role: sum(row.get("role", "").strip().lower() == role for row in plan_rows)
                for role in ("train", "calibration", "test")
            },
        },
        "design_metadata": design_metadata,
        "claim_guard": "A pass authorizes the next pipeline stage; it is not a scientific accuracy claim.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("design", "lf", "hf", "final"), required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--design-manifest", type=Path)
    parser.add_argument("--lf-root", type=Path)
    parser.add_argument("--hf-plan", type=Path)
    parser.add_argument("--hf-root", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-lf", type=int, default=10_000)
    parser.add_argument("--max-lf", type=int, default=20_000)
    parser.add_argument("--min-hf", type=int, default=50)
    parser.add_argument("--hf-extraction", choices=HF_EXTRACTIONS,
                        default="comsol-interp",
                        help="HF provenance required in every bundle")
    parser.add_argument("--max-hf", type=int, default=100)
    args = parser.parse_args(argv)
    if min(args.min_lf, args.max_lf, args.min_hf, args.max_hf) < 1:
        parser.error("count bounds must be positive")
    report = validate(
        args.stage, args.design, args.design_manifest, args.lf_root,
        args.hf_plan, args.hf_root, args.targets, args.holdout,
        args.min_lf, args.max_lf, args.min_hf, args.max_hf,
        args.hf_extraction,
    )
    if args.output:
        write_json(args.output, report, overwrite=True)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
