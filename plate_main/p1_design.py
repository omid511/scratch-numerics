#!/usr/bin/env python3
"""Versioned design-table and run-manifest utilities for the full P1 workflow.

The existing LF implementation currently supports the five geometry variables
in ``PILOT_PARAMETER_NAMES``.  The proposal's eventual HF study must also
record boundary stiffnesses, wall thickness, independent layer thicknesses,
and aerodynamic pressure; those names are declared below, but no physical
adapter is invented here because their exact definitions are not present in
the repository.  A final full study must supply that adapter explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from lhs_sampling import PARAM_RANGES, generate_lhs, scale_to_physical


PILOT_PARAMETER_NAMES = tuple(PARAM_RANGES)
# Explicit proposal-level names.  These are a contract for planning and
# provenance only until the LF/HF physics adapters define their units/maps.
PROPOSAL_PARAMETER_NAMES = (
    "k1", "k2", "k3", "k4", "k5", "cell_wall_thickness",
    "theta_c", "h_top", "h_bottom", "aerodynamic_pressure",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_id_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    if not stem.startswith("run_"):
        raise ValueError(f"Expected run_XXXX.npz, received {path}")
    try:
        run_id = int(stem[4:])
    except ValueError as error:
        raise ValueError(f"Invalid run bundle name: {path}") from error
    if run_id < 1:
        raise ValueError(f"Run ID must be positive: {path}")
    return run_id


def read_design_table(
    path: str | Path, parameter_names: Sequence[str] = PILOT_PARAMETER_NAMES,
) -> list[tuple[int, dict[str, float]]]:
    """Read and validate an immutable design table with explicit IDs when present."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    names = tuple(parameter_names)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if not set(names).issubset(fields):
            raise ValueError(f"Design table must contain {names}; found {fields}")
        id_column = next((column for column in ("run_id", "run") if column in fields), None)
        result: list[tuple[int, dict[str, float]]] = []
        seen: set[int] = set()
        for row_index, row in enumerate(reader, 1):
            try:
                run_id = int(row[id_column]) if id_column else row_index
                values = {name: float(row[name]) for name in names}
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"Invalid design row {row_index}") from error
            if run_id < 1 or run_id in seen:
                raise ValueError(f"Duplicate or nonpositive run ID {run_id}")
            if not np.isfinite(list(values.values())).all():
                raise ValueError(f"Nonfinite design row {run_id}")
            seen.add(run_id)
            result.append((run_id, values))
    if not result:
        raise ValueError(f"Empty design table: {path}")
    return result


def write_design_table(
    path: str | Path, rows: Sequence[tuple[int, Mapping[str, float]]],
    parameter_names: Sequence[str] = PILOT_PARAMETER_NAMES,
    overwrite: bool = False,
) -> None:
    """Write a design table without silently replacing an existing artifact."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite design table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(parameter_names)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", *names])
            for run_id, values in rows:
                writer.writerow([int(run_id), *(float(values[name]) for name in names)])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_pilot_lhs(
    count: int, seed: int = 42, distribution: str = "legacy",
) -> list[tuple[int, dict[str, float]]]:
    """Generate a new explicit-ID table using the existing P1 sampler."""
    if count < 1:
        raise ValueError("count must be positive")
    if distribution not in {"legacy", "uniform"}:
        raise ValueError("distribution must be legacy or uniform")
    unit = generate_lhs(count, len(PILOT_PARAMETER_NAMES), seed)
    if distribution == "legacy":
        values = scale_to_physical(unit)
    else:
        bounds = np.asarray(list(PARAM_RANGES.values()), dtype=float)
        values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return [
        (index, {name: float(row[column]) for column, name in enumerate(PILOT_PARAMETER_NAMES)})
        for index, row in enumerate(values, 1)
    ]


def write_json(path: str | Path, payload: Mapping[str, object], overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_design_manifest(
    design_path: str | Path, manifest_path: str | Path, *, count: int,
    seed: int, distribution: str, scope: str = "pilot-five-variable",
    overwrite: bool = False,
) -> dict[str, object]:
    design_path = Path(design_path)
    rows = read_design_table(design_path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "p1_design",
        "scope": scope,
        "parameter_names": list(PILOT_PARAMETER_NAMES),
        "proposal_parameter_names": list(PROPOSAL_PARAMETER_NAMES),
        "count": len(rows),
        "requested_count": int(count),
        "seed": int(seed),
        "distribution": distribution,
        "run_id_policy": "explicit one-based run_id; row order is not an identity",
        "design_sha256": file_sha256(design_path),
        "source": "lhs_sampling.generate_lhs/scale_to_physical",
        "simulation_adapter_status": {
            "pilot_five_variable_lf": "implemented",
            "proposal_boundary_stiffness_pressure_adapter": "not-defined-in-repository",
        },
    }
    write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


def write_role_plan(
    path: str | Path, design_rows: Sequence[tuple[int, Mapping[str, float]]],
    roles: Mapping[int, str], *, selection_scores: Mapping[int, float] | None = None,
    selection_ranks: Mapping[int, int] | None = None,
    parameter_names: Sequence[str] = PILOT_PARAMETER_NAMES,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite HF plan: {path}")
    selection_scores = selection_scores or {}
    selection_ranks = selection_ranks or {}
    names = tuple(parameter_names)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", *names, "role", "selection_rank", "selection_score"])
            for run_id, values in design_rows:
                if run_id not in roles:
                    continue
                writer.writerow([
                    run_id, *(values[name] for name in names), roles[run_id],
                    selection_ranks.get(run_id, ""), selection_scores.get(run_id, ""),
                ])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_run_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        runs = list(dict.fromkeys(int(token.strip()) for token in value.split(",") if token.strip()))
    except ValueError as error:
        raise ValueError("Run list must contain comma-separated integers") from error
    if not runs or any(run < 1 for run in runs):
        raise ValueError("Run list must contain positive integers")
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distribution", choices=("legacy", "uniform"), default="legacy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = generate_pilot_lhs(args.count, args.seed, args.distribution)
    write_design_table(args.output, rows, overwrite=args.overwrite)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = write_design_manifest(
        args.output, manifest_path, count=args.count, seed=args.seed,
        distribution=args.distribution, overwrite=args.overwrite,
    )
    print(json.dumps({"design": str(args.output.resolve()), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
