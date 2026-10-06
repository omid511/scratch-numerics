#!/usr/bin/env python3
"""Measure LF design sensitivity and select a fixed HF run plan.

The selection uses only LF outputs and the frozen design table.  It must be
run before any HF results are inspected.  A weighted maximin subset is chosen
from the existing LF candidate pool, then divided into train/calibration/test
roles deterministically.  This prevents using expensive HF outcomes to tune
the design itself.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from p1_design import (
    PILOT_PARAMETER_NAMES, PROPOSAL_PARAMETER_NAMES, file_sha256,
    read_design_table, write_json, write_role_plan,
)


def design_parameter_names(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        fields = tuple(next(csv.reader(stream)))
    names = tuple(field for field in fields if field != "run_id")
    if names not in (PILOT_PARAMETER_NAMES, PROPOSAL_PARAMETER_NAMES):
        raise ValueError(f"Unsupported design parameter names: {names}")
    return names



def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return 0.0
    xr = rankdata(x)
    yr = rankdata(y)
    xr -= xr.mean()
    yr -= yr.mean()
    denominator = np.linalg.norm(xr) * np.linalg.norm(yr)
    return float(xr @ yr / denominator) if denominator else 0.0
def _standardized_coefficients(parameters: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    x = (parameters - parameters.mean(axis=0)) / np.maximum(parameters.std(axis=0), 1e-12)
    y = (outputs - outputs.mean(axis=0)) / np.maximum(outputs.std(axis=0), 1e-12)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    return coefficients.T


def load_lf_frequency_features(
    design_rows: Sequence[tuple[int, Mapping[str, float]]], lf_root: str | Path,
    modes: int, parameter_names: Sequence[str] = PILOT_PARAMETER_NAMES,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Load only frequencies, avoiding a full 80x80 field matrix in memory."""
    lf_root = Path(lf_root)
    parameters = []
    features = []
    valid_runs = []
    feature_names = [f"frequency_mode_{mode:02d}" for mode in range(1, modes + 1)]
    feature_names += [f"frequency_ratio_mode_{mode:02d}" for mode in range(2, modes + 1)]
    for run_id, values in design_rows:
        path = lf_root / f"run_{run_id:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing LF bundle for design run {run_id}: {path}")
        with np.load(path, allow_pickle=False) as archive:
            frequencies = np.asarray(archive["frequencies"], dtype=float)
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("run_id") != run_id or len(frequencies) < modes:
                raise ValueError(f"LF bundle {path} has incompatible run/mode metadata")
            if not np.isfinite(frequencies[:modes]).all() or np.any(frequencies[:modes] <= 0):
                raise ValueError(f"LF bundle {path} has invalid frequencies")
        base = frequencies[:modes]
        ratios = base[1:] / base[0]
        parameters.append([values[name] for name in parameter_names])
        features.append(np.r_[base, ratios])
        valid_runs.append(run_id)
    return np.asarray(features), feature_names, valid_runs


def compute_sensitivity(
    parameters: np.ndarray, outputs: np.ndarray,
    parameter_names: Sequence[str] = PILOT_PARAMETER_NAMES,
) -> tuple[list[dict[str, object]], np.ndarray]:
    if parameters.ndim != 2 or outputs.ndim != 2 or len(parameters) != len(outputs):
        raise ValueError("Sensitivity matrices have inconsistent shapes")
    spearman = np.asarray([
        [_rank_correlation(parameters[:, column], outputs[:, target])
         for target in range(outputs.shape[1])]
        for column in range(parameters.shape[1])
    ])
    coefficients = _standardized_coefficients(parameters, outputs)
    rank_score = np.median(np.abs(spearman), axis=1)
    coefficient_score = np.median(np.abs(coefficients), axis=0)
    raw = .5 * rank_score + .5 * coefficient_score
    normalized = raw / max(float(np.sum(raw)), 1e-30)
    # Keep a small positive floor so an apparently inactive variable remains
    # represented in the space-filling metric instead of being discarded.
    weights = .05 / len(parameter_names) + .95 * normalized
    records = []
    for index, name in enumerate(parameter_names):
        records.append({
            "parameter": name,
            "spearman_median_abs": float(rank_score[index]),
            "spearman_max_abs": float(np.max(np.abs(spearman[index]))),
            "standardized_coefficient_median_abs": float(coefficient_score[index]),
            "score": float(raw[index]),
            "sampling_weight": float(weights[index]),
        })
    return records, weights


def _scaled_parameters(parameters: np.ndarray) -> np.ndarray:
    lower = parameters.min(axis=0)
    scale = np.where(np.ptp(parameters, axis=0) > 1e-12, np.ptp(parameters, axis=0), 1.0)
    return (parameters - lower) / scale


def _weighted_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    difference = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(weights[None, None, :] * difference * difference, axis=2))


def weighted_maximin_indices(
    parameters: np.ndarray, count: int, weights: np.ndarray, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic weighted space-filling subset."""
    if count < 1 or count > len(parameters):
        raise ValueError("Selection count is outside the candidate range")
    scaled = _scaled_parameters(parameters)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (parameters.shape[1],) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Sensitivity weights must be positive and match parameters")
    center = np.full(parameters.shape[1], .5)
    first_score = np.sum(weights * (scaled - center) ** 2, axis=1)
    first_candidates = np.flatnonzero(np.isclose(first_score, np.max(first_score)))
    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(first_candidates))]
    min_distance = _weighted_distance(scaled, scaled[selected], weights)[:, 0]
    selection_scores = np.full(len(parameters), np.nan)
    selection_scores[selected[0]] = float(first_score[selected[0]])
    for _ in range(1, count):
        # A small deterministic extremeness term protects influential tails
        # without defeating the maximin coverage objective.
        extremeness = np.sum(weights * (scaled - .5) ** 2, axis=1)
        criterion = min_distance + .05 * extremeness
        criterion[selected] = -np.inf
        best = np.flatnonzero(np.isclose(criterion, np.max(criterion)))
        chosen = int(rng.choice(best))
        selected.append(chosen)
        selection_scores[chosen] = float(min_distance[chosen])
        distance = _weighted_distance(scaled, scaled[[chosen]], weights)[:, 0]
        min_distance = np.minimum(min_distance, distance)
    return np.asarray(selected, dtype=int), selection_scores


def _assign_roles(
    selected: np.ndarray, parameters: np.ndarray, weights: np.ndarray,
    n_test: int, n_calibration: int, seed: int,
) -> tuple[dict[int, str], dict[int, int], dict[int, float]]:
    if n_test < 1 or n_calibration < 1 or n_test + n_calibration >= len(selected):
        raise ValueError("Need at least one training run plus test and calibration runs")
    roles: dict[int, str] = {}
    ranks: dict[int, int] = {}
    scores: dict[int, float] = {}
    # Test points are selected first and spread across the weighted design.
    test_local, test_scores = weighted_maximin_indices(
        parameters, n_test, weights, seed=seed + 101
    )
    test_set = set(test_local.tolist())
    remaining = np.asarray([i for i in range(len(selected)) if i not in test_set], dtype=int)
    calibration_local_relative, calibration_scores = weighted_maximin_indices(
        parameters[remaining], n_calibration, weights, seed=seed + 202
    )
    calibration_set = set(remaining[calibration_local_relative].tolist())
    selected_order = list(test_local.tolist()) + list(remaining[calibration_local_relative].tolist())
    selected_order += [i for i in range(len(selected)) if i not in test_set and i not in calibration_set]
    for rank, local in enumerate(selected_order, 1):
        run_id = int(selected[local])
        if local in test_set:
            role = "test"
            score = float(test_scores[local])
        elif local in calibration_set:
            role = "calibration"
            score = float(calibration_scores[np.flatnonzero(remaining == local)[0]])
        else:
            role = "train"
            score = float(np.nan)
        roles[run_id] = role
        ranks[run_id] = rank
        scores[run_id] = score
    return roles, ranks, scores


def write_sensitivity_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--lf", type=Path, required=True,
                        help="LF root containing run_XXXX.npz files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--hf-count", type=int, default=80)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--calibration-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.modes, args.hf_count, args.test_count, args.calibration_count) < 1:
        parser.error("mode and selection counts must be positive")
    if not 50 <= args.hf_count <= 100:
        parser.error("Proposal 1 HF count must be between 50 and 100")
    parameter_names = design_parameter_names(args.samples)
    design_rows = read_design_table(args.samples, parameter_names)
    outputs, feature_names, valid_runs = load_lf_frequency_features(
        design_rows, args.lf, args.modes, parameter_names
    )
    values_by_run = {run_id: values for run_id, values in design_rows}
    parameters = np.asarray([
        [values_by_run[run_id][name] for name in parameter_names]
        for run_id in valid_runs
    ])
    sensitivity_records, weights = compute_sensitivity(
        parameters, outputs, parameter_names
    )
    selected_local, _ = weighted_maximin_indices(
        parameters, args.hf_count, weights, seed=args.seed
    )
    selected_runs = np.asarray([valid_runs[index] for index in selected_local], dtype=int)
    roles, ranks, role_scores = _assign_roles(
        selected_runs, parameters[selected_local], weights,
        args.test_count, args.calibration_count, args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_sensitivity_csv(args.output / "sensitivity.csv", sensitivity_records)
    selected_rows = [row for row in design_rows if row[0] in roles]
    write_role_plan(
        args.output / "hf_plan.csv", selected_rows, roles,
        selection_scores=role_scores, selection_ranks=ranks,
        parameter_names=parameter_names,
    )
    np.savez_compressed(
        args.output / "lf_features.npz", run_ids=np.asarray(valid_runs, dtype=int),
        parameters=parameters, outputs=outputs, feature_names=np.asarray(feature_names),
        parameter_names=np.asarray(parameter_names),
    )
    role_counts = {
        role: sum(value == role for value in roles.values())
        for role in ("train", "calibration", "test")
    }
    manifest = {
        "schema_version": 1,
        "artifact": "p1_hf_plan",
        "design_sha256": file_sha256(args.samples),
        "design_path": str(args.samples.resolve()),
        "lf_root": str(args.lf.resolve()),
        "lf_run_count": len(valid_runs),
        "lf_feature_names": feature_names,
        "parameter_names": list(parameter_names),
        "sensitivity": sensitivity_records,
        "sampling_weights": weights.tolist(),
        "selection": {
            "method": "weighted maximin on LF frequency/features",
            "seed": args.seed, "hf_count": args.hf_count,
            "role_counts": role_counts,
            "role_policy": "roles fixed before HF simulation; no HF labels used",
        },
        "run_ids": {
            role: sorted(int(run) for run, value in roles.items() if value == role)
            for role in ("train", "calibration", "test")
        },
    }
    write_json(args.output / "hf_plan.json", manifest)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "role_counts": role_counts,
        "weights": dict(zip(parameter_names, weights.tolist())),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
