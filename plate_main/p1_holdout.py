#!/usr/bin/env python3
"""Evaluate P1 on predeclared HF test runs with run-level calibration.

The script consumes a target revision built from a fixed HF plan.  The plan
must label each HF design as ``train``, ``calibration``, or ``test`` before HF
results are generated.  The latent model is fitted only on train runs; the
calibration runs determine simultaneous run-level conformal multipliers; the
test runs are touched exactly once for the final report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from p1_design import file_sha256, write_json
from p1_surrogate import P1Dataset, LatentGPModel, evaluate_predictions


Z_BY_COVERAGE = {50: .67448975, 90: 1.64485363, 95: 1.95996398}


def read_roles(path: str | Path) -> dict[int, str]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"run_id", "role"}.issubset(rows[0]):
        raise ValueError("HF plan must contain run_id and role columns")
    roles: dict[int, str] = {}
    for row in rows:
        run = int(row["run_id"])
        role = row["role"].strip().lower()
        if role not in {"train", "calibration", "test"} or run in roles:
            raise ValueError(f"Invalid or duplicate HF role for run {run}")
        roles[run] = role
    return roles


def _finite_quantile(values: Sequence[float], coverage: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    if not 0 < coverage < 100:
        raise ValueError("Coverage must be between 0 and 100")
    rank = ((len(values) + 1) * coverage + 99) // 100
    if rank > len(values):
        minimum = math.ceil(coverage / (100 - coverage))
        raise ValueError(
            f"Finite {coverage}% intervals require at least {minimum} calibration runs; "
            f"received {len(values)}"
        )
    return float(np.partition(values, rank - 1)[rank - 1])


class RunConformalCalibrator:
    """Run-level simultaneous conformal scale for GP intervals."""

    def __init__(self):
        self.field_multiplier: dict[int, float] = {}
        self.frequency_multiplier: dict[int, float] = {}
        self.calibration_runs: list[int] = []
        self.field_score_by_run: dict[int, float] = {}
        self.frequency_score_by_run: dict[int, float] = {}
        self.unsupported_coverages: dict[int, str] = {}

    def fit(
        self, dataset: P1Dataset, rows: Sequence[int], prediction: Mapping[str, np.ndarray],
    ) -> "RunConformalCalibrator":
        rows = np.asarray(rows, dtype=int)
        fields = np.asarray(prediction["correction"], dtype=float)
        field_std = np.maximum(np.asarray(prediction["correction_std"], dtype=float), 1e-12)
        frequencies = np.asarray(prediction["frequency"], dtype=float)
        frequency_std = np.maximum(np.asarray(prediction["frequency_std"], dtype=float), 1e-12)
        if fields.shape != (len(rows), *dataset.grid_shape):
            raise ValueError("Calibration field shape mismatch")
        self.calibration_runs = sorted(np.unique(dataset.run_ids[rows]).astype(int).tolist())
        if len(self.calibration_runs) < 3:
            raise ValueError("At least three calibration runs are required")
        self.field_multiplier.clear()
        self.frequency_multiplier.clear()
        self.field_score_by_run.clear()
        self.frequency_score_by_run.clear()
        self.unsupported_coverages.clear()
        interior = dataset.interior_mask
        field_scores = []
        frequency_scores = []
        for run in self.calibration_runs:
            local = np.flatnonzero(dataset.run_ids[rows] == run)
            error = np.abs(fields[local] - dataset.correction[rows[local]])
            # Maximum over accepted modes and interior pixels keeps the unit of
            # exchangeability at the HF design/run, not at individual pixels.
            field_score = float(np.max(error[:, interior] / field_std[local][:, interior]))
            self.field_score_by_run[run] = field_score
            field_scores.append(field_score)
            freq_error = np.abs(frequencies[local] - dataset.f_hf[rows[local]])
            valid = np.isfinite(freq_error) & np.isfinite(frequency_std[local])
            if np.any(valid):
                score = float(np.max(freq_error[valid] / frequency_std[local][valid]))
                self.frequency_score_by_run[run] = score
                frequency_scores.append(score)
        for coverage in (90, 95):
            minimum = math.ceil(coverage / (100 - coverage))
            if len(field_scores) < minimum:
                self.unsupported_coverages[coverage] = (
                    f"Finite {coverage}% intervals require at least {minimum} "
                    f"calibration runs; received {len(field_scores)}"
                )
                continue
            self.field_multiplier[coverage] = _finite_quantile(field_scores, coverage)
            if len(frequency_scores) >= minimum:
                self.frequency_multiplier[coverage] = _finite_quantile(frequency_scores, coverage)
        return self

    def adjusted_prediction(
        self, prediction: Mapping[str, np.ndarray], coverage: int = 90,
    ) -> dict[str, np.ndarray]:
        if coverage not in self.field_multiplier:
            raise ValueError(self.unsupported_coverages.get(
                coverage, f"No field calibration for {coverage}% coverage"))
        field_scale = self.field_multiplier[coverage] / Z_BY_COVERAGE[coverage]
        result = dict(prediction)
        result["correction_std"] = np.asarray(prediction["correction_std"]) * field_scale
        if coverage in self.frequency_multiplier:
            frequency_scale = self.frequency_multiplier[coverage] / Z_BY_COVERAGE[coverage]
            result["frequency_std"] = np.asarray(prediction["frequency_std"]) * frequency_scale
        return result

    def state(self) -> dict[str, object]:
        return {
            "method": "run-level simultaneous conformal normalized residual",
            "quantile_rule": "ceil((n + 1) * coverage / 100)-th smallest run score",
            "unsupported_coverages": {
                str(key): value for key, value in self.unsupported_coverages.items()
            },
            "calibration_runs": self.calibration_runs,
            "field_multiplier": {str(key): value for key, value in self.field_multiplier.items()},
            "frequency_multiplier": {str(key): value for key, value in self.frequency_multiplier.items()},
            "field_score_by_run": {str(key): value for key, value in self.field_score_by_run.items()},
            "frequency_score_by_run": {str(key): value for key, value in self.frequency_score_by_run.items()},
        }


def _write_figures(
    output: Path, dataset: P1Dataset, rows: np.ndarray,
    raw: Mapping[str, np.ndarray], calibrated: Mapping[str, np.ndarray], dpi: int,
    calibrated_coverage: int,
) -> None:
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    raw_fields = np.asarray(raw["correction"])
    calibrated_std = np.asarray(calibrated["correction_std"])
    error = raw_fields - dataset.correction[rows]
    interior = dataset.interior_mask
    coverage = float(np.mean(
        np.abs(error[:, interior])
        <= Z_BY_COVERAGE[calibrated_coverage] * calibrated_std[:, interior]))
    mean_rms = np.sqrt(np.mean(error[:, interior] ** 2, axis=1))
    worst = int(np.argmax(mean_rms))
    extent = [float(dataset.x[0]), float(dataset.x[-1]), float(dataset.y[0]), float(dataset.y[-1])]
    scale = max(float(np.max(np.abs(dataset.correction[rows[worst]]))),
                float(np.max(np.abs(raw_fields[worst]))), 1e-12)
    error_scale = max(float(np.max(np.abs(error[worst]))), 1e-12)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    fields = (
        (dataset.correction[rows[worst]], "Held-out HF−LF", "RdBu_r", scale),
        (raw_fields[worst], "Raw prediction", "RdBu_r", scale),
        (error[worst], "Prediction error", "RdBu_r", error_scale),
        (calibrated_std[worst], f"Calibrated {calibrated_coverage}% scale",
         "viridis", float(np.max(calibrated_std[worst]))),
    )
    for axis, (field, title, cmap, vmax) in zip(axes, fields):
        image = axis.imshow(field, origin="lower", extent=extent, aspect="equal", cmap=cmap,
                            vmin=-vmax if cmap == "RdBu_r" else 0, vmax=vmax)
        axis.set(title=title, xlabel="x (m)", ylabel="y (m)")
        fig.colorbar(image, ax=axis, shrink=.82)
    fig.suptitle(
        f"Strict HF test worst field: run {dataset.run_ids[rows[worst]]}, "
        f"mode {dataset.mode_ids[rows[worst]]}; "
        f"pointwise {calibrated_coverage}% coverage={coverage:.3f}"
    )
    fig.savefig(output / "heldout_worst_field.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output / "heldout_worst_field.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", default="corrections_full")
    parser.add_argument("--plan", type=Path, required=True,
                        help="hf_plan.csv generated before HF simulation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        help="Reuse a saved train-only latent GP without refitting")
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--calibrated-coverage", type=int, choices=(90, 95), default=90)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    revision = args.data / args.name if args.data.is_dir() and not (args.data / "training.npz").exists() else args.data
    dataset = P1Dataset.load(revision)
    roles = read_roles(args.plan)
    accepted_roles = {role: [] for role in ("train", "calibration", "test")}
    for index, run in enumerate(dataset.run_ids.astype(int).tolist()):
        if run not in roles:
            raise ValueError(f"Target row run {run} is absent from the predeclared HF plan")
        accepted_roles[roles[run]].append(index)
    if any(not rows for rows in accepted_roles.values()):
        raise ValueError("Every train/calibration/test role needs accepted target rows")
    train_rows = np.asarray(accepted_roles["train"], dtype=int)
    calibration_rows = np.asarray(accepted_roles["calibration"], dtype=int)
    test_rows = np.asarray(accepted_roles["test"], dtype=int)
    calibration_count = len(np.unique(dataset.run_ids[calibration_rows]))
    minimum_calibration = math.ceil(
        args.calibrated_coverage / (100 - args.calibrated_coverage))
    if calibration_count < minimum_calibration:
        raise ValueError(
            f"Finite {args.calibrated_coverage}% intervals require at least "
            f"{minimum_calibration} calibration runs; received {calibration_count}"
        )
    train_modes = set(dataset.mode_ids[train_rows].astype(int).tolist())
    missing_modes = sorted(set(dataset.mode_ids[test_rows].astype(int).tolist()) - train_modes)
    if missing_modes:
        raise ValueError(f"Test contains modes absent from training rows: {missing_modes}")

    if args.model is None:
        model = LatentGPModel(args.latent_dim, seed=42, max_iterations=args.max_iterations)
        model.fit(dataset, train_rows)
    else:
        model = LatentGPModel.load(args.model)
        if (model.parameter_names != dataset.parameter_names
                or model.grid_shape != dataset.grid_shape
                or not np.array_equal(model.training_rows, train_rows)
                or not np.array_equal(model.training_runs, np.unique(dataset.run_ids[train_rows]))):
            raise ValueError("Saved model does not match the frozen train-only target rows")
    calibration_prediction = model.predict_rows(dataset, calibration_rows)
    calibrator = RunConformalCalibrator().fit(dataset, calibration_rows, calibration_prediction)
    raw_test = model.predict_rows(dataset, test_rows)
    calibrated_test = calibrator.adjusted_prediction(raw_test, args.calibrated_coverage)
    raw_records, raw_summary = evaluate_predictions(dataset, test_rows, raw_test, "latent_gp_holdout_raw", 0)
    calibrated_records, calibrated_summary = evaluate_predictions(
        dataset, test_rows, calibrated_test, "latent_gp_holdout_calibrated", 0
    )
    # Only the requested level has been conformally calibrated.
    other_levels = set(Z_BY_COVERAGE) - {args.calibrated_coverage}
    for quantity in ("field", "frequency"):
        for level in other_levels:
            key = f"{quantity}_coverage_{level}"
            for record in calibrated_records:
                record.pop(key, None)
            for statistic in ("mean", "median", "p95"):
                calibrated_summary.pop(f"{key}_{statistic}", None)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "latent_gp_train", dataset)
    with (args.output / "holdout_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        records = raw_records + calibrated_records
        keys = list(records[0])
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(records)
    np.savez_compressed(
        args.output / "holdout_predictions.npz", run_ids=dataset.run_ids[test_rows],
        mode_ids=dataset.mode_ids[test_rows], correction_true=dataset.correction[test_rows],
        correction_raw=raw_test["correction"], correction_raw_std=raw_test["correction_std"],
        correction_calibrated_std=calibrated_test["correction_std"],
        f_hf_true=dataset.f_hf[test_rows], f_hf_raw=raw_test["frequency"],
        f_hf_raw_std=raw_test["frequency_std"], f_hf_calibrated_std=calibrated_test["frequency_std"],
    )
    calibration_state = calibrator.state()
    write_json(args.output / "calibration.json", calibration_state)
    report = {
        "schema_version": 1,
        "artifact": "p1_strict_holdout",
        "data_revision": str(revision.resolve()),
        "data_sha256": file_sha256(revision / "training.npz"),
        "plan_sha256": file_sha256(args.plan),
        "role_runs": {role: sorted(run for run, value in roles.items() if value == role)
                       for role in ("train", "calibration", "test")},
        "accepted_rows": {role: len(rows) for role, rows in accepted_roles.items()},
        "raw_summary": raw_summary,
        "calibrated_summary": calibrated_summary,
        "calibration": calibration_state,
        "calibrated_coverage": args.calibrated_coverage,
        "reused_model": str(args.model.resolve()) if args.model else None,
        "claim_guard": "strict held-out HF pilot result; not a universal accuracy claim",
    }
    write_json(args.output / "holdout_metrics.json", report)
    _write_figures(
        args.output / "figures", dataset, test_rows, raw_test, calibrated_test,
        args.dpi, args.calibrated_coverage)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "raw_field_rms_median": raw_summary.get("field_interior_rms_median"),
        f"calibrated_field_coverage_{args.calibrated_coverage}_median": (
            calibrated_summary.get(f"field_coverage_{args.calibrated_coverage}_median")),
        f"field_multiplier_{args.calibrated_coverage}": (
            calibrator.field_multiplier[args.calibrated_coverage]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
