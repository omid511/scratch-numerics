#!/usr/bin/env python3
"""Training-design-only nested CV for P4 ridge features; no held-out scoring.

Usage: OPENBLAS_NUM_THREADS=1 python experiment_p4_ridge_cv.py --modal
Each outer fold tunes alpha on three inner design folds, including scaling.
Intervals bootstrap paired OOF design errors; they do not include model-refit
uncertainty and are exploratory, not multiplicity-adjusted hypothesis tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge, fit_ridge_scaling
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from mechanics.p4_margin_estimation import sidecar
from train_p4_expanded import build_clips, load_dataset

ALPHAS = np.logspace(-6, 6, 25)
CLEAN_MEMBERS = [n for n in sidecar.member_names("G4")
                 if n not in {"cal_scale", "dom_freq", "bandwidth", "env_slope"}]


def design_folds(ids, count, seed):
    designs = np.unique(ids).copy()
    if count < 2 or count > len(designs):
        raise ValueError("fold count must be between 2 and the design count")
    np.random.default_rng(seed).shuffle(designs)
    return [np.flatnonzero(np.isin(ids, group))
            for group in np.array_split(designs, count)]


def ridge_predictions(x, y, query, alphas):
    """Same centering/scaling and unpenalized intercept as PhysicsFeatureRidge."""
    mean, scale, active = fit_ridge_scaling(x)
    z, q = (x - mean) / scale, (query - mean) / scale
    z[:, ~active] = 0.0
    q[:, ~active] = 0.0
    gram = z.T @ z
    rhs = z.T @ (y - y.mean())
    return np.column_stack([
        q @ np.linalg.solve(gram + a * np.eye(x.shape[1]), rhs) + y.mean()
        for a in alphas
    ])


def per_design_mae(y, predictions, ids):
    error = np.abs(predictions - y[:, None])
    return np.stack([error[ids == did].mean(axis=0) for did in np.unique(ids)])


def nested_cv(x, y, ids, *, seed=42, evaluation_x=None):
    """Tune on x only; optional row-aligned evaluation_x tests input shift."""
    query = x if evaluation_x is None else np.asarray(evaluation_x)
    if query.shape != x.shape or not np.isfinite(query).all():
        raise ValueError("evaluation features must be finite and row-aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite ridge inputs")
    predictions = np.full(len(y), np.nan)
    records = []
    all_rows = np.arange(len(y))
    for fold, outer_test in enumerate(design_folds(ids, 5, seed)):
        outer_train = np.setdiff1d(all_rows, outer_test)
        inner_ids = ids[outer_train]
        inner_scores = []
        for inner_val in design_folds(inner_ids, 3, seed + 100 + fold):
            inner_train = np.setdiff1d(np.arange(len(outer_train)), inner_val)
            fit, val = outer_train[inner_train], outer_train[inner_val]
            if set(ids[fit]) & set(ids[val]):
                raise AssertionError("inner design leakage")
            pred = ridge_predictions(x[fit], y[fit], x[val], ALPHAS)
            inner_scores.append(per_design_mae(y[val], pred, ids[val]))
        scores = np.concatenate(inner_scores).mean(axis=0)
        # Prefer stronger regularization only on exact ties.
        best = len(ALPHAS) - 1 - int(np.argmin(scores[::-1]))
        alpha = float(ALPHAS[best])
        predictions[outer_test] = ridge_predictions(
            x[outer_train], y[outer_train], query[outer_test], [alpha])[:, 0]
        if set(ids[outer_train]) & set(ids[outer_test]):
            raise AssertionError("outer design leakage")
        records.append({"fold": fold, "alpha": alpha,
                        "train_designs": np.unique(ids[outer_train]).tolist(),
                        "validation_designs": np.unique(ids[outer_test]).tolist(),
                        "inner_design_mae_by_alpha": scores.tolist(),
                        "inactive_features": np.flatnonzero(
                            ~fit_ridge_scaling(x[outer_train])[2]).tolist(),
                        "outer_design_mae": float(per_design_mae(
                            y[outer_test], predictions[outer_test, None],
                            ids[outer_test]).mean())})
    if not np.isfinite(predictions).all():
        raise AssertionError("incomplete OOF coverage")
    design_errors = dict(zip(np.unique(ids).tolist(), per_design_mae(
        y, predictions[:, None], ids)[:, 0].tolist()))
    return {"design_mae": float(np.mean(list(design_errors.values()))),
            "clip_mae": float(np.mean(np.abs(predictions - y))),
            "per_design_mae": design_errors, "folds": records,
            "oof_predictions": predictions.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="p4_dataset_saturation")
    parser.add_argument("--output-dir", default="p4_groups/ridge_cv")
    parser.add_argument("--modal", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    signals, margins, velocities, design_ids, realization_ids, meta = load_dataset(args.dataset)
    train_ids = set(meta["train_designs"])
    if train_ids & (set(meta["val_designs"]) | set(meta["test_designs"])):
        raise ValueError("dataset design splits overlap")
    provenance = meta["provenance"]
    clips, _ = build_clips(signals, margins, velocities, design_ids, train_ids,
                          dts=provenance["dts"], realization_ids=realization_ids,
                          sat_fracs=provenance["sat_fracs"])
    ridge = PhysicsFeatureRidge()
    base = np.stack([ridge._row(c) for c in clips])
    side = sidecar.extract(clips, CLEAN_MEMBERS)
    y = np.array([c.margin for c in clips])
    ids = np.array([c.design_id for c in clips])
    arms = {"baseline": base,
            "+zcr": np.column_stack([base, side[:, CLEAN_MEMBERS.index("zcr")]]),
            "+log_dt": np.column_stack([base, side[:, CLEAN_MEMBERS.index("log_dt")]]),
            "+bundle12_clean": np.column_stack([base, side])}
    results = {"protocol": {"outer_folds": 5, "inner_folds": 3, "seed": 42,
                           "alpha_grid": ALPHAS.tolist(),
                           "tuning_metric": "equal-weight mean per-design MAE",
                           "scaling": "fit on each inner/outer training fold only",
                           "held_out_selection_calibration_test_scored": False,
                           "bootstrap_warning": "paired OOF design resampling; excludes refit uncertainty; exploratory"},
               "dataset": args.dataset, "n_clips": len(clips),
               "n_designs": len(train_ids), "clean_members": CLEAN_MEMBERS,
               "source_sha256": {}, "records": {}}
    for name, x in arms.items():
        results["records"][name] = nested_cv(x, y, ids)
        print(name, results["records"][name]["design_mae"], flush=True)
    if args.modal:
        from mechanics.p4_margin_estimation.modal_features import (
            FEATURE_NAMES, QUALITY_INDICES, extract_modal_features,
        )
        rows = []
        for i, clip in enumerate(clips):
            rows.append(extract_modal_features(clip.sensor_signals, clip.dt))
            if (i + 1) % 500 == 0:
                print(f"Modal extraction {i + 1}/{len(clips)}", flush=True)
        modal = np.stack(rows)
        arms["+modal_quality"] = np.column_stack([base, modal[:, QUALITY_INDICES]])
        arms["+modal"] = np.column_stack([base, modal])
        arms["+bundle12_clean+modal"] = np.column_stack([base, side, modal])
        results["modal_features"] = list(FEATURE_NAMES)
        results["modal_settings"] = {"method": "multichannel LS-ESPRIT",
                                     "max_modes": 8, "lag": 64, "cap": 50.0,
                                     "relative_singular_threshold": 1e-6,
                                     "fit_window": "prefix before first sensor-cap hit"}
        results["modal_quality_summary"] = {
            "available_fraction": float(modal[:, 14].mean()),
            "mode_count_quantiles": np.quantile(modal[:, 11], [0, .5, 1]).tolist(),
            "fit_error_quantiles": np.quantile(modal[:, 12], [.1, .5, .9]).tolist(),
            "usable_fraction_quantiles": np.quantile(modal[:, 13], [0, .5, 1]).tolist(),
        }
        for name in ("+modal_quality", "+modal", "+bundle12_clean+modal"):
            results["records"][name] = nested_cv(arms[name], y, ids)
            print(name, results["records"][name]["design_mae"], flush=True)
    baseline = results["records"]["baseline"]["per_design_mae"]
    for name, record in results["records"].items():
        record["paired_vs_baseline"] = paired_design_comparison(
            record["per_design_mae"].items(), baseline.items(), n_bootstrap=10000, seed=42)
    if args.modal:
        results["modal_comparisons"] = {
            f"{left}_minus_{right}": paired_design_comparison(
                results["records"][left]["per_design_mae"].items(),
                results["records"][right]["per_design_mae"].items(),
                n_bootstrap=10000, seed=42)
            for left, right in (("+modal", "+modal_quality"),
                                ("+bundle12_clean+modal", "+bundle12_clean"))
        }
    sources = [Path(__file__), ROOT / "src/mechanics/p4_margin_estimation/baselines.py",
               ROOT / "src/mechanics/p4_margin_estimation/sidecar.py"]
    if args.modal:
        sources.append(ROOT / "src/mechanics/p4_margin_estimation/modal_features.py")
    results["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources}
    np.savez_compressed(out / "features.npz", baseline=base, sidecar=side, y=y,
                        design_ids=ids, **({"modal": modal} if args.modal else {}))
    result_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
