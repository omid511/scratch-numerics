#!/usr/bin/env python3
"""Training-design-only sensor placement/count study with nested ridge CV.

Fixed 25-position grid; nested greedy 4/8/16 layouts; original eight and
original-eight-plus-eight reference. Both outer and inner folds independently
select layouts using only their training designs' modal observability.
Noise is a controlled sensitivity probe, NOT a calibrated hardware model:
shared white noise, sigma=0/1/5% of original-eight calibration RMS, before
per-channel normalization. Candidate noise realizations are shared by layouts.
"""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from experiment_p4_ridge_cv import ALPHAS, design_folds, ridge_predictions, per_design_mae
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.modal_features import estimate_modes, modal_feature_row
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison

ARMS = ("current8", "greedy4", "greedy8", "greedy16", "extend16")
NOISE = (0.0, 0.01, 0.05)


def greedy_order(grams, count, initial=()):
    """Maximize mean per-training-design regularized logdet information."""
    g = np.asarray(grams, dtype=float)
    if g.ndim != 4 or not len(g) or not np.isfinite(g).all():
        raise ValueError("expected finite design/candidate/state/state information")
    chosen = list(initial)
    if len(set(chosen)) != len(chosen) or count < len(chosen) or count > g.shape[1]:
        raise ValueError("invalid sensor budget or initial layout")
    if any(i < 0 or i >= g.shape[1] for i in chosen):
        raise ValueError("initial sensor outside candidate pool")
    current = np.broadcast_to(1e-3 * np.eye(g.shape[-1]),
                              (len(g), g.shape[-1], g.shape[-1])).copy()
    if chosen:
        current += g[:, chosen].sum(axis=1)
    while len(chosen) < count:
        remaining = [i for i in range(g.shape[1]) if i not in chosen]
        signs, logdet = np.linalg.slogdet(current[:, None] + g[:, remaining])
        if np.any(signs <= 0):
            raise ValueError("non-positive regularized observability matrix")
        pick = remaining[int(np.argmax(logdet.mean(axis=0)))]
        chosen.append(pick)
        current += g[:, pick]
    return chosen


def choose_layouts(grams, original):
    order = greedy_order(grams, 16)
    extension = greedy_order(grams, 16, original)
    return {"current8": sorted(original), "greedy4": sorted(order[:4]),
            "greedy8": sorted(order[:8]), "greedy16": sorted(order),
            "extend16": sorted(extension)}


def make_plan(ids, design_ids, grams, original):
    all_rows = np.arange(len(ids))
    def select(rows):
        training = np.unique(ids[rows])
        return choose_layouts(grams[np.isin(design_ids, training)], original)
    plan = []
    for fold, test in enumerate(design_folds(ids, 5, 42)):
        train = np.setdiff1d(all_rows, test)
        inners = []
        for inner_val in design_folds(ids[train], 3, 142 + fold):
            fit = train[np.setdiff1d(np.arange(len(train)), inner_val)]
            val = train[inner_val]
            inners.append({"fit": fit.tolist(), "val": val.tolist(), "layouts": select(fit)})
        plan.append({"fold": fold, "fit": train.tolist(), "val": test.tolist(),
                     "layouts": select(train), "inner": inners})
    final = choose_layouts(grams, original)
    unique = {tuple(layout) for entry in plan for layout in entry["layouts"].values()}
    unique.update(tuple(layout) for entry in plan for inner in entry["inner"]
                  for layout in inner["layouts"].values())
    unique.update(tuple(layout) for layout in final.values())
    return plan, final, sorted(unique)


def recovery_metrics(modes, truth):
    """One-to-one frequency match; solver truth is diagnostic only.

    Matched means <=max(2 Hz, 5% of true frequency). Alpha errors are
    conditional on matching; missing critical mode has NaN error and must
    be reported together with its recovery fraction.
    """
    f = modes["frequency"]
    out = np.array([0.0, np.nan, np.nan, 0.0])
    if not len(f):
        return out
    target = truth.imag / (2 * np.pi)
    costs = np.abs(f[:, None] - target[None]) / np.maximum(2.0, .05 * target)[None]
    estimated, actual = linear_sum_assignment(costs)
    matched = costs[estimated, actual] <= 1.0
    out[0] = matched.sum() / len(truth)
    if matched.any():
        out[1] = np.mean(np.abs(modes["alpha"][estimated[matched]] - truth.real[actual[matched]]))
    critical = np.flatnonzero((actual == 0) & matched)
    if critical.size:
        out[2] = abs(modes["alpha"][estimated[critical[0]]] - truth[0].real)
        out[3] = 1.0
    return out


def feature_shard(did, candidates, source, output, layouts, fingerprint):
    with threadpool_limits(limits=1):
        return _feature_shard(did, candidates, source, output, layouts, fingerprint)


def _feature_shard(did, candidates, source, output, layouts, fingerprint):
    with np.load(Path(candidates) / f"{did}.npz") as shard:
        raw, rows = shard["raw"], shard["source_indices"]
        truth, noise_scale = shard["poles"], shard["noise_scale"]
    with np.load(Path(source) / "metadata_arrays.npz") as a:
        dt = a["dts"][rows]
    features = np.empty((len(NOISE), len(layouts), len(rows), 22))
    diagnostic = np.empty((len(NOISE), len(layouts), len(rows), 4))
    ridge = PhysicsFeatureRidge()
    layout_rows = [np.asarray(layout, dtype=np.intp) for layout in layouts]
    for j, index in enumerate(rows):
        white = np.random.default_rng(np.random.SeedSequence([928, int(index)])).normal(size=raw[j].shape)
        for ni, level in enumerate(NOISE):
            measured = raw[j] + level * noise_scale[j] * white
            # Per-channel calibration commutes with selecting channels. Raw
            # responses remain retained for future global-normalization studies.
            normalized = np.clip(causal_calibration_normalize(
                measured, calibration_samples=51, normalize_mode="per_channel"), -50, 50).astype(np.float32)
            for li, layout in enumerate(layout_rows):
                signal = normalized[layout]
                modes = estimate_modes(signal, float(dt[j]))
                features[ni, li, j] = np.concatenate([
                    ridge._extract_features(signal, dt=float(dt[j])), modal_feature_row(modes)])
                diagnostic[ni, li, j] = recovery_metrics(modes, truth[j])
    if not np.isfinite(features).all():
        raise ValueError(f"nonfinite sensor features: {did}")
    path = Path(output) / f"{did}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, features=features, recovery=diagnostic,
                            source_indices=rows, fingerprint=fingerprint)
    temporary.replace(path)
    return did


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnostic_summary(d):
    def quantiles(column):
        valid = column[np.isfinite(column)]
        return np.quantile(valid, [.5, .9]).tolist() if len(valid) else None
    return {"mean_mode_recovery_fraction": float(d[:, 0].mean()),
            "matched_alpha_abs_error_median_p90": quantiles(d[:, 1]),
            "critical_alpha_abs_error_median_p90": quantiles(d[:, 2]),
            "critical_recovery_fraction": float(d[:, 3].mean())}


def evaluate_arm(arm, ni, features, diagnostics, y, ids, plan, layout_index, *, transfer=False):
    predictions = np.full(len(y), np.nan)
    observed = np.full((len(y), 4), np.nan)
    folds = []
    train_noise = 0 if transfer else ni
    for outer in plan:
        inner_errors = []
        for inner in outer["inner"]:
            fit, val = np.array(inner["fit"]), np.array(inner["val"])
            li = layout_index[tuple(inner["layouts"][arm])]
            if set(ids[fit]) & set(ids[val]):
                raise AssertionError("inner design leakage")
            x = features[train_noise, li]
            pred = ridge_predictions(x[fit], y[fit], x[val], ALPHAS)
            inner_errors.append(per_design_mae(y[val], pred, ids[val]))
        scores = np.concatenate(inner_errors).mean(axis=0)
        best = len(ALPHAS) - 1 - int(np.argmin(scores[::-1]))
        alpha = float(ALPHAS[best])
        fit, val = np.array(outer["fit"]), np.array(outer["val"])
        li = layout_index[tuple(outer["layouts"][arm])]
        if set(ids[fit]) & set(ids[val]):
            raise AssertionError("outer design leakage")
        predictions[val] = ridge_predictions(features[train_noise, li, fit], y[fit],
                                             features[ni, li, val], [alpha])[:, 0]
        observed[val] = diagnostics[ni, li, val]
        folds.append({"fold": outer["fold"], "alpha": alpha,
                      "layout": outer["layouts"][arm],
                      "inner_design_mae_by_alpha": scores.tolist(),
                      "outer_design_mae": float(per_design_mae(y[val], predictions[val, None], ids[val]).mean())})
    if not np.isfinite(predictions).all():
        raise AssertionError("incomplete sensor OOF predictions")
    per_design = dict(zip(np.unique(ids).tolist(), per_design_mae(y, predictions[:, None], ids)[:, 0].tolist()))
    near = np.abs(y) < .15
    return {"design_mae": float(np.mean(list(per_design.values()))),
            "clip_mae": float(np.mean(np.abs(predictions - y))),
            "near_boundary_design_mae": float(per_design_mae(y[near], predictions[near, None], ids[near]).mean()),
            "per_design_mae": per_design, "oof_predictions": predictions.tolist(),
            "folds": folds, "identification": diagnostic_summary(observed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="p4_dataset_saturation")
    parser.add_argument("--candidates", default="p4_sensor_candidates")
    parser.add_argument("--output", default="p4_groups/sensor_layout")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source, candidate, output = Path(args.source), Path(args.candidates), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "feature_shards"
    cache.mkdir(exist_ok=True)
    manifest = json.loads((candidate / "manifest.json").read_text())
    source_meta = json.loads((source / "metadata.json").read_text())
    designs = np.array(sorted(source_meta["train_designs"]))
    if designs.tolist() != manifest["training_designs"]:
        raise ValueError("candidate/source training designs disagree")
    a = dict(np.load(source / "metadata_arrays.npz"))
    train_rows = np.flatnonzero(np.isin(a["design_ids"], designs))
    y, ids = a["margins"][train_rows].astype(float), a["design_ids"][train_rows]
    grams, covered, max_replay = [], [], 0.0
    for did in designs:
        with np.load(candidate / f"{did}.npz") as shard:
            if shard["fingerprint"].item() != manifest["fingerprint"]:
                raise ValueError(f"candidate fingerprint mismatch: {did}")
            grams.append(shard["gram"])
            covered.extend(shard["source_indices"].tolist())
            max_replay = max(max_replay, float(shard["max_replay_error"]))
    if sorted(covered) != train_rows.tolist():
        raise ValueError("candidate coverage is incomplete, duplicated, or outside training")
    plan, final, layouts = make_plan(ids, designs, np.stack(grams), manifest["original_indices"])
    positions = np.asarray(manifest["candidate_grid_yx"])
    source_paths = [Path(__file__), ROOT / "experiment_p4_ridge_cv.py",
                    ROOT / "src/mechanics/p4_margin_estimation/modal_features.py",
                    ROOT / "src/mechanics/p4_margin_estimation/baselines.py"]
    protocol = {"candidate_fingerprint": manifest["fingerprint"], "noise_levels": list(NOISE),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "layouts": layouts, "outer_folds": 5, "inner_folds": 3,
                "selection": "mean training-design logdet; pair-scaled 51-sample modal Gramian; regularizer 1e-3",
                "alpha_grid": ALPHAS.tolist(), "seed": 42, "noise_seed": 928,
                "normalization": "per-channel first 51 samples; sensor cap +/-50",
                "noise": "Gaussian before normalization; sigma fraction of original-eight raw calibration RMS; shared candidate noise",
                "interpretation": "controlled noise sensitivity, not calibrated real sensor noise",
                "held_out_selection_calibration_test_scored": False,
                "bootstrap": "paired OOF design resampling; excludes refit uncertainty; exploratory"}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    results_path = output / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        if results["fingerprint"] != fingerprint:
            raise ValueError("sensor experiment resume fingerprint mismatch")
    else:
        results = {"fingerprint": fingerprint, "protocol": protocol,
                   "n_designs": len(designs), "n_clips": len(y),
                   "max_acquisition_replay_error": max_replay,
                   "training_source_indices": train_rows.tolist(),
                   "training_design_ids": ids.tolist(), "fold_plan": plan,
                   "full_training_layouts": {name: {"indices": layout,
                       "grid_yx": positions[layout].tolist(),
                       "fraction_yx": (positions[layout] / 31).tolist()} for name, layout in final.items()},
                   "records": {}}
        atomic_json(results_path, results)
    pending = []
    for did in designs:
        path = cache / f"{did}.npz"
        if path.exists():
            with np.load(path) as shard:
                if shard["fingerprint"].item() != fingerprint:
                    raise ValueError(f"feature cache fingerprint mismatch: {did}")
        else:
            pending.append(str(did))
    print(f"Layouts: {len(layouts)} unique. Feature shards: {len(pending)} pending.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = [pool.submit(feature_shard, did, candidate, source, cache, layouts, fingerprint) for did in pending]
        for job in as_completed(jobs):
            print(f"Features saved: {job.result()}", flush=True)
    features = np.empty((len(NOISE), len(layouts), len(y), 22))
    diagnostics = np.empty((len(NOISE), len(layouts), len(y), 4))
    for did in designs:
        with np.load(cache / f"{did}.npz") as shard:
            rows = np.searchsorted(train_rows, shard["source_indices"])
            features[:, :, rows] = shard["features"]
            diagnostics[:, :, rows] = shard["recovery"]
    if not np.isfinite(features).all():
        raise ValueError("invalid cached ridge features")
    lookup = {tuple(layout): i for i, layout in enumerate(layouts)}
    for ni, noise in enumerate(NOISE):
        for transfer in ((False,) if ni == 0 else (False, True)):
            mode = "clean_transfer" if transfer else "matched"
            prefix = f"noise{noise:g}__{mode}"
            for arm in ARMS:
                key = prefix + "__" + arm
                if key not in results["records"]:
                    results["records"][key] = evaluate_arm(
                        arm, ni, features, diagnostics, y, ids, plan, lookup, transfer=transfer)
                    atomic_json(results_path, results)
                print(key, results["records"][key]["design_mae"], flush=True)
            baseline = results["records"][prefix + "__current8"]["per_design_mae"]
            for arm in ARMS:
                record = results["records"][prefix + "__" + arm]
                record["paired_vs_current8"] = paired_design_comparison(
                    record["per_design_mae"].items(), baseline.items(), n_bootstrap=10000, seed=42)
            results.setdefault("count_comparisons", {})[prefix] = {
                f"{left}_minus_{right}": paired_design_comparison(
                    results["records"][prefix + "__" + left]["per_design_mae"].items(),
                    results["records"][prefix + "__" + right]["per_design_mae"].items(),
                    n_bootstrap=10000, seed=42)
                for left, right in (("greedy8", "greedy4"), ("greedy16", "greedy8"))}
            atomic_json(results_path, results)
    results["complete"] = True
    atomic_json(results_path, results)
    print(f"SENSOR_STUDY_DONE {results_path}", flush=True)


if __name__ == "__main__":
    main()
