#!/usr/bin/env python3
"""P4 multi-velocity u_crit experiment: invert per-clip margins across velocities.

A clip at known velocity V with predicted margin m gives u_crit = V/(1-m).
Single-clip inversion is noise-dominated (~100 m/s); pooling K clips of one
(design, realization) unit at different velocities exposes the slope the
single-clip task cannot see. Compares GRU vs ridge per-clip margins under
identical pooling (median and least-squares V=u*(1-m) fit), plus a
train-mean baseline, all on design-split test units at cap 50.

No training: reuses frozen saturation checkpoints. Leakage guards: u_crit
never an input; velocity (always known in deployment) is the only auxiliary;
units never split across partitions (inherited design splits); denom<=0.05
rejected before inversion.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import numpy as np

from experiment_p4_phase2 import _predict
from experiment_p4_saturation import (
    _atomic_json_dump,
    _fingerprint_dataset,
    _initial_results,
    _load_or_initialize_results,
    _load_ridge_checkpoint,
    _load_torch_checkpoint,
    _make_cap_parts,
    _relative_path,
    _save_predictions,
    _sha256_file,
    _validate_dataset,
    _validate_seeds,
    MODEL_NAME,
)
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from train_p4_expanded import load_dataset

SAT_DIR = "p4_saturation_full"
CAP = 50.0
DENOM_FLOOR = 0.05


def _unit_ucrit(bundle: dict[str, Any]) -> dict[tuple[str, int], float]:
    _, _, _, design_ids, realization_ids, meta = load_dataset(str(bundle["dataset_dir"]))
    ucrits = np.asarray(meta["provenance"]["u_crits"], dtype=float)
    out: dict[tuple[str, int], float] = {}
    for did, rid, u in zip([str(x) for x in design_ids],
                           [int(x) for x in realization_ids], ucrits.tolist()):
        key = (did, int(rid))
        if key in out and out[key] != float(u):
            raise ValueError(f"u_crit varies within {key}")
        out[key] = float(u)
    return out


def _invert(V: float, m: float) -> float | None:
    d = 1.0 - float(m)
    if not (np.isfinite(d) and d > DENOM_FLOOR) or not np.isfinite(V):
        return None
    return float(V) / d


def _fit_u(Vs: np.ndarray, ms: np.ndarray) -> float | None:
    Vs = np.asarray(Vs, dtype=float)
    d = 1.0 - np.asarray(ms, dtype=float)
    ok = np.isfinite(d) & (d > DENOM_FLOOR) & np.isfinite(Vs)
    if ok.sum() < 2:
        return None
    denom = float(Vs[ok] @ d[ok])
    if not np.isfinite(denom) or denom == 0.0:
        return None
    return float((Vs[ok] @ Vs[ok]) / denom)


def _unit_table(names: list[str], per_unit: dict[str, list[float]]) -> dict[str, Any]:
    return {n: {"n_units": len(per_unit[n]), "mae": float(np.mean(per_unit[n]))} for n in names}


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(int(s) for s in _validate_seeds(args.seeds))
    bundle = _validate_dataset(args.dataset)
    sat_dir = Path(args.sat_dir)
    config = {
        "cap": CAP,
        "seeds": [int(s) for s in seeds],
        "denom_floor": DENOM_FLOOR,
        "pooling": ["median", "lsq_fit"],
        "sat_dir": str(sat_dir),
        "model": MODEL_NAME,
        "bootstrap": int(args.bootstrap),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records: dict[str, Any] = results["records"]

    parts = _make_cap_parts(bundle, CAP, need_clean=True)
    truth = _unit_ucrit(bundle)
    units: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {"clean": [], "raw": [], "V": []})
    for c, r in zip(parts["test"]["clean"], parts["test"]["raw"]):
        u = (str(c.design_id), int(c.realization_idx))
        units[u]["clean"].append(c)
        units[u]["raw"].append(r)
        units[u]["V"].append(float(c.velocity))
    unit_ids = sorted(units)
    y_true = np.asarray([truth[u] for u in unit_ids], dtype=float)

    train_units = sorted(set(
        (str(bundle["design_ids"][i]), int(bundle["realization_ids"][i]))
        for i in range(len(bundle["design_ids"]))
        if str(bundle["design_ids"][i]) in bundle["splits"]["train"]))
    train_mean = float(np.mean([truth[u] for u in train_units]))

    ridge = _load_ridge_checkpoint(sat_dir / "checkpoints" / "physics_ridge_traincap50.npz")
    ridge_raw = np.asarray(ridge.predict(parts["test"]["raw"]), dtype=float).ravel()
    test_raw = parts["test"]["raw"]
    ridge_per_clip: dict[tuple[str, int], list[float]] = defaultdict(list)
    key_order: list[tuple[str, int]] = []
    for c, m in zip(test_raw, ridge_raw):
        u = (str(c.design_id), int(c.realization_idx))
        if not ridge_per_clip[u]:
            key_order.append(u)
        ridge_per_clip[u].append(float(m))

    per_unit: dict[str, list[float]] = {
        "gru_median": [], "gru_fit": [], "ridge_median": [], "ridge_fit": [],
        "train_mean": [],
    }
    per_unit_true: list[float] = []
    pred_matrix: dict[str, list[float]] = {
        "gru_median": [], "gru_fit": [], "ridge_median": [], "ridge_fit": [],
    }
    for seed in seeds:
        gru = _load_torch_checkpoint(
            sat_dir / "checkpoints" / f"gru_ell_seed{int(seed)}_traincap50.pt",
            int(seed), device="cpu")
        medians: list[float] = []
        fits: list[float] = []
        for u in unit_ids:
            d = units[u]
            gm = _predict(gru, d["clean"], batch=int(args.predict_batch), device="cpu")[:, 1]
            ests = [_invert(V, m) for V, m in zip(d["V"], gm)]
            ests = [e for e in ests if e is not None]
            if ests:
                medians.append(float(np.median(ests)))
            f = _fit_u(np.asarray(d["V"]), np.asarray(gm, dtype=float))
            if f is not None:
                fits.append(f)
        key_m = f"gru_ell_quantile__seed{int(seed)}__median"
        key_f = f"gru_ell_quantile__seed{int(seed)}__lsq_fit"
        for key, vals in ((key_m, medians), (key_f, fits)):
            err = np.abs(np.asarray(vals) - y_true[:len(vals)])
            payload_np = {"y_true": y_true[:len(vals)], "pred": np.asarray(vals),
                          "abs_err": err,
                          "unit_ids": np.asarray([f"{a}:{b}" for a, b in unit_ids[:len(vals)]])}
            pred_path = output_dir / "predictions" / f"{key}.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            digest = _save_predictions(pred_path, payload_np)
            records[key] = {
                "status": "complete", "model": MODEL_NAME, "seed": int(seed),
                "cap": CAP, "pooling": "median" if "median" in key else "lsq_fit",
                "n_units": len(vals), "mae_physical_ms": float(err.mean()),
                "train_mean_baseline_ms": float(np.mean(np.abs(y_true - train_mean))),
                "checkpoint": str(sat_dir / "checkpoints" / f"gru_ell_seed{int(seed)}_traincap50.pt"),
                "checkpoint_sha256": _sha256_file(
                    sat_dir / "checkpoints" / f"gru_ell_seed{int(seed)}_traincap50.pt"),
                "prediction_file": _relative_path(pred_path, output_dir),
                "prediction_sha256": digest,
            }
        per_unit["train_mean"].extend(float(abs(t - train_mean)) for t in y_true)
        _ = per_unit_true
        pred_matrix[f"gru_median_s{seed}"] = medians
        pred_matrix[f"gru_fit_s{seed}"] = fits
        del gru
    # Ridge pooling is seed-independent.
    ridge_meds, ridge_fits = [], []
    for u in unit_ids:
        ests = [_invert(V, m) for V, m in zip(units[u]["V"], ridge_per_clip[u])]
        ests = [e for e in ests if e is not None]
        if ests:
            ridge_meds.append(float(np.median(ests)))
        f = _fit_u(np.asarray(units[u]["V"]), np.asarray(ridge_per_clip[u]))
        if f is not None:
            ridge_fits.append(f)
    for key, vals in (("physics_feature_ridge__median", ridge_meds),
                      ("physics_feature_ridge__lsq_fit", ridge_fits)):
        err = np.abs(np.asarray(vals) - y_true[:len(vals)])
        payload_np = {"y_true": y_true[:len(vals)], "pred": np.asarray(vals), "abs_err": err,
                      "unit_ids": np.asarray([f"{a}:{b}" for a, b in unit_ids[:len(vals)]])}
        pred_path = output_dir / "predictions" / f"{key}.npz"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        digest = _save_predictions(pred_path, payload_np)
        records[key] = {
            "status": "complete", "model": "physics_feature_ridge", "seed": None,
            "cap": CAP, "pooling": "median" if "median" in key else "lsq_fit",
            "n_units": len(vals), "mae_physical_ms": float(err.mean()),
            "checkpoint": str(sat_dir / "checkpoints" / "physics_ridge_traincap50.npz"),
            "checkpoint_sha256": _sha256_file(
                sat_dir / "checkpoints" / "physics_ridge_traincap50.npz"),
            "prediction_file": _relative_path(pred_path, output_dir),
            "prediction_sha256": digest,
        }
    mean_gru_med = np.mean([pred_matrix[f"gru_median_s{s}"] for s in seeds], axis=0)
    mean_gru_fit = np.mean([pred_matrix[f"gru_fit_s{s}"] for s in seeds], axis=0)
    comparisons: dict[str, Any] = {}
    pairs = [
        ("gru_mean_seed_median_minus_ridge_median", mean_gru_med, np.asarray(ridge_meds)),
        ("gru_mean_seed_fit_minus_ridge_fit", mean_gru_fit, np.asarray(ridge_fits)),
        ("gru_mean_seed_median_minus_train_mean",
         mean_gru_med, np.full_like(mean_gru_med, train_mean)),
    ]
    for name, a_vals, b_vals in pairs:
        n = min(len(a_vals), len(b_vals))
        a = [(f"u{i}", float(abs(a_vals[i] - y_true[i]))) for i in range(n)]
        b = [(f"u{i}", float(abs(b_vals[i] - y_true[i]))) for i in range(n)]
        comparisons[name] = {
            "left": name.split("_minus_")[0], "right": name.split("_minus_")[1],
            "metric": "unit_abs_err", "subtraction": "left - right",
            "status": "complete",
            "result": paired_design_comparison(a, b, n_bootstrap=int(args.bootstrap), seed=0),
        }
    results["comparisons"] = comparisons
    results["summary"] = {
        "n_test_units": len(unit_ids),
        "train_mean_baseline_ms": float(np.mean(np.abs(y_true - train_mean))),
        "gru_median_ms": float(np.mean(np.abs(mean_gru_med - y_true))),
        "gru_fit_ms": float(np.mean(np.abs(mean_gru_fit - y_true))),
        "ridge_median_ms": float(np.mean(np.abs(np.asarray(ridge_meds) - y_true))),
        "ridge_fit_ms": float(np.mean(np.abs(np.asarray(ridge_fits) - y_true))),
    }
    _atomic_json_dump(results, results_path)
    print(f"MULTIVEL_DONE units={len(unit_ids)} output={output_dir}", flush=True)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P4 multi-velocity u_crit inversion")
    p.add_argument("--dataset", "--dataset-dir", dest="dataset",
                   default=os.environ.get("P4_DATASET_DIR", "p4_dataset"))
    p.add_argument("--output-dir", "--out", dest="output_dir", default="p4_multivel_run")
    p.add_argument("--sat-dir", default=SAT_DIR)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--predict-batch", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
