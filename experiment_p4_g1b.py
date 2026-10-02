#!/usr/bin/env python3
"""G1b confound split: clean-train vs DR-train, with and without log-dt.

Completes the 2x2 around G0/G1 (additive only, new file):
  G0 (existing, p4_saturation_full): DR-train, no sidecar
  G1 (existing, p4_groups/g1):       clean-train + log_dt sidecar
  G1b-A (new): clean-train, plain GRU (no sidecar)
  G1b-B (new): DR-train + log_dt sidecar

Same cap-50 data, splits, trainer, epochs, seeds, eval/corr/slope reporting
as G1. DR uses apply_dr_with_mask seed 0 (identical to G0 training data).
Sidecar stats fit on clean train (same as G1); dt is preserved by DR so the
same stats apply to DR rows. Outputs to --output-dir (default p4_groups/g1b).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import experiment_p4_saturation as _sat
from experiment_p4_phase2 import (
    GRUEllQuantile, GRUEllQuantileSidecar, _predict, _predict_sidecar,
    train_quantile_generic, train_quantile_sidecar,
)
from experiment_p4_saturation import (
    _atomic_json_dump, _fingerprint_dataset, _load_or_initialize_results,
    _make_cap_parts, _neural_record, _record_is_complete, _relative_path,
    _ridge_record, _save_torch_checkpoint, _validate_dataset, _validate_seeds,
    ALPHA, MODEL_NAME, N_MODEL_CHANNELS,
)
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from mechanics.p4_margin_estimation.quantile_head import (
    apply_cqr_adjustment, fit_cqr_adjustment,
)
from mechanics.p4_margin_estimation import sidecar as _sc
from train_p4_expanded import apply_dr_with_mask, with_ones_mask

GROUP = "G1b"
CAP = 50.0
ARMS = ("plain_clean", "sidecar_dr")


def _corr_slope(y, med):
    y = np.asarray(y, dtype=float).ravel()
    med = np.asarray(med, dtype=float).ravel()
    ok = np.isfinite(y) & np.isfinite(med)
    y, med = y[ok], med[ok]
    if len(y) < 3 or med.std() == 0:
        return {"corr": float("nan"), "slope": float("nan"),
                "intercept": float("nan"), "n": int(len(y))}
    slope, intercept = np.polyfit(med, y, 1)
    return {"corr": float(np.corrcoef(med, y)[0, 1]), "slope": float(slope),
            "intercept": float(intercept), "n": int(len(y))}


def run(args):
    seeds = _validate_seeds(args.seeds)
    torch.set_num_threads(int(args.threads))
    bundle = _validate_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["log_dt"]
    config = {"group": GROUP, "arms": list(ARMS), "members": names,
              "cap": CAP, "seeds": [int(s) for s in seeds],
              "epochs": int(args.epochs), "batch_size": int(args.batch_size),
              "predict_batch": int(args.predict_batch), "threads": int(args.threads),
              "alpha": ALPHA, "model": "gru_plain_and_sidecar",
              "bootstrap": int(args.bootstrap)}
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records = results["records"]
    _sat._COMPARISON_OUTPUT_DIR = output_dir

    expected = []
    for arm in ARMS:
        for mode in ("fixed_transfer", "matched_retrain"):
            for seed in seeds:
                expected.append((f"{GROUP}__{arm}__{mode}__seed{int(seed)}",
                                 arm, mode, seed))
    pending = [e for e in expected if not _record_is_complete(records.get(e[0]), output_dir)]
    if not pending:
        print("G1b complete; nothing pending.", flush=True)
        return results

    parts = _make_cap_parts(bundle, CAP, need_clean=True)
    train_raw = parts["train"]["raw"]
    train_clean = with_ones_mask(train_raw)
    train_dr, n_ch = apply_dr_with_mask(train_raw, seed=0)
    assert n_ch == N_MODEL_CHANNELS, (n_ch, N_MODEL_CHANNELS)
    select_clean = parts["select"]["clean"]
    calib_parts = {"raw": parts["calib"]["raw"], "clean": parts["calib"]["clean"], "cap": CAP}
    test_clean = parts["test"]["clean"]

    stats = _sc.standardize_fit(_sc.extract(train_clean, names))
    clean_side = lambda clips: _sc.standardize_apply(_sc.extract(clips, names), stats)
    train_side = clean_side(train_clean)
    dr_side = clean_side(train_dr)  # dt preserved by DR; same stats
    sel_side = clean_side(select_clean)
    cal_side = clean_side(calib_parts["clean"])
    test_side = clean_side(test_clean)

    artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    need_plain = any(a == "plain_clean" for _, a, _, _ in pending)
    need_side = any(a == "sidecar_dr" for _, a, _, _ in pending)
    if need_plain:
        for seed in seeds:
            if not any(a == "plain_clean" and s == seed for _, a, _, s in pending):
                continue
            t0 = time.time()
            model, history = train_quantile_generic(
                lambda: GRUEllQuantile(N_MODEL_CHANNELS), train_clean,
                val_clips=select_clean, epochs=int(args.epochs), lr=1e-3,
                batch_size=int(args.batch_size), seed=int(seed),
                n_channels=N_MODEL_CHANNELS, device=args.device)
            ckpt = output_dir / "checkpoints" / f"g1b_plain_clean_seed{int(seed)}_traincap50.pt"
            digest = _save_torch_checkpoint(model, ckpt)
            model.eval()
            art = {"model": model, "path": ckpt, "sha256": digest, "history": history,
                   "train_cap": CAP, "train_time": time.time() - t0,
                   "predict_batch": int(args.predict_batch), "plain": True}
            cpred = np.asarray(_predict(model, calib_parts["clean"],
                                        batch=args.predict_batch, device=args.device),
                               dtype=float)
            cy = np.asarray([float(c.margin) for c in calib_parts["clean"]], dtype=float)
            if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
                raise ValueError("G1b plain calibration predictions invalid")
            art["cqr"] = {"adjustment": float(fit_cqr_adjustment(
                cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)),
                "cap": CAP, "calib_n": int(len(cy)),
                "calib_designs": int(len(set(str(c.design_id) for c in calib_parts["clean"])))}
            artifacts[("plain_clean", seed)] = art
    if need_side:
        for seed in seeds:
            if not any(a == "sidecar_dr" and s == seed for _, a, _, s in pending):
                continue
            t0 = time.time()
            factory = lambda: GRUEllQuantileSidecar(N_MODEL_CHANNELS, n_side=1)  # noqa: E731
            model, history = train_quantile_sidecar(
                factory, train_dr, dr_side, val_clips=select_clean, val_side=sel_side,
                epochs=int(args.epochs), lr=1e-3, batch_size=int(args.batch_size),
                seed=int(seed), n_channels=N_MODEL_CHANNELS, device=args.device)
            ckpt = output_dir / "checkpoints" / f"g1b_sidecar_dr_seed{int(seed)}_traincap50.pt"
            digest = _save_torch_checkpoint(model, ckpt)
            model.eval()
            art = {"model": model, "path": ckpt, "sha256": digest, "history": history,
                   "train_cap": CAP, "train_time": time.time() - t0,
                   "predict_batch": int(args.predict_batch), "plain": False}
            cpred = np.asarray(_predict_sidecar(model, calib_parts["clean"], cal_side,
                                                batch=args.predict_batch, device=args.device),
                               dtype=float)
            cy = np.asarray([float(c.margin) for c in calib_parts["clean"]], dtype=float)
            if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
                raise ValueError("G1b sidecar calibration predictions invalid")
            art["cqr"] = {"adjustment": float(fit_cqr_adjustment(
                cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)),
                "cap": CAP, "calib_n": int(len(cy)),
                "calib_designs": int(len(set(str(c.design_id) for c in calib_parts["clean"])))}
            artifacts[("sidecar_dr", seed)] = art
    del train_dr

    for key, arm, mode, seed in pending:
        if _record_is_complete(records.get(key), output_dir):
            continue
        art = artifacts.get((arm, int(seed)))
        if art is None:
            raise RuntimeError(f"artifact unavailable for {key}")
        extra = {"group": GROUP, "arm": arm, "members": names if "sidecar" in arm else [],
                 "sidecar_stats": stats if "sidecar" in arm else None,
                 "train_data": "clean" if "clean" in arm else "dr"}
        if art.get("plain"):
            _neural_record(key=key, results=results, output_dir=output_dir,
                           artifact=art, parts=parts, cap=CAP, mode=mode,
                           seed=int(seed), calibration=art.get("cqr", {}),
                           calibration_parts=calib_parts, args=args)
        else:
            pred = np.asarray(_predict_sidecar(art["model"], test_clean, test_side,
                                               batch=int(args.predict_batch),
                                               device=args.device), dtype=float)
            if pred.ndim != 2 or pred.shape != (len(test_clean), 3):
                raise ValueError("G1b test prediction shape invalid")
            y = np.asarray([float(c.margin) for c in test_clean], dtype=float)
            raw_lo, med, raw_hi = pred[:, 0], pred[:, 1], pred[:, 2]
            adjustment = float(dict(art.get("cqr", {}))["adjustment"])
            cqr_lo, cqr_hi = apply_cqr_adjustment(raw_lo, raw_hi, adjustment)
            cal = dict(art.get("cqr", {}))
            cal.update({"adjustment": adjustment,
                        "method": "fit_cqr_adjustment_finite_sample_higher", "alpha": ALPHA})
            dids = [str(c.design_id) for c in test_clean]
            report = _sat._report(y, med, raw_lo, raw_hi, cqr_lo, cqr_hi,
                                  parts["test"]["saturated"], parts["test"]["growth"], dids)
            payload_np = {"y_true": y, "predicted_quantiles": pred,
                          "raw_lower": raw_lo, "median": med, "raw_upper": raw_hi,
                          "cqr_lower": cqr_lo, "cqr_upper": cqr_hi,
                          "saturated": np.asarray(parts["test"]["saturated"], dtype=bool),
                          "growth_clamped": np.asarray(parts["test"]["growth"], dtype=bool),
                          "design_ids": np.asarray(dids, dtype="U"),
                          "velocities": np.asarray([float(c.velocity) for c in test_clean],
                                                   dtype=np.float32)}
            _sat._store_record(key=key, results=results, output_dir=output_dir,
                               model_name=MODEL_NAME + "_sidecar", seed=int(seed), cap=CAP,
                               mode=mode, artifact=art, report=report,
                               prediction_payload=payload_np, calibration=cal,
                               train_cap=CAP, calibration_cap=CAP, args=args,
                               extra={"prediction_batch": int(args.predict_batch)})
        records[key].update(extra)
        pred = np.load(output_dir / records[key]["prediction_file"])
        records[key]["test_corr_slope"] = _corr_slope_local(
            np.asarray(pred["y_true"]), np.asarray(pred["median"]))
        _atomic_json_dump(results, results_path)
    del parts, calib_parts, artifacts

    comparisons = {}
    for arm in ARMS:
        for seed in seeds:
            left = f"{GROUP}__{arm}__matched_retrain__seed{int(seed)}"
            a = (results["records"].get(left) or {}).get("per_design_mae")
            # paired against G0 DR baseline stored in saturation results
            g0 = _g0_per_design(seed)
            item = {"left": left, "right": f"G0_saturation_seed{int(seed)}",
                    "metric": "mae", "subtraction": "left - right (first-named minus second-named)"}
            if isinstance(a, dict) and isinstance(g0, dict) and a and sorted(a) == sorted(g0):
                item["status"] = "complete"
                item["result"] = paired_design_comparison(
                    sorted(a.items()), sorted(g0.items()),
                    n_bootstrap=args.bootstrap, seed=int(seed))
            else:
                item["status"] = "incomplete (per-design MAE missing)"
            comparisons[f"{left}_minus_G0"] = item
    results["comparisons"] = comparisons
    _atomic_json_dump(results, results_path)
    print(f"G1b_DONE records={len(results['records'])} output={output_dir}", flush=True)
    return results


def _corr_slope_local(y, med):
    y = np.asarray(y, dtype=float).ravel()
    med = np.asarray(med, dtype=float).ravel()
    ok = np.isfinite(y) & np.isfinite(med)
    y, med = y[ok], med[ok]
    if len(y) < 3 or med.std() == 0:
        return {"corr": float("nan"), "slope": float("nan"),
                "intercept": float("nan"), "n": int(len(y))}
    slope, intercept = np.polyfit(med, y, 1)
    return {"corr": float(np.corrcoef(med, y)[0, 1]), "slope": float(slope),
            "intercept": float(intercept), "n": int(len(y))}


def _g0_per_design(seed):
    import json as _json
    try:
        r = _json.load(open(_ROOT / "p4_saturation_full" / "results.json"))
        rec = r["records"].get(f"gru_ell_quantile__cap50__fixed_transfer__seed{int(seed)}")
        if rec and isinstance(rec.get("per_design_mae"), dict):
            return dict(rec["per_design_mae"])
    except (OSError, ValueError, KeyError):
        pass
    return {}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="G1b: DR confound split 2x2")
    p.add_argument("--dataset", "--dataset-dir", dest="dataset",
                   default=os.environ.get("P4_DATASET_DIR", "p4_dataset"))
    p.add_argument("--output-dir", "--out", dest="output_dir", default="p4_groups/g1b")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args(argv)


def main(argv=None):
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
