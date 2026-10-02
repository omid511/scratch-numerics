#!/usr/bin/env python3
"""G1 experiment: fair timebase (additive only).

GRU gets log_dt sidecar scalar; ridge unchanged (already uses dt).
New files only for group logic; shared code lives in sidecar.py and the
Sidecar model/trainer variants in experiment_p4_phase2.py. Existing
runners/models untouched.

Outputs to --output-dir (default p4_groups/g1). Records carry group G1,
member list, sidecar stats, and test corr/slope. Member-split rule: if the
group delta vs G0 exceeds the design-bootstrap CI, rerun member-ablated
variants before advancing (here single member, nothing to split).
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
    GRUEllQuantileSidecar, _predict_sidecar, _stack_valid_with_clips,
    train_quantile_sidecar,
)
from experiment_p4_saturation import (
    _atomic_json_dump, _fingerprint_dataset, _load_or_initialize_results,
    _make_cap_parts, _neural_record, _record_is_complete, _relative_path,
    _ridge_record, _save_torch_checkpoint, _validate_dataset, _validate_seeds,
    ALPHA, MODEL_NAME, N_MODEL_CHANNELS,
)
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from mechanics.p4_margin_estimation.quantile_head import fit_cqr_adjustment
from mechanics.p4_margin_estimation import sidecar as _sc

GROUP = "G4"
CAP = 50.0
ABLATION = "g4_proxy"


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
    cli_members = getattr(args, "members", None)
    names = [m for m in (cli_members.split(",") if cli_members else _sc.member_names(GROUP)) if m]
    if not names:
        raise ValueError("empty member list")
    tag = getattr(args, "tag", None) or "+".join(names)
    ablation = f"g4_{tag}"
    group = f"{GROUP}[{tag}]"
    config = {"group": group, "members": names, "ablation": ablation,
              "cap": CAP, "seeds": [int(s) for s in seeds],
              "epochs": int(args.epochs), "batch_size": int(args.batch_size),
              "predict_batch": int(args.predict_batch), "threads": int(args.threads),
              "alpha": ALPHA, "model": "gru_sidecar", "n_side": len(names),
              "bootstrap": int(args.bootstrap)}
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records = results["records"]
    _sat._COMPARISON_OUTPUT_DIR = output_dir

    expected = []
    for mode in ("fixed_transfer", "matched_retrain"):
        expected.append((f"{group}__physics_feature_ridge__{mode}__shared", "ridge", mode, None))
        for seed in seeds:
            expected.append((f"{group}__{MODEL_NAME}_sidecar__{mode}__seed{int(seed)}",
                             "gru", mode, seed))
    pending = [e for e in expected if not _record_is_complete(records.get(e[0]), output_dir)]
    if not pending:
        print("G4 complete; nothing pending.", flush=True)
        return results

    parts = _make_cap_parts(bundle, CAP, need_clean=True)
    train_raw = parts["train"]["raw"]
    # G1: train on clean (ones-mask) clips to isolate the sidecar effect;
    # DR variants arrive in later groups on top of this exact path.
    from train_p4_expanded import with_ones_mask
    train_clean = with_ones_mask(train_raw)
    select_clean = parts["select"]["clean"]
    calib_parts = {"raw": parts["calib"]["raw"], "clean": parts["calib"]["clean"], "cap": CAP}

    raw_side = _sc.extract(train_clean, names)
    stats = _sc.standardize_fit(raw_side)
    train_side = _sc.standardize_apply(raw_side, stats)
    sel_side = _sc.standardize_apply(_sc.extract(select_clean, names), stats)
    cal_side = _sc.standardize_apply(_sc.extract(calib_parts["clean"], names), stats)
    test_side_raw = _sc.extract(parts["test"]["clean"], names)
    test_side = _sc.standardize_apply(test_side_raw, stats)

    # G4 ridge parity: energy/sat_frac are new to ridge (slope/freq/bandwidth
    # already owned); velocity owned iff use_velocity. Fit both variants and
    # record both; the 7-feature main-study ridge stays the reference row.
    from experiment_p4_saturation import _ensure_ridge, _ridge_adjustment
    from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
    ridge_artifact = _ensure_ridge(output_dir=output_dir, records={}, train_cap=CAP,
                                   train_raw=train_raw)
    ridge_v = PhysicsFeatureRidge(use_velocity=True)
    ridge_v.fit(train_raw)
    ridge_artifact["calibration"] = _ridge_adjustment(ridge_artifact["model"], calib_parts["raw"])
    want_ridge = output_dir / "checkpoints" / f"{ablation}_physics_ridge_traincap50.npz"
    got_ridge = Path(str(ridge_artifact["path"]))
    if got_ridge.resolve() != want_ridge.resolve():
        want_ridge.parent.mkdir(parents=True, exist_ok=True)
        tmp = want_ridge.with_name(want_ridge.name + ".tmp")
        shutil.copyfile(got_ridge, tmp)
        os.replace(tmp, want_ridge)
        ridge_artifact = dict(ridge_artifact)
        ridge_artifact["path"] = want_ridge
        ridge_artifact["sha256"] = _sat._sha256_file(want_ridge)

    artifacts: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        t0 = time.time()
        factory = lambda _ns=len(names): GRUEllQuantileSidecar(N_MODEL_CHANNELS, n_side=_ns)  # noqa: E731
        model, history = train_quantile_sidecar(
            factory, train_clean, train_side, val_clips=select_clean, val_side=sel_side,
            epochs=int(args.epochs), lr=1e-3, batch_size=int(args.batch_size),
            seed=int(seed), n_channels=N_MODEL_CHANNELS, device=args.device)
        ckpt = output_dir / "checkpoints" / f"{ablation}_gru_ell_seed{int(seed)}_traincap50.pt"
        digest = _save_torch_checkpoint(model, ckpt)
        model.eval()
        art = {"model": model, "path": ckpt, "sha256": digest, "history": history,
               "train_cap": CAP, "train_time": time.time() - t0,
               "predict_batch": int(args.predict_batch)}
        cpred = np.asarray(_predict_sidecar(model, calib_parts["clean"], cal_side,
                                            batch=args.predict_batch, device=args.device),
                           dtype=float)
        cy = np.asarray([float(c.margin) for c in calib_parts["clean"]], dtype=float)
        if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
            raise ValueError("G4 calibration predictions invalid")
        art["cqr"] = {"adjustment": float(fit_cqr_adjustment(
            cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)),
            "cap": CAP, "calib_n": int(len(cy)),
            "calib_designs": int(len(set(str(c.design_id) for c in calib_parts["clean"])))}
        artifacts[seed] = art

    for key, kind, mode, seed in pending:
        if _record_is_complete(records.get(key), output_dir):
            continue
        extra = {"group": group, "members": names, "sidecar_stats": stats, "ablation": ablation}
        if kind == "ridge":
            _ridge_record(key=key, results=results, output_dir=output_dir,
                          artifact=ridge_artifact, parts=parts, cap=CAP, mode=mode,
                          calibration=ridge_artifact.get("calibration", {}),
                          calibration_parts=calib_parts, args=args)
            records[key].update(extra)
        else:
            art = artifacts.get(int(seed))
            if art is None:
                raise RuntimeError(f"GRU artifact unavailable for {key}")
            # predict with sidecar rows for test clips
            test_clean = parts["test"]["clean"]
            pred = np.asarray(_predict_sidecar(art["model"], test_clean, test_side,
                                               batch=int(args.predict_batch),
                                               device=args.device), dtype=float)
            if pred.ndim != 2 or pred.shape != (len(test_clean), 3):
                raise ValueError("G4 test prediction shape invalid")
            y = np.asarray([float(c.margin) for c in test_clean], dtype=float)
            raw_lo, med, raw_hi = pred[:, 0], pred[:, 1], pred[:, 2]
            cal = dict(art.get("cqr", {}))
            adjustment = float(cal["adjustment"])
            from mechanics.p4_margin_estimation.quantile_head import apply_cqr_adjustment
            cqr_lo, cqr_hi = apply_cqr_adjustment(raw_lo, raw_hi, adjustment)
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
            from experiment_p4_saturation import _store_record
            _store_record(key=key, results=results, output_dir=output_dir,
                          model_name=MODEL_NAME + "_sidecar", seed=int(seed), cap=CAP,
                          mode=mode, artifact=art, report=report,
                          prediction_payload=payload_np, calibration=cal,
                          train_cap=CAP, calibration_cap=CAP, args=args,
                          extra={"prediction_batch": int(args.predict_batch)})
            records[key].update(extra)
        pred = np.load(output_dir / records[key]["prediction_file"])
        records[key]["test_corr_slope"] = _corr_slope(np.asarray(pred["y_true"]),
                                                      np.asarray(pred["median"]))
        _atomic_json_dump(results, results_path)
    del parts, calib_parts, artifacts, ridge_artifact

    comparisons = {}
    for seed in seeds:
        left = f"{group}__{MODEL_NAME}_sidecar__matched_retrain__seed{int(seed)}"
        right = f"{group}__physics_feature_ridge__matched_retrain__shared"
        a = (results["records"].get(left) or {}).get("per_design_mae")
        b = (results["records"].get(right) or {}).get("per_design_mae")
        item = {"left": left, "right": right, "metric": "mae",
                "subtraction": "left - right (first-named minus second-named)"}
        if isinstance(a, dict) and isinstance(b, dict) and a and sorted(a) == sorted(b):
            item["status"] = "complete"
            item["result"] = paired_design_comparison(
                sorted(a.items()), sorted(b.items()),
                n_bootstrap=args.bootstrap, seed=int(seed))
        else:
            item["status"] = "incomplete (per-design MAE missing)"
        comparisons[f"{left}_minus_{right}"] = item
    results["comparisons"] = comparisons
    _atomic_json_dump(results, results_path)
    print(f"G1_DONE records={len(results['records'])} output={output_dir}", flush=True)
    return results


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="G4: observable design proxy (16 sidecar members)")
    p.add_argument("--dataset", "--dataset-dir", dest="dataset",
                   default=os.environ.get("P4_DATASET_DIR", "p4_dataset"))
    p.add_argument("--output-dir", "--out", dest="output_dir", default="p4_groups/g1")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--members", default=None,
                   help="comma-separated sidecar member subset (default: full GROUP set)")
    p.add_argument("--tag", default=None,
                   help="output name tag (default: members joined by +)")
    return p.parse_args(argv)


def main(argv=None):
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
