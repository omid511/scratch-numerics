#!/usr/bin/env python3
"""P4 preprocessing ablation: shared-gain DR (no per-channel gain/skew).

DR-only ablation on stored per-channel-normalized clips. Same cap-50 data,
splits, GRU trainer, ridge, targets (margin + damp), and reporting as the
main targets study; only the training augmentation changes: shared (common)
gain/bias/drift applied identically to all sensors, timing skew off. Eval
splits stay clean (ones mask) exactly as in the main study.

Preregistration (locked before seeing results):
- Margin MAE <= 0.04: plateau was preprocessing; spatial info mattered.
- Damping improves / margin flat ~0.05: modal info restored but margin
  needs design-specific mapping; denominator story strengthens.
- Neither moves: preprocessing-loss weakens substantially.
- Margin moves / damping flat: (alpha, omega)-story wrong; another
  spatial pathway at work.
- Report corr + calibration slope even if MAE flat (0.82 -> 0.90 at
  same MAE is not null).

Usage::

    PYTHONPATH=src python experiment_p4_preproc.py \
        --dataset p4_dataset_saturation --output-dir p4_preproc_run \
        --seeds 0 1 2 --epochs 20 --batch-size 64 --predict-batch 32 --threads 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import experiment_p4_saturation as _sat
from experiment_p4_phase2 import _predict
from experiment_p4_saturation import (
    _atomic_json_dump,
    _fingerprint_dataset,
    _load_or_initialize_results,
    _load_torch_checkpoint,
    _make_cap_parts,
    _neural_record,
    _record_is_complete,
    _relative_path,
    _ridge_record,
    _save_torch_checkpoint,
    _sha256_file,
    _validate_dataset,
    _validate_seeds,
    ALPHA,
    MODEL_NAME,
    N_MODEL_CHANNELS,
)
from experiment_p4_targets import (
    _apply_damp,
    _damp_map,
    _destandardize_ucrit_report,
    _target_label,
    _target_record_key,
)
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from mechanics.p4_margin_estimation.quantile_head import fit_cqr_adjustment
from train_p4_expanded import apply_dr_shared_gain

TARGETS = ("margin", "damp")
CAP = 50.0
ABLATION = "shared_gain_no_skew"


def _config(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "device": str(getattr(args, "device", "cpu")),
        "targets": list(TARGETS),
        "ablation": ABLATION,
        "cap": CAP,
        "seeds": [int(s) for s in seeds],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "predict_batch": int(args.predict_batch),
        "threads": int(args.threads),
        "alpha": ALPHA,
        "model": MODEL_NAME,
        "model_input_channels": N_MODEL_CHANNELS,
        "quantile_weights": [2.0, 1.0, 1.0],
        "train_dr_seed": 0,
        "bootstrap": int(args.bootstrap),
        "preregistration": {
            "margin_le_0.04": "plateau was preprocessing; spatial info mattered",
            "damp_up_margin_flat": "modal info restored but margin needs design mapping",
            "neither_moves": "preprocessing-loss weakens substantially",
            "margin_up_damp_flat": "(alpha,omega)-story wrong; another spatial pathway",
            "also_report": "corr + calibration slope even if MAE flat",
        },
    }


def _corr_slope(y: np.ndarray, med: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float).ravel()
    med = np.asarray(med, dtype=float).ravel()
    ok = np.isfinite(y) & np.isfinite(med)
    y, med = y[ok], med[ok]
    if len(y) < 3 or med.std() == 0:
        return {"corr": float("nan"), "slope": float("nan"), "intercept": float("nan"), "n": int(len(y))}
    slope, intercept = np.polyfit(med, y, 1)
    return {"corr": float(np.corrcoef(med, y)[0, 1]), "slope": float(slope),
            "intercept": float(intercept), "n": int(len(y))}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _validate_seeds(args.seeds)
    if args.epochs < 1 or args.batch_size < 1 or args.predict_batch < 1:
        raise ValueError("epochs, batch-size, and predict-batch must be positive")
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    device = torch.device(getattr(args, "device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.device = str(device)
    torch.set_num_threads(int(args.threads))
    bundle = _validate_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    for old in ("p4_saturation_full", "p4_targets_full", "p4_targets_damp",
                "p4_multivel_full", "p4_multivel_e2e", "p4_multivel_e2e_v2", "p4_multivel_e2e_vonly"):
        if output_dir.resolve() == (_ROOT / old).resolve():
            raise ValueError(f"--output-dir must not overwrite {old}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config(args, seeds)
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records: dict[str, Any] = results["records"]
    _sat._COMPARISON_OUTPUT_DIR = output_dir

    expected = []
    for target in TARGETS:
        for mode in ("fixed_transfer", "matched_retrain"):
            expected.append((_target_record_key(target, "physics_feature_ridge", mode, None),
                             "ridge", target, mode, None))
            for seed in seeds:
                expected.append((_target_record_key(target, MODEL_NAME, mode, seed),
                                 "gru", target, mode, seed))
    pending = [item for item in expected if not _record_is_complete(records.get(item[0]), output_dir)]
    damp_map = _damp_map(bundle) if any(item[2] == "damp" for item in pending) else None

    from experiment_p4_phase2 import train_quantile_generic, GRUEllQuantile
    from experiment_p4_saturation import (
        _ensure_ridge, _ridge_adjustment, _training_state_is_compatible,
        _load_training_state, _save_training_state, results_fingerprint,
        _training_state_path, _checkpoint_path,
    )

    for target in TARGETS:
        target_pending = [item for item in pending if item[2] == target]
        if not target_pending:
            continue
        print(f"Preparing {ABLATION}/{_target_label(target) if target != 'margin' else 'margin'} "
              f"({len(target_pending)} incomplete records)...", flush=True)
        parts = _make_cap_parts(bundle, CAP, need_clean=True)
        stats = None
        if target == "damp":
            assert damp_map is not None
            stats = _apply_damp(parts, damp_map, fit_stats=True)
            assert stats is not None
            _apply_damp(parts, damp_map, stats=stats)
        # margin target: stored margins untouched.
        calib_parts = {"raw": parts["calib"]["raw"], "clean": parts["calib"]["clean"], "cap": CAP}
        select_clean = parts["select"]["clean"]

        def _t_ckpt(kind: str, seed: int | None = None) -> Path:
            base = _checkpoint_path(output_dir, kind, CAP, seed)
            return base.parent / f"{ABLATION}_{target}_{base.name}"

        gru_artifacts: dict[int, dict[str, Any]] = {}
        ridge_artifact: dict[str, Any] | None = None
        train_dr, n_channels = apply_dr_shared_gain(parts["train"]["raw"], seed=0)
        assert n_channels == N_MODEL_CHANNELS, (n_channels, N_MODEL_CHANNELS)
        for seed in seeds:
            factory = lambda: GRUEllQuantile(N_MODEL_CHANNELS)  # noqa: E731
            t0 = __import__("time").time()
            model, history = train_quantile_generic(
                factory, train_dr, val_clips=select_clean, epochs=int(args.epochs),
                lr=1e-3, batch_size=int(args.batch_size), seed=int(seed),
                n_channels=N_MODEL_CHANNELS, device=args.device)
            digest = _save_torch_checkpoint(model, _t_ckpt("gru", seed))
            model.eval()
            art = {"model": model, "path": _t_ckpt("gru", seed), "sha256": digest,
                   "history": history, "train_cap": CAP, "train_time": __import__("time").time() - t0,
                   "predict_batch": int(args.predict_batch)}
            cpred = np.asarray(_predict(model, calib_parts["clean"],
                                        batch=args.predict_batch, device=args.device), dtype=float)
            cy = np.asarray([float(c.margin) for c in calib_parts["clean"]], dtype=float)
            if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
                raise ValueError(f"{target} calibration predictions are invalid")
            art["cqr"] = {"adjustment": float(fit_cqr_adjustment(
                cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)),
                "cap": CAP, "calib_n": int(len(cy)),
                "calib_designs": int(len(set(str(c.design_id) for c in calib_parts["clean"])))}
            gru_artifacts[seed] = art
        del train_dr
        ridge_artifact = _ensure_ridge(output_dir=output_dir, records={}, train_cap=CAP,
                                       train_raw=parts["train"]["raw"])
        ridge_artifact["calibration"] = _ridge_adjustment(ridge_artifact["model"], calib_parts["raw"])
        want_ridge = _t_ckpt("ridge")
        got_ridge = Path(str(ridge_artifact["path"]))
        if got_ridge.resolve() != want_ridge.resolve():
            import shutil
            want_ridge.parent.mkdir(parents=True, exist_ok=True)
            tmp = want_ridge.with_name(want_ridge.name + ".tmp")
            shutil.copyfile(got_ridge, tmp)
            os.replace(tmp, want_ridge)
            ridge_artifact = dict(ridge_artifact)
            ridge_artifact["path"] = want_ridge
            ridge_artifact["sha256"] = _sha256_file(want_ridge)

        for key, kind, _t, mode, seed in target_pending:
            if _record_is_complete(records.get(key), output_dir):
                continue
            extra: dict[str, Any] = {"ablation": ABLATION,
                                     "target": _target_label(target) if target != "margin" else "margin"}
            if target == "damp" and stats:
                extra["damp_standardization"] = dict(stats)
            if kind == "ridge":
                if ridge_artifact is None:
                    raise RuntimeError(f"ridge artifact unavailable for {key}")
                _ridge_record(key=key, results=results, output_dir=output_dir,
                              artifact=ridge_artifact, parts=parts, cap=CAP, mode=mode,
                              calibration=ridge_artifact.get("calibration", {}),
                              calibration_parts=calib_parts, args=args)
                records[key].update(extra)
            else:
                art = gru_artifacts.get(int(seed))
                if art is None:
                    raise RuntimeError(f"GRU artifact unavailable for {key}")
                _neural_record(key=key, results=results, output_dir=output_dir,
                               artifact=art, parts=parts, cap=CAP, mode=mode,
                               seed=int(seed), calibration=art.get("cqr", {}),
                               calibration_parts=calib_parts, args=args)
                records[key].update(extra)
            # corr + slope on test predictions (cheap, preregistered)
            pred = np.load(output_dir / records[key]["prediction_file"])
            cs = _corr_slope(np.asarray(pred["y_true"]), np.asarray(pred["median"]))
            records[key]["test_corr_slope"] = cs
            _atomic_json_dump(results, results_path)
        del parts, calib_parts, gru_artifacts, ridge_artifact

    comparisons: dict[str, Any] = {}
    for target in TARGETS:
        for seed in seeds:
            left = _target_record_key(target, MODEL_NAME, "matched_retrain", seed)
            right = _target_record_key(target, "physics_feature_ridge", "matched_retrain", None)
            name = f"{left}_minus_{right}"
            a = (results["records"].get(left) or {}).get("per_design_mae")
            b = (results["records"].get(right) or {}).get("per_design_mae")
            item: dict[str, Any] = {"left": left, "right": right, "metric": "mae",
                                    "subtraction": "left - right (first-named minus second-named)"}
            if isinstance(a, dict) and isinstance(b, dict) and a and sorted(a) == sorted(b):
                item["status"] = "complete"
                item["result"] = paired_design_comparison(
                    sorted(a.items()), sorted(b.items()),
                    n_bootstrap=args.bootstrap, seed=int(seed))
            else:
                item["status"] = "incomplete (per-design MAE missing)"
            comparisons[name] = item
    results["comparisons"] = comparisons
    _atomic_json_dump(results, results_path)
    print(f"PREPROC_DONE records={len(results['records'])} output={output_dir}", flush=True)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P4 preprocessing ablation: shared-gain DR")
    p.add_argument("--dataset", "--dataset-dir", dest="dataset",
                   default=os.environ.get("P4_DATASET_DIR", "p4_dataset"))
    p.add_argument("--output-dir", "--out", dest="output_dir", default="p4_preproc_run")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch", type=int, default=16)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _run(_parse_args(argv))


if __name__ == "__main__":
    main()
