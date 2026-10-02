#!/usr/bin/env python3
"""P4 target-ablation experiment: u_crit regression + velocity-blinded residual margin.

Additive to the saturation study: same cap-50 data, splits, DR, GRU trainer,
and reporting conventions; different prediction targets.

- ``ucrit``: per-(design, realization) flutter velocity from waveform only.
- ``resid``: velocity-blinded residual margin. A per-split affine fit of
  margin on velocity (train-design clips only) is subtracted from every clip;
  models predict the residual. The fit coefficients come from train designs
  only and are stored in the record. Compares GRU-vs-ridge on damping content
  once the definitional velocity shortcut is removed.
- ``damp``: raw spectral abscissa (label_alpha, 1/s) from the same spectrum
  contract as the flutter label. Exact solver output, no velocity algebra:
  margin = 1-V/u_crit cannot be recovered from it without V and u_crit, so
  a velocity shortcut is structurally impossible. Standardized with train
  statistics. This is the breakthrough probe: if the GRU beats ridge here,
  waveforms carry damping content the pooled envelope misses.

Justification for the blinded arm: the post-hoc oracle check showed V+u_crit
predicts margin to MAE 0.024 while true damping adds ~nothing on top, and the
velocity-only design-MAE correlates NEGATIVELY (-0.48) with GRU design-MAE --
the GRU beats velocity-only hardest exactly where velocity-only fails. The
blinded arm tests whether that residual edge is real damping content: if the
GRU beats ridge on residuals, the waveform edge is genuine; if both collapse
to the same floor, margin-from-waveform is exhausted on this dataset.

Usage::

    PYTHONPATH=src python experiment_p4_targets.py \
        --dataset p4_dataset_saturation --output-dir p4_targets_run \
        --seeds 0 1 2 --epochs 20 --batch-size 64 --predict-batch 32 --threads 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from experiment_p4_phase2 import GRUEllQuantile, _predict, train_quantile_generic
from experiment_p4_saturation import (
    _atomic_json_dump,
    _comparison,  # noqa: F401  (kept for convention parity; comparisons inlined below)
    _ensure_gru,
    _ensure_ridge,
    _fingerprint_dataset,
    _initial_results,
    _load_or_initialize_results,
    _make_cap_parts,
    _neural_record,
    _record_is_complete,
    _record_key,
    _relative_path,
    _ridge_record,
    _validate_dataset,
    _validate_seeds,
    ALPHA,
    MODEL_NAME,
    N_MODEL_CHANNELS,
    _COMPARISON_OUTPUT_DIR,  # noqa: F401
)
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from mechanics.p4_margin_estimation.quantile_head import (
    apply_cqr_adjustment,
    fit_cqr_adjustment,
)
from train_p4_expanded import _split_select_calibrate

TARGETS = ("ucrit", "resid", "damp")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_PREDICT_BATCH = 32
DEFAULT_THREADS = max(1, min(8, os.cpu_count() or 1))
CAP = 50.0


def _ucrit_key(clip: Any) -> tuple[str, int]:
    rid = getattr(clip, "realization_idx", None)
    return (str(clip.design_id), int(rid))


def _design_ucrit_map(bundle: dict[str, Any]) -> dict[tuple[str, int], float]:
    """Per-(design, realization) flutter velocity from dataset provenance.

    u_crit is recomputed per perturbed realization (not per nominal design),
    so the predictable system property is the (design, realization) unit.
    Clips key on ``(design_id, realization_idx)``; velocities within a unit
    share one u_crit exactly.
    """
    from train_p4_expanded import load_dataset

    _, _, _, design_ids, realization_ids, meta = load_dataset(str(bundle["dataset_dir"]))
    prov = meta.get("provenance", {}) if isinstance(meta, dict) else {}
    ucrits = np.asarray(prov.get("u_crits"), dtype=float)
    if ucrits.shape != (len(design_ids),):
        raise ValueError("u_crit provenance is not aligned with clips")
    out: dict[tuple[str, int], float] = {}
    for did, rid, u in zip([str(x) for x in design_ids],
                           [int(x) for x in realization_ids], ucrits.tolist()):
        if not math.isfinite(u) or u <= 0:
            raise ValueError(f"non-finite u_crit for {(did, rid)}")
        key = (did, int(rid))
        if key in out and out[key] != u:
            raise ValueError(f"u_crit varies within {(did, rid)}")
        out[key] = float(u)
    return out


def _velocity_blind(parts: dict[str, Any]) -> dict[str, float]:
    """Per-split affine margin-on-velocity fit from train-design clips only."""
    train = parts["train"]["raw"]
    v = np.asarray([float(c.velocity) for c in train], dtype=float)
    y = np.asarray([float(c.margin) for c in train], dtype=float)
    ok = np.isfinite(v) & np.isfinite(y)
    if ok.sum() < 2:
        raise ValueError("need >=2 finite train clips for velocity blinding")
    A = np.column_stack([v[ok], np.ones(ok.sum())])
    (a, b), *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return {"a": float(a), "b": float(b)}


def _damp_map(bundle: dict[str, Any]) -> dict[int, float]:
    """Per-clip spectral abscissa keyed by source row order.

    label_alpha comes from the same full-spectrum contract as the flutter
    label (solver-exact, no velocity algebra). Loaded in dataset row order;
    matched to clips via each cap part's stored source indices. Non-finite
    entries fail the run rather than silently dropping.
    """
    from train_p4_expanded import load_dataset

    _, _, _, _, _, meta = load_dataset(str(bundle["dataset_dir"]))
    prov = meta.get("provenance", {}) if isinstance(meta, dict) else {}
    arr = np.asarray(prov.get("label_alphas"), dtype=float)
    n = int(bundle["n_clips"])
    if arr.shape != (n,):
        raise ValueError(f"label_alphas shape {arr.shape} != {(n,)}")
    if not np.all(np.isfinite(arr)):
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"label_alphas has {bad} non-finite entries; refusing damp target")
    return {i: float(v) for i, v in enumerate(arr.tolist())}


def _apply_damp(parts: dict[str, Any], damp: dict[int, float],
                stats: dict[str, float] | None = None,
                fit_stats: bool = False) -> dict[str, float] | None:
    """Rewrite clip .margin fields to standardized label_alpha.

    Row matching uses each split's stored source ``indices`` (the same map
    the runner uses for growth/saturation strata), so excitation duplicates
    resolve to their own rows rather than colliding on physical keys.
    """
    vals = []
    for split in ("train", "val", "test"):
        idx = np.asarray(parts[split]["indices"]).tolist()
        for clips_key in ("raw", "clean"):
            clips = parts[split].get(clips_key, [])
            assert len(clips) == len(idx), (split, clips_key, len(clips), len(idx))
            for pos, c in enumerate(clips):
                c.margin = float(damp[int(idx[pos])])
                if clips_key == "raw":
                    vals.append(c.margin)
    for sub in ("select", "calib"):
        sub_ids = parts["select_ids"] if sub == "select" else parts["calib_ids"]
        parent_raw = parts["val"]["raw"]
        parent_idx = np.asarray(parts["val"]["indices"]).tolist()
        mask = [str(c.design_id) in sub_ids for c in parent_raw]
        sub_rows = [r for r, keep in zip(parent_idx, mask) if keep]
        n_raw, n_clean = len(parts[sub].get("raw", [])), len(parts[sub].get("clean", []))
        assert n_raw == len(sub_rows) or n_clean == len(sub_rows), (sub, n_raw, n_clean, len(sub_rows))
        for clips_key in ("raw", "clean"):
            clips = parts[sub].get(clips_key, [])
            if len(clips) != len(sub_rows):
                continue
            for pos, c in enumerate(clips):
                c.margin = float(damp[int(sub_rows[pos])])
    arr = np.asarray(vals, dtype=float)
    if fit_stats:
        return {"mean": float(arr.mean()), "std": float(arr.std() + 1e-8)}
    assert stats is not None
    for split in ("train", "val", "test", "select", "calib"):
        for clips_key in ("raw", "clean"):
            for c in parts[split].get(clips_key, []):
                c.margin = (float(c.margin) - stats["mean"]) / stats["std"]
    return None


def _retarget(parts: dict[str, Any], target: str, design_ucrit: dict[tuple[str, int], float],
              blind: dict[str, float] | None, stats: dict[str, float] | None = None,
              fit_stats: bool = False) -> dict[str, float] | None:
    """Rewrite clip .margin fields to the ablation target in place.

    ucrit: per-clip margin := (design, realization) u_crit (standardized).
    resid: per-clip margin := margin - (a*velocity + b).
    damp: handled by _apply_damp (row-matched); not here.
    """
    if target == "ucrit":
        vals = []
        for split in ("train", "val", "test", "select", "calib"):
            for c in parts[split].get("raw", []):
                c.margin = float(design_ucrit[_ucrit_key(c)])
                vals.append(c.margin)
        for split in ("train", "val", "test", "select", "calib"):
            for c in parts[split].get("clean", []):
                c.margin = float(design_ucrit[_ucrit_key(c)])
        arr = np.asarray(vals, dtype=float)
        if fit_stats:
            return {"mean": float(arr.mean()), "std": float(arr.std() + 1e-8)}
        assert stats is not None
        for split in ("train", "val", "test", "select", "calib"):
            for key in ("raw", "clean"):
                for c in parts[split].get(key, []):
                    c.margin = (float(c.margin) - stats["mean"]) / stats["std"]
        return None
    assert blind is not None
    for split in ("train", "val", "test", "select", "calib"):
        for key in ("raw", "clean"):
            for c in parts[split].get(key, []):
                c.margin = float(c.margin) - (blind["a"] * float(c.velocity) + blind["b"])
    return None

def _target_record_key(target: str, model: str, mode: str, seed: int | None) -> str:
    seed_token = "shared" if seed is None else f"seed{int(seed)}"
    return f"{target}__{model}__{mode}__{seed_token}"


def _target_label(target: str) -> str:
    return {"ucrit": "u_crit", "resid": "residual_margin", "damp": "spectral_abscissa"}[target]

def _target_config(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "device": str(getattr(args, "device", "cpu")),
        "targets": list(TARGETS),
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
        "calibration": {
            "neural": "heldout_design_CQR_fit_cqr_adjustment",
            "ridge": "heldout_design_abs_residual_finite_sample_order_statistic",
        },
        "velocity_blind": "per-split train-design affine margin-on-velocity fit",
        "ucrit_standardization": "train-clip mean/std",
    }


def _destandardize_ucrit_report(record: dict[str, Any], stats: dict[str, float]) -> None:
    """Attach physical-unit u_crit MAE alongside the standardized report."""
    scale = float(stats["std"])
    for section in ("metrics",):
        m = record.get(section)
        if isinstance(m, dict) and isinstance(m.get("mae"), (int, float)):
            m["mae_physical_ms"] = float(m["mae"]) * scale
    pd = record.get("per_design")
    if isinstance(pd, list):
        for row in pd:
            if isinstance(row, dict) and isinstance(row.get("mae"), (int, float)):
                row["mae_physical_ms"] = float(row["mae"]) * scale


def _run_targets(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _validate_seeds(args.seeds)
    if args.epochs < 1 or args.batch_size < 1 or args.predict_batch < 1:
        raise ValueError("epochs, batch-size, and predict-batch must be positive")
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    try:
        device = torch.device(getattr(args, "device", "cpu"))
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid training device: {getattr(args, 'device', 'cpu')}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {device} requested, but CUDA is not available")
    args.device = str(device)
    torch.set_num_threads(int(args.threads))

    bundle = _validate_dataset(args.dataset)
    config = _target_config(args, seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records: dict[str, Any] = results["records"]
    design_ucrit = _design_ucrit_map(bundle)

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
    import experiment_p4_saturation as sat
    sat._COMPARISON_OUTPUT_DIR = output_dir

    for target in TARGETS:
        target_pending = [item for item in pending if item[2] == target]
        if not target_pending:
            continue
        print(f"Preparing target {_target_label(target)} "
              f"({len(target_pending)} incomplete records)...", flush=True)
        parts = _make_cap_parts(bundle, CAP, need_clean=True)
        blind = _velocity_blind(parts) if target == "resid" else None
        if target == "damp":
            assert damp_map is not None
            stats = _apply_damp(parts, damp_map, fit_stats=True)
            assert stats is not None
            _apply_damp(parts, damp_map, stats=stats)
        else:
            stats = _retarget(parts, target, design_ucrit, blind, fit_stats=True) if target == "ucrit" else None
            if target == "ucrit":
                assert stats is not None
                _retarget(parts, target, design_ucrit, blind, stats=stats)
            else:
                _retarget(parts, target, design_ucrit, blind)
        calib_parts = {"raw": parts["calib"]["raw"], "clean": parts["calib"]["clean"], "cap": CAP}
        select_clean = parts["select"]["clean"]

        def _t_ckpt(kind: str, seed: int | None = None) -> Path:
            import experiment_p4_saturation as _sat
            base = _sat._checkpoint_path(output_dir, kind, CAP, seed)
            return base.parent / f"{target}_{base.name}"

        def _t_state(seed: int) -> Path:
            import experiment_p4_saturation as _sat
            base = _sat._training_state_path(output_dir, CAP, seed)
            return base.parent / f"{target}_{base.name}"

        gru_artifacts: dict[int, dict[str, Any]] = {}
        ridge_artifact: dict[str, Any] | None = None
        for seed in seeds:
            art = _ensure_gru(
                output_dir=output_dir, records={}, train_cap=CAP, seed=seed,
                train_raw=parts["train"]["raw"], select_clean=select_clean,
                epochs=args.epochs, batch_size=args.batch_size,
                predict_batch=args.predict_batch, device=args.device,
                protocol=f"{target}_matched",
            )
            want_ckpt = _t_ckpt("gru", seed)
            got_ckpt = Path(str(art["path"]))
            if got_ckpt.resolve() != want_ckpt.resolve():
                want_ckpt.parent.mkdir(parents=True, exist_ok=True)
                tmp = want_ckpt.with_name(want_ckpt.name + ".tmp")
                import shutil
                shutil.copyfile(got_ckpt, tmp)
                os.replace(tmp, want_ckpt)
                from experiment_p4_saturation import _sha256_file
                art = dict(art)
                art["path"] = want_ckpt
                art["sha256"] = _sha256_file(want_ckpt)
                stale_state = got_ckpt.parent / got_ckpt.name.replace(".pt", ".training.pt")
                if stale_state.is_file():
                    try:
                        stale_state.unlink()
                    except OSError:
                        pass
            cpred = np.asarray(_predict(art["model"], calib_parts["clean"],
                                        batch=args.predict_batch, device=args.device), dtype=float)
            cy = np.asarray([float(c.margin) for c in calib_parts["clean"]], dtype=float)
            if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
                raise ValueError(f"{target} calibration predictions are invalid")
            art["cqr"] = {
                "adjustment": float(fit_cqr_adjustment(cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)),
                "cap": CAP, "calib_n": int(len(cy)),
                "calib_designs": int(len(set(str(c.design_id) for c in calib_parts["clean"]))),
            }
            gru_artifacts[seed] = art
        ridge_artifact = _ensure_ridge(
            output_dir=output_dir, records={}, train_cap=CAP,
            train_raw=parts["train"]["raw"])
        from experiment_p4_saturation import _ridge_adjustment
        ridge_artifact["calibration"] = _ridge_adjustment(ridge_artifact["model"], calib_parts["raw"])
        want_ridge = _t_ckpt("ridge")
        got_ridge = Path(str(ridge_artifact["path"]))
        if got_ridge.resolve() != want_ridge.resolve():
            want_ridge.parent.mkdir(parents=True, exist_ok=True)
            tmp = want_ridge.with_name(want_ridge.name + ".tmp")
            import shutil
            shutil.copyfile(got_ridge, tmp)
            os.replace(tmp, want_ridge)
            from experiment_p4_saturation import _sha256_file
            ridge_artifact = dict(ridge_artifact)
            ridge_artifact["path"] = want_ridge
            ridge_artifact["sha256"] = _sha256_file(want_ridge)

        for key, kind, _t, mode, seed in target_pending:
            if _record_is_complete(records.get(key), output_dir):
                continue
            extra: dict[str, Any] = {"target": _target_label(target)}
            if target == "resid":
                extra["velocity_blind"] = dict(blind) if blind else {}
            elif target == "damp":
                extra["damp_standardization"] = dict(stats) if stats else {}
            else:
                extra["ucrit_standardization"] = dict(stats) if stats else {}
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
                if target == "ucrit" and stats:
                    _destandardize_ucrit_report(records[key], stats)
            _atomic_json_dump(results, results_path)
        del parts, calib_parts, gru_artifacts, ridge_artifact

    # Paired GRU-minus-ridge comparisons per target (per-seed + mean-seed).
    import experiment_p4_saturation as satmod
    comparisons: dict[str, Any] = {}
    for target in TARGETS:
        seed_keys = []
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
            seed_keys.append((left, right))
        lefts = [_target_record_key(target, MODEL_NAME, "matched_retrain", s) for s in seeds]
        la = [results["records"].get(k) for k in lefts]
        right = _target_record_key(target, "physics_feature_ridge", "matched_retrain", None)
        from experiment_p4_saturation import _mean_design_records
        name = f"{target}__{MODEL_NAME}__mean_seed_minus_ridge"
        item = {"left": lefts, "right": right, "metric": "mae",
                "subtraction": "mean per-design MAE over seeds (gru - ridge)"}
        if all(_record_is_complete(r, output_dir) for r in la + [results["records"].get(right)]):
            a = _mean_design_records([r for r in la if isinstance(r, dict)])
            b = (results["records"].get(right) or {}).get("per_design_mae")
            if a is not None and isinstance(b, dict) and sorted(dict(a)) == sorted(b):
                item["status"] = "complete"
                item["result"] = paired_design_comparison(
                    a, sorted(b.items()), n_bootstrap=args.bootstrap, seed=0)
            else:
                item["status"] = "incomplete (per-design sets differ)"
        else:
            item["status"] = "incomplete (record/checkpoint/prediction missing or mismatched)"
        comparisons[name] = item
    results["comparisons"] = comparisons
    _atomic_json_dump(results, results_path)
    print(f"TARGETS_DONE records={len(results['records'])} output={output_dir}", flush=True)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4 target ablation: ucrit + velocity-blinded residual")
    parser.add_argument("--dataset", "--dataset-dir", dest="dataset",
                        default=os.environ.get("P4_DATASET_DIR", "p4_dataset"),
                        help="dataset directory")
    parser.add_argument("--output-dir", "--out", dest="output_dir", default="p4_targets_run",
                        help="artifact directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--predict-batch", type=int, default=DEFAULT_PREDICT_BATCH)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _run_targets(_parse_args(argv))


if __name__ == "__main__":
    main()
