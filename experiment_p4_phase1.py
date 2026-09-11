#!/usr/bin/env python3
"""Phase-1 batch: ridge+intervals vs quantile GRU vs consistency GRU.

Compares, on identical splits/budget/windows:
  1. PhysicsFeatureRidge + split-conformal residual intervals (calib designs).
  2. QuantileGRUModel (ordered head, pinball, explicit select/val designs).
  3. QuantileGRUModel + same-state paired-consistency training.
Reference: existing quantile-TCN checkpoints (--tcn-ckpt-dir, optional).

Frozen primary readouts: held-out-nuisance (mains-hum) MAE and
safe-certification availability; near-flutter MAE; interval score.
Held-out conditions: 50/150 Hz sinusoidal interference (absent from training
DR), half-window prefixes (256 samples), DR-corrupted slice.
Go/no-go (reported, not gated): consistency earns its place only with better
robustness/decision metrics at comparable interval width.

Usage: P4_DATASET_DIR=p4_dataset_full PYTHONPATH=src python experiment_p4_phase1.py
  [--tcn-ckpt-dir p4run_full] [--seeds 0 1 2] [--epochs 20] [--out p4_phase1_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mechanics.p4_margin_estimation.quantile_gru import (
    QuantileGRUModel, build_state_pairs, train_quantile_gru, train_quantile_gru_consistency,
)
from mechanics.p4_margin_estimation.decision_metrics import (
    interval_score, paired_design_comparison, regime_metrics, ridge_residual_intervals,
    safety_availability,
)
from mechanics.p4_margin_estimation.quantile_head import QuantileMarginModel
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from train_p4_expanded import (
    _split_select_calibrate, apply_dr_with_mask, build_clips, evaluate_coverage,
    load_dataset, with_ones_mask,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)  # noqa: E731
RESULTS_PATH = "p4_phase1_results.json"
HUM_FREQS = (50.0, 150.0)
HUM_FRAC = 0.05  # amplitude as fraction of per-channel calibration RMS
PREFIX_LEN = 256


def _save_results(results, path=RESULTS_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    Path(tmp).replace(path)


def _wandb():
    try:
        import wandb
        return wandb.init(project=os.environ.get("WANDB_PROJECT", "p4-margin"),
                          mode=os.environ.get("WANDB_MODE", "offline"),
                          job_type="phase1-batch",
                          config={"hum_freqs": list(HUM_FREQS), "hum_frac": HUM_FRAC,
                                  "prefix_len": PREFIX_LEN, "lambda_cons": 0.1})
    except Exception as exc:
        P(f"wandb unavailable ({exc}) — file logging only.")
        return None


def _wlog(run, payload):
    if run is None:
        return
    try:
        run.log(payload)
    except Exception as exc:
        P(f"wandb log failed ({exc}) — continuing.")


def add_hum(clips, seed=7, freqs=HUM_FREQS, frac=HUM_FRAC):
    """Mains-hum interference: held-out family absent from training DR."""
    from train_p4_expanded import _Clip
    rng = np.random.default_rng(seed)
    out = []
    for c in clips:
        sig = np.asarray(c.sensor_signals, dtype=float)
        if sig.shape[0] % 2 == 0:
            half = sig.shape[0] // 2
            phys, mask = sig[:half], sig[half:]
        else:
            phys, mask = sig, None
        dt = getattr(c, "dt", None)
        hum = np.zeros_like(phys)
        if dt is not None and np.isfinite(dt) and dt > 0:
            t = np.arange(phys.shape[1]) * float(dt)
            cal = phys[:, :max(2, phys.shape[1] // 10)]
            rms = np.sqrt(np.mean((cal - cal.mean(axis=1, keepdims=True)) ** 2,
                                  axis=1, keepdims=True)) + 1e-12
            for f in freqs:
                ph = rng.uniform(0, 2 * np.pi, size=(phys.shape[0], 1))
                amp = rng.uniform(0.5, 1.0) * frac * rms
                hum += amp * np.sin(2 * np.pi * f * t + ph)
        noisy = phys + hum
        combined = np.concatenate([noisy, mask if mask is not None else np.ones_like(noisy)], axis=0)
        out.append(_Clip(combined, c.margin, c.velocity, c.design_id,
                         dt=getattr(c, "dt", None), time=None,
                         realization_idx=getattr(c, "realization_idx", None),
                         sat_frac=float(getattr(c, "sat_frac", 0.0))))
    return out


def prefix_clips(clips, length=PREFIX_LEN):
    from train_p4_expanded import _Clip
    out = []
    for c in clips:
        sig = np.asarray(c.sensor_signals)[:, :length]
        out.append(_Clip(sig, c.margin, c.velocity, c.design_id,
                         dt=getattr(c, "dt", None), time=None,
                         realization_idx=getattr(c, "realization_idx", None),
                         sat_frac=float(getattr(c, "sat_frac", 0.0))))
    return out


def _predict_gru(model, clips, device="cpu", batch=256):
    model.eval()
    outs = []
    with torch.inference_mode():
        for s in range(0, len(clips), batch):
            Xb = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32)
                              for c in clips[s:s + batch]]).to(device)
            outs.append(model(Xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 3))


def _clip_arrays(clips, pred):
    y = np.array([c.margin for c in clips], dtype=float)
    dids = [c.design_id for c in clips]
    sat = np.array([float(getattr(c, "sat_frac", 0.0)) > 0 for c in clips], dtype=bool)
    if pred.ndim == 2 and pred.shape[1] == 3:
        return y, pred[:, 1], pred[:, 0], pred[:, 2], sat, dids
    p = np.asarray(pred, dtype=float).ravel()
    return y, p, None, None, sat, dids


def evaluate_all(name, y, med, lo, hi, sat, dids):
    rep = {"mae": float(np.mean(np.abs(med - y)))}
    rep["regimes"] = regime_metrics(y, med, lo, hi, sat, dids)
    rep["per_design_mae"] = {}
    for d in sorted(set(dids)):
        m = np.array([dd == d for dd in dids])
        rep["per_design_mae"][d] = float(np.mean(np.abs(med[m] - y[m])))
    nf = y < 0.15
    if int(nf.sum()):
        rep["near_flutter_mae"] = float(np.mean(np.abs(med[nf] - y[nf])))
    else:
        rep["near_flutter_mae"] = float("nan")
    if lo is not None:
        rep["interval_score"] = interval_score(y, lo, hi, 0.10)
    sa = safety_availability(y, med, lo if lo is not None else med, dids)
    rep.update({k: sa[k] for k in ("false_safe_rate", "unsafe_fraction_among_safe_declared",
                                   "safe_certification_availability", "n_unsafe",
                                   "n_median_false_safe", "n_safe_declared",
                                   "n_designs_with_false_safe")})
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcn-ckpt-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    out_path = args.out

    clips_arr, margins, velocities, design_ids, realization_ids, meta = load_dataset(args.dataset)
    prov = meta.get("provenance", {})
    dts = prov.get("dts") if isinstance(prov, dict) else None
    sats = prov.get("sat_fracs") if isinstance(prov, dict) else None
    train_ids, val_ids, test_ids = (set(meta[k]) for k in ("train_designs", "val_designs", "test_designs"))
    train_clips, _ = build_clips(clips_arr, margins, velocities, design_ids, train_ids, dts=dts,
                                 realization_ids=realization_ids, sat_fracs=sats)
    val_clips, _ = build_clips(clips_arr, margins, velocities, design_ids, val_ids, dts=dts,
                               realization_ids=realization_ids, sat_fracs=sats)
    test_clips, test_vels = build_clips(clips_arr, margins, velocities, design_ids, test_ids, dts=dts,
                                        realization_ids=realization_ids, sat_fracs=sats)
    select_ids, calib_ids = _split_select_calibrate(sorted(val_ids))
    sel_clips = [c for c in val_clips if c.design_id in select_ids]
    cal_clips = [c for c in val_clips if c.design_id in calib_ids]

    train_dr, n_ch = apply_dr_with_mask(train_clips, seed=0)
    sel_clean, cal_clean = with_ones_mask(sel_clips), with_ones_mask(cal_clips)
    test_clean = with_ones_mask(test_clips)
    test_dr = apply_dr_with_mask(test_clips, seed=2)[0]
    test_hum = add_hum(test_clean)
    test_pre = prefix_clips(test_clean)
    P(f"clips: {len(train_clips)} train / {len(sel_clips)} select / {len(cal_clips)} calib / {len(test_clips)} test; ch={n_ch}")

    results = {}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
            P(f"resuming: keeping {len([k for k in results if not k.startswith('_')])} entries")
        except Exception as exc:
            P(f"could not read {args.out} ({exc}) — starting fresh.")
            results = {}
    run = _wandb()

    def done(name, ckpt=None):
        return name in results and (ckpt is None or os.path.exists(ckpt))

    # ---- ridge + residual intervals (cheap reference) ----
    if not done("ridge_interval"):
        t0 = time.time()
        ridge = PhysicsFeatureRidge()
        ridge.fit(train_dr)
        ri = ridge_residual_intervals(ridge, cal_clean, test_clean, train_clips=train_dr)
        rep = {"fit_time": time.time() - t0, "adjustment": ri["adjustment"],
               "calib_n": ri["calib_n"], "calib_designs": ri["calib_designs"]}
        for cond, clips in (("clean", test_clean), ("hum", test_hum), ("prefix", test_pre), ("dr", test_dr)):
            p = np.asarray(ridge.predict(clips), dtype=float).ravel()
            y, med, _, _, sat, dids = _clip_arrays(clips, p)
            rep[cond] = evaluate_all("ridge", y, med, ri["lower"] if cond == "clean" else p - ri["adjustment"],
                                     ri["upper"] if cond == "clean" else p + ri["adjustment"], sat, dids)
        results["ridge_interval"] = rep
        _save_results(results, args.out)
        P(f"ridge clean MAE={rep['clean']['mae']:.4f} hum MAE={rep['hum']['mae']:.4f}")
    _wlog(run, {"ridge/clean_mae": results["ridge_interval"]["clean"]["mae"],
                "ridge/hum_mae": results["ridge_interval"]["hum"]["mae"]})

    # ---- quantile GRU plain + consistency ----
    for seed in args.seeds:
        for variant, tfn, ckpt in (
            (f"qgru_s{seed}", train_quantile_gru, f"p4_qgru_s{seed}.pt"),
            (f"qgru_cons_s{seed}", train_quantile_gru_consistency, f"p4_qgru_cons_s{seed}.pt"),
        ):
            if done(variant, ckpt):
                P(f"{variant}: checkpoint+metrics present — skipping.")
                continue
            t0 = time.time()
            kw = dict(n_channels=n_ch, epochs=args.epochs, seed=seed, batch_size=64,
                      val_clips=sel_clean)
            if "cons" in variant:
                pairs = build_state_pairs(train_dr)
                P(f"  {variant}: {len(pairs)} pairs over "
                  f"{len({id(c) for pr in pairs for c in pr})}/{len(train_dr)} train clips")
                model, hist = tfn(train_dr, pairs=pairs, lambda_cons=args.lambda_cons, **kw)
            else:
                model, hist = tfn(train_dr, **kw)
            torch.save(model.state_dict(), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
            rep = {"train_time": time.time() - t0,
                   "train_loss": hist["train_loss"][-1], "val_loss": hist["val_loss"][-1]}
            for cond, clips in (("clean", test_clean), ("hum", test_hum), ("prefix", test_pre), ("dr", test_dr)):
                pred = _predict_gru(model, clips)
                y, med, lo, hi, sat, dids = _clip_arrays(clips, pred)
                rep[cond] = evaluate_all(variant, y, med, lo, hi, sat, dids)
            results[variant] = rep
            _save_results(results, args.out)
            del model, hist
            P(f"{variant}: clean MAE={rep['clean']['mae']:.4f} hum MAE={rep['hum']['mae']:.4f} "
              f"width={rep['clean']['regimes']['saturated'].get('mean_width', float('nan')):.4f}")
        _wlog(run, {f"{variant}/clean_mae": results[variant]["clean"]["mae"]
                    for variant in (f"qgru_s{seed}", f"qgru_cons_s{seed}")})

    # ---- reference TCN checkpoints ----
    if args.tcn_ckpt_dir:
        for seed in args.seeds:
            name = f"tcn_s{seed}"
            ckpt = os.path.join(args.tcn_ckpt_dir, f"p4_quantile_s{seed}.pt")
            if name in results or not os.path.exists(ckpt):
                continue
            m = QuantileMarginModel(n_channels=n_ch, hidden_dim=32, n_layers=9)
            m.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
            rep = {}
            for cond, clips in (("clean", test_clean), ("hum", test_hum), ("prefix", test_pre), ("dr", test_dr)):
                pred = _predict_gru(m, clips)
                y, med, lo, hi, sat, dids = _clip_arrays(clips, pred)
                rep[cond] = evaluate_all(name, y, med, lo, hi, sat, dids)
            results[name] = rep
            _save_results(results, args.out)

    # ---- paired design-level comparisons (per-design MAE, seeds averaged) ----
    def _family_records(prefix):
        per_seed = []
        for seed in args.seeds:
            key = prefix if prefix == "ridge_interval" else f"{prefix}_s{seed}"
            if key not in results or "per_design_mae" not in results[key].get("clean", {}):
                return None
            per_seed.append(results[key]["clean"]["per_design_mae"])
        designs = sorted(per_seed[0])
        if any(sorted(d) != designs for d in per_seed):
            return None
        return [(d, float(np.mean([rec[d] for rec in per_seed]))) for d in designs]
    comps = {}
    for a, b, tag in (("qgru", "ridge_interval", "qgru_vs_ridge"),
                      ("qgru_cons", "qgru", "cons_vs_plain"),
                      ("qgru", "tcn", "qgru_vs_tcn")):
        ra, rb = _family_records(a), _family_records(b)
        if ra is None or rb is None:
            comps[tag] = {"status": "incomplete (missing family records)"}
            continue
        try:
            comps[tag] = paired_design_comparison(ra, rb)
        except ValueError as exc:
            comps[tag] = {"status": f"incomparable: {exc}"}
    results["comparisons"] = comps
    _save_results(results, args.out)
    # ---- go/no-go readout (reported, not gated) ----
    try:
        c_plain = np.mean([results[f"qgru_s{s}"]["hum"]["mae"] for s in args.seeds])
        c_cons = np.mean([results[f"qgru_cons_s{s}"]["hum"]["mae"] for s in args.seeds])
        p_plain = np.mean([results[f"qgru_s{s}"]["prefix"]["mae"] for s in args.seeds])
        p_cons = np.mean([results[f"qgru_cons_s{s}"]["prefix"]["mae"] for s in args.seeds])
        w_plain = np.mean([results[f"qgru_s{s}"]["clean"]["regimes"]["saturated"]["mean_width"]
                           for s in args.seeds])
        w_cons = np.mean([results[f"qgru_cons_s{s}"]["clean"]["regimes"]["saturated"]["mean_width"]
                          for s in args.seeds])
        verdict = {"hum_mae_plain": c_plain, "hum_mae_cons": c_cons,
                   "prefix_mae_plain": p_plain, "prefix_mae_cons": p_cons,
                   "width_plain": w_plain, "width_cons": w_cons,
                   "earn": bool((c_cons < c_plain or p_cons < p_plain) and w_cons <= 1.05 * w_plain)}
    except KeyError:
        verdict = {"status": "incomplete"}
    results["verdict"] = verdict
    _save_results(results, args.out)
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
    P("PHASE1_DONE " + json.dumps(comps))


if __name__ == "__main__":
    main()
