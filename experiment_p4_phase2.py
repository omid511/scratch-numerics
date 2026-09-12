#!/usr/bin/env python3
"""Phase-2 batch: half-window control, 2x2 summary ablation, cap sensitivity.

Follow-up to experiment_p4_phase1.py, addressing the review corrections:
 1. Honest half-windows: truncate + renormalize from the short window's own
    calibration samples (deployed emulation), then fit/calibrate AND evaluate
    on halves (ridge-half, GRU-half, TCN-half).
 2. 2x2 backbone x summary ablation on full windows: GRU+final (exists),
    GRU+early/late/last, TCN+early/late/last (exists), TCN+final step.
 3. Calibrated (CQR) intervals for every quantile family + full
    risk-availability denominators (raw and calibrated, seed-averaged).
 4. Saturation-cap sensitivity: eval-only sweep over tighter in-memory caps.

Comparison tags store mean(first-named minus second-named) paired design
differences; see paired_design_comparison (a - b convention).

Usage: P4_DATASET_DIR=p4_dataset_full PYTHONPATH=src python experiment_p4_phase2.py
  [--ckpt-dir p4run_full] [--seeds 0 1 2] [--epochs 20] [--out p4_phase2_results.json]
Checkpoints (.pt) are read from and written to --ckpt-dir.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mechanics.p4_margin_estimation.tcn import TCNBackbone, TemporalSummary
from mechanics.p4_margin_estimation.baselines import GRUMarginModel
from mechanics.p4_margin_estimation.quantile_head import (
    SUPPORTED_QUANTILES, QuantileMarginModel, fit_cqr_adjustment, pinball_loss,
)
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
from mechanics.p4_margin_estimation.decision_metrics import (
    paired_design_comparison, regime_metrics, ridge_residual_intervals, safety_availability,
)
from mechanics.p4_margin_estimation.quantile_gru import QuantileGRUModel
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from train_p4_expanded import (
    _split_select_calibrate, apply_dr_with_mask, build_clips,
    load_dataset, with_ones_mask,
)
from experiment_p4_phase1 import add_hum

P = lambda *a, **kw: print(*a, **kw, flush=True)  # noqa: E731
QW = (2.0, 1.0, 1.0)
HALF_LEN = 256
CAPS = (50.0, 25.0, 10.0)


def _save_results(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    Path(tmp).replace(path)


def _wandb():
    try:
        import wandb
        return wandb.init(project=os.environ.get("WANDB_PROJECT", "p4-margin"),
                          mode=os.environ.get("WANDB_MODE", "offline"),
                          job_type="phase2-batch")
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


def honest_half(clips, length=HALF_LEN):
    """Truncate + renormalize from the short window's own calibration block.

    Deployed short-window emulation: a real system normalizes from its own
    short calibration window, not from a longer clip's prefix statistics.
    Operates on physical rows only; DR mask rows pass through truncated.
    """
    from train_p4_expanded import _Clip
    out = []
    for c in clips:
        sig = np.asarray(c.sensor_signals, dtype=float)
        if sig.shape[0] % 2 == 0:
            half = sig.shape[0] // 2
            phys, mask = sig[:half, :length], sig[half:, :length]
        else:
            phys, mask = sig[:, :length], None
        cal = max(2, int(length * 0.1))
        phys = causal_calibration_normalize(phys, calibration_samples=cal,
                                            normalize_mode="per_channel")
        combined = np.concatenate([phys, mask if mask is not None else np.ones_like(phys)], axis=0)
        out.append(_Clip(combined, c.margin, c.velocity, c.design_id,
                         dt=getattr(c, "dt", None), time=None,
                         realization_idx=getattr(c, "realization_idx", None),
                         sat_frac=float(getattr(c, "sat_frac", 0.0))))
    return out


def apply_cap(clips, cap):
    """In-memory tighter saturation cap (directional sensitivity only)."""
    from train_p4_expanded import _Clip
    out = []
    for c in clips:
        sig = np.asarray(c.sensor_signals, dtype=float)
        if sig.shape[0] % 2 == 0:
            half = sig.shape[0] // 2
            phys = np.clip(sig[:half], -cap, cap)
            sig = np.concatenate([phys, sig[half:]], axis=0)
        else:
            sig = np.clip(sig, -cap, cap)
        out.append(_Clip(sig, c.margin, c.velocity, c.design_id,
                         dt=getattr(c, "dt", None), time=None,
                         realization_idx=getattr(c, "realization_idx", None),
                         sat_frac=float(getattr(c, "sat_frac", 0.0))))
    return out


class GRUEllQuantile(nn.Module):
    """GRU backbone + early/late/last summary + ordered quantile head."""

    def __init__(self, n_channels=16, hidden_dim=86):
        super().__init__()
        self.quantiles = SUPPORTED_QUANTILES
        base = GRUMarginModel(n_channels, hidden_dim)
        self.gru = base.gru
        self.summary = TemporalSummary(window=64)
        self.head = nn.Linear(3 * hidden_dim, 3)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)  # (B, T, H)
        feat = self.summary(out.transpose(1, 2))  # (B, 3H)
        out = self.head(feat)
        median = out[:, 0:1]
        return torch.cat([median - F.softplus(out[:, 1:2]), median,
                          median + F.softplus(out[:, 2:3])], dim=1)


class TCNFinalQuantile(nn.Module):
    """TCN backbone + final-timestep + ordered quantile head."""

    def __init__(self, n_channels=16, hidden_dim=32, n_layers=9):
        super().__init__()
        self.quantiles = SUPPORTED_QUANTILES
        self.tcn = TCNBackbone(n_channels, hidden_dim, n_layers, sequence_length=512)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        feat = self.tcn(x)[:, :, -1]
        out = self.head(feat)
        median = out[:, 0:1]
        return torch.cat([median - F.softplus(out[:, 1:2]), median,
                          median + F.softplus(out[:, 2:3])], dim=1)


def _stack_valid(clips, n_channels):
    Xl, yl = [], []
    for c in clips:
        if not np.isfinite(c.margin):
            continue
        s = torch.tensor(np.asarray(c.sensor_signals), dtype=torch.float32)
        if not bool(torch.isfinite(s).all()):
            continue
        Xl.append(s)
        yl.append(c.margin)
    if not Xl:
        raise ValueError("no valid clips")
    return torch.stack(Xl), torch.tensor(yl, dtype=torch.float32)


def train_quantile_generic(make_model, clips, *, val_clips, epochs, lr, batch_size,
                           seed, n_channels, on_epoch=None):
    """Minimal matched quantile trainer: pinball + Adam + cosine + best-val restore."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y = _stack_valid(clips, n_channels)
    Xv, yv = _stack_valid(val_clips, n_channels)
    tids = {c.design_id for c in clips if np.isfinite(c.margin)}
    vids = {c.design_id for c in val_clips if np.isfinite(c.margin)}
    if tids & vids:
        raise ValueError(f"train/val design overlap: {sorted(tids & vids)[:5]}")
    train_dl = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size)
    model = make_model()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    hist = {"train_loss": [], "val_loss": []}
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in train_dl:
            loss = pinball_loss(model(xb), yb, SUPPORTED_QUANTILES, weights=QW)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item()
            nb += 1
        sched.step()
        hist["train_loss"].append(tot / max(nb, 1))
        model.eval()
        vt, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                vt += pinball_loss(model(xb), yb, SUPPORTED_QUANTILES, weights=QW).item()
                nv += 1
        avg = vt / max(nv, 1)
        hist["val_loss"].append(avg)
        if avg < best:
            best, best_state = avg, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if on_epoch is not None:
            on_epoch(epoch, model, hist)
        _ = rng  # deterministic DataLoader order documented via torch seed
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def _predict(model, clips, batch=256):
    model.eval()
    outs = []
    with torch.inference_mode():
        for s in range(0, len(clips), batch):
            Xb = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32)
                              for c in clips[s:s + batch]])
            outs.append(model(Xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 3))


def _arrays(clips, pred):
    y = np.array([c.margin for c in clips], dtype=float)
    dids = [c.design_id for c in clips]
    sat = np.array([float(getattr(c, "sat_frac", 0.0)) > 0 for c in clips], dtype=bool)
    p = np.asarray(pred, dtype=float)
    if p.ndim == 2 and p.shape[1] == 3:
        return y, p[:, 1], p[:, 0], p[:, 2], sat, dids
    return y, p.ravel(), None, None, sat, dids


def _per_design_mae(y, med, dids):
    """Per-design mean absolute error records for paired comparisons."""
    y = np.asarray(y, dtype=float)
    med = np.asarray(med, dtype=float)
    out = {}
    for d in sorted(set(dids)):
        msk = np.array([dd == d for dd in dids])
        out[d] = float(np.mean(np.abs(med[msk] - y[msk])))
    return out


def full_report(y, med, lo, hi, sat, dids):
    """Complete denominator-explicit uncertainty table for one evaluation."""
    rep = {"mae": float(np.mean(np.abs(med - y))),
           "regimes": regime_metrics(y, med, lo, hi, sat, dids)}
    nf = y < 0.15
    rep["near_flutter_mae"] = float(np.mean(np.abs(med[nf] - y[nf]))) if int(nf.sum()) else float("nan")
    n_safe = int((y > 0).sum())
    rep["n_safe"] = int(n_safe)
    if lo is not None:
        from mechanics.p4_margin_estimation.decision_metrics import interval_score
        rep["interval_score"] = interval_score(y, lo, hi, 0.10)
        rep["coverage"] = float(np.mean((y >= lo) & (y <= hi)))
        rep["mean_width"] = float(np.mean(hi - lo))
    sa = safety_availability(y, med, lo if lo is not None else med, dids)
    rep.update(sa)
    rep["safe_cert_avail_among_safe"] = (
        float(((lo > 0) & (y > 0)).sum() / n_safe) if (lo is not None and n_safe) else float("nan"))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=".")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="p4_phase2_results.json")
    ap.add_argument("--dataset", default=os.environ.get("P4_DATASET_DIR", "p4_dataset"))
    args = ap.parse_args()

    clips_arr, margins, velocities, design_ids, realization_ids, meta = load_dataset(args.dataset)
    prov = meta.get("provenance", {})
    dts = prov.get("dts") if isinstance(prov, dict) else None
    sats = prov.get("sat_fracs") if isinstance(prov, dict) else None
    train_ids, val_ids, test_ids = (set(meta[k]) for k in ("train_designs", "val_designs", "test_designs"))
    kw = dict(dts=dts, realization_ids=realization_ids, sat_fracs=sats)
    train_clips, _ = build_clips(clips_arr, margins, velocities, design_ids, train_ids, **kw)
    val_clips, _ = build_clips(clips_arr, margins, velocities, design_ids, val_ids, **kw)
    test_clips, _ = build_clips(clips_arr, margins, velocities, design_ids, test_ids, **kw)
    select_ids, calib_ids = _split_select_calibrate(sorted(val_ids))
    sel_clips = [c for c in val_clips if c.design_id in select_ids]
    cal_clips = [c for c in val_clips if c.design_id in calib_ids]

    train_dr, n_ch = apply_dr_with_mask(train_clips, seed=0)
    sel_clean, cal_clean = with_ones_mask(sel_clips), with_ones_mask(cal_clips)
    test_clean = with_ones_mask(test_clips)
    test_dr = apply_dr_with_mask(test_clips, seed=2)[0]
    test_hum = add_hum(test_clean)
    # Honest halves: truncate + renormalize from the short window itself.
    train_half = honest_half(train_clips)
    sel_half = honest_half(sel_clips)
    cal_half = honest_half(cal_clips)
    test_half = honest_half(test_clips)
    train_half_dr, _ = apply_dr_with_mask(train_half, seed=0)
    sel_half_c, cal_half_c, test_half_c = (with_ones_mask(x) for x in (sel_half, cal_half, test_half))
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

    def ckpt(name):
        return os.path.join(args.ckpt_dir, f"{name}.pt")

    def done(name):
        return name in results and os.path.exists(ckpt(name))

    def save_ckpt(model, name):
        torch.save(model.state_dict(), ckpt(name) + ".tmp")
        os.replace(ckpt(name) + ".tmp", ckpt(name))
    # ---- ridge-half control (fit + calibrate on halves) ----
    if "ridge_half" not in results:
        rh = PhysicsFeatureRidge()
        rh.fit(train_half_dr)
        ri = ridge_residual_intervals(rh, cal_half_c, test_half_c, train_clips=train_half_dr)
        p = np.asarray(rh.predict(test_half_c), dtype=float).ravel()
        y, med, _, _, sat, dids = _arrays(test_half_c, p)
        rep = full_report(y, med, ri["lower"], ri["upper"], sat, dids)
        rep.update({"adjustment": ri["adjustment"], "calib_n": ri["calib_n"]})
        rep["per_design_mae"] = _per_design_mae(
            np.array([c.margin for c in test_half_c]),
            np.asarray(rh.predict(test_half_c), dtype=float).ravel(),
            [c.design_id for c in test_half_c])
        results["ridge_half"] = rep
        _save_results(results, args.out)
        P(f"ridge_half MAE={rep['mae']:.4f} cov={rep.get('coverage', float('nan')):.3f}")
    _wlog(run, {"ridge_half/mae": results["ridge_half"]["mae"]})

    # ---- neural-half retrains + 2x2 ablation (full-window variant heads) ----
    jobs = []
    for seed in args.seeds:
        jobs.append((f"qgru_half_s{seed}", "gru_half", seed))
        jobs.append((f"tcn_half_s{seed}", "tcn_half", seed))
        jobs.append((f"gru_ell_s{seed}", "gru_ell", seed))
        jobs.append((f"tcn_final_s{seed}", "tcn_final", seed))
    for name, kind, seed in jobs:
        if done(name):
            P(f"{name}: checkpoint+metrics present — skipping.")
            continue
        t0 = time.time()
        if kind == "qgru_half":
            from mechanics.p4_margin_estimation.quantile_gru import train_quantile_gru
            model, hist = train_quantile_gru(train_half_dr, n_channels=n_ch, epochs=args.epochs,
                                             seed=seed, batch_size=64, val_clips=sel_half_c)
            clips_ev = test_half_c
        elif kind == "tcn_half":
            from mechanics.p4_margin_estimation.train import train as train_tcn
            model, hist, _, _ = train_tcn(None, n_channels=n_ch, hidden_dim=32, n_layers=9,
                                          epochs=args.epochs, lr=1e-3, seed=seed, batch_size=64,
                                          train_clips=train_half_dr, val_clips=sel_half_c,
                                          test_clips=test_half_c, velocities=[])
            clips_ev = test_half_c
        elif kind == "gru_ell":
            model, hist = train_quantile_generic(
                lambda: GRUEllQuantile(n_ch), train_dr, val_clips=sel_clean,
                epochs=args.epochs, lr=1e-3, batch_size=64, seed=seed, n_channels=n_ch)
            clips_ev = test_clean
        else:
            model, hist = train_quantile_generic(
                lambda: TCNFinalQuantile(n_ch), train_dr, val_clips=sel_clean,
                epochs=args.epochs, lr=1e-3, batch_size=64, seed=seed, n_channels=n_ch)
            clips_ev = test_clean
        save_ckpt(model, name)
        pred = _predict(model, clips_ev)
        y, med, lo, hi, sat, dids = _arrays(clips_ev, pred)
        rep = full_report(y, med, lo, hi, sat, dids)
        rep["train_loss"] = hist["train_loss"][-1]
        rep["val_loss"] = hist["val_loss"][-1]
        rep["train_time"] = time.time() - t0
        rep["per_design_mae"] = _per_design_mae(y, med, dids)
        # CQR on the matching calibration window.
        cal_use = cal_half_c if "half" in kind else cal_clean
        cpred = _predict(model, cal_use)
        _, _, clo, chi, _, _ = _arrays(cal_use, cpred)
        from mechanics.p4_margin_estimation.quantile_head import fit_cqr_adjustment, apply_cqr_adjustment
        cy = np.array([c.margin for c in cal_use])
        adj = fit_cqr_adjustment(clo, chi, cy, alpha=0.10)
        alo, ahi = apply_cqr_adjustment(lo, hi, adj)
        rep["cqr"] = {"adjustment": adj,
                      "coverage": float(np.mean((y >= alo) & (y <= ahi))),
                      "mean_width": float(np.mean(ahi - alo))}
        sa_c = safety_availability(y, med, alo, dids)
        rep["cqr"].update({k: sa_c[k] for k in ("false_safe_rate", "unsafe_fraction_among_safe_declared",
                                                "safe_certification_availability")})
        results[name] = rep
        _save_results(results, args.out)
        del model, hist
        P(f"{name}: MAE={rep['mae']:.4f} cov={rep.get('coverage', float('nan')):.3f} "
          f"cqr_cov={rep['cqr']['coverage']:.3f}")
        _wlog(run, {f"{name}/mae": rep["mae"]})

    # ---- saturation cap sweep (eval-only, tighter caps in memory) ----
    for cap in CAPS[1:]:
        tag = f"capsweep_{int(cap)}"
        if tag in results:
            continue
        capped = apply_cap(test_clean, cap)
        out = {}
        for name, mk, is_q in (("ridge", None, False), ("qgru_s2", "p4_qgru_s2.pt", True),
                               ("tcn_s0", "p4_quantile_s0.pt", False)):
            if name == "ridge":
                rh = PhysicsFeatureRidge()
                rh.fit(train_dr)
                p = np.asarray(rh.predict(capped), dtype=float).ravel()
                y, med, _, _, sat, dids = _arrays(capped, p)
                out[name] = {"mae": float(np.mean(np.abs(med - y)))}
                continue
            path = os.path.join(args.ckpt_dir, mk)
            if not os.path.exists(path):
                continue
            m = QuantileGRUModel(n_ch) if "qgru" in name else QuantileMarginModel(
                n_channels=n_ch, hidden_dim=32, n_layers=9)
            m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            pred = _predict(m, capped)
            y, med, lo, hi, sat, dids = _arrays(capped, pred)
            out[name] = {"mae": float(np.mean(np.abs(med - y))),
                         "coverage": float(np.mean((y >= lo) & (y <= hi)))}
            del m
        results[tag] = out
        _save_results(results, args.out)
        P(f"cap {cap}: " + " ".join(f"{k}={v['mae']:.4f}" for k, v in out.items()))

    # ---- paired comparisons (mean first-named minus second-named) ----
    def famrec_here(prefix):
        per_seed = []
        for s in args.seeds:
            key = f"{prefix}_s{s}"
            if key not in results or "per_design_mae" not in results[key]:
                return None
            per_seed.append(results[key]["per_design_mae"])
        designs = sorted(per_seed[0])
        if any(sorted(d) != designs for d in per_seed):
            return None
        return [(d, float(np.mean([rec[d] for rec in per_seed]))) for d in designs]

    def backfill_phase1():
        """Per-design MAE records for phase-1 families by re-evaluating saved checkpoints."""
        p1_path = os.path.join(args.ckpt_dir, "p4_phase1_results.json")
        recs = {}
        if not os.path.exists(p1_path):
            return recs
        specs = [(f"qgru_s{s}", "p4_qgru_s{s}.pt", True) for s in args.seeds]
        specs += [(f"qgru_cons_s{s}", "p4_qgru_cons_s{s}.pt", True) for s in args.seeds]
        specs += [(f"tcn_s{s}", "p4_quantile_s{s}.pt", False) for s in args.seeds]
        for key, pat, is_gru in specs:
            path = os.path.join(args.ckpt_dir, pat.format(s=key.rsplit("_s", 1)[1]))
            if not os.path.exists(path):
                continue
            m = QuantileGRUModel(n_ch) if is_gru else QuantileMarginModel(
                n_channels=n_ch, hidden_dim=32, n_layers=9)
            m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            pred = _predict(m, test_clean)
            y, med, _, _, _, dids = _arrays(test_clean, pred)
            per = {}
            for d in sorted(set(dids)):
                msk = np.array([dd == d for dd in dids])
                per[d] = float(np.mean(np.abs(med[msk] - y[msk])))
            recs[key] = per
            del m
        rh = PhysicsFeatureRidge()
        rh.fit(train_dr)
        p = np.asarray(rh.predict(test_clean), dtype=float).ravel()
        y = np.array([c.margin for c in test_clean])
        dids = [c.design_id for c in test_clean]
        per = {}
        for d in sorted(set(dids)):
            msk = np.array([dd == d for dd in dids])
            per[d] = float(np.mean(np.abs(p[msk] - y[msk])))
        recs["ridge_interval"] = per
        return recs

    def famavg_pooled(store, prefix):
        per_seed = [store[f"{prefix}_s{s}"] for s in args.seeds if f"{prefix}_s{s}" in store]
        if len(per_seed) != len(args.seeds):
            return None
        designs = sorted(per_seed[0])
        if any(sorted(d) != designs for d in per_seed):
            return None
        return [(d, float(np.mean([rec[d] for rec in per_seed]))) for d in designs]

    p1recs = backfill_phase1()
    comps = {}
    pairs = [(("gru_ell", "here"), ("qgru", "p1"), "gru_ell_vs_qgru"),
             (("tcn_final", "here"), ("tcn", "p1"), "tcn_final_vs_tcn"),
             (("qgru_half", "here"), ("ridge_half", "here"), "qgru_half_vs_ridge_half"),
             (("tcn_half", "here"), ("ridge_half", "here"), "tcn_half_vs_ridge_half"),
             (("qgru", "p1"), ("ridge", "p1"), "qgru_vs_ridge_check")]
    for (fa, sa), (fb, sb), tag in pairs:
        src_store = results if sa == "here" else None
        ra = None
        if sa == "here":
            tmp = {}
            for s in args.seeds:
                kk = f"{fa}_s{s}"
                if kk in results and "per_design_mae" in results[kk]:
                    tmp[kk] = results[kk]["per_design_mae"]
            if len(tmp) == len(args.seeds):
                designs = sorted(next(iter(tmp.values())))
                if all(sorted(d) == designs for d in tmp.values()):
                    ra = [(d, float(np.mean([tmp[f"{fa}_s{s}"][d] for s in args.seeds]))) for d in designs]
        else:
            if fa in p1recs:
                per_seed = [p1recs[f"{fa}_s{s}"] for s in args.seeds if f"{fa}_s{s}" in p1recs]
                if len(per_seed) == len(args.seeds):
                    designs = sorted(per_seed[0])
                    ra = [(d, float(np.mean([r[d] for r in per_seed]))) for d in designs]
            elif fa == "ridge":
                ra = sorted(p1recs.get("ridge_interval", {}).items())
                ra = [(d, float(v)) for d, v in ra] or None
        rb = None
        if sb == "here":
            if fb == "ridge_half" and "ridge_half" in results and "per_design_mae" in results["ridge_half"]:
                rb = sorted(results["ridge_half"]["per_design_mae"].items())
        else:
            if fb in p1recs:
                per_seed = [p1recs[f"{fb}_s{s}"] for s in args.seeds if f"{fb}_s{s}" in p1recs]
                if len(per_seed) == len(args.seeds):
                    designs = sorted(per_seed[0])
                    rb = [(d, float(np.mean([r[d] for r in per_seed]))) for d in designs]
            elif fb == "ridge":
                rb = sorted(p1recs.get("ridge_interval", {}).items())
                rb = [(d, float(v)) for d, v in rb] or None
        if ra is None or rb is None:
            comps[tag] = {"status": "incomplete (missing family records)"}
            continue
        try:
            comps[tag] = paired_design_comparison(ra, rb)
        except ValueError as exc:
            comps[tag] = {"status": f"incomparable: {exc}"}
    results["comparisons"] = comps
    _save_results(results, args.out)
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
    P("PHASE2_DONE")


if __name__ == "__main__":
    main()
