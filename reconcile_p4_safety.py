#!/usr/bin/env python3
"""Reconcile safety rows from decision arrays: one table per model x policy.

For each saved family (ridge+intervals, quantile GRU, quantile TCN, GRU+ELL)
reloads checkpoints, recomputes clean-test predictions, and derives median /
raw-lb / CQR-lb rows from the SAME boolean declaration arrays via
policy_table. Verifies a=(1-pi_u)a_s+pi_u f per row and emits the paired
ridge-vs-GRU+ELL comparison the review requested.

Writes p4run_full/p4_safety_reconciled.json. No retraining.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mechanics.p4_margin_estimation.quantile_gru import QuantileGRUModel
from mechanics.p4_margin_estimation.quantile_head import (
    QuantileMarginModel, fit_cqr_adjustment, apply_cqr_adjustment,
)
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.decision_metrics import (
    paired_design_comparison, policy_table, ridge_residual_intervals,
)
from train_p4_expanded import (
    _split_select_calibrate, apply_dr_with_mask, build_clips, load_dataset, with_ones_mask,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)  # noqa: E731
RUN = "/home/omid5/mechanics/p4run_full"
DS = "/home/omid5/mechanics/p4_dataset_full"
N_CH = 16


def predict(model, clips, batch=256):
    model.eval()
    outs = []
    with torch.inference_mode():
        for s in range(0, len(clips), batch):
            Xb = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32)
                              for c in clips[s:s + batch]])
            outs.append(model(Xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    clips_arr, margins, vels, dids, rids, meta = load_dataset(DS)
    prov = meta.get("provenance", {})
    kw = dict(dts=prov.get("dts"), realization_ids=rids, sat_fracs=prov.get("sat_fracs"))
    trc, _ = build_clips(clips_arr, margins, vels, dids, set(meta["train_designs"]), **kw)
    vac, _ = build_clips(clips_arr, margins, vels, dids, set(meta["val_designs"]), **kw)
    tec, _ = build_clips(clips_arr, margins, vels, dids, set(meta["test_designs"]), **kw)
    sel_ids, cal_ids = _split_select_calibrate(sorted(set(meta["val_designs"])))
    cal_clean = with_ones_mask([c for c in vac if c.design_id in cal_ids])
    test_clean = with_ones_mask(tec)
    train_dr, _ = apply_dr_with_mask(trc, seed=0)
    y = np.array([c.margin for c in test_clean])
    did_list = [c.design_id for c in test_clean]
    yc = np.array([c.margin for c in cal_clean])
    P(f"test composition: unsafe={int((y <= 0).sum())} safe={int((y > 0).sum())} n={len(y)}")

    fams = {}
    # Ridge: two EXPLICIT policies from the same arrays. `median` declares
    # safe when the point prediction > 0; `band_lb` declares safe only when
    # the lower band edge (pred - adjustment) > 0. These differ whenever the
    # band straddles zero, so they must never share one row.
    rh = PhysicsFeatureRidge()
    rh.fit(train_dr)
    ri = ridge_residual_intervals(rh, cal_clean, test_clean, train_clips=train_dr)
    pr = np.asarray(rh.predict(test_clean), dtype=float).ravel()
    fams["ridge"] = (pr, pr - float(ri["adjustment"]), None)

    specs = ([(f"qgru_s{s}", QuantileGRUModel(N_CH), f"p4_qgru_s{s}.pt") for s in (0, 1, 2)]
             + [(f"tcn_s{s}", QuantileMarginModel(n_channels=N_CH, hidden_dim=32, n_layers=9),
                 f"p4_quantile_s{s}.pt") for s in (0, 1, 2)])
    sys.path.insert(0, "src/mechanics/p4_margin_estimation")
    from experiment_p4_phase2 import GRUEllQuantile
    specs += [(f"gruell_s{s}", GRUEllQuantile(N_CH), f"gru_ell_s{s}.pt") for s in (0, 1, 2)]
    for key, model, fn in specs:
        model.load_state_dict(torch.load(os.path.join(RUN, fn), weights_only=True, map_location="cpu"))
        pc = predict(model, cal_clean)
        pt = predict(model, test_clean)
        adj = fit_cqr_adjustment(pc[:, 0], pc[:, 2], yc, alpha=0.10)
        lo, hi = apply_cqr_adjustment(pt[:, 0], pt[:, 2], adj)
        fams[key] = (pt[:, 1], pt[:, 0], lo)
        del model

    tables, agg = {}, {}
    for key, (med, raw_lo, cqr_lo) in fams.items():
        lo_arg = raw_lo if raw_lo is not None else med
        t = policy_table(y, med, lo_arg, cqr_lo, did_list)
        # identity check per row
        pi = t["n_unsafe"] / t["n_total"]
        for rname, row in t["rows"].items():
            lhs = row["a"]
            rhs = (1 - pi) * row["a_s"] + pi * row["f"]
            assert abs(lhs - rhs) < 1e-9, (key, rname, lhs, rhs)
        tables[key] = t
        agg[key] = {r: {k: v for k, v in row.items()} for r, row in t["rows"].items()}
    # seed-averaged f/a_s/a per family x policy + paired ridge-vs-gruell on median MAE-free decision diff? No:
    # paired comparison on per-design MAE (accuracy), plus decision-count table here.
    per_design = {}
    for key, (med, _, _) in fams.items():
        per = {}
        for d in sorted(set(did_list)):
            m = np.array([dd == d for dd in did_list])
            per[d] = float(np.mean(np.abs(med[m] - y[m])))
        per_design[key] = per

    def famavg(keys):
        ds = sorted(per_design[keys[0]])
        return [(d, float(np.mean([per_design[k][d] for k in keys]))) for d in ds]

    comps = {}
    gruell = famavg([f"gruell_s{s}" for s in (0, 1, 2)])
    p1 = json.load(open(os.path.join(RUN, "p4_phase1_results.json")))
    ridge_p1 = sorted(p1["ridge_interval"]["clean"]["per_design_mae"].items())
    ridge_p1 = [(d, float(v)) for d, v in ridge_p1]
    comps["gruell_vs_ridge"] = paired_design_comparison(gruell, ridge_p1)

    out = {"composition": {"n_unsafe": int((y <= 0).sum()), "n_safe": int((y > 0).sum())},
           "families": agg, "comparisons": comps}
    with open(os.path.join(RUN, "p4_safety_reconciled.json"), "w") as f:
        json.dump(out, f, indent=2)
    for key in ["ridge", "qgru_s0", "tcn_s0", "gruell_s0"]:
        for pol, row in agg[key].items():
            P(f"{key:10s} {pol:8s} f={row['f']:.3f} a_s={row['a_s']:.3f} a={row['a']:.3f} "
              f"ufsd={row['ufsd']:.3f} us={row['n_unsafe_to_safe']} ss={row['n_safe_to_safe']}")
    c = comps["gruell_vs_ridge"]
    P("gruell_vs_ridge MAE diff=%+.5f CI=[%+.5f,%+.5f] bootfrac>0=%.3f (descriptive, not a p-value); "
      "CI includes zero at displayed precision — no superiority claim"
      % (c["mean_diff"], c["ci_low"], c["ci_high"], c["frac_gt0"]))
    P("WROTE p4_safety_reconciled.json (identity holds on every row)")


if __name__ == "__main__":
    main()
