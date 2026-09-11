#!/usr/bin/env python3
"""P4 review-evidence audit: rejection, clamping, timebase, sharpness (§9 step 1).

Reads the saved dataset + results WITHOUT retraining and reports whether the
evidence contract holds. Works on legacy datasets (provenance missing → gap
reported, not a crash); regenerated datasets include per-clip dt/u_crit/clamp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _load_meta(dataset_dir: Path) -> dict:
    with open(dataset_dir / "metadata.json") as f:
        return json.load(f)


def main(dataset_dir="p4_dataset", results="p4_train_results.json"):
    d = Path(dataset_dir)
    meta = _load_meta(d)
    print("== P4 evidence-contract audit ==")
    print(f"dataset: {d}")

    # §4.3 rejections
    n_skip = meta.get("n_skipped_designs", meta.get("n_rejections", "?"))
    rej = meta.get("rejections", None)
    print(f"\n[§4.3] skipped designs: {n_skip}")
    if rej is None:
        print("  GAP: no per-design rejection ledger (regenerate with new generator).")
    else:
        from collections import Counter
        print(f"  ledger entries: {len(rej)}; phases: {dict(Counter(r.get('phase') for r in rej))}")

    # §4.1/4.2/4.4 provenance
    z_path = d / "metadata_arrays.npz"
    prov_keys = ("dts", "durations", "u_crits", "clamp_fracs", "max_log_amps", "alphas", "omega_crits",
                 "label_alphas", "label_omegas", "label_represented", "sat_fracs")
    if z_path.exists():
        with np.load(z_path, allow_pickle=True) as z:
            have = [k for k in prov_keys if k in z]
            missing = [k for k in prov_keys if k not in z]
        print(f"\n[§4.1/4.2/4.4] provenance arrays: have {have}; missing {missing}")
        if "dts" in have:
            with np.load(z_path, allow_pickle=True) as z:
                dts, durs = np.asarray(z["dts"], dtype=float), np.asarray(z["durations"], dtype=float)
                cf = np.asarray(z["clamp_fracs"], dtype=float)
            print(f"  dt range: [{np.nanmin(dts):.6f}, {np.nanmax(dts):.6f}] s "
                  f"(ratio {np.nanmax(dts)/max(np.nanmin(dts),1e-12):.1f}x across clips)")
            print(f"  duration range: [{np.nanmin(durs):.4f}, {np.nanmax(durs):.4f}] s")
            print(f"  clips with clamping: {np.mean(cf > 0):.3f}; mean clamp_frac: {np.mean(cf):.4f}; "
                  f"max: {np.max(cf):.4f}")
            if float(np.mean(cf > 0)) > 0.2:
                print("  FLAG: >20% clamped clips — network may learn saturation, see §4.2.")
            if "sat_fracs" in have:
                with np.load(z_path, allow_pickle=True) as z:
                    sf = np.asarray(z["sat_fracs"], dtype=float)
                print(f"  clips saturated (±SAT_LIMIT): {np.mean(sf > 0):.3f}; mean sat_frac: {np.mean(sf):.4f}; "
                      f"max: {np.max(sf):.4f}")
                if float(np.mean(sf > 0)) > 0.2:
                    print("  FLAG: >20% saturated clips — ADC range may be choking informative dynamics.")
        if "label_represented" in have:
            with np.load(z_path, allow_pickle=True) as z:
                rep = np.asarray(z["label_represented"], dtype=bool)
                la = np.asarray(z["label_alphas"], dtype=float)
                lo = np.asarray(z["label_omegas"], dtype=float)
                ra = np.asarray(z["alphas"], dtype=float)
            print(f"  label-path represented: {np.mean(rep):.3f}; unresolved: {np.mean(~np.isfinite(la)):.3f}")
            print(f"  mean |label_alpha - retained_alpha|: {np.nanmean(np.abs(la - ra)):.3e}")
            osc = lo > 0.0
            print(f"  label driver oscillatory: {np.mean(osc):.3f} (False ⇒ static-divergence label with no transient signature)")
            if float(np.mean(rep)) < 0.95:
                print("  FLAG: <95% label representation — margin labels and signal content diverge, see finding 3.")
            # Caveat 1: spectrum representation is frequency-dominated; check
            # stability-sign agreement separately (1e-6 1/s deadband).
            fin = np.isfinite(la)
            marg = (np.abs(la) <= 1e-6) | (np.abs(ra) <= 1e-6)
            dis = fin & ~marg & (np.sign(la) != np.sign(ra))
            print(f"  stability-sign disagreement (outside deadband): {np.mean(dis):.4f}")
            if float(np.mean(dis)) > 0.01:
                print("  FLAG: >1% sign disagreement — labels and signal disagree on growth/decay.")
    # Caveat 2: ledger/headline skip-count agreement.
    if rej is not None and "n_skipped_designs" in meta:
        if int(meta["n_skipped_designs"]) != len(rej):
            print(f"\n[§4.3] FLAG: n_skipped_designs={meta['n_skipped_designs']} != ledger {len(rej)}")
        if meta.get("manifest"):
            print(f"  manifest: {json.dumps(meta['manifest'], default=str)[:300]}...")
        else:
            print("  GAP: no experiment manifest (regenerate with new generator).")
    else:
        print("\n[§4.1/4.2/4.4] GAP: metadata_arrays.npz not found.")

    # Sharpness: model interval vs constant training-quantile interval (§2 of review)
    r_path = Path(results)
    if r_path.exists():
        with open(r_path) as f:
            res = json.load(f)
        print("\n[sharpness] saved test metrics:")
        for name in ("quantile_s0", "huber_s0", "constant_median", "velocity_linear", "growth_rate"):
            m = res.get(name, {})
            cov = (m.get("coverage") or {}).get("interval_0.05_0.95", float("nan"))
            print(f"  {name:<16} MAE={m.get('mae', float('nan')):.4f} "
                  f"width={m.get('mean_interval_width', float('nan')):.4f} cov={cov:.3f} "
                  f"mae_dr={m.get('mae_dr', float('nan')):.4f}")
        # Constant-interval reference from TRAINING margins
        with np.load(z_path, allow_pickle=True) as z2:
            margins = np.asarray(z2["margins"], dtype=float)
            dids = np.asarray([str(x) for x in np.asarray(z2["design_ids"]).tolist()])
        train_ids = set(meta.get("train_designs", []))
        test_ids = set(meta.get("test_designs", []))
        tr = margins[np.isin(dids, list(train_ids))] if train_ids else margins
        te = margins[np.isin(dids, list(test_ids))] if test_ids else margins
        lo, hi = np.quantile(tr, [0.05, 0.95])
        print(f"  constant train-q05/q95 interval: [{lo:.5f}, {hi:.5f}] width={hi-lo:.5f} "
              f"test coverage={np.mean((te >= lo) & (te <= hi)):.4f}")
        print("  Rule: ~90% model coverage is compelling only with better sharpness/decisions.")
    else:
        print(f"\n[sharpness] results file {r_path} not found.")

    print("\nDone. Full temporal-warning + mismatch experiments still require "
          "regenerated data + rolling-window evaluation (review §9 steps 3-5).")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3] if len(sys.argv) > 2 else ()))
