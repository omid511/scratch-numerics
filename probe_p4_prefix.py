#!/usr/bin/env python3
"""Prefix-length probe: how does the trained quantile margin model degrade on
short observation windows? (Review Directions B/D, §9 step 3 — lightweight
version on saved artifacts, no retraining.)

Loads the expanded-dataset test split + a trained quantile checkpoint,
evaluates median MAE / q05–q95 coverage / width on full clips and on leading
prefixes. Prefixes are reported in samples; conversion to seconds needs the
per-clip dt that only regenerated datasets carry (§4.1).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from train_p4_expanded import build_clips, load_dataset, with_ones_mask
from mechanics.p4_margin_estimation.quantile_head import QuantileMarginModel


def _metrics(model, clips, prefix=None, batch=256):
    sub = []
    for c in clips:
        sig = np.asarray(c.sensor_signals)
        if prefix is not None:
            sig = sig[:, :prefix]
        ns = type(c).__new__(type(c))
        # Shallow-copy the lightweight _Clip (slots) with truncated signals.
        ns.sensor_signals, ns.margin, ns.velocity, ns.design_id = (
            sig, c.margin, c.velocity, c.design_id)
        ns.dt, ns.time = getattr(c, "dt", None), None
        sub.append(ns)
    y_true, med, lo, hi = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(sub), batch):
            Xb = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32)
                              for c in sub[s:s + batch]])
            out = model(Xb)
            lo.append(out[:, 0].cpu().numpy())
            med.append(out[:, 1].cpu().numpy())
            hi.append(out[:, 2].cpu().numpy())
            y_true.append(np.array([c.margin for c in sub[s:s + batch]]))
    y, m, l, h = (np.concatenate(a) for a in (y_true, med, lo, hi))
    return {"n": len(y), "mae": float(np.mean(np.abs(m - y))),
            "cov90": float(np.mean((y >= l) & (y <= h))),
            "width": float(np.mean(h - l))}


def main(dataset_dir="p4_dataset", ckpt="p4_quantile_s0.pt", k_test=200):
    clips_arr, margins, vels, dids, _, meta = load_dataset(dataset_dir)
    test_ids = set(meta["test_designs"])
    prov = meta.get("provenance", {})
    dts = prov.get("dts") if isinstance(prov, dict) else None
    test_clips, _ = build_clips(clips_arr, margins, vels, dids, test_ids, dts=dts)
    test_clips = [c for c in test_clips if np.isfinite(c.margin)][:k_test]
    test_clean = with_ones_mask(test_clips)
    n_ch = test_clean[0].sensor_signals.shape[0]
    model = QuantileMarginModel(n_channels=n_ch, hidden_dim=32, n_layers=9)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    print(f"prefix probe: {len(test_clean)} test clips, ckpt={ckpt}")
    print(f"{'prefix':>8} {'MAE':>8} {'cov90':>8} {'width':>8}")
    for p in (64, 128, 256, 512):
        m = _metrics(model, test_clean, prefix=p)
        print(f"{p:>8} {m['mae']:>8.4f} {m['cov90']:>8.3f} {m['width']:>8.4f}")
    print("Note: prefixes reuse full-clip causal normalization; a deployed "
          "short-window system would renormalize from its own short calibration "
          "window, so this bounds window-dependence rather than certifying it.")


if __name__ == "__main__":
    main(*sys.argv[1:4] if len(sys.argv) > 3 else ())
