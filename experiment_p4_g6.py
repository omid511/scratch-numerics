#!/usr/bin/env python3
"""Frozen-backbone probes + leave-one-out on survivors (additive only).

Q1 (frozen probe): fix GRU backbone+summary from converged plain_clean
seed-0, train ONLY side-MLP+head 3 epochs per member subset. Separates
member signal from representation co-adaptation. If a member helps here
but hurt solo-full, the solo harm was backbone corruption, not junk.
Q2 (leave-one-out): full-bundle minus each survivor, 3 epochs, seed 0.
Survivors = members whose solo didn't catastrophically harm AND ridge
screen top: kurt, env_slope, energy, sat_frac, power_ratio, zcr, freq2.

Subsets: each solo (7) + bundle16 + no-kurt + no-env_slope + top4
(kurt,env_slope,energy,sat_frac) + top4+freq2. 12 configs x 3 epochs.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from experiment_p4_phase2 import (
    GRUEllQuantileSidecar, _predict_sidecar, _stack_valid_with_clips,
    train_quantile_generic,
)
from experiment_p4_saturation import (
    _atomic_json_dump, _fingerprint_dataset, _load_or_initialize_results,
    _make_cap_parts, _record_is_complete, _relative_path,
    _save_torch_checkpoint, _validate_dataset, _validate_seeds,
    ALPHA, MODEL_NAME, N_MODEL_CHANNELS,
)
from mechanics.p4_margin_estimation import sidecar as _sc
from mechanics.p4_margin_estimation.quantile_head import (
    apply_cqr_adjustment, fit_cqr_adjustment,
)
from train_p4_expanded import with_ones_mask

GROUP = "G6"
CAP = 50.0
SUBSETS = {
    "solo-kurt": ["kurt"], "solo-env_slope": ["env_slope"],
    "solo-energy": ["energy"], "solo-sat_frac": ["sat_frac"],
    "solo-power_ratio": ["power_ratio"], "solo-zcr": ["zcr"],
    "solo-freq2": ["freq2"],
    "bundle16": _sc.member_names("G4"),
    "no-kurt": [m for m in _sc.member_names("G4") if m != "kurt"],
    "no-env_slope": [m for m in _sc.member_names("G4") if m != "env_slope"],
    "top4": ["kurt", "env_slope", "energy", "sat_frac"],
    "top4-freq2": ["kurt", "env_slope", "energy", "sat_frac", "freq2"],
}
FROZEN_CKPT = Path("p4_groups/g1b/checkpoints/g1b_plain_clean_seed0_traincap50.pt")


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
    torch.set_num_threads(int(args.threads))
    bundle = _validate_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets = [s for s in str(getattr(args, "subsets", "")).split(",") if s] or sorted(SUBSETS)
    for s in subsets:
        if s not in SUBSETS:
            raise ValueError(f"unknown subset {s!r}")
    config = {"group": GROUP, "subsets": subsets, "cap": CAP, "seed": 0,
              "epochs": int(args.epochs), "frozen_ckpt": str(FROZEN_CKPT),
              "threads": int(args.threads), "alpha": ALPHA,
              "bootstrap": int(args.bootstrap)}
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records = results["records"]

    parts = _make_cap_parts(bundle, CAP, need_clean=True)
    train_clean = with_ones_mask(parts["train"]["raw"])
    select_clean = parts["select"]["clean"]
    calib_clean = parts["calib"]["clean"]
    test_clean = parts["test"]["clean"]
    cy = np.asarray([float(c.margin) for c in calib_clean], dtype=float)
    y = np.asarray([float(c.margin) for c in test_clean], dtype=float)

    frozen = torch.load(FROZEN_CKPT, map_location="cpu", weights_only=True)
    from torch.utils.data import DataLoader, TensorDataset
    from mechanics.p4_margin_estimation.quantile_head import (
        SUPPORTED_QUANTILES, pinball_loss,
    )
    from experiment_p4_phase2 import QW

    for sub in subsets:
        key = f"{GROUP}__{sub}__seed0"
        if _record_is_complete(records.get(key), output_dir):
            continue
        names = SUBSETS[sub]
        stats = _sc.standardize_fit(_sc.extract(train_clean, names))
        tr_s = _sc.standardize_apply(_sc.extract(train_clean, names), stats)
        va_s = _sc.standardize_apply(_sc.extract(select_clean, names), stats)
        ca_s = _sc.standardize_apply(_sc.extract(calib_clean, names), stats)
        te_s = _sc.standardize_apply(_sc.extract(test_clean, names), stats)
        X, ytr, _ = _stack_valid_with_clips(train_clean, N_MODEL_CHANNELS)
        Xv, yva, _ = _stack_valid_with_clips(select_clean, N_MODEL_CHANNELS)
        S = torch.tensor(np.asarray(tr_s, dtype=np.float32))
        Sv = torch.tensor(np.asarray(va_s, dtype=np.float32))
        torch.manual_seed(0)
        model = GRUEllQuantileSidecar(N_MODEL_CHANNELS, n_side=len(names))
        # transplant frozen backbone+summary (shapes identical by construction)
        own = model.state_dict()
        own["gru.weight_ih_l0"] = frozen["gru.weight_ih_l0"].clone()
        own["gru.weight_hh_l0"] = frozen["gru.weight_hh_l0"].clone()
        own["gru.bias_ih_l0"] = frozen["gru.bias_ih_l0"].clone()
        own["gru.bias_hh_l0"] = frozen["gru.bias_hh_l0"].clone()
        model.load_state_dict(own, strict=False)
        for p in list(model.gru.parameters()):
            p.requires_grad = False
        # summary has no params (windowing); head+side train
        optim = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        dl = DataLoader(TensorDataset(X, S, ytr), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
        t0 = time.time()
        for _ in range(int(args.epochs)):
            model.train()
            for xb, sb, yb in dl:
                loss = pinball_loss(model(xb, sb), yb, SUPPORTED_QUANTILES, weights=QW)
                optim.zero_grad()
                loss.backward()
                optim.step()
        model.eval()
        pred = np.asarray(_predict_sidecar(model, test_clean, te_s, batch=256, device="cpu"))
        med = pred[:, 1]
        cpred = np.asarray(_predict_sidecar(model, calib_clean, ca_s, batch=256, device="cpu"))
        adj = float(fit_cqr_adjustment(cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA))
        lo, hi = apply_cqr_adjustment(pred[:, 0], pred[:, 2], adj)
        mae = float(np.mean(np.abs(med - y)))
        cov = float(np.mean((y >= lo) & (y <= hi)))
        ckpt = output_dir / "checkpoints" / f"g6_{sub}_seed0.pt"
        from experiment_p4_saturation import _save_predictions
        digest = _save_torch_checkpoint(model, ckpt)
        pred_path = output_dir / "predictions" / f"{key}.npz"
        pred_digest = _save_predictions(pred_path, {"y_true": y, "median": med,
                                                    "raw_lower": pred[:, 0],
                                                    "raw_upper": pred[:, 2],
                                                    "cqr_lower": lo, "cqr_upper": hi})
        records[key] = {"status": "complete", "model": MODEL_NAME + "_frozenprobe",
                        "seed": 0, "cap": CAP, "subset": sub, "members": names,
                        "mae": mae, "coverage": cov,
                        "width": float(np.mean(hi - lo)),
                        "checkpoint": str(ckpt.relative_to(output_dir)),
                        "checkpoint_sha256": digest,
                        "prediction_file": str(pred_path.relative_to(output_dir)),
                        "prediction_sha256": pred_digest,
                        "test_corr_slope": _corr_slope(y, med),
                        "train_time_seconds": time.time() - t0}
        _atomic_json_dump(results, results_path)
    print(f"G6_DONE records={len(records)} output={output_dir}", flush=True)
    return results


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="G6: frozen-backbone probes + leave-one-out")
    p.add_argument("--dataset", "--dataset-dir", dest="dataset",
                   default=os.environ.get("P4_DATASET_DIR", "p4_dataset"))
    p.add_argument("--output-dir", "--out", dest="output_dir", default="p4_groups/g6")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--subsets", default=",".join(sorted(SUBSETS)))
    return p.parse_args(argv)


def main(argv=None):
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
