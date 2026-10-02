#!/usr/bin/env python3
"""P4 multi-velocity end-to-end u_crit training: K clips -> shared GRU -> attention -> u_crit.

New training round on the full data with new code and new features. Additive:
writes only to its own --output-dir with isolated checkpoint names, never
touches p4_saturation_full/, p4_targets_full/, or p4_multivel_full/.

Design
------
- Input: K=4 clips of one (design, realization) unit at different velocities.
  Each clip keeps its known velocity (legitimate: velocity is always known in
  deployment; u_crit is what's unknown). Units never split across partitions
  (inherited design splits).
- Model: shared GRU encoder (same GRUEllQuantile backbone, trained from
  scratch -- the margin-pretrained encoder is invariant to u_crit) over each
  clip -> per-clip embeddings + standardized velocity -> single-head attention
  -> pooled vector -> scalar u_crit (standardized with train stats).
- Loss: MSE on standardized u_crit. Selection on select-unit loss, CQR-style
  residual band on calib units for reporting only.
- Baselines in-report: train-mean, frozen-encoder inversion numbers are
  referenced from p4_multivel_full (not recomputed).

Usage::

    PYTHONPATH=src python experiment_p4_multivel_e2e.py \
        --dataset p4_dataset_saturation --output-dir p4_multivel_e2e \
        --seeds 0 1 2 --epochs 20 --threads 2
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from experiment_p4_phase2 import GRUEllQuantile
from experiment_p4_saturation import (
    _atomic_json_dump,
    _fingerprint_dataset,
    _load_or_initialize_results,
    _save_predictions,
    _save_torch_checkpoint,
    _sha256_file,
    _make_cap_parts,
    _relative_path,
    _validate_dataset,
    _validate_seeds,
    MODEL_NAME,
)
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from train_p4_expanded import apply_dr_with_mask, load_dataset

CAP = 50.0
K_CLIPS = 4
HIDDEN = 86


class UnitUcritModel(nn.Module):
    """Shared GRU encoder + velocity-aware attention pooling -> u_crit.

    Ablations (same skeleton, different information):
    - waveform_velocity: full model (default).
    - velocity_only: waveforms zeroed at the door; isolates what velocities
      alone can do under identical optimization.
    - waveform_only: velocities zeroed; isolates waveform content.
    """

    def __init__(self, n_channels: int = 16, hidden_dim: int = HIDDEN, attn_dim: int = 64,
                 ablation: str = "waveform_velocity"):
        super().__init__()
        if ablation not in ("waveform_velocity", "velocity_only", "waveform_only"):
            raise ValueError(f"unknown ablation {ablation!r}")
        self.ablation = ablation
        from mechanics.p4_margin_estimation.baselines import GRUMarginModel
        from mechanics.p4_margin_estimation.tcn import TemporalSummary
        base = GRUMarginModel(n_channels, hidden_dim)
        self.gru = base.gru
        self.summary = TemporalSummary(window=64)
        self.attn_q = nn.Linear(3 * hidden_dim + 1, attn_dim)
        self.attn_k = nn.Linear(3 * hidden_dim + 1, attn_dim)
        self.attn_v = nn.Linear(3 * hidden_dim + 1, attn_dim)
        self.out = nn.Linear(attn_dim, 1)

    def forward(self, clips: torch.Tensor, vels: torch.Tensor) -> torch.Tensor:
        # clips: (B, K, C, T), vels: (B, K) globally standardized externally
        if self.ablation == "velocity_only":
            clips = torch.zeros_like(clips)
        if self.ablation == "waveform_only":
            vels = torch.zeros_like(vels)
        B, K, C, T = clips.shape
        x = clips.reshape(B * K, C, T).transpose(1, 2)
        out, _ = self.gru(x)
        feat = self.summary(out.transpose(1, 2))  # (B*K, 3H)
        z = torch.cat([feat, vels.reshape(B * K, 1)], dim=1).reshape(B, K, -1)
        Q, Kk, Vv = self.attn_q(z), self.attn_k(z), self.attn_v(z)
        w = torch.softmax(Q @ Kk.transpose(1, 2) / math.sqrt(Q.shape[-1]), dim=-1)
        pooled = (w @ Vv).mean(dim=1)
        return self.out(pooled).squeeze(-1)

def _unit_ucrit(bundle: dict[str, Any]) -> dict[tuple[str, int], float]:
    _, _, _, design_ids, realization_ids, meta = load_dataset(str(bundle["dataset_dir"]))
    ucrits = np.asarray(meta["provenance"]["u_crits"], dtype=float)
    out: dict[tuple[str, int], float] = {}
    for did, rid, u in zip([str(x) for x in design_ids],
                           [int(x) for x in realization_ids], ucrits.tolist()):
        key = (did, int(rid))
        if key in out and out[key] != float(u):
            raise ValueError(f"u_crit varies within {key}")
        out[key] = float(u)
    return out
def _pack_unit(clips: list[Any], k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    take = rng.choice(len(clips), size=k, replace=False)
    sig = np.stack([np.asarray(clips[int(j)].sensor_signals, dtype=np.float32) for j in take])
    vel = np.asarray([float(clips[int(j)].velocity) for j in take], dtype=np.float32)
    return sig, vel


def _build_tensors(parts_split: list[Any], truth: dict[tuple[str, int], float],
                   k: int, seed: int, stats: dict[str, float] | None = None,
                   fit_stats: bool = False) -> tuple[dict[str, Any], dict[str, float]]:
    units: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for c in parts_split:
        units[(str(c.design_id), int(c.realization_idx))].append(c)
    units = {u: v for u, v in units.items() if len(v) >= k}
    if not units:
        raise ValueError("no units with >= K clips")
    rng = np.random.default_rng(seed)
    S, V, Y, ids = [], [], [], []
    for u, clips in sorted(units.items()):
        sig, vel = _pack_unit(clips, k, rng)
        S.append(sig)
        V.append(vel)
        Y.append(truth[u])
        ids.append(f"{u[0]}:{u[1]}")
    S = np.stack(S).astype(np.float32)  # (U, K, C, T)
    V = np.stack(V).astype(np.float32)
    Y = np.asarray(Y, dtype=float)
    if fit_stats:
        stats = {"mean": float(Y.mean()), "std": float(Y.std() + 1e-8),
                 "vel_mean": float(V.mean()), "vel_std": float(V.std() + 1e-8)}
    assert stats is not None
    Vn = ((V - float(stats["vel_mean"])) / float(stats["vel_std"])).astype(np.float32)
    Yn = (Y - stats["mean"]) / stats["std"]
    return ({"S": S, "V": V, "Vn": Vn, "Y": Y, "Yn": Yn.astype(np.float32),
             "ids": ids, "vel_stats": (float(stats["vel_mean"]), float(stats["vel_std"]))}, stats)


def _train_unit_model(train: dict[str, Any], select: dict[str, Any], stats: dict[str, float],
                      *, seed: int, epochs: int, batch_size: int, device: str,
                      n_channels: int, state_path: Path,
                      ablation: str = "waveform_velocity") -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UnitUcritModel(n_channels=n_channels, ablation=ablation).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    S = torch.tensor(train["S"], dtype=torch.float32)
    Vn = torch.tensor(train["Vn"], dtype=torch.float32)
    Yn = torch.tensor(train["Yn"], dtype=torch.float32)
    Sv = torch.tensor(select["S"], dtype=torch.float32)
    Vnv = torch.tensor(select["Vn"], dtype=torch.float32)
    Ynv = torch.tensor(select["Yn"], dtype=torch.float32)
    n = len(S)
    best, best_state, hist = float("inf"), None, {"train_loss": [], "select_loss": []}
    start_epoch = 0
    if state_path.is_file():
        try:
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            if isinstance(saved, dict) and saved.get("completed_epochs", 0) > 0:
                model.load_state_dict(saved["model"])
                optim.load_state_dict(saved["optimizer"])
                sched.load_state_dict(saved["scheduler"])
                best = float(saved["best"])
                best_state = saved["best_model"]
                hist = {"train_loss": list(saved["history"]["train_loss"]),
                        "val_loss": list(saved["history"]["val_loss"])}
                start_epoch = int(saved["completed_epochs"])
        except Exception:
            start_epoch, best, best_state = 0, float("inf"), None
            hist = {"train_loss": [], "select_loss": []}
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb, vb, yb = S[idx].to(device), Vn[idx].to(device), Yn[idx].to(device)
            loss = nn.functional.mse_loss(model(xb, vb), yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        train_loss = tot / max(nb, 1)
        model.eval()
        with torch.no_grad():
            sel_pred, sel_tot, sel_nb = [], 0.0, 0
            for s in range(0, len(Sv), batch_size):
                sl = slice(s, s + batch_size)
                pv = model(Sv[sl].to(device), Vnv[sl].to(device))
                sel_tot += float(nn.functional.mse_loss(
                    pv, Ynv[sl].to(device), reduction="sum").item())
                sel_nb += len(pv)
                sel_pred.append(pv.cpu())
            select_loss = sel_tot / max(sel_nb, 1)
        hist["train_loss"].append(train_loss)
        hist["select_loss"].append(select_loss)
        if select_loss < best:
            best = select_loss
            best_state = {k2: v.cpu().clone() for k2, v in model.state_dict().items()}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name(state_path.name + ".tmp")
        with open(tmp, "wb") as f:
            torch.save({"schema_version": 1, "completed_epochs": epoch + 1,
                        "model": {k2: v.cpu().clone() for k2, v in model.state_dict().items()},
                        "optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
                        "best": float(best), "best_model": best_state,
                        "history": {"train_loss": list(hist["train_loss"]),
                                    "val_loss": list(hist["select_loss"])},
                        "stats": stats, "seed": seed, "epochs": epochs}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_select_loss": best, "train_time": time.time() - t0}


def _ckpt_names(output_dir: Path, seed: int, ablation: str) -> tuple[Path, Path]:
    tag = str(ablation).replace(" ", "_")
    ckpt = output_dir / "checkpoints" / f"multivel_e2e_{tag}_seed{int(seed)}_k{K_CLIPS}.pt"
    state = output_dir / "checkpoints" / f"multivel_e2e_{tag}_seed{int(seed)}_k{K_CLIPS}.training.pt"
    return ckpt, state


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(int(s) for s in _validate_seeds(args.seeds))
    ablation = str(getattr(args, "ablation", "waveform_velocity"))
    if ablation not in ("waveform_velocity", "velocity_only", "waveform_only"):
        raise ValueError(f"unknown ablation {ablation!r}")
    torch.set_num_threads(int(args.threads))
    bundle = _validate_dataset(args.dataset)
    config = {"cap": CAP, "k_clips": K_CLIPS, "seeds": [int(s) for s in seeds],
              "epochs": int(args.epochs), "batch_size": int(args.batch_size),
              "predict_batch": int(args.predict_batch), "threads": int(args.threads),
              "model": "multivel_e2e_attention", "bootstrap": int(args.bootstrap),
              "ablation": ablation}
    output_dir = Path(args.output_dir)
    for old in ("p4_saturation_full", "p4_targets_full", "p4_multivel_full", "p4_multivel_e2e"):
        if output_dir.resolve() == (_ROOT / old).resolve():
            raise ValueError(f"--output-dir must not overwrite {old}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records: dict[str, Any] = results["records"]
    truth = _unit_ucrit(bundle)

    parts = _make_cap_parts(bundle, CAP, need_clean=False)
    train_dr, n_channels = apply_dr_with_mask(parts["train"]["raw"], seed=0)
    from train_p4_expanded import with_ones_mask
    parts_train = {"raw": train_dr}
    val_raw = parts["val"]["raw"]
    from train_p4_expanded import _split_select_calibrate
    select_ids, calib_ids = _split_select_calibrate(sorted(bundle["splits"]["val"]))
    select_raw = with_ones_mask([c for c in val_raw if str(c.design_id) in select_ids])
    calib_raw = with_ones_mask([c for c in val_raw if str(c.design_id) in calib_ids])
    test_raw = with_ones_mask(parts["test"]["raw"])

    stats: dict[str, float] | None = None
    train_t, stats = _build_tensors(parts_train["raw"], truth, K_CLIPS, seed=999, fit_stats=True)
    assert stats is not None
    select_t, _ = _build_tensors(select_raw, truth, K_CLIPS, seed=1001, stats=stats)
    calib_t, _ = _build_tensors(calib_raw, truth, K_CLIPS, seed=1002, stats=stats)
    test_t, _ = _build_tensors(test_raw, truth, K_CLIPS, seed=1003, stats=stats)
    train_mean = float(stats["mean"])
    scale = float(stats["std"])

    allpred: dict[int, np.ndarray] = {}
    for seed in seeds:
        key = f"multivel_e2e_{ablation}__seed{int(seed)}"
        ckpt_path, state_path = _ckpt_names(output_dir, seed, ablation)
        if (records.get(key) or {}).get("status") == "complete":
            try:
                if Path(records[key]["prediction_file"]).is_file() or \
                        (output_dir / records[key]["prediction_file"]).is_file():
                    p = records[key]["prediction_file"]
                    z = np.load(output_dir / p if not Path(p).is_absolute() else Path(p))
                    allpred[seed] = np.asarray(z["pred"], dtype=float)
                    continue
            except Exception:
                pass
        model, info = _train_unit_model(
            train_t, select_t, stats, seed=int(seed), epochs=int(args.epochs),
            batch_size=int(args.batch_size), device="cpu",
            n_channels=int(n_channels), state_path=state_path, ablation=ablation)
        digest = _save_torch_checkpoint(model, ckpt_path)
        St = torch.tensor(test_t["S"], dtype=torch.float32)
        Vt = torch.tensor(test_t["Vn"], dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for s in range(0, len(St), int(args.predict_batch)):
                preds.append(model(St[s:s + int(args.predict_batch)],
                                   Vt[s:s + int(args.predict_batch)]).cpu().numpy())
        pred_std = np.concatenate(preds).ravel() if preds else np.zeros(0)
        pred = pred_std * scale + train_mean
        y = test_t["Y"]
        err = np.abs(pred - y)
        calib_pred = None
        with torch.no_grad():
            Sc = torch.tensor(calib_t["S"], dtype=torch.float32)
            Vc = torch.tensor(calib_t["Vn"], dtype=torch.float32)
            cp = []
            for s in range(0, len(Sc), int(args.predict_batch)):
                cp.append(model(Sc[s:s + int(args.predict_batch)],
                                Vc[s:s + int(args.predict_batch)]).cpu().numpy())
            calib_pred = (np.concatenate(cp).ravel() if cp else np.zeros(0)) * scale + train_mean
        calib_err = np.abs(calib_pred - calib_t["Y"])
        band = float(np.quantile(calib_err, 0.9)) if len(calib_err) >= 2 else float("nan")
        pred_path = output_dir / "predictions" / f"{key}.npz"
        pred_digest = _save_predictions(pred_path, {
            "y_true": y, "pred": pred, "abs_err": err,
            "unit_ids": np.asarray(test_t["ids"]),
            "calib_abs_err": calib_err, "residual_band_90": np.asarray([band])})
        records[key] = {
            "status": "complete", "model": "multivel_e2e_attention", "seed": int(seed),
            "ablation": ablation,
            "cap": CAP, "k_clips": K_CLIPS,
            "n_train_units": len(train_t["Y"]), "n_select_units": len(select_t["Y"]),
            "n_calib_units": len(calib_t["Y"]), "n_test_units": len(y),
            "mae_physical_ms": float(err.mean()),
            "train_mean_baseline_ms": float(np.mean(np.abs(y - train_mean))),
            "residual_band_90_ms": band,
            "ucrit_standardization": dict(stats),
            "checkpoint": _relative_path(ckpt_path, output_dir),
            "checkpoint_sha256": digest,
            "prediction_file": _relative_path(pred_path, output_dir),
            "prediction_sha256": pred_digest,
            "training": {"epochs": int(args.epochs), "batch_size": int(args.batch_size),
                         "final_train_loss": float(info["history"]["train_loss"][-1]),
                         "final_select_loss": float(info["history"]["select_loss"][-1]),
                         "train_time_seconds": float(info["train_time"])},
        }
        _atomic_json_dump(results, results_path)
        allpred[seed] = pred
        del model

    y = test_t["Y"]
    mean_pred = np.mean([allpred[s] for s in seeds], axis=0)
    comparisons: dict[str, Any] = {}
    base = np.full_like(y, train_mean)
    for name, av, bv in (
        ("mean_seed_minus_train_mean", mean_pred, base),
    ):
        a = [(f"u{i}", float(abs(av[i] - y[i]))) for i in range(len(y))]
        b = [(f"u{i}", float(abs(bv[i] - y[i]))) for i in range(len(y))]
        comparisons[name] = {"metric": "unit_abs_err", "subtraction": "left - right",
                             "status": "complete",
                             "result": paired_design_comparison(
                                 a, b, n_bootstrap=int(args.bootstrap), seed=0)}
    results["comparisons"] = comparisons
    results["summary"] = {
        "n_test_units": len(y),
        "train_mean_baseline_ms": float(np.mean(np.abs(y - train_mean))),
        "mean_seed_mae_ms": float(np.mean(np.abs(mean_pred - y))),
        "per_seed_mae_ms": {str(int(s)): float(np.mean(np.abs(allpred[s] - y))) for s in seeds},
    }
    _atomic_json_dump(results, results_path)
    print(f"MULTIVELE2E_DONE units={len(y)} output={output_dir}", flush=True)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P4 multi-velocity end-to-end u_crit training")
    p.add_argument("--dataset", "--dataset-dir", dest="dataset",
                   default=os.environ.get("P4_DATASET_DIR", "p4_dataset"))
    p.add_argument("--output-dir", "--out", dest="output_dir", default="p4_multivel_e2e")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--predict-batch", type=int, default=16)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--ablation", default="waveform_velocity",
                   choices=("waveform_velocity", "velocity_only", "waveform_only"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
