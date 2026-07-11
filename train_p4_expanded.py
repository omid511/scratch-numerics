#!/usr/bin/env python3
"""Train all P4 models on the expanded dataset with design-level splits."""
from __future__ import annotations
import json
import time
import numpy as np
import torch
from pathlib import Path
from mechanics.p4_margin_estimation.train import (
    train, train_huber, train_median, evaluate_coverage, HuberMarginModel,
)
from mechanics.p4_margin_estimation.baselines import (
    ConstantMedianBaseline, PhysicsFeatureRidge, VelocityLinearBaseline,
    GrowthRateBaseline,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)


class _Clip:
    __slots__ = ("sensor_signals", "margin", "velocity")
    def __init__(self, signals, margin, velocity):
        self.sensor_signals = signals
        self.margin = margin
        self.velocity = velocity


def load_dataset(dataset_dir: str = "p4_dataset"):
    d = Path(dataset_dir)
    clips_arr = np.load(d / "clips.npz")["clips"]
    margins = np.load(d / "margins.npy")
    velocities = np.load(d / "velocities.npy")
    design_ids = np.load(d / "design_ids.npy")
    realization_ids = np.load(d / "realization_ids.npy")
    with open(d / "metadata.json") as f:
        meta = json.load(f)
    return clips_arr, margins, velocities, design_ids, realization_ids, meta


def build_clips(clips_arr, margins, velocities, design_ids_all, split_ids):
    """Build clip objects for a set of design IDs."""
    clips = []
    vel_list = []
    for i, did in enumerate(design_ids_all):
        if did in split_ids:
            c = _Clip(clips_arr[i], float(margins[i]), float(velocities[i]))
            clips.append(c)
            vel_list.append(float(velocities[i]))
    return clips, vel_list


def evaluate_point(model, clips, velocities=None, device="cpu"):
    model.eval()
    X_list, y_list, valid_vels = [], [], []
    for i, c in enumerate(clips):
        if np.isnan(c.margin):
            continue
        sig = torch.tensor(c.sensor_signals, dtype=torch.float32)
        if torch.isnan(sig).any():
            continue
        X_list.append(sig)
        y_list.append(c.margin)
        if velocities is not None and i < len(velocities):
            valid_vels.append(velocities[i])
    if not X_list:
        return {}
    X = torch.stack(X_list).to(device)
    y_true = torch.tensor(y_list, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X).squeeze(-1)
    mae = (pred - y_true).abs().mean().item()
    result = {"mae": mae, "mean_interval_width": float("nan"), "coverage": {}}
    if valid_vels:
        unique_vels = sorted(set(valid_vels))
        per_vel = {}
        for v in unique_vels:
            mask = torch.tensor([vel == v for vel in valid_vels], device=device)
            if mask.sum() == 0:
                continue
            y_v = y_true[mask]
            pred_v = pred[mask]
            per_vel[v] = {"mae": (pred_v - y_v).abs().mean().item(),
                          "n_clips": int(mask.sum().item())}
        result["per_velocity"] = per_vel
    return result


def evaluate_baseline(baseline, clips, velocities=None):
    preds = []
    y_list = []
    valid_vels = []
    for i, c in enumerate(clips):
        if np.isnan(c.margin):
            continue
        preds.append(baseline.predict(c))
        y_list.append(c.margin)
        if velocities is not None and i < len(velocities):
            valid_vels.append(velocities[i])
    if not preds:
        return {}
    preds = np.array(preds)
    y_arr = np.array(y_list)
    mae = float(np.mean(np.abs(preds - y_arr)))
    result = {"mae": mae, "mean_interval_width": float("nan"), "coverage": {}}
    if valid_vels:
        unique_vels = sorted(set(valid_vels))
        per_vel = {}
        for v in unique_vels:
            idx = [j for j, vv in enumerate(valid_vels) if vv == v]
            if not idx:
                continue
            per_vel[v] = {"mae": float(np.mean(np.abs(preds[idx] - y_arr[idx]))),
                          "n_clips": len(idx)}
        result["per_velocity"] = per_vel
    return result


if __name__ == "__main__":
    dataset_dir = "p4_dataset"
    P(f"Loading dataset from {dataset_dir}/...")
    clips_arr, margins, velocities, design_ids, realization_ids, meta = load_dataset(dataset_dir)
    P(f"  {len(margins)} clips, {meta['n_train']} train / {meta['n_val']} val / {meta['n_test']} test designs")

    train_ids = set(meta["train_designs"])
    val_ids = set(meta["val_designs"])
    test_ids = set(meta["test_designs"])

    train_clips, train_vels = build_clips(clips_arr, margins, velocities, design_ids, train_ids)
    val_clips, val_vels = build_clips(clips_arr, margins, velocities, design_ids, val_ids)
    test_clips, test_vels = build_clips(clips_arr, margins, velocities, design_ids, test_ids)

    P(f"  Clips: {len(train_clips)} train, {len(val_clips)} val, {len(test_clips)} test")
    P(f"  Velocity range: {velocities.min():.0f} - {velocities.max():.0f} m/s")

    all_clips = train_clips + val_clips + test_clips
    all_vels = train_vels + val_vels + test_vels

    results = {}
    N_SEEDS = 3
    N_CHANNELS = 8

    # ── TCN Quantile ──
    for seed in range(N_SEEDS):
        name = f"quantile_s{seed}"
        P(f"\nTraining {name}...")
        t0 = time.time()
        model, history, _, _ = train(
            None, n_channels=N_CHANNELS, hidden_dim=32, n_layers=4,
            epochs=20, lr=1e-3, seed=seed, batch_size=64,
            train_clips=train_clips, val_clips=val_clips, test_clips=test_clips,
            velocities=all_vels,
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_coverage(model, test_clips, velocities=test_vels)
        results[name] = metrics
        P(f"  MAE={metrics.get('mae', 0):.4f} width={metrics.get('mean_interval_width', 0):.4f}")
        torch.save(model.state_dict(), f"p4_{name}.pt")

    # ── TCN Huber ──
    for seed in range(N_SEEDS):
        name = f"huber_s{seed}"
        P(f"\nTraining {name}...")
        t0 = time.time()
        model, history, _, _ = train_huber(
            None, n_channels=N_CHANNELS, hidden_dim=32, n_layers=4,
            epochs=20, lr=1e-3, seed=seed, batch_size=64,
            train_clips=train_clips, val_clips=val_clips, test_clips=test_clips,
            velocities=all_vels,
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_point(model, test_clips, velocities=test_vels)
        results[name] = metrics
        P(f"  MAE={metrics.get('mae', 0):.4f}")
        torch.save(model.state_dict(), f"p4_{name}.pt")

    # ── TCN Median ──
    for seed in range(N_SEEDS):
        name = f"median_s{seed}"
        P(f"\nTraining {name}...")
        t0 = time.time()
        model, history, _, _ = train_median(
            None, n_channels=N_CHANNELS, hidden_dim=32, n_layers=4,
            epochs=20, lr=1e-3, seed=seed, batch_size=64,
            train_clips=train_clips, val_clips=val_clips, test_clips=test_clips,
            velocities=all_vels,
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_point(model, test_clips, velocities=test_vels)
        results[name] = metrics
        P(f"  MAE={metrics.get('mae', 0):.4f}")
        torch.save(model.state_dict(), f"p4_{name}.pt")

    # ── Baselines ──
    P(f"\nTraining baselines...")

    bl = ConstantMedianBaseline()
    bl.fit(train_clips)
    results["constant_median"] = evaluate_baseline(bl, test_clips, test_vels)
    P(f"  constant_median MAE={results['constant_median'].get('mae', 0):.4f}")

    bl = VelocityLinearBaseline()
    bl.fit(train_clips)
    results["velocity_linear"] = evaluate_baseline(bl, test_clips, test_vels)
    P(f"  velocity_linear MAE={results['velocity_linear'].get('mae', 0):.4f}")

    bl = GrowthRateBaseline()
    bl.fit(train_clips)
    results["growth_rate"] = evaluate_baseline(bl, test_clips, test_vels)
    P(f"  growth_rate MAE={results['growth_rate'].get('mae', 0):.4f}")

    bl = PhysicsFeatureRidge()
    bl.fit(train_clips)
    results["physics_ridge"] = evaluate_baseline(bl, test_clips, test_vels)
    P(f"  physics_ridge MAE={results['physics_ridge'].get('mae', 0):.4f}")

    # ── Summary ──
    P("\n" + "=" * 70)
    P(f"{'Model':<25} {'MAE':>8} {'Interval':>10} {'Coverage':>10}")
    P("-" * 55)
    for name, m in results.items():
        mae = m.get("mae", 0)
        iw = m.get("mean_interval_width", 0)
        cov = m.get("coverage", {}).get("interval_0.05_0.95", 0)
        P(f"{name:<25} {mae:>8.4f} {iw:>10.4f} {cov:>10.3f}")

    # Save results
    with open("p4_train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    P("\nResults saved to p4_train_results.json")
