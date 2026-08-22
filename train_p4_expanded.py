#!/usr/bin/env python3
"""Train all P4 models on the expanded dataset with design-level splits."""
from __future__ import annotations
import json
import time
import math
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path
from mechanics.p4_margin_estimation.train import (
    train, train_huber, train_median, evaluate_coverage, HuberMarginModel,
)
from mechanics.p4_margin_estimation.baselines import (
    ConstantMedianBaseline, PhysicsFeatureRidge, VelocityLinearBaseline,
    GrowthRateBaseline, GRUMarginModel, train_gru,
)


from mechanics.p4_margin_estimation.domain_randomization import (
    SensorPerturber, SensorPerturbationConfig,
)


class _GruPredictAdapter:
    """Adapt GRUMarginModel.forward to the MarginPredictor protocol."""

    def __init__(self, model):
        self.model = model

    def predict(self, clips) -> np.ndarray:
        X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in clips])
        self.model.eval()
        with torch.inference_mode():
            return self.model(X).squeeze(-1).cpu().numpy()


P = lambda *a, **kw: print(*a, **kw, flush=True)


def bootstrap_by_design(
    records: list[tuple[int, float]],
    metric_fn,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap by resampling design IDs.

    Args:
        records: list of (design_id, value) pairs.
        metric_fn: function that takes a list of values and returns a scalar.
        n_bootstrap: number of bootstrap resamples.
        seed: random seed.

    Returns:
        (observed, lower_95, upper_95).
    """
    rng = np.random.default_rng(seed)
    design_vals: dict[int, list[float]] = defaultdict(list)
    for did, val in records:
        design_vals[did].append(val)

    design_ids = list(design_vals.keys())
    observed = metric_fn([v for vals in design_vals.values() for v in vals])

    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_ids = rng.choice(design_ids, size=len(design_ids), replace=True)
        sample_vals = [v for did in sampled_ids for v in design_vals[did]]
        boot_stats[b] = metric_fn(sample_vals)

    return float(observed), float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def monotonicity_violation_rate(
    predictions: np.ndarray,
    velocities: np.ndarray,
    design_ids: np.ndarray,
) -> float:
    """Fraction of design pairs where velocity ordering is violated."""
    unique_designs = np.unique(design_ids)
    violations = 0
    total = 0
    for did in unique_designs:
        mask = design_ids == did
        v = velocities[mask]
        p = predictions[mask]
        order = np.argsort(v)
        p_sorted = p[order]
        diffs = np.diff(p_sorted)
        violations += int(np.sum(diffs > 0))
        total += len(diffs)
    return violations / max(total, 1)


class _Clip:
    __slots__ = ("sensor_signals", "margin", "velocity", "design_id")
    def __init__(self, signals, margin, velocity, design_id=None):
        self.sensor_signals = signals
        self.margin = margin
        self.velocity = velocity
        self.design_id = design_id


def apply_dr_with_mask(clips, seed=0):
    """Apply DR to clips and return (augmented_clips, n_channels).

    P1-26: Concatenates validity mask as extra channels so the model
    knows which sensor channels were dropped.
    """
    rng = np.random.default_rng(seed)
    dr_config = SensorPerturbationConfig(
        snr_db=30.0, per_channel_gain=True, per_channel_bias=True,
        gain_drift=True, colored_noise_prob=0.25,
        channel_dropout_distribution=((0, 0.60), (1, 0.25), (2, 0.15)),
        burst_dropout_prob=0.25, timing_skew=True, common_mode_fraction=0.3,
    )
    perturber = SensorPerturber(dr_config)
    augmented = []
    for c in clips:
        p = perturber.perturb(c.sensor_signals, rng)
        # Concatenate mask as extra channels: [signals; mask]
        combined = np.concatenate([p.signals, p.valid_mask], axis=0)
        aug_clip = _Clip(combined, c.margin, c.velocity, c.design_id)
        augmented.append(aug_clip)
    n_channels = clips[0].sensor_signals.shape[0] * 2  # signals + mask
    return augmented, n_channels


def load_dataset(dataset_dir: str = "p4_dataset"):
    d = Path(dataset_dir)
    # P2-1: Prefer .npy (mmap-compatible), fall back to .npz
    clips_npy = d / "clips.npy"
    clips_npz = d / "clips.npz"
    if clips_npy.exists():
        clips_arr = np.load(clips_npy)
    elif clips_npz.exists():
        clips_arr = np.load(clips_npz)["clips"]
    else:
        raise FileNotFoundError(f"No clips file found in {d}")
    margins = np.load(d / "margins.npy")
    velocities = np.load(d / "velocities.npy")
    design_ids = np.load(d / "design_ids.npy")
    realization_ids = np.load(d / "realization_ids.npy")
    with open(d / "metadata.json") as f:
        meta = json.load(f)

    # Validation: schema, shape, finite-value, zero-clip checks
    if clips_arr.ndim != 3 or clips_arr.shape[1:] != (8, 512):
        raise ValueError(f"Unexpected clip shape: {clips_arr.shape}")
    if margins.shape != (clips_arr.shape[0],):
        raise ValueError(f"Margins shape {margins.shape} != clips {clips_arr.shape[0]}")
    if not np.all(np.isfinite(clips_arr)):
        raise ValueError("Non-finite values in clips")
    n_zero_clips = int(np.all(clips_arr == 0, axis=(1, 2)).sum())
    if n_zero_clips != 0:
        raise ValueError(f"Found {n_zero_clips} all-zero clips")

    return clips_arr, margins, velocities, design_ids, realization_ids, meta


def build_clips(clips_arr, margins, velocities, design_ids_all, split_ids):
    """Build clip objects for a set of design IDs."""
    clips = []
    vel_list = []
    for i, did in enumerate(design_ids_all):
        if did in split_ids:
            c = _Clip(clips_arr[i], float(margins[i]), float(velocities[i]), design_id=did)
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
    with torch.inference_mode():
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
    valid_clips = [c for c in clips if not np.isnan(c.margin)]
    if not valid_clips:
        return {}
    preds = np.asarray(baseline.predict(valid_clips))
    y_arr = np.array([c.margin for c in valid_clips])
    valid_vels = [velocities[i] for i, c in enumerate(clips) if not np.isnan(c.margin)] if velocities is not None else []
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

    # P1-26: Apply DR to training clips and concatenate validity mask as extra channels
    P("Applying domain randomization with validity mask channels...")
    train_clips_dr, N_CHANNELS = apply_dr_with_mask(train_clips, seed=0)
    val_clips_dr, _ = apply_dr_with_mask(val_clips, seed=1)
    test_clips_dr, _ = apply_dr_with_mask(test_clips, seed=2)
    P(f"  DR applied: {N_CHANNELS} channels ({N_CHANNELS // 2} signal + {N_CHANNELS // 2} mask)")

    # ── TCN Quantile ──
    for seed in range(N_SEEDS):
        name = f"quantile_s{seed}"
        P(f"\nTraining {name}...")
        t0 = time.time()
        model, history, _, _ = train(
            None, n_channels=N_CHANNELS, hidden_dim=32, n_layers=4,
            epochs=20, lr=1e-3, seed=seed, batch_size=64,
            train_clips=train_clips_dr, val_clips=val_clips_dr, test_clips=test_clips_dr,
            velocities=all_vels,
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_coverage(model, test_clips_dr, velocities=test_vels)
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
            train_clips=train_clips_dr, val_clips=val_clips_dr, test_clips=test_clips_dr,
            velocities=all_vels,
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_point(model, test_clips_dr, velocities=test_vels)
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
            train_clips=train_clips_dr, val_clips=val_clips_dr, test_clips=test_clips_dr,
            velocities=all_vels,
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_point(model, test_clips_dr, velocities=test_vels)
        results[name] = metrics
        P(f"  MAE={metrics.get('mae', 0):.4f}")
        torch.save(model.state_dict(), f"p4_{name}.pt")

    # ── Baselines ──
    P(f"\nTraining baselines...")

    bl = ConstantMedianBaseline()
    bl.fit(train_clips_dr)
    results["constant_median"] = evaluate_baseline(bl, test_clips_dr, test_vels)
    P(f"  constant_median MAE={results['constant_median'].get('mae', 0):.4f}")

    bl = VelocityLinearBaseline()
    bl.fit(train_clips_dr)
    results["velocity_linear"] = evaluate_baseline(bl, test_clips_dr, test_vels)
    P(f"  velocity_linear MAE={results['velocity_linear'].get('mae', 0):.4f}")

    bl = GrowthRateBaseline()
    bl.fit(train_clips_dr)
    results["growth_rate"] = evaluate_baseline(bl, test_clips_dr, test_vels)
    P(f"  growth_rate MAE={results['growth_rate'].get('mae', 0):.4f}")

    bl = PhysicsFeatureRidge()
    bl.fit(train_clips_dr)
    results["physics_ridge"] = evaluate_baseline(bl, test_clips_dr, test_vels)
    P(f"  physics_ridge MAE={results['physics_ridge'].get('mae', 0):.4f}")

    # ── GRU (deep sequence baseline) ──
    for seed in range(N_SEEDS):
        name = f"gru_s{seed}"
        P(f"\nTraining {name}...")
        t0 = time.time()
        gru_model, history = train_gru(
            train_clips_dr, n_channels=N_CHANNELS, epochs=20, seed=seed,
        )
        P(f"  final_train_loss={history['train_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        results[name] = evaluate_baseline(_GruPredictAdapter(gru_model), test_clips_dr, test_vels)
        P(f"  {name} MAE={results[name].get('mae', 0):.4f}")
        torch.save(gru_model.state_dict(), f"p4_{name}.pt")

    # ── Summary ──
    P("\n" + "=" * 70)
    P(f"{'Model':<25} {'MAE':>8} {'Interval':>10} {'Coverage':>10}")
    P("-" * 55)
    for name, m in results.items():
        mae = m.get("mae", 0)
        iw = m.get("mean_interval_width", 0)
        cov = m.get("coverage", {}).get("interval_0.05_0.95", 0)
        P(f"{name:<25} {mae:>8.4f} {iw:>10.4f} {cov:>10.3f}")

    # ── Per-design metrics (quantile model) ──
    q_model = None
    q_model_name = None
    for seed in range(N_SEEDS):
        name = f"quantile_s{seed}"
        if name in results:
            q_model_name = name
            break
    if q_model_name is not None:
        from mechanics.p4_margin_estimation.quantile_head import (
            QuantileMarginModel, fit_cqr_adjustment, apply_cqr_adjustment,
        )
        q_model = QuantileMarginModel(n_channels=N_CHANNELS, hidden_dim=32, n_layers=4)
        q_model.load_state_dict(torch.load(f"p4_{q_model_name}.pt", weights_only=True))
        q_model.eval()

    if q_model is not None:
        test_design_ids = [c.design_id for c in test_clips_dr if c.design_id is not None]
        unique_test_designs = sorted(set(test_design_ids))

        P(f"\n  Per-design metrics ({len(unique_test_designs)} test designs):")
        design_maes = []
        per_design_data: dict[int, list] = defaultdict(list)
        for c in test_clips_dr:
            if c.design_id is not None and not np.isnan(c.margin):
                per_design_data[c.design_id].append(c)

        for did in unique_test_designs:
            d_clips = per_design_data[did]
            if not d_clips:
                continue
            X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in d_clips])
            y = torch.tensor([c.margin for c in d_clips], dtype=torch.float32)
            with torch.inference_mode():
                pred = q_model(X)[:, 1]  # median
            mae = (pred - y).abs().mean().item()
            design_maes.append(mae)
        if design_maes:
            P(f"    Mean MAE: {np.mean(design_maes):.4f} ± {np.std(design_maes):.4f}")
            P(f"    Median design MAE: {np.median(design_maes):.4f}")
            P(f"    Worst design MAE: {np.max(design_maes):.4f}")

    # ── Safety-critical metrics ──
    if q_model is not None:
        near_flutter_clips = [c for c in test_clips_dr if not np.isnan(c.margin) and c.margin < 0.15]
        if near_flutter_clips:
            X_nf = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in near_flutter_clips])
            y_nf = torch.tensor([c.margin for c in near_flutter_clips], dtype=torch.float32)
            with torch.inference_mode():
                quantile_pred_nf = q_model(X_nf)
            pred_nf = quantile_pred_nf[:, 1]  # median
            near_flutter_mae = (pred_nf - y_nf).abs().mean().item()
            P(f"\n  Safety-critical metrics:")
            P(f"    Near-flutter MAE (margin<0.15): {near_flutter_mae:.4f} ({len(near_flutter_clips)} clips)")

            # False-safe rate using median and lower bound
            all_X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in test_clips_dr if not np.isnan(c.margin)])
            all_y = torch.tensor([c.margin for c in test_clips_dr if not np.isnan(c.margin)], dtype=torch.float32)
            with torch.inference_mode():
                quantile_pred_all = q_model(all_X)
            all_pred_median = quantile_pred_all[:, 1]

            unsafe_mask = all_y <= 0.0
            median_false_safe = unsafe_mask & (all_pred_median > 0.0)
            n_unsafe = int(unsafe_mask.sum().item())
            if n_unsafe == 0:
                false_safe_rate = float("nan")
            else:
                false_safe_rate = median_false_safe.sum().item() / n_unsafe

            false_safe_incidence = median_false_safe.float().mean().item()

            # Lower bound false-safe
            if quantile_pred_all.shape[1] > 0:
                all_pred_lower = quantile_pred_all[:, 0]
                lower_bound_false_safe = unsafe_mask & (all_pred_lower > 0.0)
                if n_unsafe == 0:
                    lower_bound_false_safe_rate = float("nan")
                else:
                    lower_bound_false_safe_rate = lower_bound_false_safe.sum().item() / n_unsafe
                P(f"    Lower-bound false-safe rate: {lower_bound_false_safe_rate:.4f}")

            # Reciprocal metric
            safe_mask = all_y > 0.0
            false_unsafe_mask = safe_mask & (all_pred_median <= 0.0)
            false_unsafe_rate = false_unsafe_mask.sum().item() / max(int(safe_mask.sum().item()), 1)
            P(f"    False-safe rate (median): {false_safe_rate:.4f}")
            P(f"    False-safe incidence: {false_safe_incidence:.4f}")
            P(f"    False-unsafe rate: {false_unsafe_rate:.4f}")

            # Signed bias near flutter
            signed_bias = (pred_nf - y_nf).mean().item()
            P(f"    Signed bias (near-flutter): {signed_bias:+.4f}")

            # P1-21: CQR calibration on val set, evaluate on test
            all_pred_lower = quantile_pred_all[:, 0].cpu().numpy()
            all_pred_upper = quantile_pred_all[:, 2].cpu().numpy()
            all_y_np = all_y.cpu().numpy()

            # Use val clips for CQR calibration
            val_clips_valid = [c for c in val_clips_dr if not np.isnan(c.margin)]
            if val_clips_valid:
                val_X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in val_clips_valid])
                with torch.inference_mode():
                    val_pred = q_model(val_X)
                val_lower = val_pred[:, 0].cpu().numpy()
                val_upper = val_pred[:, 2].cpu().numpy()
                val_y = np.array([c.margin for c in val_clips_valid])
                cqr_adj = fit_cqr_adjustment(val_lower, val_upper, val_y, alpha=0.10)
                adj_lower, adj_upper = apply_cqr_adjustment(all_pred_lower, all_pred_upper, cqr_adj)
                cqr_coverage = float(np.mean((all_y_np >= adj_lower) & (all_y_np <= adj_upper)))
                cqr_width = float(np.mean(adj_upper - adj_lower))
                P(f"    CQR adjusted coverage: {cqr_coverage:.3f}, width: {cqr_width:.4f}")

            # P1-29: Bootstrap confidence intervals
            test_records = [
                (c.design_id, float((pred - c.margin).abs().item()))
                for c, pred in zip(
                    [c for c in test_clips_dr if not np.isnan(c.margin)],
                    all_pred_median.cpu().numpy().tolist(),
                )
                if c.design_id is not None
            ]
            if test_records:
                obs_mae, lo_mae, hi_mae = bootstrap_by_design(
                    test_records, lambda vals: float(np.mean(vals)),
                    n_bootstrap=500, seed=42,
                )
                P(f"    MAE (design-bootstrap 95% CI): {obs_mae:.4f} [{lo_mae:.4f}, {hi_mae:.4f}]")

            # P1-30: Monotonicity violations
            test_vels_arr = np.array([
                c.velocity for c in test_clips_dr if not np.isnan(c.margin)
            ])
            test_design_ids_arr = np.array([
                c.design_id for c in test_clips_dr if not np.isnan(c.margin) and c.design_id is not None
            ])
            if len(test_design_ids_arr) > 0 and len(all_y_np) == len(test_vels_arr):
                mono_rate = monotonicity_violation_rate(
                    all_pred_median.cpu().numpy(), test_vels_arr, test_design_ids_arr,
                )
                P(f"    Monotonicity violation rate: {mono_rate:.4f}")

    # Save results
    with open("p4_train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    P("\nResults saved to p4_train_results.json")
