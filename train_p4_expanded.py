#!/usr/bin/env python3
"""Train all P4 models on the expanded dataset with design-level splits."""
from __future__ import annotations
import ctypes
import gc
import json
import os
import time
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path
from mechanics.p4_margin_estimation.train import (
    train, train_huber, train_unweighted_quantile as train_median,
    train_unweighted_quantile, evaluate_coverage, HuberMarginModel,
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
_LIBC = ctypes.CDLL("libc.so.6")
_RESULTS_PATH = "p4_train_results.json"
def _save_results(results):
    """Atomically persist the results dict (colab VMs are preemptible)."""
    tmp = _RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    Path(tmp).replace(_RESULTS_PATH)
def _wandb_init(meta, n_channels):
    """Start a W&B run; offline-first so unstable VMs never block training."""
    try:
        import wandb
    except ImportError:
        P("wandb not installed — metrics go to result files only.")
        return None
    try:
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "p4-margin"),
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={"n_train": meta.get("n_train"), "n_val": meta.get("n_val"),
                    "n_test": meta.get("n_test"),
                    "n_train_designs": len(meta.get("train_designs", [])),
                    "n_val_designs": len(meta.get("val_designs", [])),
                    "n_test_designs": len(meta.get("test_designs", [])),
                    "n_channels": n_channels,
                    "provenance_available": meta.get("provenance_available", False)},
        )
    except Exception as exc:
        P(f"wandb init failed ({exc}) — continuing without run tracking.")
        return None
def _flat_metrics(name, metrics):
    """Flatten one model's metrics dict to scalar wandb entries."""
    flat = {}
    for key in ("mae", "mae_dr", "mean_interval_width", "mean_interval_width_dr"):
        val = metrics.get(key)
        if isinstance(val, (int, float)) and np.isfinite(val):
            flat[f"{name}/{key}"] = float(val)
    for key, val in (metrics.get("coverage") or {}).items():
        if isinstance(val, (int, float)) and np.isfinite(val):
            flat[f"{name}/coverage_{key}"] = float(val)
    for key, val in (metrics.get("coverage_dr") or {}).items():
        if isinstance(val, (int, float)) and np.isfinite(val):
            flat[f"{name}/coverage_dr_{key}"] = float(val)
    return flat
def _wlog(run, payload):
    if run is None:
        return
    try:
        run.log(payload)
    except Exception as exc:
        P(f"wandb log failed ({exc}) — continuing.")
def _epoch_logger(run, name):
    """Per-epoch train/val curves for training-health visibility.
    Cheap on free tiers (2 scalars × 20 epochs per model). Invoked
    synchronously, so closing over the loop's ``name`` is safe.
    """
    def _cb(epoch, model, history):
        _wlog(run, {f"{name}/epoch": epoch,
                    f"{name}/train_loss": history["train_loss"][-1],
                    f"{name}/val_loss": history["val_loss"][-1]})
    return _cb


def _release_memory():
    """Return freed native memory to the OS between model runs.

    Training a TCN allocates large transient activation buffers (batch x
    hidden x seq_len per layer, forward+backward). PyTorch frees them
    promptly, but glibc keeps them in malloc arenas/fastbins, so RSS only
    ratchets upward across sequential train() calls (~+200-400MB per model,
    measured). gc.collect() drops the Python references;
    malloc_trim(0) hands the cached native pages back to the kernel,
    keeping the post-run RSS floor flat instead of climbing.

    Recommended at launch: export MALLOC_ARENA_MAX=2 (glibc reads it at
    process start; it cannot be enabled from inside Python) to limit arena
    fragmentation under torch's multi-threaded allocators.
    """
    gc.collect()
    try:
        _LIBC.malloc_trim(0)
    except Exception:
        pass  # non-glibc platform: gc.collect() alone still bounds growth


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
    """Fraction of strict-velocity design pairs where ordering is violated (same-velocity ties excluded)."""
    unique_designs = np.unique(design_ids)
    violations = 0
    total = 0
    for did in unique_designs:
        mask = design_ids == did
        v = velocities[mask]
        p = predictions[mask]
        order = np.argsort(v, kind="stable")
        v_sorted = v[order]
        p_sorted = p[order]
        # Only strict velocity increases count: same-velocity realizations
        # (ties) have arbitrary argsort order and must not vote.
        dv = np.diff(v_sorted)
        dp = np.diff(p_sorted)
        strict = dv > 0
        violations += int(np.sum(dp[strict] > 0))
        total += int(np.sum(strict))
    return violations / max(total, 1)


class _Clip:
    __slots__ = ("sensor_signals", "margin", "velocity", "design_id", "dt", "time")
    def __init__(self, signals, margin, velocity, design_id=None, dt=None, time=None):
        self.sensor_signals = signals
        self.margin = margin
        self.velocity = velocity
        self.design_id = design_id
        self.dt = dt
        self.time = time


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
        aug_clip = _Clip(combined, c.margin, c.velocity, c.design_id,
                         dt=getattr(c, "dt", None), time=getattr(c, "time", None))
        augmented.append(aug_clip)
    n_channels = clips[0].sensor_signals.shape[0] * 2  # signals + mask
    return augmented, n_channels


def with_ones_mask(clips):
    """Lift clean clips to mask-model channels with an all-valid mask."""
    import numpy as _np
    lifted = []
    for c in clips:
        sig = _np.asarray(c.sensor_signals)
        mask = _np.ones_like(sig)
        combined = _np.concatenate([sig, mask], axis=0)
        lifted.append(_Clip(combined, c.margin, c.velocity, c.design_id,
                           dt=getattr(c, "dt", None), time=getattr(c, "time", None)))
    return lifted


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
    # Label/metadata arrays: current generator writes metadata_arrays.npz;
    # legacy datasets kept them as loose .npy files.
    arrays_npz = d / "metadata_arrays.npz"
    prov: dict = {}
    if arrays_npz.exists():
        with np.load(arrays_npz, allow_pickle=True) as z:
            margins = z["margins"]
            velocities = z["velocities"]
            design_ids = z["design_ids"]
            realization_ids = z["realization_ids"]
            # Review §4.1/4.2/4.4 + finding 3 provenance: present in regenerated
            # datasets, absent in legacy ones — default to NaN/False downstream.
            for key in ("dts", "durations", "u_crits", "clamp_fracs",
                        "max_log_amps", "alphas", "omega_crits",
                        "label_alphas", "label_omegas", "label_represented",
                        "sat_fracs"):
                if key in z:
                    prov[key] = np.asarray(z[key])
    else:
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
    # Review provenance stays in-memory only (never written back to JSON).
    meta["provenance_available"] = bool(prov)
    meta["provenance"] = {k: v for k, v in prov.items()}
    return clips_arr, margins, velocities, design_ids, realization_ids, meta


def build_clips(clips_arr, margins, velocities, design_ids_all, split_ids, dts=None):
    """Build clip objects for a set of design IDs."""
    clips = []
    vel_list = []
    for i, did in enumerate(design_ids_all):
        if did in split_ids:
            dt = float(np.asarray(dts)[i]) if dts is not None else None
            if dt is not None and not np.isfinite(dt):
                dt = None
            c = _Clip(clips_arr[i], float(margins[i]), float(velocities[i]), design_id=did, dt=dt)
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
    # Review §6.4: bounded-batch inference instead of one full-partition tensor.
    y_true = torch.tensor(y_list, dtype=torch.float32, device=device)
    outs = []
    with torch.inference_mode():
        for s in range(0, len(X_list), 256):
            outs.append(model(torch.stack(X_list[s:s + 256]).to(device)).detach().cpu())
    pred = torch.cat(outs, dim=0).to(device)
    # Median-only variant is the full 3-column QuantileMarginModel trained
    # on the middle channel; reduce to that column before point metrics.
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred = pred[..., pred.shape[-1] // 2]
    pred = pred.squeeze(-1)
    mae = (pred - y_true).abs().mean().item()
    result = {"mae": mae, "mean_interval_width": float("nan"), "coverage": {}}
    if valid_vels:
        from mechanics.p4_margin_estimation.train import _velocity_bin_index
        per_vel = {}
        for bin_label, idx in sorted(_velocity_bin_index(valid_vels).items()):
            mask = torch.zeros(len(valid_vels), dtype=torch.bool, device=device)
            mask[idx] = True
            y_v = y_true[mask]
            pred_v = pred[mask]
            per_vel[bin_label] = {"mae": (pred_v - y_v).abs().mean().item(),
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
        from mechanics.p4_margin_estimation.train import _velocity_bin_index
        per_vel = {}
        for bin_label, idx in sorted(_velocity_bin_index(valid_vels).items()):
            if not idx:
                continue
            per_vel[bin_label] = {"mae": float(np.mean(np.abs(preds[idx] - y_arr[idx]))),
                                  "n_clips": len(idx)}
        result["per_velocity"] = per_vel
    return result


def _split_select_calibrate(val_sorted):
    """Split validation designs into disjoint select/calibrate halves.
    Review finding 2: checkpoint selection and CQR calibration must use
    disjoint design sets, or the split-conformal guarantee for unseen designs
    is void. Requires at least two contributing validation designs and FAILS
    rather than silently reusing selection data for calibration.
    """
    ids = sorted(set(val_sorted))
    if len(ids) < 2:
        raise ValueError(
            f"Need at least two validation designs to separate checkpoint "
            f"selection from CQR calibration, got {len(ids)}. Regenerate with "
            f"more designs instead of reusing selection data."
        )
    n_sel = max(1, len(ids) // 2)
    select_ids, calib_ids = set(ids[:n_sel]), set(ids[n_sel:])
    if not select_ids or not calib_ids or select_ids & calib_ids:
        raise ValueError(f"Invalid select/calibrate partition of {ids}")
    return select_ids, calib_ids


if __name__ == "__main__":
    dataset_dir = os.environ.get("P4_DATASET_DIR", "p4_dataset")
    P(f"Loading dataset from {dataset_dir}/...")
    clips_arr, margins, velocities, design_ids, realization_ids, meta = load_dataset(dataset_dir)
    P(f"  {len(margins)} clips ({meta['n_train']}/{meta['n_val']}/{meta['n_test']} by design split)")

    train_ids = set(meta["train_designs"])
    val_ids = set(meta["val_designs"])
    test_ids = set(meta["test_designs"])
    if not train_ids or not val_ids or not test_ids:
        raise ValueError(
            f"Design split has an empty partition (train={len(train_ids)}, "
            f"val={len(val_ids)}, test={len(test_ids)}); select/calibrate/CQR "
            f"need all three. Regenerate with more designs."
        )
    prov = meta.get("provenance", {})
    dts = prov.get("dts") if isinstance(prov, dict) else None
    train_clips, train_vels = build_clips(clips_arr, margins, velocities, design_ids, train_ids, dts=dts)
    val_clips, val_vels = build_clips(clips_arr, margins, velocities, design_ids, val_ids, dts=dts)
    test_clips, test_vels = build_clips(clips_arr, margins, velocities, design_ids, test_ids, dts=dts)
    P(f"  Clips: {len(train_clips)} train, {len(val_clips)} val, {len(test_clips)} test")
    P(f"  Velocity range: {velocities.min():.0f} - {velocities.max():.0f} m/s")
    if meta.get("provenance_available"):
        P(f"  Provenance arrays present (dt/duration/u_crit/clamp).")
    else:
        P(f"  WARNING: legacy dataset without per-clip provenance (dt/u_crit/clamp missing).")
    # Review §5.1 + finding 2: split validation DESIGNS into disjoint select
    # (checkpoint) vs calibrate (CQR) halves. Fails on fewer than two
    # validation designs rather than reusing selection data for calibration.
    select_ids, calib_ids = _split_select_calibrate(sorted(val_ids))
    sel_clips = [c for c in val_clips if c.design_id in select_ids]
    cal_clips = [c for c in val_clips if c.design_id in calib_ids]
    if not sel_clips or not cal_clips:
        raise ValueError(
            f"Empty select ({len(sel_clips)}) or calibrate ({len(cal_clips)}) "
            f"clip set; both partitions must contribute clips."
        )
    P(f"  Val split: {len(select_ids)} select / {len(calib_ids)} calibrate designs "
      f"({len(sel_clips)}/{len(cal_clips)} clips)")
    all_clips = train_clips + val_clips + test_clips
    all_vels = train_vels + val_vels + test_vels
    # Crash/OOM resume: pick up finished models from a previous partial run.
    # A model is skipped only when its checkpoint AND metrics exist AND the
    # dataset fingerprint matches — same data, splits, and protocol, so the
    # resumed run is identical to an uninterrupted one.
    results = {}
    _resume_key = {"n_clips": len(margins), "n_train": len(train_clips),
                   "n_val": len(val_clips), "n_test": len(test_clips),
                   "provenance": bool(meta.get("provenance_available"))}
    if os.path.exists(_RESULTS_PATH):
        try:
            _prev = json.load(open(_RESULTS_PATH))
            if _prev.get("_resume_key") == _resume_key:
                results = {k: v for k, v in _prev.items() if not k.startswith("_")}
                P(f"  Resuming: {len(results)} finished model(s) kept "
                  f"({sorted(results)[:6]}{'...' if len(results) > 6 else ''})")
            else:
                P("  Previous results are from a different dataset — starting fresh.")
        except Exception as exc:
            P(f"  Could not read previous results ({exc}) — starting fresh.")
    results["_resume_key"] = _resume_key
    def _skip_if_done(_name):
        """True when a checkpointed model with metrics can be reused as-is."""
        if _name in results and os.path.exists(f"p4_{_name}.pt"):
            P(f"  {_name}: checkpoint + metrics present — skipping (identical protocol).")
            return True
        return False
    # P1-26: Apply DR to TRAINING clips only; val/test stay clean (ones mask)
    # so primary MAE/coverage/CQR measure clean accuracy. DR val/test are
    # kept as a separate robustness slice.
    P("Applying domain randomization with validity mask channels...")
    train_clips_dr, N_CHANNELS = apply_dr_with_mask(train_clips, seed=0)
    sel_clips_clean = with_ones_mask(sel_clips)
    cal_clips_clean = with_ones_mask(cal_clips)
    val_clips_clean = with_ones_mask(val_clips)
    test_clips_clean = with_ones_mask(test_clips)
    val_clips_dr, _ = apply_dr_with_mask(val_clips, seed=1)
    test_clips_dr, _ = apply_dr_with_mask(test_clips, seed=2)
    P(f"  DR applied: {N_CHANNELS} channels ({N_CHANNELS // 2} signal + {N_CHANNELS // 2} mask)")
    P(f"  Primary metrics on clean val/test (ones mask); DR slice kept for robustness.")
    wrun = _wandb_init(meta, N_CHANNELS)
    # ── TCN Quantile ──
    for seed in range(N_SEEDS):
        name = f"quantile_s{seed}"
        if _skip_if_done(name):
            _wlog(wrun, _flat_metrics(name, results[name]))
            continue
        P(f"\nTraining {name}...")
        t0 = time.time()
        model, history, _, _ = train(
            None, n_channels=N_CHANNELS, hidden_dim=32, n_layers=9,
            epochs=20, lr=1e-3, seed=seed, batch_size=64,
            train_clips=train_clips_dr, val_clips=sel_clips_clean, test_clips=test_clips_clean,
            velocities=all_vels, on_epoch=_epoch_logger(wrun, name),
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_coverage(model, test_clips_clean, velocities=test_vels)
        dr_metrics = evaluate_coverage(model, test_clips_dr, velocities=test_vels)
        metrics["mae_dr"] = dr_metrics.get("mae")
        metrics["coverage_dr"] = dr_metrics.get("coverage")
        metrics["mean_interval_width_dr"] = dr_metrics.get("mean_interval_width")
        results[name] = metrics
        _save_results(results)
        _wlog(wrun, _flat_metrics(name, metrics))
        P(f"  MAE={metrics.get('mae', 0):.4f} width={metrics.get('mean_interval_width', 0):.4f}")
        torch.save(model.state_dict(), f"p4_{name}.pt")
        del model, history
        _release_memory()
    # ── TCN Huber ──
    for seed in range(N_SEEDS):
        name = f"huber_s{seed}"
        if _skip_if_done(name):
            _wlog(wrun, _flat_metrics(name, results[name]))
            continue
        P(f"\nTraining {name}...")
        t0 = time.time()
        model, history, _, _ = train_huber(
            None, n_channels=N_CHANNELS, hidden_dim=32, n_layers=9,
            epochs=20, lr=1e-3, seed=seed, batch_size=64,
            train_clips=train_clips_dr, val_clips=sel_clips_clean, test_clips=test_clips_clean,
            velocities=all_vels, on_epoch=_epoch_logger(wrun, name),
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_point(model, test_clips_clean, velocities=test_vels)
        metrics["mae_dr"] = evaluate_point(model, test_clips_dr, velocities=test_vels).get("mae")
        results[name] = metrics
        _save_results(results)
        _wlog(wrun, _flat_metrics(name, metrics))
        P(f"  MAE={metrics.get('mae', 0):.4f}")
        torch.save(model.state_dict(), f"p4_{name}.pt")
        del model, history
        _release_memory()
    # ── TCN Median ──
    for seed in range(N_SEEDS):
        name = f"median_s{seed}"
        if _skip_if_done(name):
            _wlog(wrun, _flat_metrics(name, results[name]))
            continue
        P(f"\nTraining {name}...")
        t0 = time.time()
        model, history, _, _ = train_unweighted_quantile(
            None, n_channels=N_CHANNELS, hidden_dim=32, n_layers=9,
            epochs=20, lr=1e-3, seed=seed, batch_size=64,
            train_clips=train_clips_dr, val_clips=sel_clips_clean, test_clips=test_clips_clean,
            velocities=all_vels, on_epoch=_epoch_logger(wrun, name),
        )
        P(f"  train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        metrics = evaluate_point(model, test_clips_clean, velocities=test_vels)
        metrics["mae_dr"] = evaluate_point(model, test_clips_dr, velocities=test_vels).get("mae")
        results[name] = metrics
        _save_results(results)
        _wlog(wrun, _flat_metrics(name, metrics))
        P(f"  MAE={metrics.get('mae', 0):.4f}")
        torch.save(model.state_dict(), f"p4_{name}.pt")
        del model, history
        _release_memory()

    # ── Baselines ──
    P(f"\nTraining baselines...")

    for _bl_name, _bl_cls in (("constant_median", ConstantMedianBaseline),
                             ("velocity_linear", VelocityLinearBaseline),
                             ("growth_rate", GrowthRateBaseline),
                             ("physics_ridge", PhysicsFeatureRidge)):
        bl = _bl_cls()
        bl.fit(train_clips_dr)
        results[_bl_name] = evaluate_baseline(bl, test_clips_clean, test_vels)
        results[_bl_name]["mae_dr"] = evaluate_baseline(bl, test_clips_dr, test_vels).get("mae")
        _save_results(results)
        _wlog(wrun, _flat_metrics(_bl_name, results[_bl_name]))
        P(f"  {_bl_name} MAE={results[_bl_name].get('mae', 0):.4f}")

    # ── GRU (deep sequence baseline) ──
    for seed in range(N_SEEDS):
        name = f"gru_s{seed}"
        if _skip_if_done(name):
            _wlog(wrun, _flat_metrics(name, results[name]))
            continue
        P(f"\nTraining {name}...")
        t0 = time.time()
        gru_model, history = train_gru(
            train_clips_dr, n_channels=N_CHANNELS, epochs=20, seed=seed,
            batch_size=64, val_clips=sel_clips_clean, velocities=train_vels,
            on_epoch=_epoch_logger(wrun, name),
        )
        P(f"  final_train_loss={history['train_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")
        results[name] = evaluate_baseline(_GruPredictAdapter(gru_model), test_clips_clean, test_vels)
        results[name]["mae_dr"] = evaluate_baseline(_GruPredictAdapter(gru_model), test_clips_dr, test_vels).get("mae")
        _save_results(results)
        _wlog(wrun, _flat_metrics(name, results[name]))
        P(f"  {name} MAE={results[name].get('mae', 0):.4f}")
        torch.save(gru_model.state_dict(), f"p4_{name}.pt")
        del gru_model, history
        _release_memory()
    # ── Summary ──
    P("\n" + "=" * 70)
    P(f"{'Model':<25} {'MAE':>8} {'Interval':>10} {'Coverage':>10}")
    P("-" * 55)
    for name, m in results.items():
        if name.startswith("_"):
            continue
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
        q_model = QuantileMarginModel(n_channels=N_CHANNELS, hidden_dim=32, n_layers=9)
        q_model.load_state_dict(torch.load(f"p4_{q_model_name}.pt", weights_only=True))
        q_model.eval()

    if q_model is not None:
        test_design_ids = [c.design_id for c in test_clips_clean if c.design_id is not None]
        unique_test_designs = sorted(set(test_design_ids))

        P(f"\n  Per-design metrics ({len(unique_test_designs)} test designs):")
        design_maes = []
        per_design_data: dict[int, list] = defaultdict(list)
        for c in test_clips_clean:
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
        near_flutter_clips = [c for c in test_clips_clean if not np.isnan(c.margin) and c.margin < 0.15]
        false_safe_rate = lower_bound_false_safe_rate = float("nan")
        false_safe_incidence = false_unsafe_rate = float("nan")
        near_flutter_mae = signed_bias = cqr_coverage = cqr_width = float("nan")
        mono_rate = None
        bootstrap_ci = None
        if near_flutter_clips:
            def _batched_quantiles(clips_list):
                outs = []
                with torch.inference_mode():
                    for s in range(0, len(clips_list), 256):
                        Xb = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in clips_list[s:s + 256]])
                        outs.append(q_model(Xb).cpu())
                return torch.cat(outs, dim=0) if outs else torch.empty(0, 3)
            quantile_pred_nf = _batched_quantiles(near_flutter_clips)
            y_nf = torch.tensor([c.margin for c in near_flutter_clips], dtype=torch.float32)
            pred_nf = quantile_pred_nf[:, 1]  # median column
            near_flutter_mae = (pred_nf - y_nf).abs().mean().item()
            P(f"\n  Safety-critical metrics:")
            P(f"    Near-flutter MAE (margin<0.15): {near_flutter_mae:.4f} ({len(near_flutter_clips)} clips)")
            valid_test = [c for c in test_clips_clean if not np.isnan(c.margin)]
            quantile_pred_all = _batched_quantiles(valid_test)
            all_y = torch.tensor([c.margin for c in valid_test], dtype=torch.float32)
            all_pred_median = quantile_pred_all[:, 1]

            unsafe_mask = all_y <= 0.0
            median_false_safe = unsafe_mask & (all_pred_median > 0.0)
            n_unsafe = int(unsafe_mask.sum().item())
            false_safe_rate = (median_false_safe.sum().item() / n_unsafe) if n_unsafe else float("nan")
            false_safe_incidence = median_false_safe.float().mean().item()

            all_pred_lower_t = quantile_pred_all[:, 0]
            lower_bound_false_safe = unsafe_mask & (all_pred_lower_t > 0.0)
            lower_bound_false_safe_rate = (lower_bound_false_safe.sum().item() / n_unsafe) if n_unsafe else float("nan")
            P(f"    Lower-bound false-safe rate: {lower_bound_false_safe_rate:.4f}")

            safe_mask = all_y > 0.0
            false_unsafe_mask = safe_mask & (all_pred_median <= 0.0)
            false_unsafe_rate = false_unsafe_mask.sum().item() / max(int(safe_mask.sum().item()), 1)
            P(f"    False-safe rate (median): {false_safe_rate:.4f}")
            P(f"    False-safe incidence: {false_safe_incidence:.4f}")
            P(f"    False-unsafe rate: {false_unsafe_rate:.4f}")

            signed_bias = (pred_nf - y_nf).mean().item()
            P(f"    Signed bias (near-flutter): {signed_bias:+.4f}")

            all_pred_lower = all_pred_lower_t.cpu().numpy()
            all_pred_upper = quantile_pred_all[:, 2].cpu().numpy()
            all_y_np = all_y.cpu().numpy()

            # Review §5.1: calibrate on the held-out calibrate designs, never the
            # checkpoint-select designs. Batched inference bounds memory (§6.4).
            cal_valid = [c for c in cal_clips_clean if not np.isnan(c.margin)]
            if cal_valid:
                cal_preds = []
                with torch.inference_mode():
                    for s in range(0, len(cal_valid), 256):
                        Xb = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in cal_valid[s:s + 256]])
                        cal_preds.append(q_model(Xb)[:, [0, 2]].cpu())
                val_pred = torch.cat(cal_preds, dim=0)
                cqr_adj = fit_cqr_adjustment(
                    val_pred[:, 0].numpy(), val_pred[:, 1].numpy(),
                    np.array([c.margin for c in cal_valid]), alpha=0.10,
                )
                adj_lower, adj_upper = apply_cqr_adjustment(all_pred_lower, all_pred_upper, cqr_adj)
                cqr_coverage = float(np.mean((all_y_np >= adj_lower) & (all_y_np <= adj_upper)))
                cqr_width = float(np.mean(adj_upper - adj_lower))
                P(f"    CQR adjusted coverage: {cqr_coverage:.3f}, width: {cqr_width:.4f} (cal on {len(calib_ids)} held-out designs)")

            test_records = [
                (c.design_id, float(abs(pred - c.margin)))
                for c, pred in zip(
                    [c for c in test_clips_clean if not np.isnan(c.margin)],
                    all_pred_median.cpu().numpy().tolist(),
                )
                if c.design_id is not None
            ]
            if test_records:
                obs_mae, lo_mae, hi_mae = bootstrap_by_design(
                    test_records, lambda vals: float(np.mean(vals)),
                    n_bootstrap=500, seed=42,
                )
                bootstrap_ci = [lo_mae, hi_mae]
                P(f"    MAE (design-bootstrap 95% CI): {obs_mae:.4f} [{lo_mae:.4f}, {hi_mae:.4f}]")

            test_vels_arr = np.array([
                c.velocity for c in test_clips_clean if not np.isnan(c.margin)
            ])
            test_design_ids_arr = np.array([
                c.design_id for c in test_clips_clean if not np.isnan(c.margin) and c.design_id is not None
            ])
            if len(test_design_ids_arr) > 0 and len(all_y_np) == len(test_vels_arr):
                mono_rate = monotonicity_violation_rate(
                    all_pred_median.cpu().numpy(), test_vels_arr, test_design_ids_arr,
                )
                P(f"    Monotonicity violation rate: {mono_rate:.4f}")

            # Persist the safety block (previously print-only; the expanded-
            # report generator consumes results["safety_summary"]).
            results["safety_summary"] = {
                "near_flutter_mae": near_flutter_mae,
                "near_flutter_clips": len(near_flutter_clips),
                "false_safe_rate_median": false_safe_rate,
                "lower_bound_false_safe_rate": lower_bound_false_safe_rate,
                "false_safe_incidence": false_safe_incidence,
                "false_unsafe_rate": false_unsafe_rate,
                "signed_bias_near_flutter": signed_bias,
                "cqr_coverage_90": cqr_coverage,
                "cqr_width_90": cqr_width,
                "mae_design_bootstrap_ci95": bootstrap_ci,
                "monotonicity_violation_rate": mono_rate,
            }
            _save_results(results)
            _wlog(wrun, {f"safety/{k}": v for k, v in results["safety_summary"].items()
                         if isinstance(v, (int, float)) and np.isfinite(v)})
    _release_memory()
    # Save results (also saved incrementally after every model for preemptible VMs)
    _save_results(results)
    if wrun is not None:
        try:
            P(f"wandb run dir (download before stopping the VM): {wrun.dir}")
            wrun.finish()
        except Exception as exc:
            P(f"wandb finish failed ({exc}) — results file is authoritative.")
    P("\nResults saved to p4_train_results.json")
