"""P4 end-to-end pipeline: batch generation, grouped splits, multiple models."""
import time
import numpy as np
import torch
from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver, SOUND_SPEED
from mechanics.p4_margin_estimation.transient import (
    compute_eigendecomposition,
    generate_clip_from_eigendecomposition,
)
from mechanics.p4_margin_estimation.train import (
    train, train_huber, train_median, evaluate_coverage,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)


def evaluate_point(model, clips, device="cpu", velocities=None):
    """Evaluate a single-output model (MAE only, no interval/coverage)."""
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
        pred = model(X).squeeze(-1)  # (B,)
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


def make_solver():
    face = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
                    nu12=0.33, rho=2710)
    core = Material(E1=4.73e7, E2=4.73e7, G23=1.01e9, G13=1.01e9, G12=1.20e7,
                    nu12=0.98, rho=278.15)
    lam = Laminate(materials=[face, core, face], angles=[0, 0, 0],
                   z=[-0.005, -0.004, 0.004, 0.005])
    solver = FSDTSolver(L1=1.0, L2=1.0, M=6, N=6, laminate=lam, grid=(30, 30),
                        k_stiffness=1e13)
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "free"},
        top={"type": "clamped"}, bottom={"type": "free"},
    )
    return solver


if __name__ == "__main__":
    P("1. Building solver...")
    solver = make_solver()

    P("2. Finding flutter boundary...")
    t0 = time.time()
    u_crit = solver.find_flutter_velocity(
        v_lower=680.0, v_upper=3000.0, n_scan=20, tol=1.0, n_modes=8)
    P(f"   u_crit={u_crit:.1f} m/s (Mach {u_crit/SOUND_SPEED:.2f}) ({time.time()-t0:.1f}s)")

    if u_crit is None:
        P("   ERROR: no flutter boundary found. Aborting.")
        exit(1)

    # Velocity levels: denser near flutter
    # 24 levels: 8 low (680-999), 8 mid (1001-1299), 8 high (1301-0.95*u_crit)
    v_hi = u_crit * 0.95
    v_lo = max(2.0 * SOUND_SPEED, u_crit * 0.4)  # M >= 2
    low_vels = np.linspace(v_lo, 999, 8)    # exclusive end
    mid_vels = np.linspace(1001, 1299, 8)  # exclusive ends
    high_vels = np.linspace(1301, v_hi, 8)  # inclusive start
    velocity_levels = np.concatenate([low_vels, mid_vels, high_vels])
    n_realizations = 10
    n_sensors = 8

    P(f"3. Generating {len(velocity_levels)} velocity levels × {n_realizations} realizations = {len(velocity_levels)*n_realizations} clips...")

    all_clips = []
    all_velocities = []
    t0 = time.time()
    for level_idx, v in enumerate(velocity_levels):
        # Compute eigendecomposition once per velocity (sensors embedded)
        try:
            eigs = compute_eigendecomposition(solver, float(v), n_modes=8,
                                              n_sensors=n_sensors)
        except Exception as e:
            P(f"   Level {level_idx+1}/{len(velocity_levels)} (V={v:.0f}) EIG FAILED: {e}")
            continue

        # Generate n_realizations clips with different random ICs
        rng_level = np.random.default_rng(42 + level_idx)
        level_clips = []
        for r in range(n_realizations):
            try:
                clip = generate_clip_from_eigendecomposition(
                    eigs, rng_level, n_timesteps=512, u_crit=u_crit,
                )
                level_clips.append(clip)
                all_velocities.append(float(v))
            except Exception as e:
                P(f"   Level {level_idx+1} Realization {r+1} (V={v:.0f}) REJECTED: {e}")
                continue

        all_clips.extend(level_clips)
        margin = level_clips[0].margin if level_clips else float('nan')
        P(f"   Level {level_idx+1}/{len(velocity_levels)} (V={v:.0f}, margin={margin:.3f}, {len(level_clips)} clips) {time.time()-t0:.1f}s")

    P(f"   Total: {len(all_clips)} clips generated")

    # Filter valid clips (non-NaN margin)
    valid_clips = [c for c in all_clips if not np.isnan(c.margin)]
    valid_vels = [v for v, c in zip(all_velocities, all_clips) if not np.isnan(c.margin)]
    P(f"   {len(valid_clips)} valid clips (margin not NaN)")

    if len(valid_clips) < 20:
        P("   ERROR: too few valid clips. Aborting.")
        exit(1)

    # ── Train 3 models with proper train/test split ──
    models = {
        "huber": (lambda c, v: train_huber(c, n_channels=n_sensors, hidden_dim=32,
                                           n_layers=4, epochs=50, lr=1e-3, velocities=v),
                  evaluate_point),
        "median": (lambda c, v: train_median(c, n_channels=n_sensors, hidden_dim=32,
                                             n_layers=4, epochs=50, lr=1e-3, velocities=v),
                   evaluate_point),
        "quantile": (lambda c, v: train(c, n_channels=n_sensors, hidden_dim=32,
                                        n_layers=4, epochs=50, lr=1e-3, velocities=v),
                     evaluate_coverage),
    }

    results = {}
    for name, (train_fn, eval_fn) in models.items():
        P(f"\n4. Training {name} model...")
        t0 = time.time()
        model, history, test_clips, test_vels = train_fn(valid_clips, valid_vels)
        P(f"   Train loss: {history['train_loss'][-1]:.4f}, Val loss: {history['val_loss'][-1]:.4f} ({time.time()-t0:.1f}s)")

        P(f"5. Evaluating {name} model on test set...")
        metrics = eval_fn(model, test_clips, velocities=test_vels)
        results[name] = metrics

        P(f"   MAE: {metrics.get('mae', 'N/A'):.4f}")
        P(f"   Interval width: {metrics.get('mean_interval_width', 'N/A'):.4f}")
        for k, v in metrics.get("coverage", {}).items():
            P(f"   {k}: {v:.3f}")

        # Per-velocity breakdown
        if "per_velocity" in metrics:
            P(f"   Per-velocity MAE:")
            for pv, data in sorted(metrics["per_velocity"].items()):
                P(f"     V={pv:.0f}: MAE={data['mae']:.4f}, n={data['n_clips']}")

        # Near-flutter performance (margin < 0.15) on test set
        near_flutter_test = [c for c in test_clips if c.margin < 0.15]
        if near_flutter_test:
            near_metrics = eval_fn(model, near_flutter_test)
            P(f"   Near-flutter (margin<0.15): MAE={near_metrics.get('mae', 'N/A'):.4f}, n={len(near_flutter_test)}")

        # Save model
        torch.save(model.state_dict(), f"p4_{name}.pt")
        P(f"   Saved model to p4_{name}.pt")

    # ── Summary ──
    P("\n" + "="*60)
    P("SUMMARY")
    P("="*60)
    P(f"{'Model':<12} {'MAE':>8} {'Interval':>10} {'Coverage':>10}")
    P("-"*45)
    for name, m in results.items():
        P(f"{name:<12} {m.get('mae', 0):>8.4f} {m.get('mean_interval_width', 0):>10.4f} "
          f"{m.get('coverage', {}).get('interval_0.05_0.95', 0):>10.3f}")
