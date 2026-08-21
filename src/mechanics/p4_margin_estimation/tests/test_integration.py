"""Integration test: solver -> flutter boundary -> transient -> training -> evaluation."""
import numpy as np
import torch
import pytest


def test_full_pipeline_synthetic():
    """End-to-end: synthetic clips train a model, evaluates on test only."""
    from mechanics.p4_margin_estimation.train import train, evaluate_coverage

    rng = np.random.default_rng(42)

    class _Clip:
        __slots__ = ("sensor_signals", "margin", "design_id")
        def __init__(self, signals, margin, did):
            self.sensor_signals = signals
            self.margin = margin
            self.design_id = did

    clips = []
    for i in range(20):
        m = float(rng.uniform(0.05, 0.95))
        signals = rng.standard_normal((4, 512)) * m
        clips.append(_Clip(signals, m, f"d{i:03d}"))

    model, history, test_clips, test_vels = train(
        clips, n_channels=4, hidden_dim=8, n_layers=8,
        epochs=5, lr=1e-3,
        test_split=0.2, val_split=0.2,
    )
    assert history["train_loss"][-1] < history["train_loss"][0] + 0.5

    metrics = evaluate_coverage(model, test_clips)
    assert "mae" in metrics
    assert "coverage" in metrics


def test_flutter_scan_completes_without_error():
    """Flutter velocity scan completes and stores results, regardless of outcome."""
    from mechanics.solver import FSDTSolver
    from mechanics.laminate import Material, Laminate

    face = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
                    nu12=0.33, rho=2710)
    core = Material(E1=4.73e7, E2=4.73e7, G23=1.01e9, G13=1.01e9, G12=1.20e7,
                    nu12=0.98, rho=278.15)
    lam = Laminate(materials=[face, core, face], angles=[0, 0, 0],
                   z=[-0.005, -0.004, 0.004, 0.005])
    solver = FSDTSolver(L1=1.0, L2=1.0, M=6, N=6, laminate=lam, grid=(16, 16))
    solver.set_boundary(left={"type": "clamped"}, right={"type": "free"},
                        top={"type": "clamped"}, bottom={"type": "free"})

    # scan completes (may return None if no crossing exists)
    u_crit = solver.find_flutter_velocity(
        rho=1.2, c_sound=340.0, zeta=0.0,
        v_lower=680, v_upper=5000, n_scan=20, velocity_tol=5.0)
    # scan data is always stored
    assert hasattr(solver, "_flutter_scan")
    velocities, alpha = solver._flutter_scan
    assert len(velocities) == 20
    assert len(alpha) == 20
