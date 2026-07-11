"""Integration test: solver -> flutter boundary -> transient -> training -> evaluation."""
import numpy as np
import torch
import pytest


def test_full_pipeline_stable_plate():
    """End-to-end: stable plate generates clips, trains model, evaluates on test only."""
    from mechanics.solver import FSDTSolver
    from mechanics.laminate import Material, Laminate
    from mechanics.p4_margin_estimation.transient import (
        generate_transient_clip, default_sensor_xy
    )
    from mechanics.p4_margin_estimation.train import train, evaluate_coverage
    from mechanics.p4_margin_estimation.quantile_head import QuantileMarginModel

    # 1. Build solver
    al = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
                  nu12=0.33, rho=2710)
    lam = Laminate([al], [0.0], [-0.005, 0.005])
    solver = FSDTSolver(L1=0.3, L2=0.3, M=6, N=6, laminate=lam, grid=(16, 16),
                        k_stiffness=1e14)
    solver.set_boundary(left={"type": "clamped"}, right={"type": "clamped"},
                        top={"type": "clamped"}, bottom={"type": "clamped"})

    # 2. Generate clips with distinct velocities for grouped split
    rng = np.random.default_rng(42)
    velocity_levels = np.linspace(500.0, 1200.0, 6)
    sensor_xy = default_sensor_xy(16, 16, n_sensors=4)
    clips = []
    velocities = []
    for v in velocity_levels:
        try:
            clip = generate_transient_clip(solver, float(v), n_timesteps=64,
                                           n_sensors=4, n_modes=5, rng=rng,
                                           sensor_xy=sensor_xy, u_crit=2000.0)
            clips.append(clip)
            velocities.append(float(v))
        except ValueError:
            continue
    assert len(clips) >= 4, f"Too few clips: {len(clips)}"

    # 3. Train with grouped 3-way split
    model, history, test_clips, test_vels = train(
        clips, n_channels=4, hidden_dim=8, n_layers=2,
        epochs=5, lr=1e-3, velocities=velocities,
        test_split=0.2, val_split=0.2
    )
    assert history["train_loss"][-1] < history["train_loss"][0] + 0.1

    # 4. Evaluate only on test clips
    metrics = evaluate_coverage(model, test_clips, velocities=test_vels)
    assert "mae" in metrics
    assert "coverage" in metrics

    # 5. Assert no velocity overlap between train/val/test
    test_clips_set = set(id(c) for c in test_clips)
    train_vels = set(v for c, v in zip(clips, velocities) if id(c) not in test_clips_set)
    test_vels_set = set(test_vels)
    assert len(train_vels & test_vels_set) == 0, "Velocity overlap between train and test"


def test_flutter_boundary_consistency():
    """Flutter boundary via velocity scan returns a finite critical velocity."""
    from mechanics.solver import FSDTSolver
    from mechanics.laminate import Material, Laminate

    face = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
                    nu12=0.33, rho=2710)
    core = Material(E1=4.73e7, E2=4.73e7, G23=1.01e9, G13=1.01e9, G12=1.20e7,
                    nu12=0.98, rho=278.15)
    lam = Laminate(materials=[face, core, face], angles=[0, 0, 0],
                   z=[-0.005, -0.004, 0.004, 0.005])
    solver = FSDTSolver(L1=1.0, L2=1.0, M=6, N=6, laminate=lam, grid=(30, 30))
    solver.set_boundary(left={"type": "clamped"}, right={"type": "free"},
                        top={"type": "clamped"}, bottom={"type": "free"})

    u_crit = solver.find_flutter_velocity(v_lower=680, v_upper=20000, n_scan=40, tol=5.0)
    assert u_crit is not None
    assert 1000 < u_crit < 2000, f"Unexpected u_crit: {u_crit}"
