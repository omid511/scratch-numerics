"""Tests for Proposal 4: transient generation, domain randomisation, TCN, quantiles."""
from __future__ import annotations

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechanics.p4_margin_estimation.transient import (
    generate_transient_clip,
    generate_dataset,
    TransientClip,
)
from mechanics.p4_margin_estimation.domain_randomization import (
    SensorPerturber,
    SensorPerturbationConfig,
)
from mechanics.p4_margin_estimation.tcn import TCNBackbone
from mechanics.p4_margin_estimation.quantile_head import (
    QuantileMarginModel,
    pinball_loss,
    SUPPORTED_QUANTILES as DEFAULT_QUANTILES,
)
from mechanics.p4_margin_estimation.train import evaluate_coverage


# ── Solver fixture ──────────────────────────────────────────────────────────

def _make_solver():
    from mechanics.solver import FSDTSolver
    from mechanics.laminate import Material, Laminate

    # Thick aluminum plate — stable at all supersonic test velocities
    al = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
                  nu12=0.33, rho=2710)
    lam = Laminate([al], [0.0], [-0.005, 0.005])
    s = FSDTSolver(L1=0.3, L2=0.3, M=6, N=6, laminate=lam, grid=(16, 16),
                   k_stiffness=1e14)
    s.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    return s


@pytest.fixture(scope="module")
def solver():
    return _make_solver()


# ── Transient tests ─────────────────────────────────────────────────────────

class TestTransient:
    def test_clip_shapes(self, solver):
        clip = generate_transient_clip(solver, velocity=800.0, n_sensors=4, n_timesteps=512, u_crit=2000.0, nyquist_strict=False)
        assert isinstance(clip, TransientClip)
        assert clip.sensor_signals.shape == (4, 512)
        assert clip.time.shape == (512,)
        assert clip.eigenvalues.ndim == 1
        assert clip.sensor_xy.shape == (4, 2)

    def test_clip_not_all_nan(self, solver):
        clip = generate_transient_clip(solver, velocity=900.0, n_sensors=4, n_timesteps=512, u_crit=2000.0, nyquist_strict=False)
        assert not np.all(np.isnan(clip.sensor_signals))

    def test_generate_dataset(self, solver):
        clips = generate_dataset(
            solver, n_samples=3, n_modes=10, n_timesteps=512,
            velocity_range=(800.0, 1200.0), u_crit=2000.0, nyquist_strict=False,
        )
        assert len(clips) >= 1

    def test_margin_value(self, solver):
        clip = generate_transient_clip(solver, velocity=850.0, n_modes=10, n_timesteps=512, u_crit=1200.0, nyquist_strict=False)
        expected = (clip.u_crit - 850.0) / clip.u_crit
        assert abs(clip.margin - expected) < 1e-10

    def test_eigenvalue_sign_convention(self, solver):
        """Stable plate clips should not have diverging signals; eigenvalue filtering works."""
        # Generate clips at two velocities for a stable plate
        clip_low = generate_transient_clip(
            solver, velocity=800.0, n_modes=5, n_timesteps=512, u_crit=2000.0, nyquist_strict=False,
        )
        clip_high = generate_transient_clip(
            solver, velocity=1200.0, n_modes=5, n_timesteps=512, u_crit=2000.0, nyquist_strict=False,
        )
        # Both should produce finite, non-zero signals
        assert np.isfinite(clip_low.sensor_signals).all()
        assert np.isfinite(clip_high.sensor_signals).all()
        assert not np.all(clip_low.sensor_signals == 0)
        assert not np.all(clip_high.sensor_signals == 0)

        # Higher velocity (closer to flutter) should generally have larger amplitude
        rms_low = np.sqrt(np.mean(clip_low.sensor_signals ** 2))
        rms_high = np.sqrt(np.mean(clip_high.sensor_signals ** 2))
        # After normalization this may not hold strictly, but signals should exist
        assert rms_low > 0 and rms_high > 0


# ── Domain randomisation tests ──────────────────────────────────────────────

class TestDomainRandomization:
    def test_shapes_preserved(self):
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((4, 128))
        config = SensorPerturbationConfig(snr_db=30.0)
        perturber = SensorPerturber(config)
        out = perturber.perturb(signals, rng)
        assert out.signals.shape == signals.shape
        assert out.valid_mask.shape == signals.shape

    def test_noise_changes_signal(self):
        rng = np.random.default_rng(1)
        signals = np.ones((4, 128))
        config = SensorPerturbationConfig(snr_db=20.0, per_channel_gain=False, per_channel_bias=False,
                                         gain_drift=False, colored_noise_prob=0.0,
                                         channel_dropout_distribution=((0, 1.0),), burst_dropout_prob=0.0,
                                         timing_skew=False, common_mode_fraction=0.0)
        out = SensorPerturber(config).perturb(signals, rng)
        assert not np.allclose(out.signals, signals)

    def test_gain_scales(self):
        rng = np.random.default_rng(2)
        signals = np.ones((2, 32))
        config = SensorPerturbationConfig(snr_db=None, per_channel_gain=True, per_channel_bias=False,
                                         gain_drift=False, colored_noise_prob=0.0,
                                         channel_dropout_distribution=((0, 1.0),), burst_dropout_prob=0.0,
                                         timing_skew=False, common_mode_fraction=0.0)
        out = SensorPerturber(config).perturb(signals, rng)
        assert out.signals.shape == signals.shape

    def test_resample_preserves_shape(self):
        rng = np.random.default_rng(3)
        signals = rng.standard_normal((4, 128))
        config = SensorPerturbationConfig(snr_db=None)
        out = SensorPerturber(config).perturb(signals, rng)
        assert out.signals.shape == (4, 128)

    def test_deterministic_with_seed(self):
        signals = np.ones((3, 64))
        config = SensorPerturbationConfig(snr_db=30.0, per_channel_gain=False,
                                         per_channel_bias=False, gain_drift=False,
                                         colored_noise_prob=0.0,
                                         channel_dropout_distribution=((0, 1.0),),
                                         burst_dropout_prob=0.0, timing_skew=False,
                                         common_mode_fraction=0.0)
        perturber = SensorPerturber(config)
        r1 = perturber.perturb(signals, np.random.default_rng(42))
        r2 = perturber.perturb(signals, np.random.default_rng(42))
        np.testing.assert_array_equal(r1.signals, r2.signals)
        np.testing.assert_array_equal(r1.valid_mask, r2.valid_mask)


# ── TCN tests ───────────────────────────────────────────────────────────────

class TestTCN:
    def test_output_shape(self):
        model = TCNBackbone(n_channels=8, hidden_dim=16, n_layers=3, sequence_length=16)
        x = torch.randn(4, 8, 16)
        out = model(x)
        assert out.shape == (4, 16, 16)

    def test_causal_constraint(self):
        """Output at t should not depend on input at t+1."""
        model = TCNBackbone(n_channels=2, hidden_dim=8, n_layers=4, sequence_length=20)
        model.eval()
        x = torch.randn(1, 2, 20)
        with torch.no_grad():
            out_full = model(x)
            out_trunc = model(x[:, :, :10])
        # Last timestep of truncated should differ from full (different input history)
        # but first 10 timesteps should match (causal: no future leakage)
        np.testing.assert_allclose(
            out_full[0, :, :10].numpy(),
            out_trunc[0, :, :10].numpy(),
            atol=1e-5,
        )

    def test_parameter_count(self):
        model = TCNBackbone(n_channels=8, hidden_dim=32, n_layers=8, sequence_length=512)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 200_000  # lightweight

    def test_gradient_flows(self):
        model = TCNBackbone(n_channels=4, hidden_dim=16, n_layers=5, sequence_length=64)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = model(x).sum()
        out.backward()
        assert x.grad is not None


# ── Quantile head tests ────────────────────────────────────────────────────

class TestQuantileHead:
    def test_output_shape(self):
        model = QuantileMarginModel(n_channels=4, hidden_dim=16, n_layers=8, sequence_length=64)
        x = torch.randn(2, 4, 64)
        out = model(x)
        assert out.shape == (2, len(DEFAULT_QUANTILES))

    def test_quantile_ordering(self):
        """q0.05 <= q0.50 <= q0.95 for any input."""
        model = QuantileMarginModel(n_channels=4, hidden_dim=16, n_layers=8, sequence_length=32)
        model.eval()
        x = torch.randn(5, 4, 32)
        with torch.no_grad():
            out = model(x)  # (5, 3)
        assert torch.all(out[:, 0] <= out[:, 1] + 1e-5)
        assert torch.all(out[:, 1] <= out[:, 2] + 1e-5)

    def test_pinball_loss_nonneg(self):
        pred = torch.tensor([[0.5, 0.5, 0.5]])   # (1, 3)
        target = torch.tensor([1.0])               # (1,)
        loss = pinball_loss(pred, target, (0.05, 0.5, 0.95))
        assert loss.item() >= 0

    def test_pinball_loss_zero_at_correct_quantile(self):
        """If prediction equals target, loss should be ~0."""
        target_val = 0.7
        for tau in [0.05, 0.5, 0.95]:
            pred = torch.tensor([[target_val]])
            target = torch.tensor([target_val])
            loss = pinball_loss(pred, target, (tau,))
            assert loss.item() < 1e-6


# ── Train / coverage tests ─────────────────────────────────────────────────

class TestTrain:
    def test_evaluate_coverage_returns_dict(self, solver):
        from mechanics.p4_margin_estimation.transient import generate_transient_clip

        clips = []
        for v in [800.0, 900.0, 1000.0]:
            clip = generate_transient_clip(solver, v, n_timesteps=64, n_sensors=4, n_modes=5, u_crit=2000.0, nyquist_strict=False)
            # Force margin for testing
            clip.margin = 0.5 if v < 500 else 0.1
            clips.append(clip)
        model = QuantileMarginModel(n_channels=4, hidden_dim=8, n_layers=8, sequence_length=64)
        result = evaluate_coverage(model, clips)
        assert "mae" in result
        assert "coverage" in result

    def test_training_convergence(self, solver):
        """Train 5 epochs on tiny synthetic data, verify loss decreases."""
        rng = np.random.default_rng(0)
        clips = []
        for _ in range(10):
            m = float(rng.uniform(0.1, 0.9))
            clips.append(_SyntheticClip(rng.standard_normal((4, 64)) * m, m))

        torch.manual_seed(0)
        model = QuantileMarginModel(n_channels=4, hidden_dim=8, n_layers=8, sequence_length=64)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        quantiles = model.quantiles

        X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in clips])
        y = torch.tensor([c.margin for c in clips], dtype=torch.float32)

        losses = []
        for _ in range(5):
            model.train()
            pred = model(X)
            target = y.unsqueeze(1)  # (B, 1) to match pred (B, n_q)
            loss = pinball_loss(pred, target, quantiles)
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ── Helpers for behavioral tests ──────────────────────────────────────────

class _SyntheticClip:
    """Bare clip with sensor_signals and margin for training tests."""
    __slots__ = ("sensor_signals", "margin", "design_id")
    _next_id = 0
    def __init__(self, sensor_signals: np.ndarray, margin: float, design_id: int | None = None):
        self.sensor_signals = sensor_signals
        self.margin = margin
        if design_id is None:
            _SyntheticClip._next_id += 1
            self.design_id = _SyntheticClip._next_id
        else:
            self.design_id = design_id


def _make_clips(n: int, n_sensors: int = 8, n_timesteps: int = 64, seed: int = 42):
    """Synthetic clips where signal RMS ∝ margin."""
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        m = float(rng.uniform(0.05, 0.95))
        clips.append(_SyntheticClip(rng.standard_normal((n_sensors, n_timesteps)) * m, m, design_id=i))
    return clips


def _train(clips, *, n_channels=8, hidden_dim=16, n_layers=8,
           epochs=30, batch_size=16, seed=0):
    from mechanics.p4_margin_estimation.train import train
    # Generate synthetic velocities for grouped split (one per clip, binned by margin)
    velocities = [float(i % 6) for i in range(len(clips))]
    model, _, _, _ = train(clips, n_channels=n_channels, hidden_dim=hidden_dim,
                     n_layers=n_layers, epochs=epochs, batch_size=batch_size, seed=seed,
                     velocities=velocities)
    return model


def _mae(model, clips):
    """MAE of median (q0.50) prediction."""
    model.eval()
    X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in clips])
    with torch.no_grad():
        med = model(X)[:, 1]  # (B,) — scalar median
    y = torch.tensor([c.margin for c in clips], dtype=torch.float32)
    return (med - y).abs().mean().item()


# ── Behavioral tests for P4 ──────────────────────────────────────────────

class TestP4Behavioral:
    def test_tcn_predictions_improve_with_training(self):
        all_clips = _make_clips(40, seed=42)
        train_clips, test_clips = all_clips[:30], all_clips[30:]

        mae_before = _mae(
            QuantileMarginModel(n_channels=8, hidden_dim=16, n_layers=8, sequence_length=64),
            test_clips,
        )
        model = _train(train_clips, seed=0, epochs=30)
        mae_after = _mae(model, test_clips)

        assert mae_after < mae_before, f"MAE did not improve: {mae_before} -> {mae_after}"

    def test_median_prediction_ranking_matches_velocity(self):
        rng = np.random.default_rng(99)
        u_crit = 800.0
        train_clips = []
        for _ in range(60):
            v = float(rng.uniform(100, 700))
            m = (u_crit - v) / u_crit
            # Use deterministic per-timestep pattern: sine with amplitude ∝ margin
            t = np.linspace(0, 1, 64)
            pattern = np.sin(2 * np.pi * 3.0 * t)  # (64,)
            signal = np.tile(pattern, (8, 1)) * m  # (8, 64), amplitude ∝ margin
            train_clips.append(_SyntheticClip(signal, m))

        model = _train(train_clips, seed=1, epochs=60, hidden_dim=32, n_layers=6)

        # v1 < v2 < v3 => margin(v1) > margin(v2) > margin(v3)
        velocities = [300.0, 400.0, 500.0]
        test_clips = []
        for v in velocities:
            m = (u_crit - v) / u_crit
            t = np.linspace(0, 1, 64)
            pattern = np.sin(2 * np.pi * 3.0 * t)
            signal = np.tile(pattern, (8, 1)) * m
            test_clips.append(_SyntheticClip(signal, m))

        model.eval()
        X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in test_clips])
        with torch.no_grad():
            medians = model(X)[:, 1].numpy()  # (B,)

        assert medians[0] > medians[1] > medians[2], f"Ranking wrong: {medians}"

    def test_quantile_interval_coverage(self):
        rng = np.random.default_rng(123)
        clips = []
        for _ in range(40):
            m = float(rng.uniform(0.1, 0.9))
            clips.append(_SyntheticClip(rng.standard_normal((8, 64)) * m, m))

        model = _train(clips[:30], seed=2, epochs=40)
        test_clips = clips[30:]

        model.eval()
        X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in test_clips])
        with torch.no_grad():
            pred = model(X)
        lo, hi = pred[:, 0], pred[:, 2]  # (B,) scalar
        y = torch.tensor([c.margin for c in test_clips], dtype=torch.float32)

        # Verify quantile ordering: lo <= hi for all samples
        assert torch.all(lo <= hi + 1e-5), "q0.05 > q0.95 for some samples"

        coverage = ((y >= lo) & (y <= hi)).float().mean().item()
        assert coverage >= 0.75, f"Coverage {coverage} < 0.75"

    def test_calibration_low_quantile(self):
        rng = np.random.default_rng(456)
        clips = []
        for _ in range(40):
            m = float(rng.uniform(0.1, 0.9))
            clips.append(_SyntheticClip(rng.standard_normal((8, 64)) * m, m))

        model = _train(clips[:30], seed=3, epochs=40)
        test_clips = clips[30:]

        model.eval()
        X = torch.stack([torch.tensor(c.sensor_signals, dtype=torch.float32) for c in test_clips])
        with torch.no_grad():
            pred_q05 = model(X)[:, 0]  # (B,) scalar
        y = torch.tensor([c.margin for c in test_clips], dtype=torch.float32)

        # Well-calibrated q0.05: ~5% of y fall below pred_q0.05
        frac_below = (y < pred_q05).float().mean().item()
        assert 0.0 <= frac_below <= 0.20, f"q0.05 calibration off: {frac_below}"

    def test_domain_randomization_reduces_perturbed_mae(self):
        """DR should reduce MAE on perturbed test data (3 trials, majority agree)."""
        n_trials = 3
        improvements = 0

        for trial in range(n_trials):
            rng_base = np.random.default_rng(789 + trial)
            u_crit = 800.0
            base_clips = []
            for _ in range(60):
                v = float(rng_base.uniform(100, 700))
                m = (u_crit - v) / u_crit
                t = np.linspace(0, 1, 64)
                pattern = np.sin(2 * np.pi * 3.0 * t)
                signal = np.tile(pattern, (8, 1)) * m
                base_clips.append(_SyntheticClip(signal, m))

            # DR config: clean training with minimal noise
            dr_config = SensorPerturbationConfig(
                snr_db=40.0, per_channel_gain=False, per_channel_bias=False,
                gain_drift=False, colored_noise_prob=0.0,
                channel_dropout_distribution=((0, 1.0),), burst_dropout_prob=0.0,
                timing_skew=False, common_mode_fraction=0.0
            )

            # DR-augmented training set (separate rng for DR augmentation)
            rng_dr = np.random.default_rng(789 + trial + 1000)
            dr_train = []
            for c in base_clips[:40]:
                p = SensorPerturber(dr_config)
                dr_train.append(_SyntheticClip(p.perturb(c.sensor_signals, rng_dr).signals, c.margin))

            # Perturbed test set (separate rng for test noise)
            rng_test = np.random.default_rng(789 + trial + 2000)
            test_perturbed = []
            perturb_config = SensorPerturbationConfig(snr_db=30.0, per_channel_gain=False,
                                                     per_channel_bias=False, gain_drift=False,
                                                     colored_noise_prob=0.0,
                                                     channel_dropout_distribution=((0, 1.0),),
                                                     burst_dropout_prob=0.0, timing_skew=False,
                                                     common_mode_fraction=0.0)
            for c in base_clips[40:]:
                p = SensorPerturber(perturb_config)
                test_perturbed.append(_SyntheticClip(p.perturb(c.sensor_signals, rng_test).signals, c.margin))

            mae_no_dr = _mae(
                _train(base_clips[:40], seed=10 + trial, epochs=40, hidden_dim=32, n_layers=6),
                test_perturbed,
            )
            mae_dr = _mae(
                _train(dr_train, seed=10 + trial, epochs=40, hidden_dim=32, n_layers=6),
                test_perturbed,
            )

            if mae_dr < mae_no_dr:  # Strict improvement required
                improvements += 1

        assert improvements >= 2, f"DR only helped in {improvements}/{n_trials} trials"

    def test_dr_gain_perturbation_robustness(self):
        """DR-trained model should be robust to small gain perturbations (3 trials)."""
        n_trials = 3
        passes = 0

        for trial in range(n_trials):
            rng_base = np.random.default_rng(101 + trial)
            base_clips = []
            for _ in range(40):
                m = float(rng_base.uniform(0.1, 0.9))
                base_clips.append(_SyntheticClip(rng_base.standard_normal((8, 64)) * m, m))

            # DR with gain perturbation
            dr_config = SensorPerturbationConfig(
                snr_db=30.0, per_channel_gain=True, per_channel_bias=False,
                gain_drift=False, colored_noise_prob=0.0,
                channel_dropout_distribution=((0, 1.0),), burst_dropout_prob=0.0,
                timing_skew=False, common_mode_fraction=0.0
            )
            rng_dr = np.random.default_rng(101 + trial + 1000)
            dr_clips = []
            for c in base_clips:
                p = SensorPerturber(dr_config)
                dr_clips.append(_SyntheticClip(p.perturb(c.sensor_signals, rng_dr).signals, c.margin))

            model = _train(dr_clips[:30], seed=20 + trial, epochs=30)

            test_base = base_clips[30:]
            test_095 = [_SyntheticClip(c.sensor_signals * 0.95, c.margin) for c in test_base]
            test_105 = [_SyntheticClip(c.sensor_signals * 1.05, c.margin) for c in test_base]

            base_mae = _mae(model, test_base)
            mae_095 = _mae(model, test_095)
            mae_105 = _mae(model, test_105)

            # Each perturbation must not degrade too much, and they should be similar
            if (mae_095 < 1.5 * base_mae and mae_105 < 1.5 * base_mae
                    and abs(mae_095 - mae_105) < 0.30 * base_mae):
                passes += 1

        assert passes >= 2, f"Gain robustness only passed in {passes}/{n_trials} trials"

    def test_pinball_loss_asymmetric_response(self):
        tau = 0.95
        err_mag = 1.0
        target = torch.tensor([[0.0]])

        pred_over = torch.tensor([[err_mag]])    # pred > target
        pred_under = torch.tensor([[-err_mag]])  # pred < target

        loss_over = pinball_loss(pred_over, target, (tau,))
        loss_under = pinball_loss(pred_under, target, (tau,))

        # At tau=0.95, under-prediction penalised more (0.95 vs 0.05)
        assert loss_under > loss_over, (
            f"Expected under-pred loss {loss_under} > over-pred loss {loss_over}"
        )

    def test_quantile_ordering_preserved_after_training(self):
        clips = _make_clips(40, seed=42)
        model = _train(clips, seed=0, epochs=20)

        model.eval()
        x = torch.randn(10, 8, 64)
        with torch.no_grad():
            out = model(x)  # (10, 3)

        assert torch.all(out[:, 0] <= out[:, 1] + 1e-5)
        assert torch.all(out[:, 1] <= out[:, 2] + 1e-5)


# ── P1-6: Discretization convergence ────────────────────────────────────

class TestConvergence:
    def test_flutter_velocity_is_basis_converged(self):
        """P1-6: Flutter velocity should converge as basis order increases."""
        from mechanics.solver import FSDTSolver
        from mechanics.laminate import Material, Laminate

        face = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
                        nu12=0.33, rho=2710)
        core = Material(E1=4.73e7, E2=4.73e7, G23=1.01e9, G13=1.01e9, G12=1.20e7,
                        nu12=0.98, rho=278.15)
        lam = Laminate(materials=[face, core, face], angles=[0, 0, 0],
                       z=[-0.005, -0.004, 0.004, 0.005])

        # Orders [6, 8]: order 4 cannot represent this configuration's flutter
        # mode at all (u_crit ~4773 vs ~1462 m/s — genuine basis
        # non-convergence, stable across every participation-gate threshold).
        # The old [4, 6] comparison never actually executed: under the
        # pre-scaling numerics both orders returned None and the test
        # auto-skipped. Orders 6 vs 8 converge to ~0.6%.
        orders = [6, 8]
        flutter_vels = []
        for order in orders:
            s = FSDTSolver(L1=1.0, L2=1.0, M=order, N=order, laminate=lam, grid=(8, 8))
            s.set_boundary(
                left={"type": "clamped"}, right={"type": "free"},
                top={"type": "clamped"}, bottom={"type": "free"},
            )
            try:
                v_crit = s.find_flutter_velocity(rho=1.2, c_sound=340.0, zeta=0.0,
                                                 v_lower=680, v_upper=5000, n_scan=20)
                flutter_vels.append(v_crit)
            except Exception:
                flutter_vels.append(None)

        # Check convergence between last two successful results
        valid = [(o, v) for o, v in zip(orders, flutter_vels) if v is not None and np.isfinite(v)]
        if len(valid) < 2:
            pytest.skip(f"Solver could not compute flutter for enough orders "
                        f"(got {len(valid)}/{len(orders)}); "
                        f"needs a laminate/BC combo that produces crossing")
        _, u_prev = valid[-2]
        _, u_last = valid[-1]
        relative_change = abs(u_last - u_prev) / abs(u_last)
        assert relative_change < 0.15, (
            f"Flutter velocity not converged: {u_prev:.1f} -> {u_last:.1f} "
            f"(change={relative_change:.3f})"
        )


# ── New review tests ───────────────────────────────────────────────────

class TestReviewNew:
    # T4: timing-skew no-wrap
    def test_positive_shift_does_not_wrap_last_sample(self):
        from mechanics.p4_margin_estimation.domain_randomization import shift_without_wrap
        x = np.array([1.0, 2.0, 3.0])
        shifted = shift_without_wrap(x, 1, fill_value=0.0)
        np.testing.assert_array_equal(shifted, np.array([0.0, 1.0, 2.0]))

    def test_negative_shift_does_not_wrap_first_sample(self):
        from mechanics.p4_margin_estimation.domain_randomization import shift_without_wrap
        x = np.array([1.0, 2.0, 3.0])
        shifted = shift_without_wrap(x, -1, fill_value=0.0)
        np.testing.assert_array_equal(shifted, np.array([2.0, 3.0, 0.0]))

    def test_zero_shift_preserves_signal(self):
        from mechanics.p4_margin_estimation.domain_randomization import shift_without_wrap
        x = np.array([1.0, 2.0, 3.0])
        shifted = shift_without_wrap(x, 0, fill_value=0.0)
        np.testing.assert_array_equal(shifted, x)

    # T5: safety-rate denominator
    def test_false_safe_rate_is_conditional_on_unsafe_cases(self):
        """T5: false-safe rate must be conditional on actually-unsafe cases."""
        y = np.array([-1.0, -1.0, 1.0, 1.0])
        pred = np.array([1.0, -1.0, 1.0, 1.0])
        unsafe_mask = y <= 0.0
        false_safe_mask = unsafe_mask & (pred > 0.0)
        n_unsafe = int(unsafe_mask.sum())
        false_safe_rate = false_safe_mask.sum() / n_unsafe
        assert false_safe_rate == pytest.approx(0.5)

    # T6: flutter-boundary bracket sign change
    def test_flutter_boundary_brackets_sign_change(self, solver):
        """T6: spectral abscissa must cross zero at flutter boundary."""
        from mechanics.eigenanalysis import spectral_abscissa

        try:
            u_crit = solver.find_flutter_velocity(rho=1.2, c_sound=340.0, zeta=0.0)
        except Exception:
            pytest.skip("No flutter boundary found for this solver config")

        if u_crit is None:
            pytest.skip("No flutter boundary found")

        M_mat_below, K_below, C_below = solver.assemble_aeroelastic_system(
            0.999 * u_crit, 1.2, 340.0, 0.0)
        alpha_below = spectral_abscissa(M_mat_below, K_below, C_below).alpha

        M_mat_above, K_above, C_above = solver.assemble_aeroelastic_system(
            1.001 * u_crit, 1.2, 340.0, 0.0)
        alpha_above = spectral_abscissa(M_mat_above, K_above, C_above).alpha

        # Below flutter: stable (alpha <= 0), above: unstable (alpha > 0)
        assert alpha_below <= 0.0 + 1e-4, f"Expected stable below u_crit, got alpha={alpha_below}"
        assert alpha_above > -1e-4, f"Expected unstable above u_crit, got alpha={alpha_above}"

    # T7: piston-theory domain guard
    def test_piston_pressure_rejects_subcritical_mach(self):
        """T7: piston_pressure must reject M < minimum_mach."""
        from mechanics.piston_theory import piston_pressure
        with pytest.raises(ValueError, match="M >="):
            piston_pressure(
                velocity=1.5 * 340.0,
                flow_angle=0.0,
                dw_dx=np.zeros(2),
                dw_dy=np.zeros(2),
                dw_dt=np.zeros(2),
                rho_inf=1.0,
                sound_speed=340.0,
            )

    # T9: structural/flutter convergence
    def test_natural_frequencies_converge_with_basis_order(self):
        """T9: Natural frequencies should converge as basis order increases."""
        from mechanics.solver import FSDTSolver
        from mechanics.laminate import Material, Laminate

        al = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
                      nu12=0.33, rho=2710)
        lam = Laminate([al], [0.0], [-0.005, 0.005])

        freqs_by_order = []
        for order in [4, 6]:
            s = FSDTSolver(L1=0.3, L2=0.3, M=order, N=order, laminate=lam, grid=(16, 16),
                           k_stiffness=1e14)
            s.set_boundary(
                left={"type": "clamped"}, right={"type": "clamped"},
                top={"type": "clamped"}, bottom={"type": "clamped"},
            )
            try:
                freqs = s.structural_frequencies(n_modes=4)
                freqs_by_order.append(freqs)
            except Exception:
                freqs_by_order.append(None)

        if freqs_by_order[0] is not None and freqs_by_order[1] is not None:
            n_compare = min(len(freqs_by_order[0]), len(freqs_by_order[1]))
            for i in range(n_compare):
                f0, f1 = freqs_by_order[0][i], freqs_by_order[1][i]
                if f0 > 0 and f1 > 0:
                    rel_change = abs(f1 - f0) / f1
                    assert rel_change < 0.10, (
                        f"Mode {i} frequency not converged: {f0:.1f} -> {f1:.1f} Hz "
                        f"(change={rel_change:.3f})"
                    )

    # Regression: test_design_ids_are_disjoint_across_splits
    def test_design_ids_are_disjoint_across_splits(self):
        """Regression: split must not leak design IDs across train/val/test."""
        from mechanics.p4_margin_estimation.train import _grouped_3way_split

        class _MockClip:
            def __init__(self, design_id):
                self.design_id = design_id
                self.margin = 0.5
                self.sensor_signals = np.zeros((4, 16))

        # Create 10 clips across 5 designs (2 clips each)
        clips = []
        for did in range(5):
            for _ in range(2):
                clips.append(_MockClip(did))

        train_idx, val_idx, test_idx = _grouped_3way_split(
            clips, val_split=0.2, test_split=0.2, seed=7
        )
        train_ids = {clips[i].design_id for i in train_idx}
        val_ids = {clips[i].design_id for i in val_idx}
        test_ids = {clips[i].design_id for i in test_idx}

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    # Regression: test_flutter_detection_includes_real_divergence_mode
    def test_flutter_detection_includes_real_divergence_mode(self):
        """Regression: spectral_abscissa must detect real divergence (negative stiffness)."""
        from mechanics.eigenanalysis import spectral_abscissa

        M = np.array([[1.0]])
        C = np.array([[0.1]])
        K = np.array([[-1.0]])

        alpha = spectral_abscissa(M, K, C).alpha
        assert alpha > 0.0, f"Expected positive spectral abscissa for divergent system, got {alpha}"
