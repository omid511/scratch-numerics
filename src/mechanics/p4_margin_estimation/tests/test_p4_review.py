"""Tier-1 checks for P4 research-review repairs (§4.1/4.2/4.4/5.1/5.2/3.2/6.4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechanics.p4_margin_estimation.transient import (
    Eigendecomposition,
    generate_clip_from_eigendecomposition,
)
from mechanics.p4_margin_estimation.evaluate import (
    lead_time_to_event,
    prefix_warning_available,
)


def _stub_eigs(n_modes=2, n_sensors=2, real=None, imag=None, seed=0):
    rng = np.random.default_rng(seed)
    MN_eff = 4
    size = 3 * MN_eff
    eigvals = np.asarray([
        complex(real[i] if real is not None else -1.0,
                imag[i] if imag is not None else (50.0 + 100.0 * i))
        for i in range(n_modes)
    ])
    eigvecs = rng.standard_normal((size, n_modes)) + 1j * rng.standard_normal((size, n_modes))
    sensor_modes = (rng.standard_normal((n_sensors, n_modes))
                    + 1j * rng.standard_normal((n_sensors, n_modes))) * 0.1
    return Eigendecomposition(
        eigvals=eigvals, eigvecs=eigvecs, size=size, MN_eff=MN_eff,
        velocity=800.0, M_mat=np.eye(size), K_total=np.eye(size), C_total=np.eye(size),
        sensor_iy=np.arange(n_sensors), sensor_ix=np.zeros(n_sensors, dtype=int),
        vx_grid=np.eye(2)[:1].repeat(2, axis=0)[:, :1].reshape(2) if False else np.ones(2),
        vy_grid=np.ones(2), M_eff=2, N_eff=2,
        rho=1.2, c_sound=340.0, zeta=0.0, sensor_modes=sensor_modes,
        t_span=(0.0, 0.5), dt=0.5 / 16,
    )


class TestClipProvenance:
    def test_dt_duration_critical_populated(self):
        eigs = _stub_eigs()
        rng = np.random.default_rng(0)
        clip = generate_clip_from_eigendecomposition(eigs, rng, n_timesteps=16, u_crit=1000.0)
        assert clip.dt == pytest.approx(float(clip.time[1] - clip.time[0]))
        assert clip.duration == pytest.approx(float(clip.time[-1] - clip.time[0] + clip.dt))
        assert clip.critical_idx == 0
        assert clip.omega_crit == pytest.approx(abs(clip.eigenvalues[0].imag))
        assert clip.alpha == pytest.approx(float(np.max(clip.eigenvalues.real)))
        assert np.isfinite(clip.clamp_frac) and np.isfinite(clip.max_log_amp)

    def test_clamp_incidence_recorded(self):
        # Strong growth: real=200 over t∈[0,0.5) peaks at 100 → must clamp at 20.
        eigs = _stub_eigs(real=[200.0, 200.0])
        rng = np.random.default_rng(1)
        clip = generate_clip_from_eigendecomposition(eigs, rng, n_timesteps=16, u_crit=1000.0)
        assert clip.clamp_frac > 0.0
        assert clip.max_log_amp > 20.0
        # Stable modes never saturate.
        eigs_s = _stub_eigs(real=[-2.0, -3.0], seed=3)
        clip_s = generate_clip_from_eigendecomposition(
            eigs_s, np.random.default_rng(2), n_timesteps=16, u_crit=1000.0)
        assert clip_s.clamp_frac == pytest.approx(0.0)

    def test_cached_path_matches_projection(self):
        from mechanics.p4_margin_estimation.transient import _clamped_modal_response
        eigs = _stub_eigs(seed=7)
        rng = np.random.default_rng(11)
        # Predict internal RNG draws: generate_clip draws amplitudes/phases first.
        from mechanics.p4_margin_estimation.transient import sample_modal_initial_conditions
        _, coeffs = sample_modal_initial_conditions(eigs.eigvals, np.random.default_rng(11))
        t = np.linspace(0.0, 0.5, 16, endpoint=False)
        expected_raw = 2.0 * np.real(eigs.sensor_modes @_clamped_modal_response(eigs.eigvals, t, coeffs))
        clip = generate_clip_from_eigendecomposition(eigs, rng, n_timesteps=16, u_crit=1000.0)
        # Normalization rescales; compare structure via correlation, not raw values.
        for s in range(expected_raw.shape[0]):
            a, b = expected_raw[s] - expected_raw[s].mean(), clip.sensor_signals[s] - clip.sensor_signals[s].mean()
            corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            assert corr > 0.99


class TestWarningVsEvent:
    def test_separate_thresholds(self):
        # Warning at idx 2 (<=0.15), true flutter (<=0.0) at idx 6 → lead 4 samples.
        pred = np.array([0.3, 0.2, 0.1, 0.05, 0.0, -0.05, -0.1])
        true = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.05, -0.05])
        assert lead_time_to_event(pred, true) == 4.0
        assert lead_time_to_event(pred, true, dt=0.01) == pytest.approx(0.04)

    def test_no_event_is_nan(self):
        pred = np.array([0.3, 0.1, 0.1])
        true = np.array([0.3, 0.2, 0.2])  # never reaches 0.0
        assert np.isnan(lead_time_to_event(pred, true))

    def test_bad_dt_raises(self):
        with pytest.raises(ValueError):
            lead_time_to_event(np.zeros(3), np.zeros(3), dt=-1.0)

    def test_prefix_availability_accounts_calibration(self):
        assert prefix_warning_available(512, 64, 10) == 63
        assert prefix_warning_available(512, 64, 100) == 100
        assert prefix_warning_available(512, 64, None) is None


class TestGrowthBaselineCalibrated:
    def _clip(self, rate, margin, did="D000000"):
        T = 64
        t = np.arange(T) / 1000.0
        sig = np.exp(rate * t)[None, :].repeat(2, axis=0)
        return SimpleNamespace(sensor_signals=sig, margin=margin, design_id=did, dt=1 / 1000.0, time=t)

    def test_fit_maps_growth_to_margin(self):
        from mechanics.p4_margin_estimation.baselines import GrowthRateBaseline
        bl = GrowthRateBaseline()
        train = [self._clip(-2.0, 0.2, "D000001"), self._clip(-1.0, 0.1, "D000001"),
                 self._clip(1.0, -0.1, "D000002"), self._clip(2.0, -0.2, "D000002")]
        bl.fit(train)
        assert bl.coef_ < 0.0
        pred = bl.predict_margin(train)
        assert np.all(np.isfinite(pred))
        # Ordering preserved: decaying → higher margin than growing.
        assert float(pred[0]) > float(pred[3])


class TestGruExplicitVal:
    def _clip(self, margin, did):
        return SimpleNamespace(sensor_signals=np.random.default_rng(0).standard_normal((4, 32)),
                              margin=margin, design_id=did, dt=0.001, time=np.arange(32) / 1000.0)

    def test_explicit_val_trains(self):
        from mechanics.p4_margin_estimation.baselines import train_gru
        train = [self._clip(0.1, "D000001") for _ in range(8)] + [self._clip(-0.1, "D000002") for _ in range(8)]
        val = [self._clip(0.05, "D000003") for _ in range(4)]
        _, hist = train_gru(train, n_channels=4, hidden_dim=8, epochs=2, batch_size=8,
                            seed=0, val_clips=val)
        assert len(hist["train_loss"]) == 2 and len(hist["val_loss"]) == 2

    def test_overlap_raises(self):
        from mechanics.p4_margin_estimation.baselines import train_gru
        train = [self._clip(0.1, "D000001") for _ in range(4)]
        val = [self._clip(0.05, "D000001") for _ in range(2)]
        with pytest.raises(ValueError):
            train_gru(train, n_channels=4, hidden_dim=8, epochs=1, batch_size=4,
                      seed=0, val_clips=val)


class TestLoaderProvenance:
    def test_legacy_dataset_loads_without_provenance(self, tmp_path):
        import train_p4_expanded as T
        n, C, Tm = 6, 8, 512
        rng = np.random.default_rng(0)
        np.save(tmp_path / "clips.npy", rng.standard_normal((n, C, Tm)).astype(np.float32) + 2.0)
        np.savez_compressed(tmp_path / "metadata_arrays.npz",
                            margins=np.linspace(-0.1, 0.3, n).astype(np.float32),
                            velocities=np.linspace(700, 900, n).astype(np.float32),
                            design_ids=np.array([f"D{i:06d}" for i in range(n)]),
                            realization_ids=np.zeros(n, dtype=int))
        meta = {"n_train": 2, "n_val": 2, "n_test": 2,
                "train_designs": ["D000000", "D000001"], "val_designs": ["D000002", "D000003"],
                "test_designs": ["D000004", "D000005"]}
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        clips_arr, margins, vels, dids, rids, m = T.load_dataset(str(tmp_path))
        assert m["provenance_available"] is False
        clips, _ = T.build_clips(clips_arr, margins, vels, dids, set(dids[:2]))
        assert all(getattr(c, "dt", None) is None for c in clips)

    def test_new_provenance_threads_dt(self, tmp_path):
        import train_p4_expanded as T
        n, C, Tm = 4, 8, 512
        rng = np.random.default_rng(1)
        np.save(tmp_path / "clips.npy", rng.standard_normal((n, C, Tm)).astype(np.float32) + 2.0)
        dts = np.array([0.001, 0.002, 0.001, 0.002], dtype=np.float32)
        np.savez_compressed(tmp_path / "metadata_arrays.npz",
                            margins=np.zeros(n, dtype=np.float32),
                            velocities=np.ones(n, dtype=np.float32) * 800,
                            design_ids=np.array([f"D{i:06d}" for i in range(n)]),
                            realization_ids=np.zeros(n, dtype=int),
                            dts=dts, durations=dts * Tm,
                            u_crits=np.ones(n, dtype=np.float32) * 1000,
                            clamp_fracs=np.zeros(n, dtype=np.float32),
                            max_log_amps=np.zeros(n, dtype=np.float32),
                            alphas=np.zeros(n, dtype=np.float32),
                            omega_crits=np.ones(n, dtype=np.float32) * 100)
        meta = {"n_train": 2, "n_val": 1, "n_test": 1,
                "train_designs": ["D000000", "D000001"], "val_designs": ["D000002"],
                "test_designs": ["D000003"]}
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        clips_arr, margins, vels, dids, rids, m = T.load_dataset(str(tmp_path))
        assert m["provenance_available"] is True
        clips, _ = T.build_clips(clips_arr, margins, vels, dids, {"D000000"}, dts=m["provenance"]["dts"])
        assert clips[0].dt == pytest.approx(0.001)


class TestAdaptiveDtGuard:
    """Review finding 1: the representability guard must never trip on rounding."""
    def test_guard_never_trips_on_sampled_band(self):
        from mechanics.p4_margin_estimation.transient import _adaptive_dt, NYQUIST_MARGIN, _DT_EPS
        rng = np.random.default_rng(0)
        omegas = np.exp(rng.uniform(np.log(10.0), np.log(1e6), size=100_000))
        dt_nominal = 0.5 / 512
        for w in omegas:
            dt = _adaptive_dt(float(w), dt_nominal)
            assert dt > 0.0 and dt <= dt_nominal
            assert float(w) <= NYQUIST_MARGIN * (np.pi / dt) * (1.0 + _DT_EPS)

    def test_degenerate_band_returns_nominal(self):
        from mechanics.p4_margin_estimation.transient import _adaptive_dt
        assert _adaptive_dt(0.0, 0.001) == 0.001
        assert _adaptive_dt(float("nan"), 0.001) == 0.001

    def test_invalid_nominal_raises(self):
        from mechanics.p4_margin_estimation.transient import _adaptive_dt
        with pytest.raises(ValueError):
            _adaptive_dt(100.0, -0.001)


class TestSelectCalibrateSplit:
    """Review finding 2: selection and calibration designs must be disjoint."""
    def test_single_design_raises_not_leaks(self):
        import train_p4_expanded as T
        with pytest.raises(ValueError):
            T._split_select_calibrate(["D000001"])

    def test_two_designs_split_one_each(self):
        import train_p4_expanded as T
        sel, cal = T._split_select_calibrate(["D000002", "D000001"])
        assert len(sel) == 1 and len(cal) == 1 and not (sel & cal)
        assert sel | cal == {"D000001", "D000002"}

    def test_larger_split_covers_disjointly(self):
        import train_p4_expanded as T
        ids = [f"D{i:06d}" for i in range(13)]
        sel, cal = T._split_select_calibrate(ids)
        assert sel | cal == set(ids) and not (sel & cal)
        assert len(sel) == 6 and len(cal) == 7


class TestPrefixBounds:
    """Review finding 5: impossible observation windows must raise."""
    def test_calibration_beyond_observation_raises(self):
        with pytest.raises(ValueError):
            prefix_warning_available(8, 64, 0)

    def test_negative_warn_index_raises(self):
        with pytest.raises(ValueError):
            prefix_warning_available(8, 1, -3)

    def test_warn_index_past_end_raises(self):
        with pytest.raises(ValueError):
            prefix_warning_available(8, 1, 8)

    def test_window_floor_honored(self):
        assert prefix_warning_available(512, 64, 10, min_samples=128) == 127
        assert prefix_warning_available(512, 64, 200, min_samples=128) == 200
        with pytest.raises(ValueError):
            prefix_warning_available(64, 8, 10, min_samples=128)


class TestConstrainedGrowthFit:
    """Review baseline concern: non-negative slope → boundary solution, not an arbitrary map."""
    def _clip(self, rate, margin, did="D000000"):
        T = 64
        t = np.arange(T) / 1000.0
        sig = np.exp(rate * t)[None, :].repeat(2, axis=0)
        return SimpleNamespace(sensor_signals=sig, margin=margin, design_id=did, dt=1 / 1000.0, time=t)

    def test_positive_association_yields_fitted_constant(self):
        from mechanics.p4_margin_estimation.baselines import GrowthRateBaseline
        bl = GrowthRateBaseline()
        train = [self._clip(1.0, 0.1, "D000001"), self._clip(2.0, 0.2, "D000001"),
                 self._clip(3.0, 0.3, "D000002")]
        bl.fit(train)
        assert bl.coef_ == 0.0
        assert bl.intercept_ == pytest.approx(0.2)
        pred = bl.predict_margin(train)
        assert np.allclose(pred, 0.2)


class TestLabelProvenanceDefaults:
    """Review finding 3: label fields default safely on unequipped decompositions."""
    def test_stub_clip_reports_unresolved(self):
        eigs = _stub_eigs()
        clip = generate_clip_from_eigendecomposition(
            eigs, np.random.default_rng(0), n_timesteps=16, u_crit=1000.0)
        assert np.isnan(clip.label_alpha) and np.isnan(clip.label_omega)
        assert clip.label_represented is False


class TestEigMatch:
    """Review finding 3: representation match is conjugation-tolerant but mode-strict."""
    def test_conjugate_branch_matches(self):
        from mechanics.p4_margin_estimation.transient import _eig_match
        retained = np.array([1.0 + 450.0j, -2.0 + 300.0j])
        assert _eig_match(retained, complex(1.0 - 450.0j)) is True
        assert _eig_match(retained, complex(1.0 + 450.0j)) is True

    def test_different_mode_misses(self):
        from mechanics.p4_margin_estimation.transient import _eig_match
        retained = np.array([1.0 + 450.0j, -2.0 + 300.0j])
        assert _eig_match(retained, complex(0.5 + 100.0j)) is False
        assert _eig_match(np.array([]), complex(1.0 + 1.0j)) is False


class TestStabilitySign:
    """Caveat 1: spectrum match does not imply stability-sign agreement."""
    def test_caveat_case_splits(self):
        from mechanics.p4_margin_estimation.transient import _eig_match, stability_sign_agrees
        assert _eig_match(np.array([-0.001 + 10000j]), 0.001 + 10000j) is True
        assert stability_sign_agrees(0.001, -0.001) is False

    def test_agreement_cases(self):
        from mechanics.p4_margin_estimation.transient import stability_sign_agrees
        assert stability_sign_agrees(2.0, 3.0) is True
        assert stability_sign_agrees(-2.0, -3.0) is True
        assert stability_sign_agrees(0.0, -5.0) is True  # marginal deadband
        assert stability_sign_agrees(5.0, 1e-9) is True  # marginal deadband
        assert stability_sign_agrees(float("nan"), 1.0) is False


class TestPrefixEdgeCases:
    """Caveat 3: non-integer indices and unvalidated None windows raise."""
    def test_fractional_index_raises(self):
        with pytest.raises(ValueError):
            prefix_warning_available(8, 1, -0.5)

    def test_none_with_impossible_window_raises(self):
        with pytest.raises(ValueError):
            prefix_warning_available(8, 64, None)

    def test_bool_index_raises(self):
        with pytest.raises(ValueError):
            prefix_warning_available(8, 1, True)


class TestSensorSaturation:
    """Exploding normalized transients are ADC-clipped, with incidence recorded."""
    def test_growth_clip_bounded_and_flagged(self):
        from mechanics.p4_margin_estimation.transient import SAT_LIMIT
        eigs = _stub_eigs(real=[200.0, 200.0])
        clip = generate_clip_from_eigendecomposition(
            eigs, np.random.default_rng(1), n_timesteps=16, u_crit=1000.0)
        assert float(np.abs(clip.sensor_signals).max()) <= SAT_LIMIT
        assert clip.sat_frac > 0.0

    def test_stable_clip_untouched(self):
        eigs = _stub_eigs(real=[-2.0, -3.0], seed=3)
        clip = generate_clip_from_eigendecomposition(
            eigs, np.random.default_rng(2), n_timesteps=16, u_crit=1000.0)
        assert clip.sat_frac == pytest.approx(0.0)
