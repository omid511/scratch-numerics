"""Tests for real_pipeline (synthetic-only, in-memory arrays).

No disk fixtures: all tests exercise the private *_on_arrays helpers directly
so they run without hf_dataset or recorded Part-1 data on disk.
"""
import math

import numpy as np
import pytest

from mechanics.p1_multifidelity.real_pipeline import (
    RealPipelineConfig,
    _fit_on,
    _fit_on_modeconditioned,
    _mode_onehot_gp_inputs,
    _frequency_gp_on_arrays,
    _loo_cv_on_arrays,
    _split_run_indices,
)


def _make_synthetic_fields(
    n_runs: int = 6,
    n_modes: int = 3,
    ny: int = 8,
    nx: int = 8,
    seed: int = 0,
):
    """Smooth low-frequency correction fields parameterized by theta.

    Fields are products of low-frequency sines whose amplitudes depend on the
    design parameters theta, so PCA + GP have genuine structure to learn.
    """
    rng = np.random.default_rng(seed)
    gx = np.linspace(0, 1, nx)
    gy = np.linspace(0, 1, ny)
    Xg, Yg = np.meshgrid(gx, gy)
    # Boundary-compatible basis (vanishes at edges).
    envelope = Xg * (1 - Xg) * Yg * (1 - Yg)

    theta = rng.uniform(0.1, 0.9, size=(n_runs, 5))
    fields = np.zeros((n_runs, n_modes, ny, nx))
    for r in range(n_runs):
        for m in range(n_modes):
            kx, ky = m + 1, m + 2
            amp = 1.0 + 2.0 * theta[r, 0] + 0.5 * theta[r, 1]
            phase = np.pi * theta[r, 2] + 0.3 * theta[r, 3]
            fields[r, m] = (
                amp
                * envelope
                * np.sin(kx * np.pi * Xg + phase)
                * np.sin(ky * np.pi * Yg)
            )
    return theta, fields


class TestSplitByRun:
    def test_disjoint_and_complete(self):
        config = RealPipelineConfig(seed=7)
        for n_runs in (2, 5, 10, 17):
            train_idx, test_idx = _split_run_indices(n_runs, config)
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
            assert set(train_idx.tolist()) | set(test_idx.tolist()) == set(range(n_runs))
            assert len(test_idx) == min(math.ceil(config.test_fraction * n_runs), n_runs - 1)

    def test_seeded_reproducible(self):
        t1 = _split_run_indices(10, RealPipelineConfig(seed=42))
        t2 = _split_run_indices(10, RealPipelineConfig(seed=42))
        assert np.array_equal(t1[0], t2[0])
        assert np.array_equal(t1[1], t2[1])

    def test_too_few_runs_raises(self):
        with pytest.raises(ValueError):
            _split_run_indices(1, RealPipelineConfig())


class TestFitOn:
    def test_split_by_run_integrity_and_shapes(self):
        theta, fields = _make_synthetic_fields()
        config = RealPipelineConfig(d_z=4, seed=3)
        train_idx, test_idx = _split_run_indices(theta.shape[0], config)
        result = _fit_on(theta, fields, train_idx, test_idx, config)

        metrics = result["metrics"]
        assert set(metrics) == {
            "reconstruction_mse",
            "coverage_proxy_2sigma",
            "n_train_runs",
            "n_test_runs",
            "d_z",
            "per_mode_mse",
        }
        assert metrics["n_train_runs"] == len(train_idx)
        assert metrics["n_test_runs"] == len(test_idx)

        n_test_samples = len(test_idx) * fields.shape[1]
        arrays = result["arrays"]
        assert arrays["z_train"].ndim == 2
        assert arrays["z_train"].shape[1] <= config.d_z
        assert arrays["z_test_pred_mean"].shape == (len(test_idx), metrics["d_z"])
        assert arrays["z_test_pred_var"].shape == (len(test_idx), metrics["d_z"])
        assert np.all(arrays["z_test_pred_var"] >= 0)
        assert arrays["theta_test"].shape == (len(test_idx), theta.shape[1])
        # Held-out runs never appear in training data of the GP.
        assert set(arrays["theta_test"][:, 0].tolist()).isdisjoint(
            set(theta[train_idx][:, 0].tolist())
        )

        ny, nx = fields.shape[2], fields.shape[3]
        decoded_probe = result["models"]["decoder"].decode(arrays["z_test_pred_mean"])
        assert decoded_probe.shape == (len(test_idx), ny, nx)
        assert metrics["per_mode_mse"].shape == (fields.shape[1],)
        assert metrics["reconstruction_mse"] >= 0.0
        assert 0.0 <= metrics["coverage_proxy_2sigma"] <= 1.0

    def test_decoder_boundary_enforced(self):
        theta, fields = _make_synthetic_fields(n_runs=6, seed=11)
        config = RealPipelineConfig(d_z=4, seed=5)
        result = _fit_on(theta, fields, np.arange(4), np.array([4, 5]), config)
        decoded = result["models"]["decoder"].decode(result["arrays"]["z_test_pred_mean"])
        assert np.allclose(decoded[:, 0, :], 0.0, atol=1e-12)
        assert np.allclose(decoded[:, -1, :], 0.0, atol=1e-12)
        assert np.allclose(decoded[:, :, 0], 0.0, atol=1e-12)
        assert np.allclose(decoded[:, :, -1], 0.0, atol=1e-12)

    def test_overlapping_splits_rejected(self):
        theta, fields = _make_synthetic_fields()
        config = RealPipelineConfig()
        with pytest.raises(ValueError):
            _fit_on(theta, fields, np.arange(4), np.arange(3, 6), config)

    def test_smooth_structure_beats_pca_baseline(self):
        """GP-decoded held-out fields should beat predicting the train-mean field."""
        theta, fields = _make_synthetic_fields(n_runs=10, n_modes=2, seed=21)
        config = RealPipelineConfig(d_z=4, seed=9)
        train_idx, test_idx = _split_run_indices(theta.shape[0], config)
        result = _fit_on(theta, fields, train_idx, test_idx, config)

        flat = fields.reshape(-1, *fields.shape[2:])
        n_modes = fields.shape[1]
        train_flat = flat[np.repeat(train_idx, n_modes)]
        mean_field = train_flat.mean(axis=0)
        test_flat = flat[np.repeat(test_idx, n_modes)]
        baseline_mse = float(np.mean((mean_field - test_flat) ** 2))

        assert result["metrics"]["reconstruction_mse"] < baseline_mse


class TestLeaveOneRunOut:
    def test_loo_small(self):
        theta, fields = _make_synthetic_fields(n_runs=5, n_modes=2, seed=13)
        config = RealPipelineConfig(d_z=3)
        out = _loo_cv_on_arrays(theta, fields, config)
        assert len(out["per_fold_mse"]) == 5
        assert all(m >= 0 for m in out["per_fold_mse"])
        assert 0.0 <= out["mean_coverage_proxy"] <= 1.0

    def test_max_runs_limits_folds(self):
        theta, fields = _make_synthetic_fields(n_runs=6, n_modes=2, seed=17)
        out = _loo_cv_on_arrays(theta, fields, RealPipelineConfig(d_z=3), max_runs=3)
        assert len(out["per_fold_mse"]) == 3


class TestFrequencyErrorGP:
    def _easy_target(self, n=24, seed=101):
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 1, size=(n, 5))
        # Mode 0: smooth function of theta -> GP must win easily.
        y0 = 5.0 + 20.0 * np.sin(2 * np.pi * theta[:, 0])
        # Mode 1: pure noise -> nothing learnable.
        y1 = rng.normal(0, 1, size=n)
        return theta, np.stack([y0, y1], axis=1)

    def test_shapes_and_keys(self):
        theta, rel_err = self._easy_target()
        out = _frequency_gp_on_arrays(theta, rel_err, RealPipelineConfig())
        assert set(out) == {"per_mode", "mean_loo_rmse_pct", "baseline_mean_rmse_pct"}
        assert len(out["per_mode"]) == 2
        for entry in out["per_mode"]:
            assert set(entry) == {
                "loo_rmse_pct",
                "loo_mae_pct",
                "loo_coverage_2sigma",
                "baseline_rmse_pct",
            }
            assert entry["loo_rmse_pct"] >= 0
            assert entry["loo_mae_pct"] >= 0
            assert 0.0 <= entry["loo_coverage_2sigma"] <= 1.0
        expected_mean = float(np.mean([e["loo_rmse_pct"] for e in out["per_mode"]]))
        assert out["mean_loo_rmse_pct"] == pytest.approx(expected_mean)

    def test_gp_beats_mean_baseline_on_easy_target(self):
        theta, rel_err = self._easy_target()
        out = _frequency_gp_on_arrays(theta, rel_err, RealPipelineConfig())
        assert out["per_mode"][0]["loo_rmse_pct"] < out["per_mode"][0]["baseline_rmse_pct"]

    def test_unlearnable_mode_not_worse_than_explained(self):
        """On pure noise the GP may lose to the baseline — but only modestly."""
        theta, rel_err = self._easy_target()
        out = _frequency_gp_on_arrays(theta, rel_err, RealPipelineConfig())
        noise_entry = out["per_mode"][1]
        assert (
            noise_entry["loo_rmse_pct"]
            <= 3.0 * noise_entry["baseline_rmse_pct"] + 1e-9
        )


class TestModeConditioned:
    def test_gp_input_shape_and_split_by_run(self):
        """(a) GP inputs are (n_train_samples, d_theta + n_modes);
        (b) held-out runs never leak into the GP training rows."""
        theta, fields = _make_synthetic_fields(n_runs=6, n_modes=3, seed=31)
        d_theta = theta.shape[1]
        config = RealPipelineConfig(d_z=4, seed=2)
        train_idx, test_idx = _split_run_indices(theta.shape[0], config)
        result = _fit_on_modeconditioned(theta, fields, train_idx, test_idx, config)

        gp_inputs = result["models"]["gp"]._theta_train
        assert gp_inputs.shape == (len(train_idx) * 3, d_theta + 3)
        # One-hot blocks: last n_modes columns are exactly a one-hot per row.
        onehot = gp_inputs[:, d_theta:]
        assert np.all(onehot.sum(axis=1) == 1.0)
        assert set(np.unique(onehot.argmax(axis=1))) == {0, 1, 2}

        # Split-by-run: every distinct theta row in GP inputs is a STANDARDIZED
        # TRAIN run theta (pipeline z-scores theta with train stats before
        # building GP inputs), and no held-out run's standardized theta leaks in.
        mu = theta[train_idx].mean(axis=0)
        sd = theta[train_idx].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        train_thetas = {tuple(np.round(row, 9)) for row in (theta[train_idx] - mu) / sd}
        test_thetas = {tuple(np.round((theta[i] - mu) / sd, 9)) for i in test_idx}
        input_thetas = {tuple(np.round(row, 9)) for row in np.unique(gp_inputs[:, :d_theta], axis=0)}
        assert input_thetas == train_thetas
        for held_out in test_thetas:
            assert all(
                not np.allclose(held_out, row, atol=1e-6) for row in train_thetas
            )

        metrics = result["metrics"]
        assert "zero_baseline_mse" in metrics and "skill_vs_zero" in metrics
        assert metrics["reconstruction_mse"] <= metrics["zero_baseline_mse"]

    def test_noncontiguous_split_row_ordering(self):
        """With non-adjacent held-out runs, z_test_pred_mean rows must align
        with theta[test_idx] / field-sample order (run-major, modes contiguous)."""
        theta, fields = _make_synthetic_fields(n_runs=6, n_modes=3, seed=41)
        d_theta = theta.shape[1]
        config = RealPipelineConfig(d_z=4, seed=8)
        train_idx = np.array([0, 2, 4])
        test_idx = np.array([1, 3, 5])  # interleaved, non-contiguous
        result = _fit_on_modeconditioned(theta, fields, train_idx, test_idx, config)

        mu = theta[train_idx].mean(axis=0)
        sd = np.where(theta[train_idx].std(axis=0) > 0, theta[train_idx].std(axis=0), 1.0)
        gp_X_expected = _mode_onehot_gp_inputs(
            np.repeat((theta[test_idx] - mu) / sd, 3, axis=0), 3
        )
        z_mean_rebuilt, _ = result["models"]["gp"].predict(gp_X_expected)
        assert np.allclose(result["arrays"]["z_test_pred_mean"], z_mean_rebuilt)

        # Rows of different held-out runs must be distinguishable.
        per_run = result["arrays"]["z_test_pred_mean"].reshape(3, 3, -1)
        assert not np.allclose(per_run[0], per_run[1])
        assert not np.allclose(per_run[1], per_run[2])



    def test_overlapping_splits_rejected(self):
        theta, fields = _make_synthetic_fields()
        with pytest.raises(ValueError):
            _fit_on_modeconditioned(
                theta, fields, np.arange(4), np.arange(3, 6), RealPipelineConfig()
            )
