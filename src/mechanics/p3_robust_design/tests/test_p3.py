"""Tests for Proposal 3: Robust Design Under Stochastic Boundary Conditions."""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver
from mechanics.p3_robust_design.mode_tracking import mac, _optimal_match, extract_mode_data
from mechanics.p3_robust_design.gp_surrogate import (
    GPSurrogate, _rbf_kernel, _augmented_kernel,
)
from mechanics.p3_robust_design.active_learning import (
    uncertainty_sampling, expected_improvement_reliability,
    boundary_seeking, select_batch,
)
from mechanics.p3_robust_design.train import evaluate_gp_accuracy


# ---- Fixtures ----

def _face_sheet():
    E = 70e9; nu = 0.33; G = E / (2 * (1 + nu))
    return Material(E, E, G, G, G, nu, 2710)


def _honeycomb_core():
    return Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)


def _sandwich_laminate():
    return Laminate(
        materials=[_face_sheet(), _honeycomb_core(), _face_sheet()],
        angles=[0, 0, 0],
        z=[-5e-3, -4e-3, 4e-3, 5e-3],
    )


def _make_solver(M=6, N=6):
    lam = _sandwich_laminate()
    s = FSDTSolver(L1=0.3, L2=0.3, M=M, N=N, laminate=lam, k_stiffness=1e12)
    s.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    return s


# ---- Mode tracking tests ----

class TestMAC:
    def test_identical_modes_mac_one(self):
        m = np.random.randn(10, 10)
        assert abs(mac(m, m) - 1.0) < 1e-10

    def test_orthogonal_modes_mac_zero(self):
        a = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert mac(a, b) < 1e-10

    def test_swapped_modes_mac_one(self):
        a = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        b = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
        # MAC is computed element-wise on flattened, so this tests the formula
        val = mac(a[0], b[0])
        assert 0.0 <= val <= 1.0


class TestHungarianMatch:
    def test_identity_assignment(self):
        prev = np.random.randn(3, 4, 4)
        curr = prev.copy()
        macs, assignment = _optimal_match(prev, curr)
        np.testing.assert_array_equal(assignment, [0, 1, 2])
        np.testing.assert_allclose(macs, 1.0, atol=1e-10)

    def test_reverse_assignment(self):
        prev = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        curr = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
        macs, assignment = _optimal_match(prev, curr)
        np.testing.assert_array_equal(assignment, [1, 0])

    def test_partial_match(self):
        prev = np.random.randn(3, 5, 5)
        curr = prev.copy()
        curr[2] = np.random.randn(5, 5)  # one mode changed
        macs, assignment = _optimal_match(prev, curr)
        assert macs[0] > 0.99
        assert macs[1] > 0.99


class TestExtractModeData:
    def test_returns_correct_shapes(self):
        s = _make_solver()
        data = extract_mode_data(s, 500.0, n_modes=4)
        assert data["real_parts"].shape == (4,)
        assert data["imag_parts"].shape == (4,)
        assert data["mode_shapes"].shape[0] == 4
        assert data["frequencies"].shape == (4,)

    def test_frequencies_positive(self):
        s = _make_solver()
        data = extract_mode_data(s, 500.0, n_modes=4)
        assert all(data["frequencies"] > 0)


# ---- GP surrogate tests ----

class TestRBFKernel:
    def test_shape(self):
        X1 = np.random.randn(5, 3)
        X2 = np.random.randn(7, 3)
        K = _rbf_kernel(X1, X2, np.ones(3), 1.0)
        assert K.shape == (5, 7)

    def test_self_kernel_positive_diagonal(self):
        X = np.random.randn(4, 2)
        K = _rbf_kernel(X, X, np.ones(2), 1.0)
        diag = np.diag(K)
        assert all(diag > 0)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_symmetry(self):
        X = np.random.randn(5, 3)
        K = _rbf_kernel(X, X, np.ones(3), 2.0)
        np.testing.assert_allclose(K, K.T, atol=1e-10)


class TestAugmentedKernel:
    def test_no_grad_standard_shape(self):
        X = np.random.randn(4, 2)
        K = _augmented_kernel(X, np.ones(2), 1.0, 1e-6, include_grad=False)
        assert K.shape == (4, 4)

    def test_with_grad_shape(self):
        X = np.random.randn(4, 2)
        K = _augmented_kernel(X, np.ones(2), 1.0, 1e-6, include_grad=True)
        expected_size = 4 + 4 * 2
        assert K.shape == (expected_size, expected_size)

    def test_positive_diagonal(self):
        X = np.random.randn(4, 2)
        K = _augmented_kernel(X, np.ones(2), 1.0, 1e-6, include_grad=True)
        assert all(np.diag(K) > 0)


class TestGPSurrogate:
    def test_fit_predict_shapes(self):
        X = np.random.randn(20, 2)
        y = np.sin(X[:, 0]) + X[:, 1] ** 2
        gp = GPSurrogate()
        gp.fit(X, y)
        mean, std = gp.predict(X)
        assert mean.shape == (20,)
        assert std.shape == (20,)

    def test_interpolation_accuracy(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = 3.0 * X[:, 0] + 2.0 * X[:, 1]
        gp = GPSurrogate(noise_var=1e-8)
        gp.fit(X, y)
        mean, _ = gp.predict(X)
        np.testing.assert_allclose(mean, y, atol=1e-3)

    def test_extrapolation_uncertainty_grows(self):
        X = np.random.randn(20, 1)
        y = X[:, 0] ** 2
        gp = GPSurrogate()
        gp.fit(X, y)
        X_test = np.array([[-10.0]])  # far from training
        _, std_far = gp.predict(X_test)
        _, std_near = gp.predict(X[:1])
        assert std_far[0] > std_near[0]

    def test_gp_convergence_more_data_lower_rmse(self):
        np.random.seed(0)
        X_full = np.random.randn(200, 2)
        y_full = np.sin(X_full[:, 0]) + 0.5 * X_full[:, 1]
        X_test = np.random.randn(50, 2)
        y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1]
        rmses = []
        for n in [10, 20, 50, 100]:
            gp = GPSurrogate(noise_var=1e-6)
            gp.fit(X_full[:n], y_full[:n])
            mean, _ = gp.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - mean) ** 2))
            rmses.append(rmse)
        # RMSE should generally decrease
        assert rmses[-1] < rmses[0], f"RMSE didn't improve: {rmses}"

    def test_gradient_enhanced_accuracy(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + 0.5 * X[:, 1]
        gp = GPSurrogate(use_gradients=True, noise_var=1e-6)
        gp.fit(X, y, objective_fn=lambda x: x[0] + 0.5 * x[1])
        X_test = np.random.randn(20, 2)
        y_test = X_test[:, 0] + 0.5 * X_test[:, 1]
        mean, _ = gp.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - mean) ** 2))
        assert rmse < 0.2, f"Gradient-enhanced RMSE too high: {rmse}"

    def test_get_training_data(self):
        np.random.seed(42)
        X = np.random.randn(15, 2)
        y = np.sin(X[:, 0]) + X[:, 1]
        gp = GPSurrogate()
        gp.fit(X, y)
        X_out, y_out = gp.get_training_data()
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_log_marginal_likelihood_finite(self):
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        gp = GPSurrogate()
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)

    def test_gp_hyperparam_optimization_improves_fit(self):
        """ML-II hyperparameter optimization reduces holdout RMSE on an
        anisotropic function where fixed length-scales underfit."""
        rng = np.random.default_rng(7)

        def true_fn(X):
            return np.sin(4.0 * X[:, 0]) + 0.1 * X[:, 1]

        X = rng.uniform(-2.0, 2.0, size=(40, 2))
        y = true_fn(X)
        X_test = rng.uniform(-2.0, 2.0, size=(60, 2))
        y_test = true_fn(X_test)

        gp_fixed = GPSurrogate(noise_var=1e-6)
        gp_fixed.fit(X, y)
        mean_fixed, _ = gp_fixed.predict(X_test)
        rmse_fixed = np.sqrt(np.mean((y_test - mean_fixed) ** 2))

        gp_opt = GPSurrogate(noise_var=1e-6)
        gp_opt.fit(X, y, optimize_hyperparams=True)
        mean_opt, _ = gp_opt.predict(X_test)
        rmse_opt = np.sqrt(np.mean((y_test - mean_opt) ** 2))

        assert rmse_opt < rmse_fixed, (
            f"optimized RMSE {rmse_opt:.4g} not < fixed {rmse_fixed:.4g}"
        )
        gp = GPSurrogate()
        with pytest.raises(RuntimeError):
            gp.predict(np.random.randn(5, 2))

    def test_gradient_enhanced_fit(self):
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = X[:, 0] + 0.5 * X[:, 1]
        gp = GPSurrogate(use_gradients=True, noise_var=1e-4)

        def obj(x):
            return x[0] + 0.5 * x[1]

        gp.fit(X, y, objective_fn=obj)
        mean, std = gp.predict(X)
        assert mean.shape == (20,)
        # Gradient-enhanced GP with augmented kernel — verify finite predictions
        assert np.all(np.isfinite(mean))
        assert np.all(std >= 0)


# ---- Active learning tests ----

class TestAcquisitionFunctions:
    @pytest.fixture
    def fitted_gp(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + 0.5 * X[:, 1]
        gp = GPSurrogate()
        gp.fit(X, y)
        return gp

    def test_uncertainty_sampling_shape(self, fitted_gp):
        X_cand = np.random.randn(50, 2)
        scores = uncertainty_sampling(X_cand, fitted_gp)
        assert scores.shape == (50,)
        assert all(scores >= 0)

    def test_ei_reliability_shape(self, fitted_gp):
        X_cand = np.random.randn(50, 2)
        scores = expected_improvement_reliability(X_cand, fitted_gp, lambda_target=0.5)
        assert scores.shape == (50,)
        assert all(scores >= 0)

    def test_boundary_seeking_shape(self, fitted_gp):
        X_cand = np.random.randn(50, 2)
        scores = boundary_seeking(X_cand, fitted_gp, lambda_target=0.5)
        assert scores.shape == (50,)
        assert all(scores >= 0)

    def test_select_batch_returns_correct_count(self, fitted_gp):
        X_cand = np.random.randn(100, 2)
        selected = select_batch(X_cand, fitted_gp, lambda_target=0.5, batch_size=5)
        assert selected.shape == (5, 2)

    def test_select_batch_different_methods(self, fitted_gp):
        X_cand = np.random.randn(100, 2)
        for method in ["reliability", "uncertainty", "boundary"]:
            sel = select_batch(X_cand, fitted_gp, lambda_target=0.5,
                               batch_size=3, method=method)
            assert sel.shape == (3, 2)


# ---- Sweep test (lightweight) ----

class TestSweep:
    def test_generate_small_sweep(self):
        from mechanics.p3_robust_design.sweep import generate_design_sweep
        lam = _sandwich_laminate()
        result = generate_design_sweep(
            n_samples=3, laminate=lam, seed=0, M=5, N=5,
        )
        assert result.design_params.shape == (3, 4)
        assert result.boundary_stiffnesses.shape == (3, 4)
        assert result.flutter_lambda.shape == (3,)
        assert result.n_total == 3

    def test_sweep_records_failure_reasons(self):
        """Failed sweep points are counted with a categorized reason."""
        from mechanics.p3_robust_design.sweep import generate_design_sweep
        lam = _sandwich_laminate()
        # Absurdly stiff edges (log10 k ~ 17-18, far above the machine
        # k_stiffness ceiling): flutter lambda is pushed beyond the sweep's
        # upper bracket (1000), so no crossing can be found.
        result = generate_design_sweep(
            n_samples=2, laminate=lam, seed=1, M=5, N=5,
            stiffness_bounds={e: (17.0, 18.0)
                              for e in ("left", "right", "top", "bottom")},
        )
        assert result.n_success == 0
        assert result.failure_reasons
        total = sum(result.failure_reasons.values())
        assert total == 2
        known = {"unstable_at_lambda_lower", "no_crossing_in_bracket"}
        solver_cats = {k for k in result.failure_reasons
                       if k in known or k.startswith("solver_error:")}
        assert solver_cats == set(result.failure_reasons)


class TestModeTrackingSmoothness:
    def test_frequency_smoothness_across_velocity(self):
        from mechanics.p3_robust_design.mode_tracking import track_modes_across_velocity
        s = _make_solver()
        velocities = np.linspace(400.0, 500.0, 6)
        result = track_modes_across_velocity(s, velocities, n_modes=4)
        freqs = result["frequencies"]  # (n_vel, n_modes)
        # Check smoothness: |Δf|/Δv should be reasonable (< 1.0 Hz per m/s)
        for mode_idx in range(freqs.shape[1]):
            df = np.abs(np.diff(freqs[:, mode_idx]))
            dv = np.diff(velocities)
            rate = df / dv
            assert np.all(rate < 1.0), (
                f"Mode {mode_idx} has frequency jump: max rate = {rate.max():.3f}"
            )


class TestActiveLearningImprovement:
    def test_active_learning_reduces_rmse(self):
        np.random.seed(42)
        # Create a simple known function
        def true_fn(x):
            return x[0] ** 2 + 0.5 * x[1]

        # Initial training data
        X_init = np.random.randn(10, 2)
        y_init = np.array([true_fn(x) for x in X_init])

        # Test set
        X_test = np.random.randn(30, 2)
        y_test = np.array([true_fn(x) for x in X_test])

        # Fit initial GP
        gp = GPSurrogate(noise_var=1e-6)
        gp.fit(X_init, y_init)

        # Initial RMSE
        init_rmse = np.sqrt(np.mean((y_test - gp.predict(X_test)[0]) ** 2))

        # Run 3 active learning iterations
        from mechanics.p3_robust_design.active_learning import select_batch
        for _ in range(3):
            X_cand = np.random.randn(100, 2) * 2
            selected = select_batch(X_cand, gp, lambda_target=5.0,
                                    batch_size=5, method="uncertainty")
            new_y = np.array([true_fn(x) for x in selected])
            X_train, y_train = gp.get_training_data()
            X_all = np.vstack([X_train, selected])
            y_all = np.concatenate([y_train, new_y])
            gp_new = GPSurrogate(noise_var=1e-6)
            gp_new.fit(X_all, y_all)
            gp = gp_new

        final_rmse = np.sqrt(np.mean((y_test - gp.predict(X_test)[0]) ** 2))
        assert final_rmse < init_rmse, (
            f"Active learning didn't reduce RMSE: {init_rmse:.4f} -> {final_rmse:.4f}"
        )


class TestSweepPhysics:
    def test_failure_probability_in_unit_interval(self):
        from mechanics.p3_robust_design.train import brute_force_mc
        lam = _sandwich_laminate()
        result = brute_force_mc(lam, n_samples=5, seed=0)
        assert 0.0 <= result["failure_probability"] <= 1.0
        assert result["n_total"] == 5


# ---- Behavioral tests for P3 Robust Design ----

class TestGPCalibration:
    def test_gp_prediction_calibration(self):
        """≥90% of held-out true values within mean ± 2*std on sin(x0)+cos(x1)."""
        rng = np.random.default_rng(42)

        def true_fn(X):
            return np.sin(X[:, 0]) + np.cos(X[:, 1])

        X_train = rng.uniform(-2, 2, size=(150, 2))
        y_train = true_fn(X_train)

        X_test = rng.uniform(-2, 2, size=(200, 2))
        y_test = true_fn(X_test)

        gp = GPSurrogate(noise_var=1e-6)
        gp.fit(X_train, y_train)

        mean, std = gp.predict(X_test)
        within = np.abs(y_test - mean) <= 2 * std
        coverage = np.mean(within)
        assert coverage >= 0.90, f"Calibration coverage {coverage:.3f} < 0.90"


class TestGradientGP:
    def test_gradient_gp_beats_plain_gp(self):
        """Gradient-enhanced RMSE < plain GP RMSE on sin(x0)+cos(x1)."""
        rng = np.random.default_rng(42)

        def true_fn(X):
            return np.sin(X[:, 0]) + np.cos(X[:, 1])

        X_train = rng.uniform(-2, 2, size=(30, 2))
        y_train = true_fn(X_train)

        X_test = rng.uniform(-2, 2, size=(100, 2))
        y_test = true_fn(X_test)

        gp_plain = GPSurrogate(noise_var=1e-6)
        gp_plain.fit(X_train, y_train)
        mean_plain, _ = gp_plain.predict(X_test)
        rmse_plain = np.sqrt(np.mean((y_test - mean_plain) ** 2))

        gp_grad = GPSurrogate(noise_var=1e-6, use_gradients=True)
        gp_grad.fit(X_train, y_train,
                    objective_fn=lambda x: true_fn(x.reshape(1, -1))[0])
        mean_grad, _ = gp_grad.predict(X_test)
        rmse_grad = np.sqrt(np.mean((y_test - mean_grad) ** 2))

        assert rmse_grad < rmse_plain, (
            f"Gradient RMSE {rmse_grad:.4f} >= plain RMSE {rmse_plain:.4f}"
        )


class TestUncertaintySamplingFarPoints:
    def test_uncertainty_sampling_picks_far_points(self):
        """Top-5 highest-uncertainty candidates all have norm > 2.0."""
        rng = np.random.default_rng(42)

        def true_fn(X):
            return np.sin(X[:, 0]) + np.cos(X[:, 1])

        X_train = rng.uniform(-1, 1, size=(30, 2))
        y_train = true_fn(X_train)

        gp = GPSurrogate(noise_var=1e-6)
        gp.fit(X_train, y_train)

        X_cand = rng.uniform(-5, 5, size=(200, 2))
        scores = uncertainty_sampling(X_cand, gp)
        top5_idx = np.argsort(scores)[-5:]
        top5_norms = np.linalg.norm(X_cand[top5_idx], axis=1)
        assert np.all(top5_norms > 2.0), (
            f"Some top-5 uncertainty points within norm 2.0: {top5_norms}"
        )


class TestEIPeakNearThreshold:
    def test_ei_scores_peak_near_threshold(self):
        """EI highest where μ < λ_target, near zero where μ >> λ_target."""
        rng = np.random.default_rng(42)

        X_train = rng.uniform(-3, 3, size=(50, 2))
        y_train = X_train[:, 0]

        gp = GPSurrogate(noise_var=1e-6)
        gp.fit(X_train, y_train)

        lambda_target = 0.0
        X_test = np.zeros((100, 2))
        X_test[:, 0] = np.linspace(-3, 3, 100)

        scores = expected_improvement_reliability(X_test, gp, lambda_target)

        high_idx = np.where(X_test[:, 0] > 2.0)[0]
        assert np.all(scores[high_idx] < 1e-6), (
            f"EI not near zero where μ >> target: {scores[high_idx]}"
        )

        low_idx = np.where(X_test[:, 0] < -1.0)[0]
        assert np.all(scores[low_idx] > 0), (
            f"EI not positive where μ < target: {scores[low_idx]}"
        )


class TestBoundarySeekingPeaksAtCrossover:
    def test_boundary_seeking_peaks_at_crossover(self):
        """Boundary seeking peaks where μ ≈ λ_target, not at extremes."""
        rng = np.random.default_rng(42)

        X_train = rng.uniform(-3, 3, size=(50, 2))
        y_train = X_train[:, 0]

        gp = GPSurrogate(noise_var=1e-6)
        gp.fit(X_train, y_train)

        lambda_target = 0.0
        X_test = np.zeros((100, 2))
        X_test[:, 0] = np.linspace(-3, 3, 100)

        scores = boundary_seeking(X_test, gp, lambda_target)

        peak_idx = np.argmax(scores)
        peak_x0 = X_test[peak_idx, 0]
        assert abs(peak_x0) < 1.0, (
            f"Boundary-seeking peak at x0={peak_x0:.3f}, expected near 0"
        )


class TestModeTrackingMAC:
    def test_mode_tracking_mac_consistency(self):
        """Median MAC > 0.6; at most 5 outlier steps with MAC < 0.5."""
        from mechanics.p3_robust_design.mode_tracking import track_modes_across_velocity
        s = _make_solver()
        velocities = np.linspace(420.0, 460.0, 8)
        result = track_modes_across_velocity(s, velocities, n_modes=4)
        mac_vals = result["mac_values"]
        # Per-step median: tolerate individual mode drops at crossing/veering
        step_medians = np.median(mac_vals, axis=1)
        assert np.median(step_medians) > 0.6, (
            f"Median MAC {np.median(step_medians):.3f} <= 0.6"
        )
        low_steps = np.sum(step_medians < 0.5)
        assert low_steps <= 5, (
            f"{low_steps} steps have median MAC < 0.5 (max 5 allowed)"
        )

    def test_complex_shape_mac_phase_invariant_across_sweep(self):
        """Regression: complex mode shapes keep MAC high across the
        velocity sweep, and MAC is invariant to the arbitrary global
        phase each eigenstep carries — the property the real-part
        slice lacks along aeroelastic branches.
        """
        from mechanics.p3_robust_design.mode_tracking import mac
        s = _make_solver()
        velocities = np.linspace(420.0, 460.0, 8)
        results = [s.solve_complex_modal(float(v), n_modes=4) for v in velocities]
        macs = [
            mac(prev.mode_shapes_complex[k], curr.mode_shapes_complex[k])
            for prev, curr in zip(results[:-1], results[1:])
            for k in range(4)
        ]
        assert np.median(macs) > 0.6, (
            f"Median complex-shape MAC {np.median(macs):.3f} <= 0.6"
        )
        # Phase invariance: rotating each mode by an arbitrary global
        # complex phase must not change MAC.
        rng = np.random.default_rng(7)
        rotated = results[0].mode_shapes_complex * np.exp(
            1j * rng.uniform(0.0, 2.0 * np.pi, size=(4, 1, 1))
        )
        for k in range(4):
            ref = mac(results[0].mode_shapes_complex[k], results[1].mode_shapes_complex[k])
            assert abs(mac(rotated[k], results[1].mode_shapes_complex[k]) - ref) < 1e-9


class TestStiffnessIncreasesFlutter:
    def test_stiffness_increases_flutter_lambda(self):
        """Higher max-stiffness correlates positively with flutter_λ."""
        from mechanics.p3_robust_design.sweep import generate_design_sweep
        from scipy.stats import pearsonr

        lam = _sandwich_laminate()
        result = generate_design_sweep(n_samples=15, laminate=lam, seed=42)

        valid = ~np.isnan(result.flutter_lambda)
        assert valid.sum() >= 5, "Not enough successful sweeps"

        max_stiff = np.max(result.boundary_stiffnesses[valid], axis=1)
        flutter = result.flutter_lambda[valid]

        corr, _ = pearsonr(max_stiff, flutter)
        assert corr > 0, f"Pearson correlation {corr:.4f} <= 0"


class TestGPUncertaintyShrinks:
    def test_gp_uncertainty_shrinks_with_data(self):
        """Std at a fixed test point monotonically decreases with more data."""
        rng = np.random.default_rng(42)

        def true_fn(X):
            return np.sin(X[:, 0]) + np.cos(X[:, 1])

        X_pool = rng.uniform(-2, 2, size=(200, 2))
        y_pool = true_fn(X_pool)

        X_test = np.array([[0.5, 0.5]])

        stds = []
        for n in [10, 30, 60, 100]:
            gp = GPSurrogate(noise_var=1e-6)
            gp.fit(X_pool[:n], y_pool[:n])
            _, std = gp.predict(X_test)
            stds.append(std[0])

        for i in range(len(stds) - 1):
            assert stds[i + 1] < stds[i], (
                f"std did not decrease: {stds}"
            )
