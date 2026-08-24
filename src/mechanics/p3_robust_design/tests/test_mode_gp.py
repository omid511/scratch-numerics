"""Tests for the Phase-5 mode-decomposition surrogate (mode_gp.py)."""
import numpy as np
import pytest

from mechanics.laminate import Laminate, Material
from mechanics.p3_robust_design.mode_gp import ModeDecompositionGP
from mechanics.p3_robust_design.sweep import _make_solver


# ---- Toy 2-mode synthetic: two smooth functions crossing ----

def _crossing_data(n: int = 41, noise: float = 0.0):
    """Two linear branches crossing at x=0: f1 = 0.5 + x, f2 = 0.5 - x."""
    rng = np.random.default_rng(42)
    x = np.linspace(-1.0, 1.0, n)
    X = x[:, None]
    f1 = 0.5 + x
    f2 = 0.5 - x
    noise_mat = rng.normal(0, noise, (n, 2)) if noise > 0 else 0.0
    Y = np.column_stack([f1, f2]) + noise_mat
    return X, Y, f1, f2


def _fit_crossing(combination: str = "min", n: int = 41,
                  noise: float = 0.0, noise_var: float = 1e-8) -> tuple:
    X, Y, f1, f2 = _crossing_data(n, noise=noise)
    mdgp = ModeDecompositionGP(
        combination=combination,
        length_scales=np.array([0.25]),
        signal_var=1.0,
        noise_var=noise_var,
    )
    mdgp.fit(X, Y)
    return mdgp, X, f1, f2


class TestToyTwoMode:
    def test_combined_tracks_min(self):
        mdgp, X, f1, f2 = _fit_crossing("min")
        pred = mdgp.predict(X)
        true_min = np.minimum(f1, f2)
        assert np.allclose(pred.lam, true_min, atol=5e-2)

    def test_combined_tracks_max(self):
        mdgp, X, f1, f2 = _fit_crossing("max")
        pred = mdgp.predict(X)
        assert np.allclose(pred.lam, np.maximum(f1, f2), atol=5e-2)

    def test_mode_identity_switches_at_crossing(self):
        """Active mode is branch 2 left of the crossing and branch 1 right."""
        mdgp, X, _, _ = _fit_crossing()
        pred = mdgp.predict(X)
        x = X[:, 0]
        left_active = int(pred.active_mode[np.argmin(np.abs(x + 0.5))])
        right_active = int(pred.active_mode[np.argmin(np.abs(x - 0.5))])
        assert left_active != right_active

    def test_mode_probabilities_sum_to_one(self):
        mdgp, X, _, _ = _fit_crossing()
        pred = mdgp.predict(X)
        assert np.allclose(pred.mode_probabilities.sum(axis=1), 1.0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ModeDecompositionGP().predict(np.zeros((3, 1)))

    def test_invalid_combination_rejected(self):
        with pytest.raises(ValueError):
            ModeDecompositionGP(combination="mean")


class TestUncertaintyAtCrossing:
    def test_combined_std_finite_positive(self):
        """Combined std is finite and positive across the full range."""
        mdgp, X, _, _ = _fit_crossing(n=61, noise=0.02, noise_var=4e-4)
        pred = mdgp.predict(X)
        assert np.all(np.isfinite(pred.std))
        assert np.all(pred.std > 0)

    def test_mode_probabilities_valid_range(self):
        """Mode probabilities are in [0, 1] and sum to 1 everywhere."""
        mdgp, X, _, _ = _fit_crossing(n=61, noise=0.02, noise_var=4e-4)
        pred = mdgp.predict(X)
        p = pred.mode_probabilities
        assert np.all(p >= 0) and np.all(p <= 1)
        assert np.allclose(p.sum(axis=1), 1.0)

# ---- Integration: tracked modes from mode_tracking on a toy sweep ----

class TestModeTrackingIntegration:
    def _laminate(self) -> Laminate:
        E, nu, rho = 70e9, 0.33, 2700.0
        G = E / (2 * (1 + nu))
        return Laminate(
            [Material(E, E, G, G, G, nu, rho)] * 3,
            [0.0, 0.0, 0.0], [-0.006, -0.002, 0.002, 0.006],
        )
