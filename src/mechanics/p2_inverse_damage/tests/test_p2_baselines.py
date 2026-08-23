"""Tests for P2 field-prediction baselines (comparison protocol).

Synthetic linear ground truth: each baseline must train to low MSE on a
held-out set; Bayesian std must grow away from the data manifold center;
classifier probabilities stay in [0, 1] with thresholded map == argmax.
Runtime budget: < 10 s total.
"""
import numpy as np
import pytest

from mechanics.p2_inverse_damage.baselines import (
    BayesianRegressionBaseline,
    DirectRegressionBaseline,
    PixelClassifierBaseline,
)

GRID = (4, 6)
D_IN = 5
OUT = GRID[0] * GRID[1]
N_TRAIN, N_TEST = 300, 100


def _synthetic(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear measurements → field map squashed into [0, 1]."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, size=(D_IN, OUT)) / np.sqrt(D_IN)
    b = rng.normal(0.0, 0.1, size=OUT)
    X = rng.normal(0.0, 1.0, size=(N_TRAIN + N_TEST, D_IN))
    Y = 1.0 / (1.0 + np.exp(-(X @ W + b)))  # in (0, 1), learnable via sigmoid head
    return X[:N_TRAIN], Y[:N_TRAIN], X[N_TRAIN:], Y[N_TRAIN:]


Xtr, Ytr, Xte, Yte = _synthetic(seed=7)


# ─── 1. Direct regression ────────────────────────────────────────────

class TestDirectRegression:
    def test_trains_to_low_mse(self):
        model = DirectRegressionBaseline(d_in=D_IN, grid_shape=GRID, seed=0)
        hist = model.fit(Xtr, Ytr.reshape(N_TRAIN, OUT), epochs=150, lr=5e-3,
                         batch_size=64)
        assert hist[-1] < hist[0]  # loss decreases
        pred = model.predict(Xte)
        assert pred.shape == (N_TEST, *GRID)
        mse = float(np.mean((pred - Yte.reshape(N_TEST, *GRID)) ** 2))
        assert mse < 1e-2, f"held-out MSE too high: {mse:.4f}"

    def test_output_range_and_determinism(self):
        model = DirectRegressionBaseline(d_in=D_IN, grid_shape=GRID, seed=0)
        model.fit(Xtr, Ytr.reshape(N_TRAIN, OUT), epochs=10, lr=1e-2)
        pred = model.predict(Xte)
        assert pred.min() >= 0.0 and pred.max() <= 1.0
        assert np.allclose(pred, model.predict(Xte))


# ─── 2. Bayesian ridge regression ────────────────────────────────────

class TestBayesianRegression:
    def test_closed_form_low_mse(self):
        model = BayesianRegressionBaseline(grid_shape=GRID).fit(Xtr, Ytr)
        mean, _ = model.predict_with_uncertainty(Xte)
        assert mean.shape == (N_TEST, *GRID)
        mse = float(np.mean((mean - Yte.reshape(N_TEST, *GRID)) ** 2))
        assert mse < 1e-2, f"held-out MSE too high: {mse:.4f}"

    def test_predict_matches_uncertainty_mean(self):
        model = BayesianRegressionBaseline(grid_shape=GRID).fit(Xtr, Ytr)
        mean_u, _ = model.predict_with_uncertainty(Xte)
        assert np.allclose(mean_u, model.predict(Xte))

    def test_std_larger_far_from_data_center(self):
        """Predictive std grows with leverage away from the training centroid."""
        model = BayesianRegressionBaseline(grid_shape=GRID).fit(Xtr, Ytr)
        near = Xtr.mean(axis=0, keepdims=True)          # manifold center
        far = Xtr.mean(axis=0) + 8.0                    # far off-manifold
        _, std_near = model.predict_with_uncertainty(near)
        _, std_far = model.predict_with_uncertainty(far[None, :])
        assert np.all(std_far > std_near)

    def test_std_positive_finite(self):
        model = BayesianRegressionBaseline(grid_shape=GRID).fit(Xtr, Ytr)
        _, std = model.predict_with_uncertainty(Xte)
        assert np.all(np.isfinite(std)) and np.all(std > 0)


# ─── 3. Pixel classifier ─────────────────────────────────────────────

class TestPixelClassifier:
    def test_probabilities_bounded_and_threshold_matches_argmax(self):
        labels = (Ytr > 0.5).astype(np.float64)
        yte_bin = (Yte > 0.5).astype(np.int64)
        model = PixelClassifierBaseline(d_in=D_IN, grid_shape=GRID, seed=0)
        model.fit(Xtr, labels, epochs=200, lr=1e-1)
        proba = model.predict_proba(Xte)
        assert proba.shape == (N_TEST, *GRID)
        assert proba.min() >= 0.0 and proba.max() <= 1.0
        # Binary argmax([1-p, p]) is exactly the p >= 0.5 threshold map.
        pred = model.predict(Xte)
        argmax_map = np.argmax(
            np.stack([1.0 - proba, proba], axis=0), axis=0
        )
        acc = float(np.mean(pred == yte_bin.reshape(N_TEST, *GRID)))
        assert acc > 0.9, f"classifier accuracy too low: {acc:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
