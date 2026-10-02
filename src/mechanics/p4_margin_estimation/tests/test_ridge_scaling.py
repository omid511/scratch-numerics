"""Ridge must not learn extrapolation from numerical feature variation."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiment_p4_ridge_cv import ridge_predictions


def test_numerical_feature_variation_cannot_amplify_query_shift():
    rng = np.random.default_rng(24)
    x = np.column_stack([rng.normal(size=100), 1e-12 * rng.normal(size=100)])
    y = .2 * x[:, 0] + .1 * x[:, 1] / 1e-12
    query = np.array([[0., 0.], [0., .5]])
    predicted = ridge_predictions(x, y, query, [1.])[:, 0]
    np.testing.assert_allclose(predicted[0], predicted[1], atol=1e-12, rtol=0)
    expected = ridge_predictions(x[:, :1], y, query[:, :1], [1.])[:, 0]
    np.testing.assert_allclose(predicted, expected, atol=1e-12, rtol=0)


def test_all_constant_training_features_predict_training_mean():
    x = np.full((10, 3), 2.)
    y = np.linspace(-.1, .3, 10)
    prediction = ridge_predictions(x, y, np.array([[100., -20., 1.]]), [.01, 10.])
    np.testing.assert_allclose(prediction, y.mean(), atol=1e-12)


def test_supported_features_keep_signal_and_query_batch_independence():
    x = np.linspace(-1., 1., 100)[:, None]
    y = .3 * x[:, 0] - .1
    query = np.array([[-.5], [.5]])
    prediction = ridge_predictions(x, y, query, [1e-6])[:, 0]
    np.testing.assert_allclose(prediction, [-.25, .05], atol=1e-7)
    extended = ridge_predictions(x, y, np.vstack([query, [[1e8]]]), [1e-6])[:2, 0]
    np.testing.assert_allclose(prediction, extended, atol=1e-12)
