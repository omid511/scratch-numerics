"""Gradient correctness for the jointly trained residual MLP objective."""
import numpy as np
import pytest
from scipy.optimize import check_grad

from mechanics.p4_margin_estimation.skip_mlp import objective_setup


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('consistency_weight', [0., 1.])
@pytest.mark.parametrize('pair_weights', [None, np.array([1., 0., 3., 2.])])
def test_regularized_gradient_matches_finite_differences(skip, consistency_weight, pair_weights):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    # Repeated endpoints require accumulating, rather than overwriting, gradients.
    pairs = np.array([[0, 4], [0, 7], [4, 9], [2, 2]])
    theta, objective, predict = objective_setup(
        x, y, (4, 3), 42, .7, skip,
        consistency_pairs=pairs, consistency_weight=consistency_weight,
        consistency_pair_weights=pair_weights)
    if skip:
        # Nonzero skip weights exercise both the residual and its L2 gradient.
        theta[-3:] = [.2, -.1, .3]
    error = check_grad(lambda t: objective(t)[0], lambda t: objective(t)[1], theta)
    assert error < 1e-5


def test_consistency_penalizes_prediction_gap_not_target_residual_gap():
    rng = np.random.default_rng(72)
    x = rng.normal(size=(12, 3))
    y = rng.normal(size=12)
    pairs = np.array([[0, 4], [0, 7], [3, 9]])
    theta, plain, predict = objective_setup(x, y, (4, 3), 42, .7, False)
    _, paired, _ = objective_setup(
        x, y, (4, 3), 42, .7, False,
        consistency_pairs=pairs, consistency_weight=.6)
    predictions = predict(theta, x)
    expected = .3 * np.mean((predictions[pairs[:, 0]] - predictions[pairs[:, 1]]) ** 2)
    np.testing.assert_allclose(paired(theta)[0] - plain(theta)[0], expected, atol=1e-14)


@pytest.mark.parametrize('pairs, weight', [
    (None, 1.), (np.array([[0, 8]]), 1.),
    (np.array([[0., .5]]), 1.), (np.array([[0, 1]]), -1.),
])
def test_invalid_consistency_constraints_are_rejected(pairs, weight):
    with pytest.raises(ValueError):
        objective_setup(np.ones((8, 3)), np.zeros(8), (4, 3), 42, 1., False,
                        consistency_pairs=pairs, consistency_weight=weight)


def test_pair_weights_localize_penalty_without_changing_supervised_loss():
    rng = np.random.default_rng(73)
    x, y = rng.normal(size=(12, 3)), rng.normal(size=12)
    pairs = np.array([[0, 4], [0, 7], [3, 9]])
    theta, plain, predict = objective_setup(x, y, (4, 3), 42, .7, False)
    weights = np.array([1., 0., 3.])
    _, localized, _ = objective_setup(
        x, y, (4, 3), 42, .7, False, consistency_pairs=pairs,
        consistency_weight=.6, consistency_pair_weights=weights)
    _, rescaled, _ = objective_setup(
        x, y, (4, 3), 42, .7, False, consistency_pairs=pairs,
        consistency_weight=.6, consistency_pair_weights=10*weights)
    p = predict(theta, x)
    expected = .3 * np.average((p[pairs[:, 0]]-p[pairs[:, 1]])**2, weights=weights)
    np.testing.assert_allclose(localized(theta)[0]-plain(theta)[0], expected, atol=1e-14)
    np.testing.assert_allclose(localized(theta)[0], rescaled(theta)[0], atol=1e-14)
    np.testing.assert_allclose(localized(theta)[1], rescaled(theta)[1], atol=1e-14)
    # A zero-weight pair must have no effect, including its gradient endpoints.
    _, omitted, _ = objective_setup(
        x, y, (4, 3), 42, .7, False, consistency_pairs=pairs[[0, 2]],
        consistency_weight=.6, consistency_pair_weights=weights[[0, 2]])
    np.testing.assert_allclose(localized(theta)[0], omitted(theta)[0], atol=1e-14)
    np.testing.assert_allclose(localized(theta)[1], omitted(theta)[1], atol=1e-14)


@pytest.mark.parametrize('weights', [np.zeros(2), np.array([1., -1.]),
                                   np.array([1., np.nan]), np.ones(3)])
def test_invalid_pair_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        objective_setup(np.ones((8, 3)), np.zeros(8), (4, 3), 42, 1., False,
                        consistency_pairs=np.array([[0, 1], [2, 3]]),
                        consistency_weight=1., consistency_pair_weights=weights)
