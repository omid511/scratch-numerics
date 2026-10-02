"""Identification checks with known dynamics, including spatial cancellation."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechanics.p4_margin_estimation.modal_features import estimate_modes, extract_modal_features


def test_recovers_growth_and_decay_despite_spatial_cancellation_and_offsets():
    t = np.arange(512) * 0.001
    signal = (np.exp(-2 * t) * np.cos(2 * np.pi * 31 * t)
              + 0.4 * np.exp(3 * t) * np.cos(2 * np.pi * 73 * t + 0.7))
    x = np.array([signal, -signal, 0.5 * signal + 0.2, -0.5 * signal - 0.2])
    result = estimate_modes(x, 0.001)
    order = np.argsort(result["frequency"])
    np.testing.assert_allclose(result["frequency"][order], [31, 73], atol=1e-5)
    np.testing.assert_allclose(result["alpha"][order], [-2, 3], atol=1e-5)
    assert result["relative_fit_error"] < 1e-12
    rescaled = estimate_modes(x, 0.002)
    np.testing.assert_allclose(rescaled["frequency"], result["frequency"] / 2, atol=1e-5)
    np.testing.assert_allclose(rescaled["alpha"], result["alpha"] / 2, atol=1e-5)
    noisy = estimate_modes(x + np.random.default_rng(3).normal(0, 1e-4, x.shape), 0.001)
    order = np.argsort(noisy["frequency"][:2])
    np.testing.assert_allclose(noisy["frequency"][:2][order], [31, 73], atol=0.01)
    np.testing.assert_allclose(noisy["alpha"][:2][order], [-2, 3], atol=0.02)


def test_saturation_uses_unclipped_prefix_without_fitting_flattened_peaks():
    t = np.arange(512) * 0.001
    x = np.clip(np.exp(12 * t) * np.cos(2 * np.pi * 40 * t), -50, 50)[None, :]
    result = estimate_modes(x, 0.001)
    np.testing.assert_allclose(result["alpha"], [12], atol=1e-5)
    np.testing.assert_allclose(result["frequency"], [40], atol=1e-5)
    assert 0.5 < result["usable_fraction"] < 0.8


@pytest.mark.parametrize("signals", [np.ones((2, 512)), np.zeros((2, 512)),
                                    np.full((2, 512), 50.0)])
def test_unobservable_clips_report_missing_instead_of_a_physical_estimate(signals):
    row = extract_modal_features(signals, 0.001)
    assert row[14] == 0  # unavailable
    assert row[11] == 0  # no modes
    assert row[12] == 1  # reconstruction unavailable
    assert np.all(row[:11] == 0)


@pytest.mark.parametrize("dt", [None, 0, -1, float("nan")])
def test_physical_timebase_is_required(dt):
    with pytest.raises(ValueError, match="positive physical dt"):
        estimate_modes(np.ones((2, 512)), dt)


def test_nested_selection_never_uses_outer_held_out_labels():
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from experiment_p4_ridge_cv import design_folds, nested_cv

    rng = np.random.default_rng(17)
    ids = np.repeat([f"D{i:02}" for i in range(15)], 4)
    x = rng.normal(size=(len(ids), 3))
    y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(0, 0.2, len(ids))
    held_out = design_folds(ids, 5, 42)[0]
    original = nested_cv(x, y, ids)
    changed_y = y.copy()
    changed_y[held_out] += 100
    changed = nested_cv(x, changed_y, ids)
    np.testing.assert_array_equal(
        np.asarray(original["oof_predictions"])[held_out],
        np.asarray(changed["oof_predictions"])[held_out])
    assert original["folds"][0]["alpha"] == changed["folds"][0]["alpha"]
