"""Tests for Sobol sensitivity analysis."""
import numpy as np
import pytest
import sys
sys.path.insert(0, "/home/omid5/mechanics")

from data_generation import (
    sobol_sample,
    compute_sobol_indices,
)


def _ishigami(x: np.ndarray) -> np.ndarray:
    """Ishigami function: f(x) = sin(x1) + 7*sin^2(x2) + 0.1*x3^4*sin(x1).
    Known analytical Sobol indices:
        S1 = [0.3138, 0.4424, 0.0] (approx)
        ST = [0.5576, 0.4424, 0.2437] (approx)
    x in [-pi, pi]^3.
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)


def test_sobol_sample_shape():
    """Sobol sample matrix has correct dimensions."""
    n_base = 64
    D = 7
    samples = sobol_sample(n_base, D, seed=42)
    assert samples.shape == (n_base * (2 * D + 2), D)


def test_sobol_sample_range():
    """All samples in [0,1]."""
    samples = sobol_sample(32, 5, seed=42)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_sobol_sample_deterministic():
    """Same seed gives same samples."""
    a = sobol_sample(64, 3, seed=123)
    b = sobol_sample(64, 3, seed=123)
    np.testing.assert_array_equal(a, b)


def test_ishigami_s1_bounds():
    """S1 indices for Ishigami should be between 0 and 1."""
    n_base = 512
    D = 3
    samples = sobol_sample(n_base, D, seed=42)

    # Scale to [-pi, pi]
    samples_scaled = -np.pi + samples * 2 * np.pi
    Y = _ishigami(samples_scaled)[:, None]

    indices = compute_sobol_indices(Y, n_base, D, ["x1", "x2", "x3"])

    assert indices["S1"].shape == (D, 1)
    assert indices["ST"].shape == (D, 1)
    assert np.all(indices["S1"] >= -0.01)
    assert np.all(indices["S1"] <= 1.01)
    assert np.all(indices["ST"] >= -0.01)
    assert np.all(indices["ST"] <= 1.01)


def test_ishigami_indices_ranking():
    """Ishigami: x2 should dominate (ST ~0.44), x1 next (~0.56), x3 small."""
    n_base = 1024
    D = 3
    samples = sobol_sample(n_base, D, seed=42)
    samples_scaled = -np.pi + samples * 2 * np.pi
    Y = _ishigami(samples_scaled)[:, None]

    indices = compute_sobol_indices(Y, n_base, D, ["x1", "x2", "x3"])

    S1 = indices["S1"].flatten()
    ST = indices["ST"].flatten()

    # x3 should have S1 close to 0 (it only appears in higher-order interaction)
    assert S1[2] < 0.1, f"x3 S1 should be ~0, got {S1[2]}"

    # x2 should have large ST
    assert ST[1] > 0.3, f"x2 ST should be >0.3, got {ST[1]}"

    # x1 should have larger ST than x3
    assert ST[0] > ST[2], f"x1 ST ({ST[0]}) should exceed x3 ST ({ST[2]})"


def test_indices_sum_leq_one():
    """Sum of S1 indices should be <= 1 (theoretical bound)."""
    n_base = 4096
    D = 4
    samples = sobol_sample(n_base, D, seed=42)

    def quadratic(x):
        return (x[:, 0] + 0.5 * x[:, 1])[:, None]

    Y = quadratic(samples)
    indices = compute_sobol_indices(Y, n_base, D, ["a", "b", "c", "d"])

    sum_s1 = indices["S1"].sum(axis=0)
    assert np.all(sum_s1 <= 1.05), f"Sum S1 = {sum_s1}, should be <= 1"


def test_constant_output_gives_zero_indices():
    """Constant output should yield zero sensitivity."""
    n_base = 128
    D = 3
    samples = sobol_sample(n_base, D, seed=42)
    Y = np.ones((n_base * (2 * D + 2), 1)) * 42.0

    indices = compute_sobol_indices(Y, n_base, D, ["a", "b", "c"])

    # Variance is 0, so indices should be 0 (or near-0 from clamping)
    assert np.all(indices["S1"] < 0.01)
    assert np.all(indices["ST"] < 0.01)


def test_linear_model_indices():
    """Linear model f(x) = x1: S1[0] should be ~1, rest ~0."""
    n_base = 512
    D = 3
    samples = sobol_sample(n_base, D, seed=42)
    Y = samples[:, :1]  # f(x) = x1

    indices = compute_sobol_indices(Y, n_base, D, ["x1", "x2", "x3"])

    S1 = indices["S1"].flatten()
    assert S1[0] > 0.9, f"x1 S1 should be ~1, got {S1[0]}"
    assert S1[1] < 0.05, f"x2 S1 should be ~0, got {S1[1]}"
    assert S1[2] < 0.05, f"x3 S1 should be ~0, got {S1[2]}"
