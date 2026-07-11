"""Reliability-oriented active learning with acquisition functions.

Selects new sample points that improve surrogate accuracy near the
failure boundary (λ_cr ≈ threshold).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

from .gp_surrogate import GPSurrogate


def uncertainty_sampling(X_candidates: np.ndarray, gp: GPSurrogate) -> np.ndarray:
    """Acquisition: predictive uncertainty σ(x).

    Higher σ = more informative.
    """
    _, std = gp.predict(X_candidates)
    return std


def expected_improvement_reliability(
    X_candidates: np.ndarray,
    gp: GPSurrogate,
    lambda_target: float,
) -> np.ndarray:
    """Acquisition: reliability-oriented expected improvement.

    EI_rel(x) = P(λ_cr(x) < λ_target) * (λ_target - μ(x))
    when μ(x) < λ_target, else 0.

    High EI_rel means: likely to be near failure boundary and informative.
    """
    mu, std = gp.predict(X_candidates)
    std = np.maximum(std, 1e-10)

    # Probability of being below threshold
    z = (lambda_target - mu) / std
    prob_below = norm.cdf(z)

    # Improvement: how far below threshold (positive when below)
    improvement = np.maximum(lambda_target - mu, 0.0)

    return prob_below * improvement


def boundary_seeking(
    X_candidates: np.ndarray,
    gp: GPSurrogate,
    lambda_target: float,
) -> np.ndarray:
    """Acquisition: boundary seeking — μ(x) ≈ λ_target with high σ.

    Combines proximity to threshold with uncertainty.
    """
    mu, std = gp.predict(X_candidates)
    # Distance from threshold (Gaussian-like penalty)
    proximity = np.exp(-0.5 * ((mu - lambda_target) / (std + 1e-10)) ** 2)
    return proximity * std


def select_batch(
    X_candidates: np.ndarray,
    gp: GPSurrogate,
    lambda_target: float,
    batch_size: int = 5,
    method: str = "reliability",
    min_dist: float = 0.0,
) -> np.ndarray:
    """Select batch of points for active learning.

    Args:
        X_candidates: (n_cand, d) candidate points.
        gp: Fitted GP surrogate.
        lambda_target: Safety threshold for λ_cr.
        batch_size: Number of points to select.
        method: 'reliability', 'uncertainty', or 'boundary'.
        min_dist: Minimum Euclidean distance between selected points.

    Returns:
        (batch_size, d) selected points.
    """
    if method == "reliability":
        scores = expected_improvement_reliability(X_candidates, gp, lambda_target)
    elif method == "uncertainty":
        scores = uncertainty_sampling(X_candidates, gp)
    elif method == "boundary":
        scores = boundary_seeking(X_candidates, gp, lambda_target)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Greedy selection with minimum-distance dedup
    ranked = np.argsort(scores)[::-1]
    selected = []
    for idx in ranked:
        if len(selected) >= batch_size:
            break
        if min_dist > 0 and selected:
            dists = np.linalg.norm(X_candidates[selected] - X_candidates[idx], axis=1)
            if np.any(dists < min_dist):
                continue
        selected.append(idx)
    return X_candidates[selected]
