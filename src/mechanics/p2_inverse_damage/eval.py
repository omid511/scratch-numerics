"""Evaluation metrics for P2 inverse damage identification.

Simulation-Based Calibration (SBC), interval coverage, CRPS, and MSE
helpers. NumPy-only; formulas documented per function.
"""
from __future__ import annotations
import math

import numpy as np


def sbc_ranks(
    posterior_samples_per_obs: list[np.ndarray],
    truth: np.ndarray,
) -> np.ndarray:
    """Rank of the true value within each observation's posterior samples.

    For observation i with posterior samples s_i (shape (n_samp,)) and
    truth y_i::

        rank_i = #{ j : s_i[j] < y_i }   ∈ [0, n_samp]

    Under a perfectly calibrated posterior the ranks are discrete-uniform
    on {0, ..., n_samp}. Ties count as "not below", matching standard SBC.
    """
    if len(posterior_samples_per_obs) != len(truth):
        raise ValueError("one sample array per truth entry required")
    ranks = np.empty(len(truth), dtype=int)
    for i, (samples, y) in enumerate(zip(posterior_samples_per_obs, truth)):
        samples = np.asarray(samples)
        ranks[i] = int(np.sum(samples < y))
    return ranks


def _chi2_sf(x: float, k: int) -> float:
    """Survival function P(X > x) of a chi-square variable with k dof.

    Computed via the regularized upper incomplete gamma Q(k/2, x/2) using
    the Numerical Recipes continued-fraction / series pair (no scipy).
    """
    if x <= 0.0:
        return 1.0
    a, half_x = k / 2.0, x / 2.0
    if half_x < a + 1.0:  # series for P(a, x); sf = 1 - P
        term = 1.0 / a
        total = term
        n = 1
        while True:
            term *= half_x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14 or n > 1000:
                break
            n += 1
        p = total * math.exp(-half_x + a * math.log(half_x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    # continued fraction for Q(a, x) (Lentz)
    tiny = 1e-300
    b, c = half_x + 1.0 - a, 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-half_x + a * math.log(half_x) - math.lgamma(a)) * h
    return max(0.0, min(1.0, q))


def sbc_rank_uniformity_pvalue(
    ranks: np.ndarray,
    n_bins: int | None = None,
) -> float:
    """Chi-square goodness-of-fit p-value for rank uniformity.

    Ranks are histogrammed into ``n_bins`` equal bins over [0, max_rank];
    under calibration each bin holds n/n_bins counts in expectation. The
    statistic X² = Σ (O - E)² / E has approximately n_bins - 1 dof; returns
    its survival-function p-value. Large p ⇒ no evidence against uniform
    (well-calibrated) ranks.
    """
    ranks = np.asarray(ranks)
    if n_bins is None:
        n_bins = int(ranks.max()) + 1
    observed = np.histogram(ranks, bins=n_bins, range=(0, n_bins))[0].astype(float)
    expected = ranks.size / n_bins
    stat = float(np.sum((observed - expected) ** 2 / expected))
    return _chi2_sf(stat, n_bins - 1)


def interval_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Fraction of truths inside their prediction intervals.

    coverage = (1/n) Σ 1[lower_i <= y_i <= upper_i]
    """
    lower, upper, truth = np.asarray(lower), np.asarray(upper), np.asarray(truth)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def crps(
    sample_ensemble: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Fair empirical Continuous Ranked Probability Score.

    With ensemble members x_1..x_m per observation and truth y::

        CRPS = E_x|X - y|  -  1/(2m(m-1)) ΣΣ_{i≠j} |x_i - x_j|

    The 1/(m-1) normalization is the unbiased ("fair") estimator of
    E|X - X'| — it removes the self-pair bias of the naive estimator
    (Zamo & Naveau, 2018). Averaged over observations.
    """
    ens = np.asarray(sample_ensemble, dtype=float)  # (n_obs, m)
    y = np.asarray(truth, dtype=float)              # (n_obs,)
    if ens.ndim != 2 or ens.shape[0] != y.shape[0]:
        raise ValueError("sample_ensemble must be (n_obs, n_samp)")
    n_obs, m = ens.shape
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 ensemble members")
    term_abs = np.mean(np.abs(ens - y[:, None]))
    diffs = np.abs(ens[:, :, None] - ens[:, None, :])
    spread = diffs.sum() / (2.0 * m * (m - 1))
    return float(term_abs - spread)


def posterior_mean_mse(mean_field: np.ndarray, truth_field: np.ndarray) -> float:
    """Mean squared error between predicted and true damage fields."""
    mean_field, truth_field = np.asarray(mean_field), np.asarray(truth_field)
    if mean_field.shape != truth_field.shape:
        raise ValueError(f"shape mismatch: {mean_field.shape} vs {truth_field.shape}")
    return float(np.mean((mean_field - truth_field) ** 2))


def oracle_bound_mse(
    noiseless_ref_model_preds: np.ndarray,
    truth: np.ndarray,
) -> float:
    """MSE of precomputed noise-free inverse predictions (oracle bound).

    The oracle is defined as the noise-free inverse solution computed by
    the caller; this helper only scores the provided predictions against
    ground truth, giving the floor any noisy pipeline is compared against.
    """
    return posterior_mean_mse(noiseless_ref_model_preds, truth)
