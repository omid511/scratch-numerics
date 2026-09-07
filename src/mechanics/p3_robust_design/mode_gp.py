"""Mode-decomposition surrogate (roadmap Phase 5).

Fits one independent GP per tracked mode branch (from
``mode_tracking``) mapping design parameters theta -> lambda_cr
contribution of that mode, then combines them via min() (flutter load is
governed by the first mode to go unstable; use ``combination="max"`` for
quantities where the upper branch governs).

Uncertainty from the per-mode GP posteriors is propagated through the
min/max combination by moment matching under independent per-mode
posteriors: the active-mode identity J is treated as categorical with
probabilities computed numerically from the per-mode Gaussians, giving

    Var[lambda_cr] ~= sum_j p_j sigma_j^2
                    + sum_j p_j mu_j^2 - (sum_j p_j mu_j)^2,

i.e. within-mode posterior variance plus a between-branch spread term
that peaks near mode crossings, where the active mode switches.

Gate context (Phase 2): per-design fraction above MAC 0.8 gate = 62% (< 70% threshold — Phase-5 SKIPPED per the roadmap's corrected per-design gate metric). The ModeDecompositionGP below is implemented as dead-per-roadmap code whose stated justification was repudiated by the corrected feasibility analysis. It is retained for future use if a plate configuration with actual mode crossings is identified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .gp_surrogate import GPSurrogate

__all__ = ["ModeDecompositionGP", "ModePrediction"]


@dataclass
class ModePrediction:
    """Combined prediction from a :class:`ModeDecompositionGP`.

    Attributes:
        lam: (n_test,) combined point estimate, ``min_j``/``max_j`` of the
            per-mode GP means depending on the combination convention.
        std: (n_test,) posterior standard deviation propagated through the
            min/max combination (within-mode variance + between-branch
            spread of the active-mode identity).
        active_mode: (n_test,) integer index of the governing mode
            (argmin/argmax of the per-mode means).
        mode_probabilities: (n_test, n_modes) probability that each mode
            is the governing one, under independent per-mode posteriors.
        near_crossing: (n_test,) bool mask flagging points where the top
            two branch means are closer than the crossing tolerance;
            roadmap step 2 says to send these to direct solver validation.
        per_mode_mean: (n_test, n_modes) per-mode GP means.
        per_mode_std: (n_test, n_modes) per-mode GP posterior stds.
    """

    lam: np.ndarray
    std: np.ndarray
    active_mode: np.ndarray
    mode_probabilities: np.ndarray
    near_crossing: np.ndarray
    per_mode_mean: np.ndarray
    per_mode_std: np.ndarray


def _active_probabilities(mu: np.ndarray, sigma: np.ndarray,
                          minimize: bool) -> np.ndarray:
    """Probability that each mode is the min/max, per test point.

    Under independent posteriors f_j ~ N(mu_j, sigma_j^2):

        p_j = integral phi_j(z) * prod_{k != j} P(f_k > z) dz   (min)

    computed by trapezoid quadrature on a shared z-grid spanning all
    branches' +-6-sigma support.

    Args:
        mu: (n, m) per-mode means.
        sigma: (n, m) per-mode standard deviations (> 0).
        minimize: True for min-combination (lambda_cr), False for max.

    Returns:
        (n, m) probabilities, rows summing to 1.
    """
    n, m = mu.shape
    lo = (mu - 6.0 * sigma).min(axis=1)
    hi = (mu + 6.0 * sigma).max(axis=1)
    n_grid = 241
    z = np.linspace(lo, hi, n_grid, axis=1)[:, None, :]  # (n, 1, G)

    zn = (z - mu[:, :, None]) / sigma[:, :, None]  # (n, m, G)
    pdf_self = norm.pdf(zn) / sigma[:, :, None]
    # P(f_k <= z): branch k is BELOW z (i.e. beats j for the min).
    cdf_leq = norm.cdf(zn)

    # NOTE: the k != j product below is computed directly per branch. The
    # algebraically equivalent total-product-divided-by-own-factor form
    # loses tail mass when the own-tail term underflows, so it is not used.
    p = np.empty((n, m))
    if minimize:
        # j governs iff every other branch stays above z.
        surv = 1.0 - cdf_leq  # P(f_k > z), (n, m, G)
        for j in range(m):
            keep = [k for k in range(m) if k != j]
            if keep:
                others = np.prod(surv[:, keep, :], axis=1)[:, None, :]
            else:
                others = np.ones((n, 1, n_grid))
            p[:, j] = np.trapezoid(
                pdf_self[:, j:j + 1, :] * others, x=z, axis=2
            )[:, 0]
    else:
        for j in range(m):
            keep = [k for k in range(m) if k != j]
            if keep:
                others = np.prod(cdf_leq[:, keep, :], axis=1)[:, None, :]
            else:
                others = np.ones((n, 1, n_grid))
            p[:, j] = np.trapezoid(
                pdf_self[:, j:j + 1, :] * others, x=z, axis=2
            )[:, 0]
    p = np.clip(p, 0.0, None)
    total = p.sum(axis=1, keepdims=True)
    return p / np.maximum(total, 1e-300)


class ModeDecompositionGP:
    """Per-mode GP surrogate combined via min/max (Phase 5, approach b).

    For each tracked mode branch j an independent
    :class:`~mechanics.p3_robust_design.gp_surrogate.GPSurrogate` maps
    design parameters x -> lambda_cr contribution of that branch. The
    combined prediction follows the roadmap convention
    ``lambda_cr(x) = min_j f_j(x)`` (the first mode to flutter governs);
    pass ``combination="max"`` when the upper envelope governs.

    Args:
        combination: "min" (default, lambda_cr convention) or "max".
        crossing_tol: Mean-gap tolerance for flagging points as near a
            mode crossing (roadmap: |f_A(x) - f_B(x)| < eps). If None,
            defaults to 1% of the training-target range.
        length_scales: (d,) initial RBF length scales passed to each
            per-mode GP. If None, each GP derives them from its data.
        signal_var: Signal variance for each per-mode GP.
        noise_var: Observation-noise variance for each per-mode GP.
        optimize_hyperparams: Fit each per-mode GP with LML hyperparameter
            optimization (slower; see GPSurrogate.fit).
    """

    def __init__(
        self,
        combination: str = "min",
        crossing_tol: float | None = None,
        length_scales: np.ndarray | None = None,
        signal_var: float = 1.0,
        noise_var: float = 1e-6,
        optimize_hyperparams: bool = False,
    ):
        if combination not in ("min", "max"):
            raise ValueError(
                f"combination must be 'min' or 'max', got {combination!r}"
            )
        self.combination = combination
        self.crossing_tol = crossing_tol
        self._length_scales = (
            None if length_scales is None else np.array(length_scales, dtype=float)
        )
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.optimize_hyperparams = optimize_hyperparams
        self.mode_gps: list[GPSurrogate] = []
        self.n_modes = 0

    def fit(self, X: np.ndarray, y_per_mode: np.ndarray,
            objective_fns=None) -> "ModeDecompositionGP":
        """Fit one GP per tracked mode branch.

        Args:
            X: (n, d) design parameters.
            y_per_mode: per-mode training targets from the mode-tracking
                infrastructure — array of shape (n, n_modes), or a
                sequence of (n,) arrays, one per tracked branch.
            objective_fns: Must be None. Per-mode gradient enhancement is
                not implemented; any non-None value raises ValueError
                rather than being silently ignored.

        Returns:
            self.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if isinstance(y_per_mode, np.ndarray) and y_per_mode.ndim == 2:
            Y = np.asarray(y_per_mode, dtype=float)
        elif isinstance(y_per_mode, np.ndarray) and y_per_mode.ndim == 1:
            Y = np.asarray(y_per_mode, dtype=float).reshape(-1, 1)
        else:
            cols = [np.asarray(y, dtype=float).ravel() for y in y_per_mode]
            if not cols:
                raise ValueError("y_per_mode must contain at least one mode")
            Y = np.column_stack(cols)
        if Y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y_per_mode has {Y.shape[0]}"
            )
        if objective_fns is not None:
            raise ValueError(
                "ModeDecompositionGP.fit: objective_fns (per-mode gradient "
                "enhancement) is not implemented; pass objective_fns=None "
                "for plain per-mode GPs."
            )

        self.n_modes = Y.shape[1]
        self.mode_gps = []
        for j in range(self.n_modes):
            gp = GPSurrogate(
                length_scales=(
                    None if self._length_scales is None
                    else self._length_scales.copy()
                ),
                signal_var=self.signal_var,
                noise_var=self.noise_var,
            )
            gp.fit(X, Y[:, j],
                   optimize_hyperparams=self.optimize_hyperparams)
            self.mode_gps.append(gp)
        return self

    def predict(self, X: np.ndarray) -> ModePrediction:
        """Predict combined lambda_cr with propagated uncertainty.

        Args:
            X: (n, d) or (d,) design points.

        Returns:
            ModePrediction with combined mean/std, active-mode identity,
            per-mode means/stds, active-mode probabilities and a
            near-crossing flag mask.
        """
        if not self.mode_gps:
            raise RuntimeError("ModeDecompositionGP not fitted. Call fit() first.")

        X = np.atleast_2d(np.asarray(X, dtype=float))
        mus = np.empty((X.shape[0], self.n_modes))
        stds = np.empty_like(mus)
        for j, gp in enumerate(self.mode_gps):
            mus[:, j], stds[:, j] = gp.predict(X)

        # Guard against zero predictive std (exact interpolation at
        # training points breaks the probability quadrature).
        floor = 1e-12 * max(1.0, float(mus.std()))
        sigma = np.maximum(stds, floor)

        minimize = self.combination == "min"
        op = np.min if minimize else np.max
        argop = np.argmin if minimize else np.argmax

        lam = op(mus, axis=1)
        active = argop(mus, axis=1)

        p = _active_probabilities(mus, sigma, minimize)
        # Moment matching: Z = f_J with J ~ Categorical(p) and
        # independent per-mode posteriors.
        var_within = np.sum(p * sigma ** 2, axis=1)
        mean_mix = np.sum(p * mus, axis=1)
        var_between = np.sum(p * mus ** 2, axis=1) - mean_mix ** 2
        std_comb = np.sqrt(np.maximum(var_within + var_between, 0.0))

        # Crossing detection (roadmap step 2): gap between the top two
        # branch means below tolerance flags points for solver validation.
        sorted_mu = np.sort(mus, axis=1)
        if minimize:
            gap = sorted_mu[:, 1] - sorted_mu[:, 0] if self.n_modes >= 2 \
                else np.full(X.shape[0], np.inf)
        else:
            gap = sorted_mu[:, -1] - sorted_mu[:, -2] if self.n_modes >= 2 \
                else np.full(X.shape[0], np.inf)
        tol = self.crossing_tol
        if tol is None and self.mode_gps:
            y_train = np.column_stack([
                gp.get_training_data()[1] for gp in self.mode_gps
            ])
            tol = 0.01 * float(y_train.max() - y_train.min() + 1e-30)
            # Floor so constant training targets (range 0) cannot pin the
            # tolerance near 1e-32 and silently disable crossing routing.
            tol = max(tol, 1e-6 * max(1.0, float(np.abs(y_train).max())))
        near_crossing = gap < tol

        return ModePrediction(
            lam=lam,
            std=std_comb,
            active_mode=active,
            mode_probabilities=p,
            near_crossing=near_crossing,
            per_mode_mean=mus,
            per_mode_std=stds,
        )
