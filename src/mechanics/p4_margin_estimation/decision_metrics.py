"""Phase-1 evaluation slice: split-conformal intervals, decision metrics, paired design comparison.

Works against the shared clip/predictor protocols (no new dependencies):

- Clip protocol: objects with ``.sensor_signals`` float ``(C, T)``,
  ``.margin`` float, ``.design_id`` str, optional ``.dt``/``.time``.
- Predictor protocol: any object with ``.predict(clips) -> np.ndarray``
  (matches ``baselines.MarginPredictor``).

Calibration discipline: calibration clips must come from designs disjoint
from train AND test (caller-enforced). Where a function receives both train
and calibration lists it raises on train/calib design overlap.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ridge_residual_intervals",
    "interval_score",
    "safety_availability",
    "paired_design_comparison",
    "near_flutter_mask",
    "regime_metrics",
]

_DEFAULT_ALPHA = 0.10
_NEAR_FLUTTER_THRESHOLD = 0.15


def _clip_margins(clips) -> np.ndarray:
    return np.asarray([float(c.margin) for c in clips], dtype=float)


def _clip_designs(clips) -> list:
    return [getattr(c, "design_id", None) for c in clips]


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return alpha


def ridge_residual_intervals(
    predictor,
    calib_clips,
    test_clips,
    alpha: float = _DEFAULT_ALPHA,
    train_clips=None,
) -> dict:
    """Split-conformal intervals from calibration absolute residuals.

    The predictor must already be fitted by the caller. The adjustment is
    ``quantile(|y_calib - pred_calib|, 1 - alpha)`` over finite calibration
    residuals; test intervals are ``pred +/- adjustment``.

    Raises
    ------
    ValueError
        If ``calib_clips`` is empty, if fewer than 2 finite calibration
        residuals exist, if ``alpha`` is not in (0, 1), or if ``train_clips``
        is given and shares any design id with ``calib_clips``.

    Returns a dict with ``lower``/``upper`` np.ndarray (one row per test
    clip), ``adjustment`` float, ``calib_n`` (finite residuals used), and
    ``calib_designs`` (unique calibration design ids seen).
    """
    alpha = _check_alpha(alpha)
    calib_clips = list(calib_clips)
    test_clips = list(test_clips)
    if len(calib_clips) == 0:
        raise ValueError("Calibration set is empty")
    if train_clips is not None:
        train_ids = {d for d in _clip_designs(list(train_clips)) if d is not None}
        calib_ids = {d for d in _clip_designs(calib_clips) if d is not None}
        overlap = train_ids & calib_ids
        if overlap:
            raise ValueError(f"train/calib design overlap: {sorted(overlap)[:5]}")

    y_calib = _clip_margins(calib_clips)
    pred_calib = np.asarray(predictor.predict(calib_clips), dtype=float).ravel()
    if pred_calib.shape != y_calib.shape:
        raise ValueError(
            f"predictor output shape {pred_calib.shape} != calib shape {y_calib.shape}"
        )
    resid = np.abs(y_calib - pred_calib)
    finite = resid[np.isfinite(resid)]
    if finite.size < 2:
        raise ValueError(f"need >=2 finite calibration residuals, got {finite.size}")
    adjustment = float(np.quantile(finite, 1.0 - alpha))

    pred_test = np.asarray(predictor.predict(test_clips), dtype=float).ravel()
    if pred_test.shape != (len(test_clips),):
        raise ValueError(
            f"predictor output shape {pred_test.shape} != test size {(len(test_clips),)}"
        )
    calib_designs = len({d for d in _clip_designs(calib_clips) if d is not None})
    return {
        "lower": pred_test - adjustment,
        "upper": pred_test + adjustment,
        "adjustment": adjustment,
        "calib_n": int(finite.size),
        "calib_designs": int(calib_designs),
    }


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = _DEFAULT_ALPHA,
) -> float:
    """Mean Winkler (interval) score.

    ``mean(upper - lower + (2/alpha) * under-penalty + (2/alpha) * over-penalty)``
    where the under-penalty is ``lower - y`` when ``y < lower`` (else 0) and
    the over-penalty is ``y - upper`` when ``y > upper`` (else 0).

    Raises ValueError on shape mismatch, empty input, any non-finite entry,
    any ``lower > upper`` entry, or ``alpha`` not in (0, 1).
    """
    alpha = _check_alpha(alpha)
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch: {y.shape} vs {lo.shape} vs {hi.shape}")
    if y.size == 0:
        raise ValueError("inputs are empty")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("inputs must be finite")
    if bool(np.any(lo > hi)):
        raise ValueError("lower exceeds upper on some entries")
    width = hi - lo
    under = np.where(y < lo, lo - y, 0.0)
    over = np.where(y > hi, y - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (under + over)))


def safety_availability(
    y_true: np.ndarray,
    median_pred: np.ndarray,
    lower_pred: np.ndarray,
    design_ids,
) -> dict:
    """Safety/availability decision metrics at the zero-margin boundary.

    Unsafe means true margin ``<= 0``; a safe declaration means the relevant
    prediction (here the median) is ``> 0``; certification means
    ``lower > 0``.

    NaN cases (all documented, never crashing): ``false_safe_rate`` is NaN
    when there are no unsafe clips; ``unsafe_fraction_among_safe_declared``
    is NaN when nothing is declared safe; ``safe_certification_availability``
    is NaN when the input is empty. Counts are always exact ints (zero when
    the corresponding set is empty).
    """
    y = np.asarray(y_true, dtype=float).ravel()
    med = np.asarray(median_pred, dtype=float).ravel()
    lo = np.asarray(lower_pred, dtype=float).ravel()
    designs = list(design_ids)
    if not (y.shape == med.shape == lo.shape) or len(designs) != y.size:
        raise ValueError("y_true/median_pred/lower_pred/design_ids length mismatch")
    if y.size and not (
        np.all(np.isfinite(y)) and np.all(np.isfinite(med)) and np.all(np.isfinite(lo))
    ):
        raise ValueError("y_true/median_pred/lower_pred must be finite")

    unsafe = y <= 0
    declared = med > 0
    certified = lo > 0
    n_unsafe = int(unsafe.sum())
    n_declared = int(declared.sum())
    n_med_fs = int((unsafe & declared).sum())
    n_lo_fs = int((unsafe & certified).sum())
    designs = np.asarray(designs, dtype=object)
    return {
        # P(declared safe | unsafe); NaN when n_unsafe == 0.
        "false_safe_rate": (n_med_fs / n_unsafe) if n_unsafe else float("nan"),
        # P(unsafe | declared safe); NaN when n_safe_declared == 0.
        "unsafe_fraction_among_safe_declared": (
            (n_med_fs / n_declared) if n_declared else float("nan")
        ),
        # Fraction certifiable via the lower bound; NaN when input empty.
        "safe_certification_availability": (
            float(certified.mean()) if y.size else float("nan")
        ),
        "n_unsafe": n_unsafe,
        "n_median_false_safe": n_med_fs,
        "n_lower_false_safe": n_lo_fs,
        "n_safe_declared": n_declared,
        "n_designs_with_false_safe": int(
            len(set(designs[unsafe & declared].tolist())) if n_med_fs else 0
        ),
        "n_unsafe_designs": int(len(set(designs[unsafe].tolist())) if n_unsafe else 0),
        "n_total": int(y.size),
    }


def paired_design_comparison(
    records_a,
    records_b,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired design-level comparison of two per-design record lists.

    Each record list holds ``(design_id, value)`` pairs. Both lists must
    cover the identical design set (no missing/extra/duplicated ids), else
    ValueError. The paired difference is ``a - b`` per design; uncertainty
    comes from resampling designs with replacement and averaging within each
    resample (a minimal local version of the trainer's bootstrap_by_design
    pattern).

    Values must be finite (ValueError otherwise); ``n_bootstrap`` must be a
    positive int. Returns ``mean_diff``, the 95% bootstrap CI
    (``ci_low``/``ci_high``), ``frac_gt0`` (fraction of bootstrap means > 0),
    plus ``n_designs``/``n_bootstrap``.
    """
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    a = list(records_a)
    b = list(records_b)
    if len(a) != len(b):
        raise ValueError(f"record count mismatch: {len(a)} vs {len(b)}")
    ids_a = [r[0] for r in a]
    ids_b = [r[0] for r in b]
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise ValueError("duplicated design ids within a record list")
    if set(ids_a) != set(ids_b):
        raise ValueError(
            f"design set mismatch: only_a={sorted(set(ids_a) - set(ids_b))[:5]} "
            f"only_b={sorted(set(ids_b) - set(ids_a))[:5]}"
        )
    order = sorted(set(ids_a))
    da = {r[0]: float(r[1]) for r in a}
    db = {r[0]: float(r[1]) for r in b}
    diffs = np.array([da[d] - db[d] for d in order], dtype=float)
    if not np.all(np.isfinite(diffs)):
        raise ValueError("record values must be finite")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(order), size=(n_bootstrap, len(order)))
    boot_means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.quantile(boot_means, 0.025)),
        "ci_high": float(np.quantile(boot_means, 0.975)),
        "frac_gt0": float((boot_means > 0).mean()),
        "n_designs": int(len(order)),
        "n_bootstrap": int(n_bootstrap),
    }


def near_flutter_mask(margins: np.ndarray, threshold: float = _NEAR_FLUTTER_THRESHOLD) -> np.ndarray:
    """Boolean mask of clips whose margin is below ``threshold`` (default 0.15).

    Raises ValueError if ``threshold`` is not finite. Non-finite margins map
    to False (comparison semantics), they do not raise.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    return np.asarray(margins, dtype=float) < threshold

def regime_metrics(y_true, median_pred, lower_pred, upper_pred, sat_flag, design_ids,
                   alpha: float = _DEFAULT_ALPHA) -> dict:
    """Accuracy/interval/decision metrics split by saturation regime.
    Review finding 2: aggregate rankings hide regime-specific behavior, so
    every comparison reports saturated / unsaturated slices separately with
    exact counts and affected design counts. ``sat_flag`` is boolean per
    clip (True = saturated). Interval inputs may be None for point-only
    models (coverage/width/score report NaN). Returns
    ``{'saturated': {...}, 'unsaturated': {...}}`` with n, n_designs, mae,
    coverage, mean_width, interval_score, false_safe_rate,
    unsafe_fraction_among_safe_declared, safe_certification_availability.
    """
    alpha = _check_alpha(alpha)
    y = np.asarray(y_true, dtype=float)
    med = np.asarray(median_pred, dtype=float)
    sat = np.asarray(sat_flag, dtype=bool)
    dids = list(design_ids)
    if not (len(y) == len(med) == len(sat) == len(dids)):
        raise ValueError("input length mismatch")
    lo = None if lower_pred is None else np.asarray(lower_pred, dtype=float)
    hi = None if upper_pred is None else np.asarray(upper_pred, dtype=float)
    if (lo is None) != (hi is None):
        raise ValueError("lower/upper must both be given or both None")
    if lo is not None and not (len(lo) == len(hi) == len(y)):
        raise ValueError("interval length mismatch")
    out = {}
    for name, mask in (("saturated", sat), ("unsaturated", ~sat)):
        idx = np.flatnonzero(mask)
        div = {"n": int(len(idx)), "n_designs": int(len({dids[i] for i in idx.tolist()}))}
        if len(idx) == 0:
            div.update({"mae": float("nan"), "coverage": float("nan"),
                        "mean_width": float("nan"), "interval_score": float("nan")})
        else:
            div["mae"] = float(np.mean(np.abs(med[idx] - y[idx])))
            if lo is None:
                div.update({"coverage": float("nan"), "mean_width": float("nan"),
                            "interval_score": float("nan")})
            else:
                div["coverage"] = float(np.mean((y[idx] >= lo[idx]) & (y[idx] <= hi[idx])))
                div["mean_width"] = float(np.mean(hi[idx] - lo[idx]))
                div["interval_score"] = interval_score(y[idx], lo[idx], hi[idx], alpha)
        # safety_availability handles empty inputs with documented NaNs.
        sa = safety_availability(y[idx], med[idx], lo[idx] if lo is not None else med[idx],
                                 [dids[i] for i in idx.tolist()])
        div.update(sa)
        out[name] = div
    return out
