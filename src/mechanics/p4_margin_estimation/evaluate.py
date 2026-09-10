"""Phase-6 warning metrics downstream of continuous margin predictions.

The margin model produces continuous predictions; this module implements the
separate decision layer that turns those predictions into warnings and scores
the warnings. All functions operate on plain numpy arrays so they can be used
against any predictor (TCN quantile medians, GRU, growth-rate baseline).

Data semantics (matching ``TransientClip`` and ``generate_p4_dataset.py``):

- A clip's scalar label is ``margin = (u_crit - velocity) / u_crit``:
  positive subcritical, zero flutter, negative supercritical.
- For lead-time analysis each clip contributes one *margin series* sampled on
  its common time grid (``TransientClip.time``, shape ``(n_timesteps,)``):
  the predicted margin series from the decision layer and the corresponding
  true margin series. A clip "crosses" when its margin falls to or below the
  warning threshold (margin shrinking toward/past 0 means approaching
  flutter). Time ordering follows ascending sample index; convert indices to
  physical seconds via the caller's ``dt`` if needed.
- Clip-level warning flags are booleans aligned across clips; a "failure"
  flag marks clips whose true margin actually crosses the threshold.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "broadcast_scalar_to_series",
    "compute_lead_time",
    "lead_time_to_event",
    "prefix_warning_available",
    "lead_time_precision_recall",
    "false_alarm_rate",
    "roc_auc",
]


def broadcast_scalar_to_series(
    scalar_margins: np.ndarray,
    n_timesteps: int,
) -> np.ndarray:
    """Broadcast per-clip scalar margins to constant series.

    Current margin heads (TCN quantile median, GRU, growth-rate) emit one
    scalar margin per clip. Lead-time analysis needs a margin SERIES per
    clip on its time grid; this adapter broadcasts each scalar to a
    constant ``(n_clips, n_timesteps)`` series. Honest caveat: a constant
    series warns at index 0 or never, so lead times degenerate to
    ``0``/``NaN`` — genuine early-warning evaluation requires a
    per-timestep head. Do not fabricate ramps; use this only to run the
    warning metrics without inventing temporal structure.
    """
    scalars = np.asarray(scalar_margins, dtype=float).ravel()
    if n_timesteps < 1:
        raise ValueError(f"n_timesteps must be positive, got {n_timesteps}")
    return np.repeat(scalars[:, None], int(n_timesteps), axis=1)


def compute_lead_time(
    pred_margin_series: np.ndarray,
    true_margin_series: np.ndarray,
    warn_threshold: float = 0.15,
) -> float:
    """Lead time (in samples) between predicted and true threshold crossings.

    The first index where the predicted series is ``<= warn_threshold`` is the
    warning time; the first index where the true series is ``<=
    warn_threshold`` is the failure time. Lead time = failure_index -
    warning_index in samples (positive means the warning preceded failure).

    Returns NaN when either series never crosses the threshold (no warning,
    or no actual instability to warn about), since lead time is undefined.

    Requires per-timestep SERIES of shape ``(T,)`` (not per-clip scalars):
    scalar heads must go through :func:`broadcast_scalar_to_series` first
    (with its degenerate-lead-time caveat). A 0-d/scalar input raises an
    informative error instead of silently returning a fake metric.
    """
    pred = np.asarray(pred_margin_series)
    true = np.asarray(true_margin_series)
    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError(
            "compute_lead_time needs one 1-d margin SERIES per clip with shape "
            f"(T,), got {pred.shape} vs {true.shape}. Broadcast scalar margins "
            "with broadcast_scalar_to_series() first and index one row per clip."
        )
    if pred.shape != true.shape:
        raise ValueError(f"series shape mismatch: {pred.shape} vs {true.shape}")

    warn_idx = np.flatnonzero(pred <= warn_threshold)
    fail_idx = np.flatnonzero(true <= warn_threshold)
    if warn_idx.size == 0 or fail_idx.size == 0:
        return float("nan")
    return float(fail_idx[0] - warn_idx[0])


def lead_time_precision_recall(
    warned_flags: np.ndarray,
    failure_within_horizon_flags: np.ndarray,
) -> dict[str, float]:
    """Precision/recall of clip-level warnings at a given horizon.

    Parameters
    ----------
    warned_flags:
        Boolean array; True where the decision layer warned the clip before
        the horizon elapsed (e.g. predicted margin crossed the threshold with
        enough samples remaining).
    failure_within_horizon_flags:
        Boolean array; True where the clip's true margin actually crossed the
        threshold within the horizon.

    Precision: fraction of warnings that correspond to real imminent
    failures. Recall: fraction of real imminent failures that were warned.
    Both are NaN when their denominator is zero (no warnings / no failures).
    """
    warned = np.asarray(warned_flags, dtype=bool)
    failed = np.asarray(failure_within_horizon_flags, dtype=bool)
    if warned.shape != failed.shape:
        raise ValueError(f"flag shape mismatch: {warned.shape} vs {failed.shape}")

    n_warned = int(warned.sum())
    n_failed = int(failed.sum())
    hits = int((warned & failed).sum())
    return {
        "precision": hits / n_warned if n_warned else float("nan"),
        "recall": hits / n_failed if n_failed else float("nan"),
    }


def false_alarm_rate(
    warned_flags: np.ndarray,
    failure_within_horizon_flags: np.ndarray,
) -> float:
    """Fraction of stable clips (true margin never crosses) that were warned.

    Computed only over clips whose true margin does not cross within the
    horizon. NaN when there are no stable clips.
    """
    warned = np.asarray(warned_flags, dtype=bool)
    failed = np.asarray(failure_within_horizon_flags, dtype=bool)
    if warned.shape != failed.shape:
        raise ValueError(f"flag shape mismatch: {warned.shape} vs {failed.shape}")
    stable = ~failed
    n_stable = int(stable.sum())
    if n_stable == 0:
        return float("nan")
    return float((warned & stable).sum() / n_stable)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC without sklearn.

    ``scores`` are continuous decision scores (e.g. negative predicted
    margin); ``labels`` are binary ground-truth failure flags. AUC equals the
    Mann-Whitney U statistic normalized to [0, 1]: probability that a random
    positive outranks a random negative, with ties counting 0.5. Ties handled
    via average ranks. NaN when only one class is present.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {labels.shape}")
    pos = labels.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    # Average ranks for ties (merge sort keeps equal scores adjacent).
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1

    rank_sum_pos = ranks[pos].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def lead_time_to_event(
    pred_margin_series: np.ndarray,
    true_margin_series: np.ndarray,
    warn_threshold: float = 0.15,
    event_threshold: float = 0.0,
    dt: float | None = None,
) -> float:
    """Lead time between a WARNING crossing and the INSTABILITY event.
    Review §3.2: ``compute_lead_time`` uses one threshold for both series,
    which measures warning-region anticipation, not flutter anticipation.
    Here the warning time is the first predicted ``<= warn_threshold`` and
    the event time is the first true ``<= event_threshold`` (default margin
    zero = flutter). Positive means warning preceded instability.
    With ``dt`` (seconds/sample) the result is in seconds, otherwise samples.
    NaN when either never crosses. Both inputs must be 1-d ``(T,)`` series.
    """
    pred = np.asarray(pred_margin_series)
    true = np.asarray(true_margin_series)
    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError(
            "lead_time_to_event needs one 1-d margin SERIES per clip with shape "
            f"(T,), got {pred.shape} vs {true.shape}."
        )
    if pred.shape != true.shape:
        raise ValueError(f"series shape mismatch: {pred.shape} vs {true.shape}")
    warn_idx = np.flatnonzero(pred <= warn_threshold)
    event_idx = np.flatnonzero(true <= event_threshold)
    if warn_idx.size == 0 or event_idx.size == 0:
        return float("nan")
    lead_samples = float(event_idx[0] - warn_idx[0])
    if dt is not None:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid dt: {dt}")
        return lead_samples * float(dt)
    return lead_samples


def prefix_warning_available(
    n_timesteps: int,
    calibration_samples: int,
    warn_index: int | None,
    min_samples: int = 1,
) -> int | None:
    """Earliest sample index at which a prefix prediction is actually available.
    Review §3.1: a complete-clip scalar prediction is not available at the
    start of its clip. Causal calibration needs ``calibration_samples`` and a
    rolling predictor needs ``min_samples`` (its window length); a warning at
    ``warn_index`` is only actionable once all required samples exist.
    Returns None when no warning fired (``warn_index`` None/NaN).
    Raises on impossible windows: non-positive counts, calibration or window
    longer than the observation, a warning index outside ``[0, n_timesteps)``,
    or a non-integer warning index (silent truncation would misstate timing).
    Window validation runs even when no warning fired, so ``None`` never
    masks a misconfigured observation budget.
    Note: pass the predictor's true window length as ``min_samples`` — the
    default 1 covers only calibration delay, not rolling-window availability.
    """
    if n_timesteps <= 0 or calibration_samples <= 0 or min_samples <= 0:
        raise ValueError("n_timesteps, calibration_samples and min_samples must be positive")
    if calibration_samples > n_timesteps:
        raise ValueError(f"calibration_samples ({calibration_samples}) exceeds observed n_timesteps ({n_timesteps})")
    if min_samples > n_timesteps:
        raise ValueError(f"min_samples ({min_samples}) exceeds observed n_timesteps ({n_timesteps})")
    if warn_index is None or (isinstance(warn_index, float) and np.isnan(warn_index)):
        return None
    if isinstance(warn_index, bool) or not isinstance(warn_index, (int, np.integer)):
        raise ValueError(f"warn_index must be an integer sample index, got {warn_index!r}")
    w = int(warn_index)
    if w < 0 or w >= n_timesteps:
        raise ValueError(f"warn_index ({warn_index}) outside observed [0, {n_timesteps})")
    return int(max(w, int(calibration_samples) - 1, int(min_samples) - 1))
