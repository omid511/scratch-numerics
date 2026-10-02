#!/usr/bin/env python3
"""G1 sidecar inputs for P4 quantile models (additive only).

G1 member: log_dt. Later groups append members here without touching
existing code paths:
  G2: velocity
  G4: cal_scale, dom_freq, bandwidth, env_slope, energy, sat_frac,
      freq2, power_ratio, xcorr, phase_slope, is_adapted, zcr, skew, kurt,
      v_over_bin, dist_prior

Scalar contract: every member maps one clip -> one finite float. NaN or
non-finite values fall back to the member's declared fallback so a missing
field never kills a batch. Standardization uses train-split mean/std
computed once in the runner and stored in results.json.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


def _clip_dt(clip: Any) -> float | None:
    dt = getattr(clip, "dt", None)
    try:
        dt = float(dt) if dt is not None else None
    except (TypeError, ValueError):
        return None
    if dt is None or not math.isfinite(dt) or dt <= 0:
        return None
    return dt


def _member_log_dt(clip: Any) -> float:
    dt = _clip_dt(clip)
    if dt is None:
        return float("nan")
    return math.log(dt)


def _member_velocity(clip: Any) -> float:
    v = getattr(clip, "velocity", None)
    try:
        v = float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")
    return v



def _signals_2d(clip: Any) -> np.ndarray | None:
    try:
        s = np.asarray(clip.sensor_signals, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if s.ndim != 2 or s.shape[0] < 1 or s.shape[1] < 8:
        return None
    # Mask-lifted clips carry [signals; mask] rows. Spectral/energy members
    # must read physical rows only; mask rows (0/1) would corrupt FFTs and
    # energy. Same binary-second-half rule as baselines._split_signals_mask.
    n = s.shape[0]
    if n % 2 == 0 and n >= 2:
        second = s[n // 2:]
        if second.size and bool(np.all((second == 0.0) | (second == 1.0))):
            s = s[:n // 2]
    if not np.all(np.isfinite(s)):
        return None
    return s


def _cal_block(clip: Any, n: int = 51) -> np.ndarray | None:
    s = _signals_2d(clip)
    if s is None:
        return None
    return s[:, :min(n, s.shape[1])]


def _member_cal_scale(clip: Any) -> float:
    c = _cal_block(clip)
    if c is None:
        return float("nan")
    off = c.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean((c - off) ** 2)))


def _member_energy(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None:
        return float("nan")
    return float(np.mean(s ** 2))


def _member_sat_frac(clip: Any) -> float:
    try:
        v = float(getattr(clip, "sat_frac", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return v


def _member_is_adapted(clip: Any) -> float:
    dt = _clip_dt(clip)
    if dt is None:
        return float("nan")
    return 0.0 if abs(dt - 0.5 / 512) < 1e-12 else 1.0


def _fft_peak(s: np.ndarray, dt: float | None) -> tuple[float, float, float, float]:
    """Return (dom_freq_hz, freq2_hz, peak_ratio, bandwidth_hz)."""
    pooled = s.mean(axis=0)
    n = pooled.shape[0]
    vals = np.fft.rfft(pooled)
    mag = np.abs(vals)
    if len(mag) < 3:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    use_dt = dt if dt is not None else 1.0
    freqs = np.fft.rfftfreq(n, d=use_dt)
    mag_nodc = mag.copy()
    mag_nodc[0] = 0.0
    tot = float(mag.sum())
    if not np.isfinite(tot) or tot <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    i1 = int(np.argmax(mag_nodc))
    p1 = float(mag[i1])
    rest = mag_nodc.copy()
    rest[max(0, i1 - 1):i1 + 2] = 0.0
    p2 = float(rest.max())
    f1 = float(freqs[i1]) if i1 < len(freqs) else float("nan")
    i2 = int(np.argmax(rest))
    f2 = float(freqs[i2]) if i2 < len(freqs) else float("nan")
    power = mag ** 2
    tot_p = float(power.sum()) + 1e-30
    mean = float((freqs * power).sum() / tot_p)
    bw = float(np.sqrt(((freqs - mean) ** 2 * power).sum() / tot_p))
    return (f1, f2, p1 / max(p2, 1e-30), bw)


def _member_dom_freq(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None:
        return float("nan")
    return _fft_peak(s, _clip_dt(clip))[0]


def _member_freq2(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None:
        return float("nan")
    return _fft_peak(s, _clip_dt(clip))[1]


def _member_power_ratio(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None:
        return float("nan")
    return _fft_peak(s, _clip_dt(clip))[2]


def _member_bandwidth(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None:
        return float("nan")
    return _fft_peak(s, _clip_dt(clip))[3]


def _member_env_slope(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None:
        return float("nan")
    env = np.sqrt(np.mean(s ** 2, axis=0) + 1e-30)
    t = np.arange(env.shape[0], dtype=np.float64)
    la = np.log(env + 1e-30)
    ok = np.isfinite(la)
    if ok.sum() < 8:
        return float("nan")
    A = np.column_stack([t[ok], np.ones(int(ok.sum()))])
    coef, *_ = np.linalg.lstsq(A, la[ok], rcond=None)
    dt = _clip_dt(clip)
    return float(coef[0] / dt) if dt is not None else float(coef[0])


def _member_xcorr(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None or s.shape[0] < 2:
        return float("nan")
    z = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-30)
    c = np.corrcoef(z)
    iu = np.triu_indices(c.shape[0], k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _member_phase_slope(clip: Any) -> float:
    s = _signals_2d(clip)
    if s is None or s.shape[0] < 2 or s.shape[1] < 16:
        return float("nan")
    ref = s[0]
    slopes = []
    for j in range(1, s.shape[0]):
        cc = np.correlate(s[j] - s[j].mean(), ref - ref.mean(), mode="full")
        lag = int(np.argmax(cc)) - (len(ref) - 1)
        slopes.append(abs(lag))
    return float(np.median(slopes))


def _flat(clip: Any) -> np.ndarray | None:
    s = _signals_2d(clip)
    if s is None:
        return None
    v = s.ravel()
    v = v[np.isfinite(v)]
    if v.size < 16:
        return None
    return v


def _member_zcr(clip: Any) -> float:
    v = _flat(clip)
    if v is None:
        return float("nan")
    c = v - np.median(v)
    return float(np.mean(c[:-1] * c[1:] < 0.0))


def _member_skew(clip: Any) -> float:
    v = _flat(clip)
    if v is None:
        return float("nan")
    sd = float(v.std())
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean(((v - v.mean()) / sd) ** 3))


def _member_kurt(clip: Any) -> float:
    v = _flat(clip)
    if v is None:
        return float("nan")
    sd = float(v.std())
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean(((v - v.mean()) / sd) ** 4) - 3.0)


def member_names(group: str) -> list[str]:
    """Ordered member names for a group tag (cumulative)."""
    order = ["log_dt"]
    if group in ("G2", "G3", "G4", "G5"):
        order += ["velocity"]
    if group in ("G3", "G4", "G5"):
        order += ["cal_scale"]
    if group in ("G4", "G5"):
        order += ["dom_freq", "bandwidth", "env_slope", "energy", "sat_frac",
                  "freq2", "power_ratio", "xcorr", "phase_slope", "is_adapted",
                  "zcr", "skew", "kurt", "v_over_bin", "dist_prior"]
    return [m for m in order if m in MEMBERS]


def extract(clips: list[Any], names: list[str]) -> np.ndarray:
    """Extract (n_clips, n_members) array; non-finite -> member fallback."""
    cols = []
    for name in names:
        fn, fallback = MEMBERS[name]
        col = []
        for c in clips:
            try:
                v = float(fn(c))
            except (TypeError, ValueError):
                v = float("nan")
            if not math.isfinite(v):
                v = fallback
            col.append(v)
        cols.append(np.asarray(col, dtype=np.float64))
    if not cols:
        return np.zeros((len(clips), 0), dtype=np.float64)
    return np.stack(cols, axis=1)


def standardize_fit(arr: np.ndarray) -> dict[str, list[float]]:
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0) + 1e-8
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return {"mean": [float(v) for v in mu], "std": [float(v) for v in sd]}


def standardize_apply(arr: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    mu = np.asarray(stats["mean"], dtype=np.float64)
    sd = np.asarray(stats["std"], dtype=np.float64)
    return (np.asarray(arr, dtype=np.float64) - mu) / sd

MEMBERS: dict[str, tuple[Callable[[Any], float], float]] = {
    "log_dt": (_member_log_dt, math.log(0.5 / 512)),
    "velocity": (_member_velocity, 1500.0),
    "cal_scale": (_member_cal_scale, 1.0),
    "dom_freq": (_member_dom_freq, 100.0),
    "bandwidth": (_member_bandwidth, 50.0),
    "env_slope": (_member_env_slope, 0.0),
    "energy": (_member_energy, 1.0),
    "sat_frac": (_member_sat_frac, 0.0),
    "freq2": (_member_freq2, 200.0),
    "power_ratio": (_member_power_ratio, 10.0),
    "xcorr": (_member_xcorr, 0.5),
    "phase_slope": (_member_phase_slope, 0.0),
    "is_adapted": (_member_is_adapted, 0.0),
    "zcr": (_member_zcr, 0.1),
    "skew": (_member_skew, 0.0),
    "kurt": (_member_kurt, 0.0),
}

