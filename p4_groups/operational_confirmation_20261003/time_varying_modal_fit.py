"""Waveform-only stationary and linearly time-varying single-mode fits."""
from collections.abc import Mapping
import time

import numpy as np
from scipy.optimize import least_squares


class _FitIssue(Exception):
    """A numerical condition that makes this fit unusable."""

    def __init__(self, status, message, rank=0):
        super().__init__(message)
        self.status = status
        self.rank = int(rank)


def _finite_pair(settings, key):
    try:
        pair = tuple(float(value) for value in settings[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"solver setting {key!r} must be a finite two-value bound") from exc
    if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] >= pair[1]:
        raise ValueError(f"solver setting {key!r} must have increasing finite bounds")
    return pair


def _solver_options(settings):
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be the frozen protocol solver mapping")
    method = str(settings.get("method", "")).lower()
    jacobian = str(settings.get("jacobian", "")).lower()
    if "trf" not in method or "3-point" not in jacobian:
        raise ValueError("solver settings must specify trf and a 3-point Jacobian")

    bounds = {
        "alpha": _finite_pair(settings, "mid_growth_bounds_per_second"),
        "alpha_slope": _finite_pair(settings, "growth_slope_bounds_per_second_squared"),
        "frequency_slope": _finite_pair(settings, "frequency_slope_bounds_hz_per_second"),
    }
    try:
        halfwidth = float(settings["frequency_halfwidth_cycles"])
        max_nfev = int(settings["maximum_evaluations"])
        tolerances = tuple(float(settings[key]) for key in ("ftol", "xtol", "gtol"))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("solver settings have invalid frequency or termination values") from exc
    if (not np.isfinite(halfwidth) or halfwidth <= 0 or max_nfev <= 0
            or len(tolerances) != 3 or not np.isfinite(tolerances).all()
            or min(tolerances) <= 0):
        raise ValueError("solver frequency width, evaluation limit, and tolerances must be positive")
    return bounds, halfwidth, max_nfev, tolerances


def _failure(message, status, started, *, nfev=0, seed_frequency=None,
             basis_rank=0, bound_hits=None):
    return {
        "success": False,
        "message": str(message),
        "alpha_end": None,
        "frequency_end": None,
        "parameters": None,
        "relative_fit_error": None,
        "nfev": int(nfev),
        "status": status,
        "bound_hits": [] if bound_hits is None else list(bound_hits),
        "seed_frequency": seed_frequency,
        "basis_rank": int(basis_rank),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _fft_seed(centered, dt, duration):
    """Pick the aggregate demeaned-Hann FFT-power peak in the physical band."""
    n_samples = centered.shape[1]
    nyquist = 0.5 / dt
    low_frequency, high_frequency = 1.0, nyquist - 1.0
    if high_frequency <= low_frequency:
        return None, low_frequency, high_frequency

    window = np.hanning(n_samples)
    with np.errstate(over="ignore", invalid="ignore"):
        spectrum = np.fft.rfft(centered * window[None, :], n=16 * n_samples, axis=1)
        power = np.sum(spectrum.real * spectrum.real + spectrum.imag * spectrum.imag, axis=0)
    frequencies = np.fft.rfftfreq(16 * n_samples, d=dt)
    in_band = (frequencies >= low_frequency) & (frequencies <= high_frequency)
    if (not in_band.any() or not np.isfinite(power).all()
            or not np.any(power[in_band] > 0.0)):
        return None, low_frequency, high_frequency
    indices = np.flatnonzero(in_band)
    peak = int(indices[np.argmax(power[in_band])])
    seed = float(frequencies[peak])
    if not np.isfinite(seed) or duration <= 0:
        return None, low_frequency, high_frequency
    return seed, low_frequency, high_frequency


def _bound_hits(solution, lower, upper, names):
    hits = []
    for value, lo, hi, name in zip(solution, lower, upper, names):
        fraction = min((value - lo) / (hi - lo), (hi - value) / (hi - lo))
        if fraction <= 1e-5:
            hits.append(name)
    return hits


def _projected_residual(theta, varying, basis_time, signal, rank_state):
    """Profile channel offsets/cosine/sine coefficients at one pole iterate."""
    if varying:
        alpha_mid, alpha_slope, frequency_mid, frequency_slope = theta
    else:
        alpha_mid, frequency_mid = theta
        alpha_slope = frequency_slope = 0.0

    u = basis_time
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        envelope = np.exp(alpha_mid * u + 0.5 * alpha_slope * u * u)
        phase = 2.0 * np.pi * (frequency_mid * u + 0.5 * frequency_slope * u * u)
        basis = np.column_stack((np.ones_like(u), envelope * np.cos(phase),
                                 envelope * np.sin(phase)))
    if not np.isfinite(basis).all():
        raise _FitIssue("non_finite_basis", "non-finite modal basis during optimization")

    try:
        coefficients, _, rank, _ = np.linalg.lstsq(basis, signal.T, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise _FitIssue("linear_solve_failed", f"linear coefficient solve failed: {exc}",
                        rank_state[0]) from exc
    rank_state[0] = int(rank)
    if rank != 3:
        raise _FitIssue("rank_deficient", f"linear modal basis rank is {rank}, expected 3", rank)

    residual = basis @ coefficients - signal.T
    if not np.isfinite(residual).all():
        raise _FitIssue("non_finite_residual", "non-finite projected waveform residual", rank)
    return residual.ravel()


def fit_single_mode(signal, dt, *, varying, settings, stationary=None):
    """Fit one shared complex mode with channel-specific linear coefficients.

    ``signal`` must be calibrated channels-by-samples data.  The fitter has no
    access to references or labels.  For a varying fit, ``stationary`` must be
    a successful result from the stationary fit and supplies its waveform-fit
    midpoint poles as the warm start.
    """
    started = time.perf_counter()
    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be a finite positive scalar") from exc
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive scalar")
    if not isinstance(varying, (bool, np.bool_)):
        raise ValueError("varying must be a boolean")
    varying = bool(varying)

    raw = np.asarray(signal)
    if np.iscomplexobj(raw):
        raise ValueError("signal must be real-valued calibrated measurements")
    try:
        x = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("signal must be a finite real channels-by-samples array") from exc
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 4:
        raise ValueError("signal must have at least one channel and four samples")
    if not np.isfinite(x).all():
        return _failure("signal must contain only finite measurements", "non_finite_input", started)
    if (np.abs(x) >= 50.0).any():
        return _failure("signal reaches the calibrated saturation cap50", "saturated_input", started)

    with np.errstate(over="ignore", invalid="ignore"):
        centered = x - np.mean(x, axis=1, keepdims=True)
        centered_energy = float(np.sum(centered * centered))
    if not np.isfinite(centered_energy) or centered_energy <= 0.0:
        return _failure("signal must have finite, nonzero centered energy",
                        "invalid_centered_energy", started)

    bounds, halfwidth_cycles, max_nfev, tolerances = _solver_options(settings)
    n_samples = x.shape[1]
    # The cached protocol defines an N-sample window as the span (N-1)*dt.
    duration = (n_samples - 1) * dt
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("signal window duration must be finite and positive")
    tau = (np.arange(n_samples, dtype=np.float64) - 0.5 * (n_samples - 1)) * dt
    u = tau / duration

    if varying:
        if not isinstance(stationary, Mapping) or stationary.get("success") is not True:
            return _failure("successful stationary waveform fit required for warm start",
                            "stationary_warm_start_unavailable", started)
        stationary_parameters = stationary.get("parameters")
        if not isinstance(stationary_parameters, Mapping):
            return _failure("stationary fit has no parameter estimates",
                            "stationary_warm_start_unavailable", started)
        try:
            warm_parameters = {
                key: float(stationary_parameters[key])
                for key in ("alpha_mid", "alpha_slope", "frequency_mid", "frequency_slope")
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            return _failure("stationary fit parameters are incomplete",
                            "stationary_warm_start_unavailable", started)
        if (not np.isfinite(tuple(warm_parameters.values())).all()
                or warm_parameters["alpha_slope"] != 0.0
                or warm_parameters["frequency_slope"] != 0.0):
            return _failure("stationary fit parameters are not finite stationary poles",
                            "stationary_warm_start_unavailable", started)

    seed, frequency_min, frequency_max = _fft_seed(centered, dt, duration)
    if seed is None:
        status = "collapsed_frequency_range" if frequency_max <= frequency_min else "no_fft_peak_in_band"
        return _failure("no usable Hann-windowed FFT peak in the physical frequency band",
                        status, started)

    mid_frequency_min = max(frequency_min, seed - halfwidth_cycles / duration)
    mid_frequency_max = min(frequency_max, seed + halfwidth_cycles / duration)
    if not np.isfinite((mid_frequency_min, mid_frequency_max)).all() or mid_frequency_min >= mid_frequency_max:
        return _failure("frequency-mid search bounds collapsed", "collapsed_frequency_mid_bounds",
                        started, seed_frequency=seed)

    alpha_min, alpha_max = bounds["alpha"]
    alpha_slope_min, alpha_slope_max = bounds["alpha_slope"]
    frequency_slope_min, frequency_slope_max = bounds["frequency_slope"]

    if varying:
        slope_cap = min(
            2.0 * (mid_frequency_min - frequency_min) / duration,
            2.0 * (frequency_max - mid_frequency_max) / duration,
        )
        frequency_slope_min = max(frequency_slope_min, -slope_cap)
        frequency_slope_max = min(frequency_slope_max, slope_cap)
        if (not np.isfinite((frequency_slope_min, frequency_slope_max)).all()
                or frequency_slope_min >= frequency_slope_max):
            return _failure("frequency-slope bounds collapse while keeping both endpoint frequencies physical",
                            "collapsed_frequency_slope_bounds", started, seed_frequency=seed)

    if varying:
        lower = np.array((alpha_min * duration, alpha_slope_min * duration ** 2,
                          mid_frequency_min * duration, frequency_slope_min * duration ** 2))
        upper = np.array((alpha_max * duration, alpha_slope_max * duration ** 2,
                          mid_frequency_max * duration, frequency_slope_max * duration ** 2))
        names = ("alpha_mid", "alpha_slope", "frequency_mid", "frequency_slope")
        x0 = np.array((warm_parameters["alpha_mid"] * duration, 0.0,
                      warm_parameters["frequency_mid"] * duration, 0.0))
    else:
        lower = np.array((alpha_min * duration, mid_frequency_min * duration))
        upper = np.array((alpha_max * duration, mid_frequency_max * duration))
        names = ("alpha_mid", "frequency_mid")
        x0 = np.array((0.0, seed * duration))

    if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower >= upper)):
        return _failure("nonlinear solver bounds collapsed", "collapsed_solver_bounds",
                        started, seed_frequency=seed)
    if not np.isfinite(x0).all() or np.any(x0 < lower) or np.any(x0 > upper):
        return _failure("initial point is outside the fitted bounds", "invalid_initial_point",
                        started, seed_frequency=seed)

    rank_state = [0]
    evaluations = [0]

    def residual(theta):
        evaluations[0] += 1
        return _projected_residual(theta, varying, u, x, rank_state)

    try:
        result = least_squares(
            residual, x0, bounds=(lower, upper), method="trf", jac="3-point",
            max_nfev=max_nfev, ftol=tolerances[0], xtol=tolerances[1], gtol=tolerances[2],
        )
    except _FitIssue as exc:
        return _failure(str(exc), exc.status, started, nfev=evaluations[0],
                        seed_frequency=seed, basis_rank=exc.rank or rank_state[0])

    candidate = np.asarray(result.x, dtype=np.float64)
    candidate_hits = (_bound_hits(candidate, lower, upper, names)
                      if candidate.shape == lower.shape and np.isfinite(candidate).all() else [])
    if not result.success:
        return _failure(result.message, int(result.status), started, nfev=int(result.nfev),
                        seed_frequency=seed, basis_rank=rank_state[0], bound_hits=candidate_hits)
    if candidate.shape != lower.shape or not np.isfinite(candidate).all():
        return _failure("solver returned non-finite or malformed parameters", "non_finite_solution",
                        started, nfev=int(result.nfev), seed_frequency=seed,
                        basis_rank=rank_state[0])

    try:
        final_residual = _projected_residual(candidate, varying, u, x, rank_state)
    except _FitIssue as exc:
        return _failure(str(exc), exc.status, started, nfev=int(result.nfev),
                        seed_frequency=seed, basis_rank=exc.rank or rank_state[0],
                        bound_hits=candidate_hits)
    with np.errstate(over="ignore", invalid="ignore"):
        residual_energy = float(np.dot(final_residual, final_residual))
        relative_fit_error = float(residual_energy / centered_energy)

    if varying:
        alpha_mid_scaled, alpha_slope_scaled, frequency_mid_scaled, frequency_slope_scaled = candidate
    else:
        alpha_mid_scaled, frequency_mid_scaled = candidate
        alpha_slope_scaled = frequency_slope_scaled = 0.0
    parameters = {
        "alpha_mid": float(alpha_mid_scaled / duration),
        "alpha_slope": float(alpha_slope_scaled / duration ** 2),
        "frequency_mid": float(frequency_mid_scaled / duration),
        "frequency_slope": float(frequency_slope_scaled / duration ** 2),
    }
    alpha_end = parameters["alpha_mid"] + 0.5 * duration * parameters["alpha_slope"]
    frequency_end = parameters["frequency_mid"] + 0.5 * duration * parameters["frequency_slope"]
    output_values = (relative_fit_error, alpha_end, frequency_end, *parameters.values())
    if not np.isfinite(residual_energy) or not np.isfinite(output_values).all():
        return _failure("solver result has non-finite fit diagnostics", "non_finite_fit_diagnostics",
                        started, nfev=int(result.nfev), seed_frequency=seed,
                        basis_rank=rank_state[0], bound_hits=candidate_hits)

    return {
        "success": True,
        "message": str(result.message),
        "alpha_end": float(alpha_end),
        "frequency_end": float(frequency_end),
        "parameters": parameters,
        "relative_fit_error": float(relative_fit_error),
        "nfev": int(result.nfev),
        "status": int(result.status),
        "bound_hits": candidate_hits,
        "seed_frequency": seed,
        "basis_rank": int(rank_state[0]),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
