"""Controlled S/M/R operating-point acquisition feasibility helper (P4 only).

Feasibility checkpoint for the frontier-directed single-versus-multiple
operating-point experiment. No fitting, no campaign, no training.

Contract (fixed aperture, do not change without dispatcher approval):

- 448 samples at 1 kHz: observed times exactly ``np.arange(448) / 1000``.
  First-to-last span 447 ms.
- Plateau sample indices ``0:128``, ``160:288``, ``320:448`` (128 each).
- Smooth cubic velocity transitions occupy time ``[0.128, 0.160]`` and
  ``[0.288, 0.320]`` (32 samples each, indices ``128:160`` and ``288:320``).
  Transition samples count toward the aperture.
- Five integration pieces (inclusive endpoints)::

      [0.000, 0.128], [0.128, 0.160], [0.160, 0.288], [0.288, 0.320], [0.320, 0.447]

  with observed index ranges ``(0,128)``, ``(128,160)``, ``(160,288)``,
  ``(288,320)``, ``(320,447)``. Concatenated without duplicating shared
  boundary samples this yields exactly 448 observed samples.
- Arms (absolute velocities, identical across designs):

  - S: ``(Uend, Uend, Uend)``
  - M: ``(Uend-delta, Uend+delta, Uend)``
  - R: ``(Uend+delta, Uend-delta, Uend)``

  ``endpoint`` (Uend) and ``delta`` are absolute physical velocities
  (m/s). They MUST be supplied by the dispatcher from original
  proper-training information only (shared fitting designs
  D000007/D000033/D000068/D000087). This module NEVER derives them from
  any evaluated design's ``ucrit``, true margin, or pole information, and
  NEVER takes ``ucrit``/design-margin inputs. Computing true margin
  afterwards remains scoring-only.
- Preserve 16 sensors, the causal 51-sample calibration support, caps, the
  existing initial-only excitation law, and continuous mechanical state.
  Native missing outputs stay missing (NaN + ``valid=False``); they are
  never re-qualified here.
- Existing dynamics matrices (``base``/``stiffness``/``damping``) and air
  ``rho``/``c`` are reused. No target-ratio inputs.

Interfaces exposed (runnable import)::

    plateau_velocities(endpoint, delta, arm) -> (v0, v1, v2)
    velocity_at(time, endpoint, delta, arm) -> float | np.ndarray
    absolute_matrix(plate, velocity, full=False) -> np.ndarray
    acquire(plate, endpoint, delta, arm, replicate=0,
            coefficient_step=.001, full=False) -> StateAcquisition
    acquire_with_diagnostics(...) -> (StateAcquisition, dict)
    chronology(endpoint, delta) -> dict
    refinement_study(plate, endpoint, delta, arm, replicate=0,
                     steps=(.010, .004, .002, .001, .0005, .00025),
                     full=False) -> dict
    rom_comparison(plate, endpoint, delta, arm, replicate=0,
                   coefficient_step=.00025) -> dict
    numerical_report(plate, endpoint, delta, replicate=0) -> dict

Exact checks performed (see :func:`numerical_report`):

- Observed grid equals ``np.arange(448)/1000`` exactly; five piece grids
  are individually uniform and aligned to observed samples.
- No frozen-coefficient block spans a piece boundary (each piece calls
  :func:`propagate_state` separately).
- Same state object is passed at every piece boundary (no resets); boundary
  response agreement is measured as replay evidence.
- Numerical tolerance: reduced ``.001`` vs finest ``.00025`` whole-response
  relative RMS ``<= .005`` (existing step-halving scale).
- ROM tolerance: reduced-finest vs full-finest whole-response relative RMS
  ``<= .02`` (existing protocol scale).
- Failures are reported, never absorbed by relaxing tolerances.

Tradeoffs and limitations:

- :func:`propagate_state` uses midpoint-frozen exponentials. Within constant
  plateaus any block size is exact (up to floating point); within cubic
  transitions accuracy depends on ``coefficient_step``. Splitting at piece
  boundaries removes cross-boundary smearing but does not validate the
  physical admissibility of the 32 ms cubic rate itself.
- ``coefficient_step < .001`` requires a finer integration grid (substeps)
  decimated back to 1 kHz. Cost scales with ``1/dt``; ``.00025`` costs ~4x
  the observed rate. Only the finest reference is computed once per arm and
  reused for all coarser comparisons.
- Full-model exponentials are ``O(n^3)`` in the full state dimension.
  ``numerical_report`` computes the full model ONLY at the finest meaningful
  step (``.00025``), once per arm, and reuses it. It never runs the full
  model at every refinement step.
- Reduced and full states live in different dimensions, so ROM "final-state
  error" is reported in response space (final observed column), not by
  subtracting incompatible state vectors. Same-model refinement reports a
  true state-space distance.
- A numerical pass does NOT imply physical validity. Supersonic
    piston-theory membership (``mach >= 2``) is checked, but peak transition
    rates/accelerations are reported for dispatcher/frontier judgement, not
  claimed valid. No rate or domain bound is invented here.
- Excitation is initial-only (``forcing=None``). No pulses are added.

Why no existing ``physical_pulse`` falls in 0-447 ms:

    ``trajectory_physics.physical_pulse(t)`` is a 1 N transverse half-sine of
    20 ms duration every 2 s starting at 1 s: nonzero only on
    ``[1.00, 1.02)``, ``[3.00, 3.02)``, ... . On the closed aperture
    ``[0.000, 0.447]`` it is identically zero, so ``forcing=None`` reproduces
    the existing initial-only law exactly without adding actuation.
    Calibration (first 51 samples, indices ``0:51``) lies wholly inside the
    first plateau, so the causal ``causal_observation(..., calibration_samples=51,
    cap=50.)`` support is unchanged downstream.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from mechanics.p4_margin_estimation.operational import (
    StateAcquisition,
    propagate_state,
)

__all__ = [
    "SAMPLE_DT",
    "N_SAMPLES",
    "OBS_TIMES",
    "TRANSITION_0",
    "TRANSITION_1",
    "PIECE_INDEX_RANGES",
    "PLATEAU_SLICES",
    "REFINEMENT_STEPS",
    "NUMERICAL_TOL",
    "ROM_TOL",
    "DEFAULT_STEP",
    "FINEST_STEP",
    "ARMS",
    "plateau_velocities",
    "velocity_at",
    "absolute_matrix",
    "acquire",
    "acquire_with_diagnostics",
    "chronology",
    "refinement_study",
    "rom_comparison",
    "numerical_report",
]

# ---------------------------------------------------------------------------
# Fixed aperture constants (do not tune per design).
# ---------------------------------------------------------------------------

SAMPLE_DT = 0.001
N_SAMPLES = 448
OBS_TIMES = np.arange(N_SAMPLES, dtype=float) / 1000.0

TRANSITION_0 = (0.128, 0.160)
TRANSITION_1 = (0.288, 0.320)

# Inclusive observed index ranges per piece: (first, last) sample index.
PIECE_INDEX_RANGES = ((0, 128), (128, 160), (160, 288), (288, 320), (320, 447))

PLATEAU_SLICES = {
    "plateau0": slice(0, 128),
    "plateau1": slice(160, 288),
    "plateau2": slice(320, 448),
}
TRANSITION_SLICES = {
    "transition0": slice(128, 160),
    "transition1": slice(288, 320),
}

REFINEMENT_STEPS = (0.010, 0.004, 0.002, 0.001, 0.0005, 0.00025)
DEFAULT_STEP = 0.001
FINEST_STEP = 0.00025

NUMERICAL_TOL = 0.005
ROM_TOL = 0.02

ARMS = ("S", "M", "R")


# ---------------------------------------------------------------------------
# Schedule.
# ---------------------------------------------------------------------------

def _normalize_arm(arm) -> str:
    if not isinstance(arm, str):
        raise ValueError("arm must be one of 'S', 'M', 'R'")
    key = arm.strip().upper()
    if key not in ARMS:
        raise ValueError("arm must be one of 'S', 'M', 'R'")
    return key


def plateau_velocities(endpoint, delta, arm):
    """Absolute plateau velocities ``(v0, v1, v2)`` in m/s.

    No ``ucrit``/design/margin inputs. ``endpoint``/``delta`` are absolute and
    identical across designs; the dispatcher derives them from original
    proper-training information only.
    """
    try:
        endpoint_f = float(endpoint)
    except (TypeError, ValueError):
        raise ValueError("endpoint must be a finite velocity")
    try:
        delta_f = float(delta)
    except (TypeError, ValueError):
        raise ValueError("delta must be a finite velocity increment")
    if not np.isfinite(endpoint_f) or not np.isfinite(delta_f):
        raise ValueError("endpoint/delta must be finite")
    if endpoint_f <= 0:
        raise ValueError("endpoint must be positive")
    if delta_f < 0:
        raise ValueError("delta must be non-negative")
    key = _normalize_arm(arm)
    if key == "S":
        return (endpoint_f, endpoint_f, endpoint_f)
    if key == "M":
        return (endpoint_f - delta_f, endpoint_f + delta_f, endpoint_f)
    return (endpoint_f + delta_f, endpoint_f - delta_f, endpoint_f)


def _smoothstep(fraction):
    fraction = np.asarray(fraction, dtype=float)
    return 3.0 * fraction**2 - 2.0 * fraction**3


def velocity_at(time, endpoint, delta, arm):
    """Absolute velocity schedule in m/s with exact cubic C1 transitions.

    Plateaus: ``t < 0.128`` -> ``v0``; ``0.160 < t < 0.288`` -> ``v1``;
    ``t > 0.320`` -> ``v2``. Transitions on ``[0.128, 0.160]`` and
    ``[0.288, 0.320]`` use Hermite smoothstep
    ``v0 + (v1-v0)*(3s^2-2s^3)`` with ``s`` the normalized time. Endpoint
    slopes are zero, matching the neighbouring constant plateaus, so value
    and first derivative are continuous (C1). Outside ``[0, 0.447]`` the
    nearest plateau value is returned (extension, not acquisition).
    """
    v0, v1, v2 = plateau_velocities(endpoint, delta, arm)
    scalar = np.ndim(time) == 0
    t = np.atleast_1d(np.asarray(time, dtype=float))
    out = np.full(t.shape, np.nan, dtype=float)
    t0a, t1a = TRANSITION_0
    t0b, t1b = TRANSITION_1
    span_a = t1a - t0a
    span_b = t1b - t0b
    m_plateau0 = t < t0a
    m_trans0 = (t >= t0a) & (t <= t1a)
    m_plateau1 = (t > t1a) & (t < t0b)
    m_trans1 = (t >= t0b) & (t <= t1b)
    m_plateau2 = t > t1b
    out[m_plateau0] = v0
    if np.any(m_trans0):
        s = (t[m_trans0] - t0a) / span_a
        out[m_trans0] = v0 + (v1 - v0) * (3.0 * s**2 - 2.0 * s**3)
    out[m_plateau1] = v1
    if np.any(m_trans1):
        s = (t[m_trans1] - t0b) / span_b
        out[m_trans1] = v1 + (v2 - v1) * (3.0 * s**2 - 2.0 * s**3)
    out[m_plateau2] = v2
    if scalar:
        return float(out[0])
    return out


# ---------------------------------------------------------------------------
# Dynamics (absolute velocity, never ratio/ucrit).
# ---------------------------------------------------------------------------

def absolute_matrix(plate, velocity, full=False):
    """Velocity-dependent dynamics matrix at an absolute velocity.

    Reuses the plate's existing ``base``/``stiffness``/``damping`` matrices and
    design air ``rho_air``/``c_air`` exactly as ``PlateDynamics.matrix`` does,
    but with ``velocity`` given directly in m/s. Never reads ``plate.ucrit``
    and never takes a ratio/margin input.
    """
    try:
        value = float(velocity)
    except (TypeError, ValueError):
        raise ValueError("velocity must be a finite absolute velocity")
    if not np.isfinite(value):
        raise ValueError("velocity must be finite")
    matrices = plate.full if full else plate.reduced
    try:
        base = np.asarray(matrices["base"], dtype=float)
        stiffness = np.asarray(matrices["stiffness"], dtype=float)
        damping = np.asarray(matrices["damping"], dtype=float)
        rho = float(plate.design.rho_air)
        sound = float(plate.design.c_air)
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("plate must expose reduced/full base/stiffness/damping and design rho_air/c_air")
    if not np.isfinite(rho) or not np.isfinite(sound) or rho <= 0 or sound <= 0:
        raise ValueError("invalid design air properties")
    mach = value / sound
    if mach < 2:
        raise ValueError("trajectory left configured Mach>=2 piston-theory domain")
    dynamic = rho * value * value / math.sqrt(mach * mach - 1.0)
    coefficient = dynamic * (mach * mach - 2.0) / (mach * mach - 1.0) / value
    if base.shape != stiffness.shape or base.shape != damping.shape or base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("invalid dynamics matrices")
    return base + dynamic * stiffness + coefficient * damping


# ---------------------------------------------------------------------------
# Piecewise acquisition.
# ---------------------------------------------------------------------------

def _integration_grids(coefficient_step):
    """Per-piece integration grids aligned to observed 1 kHz samples.

    Returns ``(grids, ratio, dt)`` where each grid covers one piece inclusive
    of both endpoints. ``ratio`` is the number of integration steps per
    observed sample (1 for ``coefficient_step >= .001``). Decimating by
    ``ratio`` recovers exactly the observed samples.
    """
    try:
        coeff = float(coefficient_step)
    except (TypeError, ValueError):
        raise ValueError("coefficient_step must be a positive number")
    if not np.isfinite(coeff) or coeff <= 0:
        raise ValueError("coefficient_step must be positive")
    if coeff >= SAMPLE_DT - 1e-12:
        ratio = 1
        dt = SAMPLE_DT
    else:
        ratio = int(math.ceil(SAMPLE_DT / coeff - 1e-9))
        if ratio < 1:
            ratio = 1
        dt = SAMPLE_DT / ratio
        if dt > coeff * (1.0 + 1e-9):
            raise ValueError("cannot align substeps to 1 kHz samples")
    grids = []
    for i0, i1 in PIECE_INDEX_RANGES:
        if ratio == 1:
            grid = np.arange(i0, i1 + 1, dtype=float) / 1000.0
        else:
            j0, j1 = i0 * ratio, i1 * ratio
            grid = np.arange(j0, j1 + 1, dtype=float) / (1000.0 * ratio)
        grids.append(grid)
    return grids, ratio, dt


def _validate_acquire_args(endpoint, delta, arm, replicate, coefficient_step, full):
    plateau_velocities(endpoint, delta, arm)  # validates endpoint/delta/arm
    if isinstance(replicate, bool) or not isinstance(replicate, (int, np.integer)):
        raise ValueError("replicate must be a non-negative int")
    if int(replicate) < 0:
        raise ValueError("replicate must be a non-negative int")
    try:
        coeff = float(coefficient_step)
    except (TypeError, ValueError):
        raise ValueError("coefficient_step must be positive")
    if not np.isfinite(coeff) or coeff <= 0:
        raise ValueError("coefficient_step must be positive")
    if not isinstance(full, (bool, np.bool_)):
        raise ValueError("full must be bool")


def _acquire_impl(plate, endpoint, delta, arm, replicate, coefficient_step, full):
    """Core piecewise integration; returns ``(acquisition, diagnostics)``."""
    _validate_acquire_args(endpoint, delta, arm, replicate, coefficient_step, full)
    key = _normalize_arm(arm)
    coeff = float(coefficient_step)
    use_full = bool(full)

    try:
        states = plate.initial(int(replicate))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("plate.initial(replicate) must return [reduced_state, full_state]")
    try:
        initial = np.array(states[1] if use_full else states[0], dtype=float, copy=True)
    except (IndexError, TypeError, ValueError):
        raise ValueError("plate.initial(replicate) must return [reduced_state, full_state]")
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("invalid initial state")

    try:
        matrices = plate.full if use_full else plate.reduced
        projection = np.asarray(matrices["projection"], dtype=float)
        input_vector = np.asarray(matrices["force"], dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("plate must expose reduced/full projection/force")
    if projection.ndim != 2 or projection.shape[1] != len(initial):
        raise ValueError("incompatible projection/initial state")
    if input_vector.shape != initial.shape:
        raise ValueError("invalid input vector")

    endpoint_f, delta_f = float(endpoint), float(delta)

    def matrix_at(t):
        # Scalar midpoint from propagate_state; absolute velocity only.
        return absolute_matrix(plate, velocity_at(float(t), endpoint_f, delta_f, key), full=use_full)

    grids, ratio, dt = _integration_grids(coeff)

    n_channels = projection.shape[0]
    pieces_response = []
    pieces_valid = []
    piece_results = []
    boundary_states = [initial.copy()]
    boundary_gaps = []
    state = initial.copy()
    overall_termination = None
    previous_last_obs = None

    for piece_index, grid in enumerate(grids):
        result = propagate_state(
            grid,
            state,
            matrix_at,
            projection,
            # Absolute timestamp subtraction may exceed nominal dt by one ulp.
            coefficient_step=max(coeff, float(grid[1] - grid[0])),
            input_vector=input_vector,
            forcing=None,  # initial-only law; physical_pulse == 0 on [0, .447] (see module docstring).
        )
        piece_results.append(result)
        resp_obs = result.response[:, ::ratio]
        valid_obs = result.valid[::ratio]
        if piece_index > 0 and previous_last_obs is not None and resp_obs.shape[1] >= 1 and previous_last_obs.shape[0] == resp_obs.shape[0]:
            with np.errstate(invalid="ignore"):
                gap = float(np.nanmax(np.abs(resp_obs[:, 0] - previous_last_obs)))
            boundary_gaps.append(gap if np.isfinite(gap) else float("nan"))
        if piece_index == 0:
            pieces_response.append(resp_obs)
            pieces_valid.append(valid_obs)
        else:
            pieces_response.append(resp_obs[:, 1:])
            pieces_valid.append(valid_obs[1:])
        previous_last_obs = resp_obs[:, -1].copy()
        if result.termination is not None:
            overall_termination = result.termination
            state = np.array(result.final_state, dtype=float, copy=True)
            boundary_states.append(state.copy())
            # Fill every remaining piece with missing acquisition.
            for i0_next, i1_next in PIECE_INDEX_RANGES[piece_index + 1 :]:
                n_new = int(i1_next - i0_next)  # first sample duplicates boundary
                pieces_response.append(np.full((n_channels, n_new), np.nan, dtype=float))
                pieces_valid.append(np.zeros(n_new, dtype=bool))
                boundary_states.append(state.copy())
                boundary_gaps.append(float("nan"))
            break
        state = np.array(result.final_state, dtype=float, copy=True)
        boundary_states.append(state.copy())

    response = np.concatenate(pieces_response, axis=1)
    valid = np.concatenate(pieces_valid, axis=0)
    # Defensive: enforce exact 448-sample support even if a plate exposes an
    # unexpected channel count; time base is never altered.
    if response.shape[1] != N_SAMPLES or valid.shape[0] != N_SAMPLES:
        raise RuntimeError("piecewise assembly must retain exactly 448 observed samples")
    times = OBS_TIMES.copy()
    acquisition = StateAcquisition(times, response, state, valid, overall_termination)
    diagnostics = {
        "arm": key,
        "endpoint": endpoint_f,
        "delta": delta_f,
        "replicate": int(replicate),
        "full": use_full,
        "coefficient_step": coeff,
        "integration_dt": float(dt),
        "substep_ratio": int(ratio),
        "piece_index_ranges": [tuple(p) for p in PIECE_INDEX_RANGES],
        "piece_grids": [np.array(g, dtype=float, copy=True) for g in grids],
        "piece_results": list(piece_results),
        "boundary_states": [np.array(s, dtype=float, copy=True) for s in boundary_states],
        "boundary_response_gaps": [float(g) for g in boundary_gaps],
        "n_channels": int(n_channels),
    }
    return acquisition, diagnostics


def acquire(plate, endpoint, delta, arm, replicate=0, coefficient_step=.001, full=False):
    """Acquire the 448-sample S/M/R aperture with continuous state.

    Uses ``plate.initial(replicate)[1 if full else 0]`` as the single initial
    state, integrates the five schedule pieces each with its own aligned grid
    via :func:`propagate_state` (``forcing=None``), passes the same state at
    every boundary, decimates substeps (when ``coefficient_step < .001``) back
    to exactly ``np.arange(448)/1000``, and preserves NaN/``valid``/termination
    behaviour. No state resets; no coefficient block spans a boundary.
    """
    acquisition, _ = _acquire_impl(plate, endpoint, delta, arm, replicate, coefficient_step, full)
    return acquisition


def acquire_with_diagnostics(plate, endpoint, delta, arm, replicate=0, coefficient_step=.001, full=False):
    """Like :func:`acquire` but also returns boundary/replay diagnostics."""
    return _acquire_impl(plate, endpoint, delta, arm, replicate, coefficient_step, full)


def chronology(endpoint, delta):
    """Static schedule table: pieces, sample indices, and per-arm velocities.

    No plate/``ucrit`` access; purely the commanded aperture.
    """
    endpoint_f, delta_f = float(endpoint), float(delta)
    kinds = ("plateau0", "transition0", "plateau1", "transition1", "plateau2")
    pieces = []
    for (i0, i1), kind in zip(PIECE_INDEX_RANGES, kinds):
        pieces.append({"t_start": float(i0 / 1000.0), "t_end": float(i1 / 1000.0),
                       "i_start": int(i0), "i_end": int(i1), "kind": kind})
    return {
        "times": OBS_TIMES.copy(),
        "n_samples": N_SAMPLES,
        "first_to_last_span_s": float(OBS_TIMES[-1] - OBS_TIMES[0]),
        "pieces": pieces,
        "plateau_slices": {k: (int(v.start), int(v.stop)) for k, v in PLATEAU_SLICES.items()},
        "transition_slices": {k: (int(v.start), int(v.stop)) for k, v in TRANSITION_SLICES.items()},
        "plateau_velocities": {arm: plateau_velocities(endpoint_f, delta_f, arm) for arm in ARMS},
        "endpoint": endpoint_f,
        "delta": delta_f,
    }


# ---------------------------------------------------------------------------
# Error metrics.
# ---------------------------------------------------------------------------

def _relative_response_rms(test, ref, test_valid, ref_valid):
    """Relative RMS of ``test`` vs ``ref`` over mutually valid samples.

    Returns ``(rms, n_valid, unit)`` with ``unit = sqrt(mean(ref^2))`` over the
    overlap (floor ``1e-30``). ``rms`` is ``nan`` when there is no overlap.
    Inputs are ``(n_channels, n_times)`` with 1-D time validity masks.
    """
    test = np.asarray(test, dtype=float)
    ref = np.asarray(ref, dtype=float)
    test_valid = np.asarray(test_valid, dtype=bool)
    ref_valid = np.asarray(ref_valid, dtype=bool)
    if test.shape != ref.shape or test_valid.shape != ref_valid.shape or test_valid.ndim != 1 or test.shape[1] != test_valid.shape[0]:
        raise ValueError("mismatched response/validity shapes")
    overlap = test_valid & ref_valid
    n_valid = int(np.count_nonzero(overlap))
    if n_valid == 0:
        return float("nan"), 0, float("nan")
    diff = test[:, overlap] - ref[:, overlap]
    unit = float(np.sqrt(np.mean(ref[:, overlap] ** 2)))
    if not np.isfinite(unit) or unit <= 0:
        return float("nan"), n_valid, float(unit) if np.isfinite(unit) else float("nan")
    unit = max(unit, 1e-30)
    rms = float(np.sqrt(np.mean(diff**2)) / unit)
    return rms, n_valid, unit


def _state_distance(test_state, ref_state):
    """Absolute and relative L2 state distance (same-dimension models only)."""
    test_state = np.asarray(test_state, dtype=float)
    ref_state = np.asarray(ref_state, dtype=float)
    if test_state.shape != ref_state.shape or test_state.ndim != 1:
        raise ValueError("incomparable state dimensions")
    absolute = float(np.linalg.norm(test_state - ref_state))
    denom = max(float(np.linalg.norm(ref_state)), 1e-30)
    return absolute, float(absolute / denom)


def refinement_study(plate, endpoint, delta, arm, replicate=0, steps=REFINEMENT_STEPS, full=False):
    """Step-refinement of one arm against its finest reference.

    The finest step (``min(steps)``) is integrated once and reused for every
    coarser comparison. Each row reports whole-aperture plus per-plateau
    relative RMS and the final-state distance. For ``full=True`` both test
    and reference are full-model states (comparable); ROM comparison across
    model orders belongs in :func:`rom_comparison`.
    """
    key = _normalize_arm(arm)
    step_list = [float(s) for s in steps]
    if len(step_list) < 2:
        raise ValueError("at least two refinement steps required")
    if any(not np.isfinite(s) or s <= 0 for s in step_list):
        raise ValueError("refinement steps must be positive")
    finest = min(step_list)
    reference = acquire(plate, endpoint, delta, key, replicate=replicate, coefficient_step=finest, full=full)
    rows = []
    cache = {}
    for step in step_list:
        if step == finest:
            test = reference
        elif step in cache:
            test = cache[step]
        else:
            test = acquire(plate, endpoint, delta, key, replicate=replicate, coefficient_step=step, full=full)
            cache[step] = test
        whole, n_valid, unit = _relative_response_rms(test.response, reference.response, test.valid, reference.valid)
        per_plateau = {}
        for name, sl in PLATEAU_SLICES.items():
            rms, n_v, _ = _relative_response_rms(
                test.response[:, sl], reference.response[:, sl], test.valid[sl], reference.valid[sl]
            )
            per_plateau[name] = {"rms": rms, "n_valid": int(n_v)}
        try:
            state_abs, state_rel = _state_distance(test.final_state, reference.final_state)
        except ValueError:
            state_abs, state_rel = float("nan"), float("nan")
        rows.append(
            {
                "step": float(step),
                "whole_rms": whole,
                "n_valid_overlap": int(n_valid),
                "reference_unit": unit,
                "per_plateau": per_plateau,
                "final_state_abs": state_abs,
                "final_state_rel": state_rel,
                "test_termination": test.termination,
                "test_n_valid": int(np.count_nonzero(test.valid)),
            }
        )
    return {
        "arm": key,
        "endpoint": float(endpoint),
        "delta": float(delta),
        "replicate": int(replicate),
        "full": bool(full),
        "steps": [float(s) for s in step_list],
        "finest_step": float(finest),
        "reference_termination": reference.termination,
        "reference_n_valid": int(np.count_nonzero(reference.valid)),
        "rows": rows,
    }


def rom_comparison(plate, endpoint, delta, arm, replicate=0, coefficient_step=.00025):
    """Reduced-finest vs full-finest response at one shared step.

    Exactly two integrations (one reduced, one full); no full-model sweep.
    State vectors are dimensionally incomparable, so the "final error" is the
    final observed response column distance (plus its unit-normalized form),
    stated explicitly.
    """
    key = _normalize_arm(arm)
    coeff = float(coefficient_step)
    reduced = acquire(plate, endpoint, delta, key, replicate=replicate, coefficient_step=coeff, full=False)
    full_acq = acquire(plate, endpoint, delta, key, replicate=replicate, coefficient_step=coeff, full=True)
    whole, n_valid, unit = _relative_response_rms(reduced.response, full_acq.response, reduced.valid, full_acq.valid)
    per_plateau = {}
    for name, sl in PLATEAU_SLICES.items():
        rms, n_v, _ = _relative_response_rms(
            reduced.response[:, sl], full_acq.response[:, sl], reduced.valid[sl], full_acq.valid[sl]
        )
        per_plateau[name] = {"rms": rms, "n_valid": int(n_v)}
    overlap = reduced.valid & full_acq.valid
    if np.count_nonzero(overlap) == 0:
        final_abs, final_rel = float("nan"), float("nan")
    else:
        last = int(np.flatnonzero(overlap)[-1])
        if reduced.valid[last] and full_acq.valid[last]:
            diff = reduced.response[:, last] - full_acq.response[:, last]
            final_abs = float(np.sqrt(np.mean(diff**2)))
            final_rel = float(final_abs / max(unit, 1e-30)) if np.isfinite(unit) and unit > 0 else float("nan")
        else:
            final_abs, final_rel = float("nan"), float("nan")
    return {
        "arm": key,
        "endpoint": float(endpoint),
        "delta": float(delta),
        "replicate": int(replicate),
        "coefficient_step": coeff,
        "whole_rms": whole,
        "n_valid_overlap": int(n_valid),
        "full_unit": unit,
        "per_plateau": per_plateau,
        "final_response_abs": final_abs,
        "final_response_rel": final_rel,
        "reduced_termination": reduced.termination,
        "full_termination": full_acq.termination,
        "reduced_n_valid": int(np.count_nonzero(reduced.valid)),
        "full_n_valid": int(np.count_nonzero(full_acq.valid)),
        "note": "state spaces differ (reduced 2*count vs full 2*N); final error is response-space, not state subtraction",
    }


# ---------------------------------------------------------------------------
# Numerical report (feasibility checkpoint, not a validity claim).
# ---------------------------------------------------------------------------

def numerical_report(plate, endpoint, delta, replicate=0):
    """Feasibility convergence report over S/M/R; separates numerics from physics.

    - Reduced refinement at ``REFINEMENT_STEPS`` per arm vs that arm's finest
      (``.00025``) reference: whole + per-plateau RMS and final-state error.
    - Full-vs-reduced comparison per arm ONLY at the finest meaningful step
      (``.00025``): whole + per-plateau RMS and final-response error.
    - Tolerances (existing protocol scales): numerical RMS ``<= .005`` applied
      to the operational default step (``.001``) vs finest; ROM RMS ``<= .02``.
      Any exceedance (or missing overlap) is a reported failure; tolerances
      are never relaxed here.
    - Chronology/support plus continuous-replay evidence (boundary states and
      boundary response agreement proving the same state was passed).
    - Physical-validity section reports supersonic membership and peak cubic
      transition rates WITHOUT claiming they are admissible. A numerical pass
      is not evidence of physical contrast, coverage, or estimator success.
    """
    endpoint_f, delta_f = float(endpoint), float(delta)
    if not np.isfinite(endpoint_f) or not np.isfinite(delta_f) or endpoint_f <= 0 or delta_f < 0:
        raise ValueError("endpoint must be positive finite; delta non-negative finite")
    rep = int(replicate)
    if rep < 0:
        raise ValueError("replicate must be non-negative")

    table = chronology(endpoint_f, delta_f)

    # -- Support: exact grid, sensors, calibration, excitation. ---------------
    times_ok = bool(np.array_equal(np.asarray(table["times"]), np.arange(N_SAMPLES, dtype=float) / 1000.0))
    try:
        n_sensors = int(np.asarray(plate.reduced["projection"]).shape[0])
    except (AttributeError, KeyError, TypeError, ValueError):
        n_sensors = None
    support = {
        "times_equal_arange_448_over_1000": times_ok,
        "n_samples": N_SAMPLES,
        "first_to_last_span_s": float(table["first_to_last_span_s"]),
        "n_sensors": n_sensors,
        "n_sensors_expected": 16,
        "sensors_match": bool(n_sensors == 16) if n_sensors is not None else None,
        "calibration_note": "downstream causal_observation uses leading 51 samples (indices 0:51), wholly inside plateau0 (0:128); caps and 16-sensor handling unchanged",
        "excitation_note": "initial-only law: acquire passes forcing=None with the plate force vector; no pulses added",
        "pulse_note": "existing physical_pulse is nonzero only on [1.00,1.02)+2k s, hence identically zero on [0.000,0.447]; forcing=None reproduces it exactly here",
        "missing_note": "terminated tails stay NaN with valid=False and are never re-qualified",
    }

    # -- Replay evidence at the operational default step. ----------------------
    replay = {}
    max_gap = 0.0
    replay_ok = True
    for arm in ARMS:
        _, diag = _acquire_impl(plate, endpoint_f, delta_f, arm, rep, DEFAULT_STEP, False)
        gaps = [g for g in diag["boundary_response_gaps"] if np.isfinite(g)]
        arm_max = float(max(gaps)) if gaps else float("nan")
        if np.isfinite(arm_max):
            max_gap = max(max_gap, arm_max)
        # Continuity holds by construction (shared state object); the measured
        # gap guards against an accidental reset between pieces.
        continuous = bool(np.isfinite(arm_max) and arm_max <= 1e-9)
        replay_ok = bool(replay_ok and continuous)
        replay[arm] = {
            "coefficient_step": DEFAULT_STEP,
            "n_pieces": len(diag["piece_results"]),
            "boundary_response_gaps": [float(g) for g in diag["boundary_response_gaps"]],
            "max_boundary_gap": arm_max,
            "boundary_state_norms": [float(np.linalg.norm(s)) for s in diag["boundary_states"]],
            "continuous": continuous,
            "piece_terminations": [r.termination for r in diag["piece_results"]],
        }

    # -- Reduced refinement per arm (finest computed once per arm). -----------
    refinement = {}
    numerical_failures = []
    for arm in ARMS:
        study = refinement_study(plate, endpoint_f, delta_f, arm, replicate=rep, steps=REFINEMENT_STEPS, full=False)
        default_row = next(r for r in study["rows"] if r["step"] == DEFAULT_STEP)
        default_rms = default_row["whole_rms"]
        ok = bool(np.isfinite(default_rms) and default_rms <= NUMERICAL_TOL)
        if not ok:
            numerical_failures.append(f"{arm}: default-vs-finest whole RMS {default_rms} exceeds {NUMERICAL_TOL}")
        if study["reference_termination"] is not None:
            numerical_failures.append(f"{arm}: finest reference terminated ({study['reference_termination']})")
            ok = False
        refinement[arm] = {**study, "default_vs_finest_whole_rms": default_rms, "numerical_pass": ok}

    # -- ROM comparison per arm at the finest meaningful step only. -----------
    rom = {}
    rom_failures = []
    for arm in ARMS:
        comp = rom_comparison(plate, endpoint_f, delta_f, arm, replicate=rep, coefficient_step=FINEST_STEP)
        ok = bool(np.isfinite(comp["whole_rms"]) and comp["whole_rms"] <= ROM_TOL)
        if not ok:
            rom_failures.append(f"{arm}: ROM whole RMS {comp['whole_rms']} exceeds {ROM_TOL}")
        if comp["reduced_termination"] is not None or comp["full_termination"] is not None:
            rom_failures.append(
                f"{arm}: ROM comparison has termination (reduced={comp['reduced_termination']}, full={comp['full_termination']})"
            )
            ok = False
        rom[arm] = {**comp, "rom_pass": ok}

    # -- Physical validity (reported, never invented). -------------------------
    try:
        sound = float(plate.design.c_air)
        rho = float(plate.design.rho_air)
    except (AttributeError, TypeError, ValueError):
        sound, rho = float("nan"), float("nan")
    all_velocities = [v for arm in ARMS for v in plateau_velocities(endpoint_f, delta_f, arm)]
    min_v, max_v = float(min(all_velocities)), float(max(all_velocities))
    min_mach = float(min_v / sound) if np.isfinite(sound) and sound > 0 else float("nan")
    peak_rates = {}
    for arm in ARMS:
        v0, v1, v2 = plateau_velocities(endpoint_f, delta_f, arm)
        # Cubic smoothstep peak slope: 1.5*|dv|/T with T=.032 s.
        peak_rates[arm] = {
            "transition0_peak_dvdt": float(1.5 * abs(v1 - v0) / 0.032),
            "transition1_peak_dvdt": float(1.5 * abs(v2 - v1) / 0.032),
        }
    supersonic_ok = bool(np.isfinite(min_mach) and min_mach >= 2.0)
    physical = {
        "rho_air": rho,
        "c_air": sound,
        "min_plateau_velocity": min_v,
        "max_plateau_velocity": max_v,
        "min_mach": min_mach,
        "piston_domain_membership_mach_ge_2": supersonic_ok,
        "peak_transition_rates": peak_rates,
        "note": (
            "Numerical convergence below does not establish that these velocities, "
            "this delta, or the 32 ms cubic rates are physically admissible, produce "
            "operating-point contrast, or support any accuracy target. Rate/domain "
            "admissibility, contrast, coverage, and gate/floor accounting remain open "
            "for the dispatcher/frontier checkpoint. No bound is invented here."
        ),
    }
    if not supersonic_ok:
        rom_failures.append(f"configured piston-theory domain violated: min mach {min_mach} < 2")

    numerical_pass = bool(all(refinement[a]["numerical_pass"] for a in ARMS))
    rom_pass = bool(all(rom[a]["rom_pass"] for a in ARMS))
    overall_pass = bool(numerical_pass and rom_pass and replay_ok and times_ok and supersonic_ok)
    failures = list(numerical_failures) + list(rom_failures)
    if not replay_ok:
        failures.append(f"state-continuity replay gap {max_gap} exceeds 1e-9")
    if not times_ok:
        failures.append("observed support is not exactly np.arange(448)/1000")

    checks = [
        "observed times exactly np.arange(448)/1000 with 447 ms span",
        "five separately integrated pieces with aligned grids; no frozen-coefficient block spans a boundary",
        "single plate.initial(replicate)[reduced/full] state passed unchanged across pieces (boundary-gap replay check)",
        "initial-only excitation (forcing=None); physical_pulse identically zero on aperture",
        "reduced .001-vs-.00025 whole RMS <= .005 per arm (S/M/R)",
        "reduced-finest vs full-finest whole RMS <= .02 per arm at .00025 only",
        "configured piston-theory membership min(plateau velocities)/c_air >= 2 reported, not independent physical validation",
        "missing tails preserved as NaN/valid=False; no re-qualification",
    ]
    tradeoffs = [
        "midpoint-frozen exponentials: exact on plateaus, step-dependent in cubic transitions",
        "substep grids (.0005/.00025) decimated to 1 kHz; .00025 costs ~4x integration steps",
        "full model run only at .00025 (3 runs total); reduced finest reused across comparisons",
        "ROM final error in response space (incompatible state dimensions)",
        "report separates numerical convergence from physical validity; transition-rate admissibility unresolved",
    ]

    return {
        "inputs": {
            "endpoint": endpoint_f,
            "delta": delta_f,
            "replicate": rep,
            "arms": list(ARMS),
            "refinement_steps": list(REFINEMENT_STEPS),
            "default_step": DEFAULT_STEP,
            "finest_step": FINEST_STEP,
        },
        "interfaces": {
            "velocity_at": "velocity_at(time, endpoint, delta, arm)",
            "absolute_matrix": "absolute_matrix(plate, velocity, full=False)",
            "acquire": "acquire(plate, endpoint, delta, arm, replicate=0, coefficient_step=.001, full=False)",
            "numerical_report": "numerical_report(plate, endpoint, delta, replicate=0)",
        },
        "chronology": table,
        "support": support,
        "replay": {"per_arm": replay, "max_boundary_gap": float(max_gap), "continuous": bool(replay_ok)},
        "refinement": refinement,
        "rom": rom,
        "tolerances": {"numerical_rms_max": NUMERICAL_TOL, "rom_rms_max": ROM_TOL, "boundary_gap_max": 1e-9},
        "physical_validity": physical,
        "numerical_pass": numerical_pass,
        "rom_pass": rom_pass,
        "overall_pass": overall_pass,
        "failures": failures,
        "checks": checks,
        "tradeoffs": tradeoffs,
    }
