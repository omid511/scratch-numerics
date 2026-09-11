"""Transient response generation via full state-space propagation.

Given an FSDT solver and a flight velocity, reconstructs the time-domain
transient displacement at selected sensor locations.

Uses the shared eigenanalysis module for consistent eigensolving and filtering.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from mechanics.eigenanalysis import solve_eigenproblem, EigenFilter, TRANSIENT_FILTER, spectral_abscissa
logger = logging.getLogger(__name__)
NYQUIST_MARGIN = 0.90
# Numerical safety for the adaptive-timebase guard (review finding 1):
# dt = MARGIN*pi/omega makes the representability guard an exact equality in
# real arithmetic, so floating-point rounding alone could reject valid modes
# (6,310/100,000 sampled frequencies tripped the unguarded comparison). The
# (1 - _DT_EPS) factor puts dt strictly inside the representable set and the
# guard below carries a matching relative tolerance.
_DT_EPS = 1e-9
def _adaptive_dt(omega_max: float, dt_nominal: float) -> float:
    """Sampling interval fitting omega_max under NYQUIST_MARGIN of Nyquist.
    Returns dt_nominal for empty/non-finite/non-positive bands; the result is
    always strictly inside the representability guard, never on its edge.
    """
    if not np.isfinite(dt_nominal) or dt_nominal <= 0.0:
        raise ValueError(f"Invalid dt_nominal: {dt_nominal}")
    if not np.isfinite(omega_max) or omega_max <= 0.0:
        return dt_nominal
    return min(dt_nominal, NYQUIST_MARGIN * np.pi / omega_max * (1.0 - _DT_EPS))


def _eig_match(eigvals: np.ndarray, target: complex, tol: float = 1e-6) -> bool:
    """Whether ``target`` is represented in ``eigvals``, up to conjugation.
    The label solve maximizes over both ±imag branches, so its critical
    eigenvalue is arbitrary up to conjugation (same physical oscillation,
    distance 2|omega| for the mirror branch). Only a genuinely different
    mode should flag as unrepresented.
    """
    eigvals = np.asarray(eigvals)
    if eigvals.size == 0 or not (np.isfinite(target.real) and np.isfinite(target.imag)):
        return False
    _tol = tol * (1.0 + abs(target))
    _d = np.minimum(np.abs(eigvals - target), np.abs(eigvals - np.conj(target)))
    return bool(np.any(_d <= _tol))


SIGN_TOL = 1e-6  # 1/s deadband: |alpha| below this is marginally stable either way.
SAT_LIMIT = 50.0  # Sensor saturation: post-normalization ADC range (±50× calibration RMS, ~34 dB headroom).
def stability_sign_agrees(label_alpha: float, retained_alpha: float, tol: float = SIGN_TOL) -> bool:
    """Whether label and retained spectra agree on stability sign.
    Review caveat 1: :func:`_eig_match` is spectrum matching whose tolerance
    scales with the full complex eigenvalue (frequency-dominated), so
    opposite-sign growth rates at matched frequency still match. This
    compares signs separately: both stable, both unstable, or either inside
    the near-zero deadband (marginal — sign numerically undecidable, counted
    as agree). NaN (unresolved label solve) disagrees.
    """
    if not (np.isfinite(label_alpha) and np.isfinite(retained_alpha)):
        return False
    if abs(label_alpha) <= tol or abs(retained_alpha) <= tol:
        return True
    return bool(np.sign(label_alpha) == np.sign(retained_alpha))

def _label_path_consistency(M_mat, K_total, C_total, eigvals, velocity):
    """Replicate the flutter-label contract at one velocity and compare.
    Runs ``spectral_abscissa`` with the label gates (eta_w_min=1e-3, no
    oscillatory/band restriction — as in ``_max_real_eigenvalue``) and checks
    the label-driving mode against the retained signal spectrum.
    One extra dense solve per cached velocity (~17% generation overhead);
    an unresolved label solve warns and returns (None, nan, False) rather
    than killing generation, mirroring the flutter scan's tolerance.
    Returns (label_crit|None, label_alpha, label_represented).
    """
    try:
        _label = spectral_abscissa(M_mat, K_total, C_total, eta_w_min=1e-3)
        label_crit: complex | None = complex(_label.critical_eigenvalue)
        label_alpha = float(_label.alpha)
        return label_crit, label_alpha, _eig_match(eigvals, label_crit)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        logger.warning("Label-path abscissa unresolved at V=%.1f: %s", velocity, exc)
        return None, float("nan"), False


# NOTE: a previous `id(solver)`-keyed flutter-velocity cache lived here. It was
# removed: `id()` keys collide after garbage collection and the key ignored
# rho/c_sound/zeta/velocity bounds, so reviving it would return stale
# velocities. Callers compute `u_crit` explicitly per design/condition.


@dataclass
class Eigendecomposition:
    """Cached eigendecomposition for a single velocity."""
    eigvals: np.ndarray
    eigvecs: np.ndarray
    size: int
    MN_eff: int
    velocity: float
    M_mat: np.ndarray
    K_total: np.ndarray
    C_total: np.ndarray
    sensor_iy: np.ndarray
    sensor_ix: np.ndarray
    vx_grid: np.ndarray
    vy_grid: np.ndarray
    M_eff: int
    N_eff: int
    rho: float
    c_sound: float
    zeta: float
    sensor_modes: np.ndarray | None = None  # P2-2: cached (n_sensors, n_modes) projection
    # P2-3: adaptive sampling time base chosen by compute_eigendecomposition
    # so the retained mode band fits under 90% of Nyquist. generate_clip uses
    t_span: tuple = (0.0, 0.5)
    dt: float | None = None
    # Label-path consistency (review finding 3): critical eigenvalue from the
    # same full-spectrum spectral-abscissa contract the flutter label uses
    # (eta_w_min=1e-3, no oscillatory/band restriction), evaluated at the clip
    # velocity, plus whether it is represented in the retained signal modes.
    # None/nan when the label-path solve itself is unresolved.
    label_crit: complex | None = None
    label_alpha: float = float("nan")
    label_represented: bool = False


@dataclass
class TransientClip:
    """One transient clip at a single velocity."""
    sensor_signals: np.ndarray   # (n_sensors, n_timesteps)
    time: np.ndarray             # (n_timesteps,)
    velocity: float
    u_crit: float | None
    margin: float                # (u_crit - velocity) / u_crit  or NaN
    eigenvalues: np.ndarray      # complex, (n_modes,)
    sensor_xy: np.ndarray        # (n_sensors, 2) — grid indices [iy, ix]
    design_id: str = ""          # unique ID for grouped splitting
    # Provenance (review §4.1/4.2/4.4): sampling + saturation + critical mode.
    # Defaults preserve backward compatibility with older call sites/tests.
    dt: float | None = None             # seconds per sample (time[1]-time[0])
    duration: float | None = None       # physical clip length in seconds
    clamp_frac: float = 0.0             # fraction of (mode,time) growth args hitting clamp bounds
    max_log_amp: float = 0.0            # max Re(lambda*t) before clamping
    critical_idx: int = 0               # index into eigenvalues of most-dangerous oscillatory mode
    alpha: float = 0.0                  # spectral abscissa over retained modes (max Re)
    omega_crit: float = 0.0             # |Im| of the critical mode
    # Label-path consistency (review finding 3): abscissa + critical mode from
    # the flutter-label contract at the clip velocity. A label_represented of
    # False means margin zero is driven by a mode absent from the signal.
    label_alpha: float = float("nan")
    label_omega: float = float("nan")   # |Im| of the label-path critical mode
    label_represented: bool = False
    sat_frac: float = 0.0               # fraction of samples hitting ±SAT_LIMIT


def default_sensor_xy(ny: int, nx: int, n_sensors: int = 8) -> np.ndarray:
    """Generate fixed interior sensor positions."""
    iy_vals = np.linspace(1, ny - 2, int(np.ceil(np.sqrt(n_sensors))), dtype=int)
    ix_vals = np.linspace(1, nx - 2, int(np.ceil(np.sqrt(n_sensors))), dtype=int)
    iy_grid, ix_grid = np.meshgrid(iy_vals, ix_vals)
    coords = np.column_stack([iy_grid.ravel(), ix_grid.ravel()])[:n_sensors]
    return coords


def most_dangerous_oscillatory_mode(eigvals):
    oscillatory = np.flatnonzero(eigvals.imag > 0.0)
    if oscillatory.size == 0:
        raise RuntimeError("No positive-imaginary oscillatory mode is available")
    return int(oscillatory[np.argmax(eigvals[oscillatory].real)])


def sample_modal_initial_conditions(eigvals, rng):
    n_modes = eigvals.size
    amplitudes = rng.lognormal(mean=0.0, sigma=0.5, size=n_modes)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
    amplitudes /= max(np.linalg.norm(amplitudes), np.finfo(float).tiny)
    coefficients = amplitudes * np.exp(1j * phases)
    return amplitudes, coefficients


def validate_sensor_indices(sensor_iy, sensor_ix, *, grid_shape):
    pairs = list(zip(sensor_iy.tolist(), sensor_ix.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"Duplicate sensor grid locations: {pairs}")
    ny, nx = grid_shape
    for iy, ix in pairs:
        if not (0 <= iy < ny and 0 <= ix < nx):
            raise ValueError(f"Sensor index out of bounds: {(iy, ix)}")


def causal_calibration_normalize(signals, calibration_samples=64, eps=1e-8,
                                 normalize_mode="per_channel"):
    """Causally normalize from the leading calibration window.

    `normalize_mode="per_channel"` (default, legacy): per-channel offset and
    scale. `normalize_mode="global"`: per-channel offsets but a SINGLE global
    scale from the whole calibration block, preserving inter-channel
    amplitude ratios (flutter mode-shape cue) that per-channel scaling
    erases. Both modes use only the leading window (causal).
    """
    if signals.ndim != 2:
        raise ValueError(f"Expected (channels, time), got {signals.shape}")
    if not 1 <= calibration_samples <= signals.shape[1]:
        raise ValueError("Invalid calibration_samples")
    if normalize_mode not in ("per_channel", "global"):
        raise ValueError(f"Unknown normalize_mode: {normalize_mode!r}")
    calibration = signals[:, :calibration_samples]
    offset = calibration.mean(axis=1, keepdims=True)
    if normalize_mode == "global":
        scale = np.sqrt(np.mean((calibration - offset) ** 2))
        scale = max(float(scale), eps)
        return (signals - offset) / scale
    scale = np.sqrt(np.mean((calibration - offset) ** 2, axis=1, keepdims=True))
    scale = np.maximum(scale, eps)
    return (signals - offset) / scale


def modal_exponentials(eigvals, time, *, max_real_exponent=50.0):
    exponents = eigvals[:, None] * time[None, :]
    maximum = float(np.max(exponents.real))
    if maximum > max_real_exponent:
        raise FloatingPointError(
            f"Transient exceeds configured dynamic range: max Re(lambda*t)={maximum:.3f}"
        )
    values = np.exp(exponents)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite modal exponential")
    return values


def compute_eigendecomposition(
    solver,
    velocity: float,
    n_modes: int = 20,
    n_sensors: int = 8,
    sensor_xy: np.ndarray | None = None,
    t_span: tuple = (0.0, 0.5),
    n_timesteps: int = 512,
    rho: float = 1.2,
    c_sound: float = 340.0,
    zeta: float = 0.0,
    filt: EigenFilter = TRANSIENT_FILTER,
    min_modes: int = 2,
    nyquist_strict: bool = True,
) -> Eigendecomposition:
    """Solve the eigenvalue problem once and cache the result.

    Uses the shared solve_eigenproblem for consistent filtering.
    Raises RuntimeError if fewer than min_modes modes pass filtering.
    """
    from mechanics.piston_theory import validate_mach

    M_inf = velocity / c_sound
    validate_mach(M_inf, strict_high_mach=True, min_mach=2.0)

    M_mat, K_total, C_total = solver.assemble_aeroelastic_system(
        velocity, rho, c_sound, zeta,
    )

    # P2-3: ADAPTIVE sampling window — determine the retained modes' frequency
    # band FIRST, then choose dt so that 90% of Nyquist covers it. The output
    # series LENGTH (n_timesteps) is unchanged; only dt (and hence the clip
    # duration t1 - t0) adapts.
    t0, t1 = map(float, t_span)
    if t1 <= t0:
        raise ValueError(f"Invalid t_span: {t_span}")
    if n_timesteps < 2:
        raise ValueError("n_timesteps must be at least 2")
    dt_nominal = (t1 - t0) / n_timesteps

    result = solve_eigenproblem(
        M_mat, K_total, C_total,
        filt=filt,
        require_positive_imag=True,
        n_modes=0,
    )

    if len(result.eigvals) > 0:
        cand_vals, cand_vecs = result.eigvals, result.eigvecs
    else:
        # The unscaled linearized pencil mixes O(1) identity blocks with
        # O(1e14) stiffness blocks, so BOTH its eigenpairs AND the unscaled
        # residual metric are unreliable: solve_eigenproblem may retain zero
        # modes purely for conditioning reasons. Redo the eigensolve on the
        # DYNAMICALLY SCALED pencil (same fix as spectral_abscissa): gamma =
        # sqrt(||K||_F/||M||_F), K_t=K/gamma^2, C_t=C/gamma; eigenvalues map
        # back via s = gamma*s_hat, eigenvector layout is unchanged.
        from mechanics.eigenanalysis import (
            _qep_backward_error, _scale_qep, _transverse_participation,
        )

        logger.warning(
            "solve_eigenproblem retained 0 modes at V=%.1f (unscaled residual "
            "gate); re-solving the dynamically scaled pencil", velocity,
        )
        size_all = result.size
        gamma_s, M_t, K_t, C_t = _scale_qep(M_mat, K_total, C_total)
        Z = np.zeros((size_all, size_all))
        I = np.eye(size_all)
        A_s = np.block([[Z, I], [-K_t, -C_t]])
        B_s = np.block([[I, Z], [Z, M_t]])
        w_hat, V_hat = linalg.eig(A_s, B_s)

        ok = np.isfinite(w_hat) & (w_hat.imag > 0.0)
        idx_ok = np.flatnonzero(ok)
        if idx_ok.size == 0:
            raise RuntimeError(f"No physical modes found at V={velocity:.1f}")
        res_scaled = _qep_backward_error(
            w_hat[idx_ok], V_hat[:size_all, idx_ok], M_t, K_t, C_t,
        )
        # Conditioning-aware tolerance floor: even well-computed eigenvectors
        # of a damping-dominated stiff pencil carry scaled backward error
        # ~1e-7..1e-6, so the filter's nominal 1e-7 would reject everything
        # for numerical rather than physical reasons.
        res_tol = max(filt.residual_max, 1e-5)
        keep = idx_ok[res_scaled < res_tol]
        # Apply the participation part of the filter that the primary path
        # enforces: without it, zero-transverse-content constraint artifacts
        # (eta_w == 0) occupy mode slots and can win the 'most dangerous
        # oscillatory mode' selection on numerical dust alone.
        if filt.eta_w_min > 0.0:
            eta = _transverse_participation(V_hat[:size_all, keep], size_all)
            physical = eta >= filt.eta_w_min
        else:
            physical = np.ones(keep.size, dtype=bool)
        cand_vals = gamma_s * w_hat[keep[physical]]
        cand_vecs = V_hat[:size_all, keep[physical]]
        # Apply the frequency-band part of the filter that still makes sense.
        om_lo = filt.omega_min if filt.omega_min is not None else 0.0
        om_hi = filt.omega_max if filt.omega_max is not None else np.inf
        band = (np.abs(cand_vals.imag) >= om_lo) & (np.abs(cand_vals.imag) <= om_hi)
        cand_vals, cand_vecs = cand_vals[band], cand_vecs[:, band]
    # Review §4.1: select the RETAINED modes first, then set the timebase from
    # the retained band only. The previous code derived dt from all candidates,
    # so high-frequency modes that never entered the signal could shorten the
    # window and hide slow near-flutter growth.
    if cand_vals.size == 0:
        raise RuntimeError(f"No candidate modes at V={velocity:.1f}")
    # Sort candidates by frequency; critical (most-dangerous oscillatory) first.
    freq_order = np.argsort(np.abs(cand_vals.imag))
    critical_cand = most_dangerous_oscillatory_mode(cand_vals)
    keep_cand = [int(critical_cand)]
    for idx in (int(i) for i in freq_order):
        if idx not in keep_cand:
            keep_cand.append(idx)
        if len(keep_cand) >= n_modes:
            break
    keep_cand = np.array(keep_cand[:n_modes])
    retained_vals = cand_vals[keep_cand]
    retained_vecs = cand_vecs[:, keep_cand]
    # Timebase from the retained band: dt adapts so retained modes fit under
    # 90% of Nyquist. Retained modes are never silently dropped for Nyquist
    # reasons — dt moves instead.
    omega_ret = np.abs(retained_vals.imag)
    omega_max = float(omega_ret.max()) if omega_ret.size else 0.0
    dt = _adaptive_dt(omega_max, dt_nominal)
    nyquist_omega = np.pi / dt
    t1_adapted = t0 + n_timesteps * dt
    if bool(np.any(omega_ret > NYQUIST_MARGIN * nyquist_omega * (1.0 + _DT_EPS))):
        raise RuntimeError(
            f"Retained band exceeds {NYQUIST_MARGIN:.0%} of Nyquist ({nyquist_omega:.1f} rad/s) at "
            f"V={velocity:.1f} even after adaptive dt selection"
        )
    # Reorder retained: critical mode first, then remaining by frequency.
    freq_order_ret = np.argsort(retained_vals.imag)
    critical_pos = int(np.flatnonzero(keep_cand == int(critical_cand))[0])
    order = [critical_pos] + [i for i in (int(j) for j in freq_order_ret) if i != critical_pos]
    order = np.array(order)
    eigvals = retained_vals[order]
    eigvecs = retained_vecs[:, order]
    # Critical mode is index 0 by construction.
    critical_idx = 0
    if len(eigvals) < min_modes:
        raise RuntimeError(
            f"Only {len(eigvals)} physical modes found at V={velocity:.1f} "
            f"(need {min_modes})"
        )
    # Review finding 3: replicate the label-path contract (full validated
    # spectrum, eta_w_min=1e-3, no oscillatory/band restriction — the same
    # gates find_flutter_velocity uses via _max_real_eigenvalue) at the clip
    # velocity, and check the label-driving mode is represented in the
    # retained signal modes. A static-divergence (imag≈0) or out-of-band
    # driver crosses margin zero with no observable transient signature;
    # that must be measured per clip, not assumed from shared infrastructure.
    label_crit, label_alpha, label_represented = _label_path_consistency(
        M_mat, K_total, C_total, eigvals, velocity)

    # Sensor locations
    ny = solver.grid[1]
    nx = solver.grid[0]
    if sensor_xy is not None:
        sensor_iy = sensor_xy[:, 0]
        sensor_ix = sensor_xy[:, 1]
    else:
        coords = default_sensor_xy(ny, nx, n_sensors)
        sensor_iy = coords[:, 0]
        sensor_ix = coords[:, 1]

    validate_sensor_indices(sensor_iy, sensor_ix, grid_shape=(ny, nx))

    # P2-2: Precompute sensor-mode projection matrix
    # Project basis-space w-DOF eigenvectors to sensor locations using basis evaluation
    n_sensors_actual = len(sensor_iy)
    n_phys = len(eigvals)
    M_eff = solver._vx_grid.shape[0]
    N_eff = solver._vy_grid.shape[0]
    w_start_phys = 2 * result.MN_eff
    sensor_modes = np.zeros((n_sensors_actual, n_phys), dtype=complex)
    # w-DOFs are in basis space: (MN_eff, n_phys) -> (M_eff, N_eff, n_phys)
    w_vecs = eigvecs[w_start_phys:w_start_phys + result.MN_eff, :]
    w_3d = w_vecs.reshape(M_eff, N_eff, n_phys)
    for s_idx in range(n_sensors_actual):
        iy, ix = sensor_iy[s_idx], sensor_ix[s_idx]
        proj_vx = solver._vx_grid[:, ix]  # (M_eff,)
        proj_vy = solver._vy_grid[:, iy]  # (N_eff,)
        sensor_modes[s_idx] = np.einsum('i,ijk,j->k', proj_vx, w_3d, proj_vy)

    return Eigendecomposition(
        eigvals=eigvals,
        eigvecs=eigvecs,
        size=result.size,
        MN_eff=result.MN_eff,
        velocity=velocity,
        M_mat=M_mat,
        K_total=K_total,
        C_total=C_total,
        sensor_iy=sensor_iy,
        sensor_ix=sensor_ix,
        vx_grid=solver._vx_grid,
        vy_grid=solver._vy_grid,
        M_eff=solver._vx_grid.shape[0],
        N_eff=solver._vy_grid.shape[0],
        rho=rho,
        c_sound=c_sound,
        zeta=zeta,
        sensor_modes=sensor_modes,
        t_span=(t0, t1_adapted),
        dt=dt,
        label_crit=label_crit,
        label_alpha=label_alpha,
        label_represented=label_represented,
    )


# Clamp for unstable growth: exp(20) ~ 4.9e8 keeps ordering/finite in float64
# while exp(>709) overflows. Floor -50 keeps decaying modes finite.
_MAX_LOG_AMP = 20.0
_MIN_LOG_AMP = -50.0


def _clamped_modal_response(eigvals, t, coeffs):
    """Modal responses with real-part-clamped growth (finite, order-preserving)."""
    real_arg = np.clip(eigvals.real[:, None] * t[None, :], _MIN_LOG_AMP, _MAX_LOG_AMP)
    osc = np.exp(1j * eigvals.imag[:, None] * t[None, :])
    return coeffs[:, None] * np.exp(real_arg) * osc


def generate_clip_from_eigendecomposition(
    eigs: Eigendecomposition,
    rng: np.random.Generator,
    t_span: tuple | None = None,
    n_timesteps: int = 512,
    normalize_window_frac: float = 0.1,
    u_crit: float | None = None,
    design_id: str | None = None,
    normalize_mode: str = "per_channel",
) -> TransientClip:
    """Generate one clip from cached eigendecomposition with random ICs.

    Uses log-normal modal amplitudes so all modes receive meaningful
    excitation with independent random initial conditions.

    ``t_span=None`` (default) uses the ADAPTIVE time base stored on the
    Eigendecomposition by compute_eigendecomposition, so the sampled grid
    always matches the dt chosen for the retained mode band. Passing an
    explicit t_span keeps the legacy behavior.

    ``design_id=None`` (default) falls back to the legacy ``v<velocity>``
    label for single-solver scripts. Multi-design callers must pass the
    nominal design ID so grouped splits group by design, not velocity.
    """
    if t_span is None:
        t_span = getattr(eigs, "t_span", None) or (0.0, 0.5)
    eigvals = eigs.eigvals
    eigvecs = eigs.eigvecs
    size = eigs.size
    MN_eff = eigs.MN_eff
    w_start = 2 * MN_eff

    n_phys = len(eigvals)

    # Independent random modal amplitudes (no stability-informed boosting)
    amplitudes, coeffs_modal = sample_modal_initial_conditions(eigvals, rng)

    t = np.linspace(t_span[0], t_span[1], n_timesteps, endpoint=False)

    # Propagate using positive-imaginary eigenvalues only
    # Each selected mode has Im(s) > 0, contribution is 2*Re[c_k * v_k * exp(s_k * t)]
    # Overflow control: clamp the REAL growth argument to [_MIN_LOG_AMP,
    # _MAX_LOG_AMP] so supercritical (negative-margin) clips stay finite and
    # ordered instead of overflowing to inf and being rejected (which biased
    # the dataset toward stable clips). Oscillation phase is unclamped.
    # Review §4.2: record saturation incidence — clamped trajectories are not
    # exact linear responses and must be auditable per clip.
    real_arg = eigvals.real[:, None] * t[None, :]
    max_log_amp = float(np.max(real_arg)) if real_arg.size else 0.0
    min_log_amp = float(np.min(real_arg)) if real_arg.size else 0.0
    clamp_frac = float(np.mean((real_arg > _MAX_LOG_AMP) | (real_arg < _MIN_LOG_AMP))) if real_arg.size else 0.0
    # Map w-DOFs to grid at sensor locations (P2-2: use cached sensor_modes).
    # Review §6.1: skip the full-field w_all reconstruction on the cached path —
    # it was built then discarded. Only the uncached fallback needs it.
    n_sensors = len(eigs.sensor_iy)
    if eigs.sensor_modes is not None:
        # Use cached projection: signals = sensor_modes @ modal_response
        # (growth-clamped; see above).
        modal_response = _clamped_modal_response(eigvals, t, coeffs_modal)  # (n_phys, n_t)
        sensor_signals = 2.0 * np.real(eigs.sensor_modes @ modal_response)
    else:
        w_all = np.zeros((MN_eff, n_timesteps))
        for k in range(n_phys):
            lam_k = eigvals[k]
            v_k = eigvecs[:size, k]
            c_k = coeffs_modal[k]
            exp_t = np.exp(np.clip(lam_k.real * t, _MIN_LOG_AMP, _MAX_LOG_AMP))
            cos_t = np.cos(lam_k.imag * t)
            sin_t = np.sin(lam_k.imag * t)
            w_v_real = v_k[w_start:w_start + MN_eff].real
            w_v_imag = v_k[w_start:w_start + MN_eff].imag
            c_real = c_k.real
            c_imag = c_k.imag
            real_part = c_real * w_v_real - c_imag * w_v_imag
            imag_part = -c_imag * w_v_real - c_real * w_v_imag
            contrib = 2.0 * (np.outer(real_part, cos_t * exp_t) + np.outer(imag_part, sin_t * exp_t))
            w_all += contrib
        sensor_signals = np.zeros((n_sensors, n_timesteps))
        w_3d = w_all.reshape(eigs.M_eff, eigs.N_eff, n_timesteps)
        for s_idx in range(n_sensors):
            iy, ix = eigs.sensor_iy[s_idx], eigs.sensor_ix[s_idx]
            proj_vy = eigs.vy_grid[:, iy]
            proj_vx = eigs.vx_grid[:, ix]
            sensor_signals[s_idx] = np.einsum('i,ijt,j->t', proj_vx, w_3d, proj_vy)

    # Causal normalization using fixed calibration window. Minimum two
    # samples: a single-sample window has zero variance, forcing the eps
    # floor and amplifying the clip ~1e8× (found via stub tests at
    # n_timesteps=16; production uses 51).
    calibration_samples = max(2, int(n_timesteps * normalize_window_frac))
    sensor_signals = causal_calibration_normalize(
        sensor_signals, calibration_samples=calibration_samples,
        normalize_mode=normalize_mode)
    # Sensor saturation: a growing transient normalized by its quiet prefix
    # can exceed any physical ADC range (e.g. 2e8× calibration RMS). Clamp to
    # ±SAT_LIMIT and record incidence — same family as growth clamping (§4.2).
    sat_frac = float(np.mean(np.abs(sensor_signals) > SAT_LIMIT))
    if sat_frac > 0.0:
        sensor_signals = np.clip(sensor_signals, -SAT_LIMIT, SAT_LIMIT)
    # Reject non-finite clips
    if not np.isfinite(sensor_signals).all():
        raise RuntimeError("Non-finite sensor signal after propagation")

    # Reject all-zero clips
    if np.all(sensor_signals == 0):
        raise RuntimeError("All-zero sensor signal after propagation")
    # Margin
    margin = float("nan")
    if u_crit is not None and u_crit > 0:
        margin = (u_crit - eigs.velocity) / u_crit
    if design_id is None:
        design_id = f"v{eigs.velocity:.0f}"
    dt_clip = float(t[1] - t[0]) if len(t) > 1 else (float(eigs.dt) if eigs.dt else 0.0)
    duration = float(t[-1] - t[0] + dt_clip) if len(t) else 0.0
    alpha = float(np.max(eigvals.real)) if len(eigvals) else 0.0
    try:
        crit = int(most_dangerous_oscillatory_mode(eigvals))
    except RuntimeError:
        crit = 0
    omega_crit = float(abs(eigvals[crit].imag)) if len(eigvals) else 0.0
    return TransientClip(
        sensor_signals=sensor_signals,
        time=t,
        velocity=eigs.velocity,
        u_crit=u_crit,
        margin=margin,
        eigenvalues=eigvals,
        sensor_xy=np.column_stack([eigs.sensor_iy, eigs.sensor_ix]),
        design_id=design_id,
        dt=dt_clip,
        duration=duration,
        clamp_frac=float(clamp_frac),
        max_log_amp=float(max_log_amp),
        critical_idx=int(crit),
        alpha=float(alpha),
        omega_crit=float(omega_crit),
        label_alpha=float(getattr(eigs, "label_alpha", float("nan"))),
        label_omega=float(abs(eigs.label_crit.imag)) if getattr(eigs, "label_crit", None) is not None else float("nan"),
        label_represented=bool(getattr(eigs, "label_represented", False)),
        sat_frac=float(sat_frac),
    )


def generate_transient_clips_batch(
    solver,
    velocity: float,
    n_realizations: int = 5,
    n_modes: int = 20,
    n_sensors: int = 8,
    t_span: tuple | None = None,
    n_timesteps: int = 512,
    seed: int | None = None,
    normalize_window_frac: float = 0.1,
    sensor_xy: np.ndarray | None = None,
    u_crit: float | None = None,
    rho: float = 1.2,
    c_sound: float = 340.0,
    zeta: float = 0.0,
    nyquist_strict: bool = True,
    design_id: str | None = None,
    normalize_mode: str = "per_channel",
) -> list[TransientClip]:
    """Generate multiple clips at the same velocity with different random ICs.

    Calls compute_eigendecomposition ONCE (expensive) and reuses the
    eigendecomposition for all realizations.

    ``t_span=None`` (default) uses the adaptive eigendecomposition window
    for the clip grid; pass an explicit window for legacy fixed-window
    behavior. ``design_id`` is forwarded to each clip (default: velocity
    label).
    """
    eigs = compute_eigendecomposition(
        solver, velocity, n_modes, n_sensors, sensor_xy,
        t_span=t_span if t_span is not None else (0.0, 0.5),
        n_timesteps=n_timesteps,
        rho=rho, c_sound=c_sound, zeta=zeta,
        nyquist_strict=nyquist_strict,
    )
    rng = np.random.default_rng(seed)
    clips = []
    for _ in range(n_realizations):
        clip = generate_clip_from_eigendecomposition(
            eigs, rng, t_span, n_timesteps, normalize_window_frac, u_crit,
            design_id=design_id, normalize_mode=normalize_mode,
        )
        clips.append(clip)
    return clips


def generate_transient_clip(
    solver,
    velocity: float,
    t_span: tuple[float, float] | None = None,
    n_timesteps: int = 512,
    n_sensors: int = 8,
    n_modes: int = 20,
    rng: np.random.Generator | None = None,
    normalize_window_frac: float = 0.1,
    sensor_xy: np.ndarray | None = None,
    u_crit: float | None = None,
    rho: float = 1.2,
    c_sound: float = 340.0,
    zeta: float = 0.0,
    nyquist_strict: bool = True,
    design_id: str | None = None,
    normalize_mode: str = "per_channel",
) -> TransientClip:
    """Generate a single transient clip via full state-space propagation.

    Parameters
    ----------
    solver : FSDTSolver
    velocity : float  — flight velocity (m/s)
    t_span : (t_start, t_end) seconds, or None for the adaptive window
    n_timesteps : number of time samples
    n_sensors : how many grid points to sample
    n_modes : modes to keep from eigen-solve
    rng : random generator for reproducibility
    normalize_window_frac : fraction of clip used for causal normalization
    sensor_xy : optional (n_sensors, 2) array of [iy, ix] grid indices
    rho : air density (kg/m^3)
    c_sound : speed of sound (m/s)
    zeta : structural damping ratio

    Returns
    -------
    TransientClip
    """
    if rng is None:
        rng = np.random.default_rng()

    eigs = compute_eigendecomposition(
        solver, velocity, n_modes, n_sensors, sensor_xy,
        t_span=t_span if t_span is not None else (0.0, 0.5),
        n_timesteps=n_timesteps,
        rho=rho, c_sound=c_sound, zeta=zeta,
        nyquist_strict=nyquist_strict,
    )

    return generate_clip_from_eigendecomposition(
        eigs, rng, t_span, n_timesteps, normalize_window_frac, u_crit,
        design_id=design_id, normalize_mode=normalize_mode,
    )


def generate_dataset(
    solver,
    n_samples: int = 100,
    velocity_range: tuple[float, float] = (400.0, 800.0),
    t_span: tuple[float, float] | None = None,
    n_timesteps: int = 512,
    n_sensors: int = 8,
    n_modes: int = 20,
    seed: int = 42,
    sensor_xy: np.ndarray | None = None,
    u_crit: float | None = None,
    nyquist_strict: bool = True,
    design_id: str | None = None,
    normalize_mode: str = "per_channel",
) -> list[TransientClip]:
    """Generate a dataset of transient clips across random velocities.

    Returns a list of TransientClip objects.
    """
    rng = np.random.default_rng(seed)
    velocities = rng.uniform(*velocity_range, size=n_samples)
    clips = []
    for v in velocities:
        try:
            clip = generate_transient_clip(
                solver, float(v), t_span, n_timesteps, n_sensors, n_modes, rng=rng,
                sensor_xy=sensor_xy, u_crit=u_crit, nyquist_strict=nyquist_strict,
                design_id=design_id, normalize_mode=normalize_mode,
            )
            clips.append(clip)
        except (ValueError, AttributeError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as e:
            logger.debug("clip generation failed at v=%.1f: %s", v, e)
            continue
    return clips
