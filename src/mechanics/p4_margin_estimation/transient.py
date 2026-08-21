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

from mechanics.eigenanalysis import solve_eigenproblem, EigenFilter, TRANSIENT_FILTER

logger = logging.getLogger(__name__)

_u_crit_cache: dict[int, float | None] = {}


def _get_u_crit(solver, rho, c_sound, zeta,
                v_lower=None, v_upper=3000.0) -> float | None:
    """Cache flutter velocity by solver identity."""
    if v_lower is None:
        v_lower = 2.0 * c_sound * 1.01  # M >= 2.0 lower bound
    key = id(solver)
    if key not in _u_crit_cache:
        _u_crit_cache[key] = solver.find_flutter_velocity(
            rho=rho, c_sound=c_sound, zeta=zeta,
            v_lower=v_lower, v_upper=v_upper,
            n_scan=10, velocity_tol=5.0,
        )
    return _u_crit_cache[key]


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


def causal_calibration_normalize(signals, calibration_samples=64, eps=1e-8):
    if signals.ndim != 2:
        raise ValueError(f"Expected (channels, time), got {signals.shape}")
    if not 1 <= calibration_samples <= signals.shape[1]:
        raise ValueError("Invalid calibration_samples")
    calibration = signals[:, :calibration_samples]
    offset = calibration.mean(axis=1, keepdims=True)
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

    # Modes above Nyquist (π/dt) will alias in the sampled output.
    # Hard-fail if retained modes exceed Nyquist — oversampling is the remedy.
    t0, t1 = map(float, t_span)
    if t1 <= t0:
        raise ValueError(f"Invalid t_span: {t_span}")
    if n_timesteps < 2:
        raise ValueError("n_timesteps must be at least 2")
    dt = (t1 - t0) / n_timesteps
    nyquist_omega = np.pi / dt

    result = solve_eigenproblem(
        M_mat, K_total, C_total,
        filt=filt,
        require_positive_imag=True,
        n_modes=0,
    )

    # Nyquist filter: reject modes that cannot be represented
    omega = np.abs(result.eigvals.imag)
    representable = omega <= 0.90 * nyquist_omega

    filtered_eigvals = result.eigvals[representable]
    filtered_eigvecs = result.eigvecs[:, representable]

    if filtered_eigvals.size == 0:
        msg = (
            f"No modes below 90% of Nyquist ({nyquist_omega:.1f} rad/s) at V={velocity:.1f}; "
            f"increase n_timesteps or reduce t_span"
        )
        if nyquist_strict:
            raise ValueError(msg)
        else:
            # P0-9: result.eigvals was post-TRANSIENT_FILTER which already killed
            # them. Fall back to eigvals_all but apply minimal physical filtering:
            # positive imaginary part, basic residual, and relaxed Nyquist bound.
            logger.warning("%s — falling back to all finite eigenvalues (may alias)", msg)
            phys_mask = result.eigvals_all.imag > 0
            phys_mask &= result.residuals_all < 0.1 if hasattr(result, 'residuals_all') else True
            if phys_mask.sum() == 0:
                raise RuntimeError(msg)
            filtered_eigvals = result.eigvals_all[phys_mask]
            filtered_eigvecs = result.eigvecs_all[:, phys_mask]

    if filtered_eigvals.size < n_modes:
        logger.warning(
            "Only %d modes below Nyquist (requested %d); using all %d",
            filtered_eigvals.size, n_modes, filtered_eigvals.size,
        )
        n_modes = filtered_eigvals.size

    # Ensure the critical flutter mode is always included
    n_phys = len(filtered_eigvals)
    if n_phys == 0:
        raise RuntimeError(
            f"No physical modes found at V={velocity:.1f}"
        )

    # Sort by frequency for the main set
    freq_order = np.argsort(filtered_eigvals.imag)

    # Identify the most dangerous oscillatory mode
    critical_idx = most_dangerous_oscillatory_mode(filtered_eigvals)

    # Build keep list: critical mode first, then fill by frequency
    keep = [critical_idx]
    for idx in freq_order:
        if idx not in keep:
            keep.append(idx)
        if len(keep) >= n_modes or len(keep) >= n_phys:
            break

    keep = np.array(keep[:n_modes])

    eigvals = filtered_eigvals[keep]
    eigvecs = filtered_eigvecs[:, keep]

    if len(eigvals) < min_modes:
        raise RuntimeError(
            f"Only {len(eigvals)} physical modes found at V={velocity:.1f} "
            f"(need {min_modes})"
        )

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
    )


def generate_clip_from_eigendecomposition(
    eigs: Eigendecomposition,
    rng: np.random.Generator,
    t_span: tuple = (0.0, 0.5),
    n_timesteps: int = 512,
    normalize_window_frac: float = 0.1,
    u_crit: float | None = None,
) -> TransientClip:
    """Generate one clip from cached eigendecomposition with random ICs.

    Uses log-normal modal amplitudes so all modes receive meaningful
    excitation with independent random initial conditions.
    """
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
    # P1-15: Overflow control for unstable exponentials
    modal_exponentials(eigvals, t, max_real_exponent=1000.0)
    w_all = np.zeros((MN_eff, n_timesteps))

    for k in range(n_phys):
        lam_k = eigvals[k]
        v_k = eigvecs[:size, k]
        c_k = coeffs_modal[k]

        exp_t = np.exp(lam_k.real * t)
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

    # Map w-DOFs to grid at sensor locations (P2-2: use cached sensor_modes)
    n_sensors = len(eigs.sensor_iy)
    if eigs.sensor_modes is not None:
        # Use cached projection: signals = sensor_modes @ modal_response
        modal_response = coeffs_modal[:, None] * np.exp(eigvals[:, None] * t[None, :])  # (n_phys, n_t)
        sensor_signals = 2.0 * np.real(eigs.sensor_modes @ modal_response)
    else:
        sensor_signals = np.zeros((n_sensors, n_timesteps))
        w_3d = w_all.reshape(eigs.M_eff, eigs.N_eff, n_timesteps)
        for s_idx in range(n_sensors):
            iy, ix = eigs.sensor_iy[s_idx], eigs.sensor_ix[s_idx]
            proj_vy = eigs.vy_grid[:, iy]
            proj_vx = eigs.vx_grid[:, ix]
            sensor_signals[s_idx] = np.einsum('i,ijt,j->t', proj_vx, w_3d, proj_vy)

    # Causal normalization using fixed calibration window
    calibration_samples = max(1, int(n_timesteps * normalize_window_frac))
    sensor_signals = causal_calibration_normalize(sensor_signals, calibration_samples=calibration_samples)

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

    return TransientClip(
        sensor_signals=sensor_signals,
        time=t,
        velocity=eigs.velocity,
        u_crit=u_crit,
        margin=margin,
        eigenvalues=eigvals,
        sensor_xy=np.column_stack([eigs.sensor_iy, eigs.sensor_ix]),
        design_id=f"v{eigs.velocity:.0f}",
    )


def generate_transient_clips_batch(
    solver,
    velocity: float,
    n_realizations: int = 5,
    n_modes: int = 20,
    n_sensors: int = 8,
    t_span: tuple = (0.0, 0.5),
    n_timesteps: int = 512,
    seed: int | None = None,
    normalize_window_frac: float = 0.1,
    sensor_xy: np.ndarray | None = None,
    u_crit: float | None = None,
    rho: float = 1.2,
    c_sound: float = 340.0,
    zeta: float = 0.0,
    nyquist_strict: bool = True,
) -> list[TransientClip]:
    """Generate multiple clips at the same velocity with different random ICs.

    Calls compute_eigendecomposition ONCE (expensive) and reuses the
    eigendecomposition for all realizations.
    """
    eigs = compute_eigendecomposition(
        solver, velocity, n_modes, n_sensors, sensor_xy,
        t_span=t_span, n_timesteps=n_timesteps,
        rho=rho, c_sound=c_sound, zeta=zeta,
        nyquist_strict=nyquist_strict,
    )
    rng = np.random.default_rng(seed)
    clips = []
    for _ in range(n_realizations):
        clip = generate_clip_from_eigendecomposition(
            eigs, rng, t_span, n_timesteps, normalize_window_frac, u_crit,
        )
        clips.append(clip)
    return clips


def generate_transient_clip(
    solver,
    velocity: float,
    t_span: tuple[float, float] = (0.0, 0.5),
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
) -> TransientClip:
    """Generate a single transient clip via full state-space propagation.

    Parameters
    ----------
    solver : FSDTSolver
    velocity : float  — flight velocity (m/s)
    t_span : (t_start, t_end) seconds
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
        t_span=t_span, n_timesteps=n_timesteps,
        rho=rho, c_sound=c_sound, zeta=zeta,
        nyquist_strict=nyquist_strict,
    )

    return generate_clip_from_eigendecomposition(
        eigs, rng, t_span, n_timesteps, normalize_window_frac, u_crit,
    )


def generate_dataset(
    solver,
    n_samples: int = 100,
    velocity_range: tuple[float, float] = (400.0, 800.0),
    t_span: tuple[float, float] = (0.0, 0.5),
    n_timesteps: int = 512,
    n_sensors: int = 8,
    n_modes: int = 20,
    seed: int = 42,
    sensor_xy: np.ndarray | None = None,
    u_crit: float | None = None,
    nyquist_strict: bool = True,
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
            )
            clips.append(clip)
        except (ValueError, AttributeError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as e:
            logger.debug("clip generation failed at v=%.1f: %s", v, e)
            continue
    return clips
