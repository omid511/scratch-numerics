"""Transient response generation via full state-space propagation.

Given an FSDT solver and a flight velocity, reconstructs the time-domain
transient displacement at selected sensor locations.

Uses the complete eigenvalue decomposition with well-conditioned modes
(no `.real` truncation of individual modes).  Causal normalization
avoids future information leakage.
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from scipy import linalg

logger = logging.getLogger(__name__)

_u_crit_cache: dict[int, float | None] = {}


def _get_u_crit(solver, v_lower=680.0, v_upper=3000.0) -> float | None:
    """Cache flutter velocity by solver identity."""
    key = id(solver)
    if key not in _u_crit_cache:
        _u_crit_cache[key] = solver.find_flutter_velocity(
            v_lower=v_lower, v_upper=v_upper, n_scan=10, tol=5.0)
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


def default_sensor_xy(ny: int, nx: int, n_sensors: int = 8) -> np.ndarray:
    """Generate fixed interior sensor positions."""
    iy_vals = np.linspace(1, ny - 2, int(np.ceil(np.sqrt(n_sensors))), dtype=int)
    ix_vals = np.linspace(1, nx - 2, int(np.ceil(np.sqrt(n_sensors))), dtype=int)
    iy_grid, ix_grid = np.meshgrid(iy_vals, ix_vals)
    coords = np.column_stack([iy_grid.ravel(), ix_grid.ravel()])[:n_sensors]
    return coords


def compute_eigendecomposition(
    solver,
    velocity: float,
    n_modes: int = 20,
    n_sensors: int = 8,
    sensor_xy: np.ndarray | None = None,
    t_span: tuple = (0.0, 0.5),
    n_timesteps: int = 512,
) -> Eigendecomposition:
    """Solve the eigenvalue problem once and cache the result.

    Filters to well-conditioned physical modes using frequency range,
    transverse participation (eta_w), and eigenpair residual (r_k).
    """
    from mechanics.piston_theory import AIR_DENSITY, SOUND_SPEED, validate_mach

    M_inf = velocity / SOUND_SPEED
    validate_mach(M_inf, strict_high_mach=True, min_mach=2.0)

    M_mat, K_base, _ = solver._base_matrices()
    K_air, C_air = solver.assemble_aerodynamic(velocity, 0.0, AIR_DENSITY, SOUND_SPEED)
    size = M_mat.shape[0]
    MN_eff = size // 5
    Z = np.zeros((size, size))
    I_mat = np.eye(size)

    A_comp = np.block([
        [Z,                  I_mat],
        [-(K_base + K_air),  -C_air],
    ])
    B_comp = np.block([
        [I_mat, Z],
        [Z,     M_mat],
    ])

    eigvals_all, eigvecs_all = linalg.eig(A_comp, B_comp)
    finite = np.isfinite(eigvals_all)
    eigvals_all = eigvals_all[finite]
    eigvecs_all = eigvecs_all[:, finite]

    K_total = K_base + K_air
    C_total = C_air
    w_start = 2 * MN_eff
    w_end = 3 * MN_eff

    n_total = len(eigvals_all)
    eigvecs_phys = eigvecs_all[:size, :]
    freqs = np.abs(eigvals_all.imag)

    # Anti-alias filter: compute Nyquist from sample rate
    fs = n_timesteps / (t_span[1] - t_span[0])  # sample rate in Hz
    f_nyquist = 0.4 * fs  # conservative Nyquist cutoff (Hz)
    freq_max_rad = f_nyquist * 2 * np.pi  # convert to rad/s

    is_physical = np.zeros(n_total, dtype=bool)
    for k in range(n_total):
        if freqs[k] < 10 or freqs[k] > freq_max_rad:
            continue
        if eigvals_all.imag[k] <= 0:
            continue
        q = eigvecs_phys[:, k]
        w_block = q[w_start:w_end]
        q_norm = np.abs(q).sum() + 1e-30
        eta_w = np.abs(w_block).sum() / q_norm
        if eta_w <= 0.001:
            continue
        s = eigvals_all[k]
        Mq = M_mat @ q
        Kq = K_total @ q
        Cq = C_total @ q
        numer = np.linalg.norm(s**2 * Mq + s * Cq + Kq)
        denom = abs(s)**2 * np.linalg.norm(Mq) + abs(s) * np.linalg.norm(Cq) + np.linalg.norm(Kq)
        r_k = numer / (denom + 1e-30)
        if r_k >= 0.1:
            continue
        is_physical[k] = True

    good_idx = np.where(is_physical)[0]
    good_idx = good_idx[np.argsort(eigvals_all[good_idx].imag)]
    good_idx = good_idx[:n_modes]

    eigvals = eigvals_all[good_idx]
    eigvecs = eigvecs_all[:, good_idx]

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

    return Eigendecomposition(
        eigvals=eigvals,
        eigvecs=eigvecs,
        size=size,
        MN_eff=MN_eff,
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
    excitation, with the flutter pair (2 least-stable modes) guaranteed
    at least 20% of total initial amplitude.
    """
    eigvals = eigs.eigvals
    eigvecs = eigs.eigvecs
    size = eigs.size
    MN_eff = eigs.MN_eff
    w_start = 2 * MN_eff

    n_phys = len(eigvals)

    # Identify flutter pair (2 least-stable modes: highest Re eigenvalue)
    re_parts = eigvals.real
    flutter_pair_idx = np.argsort(re_parts)[-2:]

    # Random modal amplitudes (log-normal to ensure all modes get excitation)
    amplitudes = rng.lognormal(0, 0.5, size=n_phys)
    # Ensure flutter pair gets meaningful excitation: at least 20% of total
    amp_median = np.median(amplitudes)
    amplitudes[flutter_pair_idx] = np.maximum(
        amplitudes[flutter_pair_idx], 0.2 * amp_median
    )

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=n_phys)

    # Build modal coefficients: amplitude * exp(i*phase)
    coeffs_modal = amplitudes * np.exp(1j * phases)

    # Build initial state from modal combination: x0 = V_re @ c
    # where c = coeffs_modal expressed in real-valued coordinates
    x0 = np.zeros(2 * size)
    for k in range(n_phys):
        lam_k = eigvals[k]
        v_k = eigvecs[:size, k]
        c_k = coeffs_modal[k]
        # Conjugate pair contribution to initial state
        x0[:size] += (c_k * v_k).real
        # Velocity half: s * x contribution
        x0[size:] += (c_k * lam_k * v_k).real

    # Propagate using conjugate-pair real decomposition
    t = np.linspace(t_span[0], t_span[1], n_timesteps)
    w_all = np.zeros((MN_eff, n_timesteps))
    seen = set()

    for k in range(n_phys):
        if k in seen:
            continue
        lam_k = eigvals[k]
        v_k = eigvecs[:size, k]
        c_k = coeffs_modal[k]

        conj_k = None
        for j in range(k + 1, n_phys):
            if j not in seen and np.abs(eigvals[j] - np.conj(lam_k)) < 1e-8 * max(1, np.abs(lam_k)):
                conj_k = j
                break

        if conj_k is not None:
            seen.add(k)
            seen.add(conj_k)
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
        else:
            seen.add(k)
            w_v = v_k[w_start:w_start + MN_eff]
            modal_coeff = c_k * w_v
            exp_all = np.exp(lam_k.real * t)
            phase = np.exp(1j * lam_k.imag * t)
            contrib = np.real(modal_coeff[:, None] * (exp_all * phase)[None, :])
            w_all += contrib

    # Map w-DOFs to grid at sensor locations
    n_sensors = len(eigs.sensor_iy)
    n_sensors_actual = n_sensors
    sensor_signals = np.zeros((n_sensors_actual, n_timesteps))
    w_3d = w_all.reshape(eigs.M_eff, eigs.N_eff, n_timesteps)
    for s_idx in range(n_sensors):
        iy, ix = eigs.sensor_iy[s_idx], eigs.sensor_ix[s_idx]
        proj_vy = eigs.vy_grid[:, iy]
        proj_vx = eigs.vx_grid[:, ix]
        sensor_signals[s_idx] = np.einsum('i,ijt,j->t', proj_vx, w_3d, proj_vy)

    # Causal normalization: initial-window RMS
    n_init = max(1, int(n_timesteps * normalize_window_frac))
    init_window = sensor_signals[:, :n_init]
    s0 = np.sqrt(np.mean(init_window**2) + 1e-12)
    sensor_signals = sensor_signals / s0

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
) -> list[TransientClip]:
    """Generate multiple clips at the same velocity with different random ICs.

    Calls solve_complex_modal ONCE (expensive) and reuses the
    eigendecomposition for all realizations.
    """
    eigs = compute_eigendecomposition(solver, velocity, n_modes, n_sensors, sensor_xy)
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
) -> TransientClip:
    """Generate a single transient clip via full state-space propagation.

    Uses all eigenpairs from the generalized companion pencil, filters
    to well-conditioned modes, and propagates x(t) = V_re V_re⁻¹ x₀
    where V_re contains real-part projections of conjugate pairs.

    Parameters
    ----------
    solver : FSDTSolver
    velocity : float  — flight velocity (m/s)
    t_span : (t_start, t_end) seconds
    n_timesteps : number of time samples
    n_sensors : how many grid points to sample
    n_modes : modes to keep from eigen-solve (for eigenvalue storage)
    rng : random generator for reproducibility
    normalize_window_frac : fraction of clip used for causal normalization
    sensor_xy : optional (n_sensors, 2) array of [iy, ix] grid indices

    Returns
    -------
    TransientClip
    """
    if rng is None:
        rng = np.random.default_rng()

    from mechanics.piston_theory import AIR_DENSITY, SOUND_SPEED, validate_mach

    M_inf = velocity / SOUND_SPEED
    validate_mach(M_inf, strict_high_mach=True, min_mach=2.0)

    result = solver.solve_complex_modal(velocity, n_modes=n_modes)
    eigenvalues = result.eigenvalues          # complex, (n_modes,)
    mode_shapes = result.mode_shapes          # (n_modes, ny, nx)
    ny, nx = mode_shapes.shape[1], mode_shapes.shape[2]

    # Pick sensor locations on the grid
    if sensor_xy is not None:
        sensor_iy = sensor_xy[:, 0]
        sensor_ix = sensor_xy[:, 1]
        n_sensors = len(sensor_iy)
    else:
        # Fixed interior sensors: avoid boundary nodes
        coords = default_sensor_xy(ny, nx, n_sensors)
        sensor_iy = coords[:, 0]
        sensor_ix = coords[:, 1]

    # Build full generalized companion pencil
    M_mat, K_base, _ = solver._base_matrices()
    K_air, C_air = solver.assemble_aerodynamic(velocity, 0.0, AIR_DENSITY, SOUND_SPEED)
    size = M_mat.shape[0]
    MN_eff = size // 5
    Z = np.zeros((size, size))
    I_mat = np.eye(size)

    A_comp = np.block([
        [Z,                  I_mat],
        [-(K_base + K_air),  -C_air],
    ])
    B_comp = np.block([
        [I_mat, Z],
        [Z,     M_mat],
    ])

    # Solve generalized eigenvalue problem — all eigenvalues
    eigvals_all, eigvecs_all = linalg.eig(A_comp, B_comp)
    finite = np.isfinite(eigvals_all)
    eigvals_all = eigvals_all[finite]
    eigvecs_all = eigvecs_all[:, finite]

    # ── Select well-conditioned conjugate pairs ──
    # Use the SAME filtering as _max_real_eigenvalue: frequency range,
    # transverse participation (eta_w), and eigenpair residual (r_k).
    n_total = len(eigvals_all)

    K_total = K_base + K_air
    C_total = C_air
    MN_eff = size // 5
    w_start = 2 * MN_eff
    w_end = 3 * MN_eff

    # Compute transverse participation and residual for each eigenpair
    eigvecs_phys = eigvecs_all[:size, :]  # displacement half
    freqs = np.abs(eigvals_all.imag)  # rad/s

    is_physical = np.zeros(n_total, dtype=bool)
    for k in range(n_total):
        # 1. Physical frequency range
        if freqs[k] < 10 or freqs[k] > 100000:
            continue
        # 2. Positive imaginary part (one-sided)
        if eigvals_all.imag[k] <= 0:
            continue
        # 3. Transverse participation: eta_w > 0.001
        q = eigvecs_phys[:, k]
        w_block = q[w_start:w_end]
        q_norm = np.abs(q).sum() + 1e-30
        eta_w = np.abs(w_block).sum() / q_norm
        if eta_w <= 0.001:
            continue
        # 4. Eigenpair residual: r_k < 0.1
        s = eigvals_all[k]
        Mq = M_mat @ q
        Kq = K_total @ q
        Cq = C_total @ q
        numer = np.linalg.norm(s**2 * Mq + s * Cq + Kq)
        denom = abs(s)**2 * np.linalg.norm(Mq) + abs(s) * np.linalg.norm(Cq) + np.linalg.norm(Kq)
        r_k = numer / (denom + 1e-30)
        if r_k >= 0.1:
            continue
        is_physical[k] = True

    good_idx = np.where(is_physical)[0]
    # Sort by frequency
    good_idx = good_idx[np.argsort(eigvals_all[good_idx].imag)]

    # Take up to n_modes modes
    good_idx = good_idx[:n_modes]

    eigvals = eigvals_all[good_idx]
    eigvecs = eigvecs_all[:, good_idx]

    # Reject clip if any selected mode is unstable (Re > 0)
    if np.any(eigvals.real > 0):
        raise ValueError(
            f"Unstable mode detected at V={velocity:.1f} "
            f"(max Re={eigvals.real.max():.4f}), skipping clip"
        )

    # ── Real physical initial state ──
    x0 = np.zeros(2 * size)
    w_start = 2 * MN_eff
    n_init_dofs = min(n_sensors, MN_eff)
    init_dofs = rng.choice(MN_eff, size=n_init_dofs, replace=False)
    x0[w_start + init_dofs] = 0.1 * rng.standard_normal(n_init_dofs)
    x0[size:] = 0.01 * rng.standard_normal(size)

    # ── Propagate using conjugate-pair real decomposition ──
    # For each pair (λ, λ*), contribution is 2 Re[c_k v_k exp(λ_k t)]
    # where c_k = (v_k^H x₀) / (v_k^H v_k) for right eigenvectors
    t = np.linspace(t_span[0], t_span[1], n_timesteps)

    # Compute modal coefficients via pseudoinverse (handles conditioning)
    coeffs = np.linalg.pinv(eigvecs) @ x0  # (n_modes,)

    # Build modal contribution: sum over conjugate pairs
    w_all = np.zeros((MN_eff, n_timesteps))
    seen = set()
    for k in range(len(eigvals)):
        if k in seen:
            continue
        lam_k = eigvals[k]
        v_k = eigvecs[:size, k]

        # Find conjugate partner (if any)
        conj_k = None
        for j in range(k + 1, len(eigvals)):
            if j not in seen and np.abs(eigvals[j] - np.conj(lam_k)) < 1e-8 * max(1, np.abs(lam_k)):
                conj_k = j
                break

        c_k = coeffs[k]

        if conj_k is not None:
            # Conjugate pair: x(t) += 2 Re[c_k v_k exp(λ_k t)]
            seen.add(k)
            seen.add(conj_k)
            exp_t = np.exp(lam_k.real * t)  # (n_t,)
            cos_t = np.cos(lam_k.imag * t)  # (n_t,)
            sin_t = np.sin(lam_k.imag * t)  # (n_t,)
            v_real = v_k.real  # (size,)
            v_imag = v_k.imag
            c_real = c_k.real
            c_imag = c_k.imag
            w_v_real = v_real[w_start:w_start + MN_eff]  # (MN_eff,)
            w_v_imag = v_imag[w_start:w_start + MN_eff]  # (MN_eff,)
            real_part = c_real * w_v_real - c_imag * w_v_imag  # (MN_eff,)
            imag_part = -c_imag * w_v_real - c_real * w_v_imag  # (MN_eff,)
            contrib = 2.0 * (np.outer(real_part, cos_t * exp_t) + np.outer(imag_part, sin_t * exp_t))
            w_all += contrib
        else:
            # Unpaired mode (shouldn't happen for real system, but handle)
            seen.add(k)
            c_k = coeffs[k]
            w_v = v_k[w_start:w_start + MN_eff]  # (MN_eff,) complex
            modal_coeff = c_k * w_v  # (MN_eff,) complex
            exp_all = np.exp(lam_k.real * t)  # (n_t,)
            phase = np.exp(1j * lam_k.imag * t)  # (n_t,)
            contrib = np.real(modal_coeff[:, None] * (exp_all * phase)[None, :])  # (MN_eff, n_t)
            w_all += contrib

    # w_all is (MN_eff, n_t)

    # Map w-DOFs to grid at sensor locations
    vx_grid = solver._vx_grid  # (M_eff, grid_nx)
    vy_grid = solver._vy_grid  # (N_eff, grid_ny)
    M_eff = vx_grid.shape[0]
    N_eff = vy_grid.shape[0]

    sensor_signals = np.zeros((n_sensors, n_timesteps))
    w_3d = w_all.reshape(M_eff, N_eff, n_timesteps)
    for s_idx in range(n_sensors):
        iy, ix = sensor_iy[s_idx], sensor_ix[s_idx]
        proj_vy = vy_grid[:, iy]  # (N_eff,)
        proj_vx = vx_grid[:, ix]  # (M_eff,)
        sensor_signals[s_idx] = np.einsum('i,ijt,j->t', proj_vx, w_3d, proj_vy)

    # ── Causal normalization: initial-window RMS ──
    n_init = max(1, int(n_timesteps * normalize_window_frac))
    init_window = sensor_signals[:, :n_init]
    s0 = np.sqrt(np.mean(init_window**2) + 1e-12)
    sensor_signals = sensor_signals / s0

    # Margin — use pre-computed u_crit if provided, otherwise scan
    margin = float("nan")
    if u_crit is not None and u_crit > 0:
        margin = (u_crit - velocity) / u_crit
    else:
        try:
            u_crit_computed = _get_u_crit(solver)
            if u_crit_computed is not None and u_crit_computed > 0:
                u_crit = u_crit_computed
                margin = (u_crit - velocity) / u_crit
        except (ValueError, AttributeError, np.linalg.LinAlgError, RuntimeError) as e:
            logger.debug("flutter velocity scan failed at v=%.1f: %s", velocity, e)

    return TransientClip(
        sensor_signals=sensor_signals,
        time=t,
        velocity=velocity,
        u_crit=u_crit,
        margin=margin,
        eigenvalues=eigenvalues,
        sensor_xy=np.column_stack([sensor_iy, sensor_ix]),
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
                sensor_xy=sensor_xy, u_crit=u_crit,
            )
            clips.append(clip)
        except (ValueError, AttributeError, np.linalg.LinAlgError) as e:
            logger.debug("clip generation failed at v=%.1f: %s", v, e)
            continue
    return clips
