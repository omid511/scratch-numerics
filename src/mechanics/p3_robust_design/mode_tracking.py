"""MAC-based mode tracking across velocity sweeps.

Tracks mode identity via Modal Assurance Criterion (MAC) with
Hungarian-optimal assignment. Used to determine if mode-decomposition
surrogate is feasible.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..solver import FSDTSolver


def mac(a: np.ndarray, b: np.ndarray) -> float:
    """Modal Assurance Criterion between two mode shapes.

    Uses conjugate inner products, so MAC is invariant to a global
    complex phase on either shape.
    """
    a_flat, b_flat = a.flatten(), b.flatten()
    num = abs(np.vdot(a_flat, b_flat)) ** 2
    den = np.vdot(a_flat, a_flat).real * np.vdot(b_flat, b_flat).real + 1e-30
    return float(num / den)


def _optimal_match(
    prev: np.ndarray,
    curr: np.ndarray,
    prev_freqs: np.ndarray | None = None,
    curr_freqs: np.ndarray | None = None,
    w_f: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-optimal mode assignment from prev to curr modes.

    Cost per pair is ``1 - MAC``. When frequency arrays are supplied, a
    small frequency-proximity term ``w_f * |f_i - f_j| / f_scale`` is
    added, where ``f_scale`` is the mean of the two frequency arrays.
    This stabilizes assignment of near-degenerate modes (e.g. mixed
    (m,n)/(n,m) pairs) that MAC alone cannot disambiguate.

    Args:
        prev: (n, ...) mode shapes at previous step (real or complex).
        curr: (n, ...) mode shapes at current step.
        prev_freqs: optional (n,) frequencies at previous step.
        curr_freqs: optional (n,) frequencies at current step.
        w_f: weight of the frequency-proximity term.

    Returns:
        macs: MAC values for each assigned mode pair.
        assignment: mapping from prev index -> curr index.
    """
    n = len(curr)
    cost = np.zeros((n, n))
    use_freqs = prev_freqs is not None and curr_freqs is not None
    if use_freqs:
        f_scale = 0.5 * (np.mean(prev_freqs) + np.mean(curr_freqs)) + 1e-30
    for i in range(n):
        for j in range(n):
            cost[i, j] = 1.0 - mac(curr[i], prev[j])
            if use_freqs:
                cost[i, j] += w_f * abs(curr_freqs[i] - prev_freqs[j]) / f_scale
    rows, cols = linear_sum_assignment(cost)
    assignment = np.empty(n, dtype=int)
    macs = np.empty(n)
    for curr_i, prev_i in zip(rows, cols):
        assignment[prev_i] = curr_i
        macs[prev_i] = 1.0 - cost[curr_i, prev_i]
    return macs, assignment


def track_modes_across_velocity(
    solver: FSDTSolver,
    velocities: np.ndarray,
    n_modes: int = 4,
) -> dict:
    """Track mode identity across a velocity sweep.

    Args:
        solver: FSDTSolver with boundary conditions set.
        velocities: Sorted array of velocities to sweep.
        n_modes: Number of modes to track.

    Returns:
        dict with keys:
            'frequencies': (n_vel, n_modes) tracked frequencies
            'damping': (n_vel, n_modes) tracked damping ratios
            'mac_values': (n_vel-1, n_modes) MAC between consecutive steps
            'mode_labels': (n_vel, n_modes) integer labels (0..n_modes-1)
    """
    n_vel = len(velocities)
    freqs_all = np.zeros((n_vel, n_modes))
    damp_all = np.zeros((n_vel, n_modes))
    mac_all = np.zeros((n_vel - 1, n_modes))
    labels = np.zeros((n_vel, n_modes), dtype=int)

    prev_modes = None
    prev_freqs = None
    prev_damp = None
    current_labels = np.arange(n_modes)

    for vi, v in enumerate(velocities):
        result = solver.solve_complex_modal(float(v), n_modes=n_modes)
        freqs = result.frequencies[:n_modes]
        damp = np.array([
            -e.real / (abs(e) + 1e-30) for e in result.eigenvalues[:n_modes]
        ])
        # Prefer complex mode shapes: MAC on complex shapes is phase-
        # invariant, so it stays high even when the real-part slice
        # rotates spatially along an aeroelastic branch.
        complex_shapes = getattr(result, "mode_shapes_complex", None)
        mode_shapes = (
            complex_shapes[:n_modes]
            if complex_shapes is not None
            else result.mode_shapes[:n_modes]
        )

        if prev_modes is not None:
            mac_vals, assignment = _optimal_match(
                prev_modes, mode_shapes, prev_freqs, freqs
            )
            freqs_all[vi] = freqs[assignment]
            damp_all[vi] = damp[assignment]
            mac_all[vi - 1] = mac_vals
            labels[vi] = current_labels[assignment]
            current_labels = labels[vi].copy()
        else:
            freqs_all[vi] = freqs
            damp_all[vi] = damp
            labels[vi] = current_labels

        prev_modes = mode_shapes

        prev_freqs = freqs_all[vi]
        prev_damp = damp_all[vi]

    return {
        "frequencies": freqs_all,
        "damping": damp_all,
        "mac_values": mac_all,
        "mode_labels": labels,
    }


def track_modes_across_parameter(
    solver_factory,
    param_name: str,
    param_values,
    *,
    velocity: float = 1.0,
    n_modes: int = 6,
    freq_weight: float = 0.1,
) -> dict:
    """Track mode identity as ONE design parameter varies.

    Design-space generalization of :func:`track_modes_across_velocity`: the
    sweep coordinate is built by ``solver_factory(value)`` instead of a
    velocity on a fixed solver. Matching uses the identical cost model
    (MAC distance plus frequency-proximity term) as the velocity tracker.

    Args:
        solver_factory: Callable ``value -> FSDTSolver`` with boundary
            conditions already configured.
        param_name: Label for the swept parameter (reporting only).
        param_values: Sequence of parameter values, one solver build per value.
        velocity: Free-stream velocity in m/s passed to
            ``solve_complex_modal`` (must be supersonic, Mach > 1).
        n_modes: Number of modes to track.
        freq_weight: Weight of the frequency-proximity term in the
            assignment cost (same convention as ``_optimal_match``).

    Returns:
        dict with keys:
            'param_name': the swept parameter label
            'param_values': (n_steps,) swept values
            'frequencies': (n_steps, n_modes) tracked frequencies (Hz);
                rows after the first are reordered onto the previous step's
                branch labels via the Hungarian assignment
            'mac_values': (n_steps-1, n_modes) MAC between consecutive steps
            'mode_labels': (n_steps, n_modes) integer branch labels
            'summary': {'n_steps', 'median_step_mac', 'worst_step_mac'}
    """
    param_values = [float(v) for v in param_values]
    n_steps = len(param_values)
    if n_steps < 1:
        raise ValueError("param_values must contain at least one value")

    freqs_all = np.zeros((n_steps, n_modes))
    mac_all = np.zeros((max(n_steps - 1, 0), n_modes))
    labels = np.zeros((n_steps, n_modes), dtype=int)

    prev_modes = None
    prev_freqs = None
    current_labels = np.arange(n_modes)

    for si, value in enumerate(param_values):
        solver = solver_factory(value)
        result = solver.solve_complex_modal(velocity, n_modes=n_modes)
        if len(result.frequencies) < n_modes:
            raise ValueError(
                f"solver returned {len(result.frequencies)} modes at "
                f"{param_name}={value}, expected >= {n_modes}"
            )
        freqs = result.frequencies[:n_modes]
        # Complex shapes: MAC is phase-invariant, matching the velocity
        # tracker's convention.
        complex_shapes = getattr(result, "mode_shapes_complex", None)
        mode_shapes = (
            complex_shapes[:n_modes]
            if complex_shapes is not None
            else result.mode_shapes[:n_modes]
        )

        if prev_modes is not None:
            mac_vals, assignment = _optimal_match(
                prev_modes, mode_shapes, prev_freqs, freqs, w_f=freq_weight
            )
            freqs_all[si] = freqs[assignment]
            mac_all[si - 1] = mac_vals
            labels[si] = current_labels[assignment]
            current_labels = labels[si].copy()
        else:
            freqs_all[si] = freqs
            labels[si] = current_labels

        prev_modes = mode_shapes
        prev_freqs = freqs_all[si]

    return {
        "param_name": param_name,
        "param_values": np.asarray(param_values),
        "frequencies": freqs_all,
        "mac_values": mac_all,
        "mode_labels": labels,
        "summary": {
            "n_steps": n_steps,
            "median_step_mac": float(np.median(mac_all)) if n_steps > 1 else 1.0,
            "worst_step_mac": float(mac_all.min()) if n_steps > 1 else 1.0,
        },
    }


def extract_mode_data(
    solver: FSDTSolver,
    velocity: float,
    n_modes: int = 4,
) -> dict:
    """Extract per-mode eigenvalue real parts at a given velocity.

    Returns:
        dict with 'real_parts', 'imag_parts', 'mode_shapes', 'frequencies'.
    """
    result = solver.solve_complex_modal(velocity, n_modes=n_modes)
    return {
        "real_parts": np.array([e.real for e in result.eigenvalues[:n_modes]]),
        "imag_parts": np.array([e.imag for e in result.eigenvalues[:n_modes]]),
        "mode_shapes": result.mode_shapes[:n_modes],
        "frequencies": result.frequencies[:n_modes],
    }
