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
    """Modal Assurance Criterion between two mode shapes."""
    a_flat, b_flat = a.flatten(), b.flatten()
    num = abs(np.dot(a_flat, b_flat)) ** 2
    den = np.dot(a_flat, a_flat) * np.dot(b_flat, b_flat) + 1e-30
    return float(num / den)


def _optimal_match(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-optimal mode assignment from prev to curr modes.

    Returns:
        macs: MAC values for each assigned mode pair.
        assignment: mapping from prev index -> curr index.
    """
    n = len(curr)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = 1.0 - mac(curr[i], prev[j])
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
        mode_shapes = result.mode_shapes[:n_modes]

        if prev_modes is not None:
            mac_vals, assignment = _optimal_match(prev_modes, mode_shapes)
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
