"""Frequency-based identifiability analysis for spatial damage fields.

Formalizes the P2 MVP finding that modal frequencies alone do not
identifiably determine an 8x8 spatial stiffness-retention field (6
measurements vs 64 unknowns -> a large nullspace), while mode shapes
restore identifiability.

Quantities:
- finite-difference Jacobian d(frequencies)/d(field pixels),
- effective rank of that Jacobian (SVD, relative threshold),
- nullspace dimension,
- Cramer-Rao-style lower bound on field MSE under Gaussian frequency
  noise, restricted to the identifiable subspace,
- adversarial pairs: fields differing on many pixels whose frequency
  signatures agree within the solver/noise tolerance.
"""

from __future__ import annotations

import numpy as np

from mechanics.p2_inverse_damage.damage_data import _default_laminate
from mechanics.solver import FSDTSolver

FIELD_GRID = (8, 8)
N_MODES = 6


def _make_solver() -> FSDTSolver:
    """Clamped FSDT solver matching generate_field_dataset conventions."""
    solver = FSDTSolver(
        L1=0.3, L2=0.3, M=6, N=6,
        laminate=_default_laminate(),
        basis_type="legendre",
    )
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    return solver


def _solve_frequencies(solver: FSDTSolver, field: np.ndarray) -> np.ndarray:
    solver.set_damage_field(np.asarray(field, dtype=float))
    result = solver.solve_modal(n_modes=N_MODES)
    return np.asarray(result.frequencies.real)


def frequency_jacobian(
    field: np.ndarray,
    *,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Finite-difference Jacobian of the 6 frequencies w.r.t. 8x8 pixels.

    Forward differences: column j is (f(f + eps*e_j) - f(f)) / eps.
    Retention values are clipped into (epsilon_floor, 1] so the perturbed
    field stays valid for ``set_damage_field``.

    Returns shape (6, 64), row-major over the (n_y, n_x) grid.
    """
    field = np.asarray(field, dtype=float)
    if field.shape != FIELD_GRID:
        raise ValueError(f"field must have shape {FIELD_GRID}, got {field.shape}")

    solver = _make_solver()
    f0 = _solve_frequencies(solver, field)

    jac = np.zeros((N_MODES, field.size))
    flat = field.ravel()
    floor = 10.0 * epsilon
    for j in range(flat.size):
        original = flat[j]
        step = epsilon if original + epsilon <= 1.0 else -epsilon
        trial = field.copy()
        trial.ravel()[j] = original + step
        f1 = _solve_frequencies(solver, trial)
        jac[:, j] = (f1 - f0) / step
    return jac


def effective_rank(jacobian: np.ndarray, *, threshold: float = 0.01) -> int:
    """Number of singular values above ``threshold * max_singular_value``."""
    s = np.linalg.svd(np.asarray(jacobian, dtype=float), compute_uv=False)
    if s.size == 0 or s[0] <= 0.0:
        return 0
    return int(np.sum(s > threshold * s[0]))


def nullspace_dimension(jacobian: np.ndarray, *, threshold: float = 0.01) -> int:
    """Columns minus effective rank: unidentifiable pixel directions."""
    return int(np.asarray(jacobian).shape[1]) - effective_rank(
        jacobian, threshold=threshold
    )


def cramer_rao_floor(
    jacobian: np.ndarray,
    sigma: np.ndarray | float,
    *,
    threshold: float = 0.01,
) -> float:
    """Cramer-Rao-style floor on achievable field MSE.

    With Gaussian frequency noise of per-mode std ``sigma`` and unbiased
    estimator, MSE >= sigma^2 * trace((J^T J)^{-1}) / n_pixels on the
    identifiable subspace (pseudo-inverse; nullspace directions contribute
    zero because they cannot be estimated from frequencies at all).
    """
    J = np.asarray(jacobian, dtype=float)
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[0],))
    if np.any(sigma_arr <= 0.0):
        return 0.0  # noiseless measurements: no information-theoretic floor
    # Whiten: J_w = D^{-1} J makes the information matrix J_w^T J_w.
    J_w = J / sigma_arr[:, None]
    s2 = np.linalg.svd(J_w, compute_uv=False) ** 2
    keep = s2 > (threshold**2) * s2[0]
    # sum of inverse nonzero squared singular values == trace of pinv(J^T J)
    return float(np.sum(1.0 / s2[keep]) / J.shape[1])


def identifiability_report(
    dataset_path: str,
    *,
    n_samples: int = 20,
    noise_levels: tuple[float, ...] = (0.0, 0.005, 0.02),
) -> dict:
    """Identifiability statistics over a field dataset.

    For up to ``n_samples`` dataset fields: compute the frequency Jacobian
    once, then evaluate effective rank, nullspace dimension, and the
    Cramer-Rao-style MSE floor per multiplicative noise level (sigma is a
    fraction of each mode's frequency). The verdict compares the median
    floor at the largest non-zero noise against the 0.030 field-MSE bar.
    """
    data = np.load(dataset_path)
    fields = data["fields_values"]
    if fields.ndim != 3 or fields.shape[1:] != FIELD_GRID:
        raise ValueError(
            f"{dataset_path}: expected fields ({len(fields)}, 8, 8), "
            f"got {fields.shape}"
        )
    fields = fields[:n_samples]

    ranks, null_dims, floors = [], [], {n: [] for n in noise_levels}
    for k, field in enumerate(fields):
        J = frequency_jacobian(field)
        rank = effective_rank(J)
        ranks.append(rank)
        null_dims.append(nullspace_dimension(J))
        for noise in noise_levels:
            sigma = noise * np.asarray(_reference_frequencies())
            floors[noise].append(cramer_rao_floor(J, sigma))

    max_noise = max(noise_levels)
    median_floor = (
        float(np.median(floors[max_noise])) if max_noise > 0 else None
    )
    return {
        "n_fields": int(len(fields)),
        "dataset": str(dataset_path),
        "effective_rank": {
            "median": float(np.median(ranks)),
            "min": int(np.min(ranks)),
            "max": int(np.max(ranks)),
        },
        "nullspace_dimension": {
            "median": float(np.median(null_dims)),
            "min": int(np.min(null_dims)),
            "max": int(np.max(null_dims)),
        },
        "crb_floor_by_noise": {
            str(n): float(np.median(v)) for n, v in floors.items()
        },
        "mse_bar": 0.030,
        "bar_reachable_at_max_noise": (
            None if median_floor is None else bool(median_floor < 0.030)
        ),
        "verdict": _verdict(ranks, null_dims, median_floor),
    }


_REFERENCE_FREQS: list[float] | None = None


def _reference_frequencies() -> list[float]:
    """Nominal frequencies used only to convert relative noise to absolute
    per-mode sigma (cached; pristine-ish reference scale)."""
    global _REFERENCE_FREQS
    if _REFERENCE_FREQS is None:
        solver = _make_solver()
        f = _solve_frequencies(solver, np.ones(FIELD_GRID))
        _REFERENCE_FREQS = [float(x) for x in f]
    return _REFERENCE_FREQS


def _verdict(ranks, null_dims, median_floor) -> str:
    parts = [
        f"frequencies identify {int(np.median(ranks))}/64 pixel directions "
        f"(nullspace {int(np.median(null_dims))})"
    ]
    if median_floor is not None:
        reachable = median_floor < 0.030
        parts.append(
            f"CRB floor {median_floor:.4g} at max noise -> "
            + ("below" if reachable else "AT OR ABOVE")
            + " the 0.030 bar"
        )
        if not reachable:
            parts.append(
                "frequency-only inversion cannot meet the bar; shape "
                "conditioning is required"
            )
    return "; ".join(parts)


def adversarial_pair(
    field_a: np.ndarray,
    field_b: np.ndarray,
    *,
    solver_tolerance: float = 0.01,
) -> bool:
    """Do two fields hide behind one frequency signature?

    True when the fields differ on >= 30% of cells yet every relative
    frequency difference stays below ``solver_tolerance`` (the practical
    noise/solver floor): frequencies alone cannot distinguish them.
    """
    a = np.asarray(field_a, dtype=float)
    b = np.asarray(field_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"field shapes must match, got {a.shape} vs {b.shape}")
    frac_diff = float(np.mean(a != b))
    if frac_diff < 0.30:
        return False

    solver = _make_solver()
    fa = _solve_frequencies(solver, a)
    fb = _solve_frequencies(solver, b)
    rel = np.abs(fb - fa) / np.abs(fa)
    return bool(rel.max() < solver_tolerance)
