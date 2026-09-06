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

from mechanics.p2_inverse_damage.damage_data import (
    _canonicalize_mode_shape,
    _default_laminate,
)
from mechanics.solver import FSDTSolver

FIELD_GRID = (8, 8)
N_MODES = 6


def _make_solver(grid: tuple[int, int] = (64, 64)) -> FSDTSolver:
    """Clamped FSDT solver matching generate_field_dataset conventions."""
    solver = FSDTSolver(
        L1=0.3, L2=0.3, M=6, N=6,
        laminate=_default_laminate(),
        basis_type="legendre", grid=grid,
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
        trial.ravel()[j] = min(max(original + step, floor), 1.0)
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
    zero because they cannot be estimated from frequencies at all, so this
    is a subspace-restricted floor, not a full-field bound).

    Returns ``inf`` when no pixel direction clears the rank threshold
    (nothing identifiable: unbounded variance) or when any per-mode noise
    std is nonpositive (the noiseless limit is singular for a
    rank-deficient Jacobian).
    """
    J = np.asarray(jacobian, dtype=float)
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[0],))
    if np.any(sigma_arr <= 0.0):
        return float("inf")
    # Whiten: J_w = D^{-1} J makes the information matrix J_w^T J_w.
    J_w = J / sigma_arr[:, None]
    s2 = np.linalg.svd(J_w, compute_uv=False) ** 2
    if s2.size == 0 or not np.all(np.isfinite(s2)) or s2[0] <= 0.0:
        return float("inf")
    keep = s2 > (threshold**2) * s2[0]
    if not np.any(keep):
        return float("inf")
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
    damage_tol: float = 1e-6,
) -> bool:
    """Do two fields hide behind one frequency signature?

    True when the fields differ by more than ``damage_tol`` on >= 30% of
    cells yet every relative frequency difference stays below
    ``solver_tolerance`` (the practical noise/solver floor): frequencies
    alone cannot distinguish them. The tolerance keeps solver-noise-level
    jitter on continuous retention values from counting as damage.
    """
    a = np.asarray(field_a, dtype=float)
    b = np.asarray(field_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"field shapes must match, got {a.shape} vs {b.shape}")
    frac_diff = float(np.mean(np.abs(a - b) > damage_tol))
    if frac_diff < 0.30:
        return False

    solver = _make_solver()
    fa = _solve_frequencies(solver, a)
    fb = _solve_frequencies(solver, b)
    rel = np.abs(fb - fa) / np.abs(fa)
    return bool(rel.max() < solver_tolerance)


def _ridge_field_mse(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    *,
    lam: float = 1.0,
) -> float:
    """Closed-form ridge (with intercept) train-to-test field MSE.

    Numpy-only probe used to measure channel skill on a dataset: no
    iterations, no RNG. Inputs are per-sample feature rows; targets are
    (n, gy, gx) retention fields.
    """
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64).reshape(len(Xtr), -1)
    Yte = np.asarray(Y_test, dtype=np.float64).reshape(len(Xte), -1)
    Xa = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xe = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    W = np.linalg.solve(A, Xa.T @ Ytr)
    return float(np.mean((Xe @ W - Yte) ** 2))


def shape_sensitivity_ratio(
    dataset_path: str,
    *,
    n_samples: int = 5,
    freq_only_mse: float | None = None,
    shape_mse: float | None = None,
    freq_crb_floor: float | None = None,
) -> dict:
    """Quantify how much more identifiable the shape channel is.

    Every headline number is measured from ``dataset_path`` unless the
    caller overrides it explicitly (``None`` means "compute"): a ridge
    probe maps log-frequencies to fields (``freq_only_mse``) and stored
    mode shapes — or, failing that, stored shape summaries — to fields
    (``shape_mse``), both fit on the dataset's train split and scored on
    its test split; ``freq_crb_floor`` is the median
    :func:`cramer_rao_floor` over the first ``n_samples`` fields at 2%
    relative noise, reusing the Jacobians already computed for the rank
    check below.

    The ratio freq_CRB / shape_MSE estimates how many times the shape
    channel exceeds the frequency channel in spatial identification
    capability. A ratio >> 1 means shapes are strictly required.
    ``shape_mse`` is ``None`` (and the conclusion says so) when the
    dataset stores neither mode shapes nor summaries.
    """
    data = np.load(dataset_path)
    fields = np.asarray(data["fields_values"], dtype=np.float64)
    freqs = np.asarray(data["meas_frequencies"], dtype=np.float64)
    train_idx = np.asarray(data["split_train"], dtype=int)
    test_idx = np.asarray(data["split_test"], dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"{dataset_path}: need non-empty train and test splits, got "
            f"{len(train_idx)} train / {len(test_idx)} test rows"
        )
    log_freqs = np.log(np.maximum(freqs, 1e-12))

    if freq_only_mse is None:
        freq_only_mse = _ridge_field_mse(
            log_freqs[train_idx], fields[train_idx],
            log_freqs[test_idx], fields[test_idx],
        )

    if shape_mse is None:
        shape_key = next(
            (k for k in ("meas_mode_shapes", "meas_summaries", "summaries")
             if k in data.files),
            None,
        )
        if shape_key is None:
            shape_mse = None
        else:
            feats = np.asarray(data[shape_key], dtype=np.float64)
            shape_mse = _ridge_field_mse(
                feats.reshape(len(feats), -1)[train_idx], fields[train_idx],
                feats.reshape(len(feats), -1)[test_idx], fields[test_idx],
            )

    # Frequency Jacobian ranks (and CRB floor) on the first n_samples fields.
    sample_fields = fields[:n_samples]
    ranks = []
    crb_floors = []
    ref = np.asarray(_reference_frequencies())
    for field in sample_fields:
        J = frequency_jacobian(field)
        ranks.append(effective_rank(J))
        crb_floors.append(cramer_rao_floor(J, 0.02 * ref))
    if freq_crb_floor is None:
        freq_crb_floor = float(np.median(crb_floors))

    if shape_mse is None:
        shape_better = False
        shape_below_bar = False
        improvement = None
        conclusion = (
            "Shape channel unavailable in this dataset (no stored mode "
            "shapes or summaries); frequency-only verdict stands."
        )
    else:
        shape_better = bool(shape_mse < freq_only_mse)
        shape_below_bar = bool(shape_mse < 0.030)
        improvement = (
            freq_only_mse / shape_mse if shape_mse > 0 else None
        )
        conclusion = (
            "Shape channel carries exploitable spatial signal that the "
            "frequency channel lacks; freq-only CRB floor exceeds the bar."
            if shape_better and freq_crb_floor > 0.030
            else "Shape channel does not clearly improve over freq-only."
        )
    results = {
        "freq_only_ridge_mse": freq_only_mse,
        "shape_ridge_mse": shape_mse,
        "improvement_ratio": improvement,
        "freq_crb_floor_at_2pct_noise": freq_crb_floor,
        "shape_better": shape_better,
        "shape_mse_below_bar": shape_below_bar,
        "freq_crb_above_bar": bool(freq_crb_floor > 0.030),
        "conclusion": conclusion,
    }

    results["freq_jacobian_effective_ranks"] = ranks
    results["freq_jacobian_median_rank"] = int(np.median(ranks))
    results["expected_rank_if_full"] = int(
        sample_fields.shape[1] * sample_fields.shape[2]
    )
    results["nullspace_dims"] = [
        int(sample_fields.shape[1] * sample_fields.shape[2]) - r
        for r in ranks
    ]
    return results
