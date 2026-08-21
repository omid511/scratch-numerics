"""Bridge real Part-1 HF correction data into the multifidelity learning stack.

This module applies the Phase 2/3/5 roadmap to RECORDED Part-1 high-fidelity
simulation data (not synthetic data): PCA compression of correction fields,
a GP from design parameters theta to latent codes, and held-out evaluation.

Provenance: every public entry point loads recorded HF simulation outputs via
``mechanics.p1_multifidelity.hf_dataset`` (contract: ``PART1_PARAMS``,
``load_design``, ``load_correction_fields``, ``load_frequency_errors``,
``GRID_RES``). Metrics reported here therefore describe generalization to
unseen *recorded* designs, not synthetic constructions.

Charter rule enforced throughout: data splits are BY RUN. All modes of a run
stay together in train or test; a run never appears in both sets.

Coverage caveat: ``coverage_proxy_2sigma`` is a PROXY, NOT a calibrated
credible/coverage interval. The GP posterior variance lives in latent space;
we propagate it through the linear PCA decoder as an approximate pointwise
field std and count elementwise hits of a +/-2-sigma interval. No calibration
(e.g. conformal or posterior recalibration) is performed, so values should be
read as a rough uncertainty-quality diagnostic only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .decoder import CorrectionDecoder
from .encoder import CorrectionEncoder
from .gp_model import LatentGP, RBFKernel


@dataclass
class RealPipelineConfig:
    """Configuration for the real-data field-correction pipeline.

    ``test_fraction`` is interpreted as a fraction of HELD-OUT RUNS (charter
    rule: split by run, not by field/mode sample).
    """

    d_z: int = 16
    length_scale: float = 1.0
    noise: float = 1e-4
    seed: int = 42
    test_fraction: float = 0.2
    standardize_theta: bool = True


# ---------------------------------------------------------------------------
# Private array-level helpers (tests exercise these directly on in-memory
# arrays; public functions load from disk and delegate here).
# ---------------------------------------------------------------------------


def _split_run_indices(n_runs: int, config: RealPipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Seeded split of run indices into (train_idx, test_idx).

    ``test_fraction`` is a fraction of runs, rounded UP (ceil). Guarantees:
    disjoint index sets whose union is ``arange(n_runs)``, at least one train
    run whenever ``n_runs >= 2``.
    """
    if n_runs < 2:
        raise ValueError(f"need at least 2 runs to split, got {n_runs}")
    n_test = int(math.ceil(config.test_fraction * n_runs))
    n_test = min(max(n_test, 0), n_runs - 1)
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n_runs)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def _standardize_fit(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std for theta standardization (std=1 where degenerate)."""
    mu = theta.mean(axis=0)
    sd = theta.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def _fields_to_samples(fields: np.ndarray) -> np.ndarray:
    """(n_runs, n_modes, ny, nx) -> per-(run, mode) samples (n_runs*n_modes, ny, nx).

    The encoder flattens grids internally, so samples stay as (ny, nx) grids.
    """
    n_runs, n_modes = fields.shape[0], fields.shape[1]
    return fields.reshape(n_runs * n_modes, *fields.shape[2:])


def _sample_run_ids(n_runs: int, n_modes: int) -> np.ndarray:
    """Run id of each flattened per-(run, mode) sample."""
    return np.repeat(np.arange(n_runs), n_modes)


def _fit_on(
    theta: np.ndarray,
    fields: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: RealPipelineConfig,
) -> dict:
    """Fit PCA encoder + theta->latent GP on TRAIN runs, evaluate on TEST runs.

    Args:
        theta: (n_runs, d_theta) design parameters.
        fields: (n_runs, n_modes, ny, nx) correction fields.
        train_idx / test_idx: run indices; MUST be disjoint (split-by-run).

    Returns dict with 'metrics', 'models', 'arrays' (see run_field_pipeline).
    """
    if set(train_idx.tolist()) & set(test_idx.tolist()):
        raise ValueError("train and test runs overlap: split must be by run")

    n_runs, n_modes, ny, nx = fields.shape
    X = _fields_to_samples(fields)  # (n_runs*n_modes, ny, nx)

    # Flattened-sample indices belonging to each run set.
    run_ids = _sample_run_ids(n_runs, n_modes)
    train_mask = np.isin(run_ids, train_idx)
    test_mask = np.isin(run_ids, test_idx)

    X_train, X_test = X[train_mask], X[test_mask]

    # --- Encoder: fit on TRAIN fields only ---
    encoder = CorrectionEncoder(d_z=config.d_z, grid_size=(ny, nx))
    encoder.fit(X_train)
    z_train = encoder.encode(X_train)  # (n_train_samples, d_z_eff)
    decoder = CorrectionDecoder(encoder)

    # --- Theta handling: repeat per mode; standardize with TRAIN stats ---
    theta_train_runs = theta[train_idx]
    theta_test_runs = theta[test_idx]
    if config.standardize_theta:
        mu, sd = _standardize_fit(theta_train_runs)
        theta_train_s = (theta_train_runs - mu) / sd
        theta_test_s = (theta_test_runs - mu) / sd
    else:
        theta_train_s = theta_train_runs
        theta_test_s = theta_test_runs
    theta_per_sample_train = np.repeat(theta_train_s, n_modes, axis=0)

    # --- GP: standardized/repeated theta -> latent z ---
    gp = LatentGP(
        d_z=z_train.shape[1],
        kernel=RBFKernel(length_scale=config.length_scale),
        noise=config.noise,
    )
    gp.fit(theta_per_sample_train, z_train)

    # --- Held-out prediction: one latent code per RUN (theta identical across
    # modes), broadcast to each mode of that run for field-level comparison ---
    z_mean, z_var = gp.predict(theta_test_s)  # (n_test_runs, d_z), (n_test_runs, d_z)
    n_rep = n_modes
    z_pred_samples = np.repeat(z_mean, n_rep, axis=0)  # aligned with X_test order
    z_var_samples = np.repeat(z_var, n_rep, axis=0)

    decoded = decoder.decode(z_pred_samples, apply_boundary=True)  # (n_test_samples, ny, nx)
    true_fields = X_test
    err = decoded - true_fields

    reconstruction_mse = float(np.mean(err**2))

    pointwise_bias_map = err.mean(axis=0)  # (ny, nx) mean signed error

    # Coverage proxy: propagate latent variance through the LINEAR PCA decode.
    # decode(z) = (z @ C + mean); Var[pixel] = sum_d Var[z_d] * C_d[pixel]^2.
    comp = encoder.components_.reshape(-1, ny, nx)  # (d_z_eff, ny, nx)
    var_field = np.einsum("nd,dhw->hw", z_var_samples, comp**2)
    sigma_field = np.sqrt(var_field)
    covered = np.abs(err) <= 2.0 * sigma_field
    coverage_proxy_2sigma = float(covered.mean())

    per_mode_mse = np.zeros(n_modes)
    test_sample_run_ids = run_ids[test_mask]
    for m in range(n_modes):
        # Samples are ordered run-major within each run set: modes contiguous.
        sel = np.arange(m, len(test_sample_run_ids), n_modes)
        per_mode_mse[m] = float(np.mean(err[sel] ** 2))

    metrics = {
        "reconstruction_mse": reconstruction_mse,
        "coverage_proxy_2sigma": coverage_proxy_2sigma,
        "n_train_runs": int(len(train_idx)),
        "n_test_runs": int(len(test_idx)),
        "d_z": int(z_train.shape[1]),
        "per_mode_mse": per_mode_mse,
    }
    models = {"encoder": encoder, "decoder": decoder, "gp": gp}
    arrays = {
        "z_train": z_train,
        "z_test_pred_mean": z_mean,
        "z_test_pred_var": z_var,
        "theta_test": theta_test_runs,
    }
    return {"metrics": metrics, "models": models, "arrays": arrays}

def _loo_cv_on_arrays(
    theta: np.ndarray,
    fields: np.ndarray,
    config: RealPipelineConfig,
    max_runs: int | None = None,
) -> dict:
    """Leave-one-run-out CV over in-memory arrays (public fn delegates here)."""
    n_runs = theta.shape[0]
    if max_runs is not None:
        max_runs = min(max_runs, n_runs)
        theta, fields = theta[:max_runs], fields[:max_runs]
        n_runs = max_runs
    if n_runs < 3:
        raise ValueError(f"LOO needs at least 3 runs, got {n_runs}")

    per_fold_mse: list[float] = []
    per_fold_coverage: list[float] = []
    all_idx = np.arange(n_runs)
    for i in range(n_runs):
        train_idx = np.delete(all_idx, i)
        test_idx = np.array([i])
        result = _fit_on(theta, fields, train_idx, test_idx, config)
        per_fold_mse.append(result["metrics"]["reconstruction_mse"])
        per_fold_coverage.append(result["metrics"]["coverage_proxy_2sigma"])

    return {
        "mean_reconstruction_mse": float(np.mean(per_fold_mse)),
        "per_fold_mse": per_fold_mse,
        "mean_coverage_proxy": float(np.mean(per_fold_coverage)),
    }


def _frequency_gp_on_arrays(
    theta: np.ndarray,
    rel_err_pct: np.ndarray,
    config: RealPipelineConfig,
) -> dict:
    """LOO GP per frequency mode: standardized theta -> rel_err_pct[:, m].

    Also computes a mean-baseline (predict the TRAIN-mean error) per mode so
    the GP's skill can be judged against the trivial predictor.
    """
    theta = np.asarray(theta, dtype=np.float64)
    rel_err_pct = np.asarray(rel_err_pct, dtype=np.float64)
    n, n_modes = rel_err_pct.shape
    if theta.shape[0] != n:
        raise ValueError("theta rows must match rel_err_pct rows")
    if n < 3:
        raise ValueError(f"need at least 3 designs for LOO, got {n}")

    per_mode: list[dict] = []
    for m in range(n_modes):
        y = rel_err_pct[:, m]
        preds = np.empty(n)
        pred_vars = np.empty(n)
        baseline_sq_errs = np.empty(n)
        for i in range(n):
            tr = np.delete(np.arange(n), i)
            te = np.array([i])
            th_tr = theta[tr]
            if config.standardize_theta:
                mu, sd = _standardize_fit(th_tr)
                th_tr_s = (th_tr - mu) / sd
                th_te_s = (theta[te] - mu) / sd
            else:
                th_tr_s, th_te_s = th_tr, theta[te]
            gp = LatentGP(
                d_z=1,
                kernel=RBFKernel(length_scale=config.length_scale),
                noise=config.noise,
            )
            gp.fit(th_tr_s, y[tr][:, np.newaxis])
            mean_i, var_i = gp.predict(th_te_s)
            preds[i] = mean_i[0, 0]
            pred_vars[i] = var_i[0, 0]
            baseline_sq_errs[i] = (y[tr].mean() - y[i]) ** 2

        errs = preds - y
        loo_rmse_pct = float(np.sqrt(np.mean(errs**2)))
        loo_mae_pct = float(np.mean(np.abs(errs)))
        loo_coverage = float(np.mean(np.abs(errs) <= 2.0 * np.sqrt(pred_vars)))
        baseline_rmse_pct = float(np.sqrt(np.mean(baseline_sq_errs)))

        per_mode.append(
            {
                "loo_rmse_pct": loo_rmse_pct,
                "loo_mae_pct": loo_mae_pct,
                "loo_coverage_2sigma": loo_coverage,
                "baseline_rmse_pct": baseline_rmse_pct,
            }
        )

    mean_loo_rmse = float(np.mean([d["loo_rmse_pct"] for d in per_mode]))
    baseline_mean_rmse = float(np.mean([d["baseline_rmse_pct"] for d in per_mode]))
    return {
        "per_mode": per_mode,
        "mean_loo_rmse_pct": mean_loo_rmse,
        "baseline_mean_rmse_pct": baseline_mean_rmse,
    }


# ---------------------------------------------------------------------------
# Public entry points (load recorded Part-1 data, delegate to helpers above).
# ---------------------------------------------------------------------------


def _load_hf_dataset():
    """Lazy import of hf_dataset so this module imports without it present."""
    try:
        from . import hf_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "mechanics.p1_multifidelity.hf_dataset is required to load real "
            "Part-1 data but is not available; use the private *_on_arrays "
            "helpers for in-memory operation."
        ) from exc
    return hf_dataset


def run_field_pipeline(data_root, config: RealPipelineConfig | None = None, max_modes: int = 10) -> dict:
    """Full field-correction MVP on recorded Part-1 data.

    Loads design params and correction fields, splits runs by a seeded choice
    (held-out fraction = ceil(test_fraction * n_runs) RUNS), fits the PCA
    correction encoder on TRAIN fields only, fits a GP theta->z, then decodes
    held-out predictions and reports metrics.

    Args:
        data_root: root directory containing the recorded Part-1 dataset.
        config: pipeline configuration (defaults constructed if None).
        max_modes: truncate to the first ``max_modes`` correction-field modes.

    Returns dict with keys:
        'metrics': reconstruction_mse, coverage_proxy_2sigma (PROXY only —
            see module docstring), n_train_runs, n_test_runs, d_z,
            per_mode_mse (array over modes).
        'models': {'encoder', 'decoder', 'gp'} fitted objects.
        'arrays': {'z_train', 'z_test_pred_mean', 'z_test_pred_var',
            'theta_test'}.
    """
    hf_dataset = _load_hf_dataset()
    config = config or RealPipelineConfig()

    theta = np.asarray(hf_dataset.load_design(data_root), dtype=np.float64)
    fields = np.asarray(hf_dataset.load_correction_fields(data_root), dtype=np.float64)
    if max_modes is not None:
        fields = fields[:, :max_modes]

    train_idx, test_idx = _split_run_indices(theta.shape[0], config)
    return _fit_on(theta, fields, train_idx, test_idx, config)


def leave_one_run_out_cv(data_root, config: RealPipelineConfig | None = None, max_runs: int | None = None) -> dict:
    """Leave-one-run-out cross-validation on recorded Part-1 data.

    Each fold holds out exactly one run (all its modes together); the rest are
    used for fitting. ``max_runs`` limits the number of runs processed (first
    ``max_runs`` runs) for smoke tests / quick estimates.

    Returns {'mean_reconstruction_mse', 'per_fold_mse', 'mean_coverage_proxy'}.
    """
    hf_dataset = _load_hf_dataset()
    config = config or RealPipelineConfig()

    theta = np.asarray(hf_dataset.load_design(data_root), dtype=np.float64)
    fields = np.asarray(hf_dataset.load_correction_fields(data_root), dtype=np.float64)
    return _loo_cv_on_arrays(theta, fields, config, max_runs=max_runs)


def fit_frequency_error_gp(data_root, config: RealPipelineConfig | None = None) -> dict:
    """Scalar-correction quick win: GP from theta to per-mode relative error (%).

    For each frequency mode m, fits a single-output GP mapping standardized
    theta to ``rel_err_pct[:, m]``, evaluated leave-one-design-out. A
    mean-baseline (predicting the train-mean error) is included for honest
    comparison: the GP should beat it or the gap should be explainable.

    Returns {'per_mode': [{'loo_rmse_pct', 'loo_mae_pct',
    'loo_coverage_2sigma', 'baseline_rmse_pct'}, ...], 'mean_loo_rmse_pct',
    'baseline_mean_rmse_pct'}.
    """
    hf_dataset = _load_hf_dataset()
    config = config or RealPipelineConfig()

    theta = np.asarray(hf_dataset.load_design(data_root), dtype=np.float64)
    # load_frequency_errors returns {'fsdt','comsol','abs_err','rel_err_pct'};
    # the GP target is the relative-error matrix (n_designs, n_modes).
    freq_errors = hf_dataset.load_frequency_errors(data_root)
    rel_err_pct = np.asarray(freq_errors["rel_err_pct"], dtype=np.float64)
    return _frequency_gp_on_arrays(theta, rel_err_pct, config)
