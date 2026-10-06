#!/usr/bin/env python3
"""CPU-only Proposal 1 surrogate models and grouped validation.

The module implements the remaining P1 MVP without an optional deep-learning
runtime.  It uses a boundary-enforced low-rank spatial decoder, exact ARD
Gaussian processes over design parameters, a direct coordinate-conditioned
Fourier-feature INR baseline, and a linear autoregressive co-kriging baseline.
All validation splits are by ``run_id``; accepted mode rows from one design
never cross a split boundary.

The correction target is the accepted ``HF - LF`` max-normalized, phase-aligned
field written by ``compute_corrections.py``.  The decoder represents

    delta(x, y) = b(x, y) * delta_hat(x, y)

where ``b`` is the supplied CCCC ``boundary_mask`` normalized by its maximum
(the scalar normalization is absorbed into ``delta_hat``).  The perimeter is
therefore exactly zero in predictions and is excluded from training metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from p1_pairing import PARAMETERS, boundary_mask


PARAMETER_NAMES = tuple(PARAMETERS)
MODEL_VERSION = "p1-surrogate-v1"


# ---------------------------------------------------------------------------
# Data and grouped splitting
# ---------------------------------------------------------------------------


@dataclass
class P1Dataset:
    """Accepted-only P1 correction handoff loaded from ``training.npz``."""

    x: np.ndarray
    y: np.ndarray
    boundary_mask: np.ndarray
    interior_mask: np.ndarray
    lf: np.ndarray
    hf: np.ndarray
    correction: np.ndarray
    run_ids: np.ndarray
    mode_ids: np.ndarray
    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    f_lf: np.ndarray
    f_hf: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(len(self.run_ids))

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    @property
    def modes(self) -> np.ndarray:
        return np.unique(self.mode_ids.astype(int))

    @property
    def runs(self) -> np.ndarray:
        return np.unique(self.run_ids.astype(int))

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "P1Dataset":
        path = Path(path)
        training = path / "training.npz" if path.is_dir() else path
        if not training.exists():
            raise FileNotFoundError(f"P1 training artifact not found: {training}")
        required = {
            "x", "y", "boundary_mask", "interior_mask", "lf", "hf",
            "correction", "run_ids", "mode_ids", "parameters",
            "parameter_names", "f_lf", "f_hf",
        }
        with np.load(training, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"training.npz missing fields: {sorted(missing)}")
            data = {key: archive[key] for key in required}

        names = tuple(str(value) for value in data["parameter_names"])
        if not names or len(set(names)) != len(names):
            raise ValueError("parameter_names must be nonempty and unique")
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        mask = np.asarray(data["boundary_mask"], dtype=float)
        interior = np.asarray(data["interior_mask"], dtype=bool)
        if x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(y) < 3:
            raise ValueError("P1 grid axes must be one-dimensional and nontrivial")
        if (not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
            raise ValueError("P1 grid axes must be finite and strictly increasing")
        expected_mask = boundary_mask(x, y)
        if mask.shape != (len(y), len(x)) or not np.allclose(
            mask, expected_mask, rtol=0, atol=1e-13
        ):
            raise ValueError("training boundary_mask does not match x/y")
        if interior.shape != mask.shape or not np.array_equal(interior, mask > 0):
            raise ValueError("interior_mask must equal boundary_mask > 0")
        if not np.any(interior) or not np.any(~interior):
            raise ValueError("P1 grid must contain interior and perimeter points")

        fields = np.asarray(data["correction"], dtype=float)
        lf = np.asarray(data["lf"], dtype=float)
        hf = np.asarray(data["hf"], dtype=float)
        n = len(np.asarray(data["run_ids"]))
        expected_field_shape = (n, len(y), len(x))
        if fields.shape != expected_field_shape or lf.shape != expected_field_shape:
            raise ValueError("P1 fields have inconsistent shapes")
        if hf.shape != expected_field_shape:
            raise ValueError("P1 HF field has inconsistent shape")
        if not (np.isfinite(fields).all() and np.isfinite(lf).all()
                and np.isfinite(hf).all()):
            raise ValueError("P1 fields must be finite")
        if not np.allclose(hf - lf, fields, rtol=2e-9, atol=2e-12):
            raise ValueError("P1 correction field is not exactly HF - LF")

        run_ids = np.asarray(data["run_ids"], dtype=int)
        mode_ids = np.asarray(data["mode_ids"], dtype=int)
        parameters = np.asarray(data["parameters"], dtype=float)
        f_lf = np.asarray(data["f_lf"], dtype=float)
        f_hf = np.asarray(data["f_hf"], dtype=float)
        if (run_ids.shape != (n,) or mode_ids.shape != (n,)
                or parameters.shape != (n, len(names))
                or f_lf.shape != (n,) or f_hf.shape != (n,)):
            raise ValueError("P1 row metadata has inconsistent shapes")
        if (not np.isfinite(parameters).all() or not np.isfinite(f_lf).all()
                or not np.isfinite(f_hf).all() or np.any(f_lf <= 0)
                or np.any(f_hf <= 0) or np.any(run_ids < 1)
                or np.any(mode_ids < 1)):
            raise ValueError("P1 metadata contains invalid values")
        if len(np.unique(run_ids)) < 2:
            raise ValueError("Grouped validation needs at least two runs")
        return cls(
            x=x, y=y, boundary_mask=mask, interior_mask=interior,
            lf=lf, hf=hf, correction=fields, run_ids=run_ids,
            mode_ids=mode_ids, parameters=parameters, parameter_names=names,
            f_lf=f_lf, f_hf=f_hf,
        )

    def select_runs(self, run_ids: Iterable[int]) -> "P1Dataset":
        """Return a deterministic accepted-row subset for smoke experiments."""
        selected = np.asarray(sorted({int(run) for run in run_ids}), dtype=int)
        if not len(selected):
            raise ValueError("At least one run is required")
        rows = np.flatnonzero(np.isin(self.run_ids, selected))
        if not len(rows):
            raise ValueError("Selected runs have no accepted rows")
        return P1Dataset(
            x=self.x, y=self.y, boundary_mask=self.boundary_mask,
            interior_mask=self.interior_mask, lf=self.lf[rows], hf=self.hf[rows],
            correction=self.correction[rows], run_ids=self.run_ids[rows],
            mode_ids=self.mode_ids[rows], parameters=self.parameters[rows],
            parameter_names=self.parameter_names, f_lf=self.f_lf[rows],
            f_hf=self.f_hf[rows],
        )


def grouped_run_splits(
    run_ids: Sequence[int], n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create deterministic run-level folds and row indices.

    Returns ``(train_rows, validation_rows, train_runs, validation_runs)``.
    The folds are balanced by run count, not by accepted mode-row count.
    """
    run_ids = np.asarray(run_ids, dtype=int)
    unique = np.unique(run_ids)
    if len(unique) < 2:
        raise ValueError("At least two unique runs are required")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    n_splits = min(n_splits, len(unique))
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    folds = np.array_split(shuffled, n_splits)
    result = []
    all_rows = np.arange(len(run_ids), dtype=int)
    for validation_runs in folds:
        validation_runs = np.asarray(sorted(validation_runs.tolist()), dtype=int)
        validation_mask = np.isin(run_ids, validation_runs)
        validation_rows = all_rows[validation_mask]
        train_rows = all_rows[~validation_mask]
        train_runs = np.asarray(sorted(np.unique(run_ids[~validation_mask]).tolist()), dtype=int)
        if not len(train_rows) or not len(validation_rows):
            raise ValueError("Grouped split produced an empty train/validation fold")
        result.append((train_rows, validation_rows, train_runs, validation_runs))
    return result


# ---------------------------------------------------------------------------
# Scaling, randomized spatial decoder, and exact GP
# ---------------------------------------------------------------------------


@dataclass
class ParameterScaler:
    lower: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ParameterScaler":
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Parameter values must be a finite two-dimensional array")
        lower = values.min(axis=0)
        scale = values.max(axis=0) - lower
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(lower=lower, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.lower):
            raise ValueError("Parameter shape does not match fitted scaler")
        return (values - self.lower) / self.scale

    def state(self) -> dict[str, list[float]]:
        return {"lower": self.lower.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Sequence[float]]) -> "ParameterScaler":
        return cls(np.asarray(state["lower"], dtype=float), np.asarray(state["scale"], dtype=float))


def randomized_svd(
    matrix: np.ndarray, rank: int, seed: int = 42, oversample: int = 8,
    power_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leading singular values/vectors without forming a huge covariance."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("SVD input must be a finite matrix")
    n, p = matrix.shape
    if n < 1 or p < 1:
        raise ValueError("SVD input cannot be empty")
    rank = min(int(rank), n, p)
    if rank < 1:
        raise ValueError("SVD rank must be positive")
    width = min(rank + max(0, int(oversample)), n, p)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((p, width))
    q = np.linalg.qr(matrix @ omega, mode="reduced")[0]
    for _ in range(max(0, int(power_iterations))):
        q = np.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")[0]
    projected = q.T @ matrix
    _, singular_values, components = np.linalg.svd(projected, full_matrices=False)
    return singular_values[:rank], components[:rank]


class BoundaryLatentDecoder:
    """Low-rank decoder with an exact perimeter boundary projection."""

    def __init__(self, latent_dim: int = 48, seed: int = 42):
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.boundary_mask: np.ndarray | None = None
        self.interior_mask: np.ndarray | None = None
        self.envelope: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.residual_variance: np.ndarray | None = None

    def fit(self, dataset: P1Dataset, rows: Sequence[int]) -> "BoundaryLatentDecoder":
        rows = np.asarray(rows, dtype=int)
        if not len(rows):
            raise ValueError("Decoder needs at least one training row")
        self.x, self.y = dataset.x.copy(), dataset.y.copy()
        self.boundary_mask = dataset.boundary_mask.copy()
        self.interior_mask = dataset.interior_mask.copy()
        scale = float(np.max(self.boundary_mask))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Boundary mask has no positive interior")
        self.envelope = self.boundary_mask / scale
        target = dataset.correction[rows][:, self.interior_mask]
        target = target / self.envelope[self.interior_mask][None, :]
        if not np.isfinite(target).all():
            raise ValueError("Boundary-enforced decoder target is nonfinite")
        self.mean = target.mean(axis=0)
        centered = target - self.mean
        self.singular_values, self.components = randomized_svd(
            centered, self.latent_dim, seed=self.seed
        )
        latent = centered @ self.components.T
        reconstructed = self.mean + latent @ self.components
        residual = target - reconstructed
        # Store output-space residual variance so reported intervals include
        # deterministic decoder truncation error in addition to GP uncertainty.
        residual_output = residual * self.envelope[self.interior_mask][None, :]
        self.residual_variance = np.var(residual_output, axis=0)
        return self

    def _check_fit(self) -> None:
        if any(value is None for value in (
            self.envelope, self.interior_mask, self.mean, self.components,
            self.residual_variance,
        )):
            raise RuntimeError("Decoder has not been fitted")

    def encode(self, fields: np.ndarray) -> np.ndarray:
        self._check_fit()
        fields = np.asarray(fields, dtype=float)
        if fields.ndim != 3 or fields.shape[1:] != self.envelope.shape:
            raise ValueError("Field shape does not match fitted decoder")
        raw = fields[:, self.interior_mask] / self.envelope[self.interior_mask][None, :]
        return (raw - self.mean) @ self.components.T

    def decode(self, latent: np.ndarray) -> np.ndarray:
        self._check_fit()
        latent = np.asarray(latent, dtype=float)
        if latent.ndim != 2 or latent.shape[1] != self.components.shape[0]:
            raise ValueError("Latent shape does not match fitted decoder")
        raw = self.mean[None, :] + latent @ self.components
        fields = np.zeros((len(latent), *self.envelope.shape), dtype=float)
        fields[:, self.interior_mask] = raw * self.envelope[self.interior_mask][None, :]
        return fields

    def decode_std(self, latent_std: np.ndarray) -> np.ndarray:
        """Propagate independent latent standard deviations to field space."""
        self._check_fit()
        latent_std = np.asarray(latent_std, dtype=float)
        if latent_std.ndim != 2 or latent_std.shape[1] != self.components.shape[0]:
            raise ValueError("Latent standard-deviation shape mismatch")
        variance = (latent_std**2) @ (self.components**2)
        variance *= self.envelope[self.interior_mask][None, :] ** 2
        variance += self.residual_variance[None, :]
        fields = np.zeros((len(latent_std), *self.envelope.shape), dtype=float)
        fields[:, self.interior_mask] = np.sqrt(np.maximum(variance, 0.0))
        return fields

    def reconstruction(self, fields: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(fields))

    def reconstruction_summary(self, fields: np.ndarray) -> dict[str, float]:
        prediction = self.reconstruction(fields)
        interior = self.interior_mask
        error = prediction - fields
        return {
            "interior_rms_median": float(np.median(np.sqrt(np.mean(error[:, interior] ** 2, axis=1)))),
            "interior_rms_p95": float(np.quantile(np.sqrt(np.mean(error[:, interior] ** 2, axis=1)), .95)),
            "edge_rms_median": float(np.median(np.sqrt(np.mean(error[:, ~interior] ** 2, axis=1)))),
        }

    def state_arrays(self) -> dict[str, np.ndarray]:
        self._check_fit()
        return {
            "x": self.x, "y": self.y, "boundary_mask": self.boundary_mask,
            "interior_mask": self.interior_mask, "envelope": self.envelope,
            "mean": self.mean, "components": self.components,
            "singular_values": self.singular_values,
            "residual_variance": self.residual_variance,
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], latent_dim: int, seed: int) -> "BoundaryLatentDecoder":
        decoder = cls(latent_dim=latent_dim, seed=seed)
        for key in ("x", "y", "boundary_mask", "interior_mask", "envelope", "mean", "components", "singular_values", "residual_variance"):
            setattr(decoder, key, np.asarray(arrays[key]))
        decoder.interior_mask = decoder.interior_mask.astype(bool)
        return decoder


class ExactMultiOutputGP:
    """Exact ARD RBF GP with shared kernel and independent output columns."""

    def __init__(self, max_iterations: int = 80):
        self.max_iterations = int(max_iterations)
        self.x_train: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_scale: np.ndarray | None = None
        self.length_scales: np.ndarray | None = None
        self.signal: float | None = None
        self.noise: float | None = None
        self.cholesky: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.optimization_result: dict[str, object] = {}

    @staticmethod
    def _kernel(x1: np.ndarray, x2: np.ndarray, length_scales: np.ndarray, signal: float) -> np.ndarray:
        delta = (x1[:, None, :] - x2[None, :, :]) / length_scales[None, None, :]
        return signal**2 * np.exp(-0.5 * np.sum(delta * delta, axis=2))

    @staticmethod
    def _factor(matrix: np.ndarray) -> np.ndarray:
        jitter = 1e-10
        eye = np.eye(len(matrix))
        for _ in range(7):
            try:
                return np.linalg.cholesky(matrix + jitter * eye)
            except np.linalg.LinAlgError:
                jitter *= 10.0
        raise np.linalg.LinAlgError("Unable to factor GP covariance")

    def _objective(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        d = x.shape[1]
        length_scales = np.exp(theta[:d])
        signal = float(np.exp(theta[d]))
        noise = float(np.exp(theta[d + 1]))
        covariance = self._kernel(x, x, length_scales, signal)
        covariance.flat[:: len(covariance) + 1] += noise**2
        try:
            lower = self._factor(covariance)
        except np.linalg.LinAlgError:
            return 1e100
        alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, y))
        logdet = 2.0 * np.log(np.maximum(np.diag(lower), 1e-300)).sum()
        value = .5 * float(np.sum(y * alpha))
        value += .5 * y.shape[1] * logdet
        value += .5 * y.shape[0] * y.shape[1] * math.log(2.0 * math.pi)
        return value if np.isfinite(value) else 1e100

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, optimize_columns: int | None = None,
        seed: int = 42,
    ) -> "ExactMultiOutputGP":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("GP requires matching two-dimensional x/y with at least two rows")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("GP training values must be finite")
        self.x_train = x.copy()
        self.y_mean = y.mean(axis=0)
        self.y_scale = y.std(axis=0)
        self.y_scale = np.where(self.y_scale > 1e-10, self.y_scale, 1.0)
        standardized = (y - self.y_mean) / self.y_scale
        if optimize_columns is not None and optimize_columns < standardized.shape[1]:
            rng = np.random.default_rng(seed)
            columns = np.sort(rng.choice(standardized.shape[1], int(optimize_columns), replace=False))
            objective_y = standardized[:, columns]
        else:
            columns = np.arange(standardized.shape[1])
            objective_y = standardized
        d = x.shape[1]
        initial = np.r_[np.log(np.full(d, .35)), 0.0, np.log(.08)]
        bounds = [(math.log(.03), math.log(5.0))] * d
        bounds += [(math.log(.03), math.log(5.0)), (math.log(1e-5), math.log(1.0))]
        result = minimize(
            self._objective, initial, args=(x, objective_y), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": self.max_iterations, "ftol": 1e-8},
        )
        theta = result.x if np.isfinite(result.fun) else initial
        self.length_scales = np.exp(theta[:d])
        self.signal = float(np.exp(theta[d]))
        self.noise = float(np.exp(theta[d + 1]))
        covariance = self._kernel(x, x, self.length_scales, self.signal)
        covariance.flat[:: len(covariance) + 1] += self.noise**2
        self.cholesky = self._factor(covariance)
        self.alpha = np.linalg.solve(
            self.cholesky.T, np.linalg.solve(self.cholesky, standardized)
        )
        self.optimization_result = {
            "success": bool(result.success), "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "objective": float(result.fun), "optimized_columns": int(len(columns)),
        }
        return self

    def _check_fit(self) -> None:
        if any(value is None for value in (
            self.x_train, self.y_mean, self.y_scale, self.length_scales,
            self.signal, self.noise, self.cholesky, self.alpha,
        )):
            raise RuntimeError("GP has not been fitted")

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fit()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.x_train.shape[1] or not np.isfinite(x).all():
            raise ValueError("GP prediction x has the wrong shape or is nonfinite")
        cross = self._kernel(x, self.x_train, self.length_scales, self.signal)
        mean = cross @ self.alpha
        projected = np.linalg.solve(self.cholesky, cross.T)
        variance = self.signal**2 + self.noise**2 - np.sum(projected**2, axis=0)
        variance = np.maximum(variance, 1e-12)
        std = np.sqrt(variance)[:, None] * self.y_scale[None, :]
        return mean * self.y_scale[None, :] + self.y_mean[None, :], std

    def state_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        self._check_fit()
        return {
            f"{prefix}x_train": self.x_train,
            f"{prefix}y_mean": self.y_mean,
            f"{prefix}y_scale": self.y_scale,
            f"{prefix}length_scales": self.length_scales,
            f"{prefix}signal": np.asarray([self.signal]),
            f"{prefix}noise": np.asarray([self.noise]),
            f"{prefix}cholesky": self.cholesky,
            f"{prefix}alpha": self.alpha,
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], prefix: str) -> "ExactMultiOutputGP":
        gp = cls()
        gp.x_train = np.asarray(arrays[f"{prefix}x_train"], dtype=float)
        gp.y_mean = np.asarray(arrays[f"{prefix}y_mean"], dtype=float)
        gp.y_scale = np.asarray(arrays[f"{prefix}y_scale"], dtype=float)
        gp.length_scales = np.asarray(arrays[f"{prefix}length_scales"], dtype=float)
        gp.signal = float(np.asarray(arrays[f"{prefix}signal"])[0])
        gp.noise = float(np.asarray(arrays[f"{prefix}noise"])[0])
        gp.cholesky = np.asarray(arrays[f"{prefix}cholesky"], dtype=float)
        gp.alpha = np.asarray(arrays[f"{prefix}alpha"], dtype=float)
        return gp


# ---------------------------------------------------------------------------
# Latent correction model
# ---------------------------------------------------------------------------


class LatentGPModel:
    """Boundary-enforced decoder followed by one multi-output GP per mode."""

    def __init__(self, latent_dim: int = 48, seed: int = 42, max_iterations: int = 80):
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)
        self.max_iterations = int(max_iterations)
        self.decoder: BoundaryLatentDecoder | None = None
        self.scaler: ParameterScaler | None = None
        self.parameter_names: tuple[str, ...] | None = None
        self.field_gps: dict[int, ExactMultiOutputGP] = {}
        self.frequency_gps: dict[int, ExactMultiOutputGP] = {}
        self.training_rows: np.ndarray | None = None
        self.training_runs: np.ndarray | None = None
        self.training_modes: np.ndarray | None = None
        self.grid_shape: tuple[int, int] | None = None

    def fit(self, dataset: P1Dataset, rows: Sequence[int]) -> "LatentGPModel":
        rows = np.asarray(rows, dtype=int)
        if len(np.unique(dataset.run_ids[rows])) < 2:
            raise ValueError("Latent GP fit needs at least two training runs")
        self.training_rows = rows.copy()
        self.training_runs = np.unique(dataset.run_ids[rows]).astype(int)
        self.training_modes = np.unique(dataset.mode_ids[rows]).astype(int)
        self.grid_shape = dataset.grid_shape
        self.parameter_names = dataset.parameter_names
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.decoder = BoundaryLatentDecoder(self.latent_dim, self.seed).fit(dataset, rows)
        latent = self.decoder.encode(dataset.correction[rows])
        x_scaled = self.scaler.transform(dataset.parameters[rows])
        relative_frequency = (dataset.f_hf[rows] - dataset.f_lf[rows]) / dataset.f_hf[rows]
        for mode in self.training_modes.tolist():
            mode_rows = np.flatnonzero(dataset.mode_ids[rows] == mode)
            if len(mode_rows) < 2:
                raise ValueError(f"Mode {mode} has fewer than two training rows")
            gp = ExactMultiOutputGP(self.max_iterations).fit(
                x_scaled[mode_rows], latent[mode_rows], seed=self.seed + mode
            )
            fgp = ExactMultiOutputGP(self.max_iterations).fit(
                x_scaled[mode_rows], relative_frequency[mode_rows], seed=self.seed + 1000 + mode
            )
            self.field_gps[int(mode)] = gp
            self.frequency_gps[int(mode)] = fgp
        return self

    def _check_fit(self) -> None:
        if self.decoder is None or self.scaler is None or not self.field_gps:
            raise RuntimeError("Latent GP model has not been fitted")

    def predict_design(
        self, parameters: np.ndarray, mode_ids: Sequence[int],
        f_lf: Sequence[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict correction fields for arbitrary new designs and modes."""
        self._check_fit()
        parameters = np.asarray(parameters, dtype=float)
        mode_ids = np.asarray(mode_ids, dtype=int)
        if parameters.ndim != 2 or self.parameter_names is None or parameters.shape[1] != len(self.parameter_names):
            raise ValueError(
                f"Design parameters must have shape (n, {len(self.parameter_names or ())})"
            )
        if len(mode_ids) != len(parameters):
            raise ValueError("mode_ids must match the number of designs")
        if not np.isfinite(parameters).all():
            raise ValueError("Design parameters must be finite")
        if f_lf is None:
            frequencies_lf = np.full(len(parameters), np.nan)
        else:
            frequencies_lf = np.asarray(f_lf, dtype=float)
            if frequencies_lf.shape != (len(parameters),):
                raise ValueError("f_lf must match the number of designs")
        correction = np.full((len(parameters), *self.decoder.envelope.shape), np.nan)
        correction_std = np.full_like(correction, np.nan)
        frequency = np.full(len(parameters), np.nan)
        frequency_std = np.full(len(parameters), np.nan)
        for mode in np.unique(mode_ids).astype(int).tolist():
            local = np.flatnonzero(mode_ids == mode)
            if mode not in self.field_gps:
                raise ValueError(f"No fitted GP for mode {mode}")
            scaled = self.scaler.transform(parameters[local])
            latent_mean, latent_std = self.field_gps[mode].predict(scaled)
            correction[local] = self.decoder.decode(latent_mean)
            correction_std[local] = self.decoder.decode_std(latent_std)
            if np.isfinite(frequencies_lf[local]).all():
                rel_mean, rel_std = self.frequency_gps[mode].predict(scaled)
                rel_mean = np.clip(rel_mean[:, 0], -.8, .8)
                frequency[local] = frequencies_lf[local] / (1.0 - rel_mean)
                frequency_std[local] = (
                    np.abs(frequencies_lf[local] / (1.0 - rel_mean) ** 2)
                    * rel_std[:, 0]
                )
        return {
            "correction": correction, "correction_std": correction_std,
            "frequency": frequency, "frequency_std": frequency_std,
        }

    def predict_rows(self, dataset: P1Dataset, rows: Sequence[int]) -> dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=int)
        return self.predict_design(
            dataset.parameters[rows], dataset.mode_ids[rows], dataset.f_lf[rows]
        )

    def reconstruction_summary(self, dataset: P1Dataset) -> dict[str, float]:
        self._check_fit()
        target = dataset.correction[self.training_rows]
        reconstruction = self.decoder.reconstruction(target)
        error = reconstruction - target
        interior = dataset.interior_mask
        edge = ~interior
        target_variance = float(np.var(target[:, interior], axis=0).sum())
        residual_variance = float(np.var(error[:, interior], axis=0).sum())
        return {
            "interior_rms_median": float(np.median(np.sqrt(np.mean(error[:, interior] ** 2, axis=1)))),
            "interior_rms_p95": float(np.quantile(np.sqrt(np.mean(error[:, interior] ** 2, axis=1)), .95)),
            "edge_rms_median": float(np.median(np.sqrt(np.mean(error[:, edge] ** 2, axis=1)))),
            "latent_dim": int(self.latent_dim),
            "decoder_explained_variance": (
                1.0 - residual_variance / target_variance if target_variance > 0.0 else 1.0
            ),
        }

    def save(self, directory: os.PathLike[str] | str, dataset: P1Dataset) -> None:
        self._check_fit()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = self.decoder.state_arrays()
        metadata = {
            "model_version": MODEL_VERSION,
            "model_type": "latent_gp",
            "latent_dim": self.latent_dim,
            "seed": self.seed,
            "grid_shape": list(dataset.grid_shape),
            "parameter_names": list(self.parameter_names or dataset.parameter_names),
            "training_runs": self.training_runs.tolist(),
            "training_row_indices": self.training_rows.tolist(),
            "training_rows": int(len(self.training_rows)),
            "modes": [int(mode) for mode in sorted(self.field_gps)],
            "scaler": self.scaler.state(),
            "gp_optimization": {
                "field": {str(mode): gp.optimization_result for mode, gp in self.field_gps.items()},
                "frequency": {str(mode): gp.optimization_result for mode, gp in self.frequency_gps.items()},
            },
        }
        for mode, gp in self.field_gps.items():
            arrays.update(gp.state_arrays(f"field_{mode}_"))
        for mode, gp in self.frequency_gps.items():
            arrays.update(gp.state_arrays(f"frequency_{mode}_"))
        np.savez_compressed(directory / "model.npz", **arrays)
        (directory / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: os.PathLike[str] | str) -> "LatentGPModel":
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if metadata.get("model_type") != "latent_gp":
            raise ValueError("Not a latent GP model directory")
        with np.load(directory / "model.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        model = cls(metadata["latent_dim"], metadata["seed"])
        model.parameter_names = tuple(str(value) for value in metadata["parameter_names"])
        model.decoder = BoundaryLatentDecoder.from_arrays(
            arrays, metadata["latent_dim"], metadata["seed"]
        )
        model.scaler = ParameterScaler.from_state(metadata["scaler"])
        model.training_runs = np.asarray(metadata["training_runs"], dtype=int)
        model.training_rows = np.asarray(
            metadata.get("training_row_indices", []), dtype=int
        )
        model.training_modes = np.asarray(metadata["modes"], dtype=int)
        model.grid_shape = tuple(metadata["grid_shape"])
        for mode in model.training_modes.tolist():
            model.field_gps[int(mode)] = ExactMultiOutputGP.from_arrays(arrays, f"field_{mode}_")
            model.frequency_gps[int(mode)] = ExactMultiOutputGP.from_arrays(arrays, f"frequency_{mode}_")
        return model


# ---------------------------------------------------------------------------
# Direct coordinate-conditioned INR baseline
# ---------------------------------------------------------------------------


class FourierINRBaseline:
    """Coordinate-conditioned Fourier-feature ridge residual model.

    This is the portable CPU baseline for the proposal's INR formulation.  It
    has no spatial grid restriction at inference: arbitrary coordinates can be
    queried through the same Fourier feature map.  It is intentionally a
    baseline, not presented as a trained neural MLP.
    """

    def __init__(
        self, n_random_features: int = 64, ridge: float = 1e-4,
        max_points: int = 120_000, seed: int = 42,
    ):
        self.n_random_features = int(n_random_features)
        self.ridge = float(ridge)
        self.max_points = int(max_points)
        self.seed = int(seed)
        self.scaler: ParameterScaler | None = None
        self.parameter_names: tuple[str, ...] | None = None
        self.weights: np.ndarray | None = None
        self.phase: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.boundary_mask: np.ndarray | None = None
        self.interior_mask: np.ndarray | None = None
        self.envelope: np.ndarray | None = None
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.modes: np.ndarray | None = None
        self.training_rows: np.ndarray | None = None
        self.training_points: int = 0
        self.input_dim: int | None = None

    @property
    def feature_count(self) -> int:
        if self.input_dim is None:
            raise RuntimeError("INR input dimension is not initialized")
        return 1 + self.input_dim * 2 + 2 * self.n_random_features

    def _check_fit(self) -> None:
        if any(value is None for value in (
            self.scaler, self.weights, self.phase, self.coefficients,
            self.envelope, self.interior_mask, self.input_dim,
        )):
            raise RuntimeError("INR baseline has not been fitted")

    def _features(
        self, parameters: np.ndarray, mode_ids: np.ndarray, xu: np.ndarray,
        yu: np.ndarray, lf_values: np.ndarray,
    ) -> np.ndarray:
        self._check_fit()
        parameters = np.asarray(parameters, dtype=float)
        mode_ids = np.asarray(mode_ids, dtype=int)
        xu = np.asarray(xu, dtype=float)
        yu = np.asarray(yu, dtype=float)
        lf_values = np.asarray(lf_values, dtype=float)
        n = len(xu)
        if not all(len(value) == n for value in (parameters, mode_ids, yu, lf_values)):
            raise ValueError("INR feature columns have inconsistent lengths")
        if not np.isin(mode_ids, self.modes).all():
            raise ValueError("INR query contains a mode absent from training")
        scaled = self.scaler.transform(parameters)
        mode_onehot = np.zeros((n, len(self.modes)), dtype=float)
        for index, mode in enumerate(self.modes.tolist()):
            mode_onehot[:, index] = mode_ids == mode
        q = np.column_stack([scaled, mode_onehot, xu, yu, lf_values])
        linear = np.column_stack([np.ones(n), q, q * q])
        projection = q @ self.weights + self.phase[None, :]
        return np.column_stack([linear, np.sin(projection), np.cos(projection)])

    def fit(self, dataset: P1Dataset, rows: Sequence[int]) -> "FourierINRBaseline":
        rows = np.asarray(rows, dtype=int)
        if not len(rows):
            raise ValueError("INR baseline needs training rows")
        self.training_rows = rows.copy()
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.boundary_mask = dataset.boundary_mask.copy()
        self.interior_mask = dataset.interior_mask.copy()
        self.envelope = self.boundary_mask / np.max(self.boundary_mask)
        self.x, self.y = dataset.x.copy(), dataset.y.copy()
        self.modes = np.unique(dataset.mode_ids[rows]).astype(int)
        self.parameter_names = dataset.parameter_names
        rng = np.random.default_rng(self.seed)
        self.input_dim = len(dataset.parameter_names) + len(self.modes) + 2 + 1
        self.weights = rng.normal(0.0, 1.5, size=(self.input_dim, self.n_random_features))
        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=self.n_random_features)
        gram = np.zeros((self.feature_count, self.feature_count), dtype=float)
        cross = np.zeros(self.feature_count, dtype=float)
        self.coefficients = np.zeros(self.feature_count, dtype=float)
        interior_flat = np.flatnonzero(self.interior_mask.ravel())
        if len(interior_flat) < 8:
            raise ValueError("INR needs at least eight interior grid points")
        if self.max_points < 8 * len(rows):
            raise ValueError("INR max_points must provide eight points per training row")
        per_row = min(len(interior_flat), self.max_points // len(rows))
        if per_row < 1:
            raise ValueError("INR max_points is too small for the number of rows")
        xu_grid, yu_grid = np.meshgrid(
            (dataset.x - dataset.x[0]) / (dataset.x[-1] - dataset.x[0]),
            (dataset.y - dataset.y[0]) / (dataset.y[-1] - dataset.y[0]),
        )
        for row in rows.tolist():
            selected = np.sort(rng.choice(interior_flat, per_row, replace=False))
            params = np.repeat(dataset.parameters[row][None, :], len(selected), axis=0)
            modes = np.full(len(selected), dataset.mode_ids[row], dtype=int)
            values = dataset.lf[row].ravel()[selected]
            features = self._features(
                params, modes, xu_grid.ravel()[selected], yu_grid.ravel()[selected], values
            )
            target = dataset.correction[row].ravel()[selected] / self.envelope.ravel()[selected]
            gram += features.T @ features
            cross += features.T @ target
            self.training_points += len(selected)
        regularizer = np.eye(self.feature_count) * self.ridge
        regularizer[0, 0] = self.ridge * 1e-3
        try:
            self.coefficients = np.linalg.solve(gram + regularizer, cross)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(gram + regularizer, cross, rcond=None)[0]
        return self

    def predict_points(
        self, parameters: np.ndarray, mode_ids: Sequence[int],
        xu: Sequence[float], yu: Sequence[float], lf_values: Sequence[float],
    ) -> np.ndarray:
        """Query the INR at arbitrary normalized coordinates in ``[0, 1]``."""
        parameters = np.asarray(parameters, dtype=float)
        mode_ids = np.asarray(mode_ids, dtype=int)
        xu = np.asarray(xu, dtype=float)
        yu = np.asarray(yu, dtype=float)
        lf_values = np.asarray(lf_values, dtype=float)
        if (parameters.ndim != 2
                or self.parameter_names is None
                or parameters.shape[1] != len(self.parameter_names)
                or len(mode_ids) != len(parameters) or len(xu) != len(parameters)
                or len(yu) != len(parameters) or len(lf_values) != len(parameters)):
            raise ValueError("INR point-query arrays have inconsistent lengths or parameter width")
        if (not np.isfinite(parameters).all() or not np.isfinite(xu).all()
                or not np.isfinite(yu).all() or not np.isfinite(lf_values).all()
                or np.any((xu < 0) | (xu > 1)) or np.any((yu < 0) | (yu > 1))):
            raise ValueError("INR point queries must be finite and inside the plate")
        raw = np.empty(len(parameters), dtype=float)
        for start in range(0, len(parameters), 8192):
            stop = min(start + 8192, len(parameters))
            features = self._features(
                parameters[start:stop], mode_ids[start:stop], xu[start:stop],
                yu[start:stop], lf_values[start:stop],
            )
            raw[start:stop] = features @ self.coefficients
        envelope = 16.0 * xu * (1.0 - xu) * yu * (1.0 - yu)
        return raw * envelope

    def predict_rows(self, dataset: P1Dataset, rows: Sequence[int]) -> dict[str, np.ndarray]:
        self._check_fit()
        rows = np.asarray(rows, dtype=int)
        xu_grid, yu_grid = np.meshgrid(
            (dataset.x - dataset.x[0]) / (dataset.x[-1] - dataset.x[0]),
            (dataset.y - dataset.y[0]) / (dataset.y[-1] - dataset.y[0]),
        )
        grid_x = xu_grid.ravel()
        grid_y = yu_grid.ravel()
        prediction = np.empty((len(rows), *dataset.grid_shape), dtype=float)
        chunk = 8192
        for local, row in enumerate(rows.tolist()):
            raw = np.empty(len(grid_x), dtype=float)
            for start in range(0, len(grid_x), chunk):
                stop = min(start + chunk, len(grid_x))
                params = np.repeat(dataset.parameters[row][None, :], stop - start, axis=0)
                modes = np.full(stop - start, dataset.mode_ids[row], dtype=int)
                features = self._features(
                    params, modes, grid_x[start:stop], grid_y[start:stop],
                    dataset.lf[row].ravel()[start:stop],
                )
                raw[start:stop] = features @ self.coefficients
            prediction[local] = (raw * self.envelope.ravel()).reshape(dataset.grid_shape)
        return {
            "correction": prediction,
            "correction_std": None,
            "frequency": np.full(len(rows), np.nan),
            "frequency_std": np.full(len(rows), np.nan),
        }

    def save(self, directory: os.PathLike[str] | str) -> None:
        self._check_fit()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "model.npz", x=self.x, y=self.y,
            boundary_mask=self.boundary_mask, interior_mask=self.interior_mask,
            envelope=self.envelope, weights=self.weights, phase=self.phase,
            coefficients=self.coefficients,
        )
        metadata = {
            "model_version": MODEL_VERSION, "model_type": "fourier_inr_baseline",
            "n_random_features": self.n_random_features, "ridge": self.ridge,
            "max_points": self.max_points, "seed": self.seed,
            "parameter_names": list(self.parameter_names or ()),
            "modes": self.modes.tolist(), "training_points": self.training_points,
            "scaler": self.scaler.state(),
        }
        (directory / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plain autoregressive co-kriging baseline
# ---------------------------------------------------------------------------


class LinearCoKrigingBaseline:
    """Plain ``HF = rho * LF + GP residual`` baseline.

    The residual GP predicts all interior pixels jointly with one shared design
    kernel per mode.  This deliberately skips latent compression and is the
    direct multi-fidelity baseline against which the latent model is compared.
    """

    def __init__(self, max_iterations: int = 60, seed: int = 42):
        self.max_iterations = int(max_iterations)
        self.seed = int(seed)
        self.scaler: ParameterScaler | None = None
        self.mode_models: dict[int, tuple[float, ExactMultiOutputGP]] = {}
        self.frequency_gps: dict[int, ExactMultiOutputGP] = {}
        self.interior_mask: np.ndarray | None = None
        self.boundary_mask: np.ndarray | None = None
        self.training_rows: np.ndarray | None = None

    def fit(self, dataset: P1Dataset, rows: Sequence[int]) -> "LinearCoKrigingBaseline":
        rows = np.asarray(rows, dtype=int)
        if len(np.unique(dataset.run_ids[rows])) < 2:
            raise ValueError("Co-kriging fit needs at least two training runs")
        self.training_rows = rows.copy()
        self.scaler = ParameterScaler.fit(dataset.parameters[rows])
        self.interior_mask = dataset.interior_mask.copy()
        self.boundary_mask = dataset.boundary_mask.copy()
        scaled = self.scaler.transform(dataset.parameters[rows])
        interior = self.interior_mask
        for mode in np.unique(dataset.mode_ids[rows]).astype(int).tolist():
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            lf = dataset.lf[rows[local]][:, interior]
            hf = dataset.hf[rows[local]][:, interior]
            rho = float(np.sum(lf * hf) / max(np.sum(lf * lf), 1e-30))
            residual = hf - rho * lf
            gp = ExactMultiOutputGP(self.max_iterations).fit(
                scaled[local], residual, optimize_columns=256,
                seed=self.seed + mode,
            )
            rel = (dataset.f_hf[rows[local]] - dataset.f_lf[rows[local]]) / dataset.f_hf[rows[local]]
            fgp = ExactMultiOutputGP(self.max_iterations).fit(
                scaled[local], rel, seed=self.seed + 1000 + mode,
            )
            self.mode_models[mode] = (rho, gp)
            self.frequency_gps[mode] = fgp
        return self

    def predict_rows(self, dataset: P1Dataset, rows: Sequence[int]) -> dict[str, np.ndarray]:
        if self.scaler is None or self.interior_mask is None:
            raise RuntimeError("Co-kriging baseline has not been fitted")
        rows = np.asarray(rows, dtype=int)
        prediction = np.zeros((len(rows), *dataset.grid_shape), dtype=float)
        uncertainty = np.zeros_like(prediction)
        prediction_flat = prediction.reshape(len(rows), -1)
        uncertainty_flat = uncertainty.reshape(len(rows), -1)
        interior_flat = np.flatnonzero(self.interior_mask.ravel())
        frequency = np.full(len(rows), np.nan)
        frequency_std = np.full(len(rows), np.nan)
        for mode in np.unique(dataset.mode_ids[rows]).astype(int).tolist():
            local = np.flatnonzero(dataset.mode_ids[rows] == mode)
            rho, gp = self.mode_models[mode]
            scaled = self.scaler.transform(dataset.parameters[rows[local]])
            residual_mean, residual_std = gp.predict(scaled)
            lf = dataset.lf[rows[local]][:, self.interior_mask]
            # The autoregressive model predicts HF = rho * LF + residual;
            # evaluation consumes the correction HF - LF.
            prediction_flat[np.ix_(local, interior_flat)] = (
                (rho - 1.0) * lf + residual_mean
            )
            uncertainty_flat[np.ix_(local, interior_flat)] = residual_std
            rel_mean, rel_std = self.frequency_gps[mode].predict(scaled)
            rel_mean = np.clip(rel_mean[:, 0], -.8, .8)
            f_lf = dataset.f_lf[rows[local]]
            frequency[local] = f_lf / (1 - rel_mean)
            frequency_std[local] = np.abs(f_lf / (1 - rel_mean) ** 2) * rel_std[:, 0]
        return {
            "correction": prediction, "correction_std": uncertainty,
            "frequency": frequency, "frequency_std": frequency_std,
        }


# ---------------------------------------------------------------------------
# Metrics, files, and high-resolution diagnostics
# ---------------------------------------------------------------------------


def evaluate_predictions(
    dataset: P1Dataset, rows: Sequence[int], prediction: Mapping[str, np.ndarray],
    model_name: str, fold: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = np.asarray(rows, dtype=int)
    fields = np.asarray(prediction["correction"], dtype=float)
    if fields.shape != (len(rows), *dataset.grid_shape):
        raise ValueError("Prediction field shape does not match evaluation rows")
    std_value = prediction.get("correction_std")
    std = None if std_value is None else np.asarray(std_value, dtype=float)
    if std is not None and std.shape != fields.shape:
        raise ValueError("Prediction uncertainty shape does not match fields")
    interior = dataset.interior_mask
    edge = ~interior
    errors = fields - dataset.correction[rows]
    frequency = np.asarray(prediction.get("frequency"), dtype=float)
    frequency_std = np.asarray(prediction.get("frequency_std"), dtype=float)
    records: list[dict[str, object]] = []
    for local, row in enumerate(rows.tolist()):
        interior_rms = float(np.sqrt(np.mean(errors[local, interior] ** 2)))
        edge_rms = float(np.sqrt(np.mean(errors[local, edge] ** 2)))
        target_rms = float(np.sqrt(np.mean(dataset.correction[row, interior] ** 2)))
        record: dict[str, object] = {
            "model": model_name, "fold": int(fold), "row": int(row),
            "run": int(dataset.run_ids[row]), "mode": int(dataset.mode_ids[row]),
            "field_interior_rms": interior_rms, "field_edge_rms": edge_rms,
            "field_relative_rms": interior_rms / max(target_rms, 1e-12),
        }
        if std is not None:
            point_std = np.maximum(std[local, interior], 1e-12)
            point_error = np.abs(errors[local, interior])
            for level, z in ((50, .67448975), (90, 1.64485363), (95, 1.95996398)):
                record[f"field_coverage_{level}"] = float(np.mean(point_error <= z * point_std))
            record["mean_field_std"] = float(np.mean(point_std))
        f_true = float(dataset.f_hf[row])
        if np.isfinite(frequency[local]):
            freq_error_pct = 100.0 * (frequency[local] - f_true) / f_true
            record["frequency_error_pct"] = float(freq_error_pct)
            if np.isfinite(frequency_std[local]):
                f_std = max(float(frequency_std[local]), 1e-12)
                for level, z in ((50, .67448975), (90, 1.64485363), (95, 1.95996398)):
                    record[f"frequency_coverage_{level}"] = float(
                        abs(frequency[local] - f_true) <= z * f_std
                    )
                record["frequency_std"] = f_std
        records.append(record)

    def values(key: str) -> np.ndarray:
        return np.asarray([float(record[key]) for record in records if key in record], dtype=float)

    summary: dict[str, object] = {
        "model": model_name, "fold": int(fold), "rows": len(records),
        "runs": len({int(record["run"]) for record in records}),
        "modes": sorted({int(record["mode"]) for record in records}),
    }
    for key in (
        "field_interior_rms", "field_edge_rms", "field_relative_rms",
        "mean_field_std", "frequency_error_pct", "frequency_std",
        "field_coverage_50", "field_coverage_90", "field_coverage_95",
        "frequency_coverage_50", "frequency_coverage_90", "frequency_coverage_95",
    ):
        array = values(key)
        if len(array):
            summary[f"{key}_median"] = float(np.median(array))
            summary[f"{key}_p95"] = float(np.quantile(array, .95))
            summary[f"{key}_mean"] = float(np.mean(array))
    return records, summary


def write_records(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_figure(fig, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def write_diagnostics(
    output: Path, dataset: P1Dataset, oof: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, object]], decoder: BoundaryLatentDecoder,
    dpi: int = 300,
) -> None:
    """Write publication-ready OOF plots and pointwise uncertainty maps."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "figure.dpi": 120, "savefig.dpi": dpi,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    latent_records = [record for record in records if record["model"] == "latent_gp"]
    modes = sorted({int(record["mode"]) for record in latent_records})
    if modes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].boxplot(
            [[float(record["field_interior_rms"]) for record in latent_records if int(record["mode"]) == mode] for mode in modes],
            tick_labels=[str(mode) for mode in modes], showfliers=False,
        )
        axes[0].set(xlabel="Reference mode", ylabel="OOF interior field RMS", title="Latent-GP field error")
        freq_values = [
            [float(record["frequency_error_pct"]) for record in latent_records
             if int(record["mode"]) == mode and "frequency_error_pct" in record]
            for mode in modes
        ]
        axes[1].boxplot(freq_values, tick_labels=[str(mode) for mode in modes], showfliers=False)
        axes[1].axhline(0, color="k", lw=.8)
        axes[1].set(xlabel="Reference mode", ylabel="OOF frequency error (%)", title="Latent-GP frequency correction")
        for ax in axes:
            ax.grid(axis="y", alpha=.22)
        fig.tight_layout()
        save_figure(fig, figures / "latent_gp_oof_by_mode.png", dpi)

    pred = np.asarray(oof["correction"], dtype=float)
    std = np.asarray(oof["correction_std"], dtype=float)
    valid = np.isfinite(pred).all(axis=(1, 2))
    if np.any(valid):
        true = dataset.correction[valid]
        error = pred[valid] - true
        interior = dataset.interior_mask
        abs_error = np.mean(np.abs(error), axis=0)
        point_std = np.mean(std[valid], axis=0)
        coverage = np.mean(np.abs(error) <= 1.95996398 * np.maximum(std[valid], 1e-12), axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
        extent = [float(dataset.x[0]), float(dataset.x[-1]), float(dataset.y[0]), float(dataset.y[-1])]
        for ax, field, title, cmap in zip(
            axes, (abs_error, point_std, coverage),
            ("Mean absolute OOF error", "Mean predictive standard deviation", "Pointwise 95% coverage"),
            ("magma", "viridis", "RdYlGn"),
        ):
            image = ax.imshow(field, origin="lower", extent=extent, aspect="equal", cmap=cmap)
            ax.contour(interior.astype(float), levels=[.5], colors="white", linewidths=.35,
                       extent=extent)
            ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
            fig.colorbar(image, ax=ax, shrink=.82)
        save_figure(fig, figures / "latent_gp_pointwise_diagnostics.png", dpi)

        latent_rms = np.sqrt(np.mean(error[:, interior] ** 2, axis=1))
        worst = int(np.argmax(latent_rms))
        valid_rows = np.flatnonzero(valid)
        row = int(valid_rows[worst])
        scale = max(float(np.max(np.abs(true[worst]))), float(np.max(np.abs(pred[row]))), 1e-12)
        error_scale = max(float(np.max(np.abs(error[worst]))), 1e-12)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
        extent = [float(dataset.x[0]), float(dataset.x[-1]), float(dataset.y[0]), float(dataset.y[-1])]
        images = [
            (true[worst], "HF−LF correction", "RdBu_r", scale),
            (pred[row], "OOF latent-GP prediction", "RdBu_r", scale),
            (error[worst], "Prediction error", "RdBu_r", error_scale),
            (std[row], "Predictive standard deviation", "viridis", max(float(np.max(std[row])), 1e-12)),
        ]
        for ax, (field, title, cmap, vmax) in zip(axes, images):
            image = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                              cmap=cmap, vmin=-vmax if cmap == "RdBu_r" else 0, vmax=vmax)
            ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
            fig.colorbar(image, ax=ax, shrink=.82)
        fig.suptitle(
            f"Worst OOF field: run {dataset.run_ids[row]}, mode {dataset.mode_ids[row]}"
        )
        save_figure(fig, figures / "latent_gp_worst_oof_field.png", dpi)

    singular = np.asarray(decoder.singular_values, dtype=float)
    if len(singular):
        variance = singular**2 / max(np.sum(singular**2), 1e-30)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(np.arange(1, len(variance) + 1), np.cumsum(variance), marker="o", ms=3)
        ax.set(xlabel="Latent component", ylabel="Cumulative retained variance",
               title="Spatial decoder spectrum")
        ax.set_ylim(0, 1.02); ax.grid(alpha=.22)
        save_figure(fig, figures / "decoder_spectrum.png", dpi)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run_experiment(
    dataset: P1Dataset, output: os.PathLike[str] | str, *, latent_dim: int = 48,
    n_splits: int = 5, seed: int = 42, max_iterations: int = 80,
    include_inr: bool = True, include_cokriging: bool = True,
    baseline_cv: bool = True, inr_features: int = 64,
    inr_points: int = 120_000, dpi: int = 300,
    training_selection: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train/evaluate P1 models and write all reproducible artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = grouped_run_splits(dataset.run_ids, n_splits=n_splits, seed=seed)
    all_records: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    oof_prediction = np.full_like(dataset.correction, np.nan)
    oof_std = np.full_like(dataset.correction, np.nan)
    oof_frequency = np.full(len(dataset.run_ids), np.nan)
    oof_frequency_std = np.full(len(dataset.run_ids), np.nan)
    final_decoder: BoundaryLatentDecoder | None = None

    for fold, (train_rows, validation_rows, train_runs, validation_runs) in enumerate(splits, 1):
        latent = LatentGPModel(latent_dim, seed + fold, max_iterations)
        latent.fit(dataset, train_rows)
        prediction = latent.predict_rows(dataset, validation_rows)
        oof_prediction[validation_rows] = prediction["correction"]
        oof_std[validation_rows] = prediction["correction_std"]
        oof_frequency[validation_rows] = prediction["frequency"]
        oof_frequency_std[validation_rows] = prediction["frequency_std"]
        records, summary = evaluate_predictions(dataset, validation_rows, prediction, "latent_gp", fold)
        all_records.extend(records)
        summary.update({
            "train_runs": train_runs.tolist(), "validation_runs": validation_runs.tolist(),
            "decoder": latent.reconstruction_summary(dataset),
        })
        fold_summaries.append(summary)

        if baseline_cv and include_inr:
            inr = FourierINRBaseline(inr_features, max_points=inr_points, seed=seed + fold)
            inr.fit(dataset, train_rows)
            inr_prediction = inr.predict_rows(dataset, validation_rows)
            records, summary = evaluate_predictions(dataset, validation_rows, inr_prediction, "inr_baseline", fold)
            all_records.extend(records)
            fold_summaries.append(summary)

        if baseline_cv and include_cokriging:
            co = LinearCoKrigingBaseline(max_iterations=max_iterations, seed=seed + fold)
            co.fit(dataset, train_rows)
            co_prediction = co.predict_rows(dataset, validation_rows)
            records, summary = evaluate_predictions(dataset, validation_rows, co_prediction, "cokriging_baseline", fold)
            all_records.extend(records)
            fold_summaries.append(summary)

    final_model = LatentGPModel(latent_dim, seed, max_iterations).fit(
        dataset, np.arange(dataset.n_rows, dtype=int)
    )
    final_model.save(output / "latent_gp", dataset)
    final_decoder = final_model.decoder
    if include_inr:
        final_inr = FourierINRBaseline(inr_features, max_points=inr_points, seed=seed)
        final_inr.fit(dataset, np.arange(dataset.n_rows, dtype=int))
        final_inr.save(output / "inr_baseline")
    if include_cokriging:
        # The co-kriging baseline is evaluated above; its full-fit artifact is
        # intentionally not serialized because it contains one dense GP per
        # mode and is only a comparison model, not the handoff model.
        pass

    write_records(output / "oof_metrics.csv", all_records)
    np.savez_compressed(
        output / "oof_predictions.npz", row_indices=np.arange(dataset.n_rows),
        run_ids=dataset.run_ids, mode_ids=dataset.mode_ids,
        correction=oof_prediction, correction_std=oof_std,
        f_hf_pred=oof_frequency, f_hf_std=oof_frequency_std,
    )
    summary_by_model: dict[str, dict[str, object]] = {}
    for model_name in sorted({str(record["model"]) for record in all_records}):
        model_records = [record for record in all_records if record["model"] == model_name]
        # Aggregate directly from the fold records so baseline uncertainty and
        # frequency fields are retained in the report.
        summary_by_model[model_name] = {
            "rows": len(model_records),
            "runs": len({int(record["run"]) for record in model_records}),
            "field_interior_rms_median": float(np.median([float(record["field_interior_rms"]) for record in model_records])),
            "field_interior_rms_p95": float(np.quantile([float(record["field_interior_rms"]) for record in model_records], .95)),
            "field_relative_rms_median": float(np.median([float(record["field_relative_rms"]) for record in model_records])),
        }
        for key in ("frequency_error_pct", "field_coverage_90", "field_coverage_95", "frequency_coverage_90", "frequency_coverage_95"):
            values = [float(record[key]) for record in model_records if key in record]
            if values:
                summary_by_model[model_name][f"{key}_median"] = float(np.median(values))
                summary_by_model[model_name][f"{key}_p95"] = float(np.quantile(values, .95))

    metadata = {
        "model_version": MODEL_VERSION,
        "data_rows": dataset.n_rows,
        "data_runs": len(dataset.runs),
        "data_modes": dataset.modes.tolist(),
        "latent_dim": latent_dim,
        "n_splits": len(splits),
        "seed": seed,
        "validation_policy": "grouped by run_id; accepted mode rows never split independently",
        "boundary_policy": "normalized boundary_mask multiplied into predictions; perimeter excluded from loss/RMSE",
        "scope_warning": "P1 pilot pipeline validation; not a universal 4 percent claim or held-out simulator-independent accuracy claim",
        "training_selection": dict(training_selection or {
            "role": "all accepted runs",
        }),
        "fold_summaries": fold_summaries,
        "summary_by_model": summary_by_model,
        "decoder_reconstruction": final_model.reconstruction_summary(dataset),
    }
    (output / "metrics.json").write_text(
        json.dumps(metadata, indent=2, default=_json_safe), encoding="utf-8"
    )
    write_diagnostics(output, dataset, {
        "correction": oof_prediction, "correction_std": oof_std,
    }, all_records, final_decoder, dpi=dpi)
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _train_runs_from_plan(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"run_id", "role"}.issubset(rows[0]):
        raise ValueError("Run plan must contain run_id and role")
    train = sorted({
        int(row["run_id"]) for row in rows
        if row["role"].strip().lower() == "train"
    })
    if len(train) < 2:
        raise ValueError("Run plan needs at least two train runs")
    return train


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="P1 data root containing the selected corrections revision")
    parser.add_argument("--name", default="corrections_v5",
                        help="Correction revision directory name")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for model, metrics, and figures")
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--inr-features", type=int, default=64)
    parser.add_argument("--inr-points", type=int, default=120_000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--limit-runs", type=int,
                        help="Deterministic small run subset for smoke checks only")
    parser.add_argument("--run-plan", type=Path,
                        help="Role plan CSV; train only on rows whose role is train")
    parser.add_argument("--skip-inr", action="store_true")
    parser.add_argument("--skip-cokriging", action="store_true")
    parser.add_argument("--skip-baseline-cv", action="store_true",
                        help="Fit final baselines only; not a fair grouped-CV comparison")
    args = parser.parse_args(argv)
    if args.latent_dim < 1 or args.splits < 2 or args.max_iterations < 1:
        parser.error("latent-dim, splits, and max-iterations must be positive")
    if args.inr_features < 1 or args.inr_points < 8 or args.dpi < 72:
        parser.error("inr-features/inr-points/dpi are too small")
    if args.data.is_dir() and (args.data / "training.npz").exists():
        revision = args.data
    elif args.data.is_dir():
        revision = args.data / args.name
    else:
        revision = args.data
    dataset = P1Dataset.load(revision)
    training_selection: dict[str, object] = {"role": "all accepted runs"}
    if args.run_plan is not None:
        train_runs = _train_runs_from_plan(args.run_plan)
        dataset = dataset.select_runs(train_runs)
        training_selection = {
            "role": "train",
            "run_plan": str(args.run_plan.resolve()),
            "run_plan_sha256": hashlib.sha256(args.run_plan.read_bytes()).hexdigest(),
            "run_ids": train_runs,
        }
    if args.limit_runs is not None:
        if args.limit_runs < 2:
            parser.error("limit-runs must be at least 2")
        dataset = dataset.select_runs(dataset.runs[:args.limit_runs])
    metadata = run_experiment(
        dataset, args.output, latent_dim=args.latent_dim, n_splits=args.splits,
        seed=args.seed, max_iterations=args.max_iterations,
        include_inr=not args.skip_inr, include_cokriging=not args.skip_cokriging,
        baseline_cv=not args.skip_baseline_cv, inr_features=args.inr_features,
        inr_points=args.inr_points, dpi=args.dpi,
        training_selection=training_selection,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "rows": metadata["data_rows"], "runs": metadata["data_runs"],
        "models": sorted(metadata["summary_by_model"]),
        "latent_field_rms_median": metadata["summary_by_model"]["latent_gp"]["field_interior_rms_median"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
