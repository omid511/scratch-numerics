#!/usr/bin/env python3
"""Evaluate an optional nonlinear latent decoder upgrade.

This CPU-portable upgrade keeps the stable linear latent encoder and replaces
the linear decoder with a random-feature nonlinear map from latent ``z`` to
boundary-enforced correction fields.  It is a reconstruction gate, not a
claim that a neural autoencoder is automatically justified.  Use it only if
the linear decoder's held-out reconstruction error is the observed bottleneck.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from p1_surrogate import BoundaryLatentDecoder, P1Dataset, grouped_run_splits
from p1_design import file_sha256, write_json
from p1_holdout import read_roles


class RandomFeatureNonlinearDecoder:
    """Nonlinear latent-to-field decoder with analytic uncertainty propagation."""

    def __init__(self, latent_dim: int = 48, features: int = 96, ridge: float = 1e-4, seed: int = 42):
        self.latent_dim = int(latent_dim)
        self.features = int(features)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.linear: BoundaryLatentDecoder | None = None
        self.weights: np.ndarray | None = None
        self.phase: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.residual_variance: np.ndarray | None = None

    def _feature_map(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=float)
        projection = latent @ self.weights + self.phase[None, :]
        return np.column_stack([
            np.ones(len(latent)), latent, latent * latent,
            np.sin(projection), np.cos(projection),
        ])

    def fit(self, dataset: P1Dataset, rows: Sequence[int]) -> "RandomFeatureNonlinearDecoder":
        self.linear = BoundaryLatentDecoder(self.latent_dim, self.seed).fit(dataset, rows)
        latent = self.linear.encode(dataset.correction[np.asarray(rows, dtype=int)])
        target = dataset.correction[np.asarray(rows, dtype=int)][:, dataset.interior_mask]
        target = target / self.linear.envelope[dataset.interior_mask][None, :]
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 1.0, size=(self.latent_dim, self.features))
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=self.features)
        design = self._feature_map(latent)
        gram = design.T @ design
        cross = design.T @ target
        regularizer = np.eye(design.shape[1]) * self.ridge
        regularizer[0, 0] *= 1e-3
        try:
            self.coefficients = np.linalg.solve(gram + regularizer, cross)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(gram + regularizer, cross, rcond=None)[0]
        self.residual_variance = np.zeros(target.shape[1], dtype=float)
        reconstruction = self.decode(latent)
        residual = reconstruction - dataset.correction[np.asarray(rows, dtype=int)]
        self.residual_variance = np.var(residual[:, dataset.interior_mask], axis=0)
        return self

    def _check_fit(self) -> None:
        if any(value is None for value in (
            self.linear, self.weights, self.phase, self.coefficients,
            self.residual_variance,
        )):
            raise RuntimeError("Nonlinear decoder has not been fitted")

    def encode(self, fields: np.ndarray) -> np.ndarray:
        self._check_fit()
        return self.linear.encode(fields)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        self._check_fit()
        raw = self._feature_map(latent) @ self.coefficients
        fields = np.zeros((len(raw), *self.linear.envelope.shape), dtype=float)
        fields[:, self.linear.interior_mask] = raw * self.linear.envelope[self.linear.interior_mask][None, :]
        return fields

    def decode_std(self, latent: np.ndarray, latent_std: np.ndarray) -> np.ndarray:
        """Propagate independent latent uncertainty through the nonlinear map."""
        self._check_fit()
        latent = np.asarray(latent, dtype=float)
        latent_std = np.asarray(latent_std, dtype=float)
        if latent.shape != latent_std.shape or latent.shape[1] != self.latent_dim:
            raise ValueError("Latent mean/std shapes do not match")
        result = np.zeros((len(latent), *self.linear.envelope.shape), dtype=float)
        envelope = self.linear.envelope[self.linear.interior_mask]
        for index, (value, spread) in enumerate(zip(latent, latent_std)):
            projection = value @ self.weights + self.phase
            derivative_features = np.concatenate([
                np.zeros((1, self.latent_dim)),
                np.eye(self.latent_dim),
                np.diag(2.0 * value),
                (np.cos(projection)[:, None] * self.weights.T),
                (-np.sin(projection)[:, None] * self.weights.T),
            ], axis=0)
            jacobian = derivative_features.T @ self.coefficients
            variance = np.sum((spread[:, None] * jacobian) ** 2, axis=0)
            variance *= envelope ** 2
            variance += self.residual_variance
            result[index, self.linear.interior_mask] = np.sqrt(np.maximum(variance, 0.0))
        return result

    def reconstruction_metrics(self, dataset: P1Dataset, rows: Sequence[int]) -> dict[str, float]:
        rows = np.asarray(rows, dtype=int)
        prediction = self.decode(self.encode(dataset.correction[rows]))
        error = prediction - dataset.correction[rows]
        interior = dataset.interior_mask
        values = np.sqrt(np.mean(error[:, interior] ** 2, axis=1))
        return {
            "rows": int(len(rows)),
            "runs": int(len(np.unique(dataset.run_ids[rows]))),
            "interior_rms_median": float(np.median(values)),
            "interior_rms_p95": float(np.quantile(values, .95)),
            "edge_rms_median": float(np.median(np.sqrt(np.mean(error[:, ~interior] ** 2, axis=1)))),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", default="corrections_v5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True,
                        help="Frozen HF role plan; only train runs enter the decoder gate")
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--features", type=int, default=96)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    revision = args.data / args.name if args.data.is_dir() and not (args.data / "training.npz").exists() else args.data
    dataset = P1Dataset.load(revision)
    roles = read_roles(args.plan)
    missing = sorted(set(dataset.runs.tolist()) - set(roles))
    if missing:
        raise ValueError(f"Target runs are absent from the HF plan: {missing}")
    dataset = dataset.select_runs(run for run, role in roles.items() if role == "train")
    splits = grouped_run_splits(dataset.run_ids, args.splits, args.seed)
    records = []
    for fold, (train_rows, validation_rows, _, validation_runs) in enumerate(splits, 1):
        decoder = RandomFeatureNonlinearDecoder(args.latent_dim, args.features, args.ridge, args.seed + fold)
        decoder.fit(dataset, train_rows)
        metrics = decoder.reconstruction_metrics(dataset, validation_rows)
        linear = decoder.linear.reconstruction_summary(dataset.correction[validation_rows])
        nonlinear_fields = decoder.decode(decoder.encode(dataset.correction[validation_rows]))
        linear_fields = decoder.linear.reconstruction(dataset.correction[validation_rows])
        truth = dataset.correction[validation_rows][:, dataset.interior_mask]
        nonlinear_rms = np.sqrt(np.mean(
            (nonlinear_fields[:, dataset.interior_mask] - truth) ** 2, axis=1))
        linear_rms = np.sqrt(np.mean(
            (linear_fields[:, dataset.interior_mask] - truth) ** 2, axis=1))
        metrics.update({
            "linear_interior_rms_median": linear["interior_rms_median"],
            "linear_interior_rms_p95": linear["interior_rms_p95"],
            "paired_nonlinear_minus_linear_rms_median": float(np.median(nonlinear_rms - linear_rms)),
            "nonlinear_wins": int(np.sum(nonlinear_rms < linear_rms)),
        })
        metrics.update({"fold": fold, "validation_runs": validation_runs.tolist(), "decoder": "nonlinear_random_feature"})
        records.append(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "reconstruction_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "fold", "rows", "runs", "interior_rms_median", "interior_rms_p95",
            "edge_rms_median", "linear_interior_rms_median", "linear_interior_rms_p95",
            "paired_nonlinear_minus_linear_rms_median", "nonlinear_wins", "decoder",
        ])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in writer.fieldnames})
    summary = {
        "model_version": "p1-nonlinear-decoder-upgrade-v2",
        "decoder": "random-feature nonlinear latent decoder with linear PCA encoder",
        "latent_dim": args.latent_dim, "features": args.features, "ridge": args.ridge,
        "training_selection": {
            "role": "train", "runs": dataset.runs.tolist(), "rows": dataset.n_rows,
            "plan_sha256": file_sha256(args.plan),
        },
        "folds": records,
        "interior_rms_median": float(np.median([record["interior_rms_median"] for record in records])),
        "interior_rms_p95_median": float(np.median([record["interior_rms_p95"] for record in records])),
        "linear_interior_rms_median": float(np.median([
            record["linear_interior_rms_median"] for record in records])),
        "comparison": "linear and nonlinear reconstruction on identical train-run CV folds",
        "promotion_rule": "promote only if this held-out reconstruction materially improves the linear decoder and full-model validation benefits",
    }
    write_json(args.output / "reconstruction_metrics.json", summary, overwrite=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
