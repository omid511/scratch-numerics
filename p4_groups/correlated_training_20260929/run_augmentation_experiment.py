#!/usr/bin/env python3
"""Run the fixed three-arm MLP design-fold experiment on shared.npz."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiment_p4_ridge_cv import design_folds
from experiment_p4_tabular import fit_network, score

DATA_PATH = ARTIFACT_DIR / "shared.npz"
PROTOCOL_PATH = ARTIFACT_DIR / "protocol.json"
OUTPUT_PATH = ARTIFACT_DIR / "augmentation_results.json"
EVALUATION_HASHES = (
    "experiment_p4_tabular.py",
    "experiment_p4_ridge_cv.py",
    "src/mechanics/p4_margin_estimation/baselines.py",
    "src/mechanics/p4_margin_estimation/skip_mlp.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+",
                        choices=["original3", "extra_independent4", "correlated4"],
                        default=["original3", "extra_independent4", "correlated4"])
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment evidence: {args.output}")
    data = np.load(DATA_PATH, allow_pickle=False)
    train3 = data["train3"]
    independent_extra = data["independent_extra"]
    correlated_extra = data["correlated_extra"]
    evaluation = data["evaluation"]
    evaluation_names = data["evaluation_names"].tolist()
    y = data["y"]
    ids = data["ids"]
    source_indices = data["source_indices"]

    n_rows = len(y)
    if train3.ndim != 3 or train3.shape[:2] != (3, n_rows):
        raise ValueError(f"train3 row alignment mismatch: {train3.shape}, {n_rows}")
    if independent_extra.shape != train3.shape[1:] or correlated_extra.shape != train3.shape[1:]:
        raise ValueError("extra training view row/feature alignment mismatch")
    if evaluation.shape != (len(evaluation_names), n_rows, train3.shape[2]):
        raise ValueError(f"evaluation row/feature alignment mismatch: {evaluation.shape}")
    if ids.shape != (n_rows,) or source_indices.shape != (n_rows,):
        raise ValueError("ids/source_indices row alignment mismatch")
    if len(set(evaluation_names)) != len(evaluation_names):
        raise ValueError("duplicate evaluation names")
    for name, array in (("train3", train3), ("independent_extra", independent_extra),
                        ("correlated_extra", correlated_extra), ("evaluation", evaluation),
                        ("y", y)):
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite values in {name}")

    arms = {
        "original3": train3,
        "extra_independent4": np.concatenate((train3, independent_extra[None, :, :]), axis=0),
        "correlated4": np.concatenate((train3, correlated_extra[None, :, :]), axis=0),
    }
    arms = {name: arms[name] for name in args.arms}
    fold_rows = design_folds(ids, 5, 42)
    all_designs = set(np.unique(ids).tolist())
    fold_assignments = np.full(n_rows, -1, dtype=np.int8)
    folds = []
    for fold_index, held_rows in enumerate(fold_rows):
        held_designs = np.unique(ids[held_rows])
        train_rows = np.flatnonzero(~np.isin(ids, held_designs))
        train_designs = np.unique(ids[train_rows])
        if set(train_designs.tolist()) & set(held_designs.tolist()):
            raise ValueError(f"design leakage in fold {fold_index}")
        if set(train_designs.tolist()) | set(held_designs.tolist()) != all_designs:
            raise ValueError(f"incomplete design assignment in fold {fold_index}")
        if np.any(fold_assignments[held_rows] != -1):
            raise ValueError(f"overlapping held-out rows in fold {fold_index}")
        fold_assignments[held_rows] = fold_index
        folds.append({
            "fold": fold_index,
            "train_design_ids": train_designs.tolist(),
            "test_design_ids": held_designs.tolist(),
            "test_row_indices": held_rows.tolist(),
            "test_source_indices": source_indices[held_rows].tolist(),
        })
    if np.any(fold_assignments < 0):
        raise ValueError("some rows were not assigned to an outer fold")

    arm_results = {}
    with threadpool_limits(limits=1):
        for arm_name, views in arms.items():
            oof = np.full((len(evaluation_names), n_rows), np.nan, dtype=np.float64)
            arm_folds = []
            for fold_index, (held_rows, assignment) in enumerate(zip(fold_rows, folds)):
                held_designs = np.asarray(assignment["test_design_ids"])
                train_rows = np.flatnonzero(~np.isin(ids, held_designs))
                view_count = views.shape[0]
                x_train = np.concatenate([views[:, :, :][view, train_rows, :] for view in range(view_count)], axis=0)
                y_train = np.tile(y[train_rows], view_count)
                if x_train.shape[0] != y_train.shape[0]:
                    raise ValueError(f"training feature/target length mismatch in {arm_name} fold {fold_index}")
                prediction_rows = np.zeros((len(evaluation_names), len(held_rows)), dtype=np.float64)
                predict, metadata = fit_network(
                    x_train, y_train, seed=args.seed, alpha=1., custom=False, skip=False,
                    max_iter=10000, max_fun=200000,
                )
                for eval_index in range(len(evaluation_names)):
                    prediction_rows[eval_index] = predict(evaluation[eval_index, held_rows, :])
                if not np.isfinite(prediction_rows).all():
                    raise ValueError(f"nonfinite OOF predictions in {arm_name} fold {fold_index}")
                oof[:, held_rows] = prediction_rows
                arm_folds.append({
                    **assignment,
                    "fit_metadata": metadata,
                    "train_row_count_per_view": int(len(train_rows)),
                    "training_view_count": int(view_count),
                    "training_row_count": int(len(y_train)),
                })
                print(arm_name, "seed", args.seed, "fold", fold_index,
                      "converged", metadata["success"], flush=True)
            if not np.isfinite(oof).all():
                raise ValueError(f"incomplete or nonfinite OOF predictions in {arm_name}")
            records = {
                name: score(y, oof[index], ids)
                for index, name in enumerate(evaluation_names)
            }
            arm_results[arm_name] = {"records": records, "folds": arm_folds}

    protocol = json.loads(PROTOCOL_PATH.read_text())
    protocol["base"] = (f"MLP32->16 alpha1 seed{args.seed}; training-only scaling; "
                        "same5 outer design folds seed42; no reserved splits")
    hashes = {
        "inputs": {
            "shared.npz": sha256(DATA_PATH),
            "protocol.json": sha256(PROTOCOL_PATH),
        },
        "sources": {relative: sha256(ROOT / relative) for relative in EVALUATION_HASHES},
        "execution_code": sha256(Path(__file__)),
    }
    result = {
        "protocol": protocol,
        "hashes": hashes,
        "settings": {
            "fold_count": 5,
            "fold_seed": 42,
            "model_seed": args.seed,
            "alpha": 1.0,
            "hidden_layers": [32, 16],
            "max_iter": 10000,
            "max_fun": 200000,
            "custom_objective": False,
            "skip_connection": False,
            "thread_limit": 1,
            "input_rows": n_rows,
            "feature_count": int(train3.shape[2]),
            "design_count": int(len(all_designs)),
            "evaluation_names": evaluation_names,
            "evaluation_source": "evaluation array only; no evaluation rows used for fitting",
        },
        "design_assignments": {
            "source_indices": source_indices.tolist(),
            "row_fold": fold_assignments.tolist(),
            "folds": folds,
        },
        "arms": arm_results,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "input_rows": n_rows,
        "design_count": len(all_designs),
        "folds": len(folds),
        "arms": {
            arm: [fold["fit_metadata"] for fold in result["arms"][arm]["folds"]]
            for arm in arms
        },
        "evaluation_names": evaluation_names,
    }, indent=2))


if __name__ == "__main__":
    main()
