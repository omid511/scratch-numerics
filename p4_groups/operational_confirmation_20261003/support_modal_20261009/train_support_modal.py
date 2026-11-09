"""Fit preregistered measured/oracle support MLPs on grouped TRAIN folds.

The acquisition worker owns ``dataset.npz``.  This executor owns only the fixed
MLP fits and writes one immutable artifact per fit.  No result-dependent choices
are made here: each arm uses the same original validation designs and the same
three initialization seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

# Keep BLAS/CUDA runs reproducible and avoid oversubscription on the shared host.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling

PROTOCOL_PATH = SCRIPT / "protocol.json"
DATASET_PATH = SCRIPT / "dataset.npz"
TRAINING_PATH = SCRIPT / "training.json"

SEEDS = (42, 43, 44)
ARMS = ("measured", "oracle")
ORIGINAL_FIT = "original"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    # The acquisition protocol pins source inputs and, once reviewed, the
    # executable implementation.  Never silently train against a changed file.
    for category in ("inputs", "implementation"):
        for relative, digest in protocol.get(category, {}).items():
            candidate = ROOT / relative
            if not candidate.exists():
                raise FileNotFoundError(f"Pinned protocol path is missing: {relative}")
            if sha(candidate) != digest:
                raise RuntimeError(f"Pinned protocol {category} changed: {relative}")
    return protocol


def load_data() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Acquisition dataset is missing: {DATASET_PATH}")
    with np.load(DATASET_PATH, allow_pickle=False) as z:
        required = ("features", "oracle_features", "truth", "observed", "age", "times")
        data = {name: z[name].copy() for name in required}
        data["records"] = json.loads(str(z["records"]))
        if "protocol_sha256" in z:
            data["dataset_protocol_sha256"] = str(z["protocol_sha256"])
    _validate_data(data)
    return data


def _validate_data(data: dict[str, Any]) -> None:
    measured = np.asarray(data["features"])
    oracle = np.asarray(data["oracle_features"])
    truth = np.asarray(data["truth"])
    observed = np.asarray(data["observed"])
    age = np.asarray(data["age"])
    times = np.asarray(data["times"])
    if measured.dtype != np.float64 or oracle.dtype != np.float64:
        raise TypeError("features and oracle_features must be float64")
    if measured.ndim != 3 or measured.shape[1:] != (124, 174):
        raise ValueError(f"features must be R x 124 x 174, got {measured.shape}")
    if oracle.shape != measured.shape or truth.shape != measured.shape[:2]:
        raise ValueError("oracle/features/truth shapes do not agree")
    if observed.shape != truth.shape or age.shape != truth.shape or times.shape != (truth.shape[1],):
        raise ValueError("observed/age/times shapes do not agree")
    if observed.dtype != np.bool_:
        raise TypeError("observed must be boolean")
    if len(data["records"]) != truth.shape[0]:
        raise ValueError("record count does not match dataset rows")
    if not np.isfinite(truth).all() or not np.isfinite(age).all() or not np.isfinite(times).all():
        raise ValueError("truth, age, and times must be finite")
    if not np.isfinite(measured).all() or not np.isfinite(oracle).all():
        raise ValueError("measured/oracle features must be finite, including unavailable rows")


def _record_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    designs = np.asarray([str(r["design"]) for r in records], dtype="U32")
    splits = np.asarray([str(r["split"]) for r in records], dtype="U16")
    noises = np.asarray([str(r["noise"]) for r in records], dtype="U64")
    profiles = np.asarray([str(r["profile"]) for r in records], dtype="U64")
    return designs, splits, noises, profiles


def _fit_ids(protocol: dict[str, Any]) -> list[str]:
    """Stable order: two shared original fits, then six fold-specific fits."""
    folds = protocol["evaluation_folds"]
    return [f"{ORIGINAL_FIT}_{arm}" for arm in ARMS] + [
        f"fold{fold}_{arm}" for fold in range(len(folds)) for arm in ARMS
    ]


def _proper_masks(protocol: dict[str, Any], records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    designs, splits, noises, _ = _record_arrays(records)
    fitting_noise = set(protocol["training"]["fitting_noise"])
    # Both original and expanded fits use only the two original training views.
    base = np.isin(noises, list(fitting_noise))
    validation = (splits == "validation") & base
    train_ids = set(protocol["split"]["train"])
    original = (np.isin(designs, list(train_ids))) & base
    if not original.any() or not validation.any():
        raise ValueError("missing original TRAIN or validation clean/IID1 rows")
    proper: dict[str, np.ndarray] = {ORIGINAL_FIT: original}
    for fold, excluded in enumerate(protocol["evaluation_folds"]):
        fit_ids = train_ids | (set(protocol["additional_designs"]) - set(excluded))
        mask = np.isin(designs, list(fit_ids)) & base
        if set(fit_ids) & (set(excluded) | set(protocol["split"]["validation"]) | set(protocol["split"]["held"])):
            raise ValueError(f"fold {fold} fitting population overlaps evaluation/validation/held")
        proper[f"fold{fold}"] = mask
    # Exact population checks are deliberately structural rather than based on
    # expected row counts: acquisition may add diagnostics, but never fitting IDs.
    assert set(designs[original]) == train_ids
    assert set(designs[validation]) == set(protocol["split"]["validation"])
    return original, validation, proper
def contribution_groups(records: list[dict[str, Any]], proper: np.ndarray,
                        frame_count: int) -> tuple[list[dict[str, Any]], int]:
    """Describe each proper design/noise/profile block's row-mean weight."""
    grouped: dict[tuple[str, str, str], int] = {}
    for record, keep in zip(records, proper):
        if keep:
            key = (str(record["design"]), str(record["noise"]), str(record["profile"]))
            grouped[key] = grouped.get(key, 0) + 1
    total_frames = int(proper.sum()) * int(frame_count)
    if total_frames <= 0:
        raise ValueError("proper contribution population is empty")
    groups = []
    for (design, noise, profile), record_count in sorted(grouped.items()):
        frames = int(record_count) * int(frame_count)
        groups.append(dict(design=design, noise=noise, profile=profile,
                           records=int(record_count), frames=frames,
                           normalized_weight=float(frames / total_frames),
                           per_frame_weight=float(1.0 / total_frames)))
    if sum(group["frames"] for group in groups) != total_frames:
        raise AssertionError("contribution group denominator mismatch")
    return groups, total_frames




def standardize(features: np.ndarray, observed: np.ndarray, age: np.ndarray, proper: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = features[proper][observed[proper]]
    if rows.ndim != 2 or rows.shape[1] != 174 or not len(rows):
        raise ValueError("no proper observed rows available for scaling")
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite proper observed feature row")
    mean, scale, active = fit_ridge_scaling(rows)
    normalized = (features - mean) / scale
    normalized[..., ~active] = 0.0
    normalized[~observed] = 0.0
    x = np.concatenate((normalized, observed[..., None].astype(np.float64), age[..., None]), axis=-1)
    if x.shape[-1] != 176 or not np.isfinite(x).all():
        raise ValueError("constructed input is not finite 176-wide")
    return x.astype(np.float32), (mean, scale, active)


class FrameMLP(torch.nn.Module):
    """The frozen 176 -> 32 -> 16 -> 1 ReLU framewise model."""

    def __init__(self, width: int = 176) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[..., 0]


def _state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def train_one(x: np.ndarray, y: np.ndarray, proper: np.ndarray, validation: np.ndarray,
              fit_id: str, arm: str, seed: int, protocol: dict[str, Any],
              fold: str) -> tuple[np.ndarray, dict[str, Any], dict[str, torch.Tensor]]:
    if not proper.any() or not validation.any():
        raise ValueError(f"empty proper/validation population for {fit_id} seed {seed}")
    # Seed before construction is the paired-arm initialization contract.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FrameMLP(x.shape[-1]).to(device="cuda", dtype=torch.float32)
    tx = torch.as_tensor(x[proper], device="cuda", dtype=torch.float32)
    ty = torch.as_tensor(y[proper], device="cuda", dtype=torch.float32)
    vx = torch.as_tensor(x[validation], device="cuda", dtype=torch.float32)
    vy = torch.as_tensor(y[validation], device="cuda", dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol["training"]["learning_rate"]),
                                 weight_decay=float(protocol["training"]["weight_decay"]))
    max_epochs = int(protocol["training"]["maximum_epochs"])
    patience = int(protocol["training"]["patience"])
    best = np.inf
    stale = 0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(tx) - ty) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite training loss for {fit_id} seed {seed}")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_mae = float(torch.abs(model(vx) - vy).mean().cpu())
        train_mse = float(loss.detach().cpu())
        history.append(dict(epoch=epoch + 1, train_mse=train_mse, validation_mae=validation_mae))
        if validation_mae < best:
            best = validation_mae
            best_epoch = epoch + 1
            stale = 0
            best_state = _state_to_cpu(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"no finite selected checkpoint for {fit_id} seed {seed}")
    model.load_state_dict(best_state)
    model.eval()
    tx_all = torch.as_tensor(x, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        predictions = model(tx_all).detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(predictions).all():
        raise FloatingPointError(f"nonfinite predictions for {fit_id} seed {seed}")
    metadata = dict(fit_id=fit_id, fold=fold, arm=arm, seed=seed,
                    best_epoch=best_epoch, epochs=len(history), validation_mae=best,
                    input_width=int(x.shape[-1]), hidden_widths=[32, 16],
                    parameters=int(sum(p.numel() for p in model.parameters())),
                    proper_records=int(proper.sum()), validation_records=int(validation.sum()),
                    proper_row_count=int(proper.sum() * y.shape[1]),
                    validation_row_count=int(validation.sum() * y.shape[1]), history=history)
    return predictions, metadata, best_state


def _fit_file(fit_id: str, seed: int) -> Path:
    return SCRIPT / f"{fit_id}_seed{seed}.npz"


def _checkpoint_file(fit_id: str, seed: int) -> Path:
    return SCRIPT / f"{fit_id}_seed{seed}.pt"


def train() -> None:
    protocol = load_protocol()
    if TRAINING_PATH.exists():
        raise RuntimeError("support training is immutable once published")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for support fits; no CPU training fallback")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = load_data()
    records = data["records"]
    _, validation, proper_by_fold = _proper_masks(protocol, records)
    y = np.asarray(data["truth"], dtype=np.float32)
    all_fit_ids = _fit_ids(protocol)
    predictions_rows: list[np.ndarray] = []
    fit_metadata: list[dict[str, Any]] = []
    masks_by_fit: dict[str, np.ndarray] = {}

    for fit_id in all_fit_ids:
        fold = ORIGINAL_FIT if fit_id.startswith(f"{ORIGINAL_FIT}_") else fit_id.split("_")[0]
        arm = fit_id.rsplit("_", 1)[-1]
        proper = proper_by_fold[fold]
        features = np.asarray(data["features" if arm == "measured" else "oracle_features"], dtype=np.float64)
        x, scaling = standardize(features, data["observed"], data["age"], proper)
        groups, total_frames = contribution_groups(records, proper, y.shape[1])
        row_contribution_weights = np.zeros(y.shape, dtype=np.float32)
        row_contribution_weights[proper] = np.float32(1.0 / total_frames)
        masks_by_fit[fit_id] = proper.copy()
        points_for_fit: list[np.ndarray] = []
        # Recompute both paired arms' model with each seed.  The seed controls
        # initialization only; scaling is arm-specific and fit-population-specific.
        for seed in SEEDS:
            path = _fit_file(fit_id, seed)
            if path.exists() or _checkpoint_file(fit_id, seed).exists():
                raise RuntimeError(f"partial existing support fit artifact: {path.stem}")
            point, metadata, state = train_one(x, y, proper, validation, fit_id, arm, seed, protocol, fold)
            metadata["contribution_groups"] = groups
            metadata["contribution_total_frames"] = total_frames
            metadata["contribution_per_frame_weight"] = float(1.0 / total_frames)
            row_contributions = np.full(y.shape, np.nan, dtype=np.float32)
            row_contributions[proper] = (point[proper] - y[proper]) ** 2
            checkpoint = dict(state=state, fit_id=fit_id, fold=fold, arm=arm, seed=seed,
                              protocol_sha256=sha(PROTOCOL_PATH), dataset_sha256=sha(DATASET_PATH),
                              feature_source="features" if arm == "measured" else "oracle_features",
                              input_width=176)
            torch.save(checkpoint, _checkpoint_file(fit_id, seed))
            history_array = np.asarray([[h["epoch"], h["train_mse"], h["validation_mae"]]
                                        for h in metadata["history"]], dtype=np.float64)
            np.savez_compressed(path, predictions=point, row_contributions=row_contributions,
                                row_contribution_weights=row_contribution_weights,
                                mean=scaling[0], scale=scaling[1], active=scaling[2], proper=proper,
                                validation=validation, history=history_array,
                                metadata=json.dumps(metadata), fit_id=fit_id, arm=arm, fold=fold,
                                seed=seed, protocol_sha256=sha(PROTOCOL_PATH), dataset_sha256=sha(DATASET_PATH))
            points_for_fit.append(point)
            fit_metadata.append(metadata)
            print("SUPPORT_FIT", fit_id, seed, metadata["best_epoch"], metadata["validation_mae"], flush=True)
        predictions_rows.extend(points_for_fit)

    fit_ids_per_prediction = [f"{fit_id}_seed{seed}" for fit_id in all_fit_ids for seed in SEEDS]
    predictions = np.stack(predictions_rows)
    np.savez_compressed(SCRIPT / "predictions.npz", predictions=predictions,
                        fit_ids=json.dumps(fit_ids_per_prediction), fits=json.dumps(all_fit_ids),
                        seeds=np.asarray(SEEDS), dataset_sha256=sha(DATASET_PATH),
                        protocol_sha256=sha(PROTOCOL_PATH))
    save_json(TRAINING_PATH, dict(scope=protocol.get("scope"), fit_ids=all_fit_ids,
        prediction_fit_ids=fit_ids_per_prediction, seeds=list(SEEDS), arms=list(ARMS),
        fit_count=len(fit_metadata), fits=fit_metadata,
        proper_masks={key: value.astype(int).tolist() for key, value in masks_by_fit.items()},
        dataset_sha256=sha(DATASET_PATH), protocol_sha256=sha(PROTOCOL_PATH),
        predictions_sha256=sha(SCRIPT / "predictions.npz"),
        pinned_sha256={
            "protocol": sha(PROTOCOL_PATH), "dataset": sha(DATASET_PATH),
            "implementation": sha(Path(__file__)),
            "score_implementation": sha(SCRIPT / "score_support_modal.py"),
            "verification_implementation": sha(SCRIPT / "verify_support_modal.py"),
            "scaling_helper": sha(ROOT / "src/mechanics/p4_margin_estimation/baselines.py"),
        },
        cuda=torch.cuda.get_device_name(), torch=torch.__version__,
        objective="Full-batch row-mean squared error on native all-frame outputs",
        selection="Original validation clean/IID1 all-frame MAE shared by measured/oracle and all folds",
        no_search=True))
    print("SUPPORT_TRAINING_COMPLETE", len(fit_metadata), flush=True)


def mechanism_smoke() -> dict[str, Any]:
    """Tiny CPU-only shape/causality smoke; does not train experiment fits."""
    torch.manual_seed(42)
    model = FrameMLP(176).eval()
    x = torch.zeros((2, 124, 176), dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 124)
    assert bool(torch.isfinite(y).all())
    return {"input_width": 176, "hidden_widths": [32, 16], "output_shape": list(y.shape), "finite": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "train"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == "smoke":
            print("SUPPORT_TRAIN_MECHANISM_SMOKE", json.dumps(mechanism_smoke()))
        else:
            train()
