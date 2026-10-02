#!/usr/bin/env python3
"""P4 sensor-cap sensitivity experiment on preserved pre-saturation responses.

The runner consumes ``pre_saturation.npy`` produced by the dataset generator.
Those values are normalized signals after the modal-growth safety clamp and
before the sensor ADC cap.  The historical ``clips.npy`` file remains the
cap-50 reference and is replay-checked before any training starts.

Usage::

    PYTHONPATH=src python experiment_p4_saturation.py \
        --dataset p4_dataset_full --output-dir p4_saturation_run \
        --caps 50 100 200 --seeds 0 1 2 --epochs 20 \
        --batch-size 64 --predict-batch 32 --threads 4

The output directory contains ``results.json``, atomic model checkpoints in
``checkpoints/``, and one prediction ``.npz`` audit file per complete record.
Fixed-transfer models and their calibration are fitted at cap 50.  Matched
models and calibration are fitted independently at every requested cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Match the import bootstrap used by the existing phase-2 runner.  Imports are
# intentionally the existing implementations: this file owns orchestration,
# not a second model/trainer/metric implementation.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from experiment_p4_phase2 import GRUEllQuantile, _predict, train_quantile_generic
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.decision_metrics import (
    interval_score,
    paired_design_comparison,
    policy_table,
)
from mechanics.p4_margin_estimation.quantile_head import (
    apply_cqr_adjustment,
    fit_cqr_adjustment,
)
from train_p4_expanded import (
    _split_select_calibrate,
    apply_dr_with_mask,
    build_clips,
    load_dataset,
    with_ones_mask,
)


DEFAULT_CAPS = (50.0, 100.0, 200.0)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_PREDICT_BATCH = 32
DEFAULT_THREADS = max(1, min(8, os.cpu_count() or 1))
ALPHA = 0.10
N_PHYSICAL_CHANNELS = 8
N_MODEL_CHANNELS = 16
SEQUENCE_LENGTH = 512
MODEL_NAME = "gru_ell_quantile"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Small durable-I/O and canonicalization helpers


def _json_ready(value: Any) -> Any:
    """Convert numpy/scalar values and NaN to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.generic,)):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = _json_ready(value)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_digest(value: Any) -> str:
    """Stable digest for small aligned metadata arrays.

    Object/string arrays are encoded element by element so their hash does not
    depend on numpy's object-pointer representation.  Large signal arrays use
    their source-file digest in the dataset fingerprint to avoid an additional
    contiguous multi-hundred-MB allocation.
    """
    arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(_canonical_json(list(arr.shape)).encode("utf-8"))
    h.update(b"\0")
    if arr.dtype.kind in "OUS":
        for item in arr.reshape(-1).tolist():
            raw = str(item).encode("utf-8", errors="replace")
            h.update(len(raw).to_bytes(8, "little"))
            h.update(raw)
    else:
        h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def _cap_token(cap: float) -> str:
    text = format(float(cap), ".12g")
    return text.replace("-", "m").replace("+", "p").replace(".", "d")


def _cap_label(cap: float) -> str:
    text = format(float(cap), ".12g")
    return text[:-2] if text.endswith(".0") else text


def _safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _configure_threads(threads: int) -> None:
    threads = int(threads)
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    # These environment variables affect BLAS/OpenMP children created by
    # feature extraction as well as torch.  They are set before training.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # Torch permits inter-op configuration only before its first parallel
        # work.  The intra-op bound above remains effective on resumed runs.
        pass


# ---------------------------------------------------------------------------
# Dataset validation and fingerprinting


def _validate_caps(caps: Iterable[float]) -> tuple[float, ...]:
    vals = tuple(float(c) for c in caps)
    if not vals:
        raise ValueError("at least one cap is required")
    if any(not math.isfinite(c) or c < 50.0 for c in vals):
        raise ValueError(f"caps must be finite and >= 50, got {vals!r}")
    if len(set(vals)) != len(vals):
        raise ValueError(f"caps must be unique, got {vals!r}")
    if 50.0 not in vals:
        raise ValueError("caps must include the cap-50 reference")
    return tuple(sorted(vals))


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    vals = tuple(int(s) for s in seeds)
    if not vals:
        raise ValueError("at least one seed is required")
    if len(set(vals)) != len(vals):
        raise ValueError(f"seeds must be unique, got {vals!r}")
    return vals


def _as_ids(value: Any, name: str, n: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (n,):
        raise ValueError(f"{name} shape {arr.shape} != {(n,)}")
    out = np.asarray([str(x) for x in arr.tolist()], dtype=object)
    if any(x == "" or x.lower() == "none" for x in out.tolist()):
        raise ValueError(f"{name} contains an empty/None identifier")
    return out


def _validate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Load labels/provenance via ``load_dataset`` and validate replayability."""
    dataset_path = Path(dataset_dir)
    clips_arr, margins, velocities, design_ids, realization_ids, meta = load_dataset(str(dataset_path))
    pre_path = dataset_path / "pre_saturation.npy"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Required {pre_path} is missing; regenerate the dataset with "
            "pre-saturation preservation enabled. Refusing cap sensitivity without it."
        )
    pre = np.load(pre_path, mmap_mode="r", allow_pickle=False)
    clips = np.asarray(clips_arr)
    if pre.ndim != 3 or pre.shape[1:] != (N_PHYSICAL_CHANNELS, SEQUENCE_LENGTH):
        raise ValueError(f"pre_saturation shape must be (n, 8, 512), got {pre.shape}")
    if clips.ndim != 3 or clips.shape != pre.shape:
        raise ValueError(f"clips shape {clips.shape} is not aligned with pre_saturation {pre.shape}")
    if pre.dtype != np.float32:
        raise ValueError(f"pre_saturation must be float32, got {pre.dtype}")
    n = int(pre.shape[0])
    pre_chunk = clips_chunk = replay_chunk = None
    # Validate signal tensors in bounded chunks.  ``load_dataset`` has already
    # materialized clips.npy; a full-size finite/replay temporary would
    # otherwise briefly duplicate that tensor on the constrained runner.
    for start in range(0, n, 32):
        stop = min(n, start + 32)
        pre_chunk = np.asarray(pre[start:stop])
        clips_chunk = np.asarray(clips[start:stop])
        if not np.all(np.isfinite(pre_chunk)):
            raise ValueError(f"pre_saturation contains non-finite values near clip {start}")
        if not np.all(np.isfinite(clips_chunk)):
            raise ValueError(f"clips contains non-finite values near clip {start}")
        replay_chunk = np.clip(pre_chunk, -50.0, 50.0)
        if not np.allclose(replay_chunk, clips_chunk, rtol=0.0, atol=1e-6):
            delta = float(np.max(np.abs(np.asarray(replay_chunk, dtype=np.float32) - clips_chunk)))
            raise ValueError(f"cap-50 replay does not match clips.npy near clip {start} (max delta {delta:g})")

    # Signal tensors are validated above; only small aligned arrays remain.
    del pre_chunk, clips_chunk, replay_chunk

    margins_arr = np.asarray(margins, dtype=float)
    velocities_arr = np.asarray(velocities, dtype=float)
    if margins_arr.shape != (n,) or velocities_arr.shape != (n,):
        raise ValueError("labels/velocities are not aligned with clips")
    if not np.all(np.isfinite(margins_arr)):
        raise ValueError("margins contain non-finite labels")
    if not np.all(np.isfinite(velocities_arr)):
        raise ValueError("velocities contain non-finite values")
    design_arr = _as_ids(design_ids, "design_ids", n)
    rid_arr = np.asarray(realization_ids)
    if rid_arr.shape != (n,):
        raise ValueError(f"realization_ids shape {rid_arr.shape} != {(n,)}")

    prov = meta.get("provenance", {}) if isinstance(meta, dict) else {}
    if not isinstance(prov, dict):
        raise ValueError("load_dataset did not return a provenance mapping")
    clamp_values = prov.get("clamp_fracs")
    if clamp_values is None:
        raise ValueError(
            "growth-clamp provenance is missing; refusing to label all clips "
            "as nonclamped"
        )
    clamp_arr = np.asarray(clamp_values, dtype=float)
    if clamp_arr.shape != (n,) or not np.all(np.isfinite(clamp_arr)):
        raise ValueError("clamp_fracs must be finite and aligned with clips")
    if np.any(clamp_arr < 0.0) or np.any(clamp_arr > 1.0):
        raise ValueError("clamp_fracs must lie in [0, 1]")

    dts = prov.get("dts")
    dts_arr = None
    if dts is not None:
        dts_arr = np.asarray(dts, dtype=float)
        if dts_arr.shape != (n,) or not np.all(np.isfinite(dts_arr)) or np.any(dts_arr <= 0.0):
            raise ValueError("dts must be finite, positive, and aligned with clips")

    # Metadata may advertise the provenance contract; if present, validate it
    # rather than silently accepting a mislabeled file.
    manifest = meta.get("manifest", {}) if isinstance(meta, dict) else {}
    pre_manifest = manifest.get("pre_saturation") if isinstance(manifest, dict) else None
    if pre_manifest is not None:
        if not isinstance(pre_manifest, dict):
            raise ValueError("manifest.pre_saturation must be a mapping")
        expected = {
            "available": True,
            "file": "pre_saturation.npy",
            "stage": "post_normalization_pre_sensor_cap",
            "modal_growth_clamp_retained": True,
        }
        for key, want in expected.items():
            if pre_manifest.get(key) != want:
                raise ValueError(
                    f"manifest.pre_saturation[{key!r}]={pre_manifest.get(key)!r}; "
                    f"expected {want!r}"
                )

    split_sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        vals = meta.get(f"{split}_designs") if isinstance(meta, dict) else None
        if not isinstance(vals, list):
            raise ValueError(f"metadata.{split}_designs must be a nonempty list")
        ids = {str(v) for v in vals}
        if not ids:
            raise ValueError(f"metadata.{split}_designs is empty")
        split_sets[split] = ids
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("train/select-calibration/test design partitions overlap")
    observed = set(design_arr.tolist())
    for split, ids in split_sets.items():
        missing = ids - observed
        if missing:
            raise ValueError(f"{split} metadata names designs with no clips: {sorted(missing)[:5]}")
        if not any(x in ids for x in design_arr.tolist()):
            raise ValueError(f"{split} clip partition is empty")

    # ``clips_arr`` was loaded by load_dataset for replay validation.  Do not
    # retain a second full signal tensor after this point; all cap variants are
    # derived from the mmap-backed pre-saturation source one cap at a time.
    del clips_arr, clips

    return {
        "dataset_dir": dataset_path,
        "pre": pre,
        "margins": margins_arr,
        "velocities": velocities_arr,
        "design_ids": design_arr,
        "realization_ids": rid_arr,
        "dts": dts_arr,
        "clamp_fracs": clamp_arr,
        "metadata": meta,
        "splits": split_sets,
        "n_clips": n,
    }


def _source_dependencies() -> list[Path]:
    return [
        _ROOT / "experiment_p4_saturation.py",
        _ROOT / "pyproject.toml",
        _ROOT / "uv.lock",
        _ROOT / "experiment_p4_phase2.py",
        _ROOT / "train_p4_expanded.py",
        _ROOT / "generate_p4_dataset.py",
        _SRC / "mechanics/p4_margin_estimation/decision_metrics.py",
        _SRC / "mechanics/p4_margin_estimation/quantile_head.py",
        _SRC / "mechanics/p4_margin_estimation/baselines.py",
        _SRC / "mechanics/p4_margin_estimation/quantile_gru.py",
        _SRC / "mechanics/p4_margin_estimation/tcn.py",
        _SRC / "mechanics/p4_margin_estimation/domain_randomization.py",
        _SRC / "mechanics/p4_margin_estimation/transient.py",
    ]


def _fingerprint_dataset(bundle: dict[str, Any], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dataset_dir = Path(bundle["dataset_dir"])
    raw_names = (
        "pre_saturation.npy",
        "clips.npy",
        "clips.npz",
        "metadata_arrays.npz",
        "margins.npy",
        "velocities.npy",
        "design_ids.npy",
        "realization_ids.npy",
        "metadata.json",
    )
    files = {}
    for name in raw_names:
        path = dataset_dir / name
        files[name] = _sha256_file(path) if path.exists() else None
    metadata_file = dataset_dir / "metadata.json"
    try:
        with open(metadata_file, encoding="utf-8") as f:
            raw_metadata = json.load(f)
    except Exception as exc:
        raise ValueError(f"cannot parse metadata.json for fingerprint: {exc}") from exc

    arrays = {
        "pre_saturation": {
            "shape": list(bundle["pre"].shape),
            "dtype": str(bundle["pre"].dtype),
            "file_sha256": files["pre_saturation.npy"],
        },
        "margins": _array_digest(bundle["margins"]),
        "velocities": _array_digest(bundle["velocities"]),
        "design_ids": _array_digest(bundle["design_ids"]),
        "realization_ids": _array_digest(bundle["realization_ids"]),
        "clamp_fracs": _array_digest(bundle["clamp_fracs"]),
    }
    if bundle["dts"] is not None:
        arrays["dts"] = _array_digest(bundle["dts"])
    dependencies = {}
    for path in _source_dependencies():
        rel = _relative_path(path, _ROOT) if path.exists() else path.name
        dependencies[rel] = _sha256_file(path) if path.exists() else None

    payload = {
        "schema_version": SCHEMA_VERSION,
        "data_files": files,
        "arrays": arrays,
        "metadata": raw_metadata,
        "source_dependencies": dependencies,
        "config": config,
    }
    fingerprint = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return fingerprint, payload


class _WandbArtifactPublisher:
    """Commit content-fingerprinted experiment inputs and records to online W&B."""

    def __init__(self, project: str):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B was requested but its SDK is unavailable; install the optional dependency with `uv sync --extra wandb`"
            ) from exc

        self.wandb = wandb
        self.project = project
        self.run = None
        self.fingerprint: str | None = None
        self.dataset_fingerprint: str | None = None
        try:
            entity = wandb.Api().default_entity
            if not isinstance(entity, str) or not entity:
                raise RuntimeError("authenticated W&B account has no default entity")
            self.entity = entity
            self.run = wandb.init(
                project=project,
                entity=entity,
                mode="online",
                job_type="p4-saturation",
                config={"runner": Path(__file__).name},
            )
            settings = getattr(self.run, "settings", None)
            if self.run is None or getattr(settings, "mode", None) != "online":
                raise RuntimeError("W&B did not create an online run")
        except Exception:
            if self.run is not None:
                try:
                    self.run.finish(exit_code=1)
                except Exception:
                    pass
            raise RuntimeError(
                "W&B authentication/entity discovery or online run initialization failed"
            ) from None
        self._finished = False
        self._defined_metric_prefixes: set[str] = set()

    def _scope(self, results: dict[str, Any]) -> dict[str, Any]:
        tracking = results.setdefault("wandb", {})
        if not isinstance(tracking, dict):
            raise ValueError("results.wandb must be a mapping")
        for key, value in (("project", self.project), ("entity", self.entity)):
            if tracking.get(key) not in (None, value):
                raise ValueError(f"existing W&B results use a different {key}")
            tracking[key] = value
        return tracking

    def _commit(self, artifact: Any) -> str:
        logged = self.run.log_artifact(artifact, aliases=["latest"])
        committed = logged.wait()
        if getattr(committed, "state", None) != "COMMITTED":
            raise RuntimeError(f"W&B artifact did not commit: {artifact.name}")
        reference = getattr(committed, "qualified_name", None)
        if not isinstance(reference, str) or not reference:
            raise RuntimeError(f"W&B committed artifact has no reference: {artifact.name}")
        return reference

    def publish_dataset(
        self,
        bundle: dict[str, Any],
        payload: dict[str, Any],
        fingerprint: str,
        results: dict[str, Any],
        results_path: Path,
    ) -> None:
        data_hashes = {
            name: value for name, value in payload["data_files"].items() if value is not None
        }
        source_hashes = payload["source_dependencies"]
        dataset_fingerprint = hashlib.sha256(
            _canonical_json(
                {"data_files": data_hashes, "source_dependencies": source_hashes}
            ).encode("utf-8")
        ).hexdigest()
        self.fingerprint = fingerprint
        self.dataset_fingerprint = dataset_fingerprint
        self.run.config.update({
            **payload["config"],
            "experiment_fingerprint": fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
        })
        tracking = self._scope(results)
        if tracking.get("dataset_artifact"):
            if tracking.get("dataset_fingerprint") != dataset_fingerprint:
                raise ValueError("existing W&B dataset artifact fingerprint differs")
            return

        metadata = {
            "experiment_fingerprint": fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "data_files_sha256": data_hashes,
            "source_dependencies_sha256": source_hashes,
            "config": payload["config"],
        }
        artifact = self.wandb.Artifact(
            name=f"p4-saturation-dataset-{dataset_fingerprint}",
            type="dataset",
            metadata=metadata,
            description="Fingerprint-verified P4 saturation input data and source provenance.",
        )
        dataset_dir = Path(bundle["dataset_dir"])
        for name, expected in data_hashes.items():
            path = dataset_dir / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise RuntimeError(f"dataset changed after fingerprinting: {name}")
            artifact.add_file(str(path), name=f"dataset/{name}")
        for path in _source_dependencies():
            relative = _relative_path(path, _ROOT) if path.exists() else path.name
            expected = source_hashes.get(relative)
            if not path.is_file() or not expected or _sha256_file(path) != expected:
                raise RuntimeError(f"source changed or is missing after fingerprinting: {relative}")
            artifact.add_file(str(path), name=f"source/{relative}")

        reference = self._commit(artifact)
        tracking.update({
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_artifact": reference,
        })
        _atomic_json_dump(results, results_path)

    def log_epoch(
        self,
        epoch: int,
        history: dict[str, list[float]],
        *,
        train_cap: float,
        seed: int,
        protocol: str,
    ) -> None:
        prefix = f"training/{protocol}/cap-{_cap_label(train_cap)}/seed-{seed}"
        if prefix not in self._defined_metric_prefixes:
            self.run.define_metric(
                f"{prefix}/train_loss", step_metric="training/epoch"
            )
            self.run.define_metric(
                f"{prefix}/val_loss", step_metric="training/epoch"
            )
            self._defined_metric_prefixes.add(prefix)
        self.run.log({
            "training/epoch": epoch + 1,
            f"{prefix}/train_loss": float(history["train_loss"][-1]),
            f"{prefix}/val_loss": float(history["val_loss"][-1]),
        })

    def publish_record(
        self,
        key: str,
        record: dict[str, Any],
        output_dir: Path,
    ) -> tuple[str, str]:
        checkpoint = output_dir / str(record["checkpoint"])
        prediction = output_dir / str(record["prediction_file"])
        results_path = output_dir / "results.json"
        for path, field in (
            (checkpoint, "checkpoint_sha256"),
            (prediction, "prediction_sha256"),
        ):
            if not path.is_file() or _sha256_file(path) != record.get(field):
                raise RuntimeError(f"local {field.removesuffix('_sha256')} failed SHA-256 before W&B upload")
        identity = {
            "experiment_fingerprint": self.fingerprint,
            "record_key": key,
            "record": record,
        }
        record_fingerprint = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()
        artifact = self.wandb.Artifact(
            name=f"p4-saturation-record-{record_fingerprint}",
            type="p4-saturation-record",
            metadata={
                **identity,
                "dataset_fingerprint": self.dataset_fingerprint,
                "record_fingerprint": record_fingerprint,
                "model": record["model"],
                "cap": record["cap"],
                "seed": record["seed"],
                "protocol": record["mode"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "prediction_sha256": record["prediction_sha256"],
            },
            description="One complete, SHA-verified P4 saturation model/prediction record.",
        )
        artifact.add_file(str(checkpoint), name=f"checkpoints/{checkpoint.name}")
        artifact.add_file(str(prediction), name=f"predictions/{prediction.name}")
        artifact.add_file(str(results_path), name="results.json")
        return self._commit(artifact), record_fingerprint

    def publish_completed_records(
        self,
        results: dict[str, Any],
        output_dir: Path,
        results_path: Path,
    ) -> None:
        for key, record in sorted(results["records"].items()):
            if (not isinstance(record, dict) or record.get("status") != "complete"
                    or record.get("wandb_artifact")
                    or not _record_is_complete(record, output_dir)):
                continue
            reference, fingerprint = self.publish_record(key, record, output_dir)
            record["wandb_artifact"] = reference
            record["wandb_artifact_fingerprint"] = fingerprint
            _atomic_json_dump(results, results_path)

    def publish_results(self, results: dict[str, Any], results_path: Path) -> None:
        stable_results = {
            key: (
                {
                    record_key: {
                        field: value for field, value in record.items()
                        if field not in {"wandb_artifact", "wandb_artifact_fingerprint"}
                    }
                    for record_key, record in value.items()
                }
                if key == "records" else value
            )
            for key, value in results.items()
            if key != "wandb"
        }
        result_fingerprint = hashlib.sha256(
            _canonical_json(stable_results).encode("utf-8")
        ).hexdigest()
        tracking = self._scope(results)
        if (tracking.get("results_artifact_fingerprint") == result_fingerprint
                and tracking.get("results_artifact")):
            return
        artifact = self.wandb.Artifact(
            name=f"p4-saturation-results-{result_fingerprint}",
            type="p4-saturation-results",
            metadata={
                "experiment_fingerprint": self.fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
                "results_fingerprint": result_fingerprint,
            },
            description="Complete P4 saturation results and per-record W&B references.",
        )
        artifact.add_file(str(results_path), name="results.json")
        reference = self._commit(artifact)
        tracking.update({
            "results_artifact_fingerprint": result_fingerprint,
            "results_artifact": reference,
        })
        _atomic_json_dump(results, results_path)

    def finish(self, exit_code: int) -> None:
        if not self._finished:
            self._finished = True
            self.run.finish(exit_code=exit_code)


# ---------------------------------------------------------------------------
# Cap-specific clip construction.  The physical/ridge path is intentionally
# explicit: it always receives exactly eight raw channels.  No even-channel
# heuristic is used here; only the helper-generated neural path has 16 rows.


def _clip_key(clip: Any) -> tuple[str, int | None, float]:
    rid = getattr(clip, "realization_idx", None)
    try:
        rid = int(rid) if rid is not None else None
    except Exception:
        rid = None
    return str(clip.design_id), rid, float(clip.velocity)


def _source_index_map(bundle: dict[str, Any]) -> dict[tuple[str, int | None, float], list[int]]:
    """Map duplicate physical keys to source rows in stable occurrence order.

    The generator intentionally has two excitation rows for each
    (design, realization, velocity).  ``_Clip`` does not carry excitation_idx,
    so a scalar key would collapse valid rows.  ``build_clips`` preserves
    source order; queues retain every occurrence and consume one per emitted
    clip.
    """
    out: dict[tuple[str, int | None, float], list[int]] = {}
    for i, (did, rid, vel) in enumerate(
        zip(
            bundle["design_ids"].tolist(),
            bundle["realization_ids"].tolist(),
            bundle["velocities"].tolist(),
        )
    ):
        try:
            rid_key = int(rid)
        except Exception:
            rid_key = None
        key = (str(did), rid_key, float(vel))
        out.setdefault(key, []).append(i)
    return out


def _indices_for_clips(
    clips: list[Any],
    source_map: dict[tuple[str, int | None, float], list[int]],
) -> np.ndarray:
    used: dict[tuple[str, int | None, float], int] = {}
    indices = []
    for clip in clips:
        key = _clip_key(clip)
        rows = source_map.get(key)
        offset = used.get(key, 0)
        if rows is None or offset >= len(rows):
            raise ValueError(f"clip key occurrence {key!r} is absent from loaded metadata")
        indices.append(rows[offset])
        used[key] = offset + 1
    return np.asarray(indices, dtype=int)


def _assert_raw_physics(clips: Iterable[Any]) -> None:
    for clip in clips:
        shape = np.asarray(clip.sensor_signals).shape
        if shape != (N_PHYSICAL_CHANNELS, SEQUENCE_LENGTH):
            raise ValueError(
                "PhysicsFeatureRidge input must be explicit raw (8, 512) "
                f"channels; got {shape}"
            )


def _assert_model_channels(clips: Iterable[Any]) -> None:
    for clip in clips:
        shape = np.asarray(clip.sensor_signals).shape
        if shape != (N_MODEL_CHANNELS, SEQUENCE_LENGTH):
            raise ValueError(f"model input must be (16, 512) after validity masks; got {shape}")


def _make_cap_parts(bundle: dict[str, Any], cap: float, need_clean: bool) -> dict[str, Any]:
    pre = bundle["pre"]
    capped = np.clip(pre, -float(cap), float(cap)).astype(np.float32, copy=False)
    sat_fracs = np.mean(np.abs(pre) > float(cap), axis=(1, 2), dtype=np.float64)
    source_map = _source_index_map(bundle)
    common = dict(
        dts=bundle["dts"],
        realization_ids=bundle["realization_ids"],
        sat_fracs=sat_fracs,
    )
    parts = {}
    for split, ids in bundle["splits"].items():
        raw, _ = build_clips(
            capped,
            bundle["margins"],
            bundle["velocities"],
            bundle["design_ids"],
            ids,
            **common,
        )
        if not raw:
            raise ValueError(f"{split} has no clips at cap {cap:g}")
        _assert_raw_physics(raw)
        idx = _indices_for_clips(raw, source_map)
        growth = bundle["clamp_fracs"][idx] > 0.0
        sat = np.asarray([float(getattr(c, "sat_frac", 0.0)) > 0.0 for c in raw], dtype=bool)
        parts[split] = {"raw": raw, "indices": idx, "growth": growth, "saturated": sat}
    select_ids, calib_ids = _split_select_calibrate(sorted(bundle["splits"]["val"]))
    parts["select_ids"] = select_ids
    parts["calib_ids"] = calib_ids
    val_raw = parts["val"]["raw"]
    val_indices = parts["val"]["indices"]
    select_mask = np.asarray([str(c.design_id) in select_ids for c in val_raw], dtype=bool)
    calib_mask = ~select_mask
    parts["select"] = {
        "raw": [c for c in val_raw if str(c.design_id) in select_ids],
        "indices": val_indices[select_mask],
        "growth": bundle["clamp_fracs"][val_indices[select_mask]] > 0.0,
        "saturated": np.asarray(
            [float(getattr(c, "sat_frac", 0.0)) > 0.0 for c in val_raw if str(c.design_id) in select_ids],
            dtype=bool,
        ),
    }
    parts["calib"] = {
        "raw": [c for c in val_raw if str(c.design_id) in calib_ids],
        "indices": val_indices[calib_mask],
        "growth": bundle["clamp_fracs"][val_indices[calib_mask]] > 0.0,
        "saturated": np.asarray(
            [float(getattr(c, "sat_frac", 0.0)) > 0.0 for c in val_raw if str(c.design_id) in calib_ids],
            dtype=bool,
        ),
    }
    if not parts["select"]["raw"] or not parts["calib"]["raw"]:
        raise ValueError("select/calibration clips are empty after design partition")
    if need_clean:
        for split in ("train", "val", "test"):
            clean = with_ones_mask(parts[split]["raw"])
            _assert_model_channels(clean)
            parts[split]["clean"] = clean
        parts["select"]["clean"] = [
            c for c in parts["val"]["clean"] if str(c.design_id) in select_ids
        ]
        parts["calib"]["clean"] = [
            c for c in parts["val"]["clean"] if str(c.design_id) in calib_ids
        ]
        _assert_model_channels(parts["select"]["clean"])
        _assert_model_channels(parts["calib"]["clean"])
    return parts


# ---------------------------------------------------------------------------
# Calibration and metric reporting


def _finite_sample_residual_adjustment(residuals: np.ndarray, alpha: float = ALPHA) -> tuple[float, int, int]:
    """Return corrected finite-sample conformal order statistic.

    ``k = ceil((n + 1)(1-alpha))`` is capped at n, and the kth smallest score
    is selected with ``partition`` (the equivalent ``higher`` quantile).  This
    is deliberately separate from the historical decision-metrics helper,
    whose ridge utility uses an uncorrected numpy quantile.
    """
    scores = np.asarray(residuals, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size < 2:
        raise ValueError(f"need >=2 finite ridge calibration residuals, got {scores.size}")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha!r}")
    n = int(scores.size)
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    value = float(np.partition(scores, k - 1)[k - 1])
    return value, k, n


def _ridge_adjustment(predictor: PhysicsFeatureRidge, calib_clips: list[Any]) -> dict[str, Any]:
    _assert_raw_physics(calib_clips)
    y = np.asarray([float(c.margin) for c in calib_clips], dtype=float)
    pred = np.asarray(predictor.predict(calib_clips), dtype=float).reshape(-1)
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        raise ValueError("ridge calibration predictions are not finite/aligned")
    adjustment, k, n = _finite_sample_residual_adjustment(np.abs(y - pred), ALPHA)
    return {
        "adjustment": adjustment,
        "calib_n": n,
        "order_statistic_k": k,
        "method": "finite_sample_abs_residual_order_statistic_higher",
        "alpha": ALPHA,
    }


def _interval_stats(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if not (y.shape == lo.shape == hi.shape) or not y.size:
        raise ValueError("interval arrays must be aligned and nonempty")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("interval arrays must be finite")
    if np.any(lo > hi):
        raise ValueError("interval lower exceeds upper")
    return {
        "interval_score": interval_score(y, lo, hi, ALPHA),
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "width": float(np.mean(hi - lo)),
        "mean_width": float(np.mean(hi - lo)),
    }


def _policy_rows(
    y: np.ndarray,
    med: np.ndarray,
    raw_lo: np.ndarray,
    cqr_lo: np.ndarray | None,
    dids: list[str],
) -> dict[str, Any]:
    table = policy_table(y, med, raw_lo, cqr_lo, dids)
    rows = {}
    for name in ("median", "raw_lb", "cqr_lb"):
        rows[name] = table["rows"].get(name)
    rows["denominators"] = {
        "n_unsafe": int(table["n_unsafe"]),
        "n_safe": int(table["n_safe"]),
        "n_total": int(table["n_total"]),
    }
    return _json_ready(rows)


def _identity_check(policies: dict[str, Any]) -> dict[str, Any]:
    den = policies.get("denominators", {})
    n_unsafe = int(den.get("n_unsafe", 0))
    n_safe = int(den.get("n_safe", 0))
    n_total = int(den.get("n_total", 0))
    rows = {}
    checks = []
    for name in ("median", "raw_lb", "cqr_lb"):
        row = policies.get(name)
        if row is None:
            rows[name] = {"status": "not_applicable", "passed": None}
            continue
        lhs = row.get("a")
        terms = []
        rhs = 0.0
        if n_safe:
            if row.get("a_s") is None:
                rows[name] = {"status": "missing_safe_denominator", "passed": None}
                continue
            rhs += (n_safe / n_total) * float(row["a_s"]) if n_total else 0.0
            terms.append("safe")
        if n_unsafe:
            if row.get("f") is None:
                rows[name] = {"status": "missing_unsafe_denominator", "passed": None}
                continue
            rhs += (n_unsafe / n_total) * float(row["f"]) if n_total else 0.0
            terms.append("unsafe")
        if lhs is None or not n_total:
            rows[name] = {"status": "empty", "passed": None}
            continue
        err = abs(float(lhs) - rhs)
        passed = bool(err <= 1e-12)
        rows[name] = {
            "status": "checked" if len(terms) == 2 else "checked_partial_denominator",
            "passed": passed,
            "lhs_a": float(lhs),
            "rhs": float(rhs),
            "absolute_error": float(err),
        }
        checks.append(passed)
    return {
        "formula": "a = (1-pi_u) a_s + pi_u f, pi_u=n_unsafe/n_total",
        "rows": rows,
        "passed": bool(all(checks)) if checks else None,
    }


def _per_design_records(
    y: np.ndarray,
    med: np.ndarray,
    raw_lo: np.ndarray,
    raw_hi: np.ndarray,
    cqr_lo: np.ndarray | None,
    cqr_hi: np.ndarray | None,
    dids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    y = np.asarray(y, dtype=float)
    med = np.asarray(med, dtype=float)
    raw_lo = np.asarray(raw_lo, dtype=float)
    raw_hi = np.asarray(raw_hi, dtype=float)
    cqr_lo_arr = None if cqr_lo is None else np.asarray(cqr_lo, dtype=float)
    cqr_hi_arr = None if cqr_hi is None else np.asarray(cqr_hi, dtype=float)
    out = []
    mae_map = {}
    for did in sorted(set(str(x) for x in dids)):
        idx = np.asarray([str(x) == did for x in dids], dtype=bool)
        rec = {
            "design_id": did,
            "n": int(idx.sum()),
            "mae": float(np.mean(np.abs(med[idx] - y[idx]))),
            "raw_interval_score": interval_score(y[idx], raw_lo[idx], raw_hi[idx]),
            "raw_coverage": float(np.mean((y[idx] >= raw_lo[idx]) & (y[idx] <= raw_hi[idx]))),
            "raw_width": float(np.mean(raw_hi[idx] - raw_lo[idx])),
        }
        if cqr_lo_arr is not None and cqr_hi_arr is not None:
            rec.update({
                "cqr_interval_score": interval_score(y[idx], cqr_lo_arr[idx], cqr_hi_arr[idx]),
                "cqr_coverage": float(np.mean((y[idx] >= cqr_lo_arr[idx]) & (y[idx] <= cqr_hi_arr[idx]))),
                "cqr_width": float(np.mean(cqr_hi_arr[idx] - cqr_lo_arr[idx])),
            })
        out.append(rec)
        mae_map[did] = rec["mae"]
    return out, mae_map


def _stratum_report(
    name: str,
    mask: np.ndarray,
    y: np.ndarray,
    med: np.ndarray,
    raw_lo: np.ndarray,
    raw_hi: np.ndarray,
    cqr_lo: np.ndarray | None,
    cqr_hi: np.ndarray | None,
    dids: list[str],
) -> dict[str, Any]:
    idx = np.asarray(mask, dtype=bool)
    if idx.shape != np.asarray(y).shape:
        raise ValueError(f"{name} stratum mask shape mismatch")
    ys = np.asarray(y)[idx]
    meds = np.asarray(med)[idx]
    los = np.asarray(raw_lo)[idx]
    his = np.asarray(raw_hi)[idx]
    ds = [str(dids[i]) for i in np.flatnonzero(idx).tolist()]
    if ys.size:
        raw = _interval_stats(ys, los, his)
        cqr = None
        if cqr_lo is not None and cqr_hi is not None:
            cqr = _interval_stats(ys, np.asarray(cqr_lo)[idx], np.asarray(cqr_hi)[idx])
        pol = _policy_rows(ys, meds, los, None if cqr_lo is None else np.asarray(cqr_lo)[idx], ds)
        identity = _identity_check(pol)
        return {
            "n": int(ys.size),
            "n_designs": int(len(set(ds))),
            "metrics": {
                "mae": float(np.mean(np.abs(meds - ys))),
                "interval_score": raw["interval_score"],
                "coverage": raw["coverage"],
                "width": raw["width"],
                "raw": raw,
                "cqr": cqr,
            },
            "policies": pol,
            "identity_check": identity,
        }
    # Empty strata are real outcomes, not fabricated zeroes.  Policy rows are
    # represented as unavailable, preserving denominator semantics.
    pol = {
        "median": None,
        "raw_lb": None,
        "cqr_lb": None,
        "denominators": {"n_unsafe": 0, "n_safe": 0, "n_total": 0},
    }
    return {
        "n": 0,
        "n_designs": 0,
        "metrics": {
            "mae": None,
            "interval_score": None,
            "coverage": None,
            "width": None,
            "raw": None,
            "cqr": None,
        },
        "policies": pol,
        "identity_check": _identity_check(pol),
    }


def _report(
    y: np.ndarray,
    med: np.ndarray,
    raw_lo: np.ndarray,
    raw_hi: np.ndarray,
    cqr_lo: np.ndarray | None,
    cqr_hi: np.ndarray | None,
    saturated: np.ndarray,
    growth_clamped: np.ndarray,
    dids: list[str],
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1)
    med = np.asarray(med, dtype=float).reshape(-1)
    raw_lo = np.asarray(raw_lo, dtype=float).reshape(-1)
    raw_hi = np.asarray(raw_hi, dtype=float).reshape(-1)
    sat = np.asarray(saturated, dtype=bool).reshape(-1)
    clamp = np.asarray(growth_clamped, dtype=bool).reshape(-1)
    dids = [str(d) for d in dids]
    if not (len(y) == len(med) == len(raw_lo) == len(raw_hi) == len(sat) == len(clamp) == len(dids)):
        raise ValueError("report arrays are not aligned")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(med))):
        raise ValueError("point arrays must be finite")
    raw = _interval_stats(y, raw_lo, raw_hi)
    cqr = None
    if cqr_lo is not None or cqr_hi is not None:
        if cqr_lo is None or cqr_hi is None:
            raise ValueError("CQR lower/upper must both be available")
        cqr = _interval_stats(y, cqr_lo, cqr_hi)
    policies = _policy_rows(y, med, raw_lo, cqr_lo, dids)
    per_design, mae_map = _per_design_records(y, med, raw_lo, raw_hi, cqr_lo, cqr_hi, dids)
    strata = {
        "saturated": _stratum_report("saturated", sat, y, med, raw_lo, raw_hi, cqr_lo, cqr_hi, dids),
        "unsaturated": _stratum_report("unsaturated", ~sat, y, med, raw_lo, raw_hi, cqr_lo, cqr_hi, dids),
        "growth_clamped": _stratum_report("growth_clamped", clamp, y, med, raw_lo, raw_hi, cqr_lo, cqr_hi, dids),
        "growth_nonclamped": _stratum_report("growth_nonclamped", ~clamp, y, med, raw_lo, raw_hi, cqr_lo, cqr_hi, dids),
    }
    return {
        "metrics": {
            "mae": float(np.mean(np.abs(med - y))),
            "interval_score": raw["interval_score"],
            "coverage": raw["coverage"],
            "width": raw["width"],
            "raw": raw,
            "cqr": cqr,
        },
        "policies": policies,
        "identity_check": _identity_check(policies),
        "strata": strata,
        "per_design": per_design,
        "per_design_mae": mae_map,
    }


# ---------------------------------------------------------------------------
# Interruption-safe neural training state
TRAINING_STATE_SCHEMA_VERSION = 1


def _training_state_path(output_dir: Path, train_cap: float, seed: int) -> Path:
    cap = _cap_token(train_cap)
    return output_dir / "checkpoints" / f"gru_ell_seed{int(seed)}_traincap{cap}.training.pt"


def _save_training_state(path: Path, state: dict[str, Any], *, expected_epochs: int) -> str:
    """Atomically persist validated completed-epoch training state."""
    from experiment_p4_phase2 import _validate_training_state
    validated = _validate_training_state(state, expected_epochs=int(expected_epochs))
    if int(validated.get("schema_version", 0)) != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported training-state schema version")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(validated, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return _sha256_file(path)


def _load_training_state(path: Path, *, expected_epochs: int) -> dict[str, Any] | None:
    """Load validated training state; corrupt/incompatible files fail the resume."""
    from experiment_p4_phase2 import _validate_training_state
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    try:
        validated = _validate_training_state(raw, expected_epochs=int(expected_epochs))
    except (ValueError, TypeError, AttributeError):
        return None
    if int(validated.get("schema_version", 0)) != TRAINING_STATE_SCHEMA_VERSION:
        return None
    return validated  # type: ignore[return-value]


def _training_state_is_compatible(
    state: dict[str, Any],
    *,
    model_factory,
    epochs: int,
    lr: float,
    batch_size: int,
    fingerprint: str | None = None,
) -> bool:
    """Check optimizer/scheduler/model shapes against this run before trusting a sidecar."""
    try:
        saved_model = state.get("model")
        if not isinstance(saved_model, dict) or not saved_model:
            return False
        probe = model_factory()
        current = probe.state_dict()
        if set(current.keys()) != set(saved_model.keys()):
            return False
        for key, tensor in current.items():
            saved = saved_model[key]
            if tuple(getattr(tensor, "shape", ())) != tuple(getattr(saved, "shape", ())):
                return False
            if str(getattr(tensor, "dtype", "")) != str(getattr(saved, "dtype", "")):
                return False
        probe_opt = torch.optim.Adam(probe.parameters(), lr=float(lr))
        probe_opt.load_state_dict(state.get("optimizer"))
        probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt, int(epochs))
        scheduler_state = state.get("scheduler")
        if scheduler_state is not None:
            probe_sched.load_state_dict(scheduler_state)
        saved_fingerprint = state.get("dataset_fingerprint")
        if fingerprint is not None and saved_fingerprint is not None and str(saved_fingerprint) != str(fingerprint):
            return False
        del probe, probe_opt, probe_sched
        return True
    except Exception:
        return False


def results_fingerprint(records: dict[str, Any]) -> str | None:
    """Best-effort experiment fingerprint for sidecar compatibility checks."""
    try:
        for record in records.values():
            if isinstance(record, dict):
                for key in ("experiment_fingerprint", "dataset_fingerprint", "fingerprint"):
                    value = record.get(key)
                    if isinstance(value, str) and value:
                        return value
    except Exception:
        pass
    return None


def _save_completed_training_state(path: Path, state: dict[str, Any], *, expected_epochs: int) -> str:
    """Persist post-training state before prediction/calibration side effects."""
    if not isinstance(state, dict) or state.get("completed_training") is not True:
        raise ValueError("completed training state must be marked complete")
    for field in ("checkpoint", "checkpoint_sha256", "model", "best_model", "history"):
        if field not in state:
            raise ValueError(f"completed training state is missing {field!r}")
    history = state.get("history")
    if not isinstance(history, dict):
        raise ValueError("completed training history must be a mapping")
    if len(history.get("train_loss", [])) != int(expected_epochs):
        raise ValueError("completed training history must cover all epochs")
    if len(history.get("val_loss", [])) != int(expected_epochs):
        raise ValueError("completed training history must cover all epochs")
    if int(state.get("epoch", -1)) != int(expected_epochs) - 1:
        raise ValueError("completed training epoch must be the final epoch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return _sha256_file(path)


def _load_completed_training_state(path: Path, *, expected_epochs: int) -> dict[str, Any] | None:
    """Load post-training state; invalid files force fresh training, never reuse."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(raw, dict) or raw.get("completed_training") is not True:
        return None
    try:
        if int(raw.get("schema_version", 0)) != TRAINING_STATE_SCHEMA_VERSION:
            return None
        if int(raw.get("epoch", -1)) != int(expected_epochs) - 1:
            return None
        history = raw.get("history")
        if not isinstance(history, dict):
            return None
        if len(history.get("train_loss", [])) != int(expected_epochs):
            return None
        if len(history.get("val_loss", [])) != int(expected_epochs):
            return None
        if not isinstance(raw.get("model"), dict) or not isinstance(raw.get("best_model"), dict):
            return None
        for field in ("checkpoint", "checkpoint_sha256"):
            if not isinstance(raw.get(field), str) or not raw.get(field):
                return None
    except Exception:
        return None
    return raw  # type: ignore[return-value]



# ---------------------------------------------------------------------------
# Atomic checkpoint and prediction persistence


def _checkpoint_path(output_dir: Path, kind: str, train_cap: float, seed: int | None = None) -> Path:
    cap = _cap_token(train_cap)
    if kind == "gru":
        return output_dir / "checkpoints" / f"gru_ell_seed{int(seed)}_traincap{cap}.pt"
    return output_dir / "checkpoints" / f"physics_ridge_traincap{cap}.npz"


def _save_torch_checkpoint(model: torch.nn.Module, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(model.state_dict(), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return _sha256_file(path)


def _load_torch_checkpoint(path: Path, seed: int, device: str = "cpu") -> torch.nn.Module:
    model = GRUEllQuantile(N_MODEL_CHANNELS)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _save_ridge_checkpoint(model: PhysicsFeatureRidge, path: Path) -> str:
    if model.mean_ is None or model.std_ is None or model.coef_ is None or model.intercept_ is None:
        raise ValueError("cannot checkpoint an unfitted ridge model")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    metadata = json.dumps({"alpha": float(model.alpha), "schema_version": SCHEMA_VERSION, "use_velocity": bool(getattr(model, "use_velocity", False))}, sort_keys=True)
    with open(tmp, "wb") as f:
        np.savez(
            f,
            mean=np.asarray(model.mean_, dtype=np.float64),
            std=np.asarray(model.std_, dtype=np.float64),
            coef=np.asarray(model.coef_, dtype=np.float64),
            intercept=np.asarray([float(model.intercept_)], dtype=np.float64),
            metadata=np.asarray(metadata),
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return _sha256_file(path)


def _load_ridge_checkpoint(path: Path) -> PhysicsFeatureRidge:
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"].item()))
        model = PhysicsFeatureRidge(alpha=float(metadata["alpha"]),
                                    use_velocity=bool(metadata.get("use_velocity", False)))
        model.mean_ = np.asarray(z["mean"], dtype=float)
        model.std_ = np.asarray(z["std"], dtype=float)
        model.coef_ = np.asarray(z["coef"], dtype=float)
        model.intercept_ = float(np.asarray(z["intercept"], dtype=float).reshape(-1)[0])
    if int(model.coef_.shape[0]) != int(model.mean_.shape[0]):
        raise ValueError("ridge checkpoint shapes disagree")
    if model.use_velocity and int(model.coef_.shape[0]) != 8:
        raise ValueError("velocity ridge checkpoint must carry 8 features")
    if not model.use_velocity and int(model.coef_.shape[0]) != 7:
        raise ValueError("plain ridge checkpoint must carry 7 features")
    return model


def _prediction_path(output_dir: Path, key: str) -> Path:
    return output_dir / "predictions" / f"{_safe_key(key)}.npz"


def _save_predictions(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return _sha256_file(path)


def _record_is_complete(record: Any, output_dir: Path) -> bool:
    if not isinstance(record, dict) or record.get("status") != "complete":
        return False
    required = (
        "model",
        "seed",
        "cap",
        "mode",
        "checkpoint",
        "checkpoint_sha256",
        "prediction_file",
        "prediction_sha256",
        "metrics",
        "policies",
        "identity_check",
        "per_design_mae",
    )
    if any(k not in record for k in required):
        return False
    try:
        ckpt = output_dir / str(record["checkpoint"])
        pred = output_dir / str(record["prediction_file"])
        return (
            ckpt.is_file()
            and pred.is_file()
            and _sha256_file(ckpt) == str(record["checkpoint_sha256"])
            and _sha256_file(pred) == str(record["prediction_sha256"])
        )
    except (OSError, ValueError):
        return False


def _known_checkpoint_hash(records: dict[str, Any], output_dir: Path, checkpoint: Path) -> str | None:
    rel = _relative_path(checkpoint, output_dir)
    for record in records.values():
        if not _record_is_complete(record, output_dir):
            continue
        if record.get("checkpoint") == rel:
            digest = str(record.get("checkpoint_sha256", ""))
            if digest and _sha256_file(checkpoint) == digest:
                return digest
    return None


def _initial_results(fingerprint: str, payload: dict[str, Any], config: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "config": config,
        "dataset": {
            "directory": str(bundle["dataset_dir"]),
            "n_clips": int(bundle["n_clips"]),
            "clip_shape": list(bundle["pre"].shape),
            "n_designs": int(len(set(bundle["design_ids"].tolist()))),
            "splits": {k: sorted(v) for k, v in bundle["splits"].items()},
            "pre_saturation": {
                "file": "pre_saturation.npy",
                "stage": "post_normalization_pre_sensor_cap",
                "modal_growth_clamp_retained": True,
                "cap50_replay_checked": True,
            },
        },
        "interpretation": {
            "coverage": (
                "Coverage is empirical on clustered clips (designs share realizations and "
                "velocities); heldout CQR calibration does not provide a theoretical "
                "new-design guarantee here."
            ),
            "growth_clipping_caveat": (
                "Modal growth exponents were safety-clamped before normalization; the "
                "growth_clamped stratum can reflect that numerical rule rather than an "
                "exact linear transient."
            ),
            "comparison_convention": (
                "paired_design_comparison reports mean(first - second) over matched "
                "design-level MAE records; bootstrap intervals are descriptive."
            ),
            "fixed_transfer_calibration": (
                "Fixed-transfer rows use the cap-50 calibration adjustment unchanged at "
                "higher caps; no recalibration is performed."
            ),
            "ridge_calibration": (
                "Ridge intervals use a heldout absolute-residual finite-sample order "
                "statistic; this is not the historical uncorrected helper quantile."
            ),
        },
        "records": {},
        "comparisons": {},
    }


def _load_or_initialize_results(
    path: Path,
    fingerprint: str,
    payload: dict[str, Any],
    config: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        if results.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing results fingerprint mismatch: dataset, source dependency, "
                "or configuration changed; refusing to mix artifacts."
            )
        if results.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Existing results schema differs; choose a new output directory")
        if not isinstance(results.get("records"), dict):
            raise RuntimeError("Existing results has no records mapping")
        results.setdefault("fingerprint_payload", payload)
        results.setdefault("config", config)
        results.setdefault("comparisons", {})
        return results
    results = _initial_results(fingerprint, payload, config, bundle)
    _atomic_json_dump(results, path)
    return results


# ---------------------------------------------------------------------------
# Model-level restart orchestration


def _ensure_gru(
    *,
    output_dir: Path,
    records: dict[str, Any],
    train_cap: float,
    seed: int,
    train_raw: list[Any],
    select_clean: list[Any],
    epochs: int,
    batch_size: int,
    predict_batch: int,
    device: str = "cpu",
    wandb_publisher: Any = None,
    protocol: str = "matched_retrain",
) -> dict[str, Any]:
    path = _checkpoint_path(output_dir, "gru", train_cap, seed)
    state_path = _training_state_path(output_dir, train_cap, seed)
    known = _known_checkpoint_hash(records, output_dir, path)
    if known is not None:
        try:
            if state_path.is_file():
                state_path.unlink()
        except OSError:
            pass
        model = _load_torch_checkpoint(path, seed, device=device)
        return {"model": model, "path": path, "sha256": known, "history": None, "train_cap": train_cap}

    fingerprint = results_fingerprint(records)
    completed = (_load_completed_training_state(state_path, expected_epochs=int(epochs))
                 if state_path.is_file() else None)
    if completed is not None:
        try:
            expected_rel = _relative_path(path, output_dir)
            ckpt_ok = (str(completed.get("checkpoint")) == expected_rel and path.is_file()
                       and _sha256_file(path) == str(completed.get("checkpoint_sha256")))
            model_state = completed.get("best_model") or completed.get("model")
            factory = lambda: GRUEllQuantile(N_MODEL_CHANNELS)
            probe = factory()
            probe.load_state_dict(model_state)
            del probe
            fingerprint_ok = (fingerprint is None or completed.get("dataset_fingerprint") in (None, fingerprint))
            history = completed.get("history")
            history_ok = (isinstance(history, dict)
                          and len(history.get("train_loss", [])) == int(epochs)
                          and len(history.get("val_loss", [])) == int(epochs))
            if ckpt_ok and fingerprint_ok and history_ok:
                model = factory()
                model.load_state_dict(model_state)
                model.to(device)
                model.eval()
                return {
                    "model": model,
                    "path": path,
                    "sha256": str(completed.get("checkpoint_sha256")),
                    "history": {"train_loss": [float(v) for v in history["train_loss"]],
                                "val_loss": [float(v) for v in history["val_loss"]]},
                    "train_cap": train_cap,
                    "train_time": 0.0,
                    "predict_batch": predict_batch,
                }
        except Exception:
            pass
        try:
            state_path.unlink()
        except OSError:
            pass
    factory = lambda: GRUEllQuantile(N_MODEL_CHANNELS)
    resume_state = None
    if state_path.is_file():
        candidate = _load_training_state(state_path, expected_epochs=int(epochs))
        if candidate is not None and not candidate.get("completed_training") and _training_state_is_compatible(
            candidate, model_factory=factory, epochs=int(epochs), lr=1e-3,
            batch_size=int(batch_size), fingerprint=fingerprint,
        ):
            resume_state = candidate
        else:
            try:
                state_path.unlink()
            except OSError:
                pass
    def _persist_completed_epoch(state: dict[str, Any]) -> None:
        state = dict(state)
        state["dataset_fingerprint"] = fingerprint
        _save_training_state(state_path, state, expected_epochs=int(epochs))
    train_dr, n_channels = apply_dr_with_mask(train_raw, seed=0)
    if n_channels != N_MODEL_CHANNELS:
        raise ValueError(f"DR helper returned {n_channels} channels, expected {N_MODEL_CHANNELS}")
    _assert_model_channels(train_dr)
    t0 = time.time()
    epoch_callback = None
    if wandb_publisher is not None:
        epoch_callback = lambda epoch, _model, history: wandb_publisher.log_epoch(
            epoch, history, train_cap=train_cap, seed=seed, protocol=protocol
        )
    model, history = train_quantile_generic(
        factory,
        train_dr,
        val_clips=select_clean,
        epochs=int(epochs),
        lr=1e-3,
        batch_size=int(batch_size),
        seed=int(seed),
        n_channels=N_MODEL_CHANNELS,
        on_epoch=epoch_callback,
        device=device,
        resume_state=resume_state,
        checkpoint_callback=_persist_completed_epoch,
    )
    del train_dr
    digest = _save_torch_checkpoint(model, path)
    # Completed epochs are committed before evaluation: an interruption after
    # this point resumes prediction/calibration instead of retraining.
    completed = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "epoch": int(epochs) - 1,
        "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": None,
        "scheduler": None,
        "best": float(history["val_loss"][-1]) if history.get("val_loss") else float("inf"),
        "best_model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "history": {"train_loss": list(history["train_loss"]), "val_loss": list(history["val_loss"])},
        "rng": None,
        "dataset_fingerprint": fingerprint,
        "completed_training": True,
        "checkpoint": _relative_path(path, output_dir),
        "checkpoint_sha256": digest,
    }
    _save_completed_training_state(state_path, completed, expected_epochs=int(epochs))
    model.eval()
    return {
        "model": model,
        "path": path,
        "sha256": digest,
        "history": history,
        "train_cap": train_cap,
        "train_time": time.time() - t0,
        "predict_batch": predict_batch,
    }


def _ensure_ridge(
    *,
    output_dir: Path,
    records: dict[str, Any],
    train_cap: float,
    train_raw: list[Any],
) -> dict[str, Any]:
    _assert_raw_physics(train_raw)
    path = _checkpoint_path(output_dir, "ridge", train_cap)
    known = _known_checkpoint_hash(records, output_dir, path)
    if known is not None:
        return {"model": _load_ridge_checkpoint(path), "path": path, "sha256": known, "train_cap": train_cap}
    model = PhysicsFeatureRidge()
    model.fit(train_raw)
    digest = _save_ridge_checkpoint(model, path)
    return {"model": model, "path": path, "sha256": digest, "train_cap": train_cap}


def _training_payload(artifact: dict[str, Any], seed: int | None, args: argparse.Namespace) -> dict[str, Any]:
    history = artifact.get("history")
    out = {
        "train_cap": float(artifact["train_cap"]),
        "epochs": int(args.epochs) if seed is not None else None,
        "batch_size": int(args.batch_size) if seed is not None else None,
        "seed": seed,
        "dr_seed": 0 if seed is not None else None,
        "quantile_weights": [2.0, 1.0, 1.0] if seed is not None else None,
    }
    if history:
        out.update({
            "final_train_loss": float(history["train_loss"][-1]),
            "final_select_loss": float(history["val_loss"][-1]),
            "train_time_seconds": float(artifact.get("train_time", 0.0)),
        })
    return out


def _store_record(
    *,
    key: str,
    results: dict[str, Any],
    output_dir: Path,
    model_name: str,
    seed: int | None,
    cap: float,
    mode: str,
    artifact: dict[str, Any],
    report: dict[str, Any],
    prediction_payload: dict[str, Any],
    calibration: dict[str, Any],
    train_cap: float,
    calibration_cap: float,
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> None:
    pred_path = _prediction_path(output_dir, key)
    pred_hash = _save_predictions(pred_path, prediction_payload)
    record = {
        "status": "complete",
        "model": model_name,
        "seed": seed,
        "cap": float(cap),
        "mode": mode,
        "train_cap": float(train_cap),
        "calibration_cap": float(calibration_cap),
        "recalibrated": bool(mode == "matched_retrain"),
        "checkpoint": _relative_path(Path(artifact["path"]), output_dir),
        "checkpoint_sha256": str(artifact["sha256"]),
        "prediction_file": _relative_path(pred_path, output_dir),
        "prediction_sha256": pred_hash,
        "calibration": calibration,
        "training": _training_payload(artifact, seed, args),
    }
    record.update(report)
    if extra:
        record.update(extra)
    results["records"][key] = _json_ready(record)
    _atomic_json_dump(results, output_dir / "results.json")
    publisher = getattr(args, "_wandb_publisher", None)
    if publisher is not None:
        reference, fingerprint = publisher.publish_record(
            key, results["records"][key], output_dir
        )
        results["records"][key]["wandb_artifact"] = reference
        results["records"][key]["wandb_artifact_fingerprint"] = fingerprint
        _atomic_json_dump(results, output_dir / "results.json")


def _neural_record(
    *,
    key: str,
    results: dict[str, Any],
    output_dir: Path,
    artifact: dict[str, Any],
    parts: dict[str, Any],
    cap: float,
    mode: str,
    seed: int,
    calibration: dict[str, Any],
    calibration_parts: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    test_clean = parts["test"]["clean"]
    pred = np.asarray(_predict(artifact["model"], test_clean, batch=int(args.predict_batch), device=args.device), dtype=float)
    if pred.ndim != 2 or pred.shape != (len(test_clean), 3):
        raise ValueError(f"quantile prediction shape {pred.shape} is not (n_test, 3)")
    y = np.asarray([float(c.margin) for c in test_clean], dtype=float)
    if not np.all(np.isfinite(pred)):
        raise ValueError("quantile predictions contain non-finite values")
    raw_lo, med, raw_hi = pred[:, 0], pred[:, 1], pred[:, 2]

    # Fixed-transfer rows reuse the already-fitted cap-50 adjustment exactly;
    # only matched rows may inspect this cap's heldout calibration partition.
    calibration = dict(calibration)
    if mode == "matched_retrain" or "adjustment" not in calibration:
        cal_clean = calibration_parts["clean"]
        cpred = np.asarray(_predict(artifact["model"], cal_clean, batch=int(args.predict_batch), device=args.device), dtype=float)
        cy = np.asarray([float(c.margin) for c in cal_clean], dtype=float)
        if cpred.ndim != 2 or cpred.shape != (len(cal_clean), 3) or not np.all(np.isfinite(cpred)):
            raise ValueError("quantile calibration prediction shape/non-finite values")
        adjustment = fit_cqr_adjustment(cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)
        calibration.update({
            "calib_n": int(len(cy)),
            "calib_designs": int(len(set(str(c.design_id) for c in cal_clean))),
        })
    else:
        adjustment = float(calibration["adjustment"])
    cqr_lo, cqr_hi = apply_cqr_adjustment(raw_lo, raw_hi, adjustment)
    calibration.update({
        "adjustment": float(adjustment),
        "method": "fit_cqr_adjustment_finite_sample_higher",
        "alpha": ALPHA,
    })
    dids = [str(c.design_id) for c in test_clean]
    report = _report(
        y,
        med,
        raw_lo,
        raw_hi,
        cqr_lo,
        cqr_hi,
        parts["test"]["saturated"],
        parts["test"]["growth"],
        dids,
    )
    payload = {
        "y_true": y,
        "predicted_quantiles": pred,
        "raw_lower": raw_lo,
        "median": med,
        "raw_upper": raw_hi,
        "cqr_lower": cqr_lo,
        "cqr_upper": cqr_hi,
        "saturated": np.asarray(parts["test"]["saturated"], dtype=bool),
        "growth_clamped": np.asarray(parts["test"]["growth"], dtype=bool),
        "design_ids": np.asarray(dids, dtype="U"),
        "velocities": np.asarray([float(c.velocity) for c in test_clean], dtype=np.float32),
    }
    _store_record(
        key=key,
        results=results,
        output_dir=output_dir,
        model_name=MODEL_NAME,
        seed=seed,
        cap=cap,
        mode=mode,
        artifact=artifact,
        report=report,
        prediction_payload=payload,
        calibration=calibration,
        train_cap=float(artifact["train_cap"]),
        calibration_cap=float(calibration.get("cap", calibration_parts.get("cap", cap))),
        args=args,
        extra={"prediction_batch": int(args.predict_batch)},
    )


def _ridge_record(
    *,
    key: str,
    results: dict[str, Any],
    output_dir: Path,
    artifact: dict[str, Any],
    parts: dict[str, Any],
    cap: float,
    mode: str,
    calibration: dict[str, Any],
    calibration_parts: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    test_raw = parts["test"]["raw"]
    cal_raw = calibration_parts["raw"]
    _assert_raw_physics(test_raw)
    _assert_raw_physics(cal_raw)
    pred = np.asarray(artifact["model"].predict(test_raw), dtype=float).reshape(-1)
    y = np.asarray([float(c.margin) for c in test_raw], dtype=float)
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        raise ValueError("ridge prediction shape/non-finite values")
    cal = dict(calibration)
    if mode == "matched_retrain" or "adjustment" not in cal:
        cal = _ridge_adjustment(artifact["model"], cal_raw)
        cal["cap"] = float(calibration_parts.get("cap", cap))
    adjustment = float(cal["adjustment"])
    raw_lo = pred - adjustment
    raw_hi = pred + adjustment
    dids = [str(c.design_id) for c in test_raw]
    report = _report(
        y,
        pred,
        raw_lo,
        raw_hi,
        None,
        None,
        parts["test"]["saturated"],
        parts["test"]["growth"],
        dids,
    )
    payload = {
        "y_true": y,
        "predicted_quantiles": np.column_stack([raw_lo, pred, raw_hi]),
        "raw_lower": raw_lo,
        "median": pred,
        "raw_upper": raw_hi,
        "cqr_lower": np.asarray([], dtype=float),
        "cqr_upper": np.asarray([], dtype=float),
        "saturated": np.asarray(parts["test"]["saturated"], dtype=bool),
        "growth_clamped": np.asarray(parts["test"]["growth"], dtype=bool),
        "design_ids": np.asarray(dids, dtype="U"),
        "velocities": np.asarray([float(c.velocity) for c in test_raw], dtype=np.float32),
    }
    cal.update({
        "kind": "ridge_absolute_residual",
        "recalibrated": bool(mode == "matched_retrain"),
    })
    _store_record(
        key=key,
        results=results,
        output_dir=output_dir,
        model_name="physics_feature_ridge",
        seed=None,
        cap=cap,
        mode=mode,
        artifact=artifact,
        report=report,
        prediction_payload=payload,
        calibration=cal,
        train_cap=float(artifact["train_cap"]),
        calibration_cap=float(cal.get("cap", calibration_parts.get("cap", cap))),
        args=args,
        extra={"prediction_batch": None},
    )


# ---------------------------------------------------------------------------
# Paired design comparisons and CLI driver


def _record_key(model: str, cap: float, mode: str, seed: int | None) -> str:
    seed_token = "shared" if seed is None else f"seed{int(seed)}"
    return f"{model}__cap{_cap_label(cap)}__{mode}__{seed_token}"


def _mean_design_records(records: list[dict[str, Any]]) -> list[tuple[str, float]] | None:
    if not records:
        return None
    maps = [r.get("per_design_mae") for r in records]
    if any(not isinstance(m, dict) or not m for m in maps):
        return None
    ids = sorted(maps[0])
    if any(sorted(m) != ids for m in maps):
        return None
    return [(did, float(np.mean([float(m[did]) for m in maps]))) for did in ids]


def _comparison(
    left_key: str,
    right_key: str,
    records: dict[str, Any],
    bootstrap: int,
    seed: int,
    metric: str = "mae",
) -> dict[str, Any]:
    left = records.get(left_key)
    right = records.get(right_key)
    base = {
        "left": left_key,
        "right": right_key,
        "metric": metric,
        "subtraction": "left - right (first-named minus second-named)",
    }
    if not (_record_is_complete(left, _COMPARISON_OUTPUT_DIR) and _record_is_complete(right, _COMPARISON_OUTPUT_DIR)):
        base["status"] = "incomplete (record/checkpoint/prediction missing or mismatched)"
        return base
    if metric != "mae":
        raise ValueError(f"unsupported paired metric {metric!r}")
    a = _mean_design_records([left])
    b = _mean_design_records([right])
    if a is None or b is None:
        base["status"] = "incomplete (per-design MAE missing)"
        return base
    base["status"] = "complete"
    base["result"] = paired_design_comparison(a, b, n_bootstrap=bootstrap, seed=seed)
    return base


# Set only during the driver; this avoids threading output_dir through the
# pure comparison helper while retaining strict artifact checks on resume.
_COMPARISON_OUTPUT_DIR = Path(".")


def _build_comparisons(
    results: dict[str, Any],
    output_dir: Path,
    caps: tuple[float, ...],
    seeds: tuple[int, ...],
    bootstrap: int,
) -> dict[str, Any]:
    global _COMPARISON_OUTPUT_DIR
    _COMPARISON_OUTPUT_DIR = output_dir
    records = results["records"]
    out: dict[str, Any] = {}
    # Per-seed neural and deterministic ridge fixed-vs-matched comparisons.
    for cap in caps:
        for seed in seeds:
            left = _record_key(MODEL_NAME, cap, "fixed_transfer", seed)
            right = _record_key(MODEL_NAME, cap, "matched_retrain", seed)
            out[f"{left}_minus_{right}"] = _comparison(left, right, records, bootstrap, seed)
        left = _record_key("physics_feature_ridge", cap, "fixed_transfer", None)
        right = _record_key("physics_feature_ridge", cap, "matched_retrain", None)
        out[f"{left}_minus_{right}"] = _comparison(left, right, records, bootstrap, 0)

    # Mean over requested neural seeds, then paired by design.  This is the
    # primary family-level comparison and keeps the averaging convention
    # explicit rather than pooling correlated clips.
    for cap in caps:
        for mode_a, mode_b in (("fixed_transfer", "matched_retrain"),):
            left_keys = [_record_key(MODEL_NAME, cap, mode_a, s) for s in seeds]
            right_keys = [_record_key(MODEL_NAME, cap, mode_b, s) for s in seeds]
            left_records = [records.get(k) for k in left_keys]
            right_records = [records.get(k) for k in right_keys]
            name = f"{MODEL_NAME}__cap{_cap_label(cap)}__mean_seed_{mode_a}_minus_{mode_b}"
            item = {
                "left": left_keys,
                "right": right_keys,
                "metric": "mae",
                "subtraction": "mean per-design MAE over seeds (fixed_transfer - matched_retrain)",
            }
            if all(_record_is_complete(r, output_dir) for r in left_records + right_records):
                a = _mean_design_records(left_records)
                b = _mean_design_records(right_records)
                if a is not None and b is not None:
                    item["status"] = "complete"
                    item["result"] = paired_design_comparison(a, b, n_bootstrap=bootstrap, seed=0)
                else:
                    item["status"] = "incomplete (per-design sets differ)"
            else:
                item["status"] = "incomplete (record/checkpoint/prediction missing or mismatched)"
            out[name] = item

    # Cap deltas expose sensitivity using the same declaration of subtraction.
    for model, model_seeds in ((MODEL_NAME, seeds), ("physics_feature_ridge", (None,))):
        for mode in ("fixed_transfer", "matched_retrain"):
            for cap in caps:
                if cap == 50.0:
                    continue
                for seed in model_seeds:
                    left = _record_key(model, cap, mode, seed)
                    right = _record_key(model, 50.0, mode, seed)
                    name = f"{left}_minus_{right}"
                    out[name] = _comparison(left, right, records, bootstrap, int(seed or 0))
    return out


def _config(args: argparse.Namespace, caps: tuple[float, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "device": str(getattr(args, "device", "cpu")),
        "caps": [float(c) for c in caps],
        "seeds": [int(s) for s in seeds],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "predict_batch": int(args.predict_batch),
        "threads": int(args.threads),
        "alpha": ALPHA,
        "model": MODEL_NAME,
        "model_input_channels": N_MODEL_CHANNELS,
        "raw_physics_channels": N_PHYSICAL_CHANNELS,
        "sequence_length": SEQUENCE_LENGTH,
        "quantile_weights": [2.0, 1.0, 1.0],
        "train_dr_seed": 0,
        "bootstrap": int(args.bootstrap),
        "calibration": {
            "neural": "heldout_design_CQR_fit_cqr_adjustment",
            "ridge": "heldout_design_abs_residual_finite_sample_order_statistic",
        },
    }


def _run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    caps = _validate_caps(args.caps)
    seeds = _validate_seeds(args.seeds)
    if args.epochs < 1 or args.batch_size < 1 or args.predict_batch < 1:
        raise ValueError("epochs, batch-size, and predict-batch must be positive")
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    try:
        device = torch.device(getattr(args, "device", "cpu"))
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid training device: {getattr(args, 'device', 'cpu')}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {device} requested, but CUDA is not available")
    args.device = str(device)
    _configure_threads(args.threads)
    bundle = _validate_dataset(args.dataset)
    config = _config(args, caps, seeds)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output-dir exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint, payload = _fingerprint_dataset(bundle, config)
    results_path = output_dir / "results.json"
    results = _load_or_initialize_results(results_path, fingerprint, payload, config, bundle)
    records: dict[str, Any] = results["records"]
    publisher = getattr(args, "_wandb_publisher", None)
    if publisher is not None:
        publisher.publish_dataset(bundle, payload, fingerprint, results, results_path)
        publisher.publish_completed_records(results, output_dir, results_path)

    expected = []
    for cap in caps:
        for mode in ("fixed_transfer", "matched_retrain"):
            expected.append((_record_key("physics_feature_ridge", cap, mode, None), "ridge", cap, mode, None))
            for seed in seeds:
                expected.append((_record_key(MODEL_NAME, cap, mode, seed), "gru", cap, mode, seed))
    pending = [item for item in expected if not _record_is_complete(records.get(item[0]), output_dir)]
    if not pending:
        results["comparisons"] = _build_comparisons(results, output_dir, caps, seeds, args.bootstrap)
        _atomic_json_dump(results, results_path)
        if publisher is not None:
            publisher.publish_results(results, results_path)
        print("All matching records/checkpoints/predictions are complete; nothing to retrain.", flush=True)
        return results

    fixed_gru_pending_any = any(
        item[1] == "gru" and item[3] == "fixed_transfer" for item in pending
    )
    fixed_ridge_pending_any = any(
        item[1] == "ridge" and item[3] == "fixed_transfer" for item in pending
    )
    fixed_gru_seeds = {
        int(item[4]) for item in pending
        if item[1] == "gru" and item[3] == "fixed_transfer"
    }
    fixed_gru: dict[int, dict[str, Any]] = {}
    fixed_ridge: dict[str, Any] | None = None

    # Cap 50 is both the historical reference and the only training source for
    # fixed transfer.  It is built before higher caps even when only a higher
    # cap record is missing, so fixed calibration cannot accidentally drift.
    for cap in caps:
        cap_pending = [item for item in pending if float(item[2]) == float(cap)]
        cap_keys = {item[0] for item in cap_pending}
        need_fixed_gru_here = any(
            item[1] == "gru" and item[3] == "fixed_transfer" for item in cap_pending
        )
        need_matched_gru = any(
            item[1] == "gru" and item[3] == "matched_retrain" for item in cap_pending
        )
        need_matched_ridge = any(
            item[1] == "ridge" and item[3] == "matched_retrain" for item in cap_pending
        )
        need_neural = need_fixed_gru_here or need_matched_gru or (
            cap == 50.0 and fixed_gru_pending_any
        )
        need_any = bool(cap_pending) or (
            cap == 50.0 and (fixed_gru_pending_any or fixed_ridge_pending_any)
        )
        if not need_any:
            continue

        print(f"Preparing cap {cap:g} ({len(cap_pending)} incomplete records)...", flush=True)
        parts = _make_cap_parts(bundle, cap, need_clean=need_neural)
        calib_parts = {"raw": parts["calib"]["raw"], "cap": cap}
        select_clean: list[Any] = []
        if need_neural:
            select_clean = parts["select"]["clean"]
            calib_parts["clean"] = parts["calib"]["clean"]

        matched_gru: dict[int, dict[str, Any]] = {}
        matched_ridge: dict[str, Any] | None = None

        if cap == 50.0:
            if fixed_gru_pending_any:
                for seed in sorted(fixed_gru_seeds):
                    fixed_gru[seed] = _ensure_gru(
                        output_dir=output_dir,
                        records=records,
                        train_cap=50.0,
                        seed=seed,
                        train_raw=parts["train"]["raw"],
                        select_clean=select_clean,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        predict_batch=args.predict_batch,
                        device=args.device,
                        wandb_publisher=publisher,
                        protocol="fixed_transfer",
                    )
                    cpred = np.asarray(
                        _predict(
                            fixed_gru[seed]["model"],
                            calib_parts["clean"],
                            batch=args.predict_batch,
                            device=args.device,
                        ),
                        dtype=float,
                    )
                    cy = np.asarray(
                        [float(c.margin) for c in calib_parts["clean"]], dtype=float
                    )
                    if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
                        raise ValueError("fixed cap-50 calibration predictions are invalid")
                    fixed_gru[seed]["cqr"] = {
                        "adjustment": float(
                            fit_cqr_adjustment(cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)
                        ),
                        "cap": 50.0,
                        "calib_n": int(len(cy)),
                        "calib_designs": int(
                            len(set(str(c.design_id) for c in calib_parts["clean"]))
                        ),
                    }
            if fixed_ridge_pending_any:
                fixed_ridge = _ensure_ridge(
                    output_dir=output_dir,
                    records=records,
                    train_cap=50.0,
                    train_raw=parts["train"]["raw"],
                )
                fixed_ridge["calibration"] = _ridge_adjustment(
                    fixed_ridge["model"], calib_parts["raw"]
                )
                fixed_ridge["calibration"]["cap"] = 50.0

            # At cap 50, matched and fixed training protocols coincide.  Reuse
            # an already established fixed artifact; otherwise train only the
            # seed(s) with an incomplete matched entry.
            for seed in seeds:
                key = _record_key(MODEL_NAME, 50.0, "matched_retrain", seed)
                if key not in cap_keys:
                    continue
                if seed in fixed_gru:
                    matched_gru[seed] = fixed_gru[seed]
                    continue
                matched_gru[seed] = _ensure_gru(
                    output_dir=output_dir,
                    records=records,
                    train_cap=50.0,
                    seed=seed,
                    train_raw=parts["train"]["raw"],
                    select_clean=select_clean,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    predict_batch=args.predict_batch,
                    device=args.device,
                    wandb_publisher=publisher,
                    protocol="matched_retrain",
                )
                cpred = np.asarray(
                    _predict(matched_gru[seed]["model"], calib_parts["clean"], batch=args.predict_batch, device=args.device),
                    dtype=float,
                )
                cy = np.asarray([float(c.margin) for c in calib_parts["clean"]], dtype=float)
                if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
                    raise ValueError("matched cap-50 calibration predictions are invalid")
                matched_gru[seed]["cqr"] = {
                    "adjustment": float(fit_cqr_adjustment(cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)),
                    "cap": 50.0,
                    "calib_n": int(len(cy)),
                    "calib_designs": int(len(set(str(c.design_id) for c in calib_parts["clean"]))),
                }
            if any(
                item[1] == "ridge" and item[3] == "matched_retrain" for item in cap_pending
            ):
                if fixed_ridge is not None:
                    matched_ridge = fixed_ridge
                else:
                    matched_ridge = _ensure_ridge(
                        output_dir=output_dir,
                        records=records,
                        train_cap=50.0,
                        train_raw=parts["train"]["raw"],
                    )
                    matched_ridge["calibration"] = _ridge_adjustment(
                        matched_ridge["model"], calib_parts["raw"]
                    )
                    matched_ridge["calibration"]["cap"] = 50.0
        else:
            if need_matched_gru:
                for seed in seeds:
                    key = _record_key(MODEL_NAME, cap, "matched_retrain", seed)
                    if key not in cap_keys:
                        continue
                    matched_gru[seed] = _ensure_gru(
                        output_dir=output_dir,
                        records=records,
                        train_cap=cap,
                        seed=seed,
                        train_raw=parts["train"]["raw"],
                        select_clean=select_clean,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        predict_batch=args.predict_batch,
                        device=args.device,
                        wandb_publisher=publisher,
                        protocol="matched_retrain",
                    )
                    cpred = np.asarray(
                        _predict(
                            matched_gru[seed]["model"],
                            calib_parts["clean"],
                            device=args.device,
                            batch=args.predict_batch,
                        ),
                        dtype=float,
                    )
                    cy = np.asarray(
                        [float(c.margin) for c in calib_parts["clean"]], dtype=float
                    )
                    if cpred.shape != (len(cy), 3) or not np.all(np.isfinite(cpred)):
                        raise ValueError(f"matched cap-{cap:g} calibration predictions are invalid")
                    matched_gru[seed]["cqr"] = {
                        "adjustment": float(
                            fit_cqr_adjustment(cpred[:, 0], cpred[:, 2], cy, alpha=ALPHA)
                        ),
                        "cap": float(cap),
                        "calib_n": int(len(cy)),
                        "calib_designs": int(
                            len(set(str(c.design_id) for c in calib_parts["clean"]))
                        ),
                    }
            if need_matched_ridge:
                matched_ridge = _ensure_ridge(
                    output_dir=output_dir,
                    records=records,
                    train_cap=cap,
                    train_raw=parts["train"]["raw"],
                )
                matched_ridge["calibration"] = _ridge_adjustment(
                    matched_ridge["model"], calib_parts["raw"]
                )
                matched_ridge["calibration"]["cap"] = float(cap)

        # Evaluate only incomplete entries.  Every call persists both the
        # prediction audit and its JSON record atomically.
        for key, kind, _, mode, seed in cap_pending:
            if _record_is_complete(records.get(key), output_dir):
                continue
            if kind == "ridge":
                artifact = fixed_ridge if mode == "fixed_transfer" else matched_ridge
                if artifact is None:
                    raise RuntimeError(f"ridge artifact unavailable for {key}")
                _ridge_record(
                    key=key,
                    results=results,
                    output_dir=output_dir,
                    artifact=artifact,
                    parts=parts,
                    cap=cap,
                    mode=mode,
                    calibration=artifact.get("calibration", {}),
                    calibration_parts=calib_parts,
                    args=args,
                )
                continue
            artifact = (
                fixed_gru.get(int(seed))
                if mode == "fixed_transfer"
                else matched_gru.get(int(seed))
            )
            if artifact is None:
                raise RuntimeError(f"GRU artifact unavailable for {key}")
            _neural_record(
                key=key,
                results=results,
                output_dir=output_dir,
                artifact=artifact,
                parts=parts,
                cap=cap,
                mode=mode,
                seed=int(seed),
                calibration=artifact.get("cqr", {}),
                calibration_parts=calib_parts,
                args=args,
            )

        # Release cap-specific tensors before constructing the next cap.  Fixed
        # model objects are intentionally retained; they are small state dicts.
        del matched_gru, matched_ridge, parts, calib_parts

    results["comparisons"] = _build_comparisons(results, output_dir, caps, seeds, args.bootstrap)
    _atomic_json_dump(results, results_path)
    if publisher is not None:
        publisher.publish_results(results, results_path)
    print(f"SATURATION_DONE records={len(results['records'])} output={output_dir}", flush=True)
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    project = getattr(args, "wandb_project", None)
    if project is None:
        return _run_experiment(args)
    if not isinstance(project, str) or not project.strip():
        raise ValueError("--wandb-project must be a nonempty project name")
    publisher = _WandbArtifactPublisher(project.strip())
    args._wandb_publisher = publisher
    try:
        results = _run_experiment(args)
    except BaseException:
        try:
            publisher.finish(exit_code=1)
        except Exception:
            pass
        raise
    publisher.finish(exit_code=0)
    return results

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4 pre-saturation sensor-cap sensitivity")
    parser.add_argument(
        "--dataset",
        "--dataset-dir",
        dest="dataset",
        default=os.environ.get("P4_DATASET_DIR", "p4_dataset"),
        help="dataset directory",
    )
    parser.add_argument("--output-dir", "--out", dest="output_dir", default="p4_saturation_run", help="artifact directory")
    parser.add_argument("--caps", nargs="+", type=float, default=list(DEFAULT_CAPS), help="caps; must include 50")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="neural model seeds")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="neural epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="neural training batch size")
    parser.add_argument("--predict-batch", type=int, default=DEFAULT_PREDICT_BATCH, help="prediction batch size (default <=32)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="bounded torch/BLAS thread count")
    parser.add_argument("--device", default="cpu", help="neural training/prediction device (default: cpu)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="paired-design bootstrap resamples")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="require online W&B artifact persistence in this project (unset for local-only runs)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
