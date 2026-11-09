"""Measured raw-waveform inputs for the fixed P4 raw-vs-modal experiment.

This module deliberately does only two things beyond loading the frozen modal
cache: it stages the measured, normalized sensor traces and exposes causal
384-sample windows.  In particular, acquisition does not estimate modes,
inspect labels, or run a predictor.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
SOURCE = SCRIPT / "support_modal_20261009_v2"
OUT = SCRIPT / "raw_modal_20261010"
DATASET = SOURCE / "dataset.npz"
PROTOCOL = SOURCE / "protocol.json"

# The source module is the frozen owner of noise generation and calibration.
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(SCRIPT)]
import experiment_support_modal_v2 as E  # noqa: E402
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling  # noqa: E402


SENSORS = 16
SAMPLES = 8001
WINDOW = 384
FRAME_FIRST_END = 128
FRAME_STRIDE = 64
FRAME_COUNT = 124
LAGS = (4, 2, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_manifest() -> tuple[list[dict], np.ndarray, str, int]:
    """Read only the row manifest needed by acquisition, never labels."""
    if not DATASET.exists():
        raise FileNotFoundError(f"missing frozen modal dataset: {DATASET}")
    with np.load(DATASET, allow_pickle=False) as archive:
        required = {"records", "times"}
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"modal dataset missing manifest fields: {sorted(missing)}")
        records_text = str(archive["records"])
        records = json.loads(records_text)
        times = np.asarray(archive["times"]).copy()
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise TypeError("dataset records must be a JSON list of objects")
    if times.ndim != 1 or times.shape != (FRAME_COUNT,):
        raise RuntimeError(f"unexpected modal frame grid: {times.shape}")
    expected_ends = FRAME_FIRST_END + FRAME_STRIDE * np.arange(FRAME_COUNT)
    ends = np.rint(times * 1000.0).astype(np.int64) + 1
    if not np.array_equal(ends, expected_ends):
        raise RuntimeError("modal dataset frame grid does not match the fixed raw aperture")
    for index, record in enumerate(records):
        for key in ("design", "replicate", "profile", "noise"):
            if key not in record:
                raise RuntimeError(f"dataset record {index} lacks {key!r}")
    return records, times.astype(np.float64, copy=False), records_text, len(records)


def _source_designs(records: Iterable[dict]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for record in records:
        design = str(record["design"])
        if design not in seen:
            seen.add(design)
            result.append(design)
    if not result:
        raise RuntimeError("modal dataset has no measured rows")
    return result


def _source_manifest(designs: Iterable[str]) -> dict[str, dict]:
    """Fingerprint all required remote-host shards before writing outputs."""
    result: dict[str, dict] = {}
    for design in designs:
        path = E.SHARDS / f"{design}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing measured source shard for {design}: {path}")
        result[design] = {
            "path": _relative(path),
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
    return result


def _base_provenance(records: list[dict], times: np.ndarray, records_text: str, source_shards: dict[str, dict]) -> dict:
    return {
        "phase": "acquire",
        "complete": False,
        "implementation_sha256": _sha256(Path(__file__)),
        "dataset_sha256": _sha256(DATASET),
        "protocol_sha256": _sha256(PROTOCOL),
        "source_dataset": _relative(DATASET),
        "source_protocol": _relative(PROTOCOL),
        "source_shards": source_shards,
        "normalization_sources": {
            _relative(Path(E.__file__)): _sha256(Path(E.__file__)),
            _relative(Path(E.M.__file__)): _sha256(Path(E.M.__file__)),
        },
        "rows": int(len(records)),
        "frames": int(len(times)),
        "samples": SAMPLES,
        "sensors": SENSORS,
        "waveforms_shape": [int(len(records)), SENSORS, SAMPLES],
        "waveforms_dtype": "float32",
        "admissible_shape": [int(len(records)), SAMPLES],
        "admissible_dtype": "bool",
        "acquisition_rows": [True] * len(records),
        "row_order": {
            "source": "support_modal_20261009_v2/dataset.npz records",
            "key": ["design", "replicate", "profile", "noise"],
            "global_index_preserved": True,
            "record_manifest_sha256": hashlib.sha256(records_text.encode("utf-8")).hexdigest(),
            "frame_ends_samples": (FRAME_FIRST_END + FRAME_STRIDE * np.arange(FRAME_COUNT)).tolist(),
            "frame_times_seconds": times.tolist(),
        },
        "normalization": {
            "noise_helper": "experiment_support_modal_v2.noise_for",
            "calibration_helper": "experiment_support_modal_v2.M.normalize",
            "leading_calibration_samples": 51,
            "cast": True,
            "float32_calibrated_signal": True,
            "clip_bounds": [-50.0, 50.0],
            "waveform_whitening": False,
            "leading_calibration_unchanged": True,
        },
        "admissibility": {
            "definition": "not command and all 16 calibrated float32 values finite and abs(value) < 50",
            "command_excluded": True,
            "all_channels_finite": True,
            "all_channels_strictly_below_absolute_cap": True,
            "global_first_cap_latch": False,
            "interpolation": False,
            "warmup_samples_in_source": False,
        },
        "scope": "Measured acquisition only; no labels, modal calls, eigensystem calls, prediction, or scoring",
    }


def _read_acquisition_json() -> dict | None:
    path = OUT / "acquisition.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _validate_shape_dtype(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if tuple(array.shape) != tuple(shape) or np.dtype(array.dtype) != np.dtype(dtype):
            raise RuntimeError(f"staged artifact mismatch for {path}: {array.shape}/{array.dtype}")
    finally:
        del array


def _metadata_matches(meta: dict, base: dict, verify_artifact_hashes: bool) -> bool:
    if meta.get("complete") is not True:
        return False
    for key in (
        "implementation_sha256",
        "dataset_sha256",
        "protocol_sha256",
        "source_shards",
        "normalization_sources",
        "rows",
        "frames",
        "samples",
        "sensors",
        "waveforms_shape",
        "waveforms_dtype",
        "admissible_shape",
        "admissible_dtype",
        "row_order",
        "normalization",
        "admissibility",
        "acquisition_rows",
    ):
        if key == "source_shards" and not base.get("source_shards"):
            continue
        if meta.get(key) != base.get(key):
            return False
    try:
        rows = int(base["rows"])
        _validate_shape_dtype(OUT / "waveforms.npy", (rows, SENSORS, SAMPLES), np.dtype("float32"))
        _validate_shape_dtype(OUT / "admissible.npy", (rows, SAMPLES), np.dtype("bool"))
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        return False
    if verify_artifact_hashes:
        for name in ("waveforms", "admissible"):
            path = OUT / f"{name}.npy"
            expected = meta.get(f"{name}_sha256")
            if not isinstance(expected, str) or _sha256(path) != expected:
                return False
    return True


def _open_measured(validate_hashes: bool = True) -> tuple[np.memmap, np.memmap]:
    records, times, records_text, rows = _load_manifest()
    source_shards = _source_manifest(_source_designs(records)) if validate_hashes else None
    base = _base_provenance(records, times, records_text, source_shards or {})
    meta = _read_acquisition_json()
    if meta is None or not _metadata_matches(meta, base, verify_artifact_hashes=validate_hashes):
        raise RuntimeError("raw acquisition artifacts are absent, incomplete, or stale")
    waveforms = np.load(OUT / "waveforms.npy", mmap_mode="r", allow_pickle=False)
    admissible = np.load(OUT / "admissible.npy", mmap_mode="r", allow_pickle=False)
    if not isinstance(waveforms, np.memmap) or not isinstance(admissible, np.memmap):
        raise RuntimeError("raw acquisition artifacts are not NumPy memmaps")
    return waveforms, admissible


def load_data() -> dict:
    """Load the frozen modal feature cache without changing any cached values."""
    with np.load(DATASET, allow_pickle=False) as archive:
        required = ("features", "truth", "observed", "age", "times", "records")
        missing = set(required).difference(archive.files)
        if missing:
            raise RuntimeError(f"modal dataset missing fields: {sorted(missing)}")
        data = {name: archive[name].copy() for name in required[:-1]}
        records = json.loads(str(archive["records"]))
    if data["features"].dtype != np.float64 or data["features"].ndim != 3 or data["features"].shape[-1] != 174:
        raise RuntimeError(f"unexpected cached modal feature array: {data['features'].shape}/{data['features'].dtype}")
    n_rows, n_frames, _ = data["features"].shape
    if data["truth"].shape != (n_rows, n_frames) or data["truth"].dtype != np.float64:
        raise RuntimeError("cached truth shape/dtype changed")
    if data["observed"].shape != (n_rows, n_frames) or data["observed"].dtype != np.bool_:
        raise RuntimeError("cached observed shape/dtype changed")
    if data["age"].shape != (n_rows, n_frames) or data["age"].dtype != np.float64:
        raise RuntimeError("cached age shape/dtype changed")
    if data["times"].shape != (n_frames,) or data["times"].dtype != np.float64:
        raise RuntimeError("cached times shape/dtype changed")
    if not isinstance(records, list) or len(records) != n_rows:
        raise RuntimeError("cached records do not preserve row order")
    data["records"] = records
    return data


def acquire() -> tuple[np.memmap, np.memmap]:
    """Stage every measured row into atomic, normalized continuous memmaps."""
    records, times, records_text, rows = _load_manifest()
    designs = _source_designs(records)
    source_shards = _source_manifest(designs)
    base = _base_provenance(records, times, records_text, source_shards)
    existing = _read_acquisition_json()
    if existing is not None and _metadata_matches(existing, base, verify_artifact_hashes=True):
        return _open_measured(validate_hashes=False)

    OUT.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    wave_tmp = OUT / f".waveforms.{os.getpid()}.{token}.npy"
    admissible_tmp = OUT / f".admissible.{os.getpid()}.{token}.npy"
    waveforms = None
    admissible = None
    try:
        waveforms = np.lib.format.open_memmap(
            wave_tmp, mode="w+", dtype=np.float32, shape=(rows, SENSORS, SAMPLES), fortran_order=False
        )
        admissible = np.lib.format.open_memmap(
            admissible_tmp, mode="w+", dtype=np.bool_, shape=(rows, SAMPLES), fortran_order=False
        )
        waveforms[:] = 0.0
        admissible[:] = False
        calibration_offsets = np.empty((rows, SENSORS), dtype=np.float64)
        calibration_scales = np.empty_like(calibration_offsets)
        groups: dict[str, list[int]] = {}
        for row, record in enumerate(records):
            groups.setdefault(str(record["design"]), []).append(row)
        processed = np.zeros(rows, dtype=bool)
        for design in designs:
            shard_path = E.SHARDS / f"{design}.npz"
            row_indices = groups[design]
            with np.load(shard_path, allow_pickle=False) as archive:
                for key in ("raw", "command", "records"):
                    if key not in archive.files:
                        raise RuntimeError(f"measured shard {design} missing {key!r}")
                raw = archive["raw"]
                command = archive["command"]
                physical_records = json.loads(str(archive["records"]))
                if raw.ndim != 3 or raw.shape[1:] != (SENSORS, SAMPLES):
                    raise RuntimeError(f"unexpected raw shape in {shard_path}: {raw.shape}")
                if command.shape != (raw.shape[0], SAMPLES):
                    raise RuntimeError(f"unexpected command shape in {shard_path}: {command.shape}")
                lookup: dict[tuple[int, str], int] = {}
                for physical_index, physical_record in enumerate(physical_records):
                    key = (int(physical_record["replicate"]), str(physical_record["profile"]))
                    if key in lookup:
                        raise RuntimeError(f"duplicate physical record in {shard_path}: {key}")
                    lookup[key] = physical_index
                for row in row_indices:
                    record = records[row]
                    key = (int(record["replicate"]), str(record["profile"]))
                    if key not in lookup:
                        raise RuntimeError(f"row {row} {design} has no matching physical source record: {key}")
                    physical_index = lookup[key]
                    signal, offset, scale = E.M.normalize(
                        raw[physical_index], E.noise_for(raw[physical_index], record), cast=True
                    )
                    signal = np.asarray(signal, dtype=np.float32)
                    if signal.shape != (SENSORS, SAMPLES):
                        raise RuntimeError(f"normalized source shape changed for row {row}: {signal.shape}")
                    calibration_offsets[row] = offset
                    calibration_scales[row] = scale
                    command_row = np.asarray(command[physical_index], dtype=bool)
                    finite = np.isfinite(signal).all(axis=0)
                    uncapped = (np.abs(signal) < np.float32(50.0)).all(axis=0)
                    valid = (~command_row) & finite & uncapped
                    staged = np.array(signal, dtype=np.float32, order="C", copy=True)
                    # Avoid NaN propagation and make the explicit mask sufficient
                    # for every downstream consumer.
                    staged[:, ~valid] = np.float32(0.0)
                    waveforms[row] = staged
                    admissible[row] = valid
                    processed[row] = True
        if not np.all(processed):
            missing = np.flatnonzero(~processed)
            raise RuntimeError(f"source row mapping incomplete; first missing rows: {missing[:8].tolist()}")
        waveforms.flush()
        admissible.flush()
        del waveforms, admissible
        waveforms = None
        admissible = None
        wave_tmp.replace(OUT / "waveforms.npy")
        admissible_tmp.replace(OUT / "admissible.npy")

        metadata = dict(base)
        metadata["waveforms_sha256"] = _sha256(OUT / "waveforms.npy")
        metadata["admissible_sha256"] = _sha256(OUT / "admissible.npy")
        metadata["calibration_offsets"] = E.clean_json(calibration_offsets)
        metadata["calibration_scales"] = E.clean_json(calibration_scales)
        metadata["complete"] = True
        _json_atomic(OUT / "acquisition.json", metadata)
        return _open_measured(validate_hashes=False)
    finally:
        if waveforms is not None:
            waveforms.flush()
            del waveforms
        if admissible is not None:
            admissible.flush()
            del admissible
        for temporary in (wave_tmp, admissible_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load_measured() -> tuple[np.memmap, np.memmap]:
    """Return read-only normalized waveform and admissibility memmaps."""
    return _open_measured(validate_hashes=True)


def raw_windows(
    waveforms: np.ndarray,
    admissible: np.ndarray,
    row_indices: Iterable[int],
    frame_indices: Iterable[int],
    zero_waveform: bool = False,
) -> np.ndarray:
    """Build causal ``[B,384,18]`` raw inputs without future sample access."""
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if rows.shape != frames.shape:
        raise ValueError(f"row/frame vectors must align: {rows.shape} != {frames.shape}")
    if waveforms.ndim != 3 or waveforms.shape[1:] != (SENSORS, SAMPLES):
        raise ValueError(f"unexpected waveform array shape: {waveforms.shape}")
    if admissible.ndim != 2 or admissible.shape != (waveforms.shape[0], SAMPLES):
        raise ValueError(f"unexpected admissibility array shape: {admissible.shape}")
    if np.any(rows < 0) or np.any(rows >= waveforms.shape[0]):
        raise IndexError("raw window row index out of range")
    if np.any(frames < 0) or np.any(frames >= FRAME_COUNT):
        raise IndexError("raw window frame index out of range")
    batch = int(rows.size)
    result = np.zeros((batch, WINDOW, SENSORS + 2), dtype=np.float32)
    if batch == 0:
        result[..., 17] = np.arange(-383, 1, dtype=np.float32) * np.float32(0.001)
        return np.ascontiguousarray(result)

    ends = FRAME_FIRST_END + FRAME_STRIDE * frames
    starts = ends - WINDOW
    positions = starts[:, None] + np.arange(WINDOW, dtype=np.int64)[None, :]
    present = (positions >= 0) & (positions < SAMPLES)
    # Safe gather indices are only used for indexing.  ``present`` below
    # explicitly removes every synthetic warmup position from both data and
    # the admissibility channel.
    safe_positions = np.clip(positions, 0, SAMPLES - 1)
    sample_admissible = admissible[rows[:, None], safe_positions] & present
    channels = np.arange(SENSORS, dtype=np.int64)[None, :, None]
    values = np.asarray(waveforms[rows[:, None, None], channels, safe_positions[:, None, :]], dtype=np.float32)
    # Both arms use the same physical admissibility before waveform erasure.
    sample_admissible &= np.isfinite(values).all(axis=1)
    sample_admissible &= (np.abs(values) < np.float32(50.0)).all(axis=1)
    if not zero_waveform:
        for channel in range(SENSORS):
            values[:, channel, :][~sample_admissible] = np.float32(0.0)
        result[:, :, :SENSORS] = np.transpose(values, (0, 2, 1))
    result[:, :, SENSORS] = sample_admissible.astype(np.float32)
    result[:, :, SENSORS + 1] = np.arange(-383, 1, dtype=np.float32) * np.float32(0.001)
    return np.ascontiguousarray(result)


def _make_slots(current: np.ndarray) -> np.ndarray:
    slots = np.zeros((*current.shape[:2], 3, current.shape[-1]), dtype=np.float32)
    for index, lag in enumerate(LAGS):
        if lag:
            slots[:, lag:, index] = current[:, :-lag]
        else:
            slots[:, :, index] = current
    return slots


def modal_inputs(data: dict, proper: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Standardize modal content on fitting rows and form the fixed 3 slots."""
    q = np.asarray(data["features"])
    observed = np.asarray(data["observed"])
    age = np.asarray(data["age"])
    if q.ndim != 3 or q.shape[-1] != 174:
        raise ValueError(f"unexpected modal feature shape: {q.shape}")
    n_rows, n_frames, _ = q.shape
    if observed.shape != (n_rows, n_frames) or age.shape != (n_rows, n_frames):
        raise ValueError("modal observed/age arrays do not align with features")
    fit_rows = np.asarray(proper, dtype=bool).reshape(-1)
    if fit_rows.shape != (n_rows,):
        raise ValueError(f"proper must be a row mask of shape {(n_rows,)}")
    fit_values = q[fit_rows][observed[fit_rows]]
    if fit_values.ndim != 2 or fit_values.shape[0] == 0:
        raise ValueError("proper contains no observed modal feature values")
    mean, scale, active = fit_ridge_scaling(fit_values)
    content = (q - mean) / scale
    content[..., ~active] = 0.0
    content[~observed] = 0.0
    current = np.concatenate((content, observed[..., None], age[..., None]), axis=-1).astype(np.float32)
    return _make_slots(current), (mean, scale, active)


if __name__ == "__main__":
    # Acquisition is intentionally the only CLI operation.  The parent owns
    # any smoke/fit/scoring orchestration and runs it on the measured host.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("acquire",))
    args = parser.parse_args()
    if args.phase == "acquire":
        acquire()
