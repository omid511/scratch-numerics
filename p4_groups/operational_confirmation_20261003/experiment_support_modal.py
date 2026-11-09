"""Acquire and combine the matched-modal support experiment.

The measured arm is deliberately boring: original rows are replayed from the
frozen additive-spectral dataset, while additional designs are acquired with
the unchanged temporal-probe plate and causal gate.  The oracle arm is a
physical-description counterfactual only.  It replaces confidently matched
physical pole/shape descriptions in measured modal slots, then calls the
frozen feature assembly again; it never selects a true critical branch.

This module owns only acquisition/combination.  Fitting and verification are
kept in the sibling support-fit implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy.linalg import eig
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(SCRIPT)]

import diagnostic_matrix as M
import experiment_headroom_oracle as H
import gpu_execution as G
import pulse_aware_probe as P
import temporal_probe as T
import trajectory_physics as TP
from cuda_backend import estimate_batch
from mechanics.p4_margin_estimation.modal_features import estimate_modes as cpu_estimate_modes

OUT = SCRIPT / "support_modal_20261009"
PROTOCOL_PATH = OUT / "protocol.json"
SOURCE_RAW = SCRIPT / "fitter_margin_comparison_20261008" / "raw_signals.npz"
SOURCE_DATASET = SCRIPT / "additive_spectral_20261009" / "dataset.npz"
HEADROOM = SCRIPT / "headroom_oracle_20261008"
O1 = SCRIPT / "o1_corrected"
SHARDS = OUT / "shards"
OLD_PROFILES = ("cross", "stable", "retreat")
BASE_NOISE = ("clean", "iid1", "correlated1", "iid5", "correlated5")
FRESH_SEEDS = (20261020, 20261021)
MAX_MODES = 8


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(archive, key: str):
    return json.loads(str(archive[key]))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value


def save_json(path: Path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def save_npz_atomic(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def contract():
    """Validate the immutable protocol inputs before touching outputs."""
    p = json.loads(PROTOCOL_PATH.read_text())
    for name, expected in {**p["inputs"], **p["implementation"]}.items():
        actual_path = ROOT / name
        if sha(actual_path) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    return p


def protocol_sha() -> str:
    return sha(PROTOCOL_PATH)


def inputs():
    d, fp = G.inputs()
    return d, fp


def split_map(p):
    return {did: split for split, designs in p["split"].items() for did in designs}


def designated_additional(p):
    """Reproduce the pre-outcome shuffle encoded by the protocol list."""
    eligible = list(p["eligible_designs"])
    rng = np.random.default_rng(p["selection_seed"])
    rng.shuffle(eligible)
    selected = eligible[: len(p["additional_designs"])]
    if selected != list(p["additional_designs"]):
        raise RuntimeError("additional design designation is not the frozen seeded shuffle")
    split = split_map(p)
    forbidden = set(p["split"]["validation"]) | set(p["split"]["held"])
    if set(selected) & forbidden or set(selected) & set(p["split"]["train"]):
        raise RuntimeError("additional design designation escaped the TRAIN-only support pool")
    return selected


def evaluation_fold(p):
    result = {}
    for fold, designs in enumerate(p["evaluation_folds"]):
        for did in designs:
            if did in result:
                raise RuntimeError(f"duplicate evaluation-fold designation: {did}")
            result[did] = int(fold)
    if set(result) != set(p["additional_designs"]):
        raise RuntimeError("evaluation folds do not partition additional designs")
    return result


def _as_schedule(description):
    return {
        "knots": np.asarray(description["knots"], dtype=float).tolist(),
        "ratios": np.asarray(description["ratios"], dtype=float).tolist(),
        "onsets": np.asarray(description["onsets"], dtype=float).tolist(),
        "amplitudes": np.asarray(description["amplitudes"], dtype=float).tolist(),
    }


def extended_schedule(design: str, replicate: int, profile: str, p):
    """Temporal-probe schedule plus the two preregistered sign-side regimes."""
    if profile in OLD_PROFILES:
        return _as_schedule(T.schedule(design, replicate, profile))
    if profile not in ("farther_stable", "unstable_exclusion"):
        raise ValueError(profile)
    index = list(p["profiles"]).index(profile)
    rng = np.random.default_rng(
        np.random.SeedSequence([20261009, int(design[1:]), replicate, index])
    )
    if profile == "farther_stable":
        initial = float(rng.uniform(0.78, 0.84))
        start = float(rng.uniform(1.0, 5.0))
        finish = min(start + float(rng.uniform(1.0, 2.0)), 6.8)
        end0 = float(rng.uniform(0.78, 0.84))
        end1 = float(rng.uniform(0.78, 0.84))
    else:
        initial = float(rng.uniform(1.015, 1.03))
        start = float(rng.uniform(1.0, 5.0))
        finish = min(start + float(rng.uniform(1.0, 2.0)), 6.8)
        end0 = float(rng.uniform(1.015, 1.03))
        end1 = float(rng.uniform(1.015, 1.03))
    onsets = 0.6 + np.arange(5) * 1.4 + rng.uniform(0.0, 0.65, 5)
    amplitudes = rng.uniform(0.5, 1.5, 5)
    return {
        "knots": [0.0, start, finish, 8.0],
        "ratios": [initial, initial, end0, end1],
        "onsets": onsets.tolist(),
        "amplitudes": amplitudes.tolist(),
    }


def noise_for(raw: np.ndarray, record: dict) -> np.ndarray:
    """Use the frozen diagnostic noise helper and fresh-draw seed convention."""
    name = record["noise"]
    did = int(record["design"][1:])
    profile = record["profile"]
    profile_index = list(("cross", "stable", "retreat", "farther_stable", "unstable_exclusion")).index(profile)
    if name in BASE_NOISE:
        seed = np.random.SeedSequence([20261010, did, int(record["replicate"]), profile_index])
        return M.noises(raw, seed)[0][name]
    for fresh_seed in FRESH_SEEDS:
        suffix = f"_fresh{fresh_seed}"
        if name.endswith(suffix):
            family = name[: -len(suffix)]
            if family not in ("iid1", "correlated1", "iid5", "correlated5"):
                break
            seed = np.random.SeedSequence([fresh_seed, did, int(record["replicate"]), profile_index])
            return M.noises(raw, seed)[0][family]
    raise ValueError(f"unknown support noise: {name}")


def allowed_noise(split: str, p):
    if split in ("train", "validation"):
        return tuple(BASE_NOISE)
    if split == "held":
        return tuple(p["noise"])
    if split == "additional_train":
        return tuple(p["noise"])
    raise ValueError(split)


def source_records():
    with np.load(SOURCE_RAW, allow_pickle=False) as z:
        raw = z["physical"].copy()
        command = z["command"].copy()
        records = scalar(z, "records")
    return raw, command, records


def canonical_data():
    with np.load(SOURCE_DATASET, allow_pickle=False) as z:
        data = {
            key: z[key].copy()
            for key in ("features", "truth", "observed", "age", "times")
        }
        data["records"] = scalar(z, "records")
    return data


def _identity_from_archive(z):
    return scalar(z, "identity") if "identity" in z else None


def _load_o1_context(d, fp, did):
    identity, plate = M.context(d, fp, {"design": did, "replicate": 0})
    audit = json.loads((O1 / f"audit_{did}.json").read_text())
    if not audit.get("acquisition_supported", False):
        raise RuntimeError(f"selected support design is not audited for acquisition: {did}")
    if audit.get("fingerprint") != fp:
        raise RuntimeError(f"stale O1 audit for support design: {did}")
    return identity, plate, audit


def _positive_modes(matrix, plate):
    roots, vectors = eig(matrix, check_finite=False)
    count = matrix.shape[0] // 2
    q = plate.basis[:, :count] @ (vectors[:count] / plate.omega[:count, None])
    participation = TP._transverse_participation(q, len(plate.omega))
    visible = (roots.imag > TP.TRANSIENT_FILTER.omega_min) & (
        participation >= TP.TRANSIENT_FILTER.eta_w_min
    )
    return roots, vectors, np.flatnonzero(visible)


def _critical_pole(roots, visible):
    if not len(visible):
        raise RuntimeError("no physically transverse positive-frequency mode")
    return roots[visible[np.argmax(roots[visible].real)]]


def validate_plate(plate, p, archived_audit):
    """Validate .78/.84/1.015/1.03 prerequisites without weakening O1 audits."""
    physical = p["physical_validation"]
    ratios = (0.78, 0.84, 0.9, 1.0, 1.015, 1.03)
    minimum_mach = min(r * plate.ucrit / plate.design.c_air for r in ratios)
    if minimum_mach < physical["minimum_mach"]:
        raise RuntimeError(f"{plate.design.design_id}: Mach domain fails at .78 ({minimum_mach})")
    if not archived_audit.get("acquisition_supported", False):
        raise RuntimeError(f"{plate.design.design_id}: existing reduced/full audit is unsupported")
    if not archived_audit.get("bandwidth_supported", False):
        raise RuntimeError(f"{plate.design.design_id}: existing retained-8 bandwidth audit is unsupported")
    extreme = []
    for ratio in ratios:
        full_roots, full_vectors, full_visible = _positive_modes(
            plate.matrix(ratio, len(plate.omega)), plate
        )
        reduced_roots, _, reduced_visible = _positive_modes(plate.matrix(ratio), plate)
        full_critical = _critical_pole(full_roots, full_visible)
        reduced_critical = _critical_pole(reduced_roots, reduced_visible)
        pole_gap = float(abs(reduced_critical - full_critical) / max(abs(full_critical), 1e-30))
        if pole_gap > physical["maximum_critical_relative_pole_gap"]:
            raise RuntimeError(f"{plate.design.design_id}: reduced/full pole gap fails at {ratio}: {pole_gap}")
        # The projected positive-frequency eigenbasis is the one used by matching.
        condition = float(np.linalg.cond(full_vectors[:, full_visible]))
        if not np.isfinite(condition) or condition > p["correspondence"]["maximum_eigenbasis_condition"]:
            raise RuntimeError(f"{plate.design.design_id}: eigenbasis condition fails at {ratio}: {condition}")
        ordered = full_roots[full_visible][np.argsort(full_roots[full_visible].imag)]
        retained8_max = float(np.max(ordered[:8].imag) / (2 * np.pi))
        if retained8_max > physical["maximum_retained8_hz"]:
            raise RuntimeError(f"{plate.design.design_id}: retained-8 bandwidth fails at {ratio}: {retained8_max}")
        # A short, projected response check catches a ROM that happens to match
        # poles but not the actual measured channels.
        times = np.arange(121, dtype=float) * 0.001
        reduced_initial, full_initial = plate.initial(0)
        full = TP.propagate_state(
            times,
            full_initial,
            lambda _t, value=ratio: plate.matrix(value, len(plate.omega)),
            plate.full["projection"],
            coefficient_step=0.001,
            input_vector=plate.full["force"],
        )
        reduced_coarse = TP.propagate_state(
            times,
            reduced_initial,
            lambda _t, value=ratio: plate.matrix(value),
            plate.reduced["projection"],
            coefficient_step=0.002,
            input_vector=plate.reduced["force"],
        )
        reduced_fine = TP.propagate_state(
            times,
            reduced_initial,
            lambda _t, value=ratio: plate.matrix(value),
            plate.reduced["projection"],
            coefficient_step=0.001,
            input_vector=plate.reduced["force"],
        )
        unit = max(float(np.sqrt(np.mean(full.response**2))), 1e-30)
        full_response = float(np.sqrt(np.mean((reduced_coarse.response - full.response) ** 2)) / unit)
        step_response = float(np.sqrt(np.mean((reduced_coarse.response - reduced_fine.response) ** 2)) / unit)
        if full_response > physical["maximum_full_response_relative_rms"]:
            raise RuntimeError(f"{plate.design.design_id}: full response fails at {ratio}: {full_response}")
        if step_response > physical["maximum_step_halving_relative_rms"]:
            raise RuntimeError(f"{plate.design.design_id}: step halving fails at {ratio}: {step_response}")
        extreme.append(
            dict(
                ratio=ratio,
                minimum_mach=float(ratio * plate.ucrit / plate.design.c_air),
                retained8_max_frequency_hz=retained8_max,
                critical_relative_pole_gap=pole_gap,
                eigenbasis_condition=condition,
                full_response_relative_rms=full_response,
                step_halving_relative_rms=step_response,
            )
        )
    return dict(
        minimum_mach=float(minimum_mach),
        archived_audit=archived_audit,
        extreme=extreme,
        acquisition_supported=True,
    )


def _source_physical_for_design(did, raw, records):
    rows = [i for i, record in enumerate(records) if record["design"] == did]
    if len(rows) != 6:
        raise RuntimeError(f"expected six original physical trajectories for {did}, found {len(rows)}")
    return rows


def _state_archive_from_rows(identity, plate, records, states, replay_errors=None):
    matrices = plate.reduced
    n = len(states)
    return {
        "states": np.asarray(states, dtype=float),
        "records": np.array(json.dumps(clean_json(records))),
        "identity": np.array(json.dumps(clean_json(identity))),
        "base": matrices["base"],
        "stiffness": matrices["stiffness"],
        "damping": matrices["damping"],
        "projection": matrices["projection"],
        "physical_basis": plate.basis[:, : plate.count] / plate.omega[: plate.count],
        "structural_dofs": np.array(len(plate.omega)),
        "replay_errors": np.zeros(n) if replay_errors is None else np.asarray(replay_errors),
        "protocol_sha256": np.array(protocol_sha()),
    }


def _existing_shard_matches(path: Path, did: str, p):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        if str(z["protocol_sha256"]) != protocol_sha() or str(z["design"]) != did:
            raise RuntimeError(f"immutable support shard has stale provenance: {path}")
    return True


def acquire_original(did, p, d, fp):
    path = SHARDS / f"{did}.npz"
    if _existing_shard_matches(path, did, p):
        print("SUPPORT_SHARD_REUSED", did, flush=True)
        return path
    raw, command, records = source_records()
    indices = _source_physical_for_design(did, raw, records)
    _, plate, audit = _load_o1_context(d, fp, did)
    validation = validate_plate(plate, p, audit)
    state_path = HEADROOM / f"states_{did}.npz"
    if not state_path.exists():
        raise RuntimeError(f"missing frozen original state replay: {state_path}")
    with np.load(state_path, allow_pickle=False) as z:
        state_records = scalar(z, "records")
        state_rows = z["states"].copy()
        identity = scalar(z, "identity")
        replay_errors = z["replay_errors"].copy()
    state_map = {(r["replicate"], r["profile"]): i for i, r in enumerate(state_records)}
    physical_records = []
    states = []
    raw_rows = []
    command_rows = []
    for source_index in indices:
        record = copy.deepcopy(records[source_index])
        record["source_index"] = int(source_index)
        raw_rows.append(raw[source_index])
        command_rows.append(command[source_index])
        state_index = state_map[(record["replicate"], record["profile"])]
        states.append(state_rows[state_index])
        physical_records.append(record)
    observer = copy.copy(plate)
    observer.reduced = dict(plate.reduced)
    observer.reduced["projection"] = np.vstack((plate.reduced["projection"], np.eye(2 * plate.count)))
    for replicate in range(p["schedule_replicates"]):
        for profile in ("farther_stable", "unstable_exclusion"):
            description = extended_schedule(did, replicate, profile, p)
            _, response, command_activity = T.acquire(observer, description, replicate)
            raw_rows.append(response[:len(plate.sensor)])
            states.append(response[len(plate.sensor):].T)
            command_rows.append(command_activity.astype(bool))
            physical_records.append(dict(design=did, split=split_map(p)[did], replicate=replicate,
                profile=profile, schedule=description, source_kind="added_regime"))
    replay_errors = np.r_[replay_errors, np.zeros(4)]
    archive = _state_archive_from_rows(identity, plate, physical_records, np.stack(states), replay_errors)
    save_npz_atomic(
        path,
        design=np.array(did),
        split=np.array(split_map(p)[did]),
        source_kind=np.array("original"),
        raw=np.stack(raw_rows),
        command=np.stack(command_rows),
        states=archive["states"],
        records=archive["records"],
        identity=archive["identity"],
        base=archive["base"],
        stiffness=archive["stiffness"],
        damping=archive["damping"],
        projection=archive["projection"],
        physical_basis=archive["physical_basis"],
        structural_dofs=archive["structural_dofs"],
        replay_errors=archive["replay_errors"],
        validation=json.dumps(validation),
        audit=json.dumps(audit),
        protocol_sha256=np.array(protocol_sha()),
        input_fingerprint=json.dumps(fp),
    )
    print("SUPPORT_SHARD_ACQUIRED", did, "original", len(physical_records), flush=True)
    return path


def acquire_additional(did, p, d, fp):
    path = SHARDS / f"{did}.npz"
    if _existing_shard_matches(path, did, p):
        print("SUPPORT_SHARD_REUSED", did, flush=True)
        return path
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("additional support acquisition requires CUDA; CPU propagation is forbidden")
    identity, plate, audit = _load_o1_context(d, fp, did)
    validation = validate_plate(plate, p, audit)
    observer = copy.copy(plate)
    observer.reduced = dict(plate.reduced)
    observer.reduced["projection"] = np.vstack((plate.reduced["projection"], np.eye(2 * plate.count)))
    raw_rows, command_rows, state_rows, physical_records = [], [], [], []
    for replicate in range(p["schedule_replicates"]):
        for profile in p["profiles"]:
            description = extended_schedule(did, replicate, profile, p)
            times, response, command_activity = T.acquire(observer, description, replicate)
            if response.shape != (len(plate.sensor) + 2 * plate.count, len(times)):
                raise RuntimeError(f"unexpected augmented acquisition shape for {did}/{profile}/{replicate}")
            if not np.isfinite(response).all() or not np.isfinite(command_activity).all():
                raise RuntimeError(f"nonfinite additional trajectory for {did}/{profile}/{replicate}")
            raw_rows.append(response[: len(plate.sensor)])
            state_rows.append(response[len(plate.sensor) :].T)
            command_rows.append(command_activity.astype(bool))
            physical_records.append(
                dict(
                    design=did,
                    split="train",
                    replicate=int(replicate),
                    profile=profile,
                    schedule=description,
                    source_kind="additional",
                )
            )
            print("SUPPORT_TRAJECTORY_ACQUIRED", did, replicate, profile, flush=True)
    archive = _state_archive_from_rows(identity, plate, physical_records, np.stack(state_rows))
    save_npz_atomic(
        path,
        design=np.array(did),
        split=np.array("train"),
        source_kind=np.array("additional"),
        raw=np.stack(raw_rows),
        command=np.stack(command_rows),
        states=archive["states"],
        records=archive["records"],
        identity=archive["identity"],
        base=archive["base"],
        stiffness=archive["stiffness"],
        damping=archive["damping"],
        projection=archive["projection"],
        physical_basis=archive["physical_basis"],
        structural_dofs=archive["structural_dofs"],
        replay_errors=archive["replay_errors"],
        validation=json.dumps(validation),
        audit=json.dumps(audit),
        protocol_sha256=np.array(protocol_sha()),
        input_fingerprint=json.dumps(fp),
    )
    print("SUPPORT_SHARD_ACQUIRED", did, "additional", len(physical_records), flush=True)
    return path


def acquire_design(did, p, d, fp):
    original = set(p["split"]["train"]) | set(p["split"]["validation"]) | set(p["split"]["held"])
    additional = designated_additional(p)
    if did in original:
        return acquire_original(did, p, d, fp)
    if did in additional:
        return acquire_additional(did, p, d, fp)
    raise ValueError(f"design is outside frozen support set: {did}")


def acquire_phase(p, requested):
    d, fp = inputs()
    all_designs = list(p["split"]["train"]) + list(p["split"]["validation"]) + list(p["split"]["held"]) + designated_additional(p)
    targets = all_designs if not requested else requested
    if len(set(targets)) != len(targets) or any(did not in all_designs for did in targets):
        raise ValueError("--design must name unique frozen original/additional designs")
    for did in targets:
        acquire_design(did, p, d, fp)
    save_json(
        OUT / "acquisition.json",
        dict(
            phase="acquire",
            designs=targets,
            original_designs=p["split"],
            additional_designs=p["additional_designs"],
            protocol_sha256=protocol_sha(),
            implementation_sha256=sha(Path(__file__)),
            source_hashes={"remote_input.npz": sha(SCRIPT / "remote_input.npz"), "source_raw": sha(SOURCE_RAW), "source_dataset": sha(SOURCE_DATASET)},
        ),
    )


def _load_shard(did):
    path = SHARDS / f"{did}.npz"
    if not path.exists():
        raise RuntimeError(f"missing immutable support shard: {path}")
    with np.load(path, allow_pickle=False) as z:
        if str(z["protocol_sha256"]) != protocol_sha() or str(z["design"]) != did:
            raise RuntimeError(f"support shard provenance mismatch: {path}")
        result = {key: z[key].copy() for key in z.files}
    result["records"] = json.loads(str(result["records"]))
    result["identity"] = json.loads(str(result["identity"]))
    result["validation"] = json.loads(str(result["validation"]))
    result["audit"] = json.loads(str(result["audit"]))
    result["input_fingerprint"] = json.loads(str(result["input_fingerprint"]))
    return result


def _archive_from_shard(shard):
    # H.pole_table consumes the same scalar-JSON archive convention as the
    # frozen headroom implementation.  Shards expose parsed dictionaries for
    # ordinary callers, so re-encode these two fields at this boundary.
    return {
        "states": shard["states"],
        "records": np.array(json.dumps(clean_json(shard["records"]))),
        "identity": np.array(json.dumps(clean_json(shard["identity"]))),
        "base": shard["base"],
        "stiffness": shard["stiffness"],
        "damping": shard["damping"],
        "projection": shard["projection"],
        "physical_basis": shard["physical_basis"],
        "structural_dofs": shard["structural_dofs"],
        "replay_errors": shard["replay_errors"],
    }


def _physical_table(shard, physical_index, schedule, p):
    archive = _archive_from_shard(shard)
    state_records = shard["records"]
    record = state_records[physical_index]
    ratios = np.interp(
        np.arange(shard["raw"].shape[-1], dtype=float) * 0.001,
        np.asarray(schedule["knots"], dtype=float),
        np.asarray(schedule["ratios"], dtype=float),
    )
    archive["states"] = shard["states"][physical_index : physical_index + 1]
    archive["records"] = np.array(json.dumps([record]))
    table = H.pole_table(archive, ratios[None], p)
    count = archive["projection"].shape[1] // 2
    eligibility = np.zeros(table["poles"].shape, dtype=bool)
    conditioning = np.empty(len(table["grid"]))
    for grid_index, ratio in enumerate(table["grid"]):
        identity = shard["identity"]
        velocity = ratio * identity["ucrit"]
        mach = velocity / identity["design"]["c_air"]
        dynamic = identity["design"]["rho_air"] * velocity**2 / np.sqrt(mach**2 - 1)
        damping = dynamic * (mach**2 - 2) / (mach**2 - 1) / velocity
        matrix = archive["base"] + dynamic * archive["stiffness"] + damping * archive["damping"]
        roots, vectors = eig(matrix, check_finite=False)
        positive = np.flatnonzero(roots.imag > 1e-8)
        from scipy.optimize import linear_sum_assignment
        rows, columns = linear_sum_assignment(abs(table["poles"][grid_index, :, None] - roots[positive][None, :]))
        if not np.array_equal(rows, np.arange(count)):
            raise RuntimeError("Incomplete tracked eigenbasis reconstruction")
        selected = positive[columns]
        participation = TP._transverse_participation(archive["physical_basis"] @ vectors[:count, selected],
            int(archive["structural_dofs"]))
        eligibility[grid_index] = (roots[selected].imag > TP.TRANSIENT_FILTER.omega_min) & (participation >= TP.TRANSIENT_FILTER.eta_w_min)
        conditioning[grid_index] = np.linalg.cond(vectors)
    table["eligible"] = eligibility
    table["eigenbasis_condition"] = conditioning
    return table



def _row_modes(signal, frames, use_gpu=True):
    """Estimate modes only for the supplied causal windows, preserving order."""
    if not frames:
        return {}
    windows = [signal[:, end - 128 : end] for end in frames]
    if use_gpu:
        modes = estimate_batch(windows)
    else:
        modes = [cpu_estimate_modes(window, 0.001, include_shapes=True) for window in windows]
    return {int(frame): mode for frame, mode in zip(frames, modes)}


def _oracle_match(mode, table, scale, start, end, p, plate, condition_cache):
    count = len(mode["alpha"])
    branches = np.full(count, -1, dtype=np.int16)
    best_second = np.full((count, 2), np.nan, dtype=float)
    reasons = np.full(count, "unresolved", dtype="<U40")
    condition = np.full(count, np.nan, dtype=float)
    sensor_reconstruction = np.full(count, np.nan, dtype=float)
    if not count:
        return branches, best_second, reasons, condition, sensor_reconstruction
    indices = table["indices"][0, start:end]
    if not len(indices):
        reasons[:] = "empty_window"
        return branches, best_second, reasons, condition, sensor_reconstruction
    lo, hi = int(indices.min()), int(indices.max())
    center = int(indices[(len(indices) - 1) // 2])
    quality = table["quality"][lo + 1 : hi + 1]
    poles = table["poles"][center]
    eligible = table["eligible"][center].copy()
    basis_condition = float(np.max(table["eigenbasis_condition"][lo:hi + 1]))
    condition[:] = basis_condition
    if not np.isfinite(basis_condition) or basis_condition > p["correspondence"]["maximum_eigenbasis_condition"]:
        reasons[:] = "condition_fail"
        return branches, best_second, reasons, condition, sensor_reconstruction
    measured_shapes = np.asarray(mode.get("mode_shapes", np.empty((0, len(scale)))), dtype=np.complex128)
    physical_shapes = table["shapes"][center] / scale[:, None]
    measured_norm = np.linalg.norm(measured_shapes, axis=1)
    physical_norm = np.linalg.norm(physical_shapes, axis=0)
    frequencies = table["poles"][indices].imag / (2 * np.pi)
    overlaps = np.zeros((count, physical_shapes.shape[1]), dtype=float)
    distances = np.full_like(overlaps, np.inf)
    for i in range(count):
        if measured_shapes.shape[0] <= i or measured_norm[i] <= 0:
            continue
        numerator = np.abs(np.conj(measured_shapes[i]) @ physical_shapes) ** 2
        denominator = measured_norm[i] ** 2 * physical_norm ** 2
        overlaps[i] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        distances[i] = np.sqrt(np.mean((float(mode["frequency"][i]) - frequencies) ** 2, axis=0))
    valid = eligible[None, :] & (distances <= p["correspondence"]["maximum_frequency_distance_hz"])
    # A unique assignment is required.  Dummy assignments are represented by
    # invalid zero-overlap edges and are rejected by the confidence checks.
    cost = np.where(valid, -overlaps, 1e6)
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(cost)
    assigned = {int(row): int(column) for row, column in zip(rows, columns) if cost[row, column] < 1e5}
    for i in range(count):
        candidates = np.flatnonzero(valid[i])
        if not len(candidates):
            reasons[i] = "no_frequency_correspondence"
            continue
        order = candidates[np.argsort(-overlaps[i, candidates], kind="stable")]
        best = float(overlaps[i, order[0]])
        second = float(overlaps[i, order[1]]) if len(order) > 1 else 0.0
        best_second[i] = [best, second]
        branch = assigned.get(i, -1)
        if branch < 0:
            reasons[i] = "nonunique_assignment"
            continue
        if branch != int(order[0]):
            reasons[i] = "nonunique_assignment"
            continue
        if best < p["correspondence"]["minimum_squared_overlap"] or best - second < p["correspondence"]["minimum_best_minus_second_squared_overlap"]:
            reasons[i] = "spatial_correspondence_ambiguous"
            continue
        if not quality[:, branch].all():
            reasons[i] = "tracking_ambiguous"
            continue
        branches[i] = branch
        reasons[i] = "resolved"
        condition[i] = basis_condition
        sensor_reconstruction[i] = table["maximum_sensor_error"]
    return branches, best_second, reasons, condition, sensor_reconstruction


def _match_and_oracle(mode, signal, raw, scale, start, end, table, p, plate, condition_cache):
    branches, scores, reasons, condition, branch_diag = _oracle_match(
        mode, table, scale, start, end, p, plate, condition_cache
    )
    oracle_mode = dict(mode)
    oracle_mode["alpha"] = np.asarray(mode["alpha"], dtype=float).copy()
    oracle_mode["frequency"] = np.asarray(mode["frequency"], dtype=float).copy()
    oracle_mode["mode_shapes"] = np.asarray(mode["mode_shapes"], dtype=np.complex128).copy()
    center = int(table["indices"][0, end - 1])
    poles = table["poles"][center]
    physical_shapes = table["shapes"][center] / scale[:, None]
    for slot, branch in enumerate(branches):
        if branch < 0:
            continue
        oracle_mode["alpha"][slot] = float(poles[branch].real)
        oracle_mode["frequency"][slot] = float(poles[branch].imag / (2 * np.pi))
        oracle_mode["mode_shapes"][slot] = physical_shapes[:, branch]
    return oracle_mode, branches, scores, reasons, condition, branch_diag

def _oracle_feature_row(signal, measured, physical, original, d, engine):
    """Replace physical coordinates without rerouting measured modal slots."""
    from mechanics.p4_margin_estimation.modal_features import physics_feature_groups
    row = M.features(signal, physical, d, engine)
    row[:7] = original[:7]
    row[41] = original[41]
    groups = physics_feature_groups(physical)
    visible = np.flatnonzero(np.asarray(measured["energy_fraction"]) >= .01)
    a0, f0 = measured["alpha"], measured["frequency"]
    order = visible[np.lexsort((f0[visible], -a0[visible]))]
    a, f, e = physical["alpha"], physical["frequency"], physical["energy_fraction"]
    groups["stability"][:] = 0
    groups["stability"][-1] = len(order)
    for slot, index in enumerate(order[:3]):
        groups["stability"][3 * slot:3 * slot + 3] = a[index], f[index], e[index]
    if len(order) > 1:
        lead, others = order[0], order[1:]
        neighbor = others[np.argmin(abs(f0[others] - f0[lead]))]
        i, j = np.triu_indices(len(visible), k=1)
        natural = np.hypot(a, 2 * np.pi * f)
        groups["spacing"][:] = (
            2 * abs(f[lead] - f[neighbor]) / (f[lead] + f[neighbor]),
            np.min(2 * abs(f[visible[i]] - f[visible[j]]) / (f[visible[i]] + f[visible[j]])),
            2 * np.hypot(a[lead] - a[neighbor], 2 * np.pi * (f[lead] - f[neighbor])) / (natural[lead] + natural[neighbor]), 1,
        )
    row[22:41] = np.concatenate(list(groups.values()))
    routing = dict(physical, alpha=np.asarray(a0), frequency=np.asarray(f0))
    row[42:107] = G.O.B.modal_spatial_row(routing, d["pairs"])
    row[107:] = G.O.B.modal_spatial_addition_row(routing, d["pairs"])
    return row



def _row_diagnostics(mode):
    alpha = np.full(MAX_MODES, np.nan)
    frequency = np.full(MAX_MODES, np.nan)
    energy = np.full(MAX_MODES, np.nan)
    shapes = np.full((MAX_MODES, 16), np.nan + 0j, dtype=np.complex128)
    count = min(MAX_MODES, len(mode.get("alpha", [])))
    if count:
        alpha[:count] = mode["alpha"][:count]
        frequency[:count] = mode["frequency"][:count]
        energy[:count] = mode["energy_fraction"][:count]
        shapes[:count] = mode["mode_shapes"][:count]
    return count, alpha, frequency, energy, shapes, float(mode.get("relative_fit_error", 1.0)), float(mode.get("usable_fraction", 0.0))


def _physical_sensor_error(table, states, projection, start, end):
    components = table["pairs"][0, start:end]
    sensor = states[start:end] @ projection.T
    denom = max(float(np.linalg.norm(sensor)), 1e-30)
    return float(np.linalg.norm(components.sum(axis=2) - sensor) / denom)


def _make_row_result(record, raw, command, d, engine, table, physical_states, p, plate, canonical=None, use_gpu=True):
    times = np.asarray(canonical["times"] if canonical is not None else np.arange(128, raw.shape[-1] + 1, 64) * 0.001 - 0.001, dtype=float)
    ends = (np.rint(times * 1000).astype(int) + 1) if canonical is not None else np.arange(128, raw.shape[-1] + 1, 64)
    if canonical is not None:
        q = canonical["features"].copy()
        observed = canonical["observed"].copy()
        age = canonical["age"].copy()
        truth = canonical["truth"].copy()
    else:
        q = np.zeros((len(ends), 174), dtype=float)
        observed = np.zeros(len(ends), dtype=bool)
        age = np.zeros(len(ends), dtype=float)
        ratio = np.interp(times, record["schedule"]["knots"], record["schedule"]["ratios"])
        truth = 1.0 - ratio
    noise = noise_for(raw, record)
    signal, offset, scale = M.normalize(raw, noise, cast=True)
    if canonical is not None:
        frames = [int(end) for end, ok in zip(ends, observed) if ok]
        modes = _row_modes(signal, frames, use_gpu=use_gpu)
    else:
        # This is the unchanged causal acquisition gate from temporal_probe:
        # command-free, finite, unsaturated 128-sample windows followed by the
        # strictly earlier 64-to-64 forecast skill test.
        candidate_frames = []
        candidate_windows = []
        for end in ends:
            start = int(end) - 128
            window = signal[:, start:int(end)]
            if not command[start:int(end)].any() and np.isfinite(window).all() and not (abs(window) >= 50).any():
                candidate_frames.append(int(end))
                candidate_windows.append(window)
        modes = {}
        if candidate_windows:
            if use_gpu:
                prefix_modes = estimate_batch([window[:, :64] for window in candidate_windows])
                skills = P.forecast_skills(candidate_windows, prefix_modes, 64, 64)
                chosen = [i for i, skill in enumerate(skills) if skill >= 0.5]
                chosen_modes = estimate_batch([candidate_windows[i] for i in chosen]) if chosen else []
            else:
                prefix_modes = [cpu_estimate_modes(window[:, :64], 0.001, include_shapes=True) for window in candidate_windows]
                skills = P.forecast_skills(candidate_windows, prefix_modes, 64, 64)
                chosen = [i for i, skill in enumerate(skills) if skill >= 0.5]
                chosen_modes = [cpu_estimate_modes(candidate_windows[i], 0.001, include_shapes=True) for i in chosen]
            for i, mode in zip(chosen, chosen_modes):
                end = candidate_frames[i]
                if len(mode.get("alpha", [])):
                    row = M.features(candidate_windows[i], mode, d, engine)
                    if np.isfinite(row).all():
                        frame = int(np.flatnonzero(ends == end)[0])
                        q[frame] = row
                        observed[frame] = True
                        modes[end] = mode
        last = 0.0
        for frame, time_value in enumerate(times):
            if observed[frame]:
                last = float(time_value)
            age[frame] = float(time_value) - last
        frames = [int(end) for end, ok in zip(ends, observed) if ok]
    oracle = q.copy()
    matched_branch = np.full((len(ends), MAX_MODES), -1, dtype=np.int16)
    match_scores = np.full((len(ends), MAX_MODES, 2), np.nan, dtype=float)
    replacement_reason = np.full((len(ends), MAX_MODES), "unobserved", dtype="<U40")
    condition = np.full((len(ends), MAX_MODES), np.nan, dtype=float)
    reconstruction = np.full((len(ends), MAX_MODES), np.nan, dtype=float)
    mode_count = np.zeros(len(ends), dtype=np.int16)
    mode_alpha = np.full((len(ends), MAX_MODES), np.nan, dtype=float)
    mode_frequency = np.full((len(ends), MAX_MODES), np.nan, dtype=float)
    mode_energy = np.full((len(ends), MAX_MODES), np.nan, dtype=float)
    mode_shapes = np.full((len(ends), MAX_MODES, raw.shape[0]), np.nan + 0j, dtype=np.complex128)
    mode_fit_error = np.full(len(ends), np.nan, dtype=float)
    mode_usable = np.full(len(ends), np.nan, dtype=float)
    condition_cache = {}
    for frame, end in enumerate(ends):
        if not observed[frame]:
            continue
        mode = modes.get(int(end))
        if mode is None or not len(mode.get("alpha", [])):
            replacement_reason[frame, :] = "unidentified"
            continue
        count, alpha, freq, energy, shapes, fit_error, usable = _row_diagnostics(mode)
        if count > MAX_MODES:
            raise RuntimeError("modal estimator exceeded frozen eight-mode width")
        mode_count[frame] = count
        mode_alpha[frame] = alpha
        mode_frequency[frame] = freq
        mode_energy[frame] = energy
        mode_shapes[frame] = shapes
        mode_fit_error[frame] = fit_error
        mode_usable[frame] = usable
        replacement_reason[frame, :count] = "unresolved"
        oracle_mode, branches, scores, reasons, cond, branch_diag = _match_and_oracle(
            mode,
            signal,
            raw,
            scale,
            int(end) - 128,
            int(end),
            table,
            p,
            plate,
            condition_cache,
        )
        matched_branch[frame, :count] = branches
        match_scores[frame, :count] = scores
        replacement_reason[frame, :count] = reasons
        condition[frame, :count] = cond
        for slot, branch in enumerate(branches):
            if branch >= 0:
                reconstruction[frame, slot] = _physical_sensor_error(
                    table, physical_states, plate.reduced["projection"], int(end) - 128, int(end)
                )
        if np.any(branches >= 0):
            oracle_row = _oracle_feature_row(signal[:, int(end) - 128:int(end)], mode, oracle_mode, q[frame], d, engine)
            if not np.isfinite(oracle_row).all():
                raise FloatingPointError(f"nonfinite matched-modal oracle row: {record}")
            oracle[frame] = oracle_row
    return dict(
        features=q,
        oracle_features=oracle,
        truth=truth,
        observed=observed,
        age=age,
        times=times,
        diagnostics=dict(
            mode_count=mode_count,
            mode_alpha=mode_alpha,
            mode_frequency=mode_frequency,
            mode_energy=mode_energy,
            mode_shapes=mode_shapes,
            mode_fit_error=mode_fit_error,
            mode_usable_fraction=mode_usable,
            matched_branch=matched_branch,
            match_scores=match_scores,
            replacement_reason=replacement_reason,
            eigenbasis_condition=condition,
            reconstruction_error=reconstruction,
        ),
    )


def _canonical_indices(data, p):
    split = split_map(p)
    allowed = []
    for i, record in enumerate(data["records"]):
        did = record["design"]
        if did not in split or record["profile"] not in OLD_PROFILES:
            continue
        names = allowed_noise(split[did], p)
        if record["noise"] in names:
            allowed.append(i)
    expected = 0
    for did in split:
        profiles = len(OLD_PROFILES)
        rows = 2 * profiles * len(allowed_noise(split[did], p))
        expected += rows
    if len(allowed) != expected:
        raise RuntimeError(f"canonical original population count mismatch: {len(allowed)} != {expected}")
    return allowed


def _physical_lookup(shards):
    lookup = {}
    for did, shard in shards.items():
        for i, record in enumerate(shard["records"]):
            lookup[(did, int(record["replicate"]), record["profile"])] = i
    return lookup


def _load_plate_cache(d, fp, designs):
    plates = {}
    validations = {}
    for did in designs:
        _, plate, _ = _load_o1_context(d, fp, did)
        plates[did] = plate
        shard = _load_shard(did)
        validations[did] = shard["validation"]
    return plates, validations


def _aggregate_diagnostics(records, diagnostics):
    result = {}
    total = dict(observed_windows=0, modal_slots=0, replacement_slots=0, unresolved_slots=0, ambiguous_slots=0, tracking_ambiguous_slots=0, condition_fail_slots=0)
    for record, diag in zip(records, diagnostics):
        key = f"{record['design']}|{record['replicate']}|{record['profile']}|{record['noise']}"
        observed = diag["mode_count"] > 0
        slots = int(diag["mode_count"].sum())
        replaced = int((diag["matched_branch"] >= 0).sum())
        reasons = diag["replacement_reason"]
        result[key] = dict(
            design=record["design"],
            split=record["split"],
            profile=record["profile"],
            noise=record["noise"],
            observed_windows=int(observed.sum()),
            modal_slots=slots,
            replacement_slots=replaced,
            unresolved_slots=int(np.count_nonzero((reasons == "unresolved") | (reasons == "unidentified"))),
            ambiguous_slots=int(np.count_nonzero(np.char.find(reasons.astype(str), "ambiguous") >= 0)),
            tracking_ambiguous_slots=int(np.count_nonzero(reasons == "tracking_ambiguous")),
            condition_fail_slots=int(np.count_nonzero(reasons == "condition_fail")),
            replacement_coverage=float(replaced / slots) if slots else None,
            maximum_eigenbasis_condition=float(np.nanmax(diag["eigenbasis_condition"])) if np.isfinite(diag["eigenbasis_condition"]).any() else None,
            maximum_reconstruction_error=float(np.nanmax(diag["reconstruction_error"])) if np.isfinite(diag["reconstruction_error"]).any() else None,
        )
        total["observed_windows"] += int(observed.sum())
        total["modal_slots"] += slots
        total["replacement_slots"] += replaced
        total["unresolved_slots"] += result[key]["unresolved_slots"]
        total["ambiguous_slots"] += result[key]["ambiguous_slots"]
        total["tracking_ambiguous_slots"] += result[key]["tracking_ambiguous_slots"]
        total["condition_fail_slots"] += result[key]["condition_fail_slots"]
    total["replacement_coverage"] = float(total["replacement_slots"] / total["modal_slots"]) if total["modal_slots"] else None
    return dict(total=total, by_design_noise_profile=result)


def combine_phase(p):
    dataset_path = OUT / "dataset.npz"
    if dataset_path.exists():
        raise RuntimeError("Acquired support dataset is immutable")
    d, fp = inputs()
    original_designs = list(p["split"]["train"]) + list(p["split"]["validation"]) + list(p["split"]["held"])
    additional_designs = designated_additional(p)
    all_designs = original_designs + additional_designs
    shards = {did: _load_shard(did) for did in all_designs}
    plates, validations = _load_plate_cache(d, fp, all_designs)
    canonical = canonical_data()
    canonical_rows = _canonical_indices(canonical, p)
    canonical_by_key = {tuple(canonical["records"][i][k] for k in ("split", "design", "replicate", "profile", "noise")): i for i in canonical_rows}
    source_raw, source_command, source_records_list = source_records()
    source_map = {(r["design"], int(r["replicate"]), r["profile"]): i for i, r in enumerate(source_records_list)}
    fold = evaluation_fold(p)
    physical_lookup = _physical_lookup(shards)
    table_cache = {}
    combined_records = []
    feature_rows, oracle_rows, truths, observed_rows, ages = [], [], [], [], []
    mode_diags = []
    canonical_delta = 0.0
    engine = SimpleNamespace(ridge=G.O.B.PhysicsFeatureRidge())

    def table_for(record, shard):
        key = (record["design"], int(record["replicate"]), record["profile"])
        if key not in table_cache:
            table_cache.clear()
            pi = physical_lookup[key]
            table_cache[key] = _physical_table(shard, pi, shard["records"][pi]["schedule"], p)
        return table_cache[key]

    # Frozen original measured rows, including fresh all-13 views for held designs.
    for canonical_index in canonical_rows:
        source_record = copy.deepcopy(canonical["records"][canonical_index])
        did = source_record["design"]
        shard = shards[did]
        physical_index = source_map[(did, int(source_record["replicate"]), source_record["profile"])]
        physical_record = shard["records"][physical_lookup[(did, int(source_record["replicate"]), source_record["profile"])]]
        table = table_for(source_record, shard)
        row_record = source_record
        row_record["source_kind"] = "original_cache"
        row_record["evaluation_fold"] = None
        row_record["accepted_windows"] = int(canonical["observed"][canonical_index].sum())
        row_record["frames"] = int(len(canonical["times"]))
        result = _make_row_result(
            row_record,
            source_raw[physical_index],
            source_command[physical_index],
            d,
            engine,
            table,
            shard["states"][physical_lookup[(did, int(source_record["replicate"]), source_record["profile"]) ]],
            p,
            plates[did],
            canonical={**{name: canonical[name][canonical_index] for name in ("features", "truth", "observed", "age")}, "times": canonical["times"]},
            use_gpu=True,
        )
        canonical_delta = max(canonical_delta, float(np.nanmax(abs(result["features"] - canonical["features"][canonical_index]))))
        combined_records.append(row_record)
        feature_rows.append(result["features"])
        oracle_rows.append(result["oracle_features"])
        truths.append(result["truth"])
        observed_rows.append(result["observed"])
        ages.append(result["age"])
        mode_diags.append(result["diagnostics"])

    # Both fitting populations include the new regimes; support is not confounded with their presence.
    for did in all_designs:
        shard = shards[did]
        split = "train" if did in additional_designs else split_map(p)[did]
        noise_names = p["noise"] if did in additional_designs or split == "held" else BASE_NOISE
        for physical_index, physical_record in enumerate(shard["records"]):
            if did in original_designs and physical_record["profile"] in OLD_PROFILES:
                continue
            table = table_for(physical_record, shard)
            for noise in noise_names:
                record = dict(
                    design=did,
                    split=split,
                    profile=physical_record["profile"],
                    replicate=int(physical_record["replicate"]),
                    noise=noise,
                    schedule=physical_record["schedule"],
                    source_kind="additional_trajectory" if did in additional_designs else "original_added_regime",
                    evaluation_fold=int(fold[did]) if did in additional_designs else None,
                )
                raw = shard["raw"][physical_index]
                result = _make_row_result(
                    record,
                    raw,
                    shard["command"][physical_index],
                    d,
                    engine,
                    table,
                    shard["states"][physical_index],
                    p,
                    plates[did],
                    canonical=None,
                    use_gpu=True,
                )
                record["accepted_windows"] = int(result["observed"].sum())
                record["frames"] = int(len(result["times"]))
                combined_records.append(record)
                feature_rows.append(result["features"])
                oracle_rows.append(result["oracle_features"])
                truths.append(result["truth"])
                observed_rows.append(result["observed"])
                ages.append(result["age"])
                mode_diags.append(result["diagnostics"])
                print("SUPPORT_COMBINED", did, physical_record["replicate"], physical_record["profile"], noise, flush=True)

    features = np.stack(feature_rows).astype(np.float64)
    oracle_features = np.stack(oracle_rows).astype(np.float64)
    truth = np.stack(truths).astype(np.float64)
    observed = np.stack(observed_rows).astype(bool)
    age = np.stack(ages).astype(np.float64)
    times = np.asarray(canonical["times"], dtype=float)
    if features.shape != oracle_features.shape or features.shape[2] != 174:
        raise RuntimeError(f"support feature shape mismatch: {features.shape}/{oracle_features.shape}")
    if truth.shape != observed.shape or age.shape != observed.shape or times.shape != (features.shape[1],):
        raise RuntimeError("support row-array shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(oracle_features).all():
        raise FloatingPointError("support dataset contains nonfinite feature rows")
    if np.any(oracle_features[~observed] != 0) or np.any(features[~observed] != 0):
        raise RuntimeError("unobserved support rows must remain zero")

    def stack_diag(name, dtype=float):
        return np.stack([diag[name] for diag in mode_diags]).astype(dtype)

    diagnostics = _aggregate_diagnostics(combined_records, mode_diags)
    diagnostics.update(
        dict(
            canonical_measured_max_abs_delta=canonical_delta,
            total_rows=int(len(combined_records)),
            original_rows=int(len(canonical_rows)),
            additional_rows=int(len(combined_records) - len(canonical_rows)),
            original_profile_regime="cross/stable/retreat from frozen source raw/state replay",
            additional_profile_regime="cross/stable/retreat plus farther_stable/unstable_exclusion from new trajectories",
            oracle_policy="matched physical branch only; no true-critical selection; unresolved slots retain measured mode",
            physical_validation=validations,
        )
    )
    save_npz_atomic(
        dataset_path,
        features=features,
        oracle_features=oracle_features,
        truth=truth,
        observed=observed,
        age=age,
        times=times,
        records=json.dumps(clean_json(combined_records)),
        mode_count=stack_diag("mode_count", np.int16),
        measured_alpha=stack_diag("mode_alpha"),
        measured_frequency=stack_diag("mode_frequency"),
        measured_energy_fraction=stack_diag("mode_energy"),
        measured_mode_shapes=stack_diag("mode_shapes"),
        measured_fit_error=stack_diag("mode_fit_error"),
        measured_usable_fraction=stack_diag("mode_usable_fraction"),
        matched_branch=stack_diag("matched_branch", np.int16),
        match_scores=stack_diag("match_scores"),
        replacement_reason=stack_diag("replacement_reason", "<U40"),
        eigenbasis_condition=stack_diag("eigenbasis_condition"),
        reconstruction_error=stack_diag("reconstruction_error"),
        replacement_diagnostics=json.dumps(clean_json(diagnostics)),
        replacement_coverage=json.dumps(clean_json(diagnostics["total"])),
        protocol_sha256=np.array(protocol_sha()),
        implementation_sha256=np.array(sha(Path(__file__))),
        input_fingerprint=json.dumps(fp),
    )
    save_json(OUT / "acquisition.json", dict(
        phase="combine",
        protocol_sha256=protocol_sha(),
        implementation_sha256=sha(Path(__file__)),
        dataset_sha256=sha(dataset_path),
        rows=int(len(combined_records)),
        features_shape=list(features.shape),
        original_rows=int(len(canonical_rows)),
        additional_rows=int(len(combined_records) - len(canonical_rows)),
        replacement_coverage=diagnostics["total"],
        physical_validation=validations,
    ))
    print("SUPPORT_COMBINED_COMPLETE", features.shape, json.dumps(clean_json(diagnostics["total"])), flush=True)


def mechanism_smoke():
    """Tiny CPU-only check for the mode-slot replacement mechanism."""
    p = contract()
    # The smoke intentionally uses the frozen feature path, but no CUDA
    # propagation or estimator.  A deterministic two-tone window exercises
    # the shape-enabled modal dictionary and all 174 output coordinates.
    t = np.arange(128) * 0.001
    shape = np.arange(1.0, 17.0)[:, None]
    signal = shape * (np.exp(-2.0 * t)[None] * np.cos(2 * np.pi * 18 * t)[None] + 0.4 * np.exp(-4.0 * t)[None] * np.cos(2 * np.pi * 31 * t)[None])
    mode = cpu_estimate_modes(signal, 0.001, include_shapes=True)
    if not len(mode["alpha"]):
        raise RuntimeError("CPU modal smoke did not identify the deterministic two-tone signal")
    d, _ = inputs()
    engine = SimpleNamespace(ridge=G.O.B.PhysicsFeatureRidge())
    row = M.features(signal.astype(np.float32), mode, d, engine)
    if row.shape != (174,) or not np.isfinite(row).all():
        raise RuntimeError(f"modal feature smoke shape/nonfinite failure: {row.shape}")
    extended = extended_schedule("D000062", 0, "farther_stable", p)
    excluded = extended_schedule("D000062", 0, "unstable_exclusion", p)
    if not (min(extended["ratios"]) >= 0.78 and max(extended["ratios"]) <= 0.84):
        raise RuntimeError("farther-stable schedule escaped preregistered range")
    if not (min(excluded["ratios"]) >= 1.015 and max(excluded["ratios"]) <= 1.03):
        raise RuntimeError("unstable-exclusion schedule escaped preregistered range")
    print("SUPPORT_MECHANISM_SMOKE", json.dumps(dict(feature_width=int(row.size), modes=int(len(mode["alpha"])), protocol_sha256=protocol_sha())), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("acquire", "combine", "smoke"), required=True)
    parser.add_argument("--design", action="append", default=None, help="one frozen design; repeat for multiple shards")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        p = contract()
        if args.phase == "smoke":
            mechanism_smoke()
        elif args.phase == "acquire":
            acquire_phase(p, args.design)
        else:
            if args.design:
                raise ValueError("--design is only valid for --phase acquire")
            combine_phase(p)


if __name__ == "__main__":
    main()
