#!/usr/bin/env python3
"""Replay TRAIN trajectories at a shared 25-location sensor candidate grid.

Preserves raw (pre-normalization) float64 displacements, solver poles for
identification diagnostics, and per-design observability contributions.
Existing eight-sensor clips/labels/timebases must replay before committing.
One atomic NPZ per design permits interruption/resume without resimulation.
"""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from generate_p4_dataset import (
    _design_seed, _perturbed_design, _sample_velocity_ratios, valid_ratios_for_design,
)
from mechanics.p4_margin_estimation.design_sampler import (
    sample_designs, make_solver_from_design, compute_design_u_crit,
)
from mechanics.p4_margin_estimation.transient import (
    compute_eigendecomposition, default_sensor_xy, sample_modal_initial_conditions,
    _clamped_modal_response, causal_calibration_normalize,
)


def candidate_positions():
    grid = np.linspace(1, 30, 5, dtype=int)
    y, x = np.meshgrid(grid, grid)
    positions = np.column_stack([y.ravel(), x.ravel()])
    old = default_sensor_xy(32, 32, 8)
    indices = [int(np.flatnonzero(np.all(positions == p, axis=1))[0]) for p in old]
    return positions, indices


def observability_contributions(eigs):
    """Finite 51-sample, modal-pair-scaled information per candidate.

    Equal variance for real/imag coordinates of each pair avoids arbitrary
    eigenvector scale/phase affecting layout scores. Averaged over velocities
    within each design later. This is an observability surrogate, not label
    information or a sensor-noise model fitted on held-out designs.
    """
    t = np.arange(51) * eigs.dt
    z = eigs.sensor_modes[:, None, :] * _clamped_modal_response(
        eigs.eigvals, t, np.ones(len(eigs.eigvals))) .T[None, :, :]
    scale = np.sqrt(np.mean(np.abs(z) ** 2, axis=(0, 1)))
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("unobservable retained mode in candidate grid")
    z = z / scale
    b = np.concatenate([2 * z.real, -2 * z.imag], axis=2)
    return np.einsum("cti,ctj->cij", b, b) / len(t)


def replay_design(design, source, output, fingerprint):
    with threadpool_limits(limits=1):
        return _replay_design(design, Path(source), Path(output), fingerprint)


def _replay_design(design, source, output, fingerprint):
    a = dict(np.load(source / "metadata_arrays.npz"))
    saved = np.load(source / "clips.npy", mmap_mode="r")
    rows = np.flatnonzero(a["design_ids"] == design.design_id)
    positions, old = candidate_positions()
    raw = np.empty((len(rows), len(positions), 512), dtype=np.float64)
    poles = np.empty((len(rows), 8), dtype=np.complex128)
    noise_scale = np.empty(len(rows))
    written = np.zeros(len(rows), dtype=bool)
    lookup = {(int(a["realization_ids"][i]), float(a["velocities"][i]),
               int(a["excitation_ids"][i])): j for j, i in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("ambiguous source acquisition keys")
    rng_base = np.random.default_rng(_design_seed(design.design_id))
    grams = []
    max_replay_error = 0.0
    for r in range(3):
        pert = _perturbed_design(design, r, rng_base)
        solver = make_solver_from_design(pert)
        u = compute_design_u_crit(solver, pert, v_lower=2 * pert.c_air * 1.01)
        if u is None:
            if np.any(a["realization_ids"][rows] == r):
                raise ValueError("source realization flutter boundary no longer replays")
            continue
        ratios = valid_ratios_for_design(_sample_velocity_ratios(rng_base),
                                         u_crit=u, c_air=pert.c_air)
        if len(ratios) < 4:
            lower = max(0.68, 2 * pert.c_air / u)
            if lower >= 1.15:
                continue
            while len(ratios) < 10:
                ratios.append(float(rng_base.uniform(lower, 1.15)))
        rng_base.shuffle(ratios)
        for ratio in ratios:
            velocity = float(ratio * u)
            keys = [(r, float(np.float32(velocity)), exc) for exc in range(2)]
            if not any(key in lookup for key in keys):
                continue  # Preserve the original accepted-clip population.
            eigs = compute_eigendecomposition(
                solver, velocity, n_modes=8, sensor_xy=positions,
                rho=pert.rho_air, c_sound=pert.c_air, zeta=pert.zeta)
            if len(eigs.eigvals) != 8:
                raise ValueError("retained modal dimension changed")
            grams.append(observability_contributions(eigs))
            t = np.linspace(*eigs.t_span, 512, endpoint=False)
            for exc, key in enumerate(keys):
                if key not in lookup:
                    continue
                j = lookup[key]
                i = rows[j]
                if written[j]:
                    raise ValueError("duplicate replay acquisition")
                ss = np.random.SeedSequence([
                    _design_seed(design.design_id), r, int(round(velocity * 1000)), exc])
                _, coefficients = sample_modal_initial_conditions(
                    eigs.eigvals, np.random.default_rng(ss.generate_state(1)[0]))
                signal = 2 * np.real(eigs.sensor_modes @ _clamped_modal_response(
                    eigs.eigvals, t, coefficients))
                replay = np.clip(causal_calibration_normalize(
                    signal[old], calibration_samples=51), -50, 50).astype(np.float32)
                error = float(np.max(np.abs(replay - saved[i])))
                if not np.allclose(replay, saved[i], atol=1e-5, rtol=3e-6):
                    raise ValueError(f"{design.design_id} row {i}: waveform replay error {error}")
                if (np.float32(1 - velocity / u) != a["margins"][i]
                        or np.float32(eigs.dt) != a["dts"][i]):
                    raise ValueError(f"{design.design_id} row {i}: label/timebase replay changed")
                raw[j] = signal
                poles[j] = eigs.eigvals
                prefix = signal[old, :51]
                noise_scale[j] = np.sqrt(np.mean((prefix - prefix.mean(axis=1, keepdims=True)) ** 2))
                written[j] = True
                max_replay_error = max(max_replay_error, error)
    if not written.all() or not np.isfinite(raw).all():
        raise ValueError(f"incomplete replay for {design.design_id}: {written.sum()}/{len(rows)}")
    target = output / f"{design.design_id}.npz"
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, raw=raw, source_indices=rows, poles=poles,
                            noise_scale=noise_scale, gram=np.mean(grams, axis=0),
                            fingerprint=fingerprint, max_replay_error=max_replay_error)
    temporary.replace(target)
    return design.design_id, len(rows), max_replay_error


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="p4_dataset_saturation")
    parser.add_argument("--output", default="p4_sensor_candidates")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--designs", nargs="+", help="optional acquisition smoke subset")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "metadata.json").read_text())
    train_ids = set(meta["train_designs"])
    selected = set(args.designs) if args.designs else train_ids
    if not selected <= train_ids:
        raise ValueError("only training designs may enter this acquisition study")
    positions, old = candidate_positions()
    sources = [Path(__file__), ROOT / "generate_p4_dataset.py",
               ROOT / "src/mechanics/p4_margin_estimation/transient.py",
               ROOT / "src/mechanics/p4_margin_estimation/design_sampler.py"]
    provenance = {"dataset": str(source), "source_hashes": {
        str(p): file_digest(p) for p in [source / "metadata.json",
                                        source / "metadata_arrays.npz", source / "clips.npy", *sources]},
        "candidate_grid_yx": positions.tolist(), "candidate_fraction_yx": (positions / 31).tolist(),
        "original_indices": old, "training_designs": sorted(train_ids),
        "raw_stage": "post-modal-growth-clamp, pre-normalization and sensor cap",
        "noise_reference": "RMS of original-eight mean-centered first 51 raw samples"}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise ValueError("candidate resume fingerprint mismatch")
    else:
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({**provenance, "fingerprint": fingerprint}, indent=2) + "\n")
        temporary.replace(manifest)
    designs = [d for d in sample_designs(meta["n_designs"], seed=meta["manifest"]["seed"])
               if d.design_id in selected]
    pending = []
    for design in designs:
        path = output / f"{design.design_id}.npz"
        if path.exists():
            with np.load(path) as record:
                if record["fingerprint"].item() != fingerprint:
                    raise ValueError(f"stale candidate shard {path}")
                if not np.isfinite(record["raw"]).all():
                    raise ValueError(f"corrupt candidate shard {path}")
        else:
            pending.append(design)
    print(f"Acquisition: {len(designs)-len(pending)} resumed, {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(replay_design, d, source, output, fingerprint) for d in pending]
        for future in as_completed(futures):
            did, count, error = future.result()
            print(f"Saved {did}: {count} clips, old-layout max error {error:.3g}", flush=True)
    print(f"CANDIDATES_DONE {len(designs)} designs in {output}", flush=True)


if __name__ == "__main__":
    main()
