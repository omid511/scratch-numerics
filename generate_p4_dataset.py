#!/usr/bin/env python3
"""Generate the expanded P4 dataset: 512 designs × 3 realizations × 9 velocities × 2 excitations.

Saves to disk as:
  metadata.json  — design parameters, velocity levels, split assignments
  clips.npz      — all sensor signals stacked (N_clips, N_sensors, T)
  margins.npy    — all margins (N_clips,)
  velocities.npy — all velocities (N_clips,)
  design_ids.npy — design ID per clip (N_clips,)
  realization_ids.npy — realization ID per clip (N_clips,)
"""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp
import json
import hashlib
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from mechanics.p4_margin_estimation.design_sampler import (
    DesignSample, sample_designs, make_solver_from_design, compute_design_u_crit,
)
from mechanics.p4_margin_estimation.transient import (
    compute_eigendecomposition, generate_clip_from_eigendecomposition,
)

logger = logging.getLogger(__name__)

N_REALIZATIONS = 3
N_DESIGNS = 512
# Continuous velocity ratios: explicit strata covering full range
N_EXCITATIONS = 2
N_SENSORS = 8
N_MODES = 8

ZETA_PERTURB = 0.2
FACE_E_PERTURB = 0.03
CORE_G_PERTURB = 0.05
RHO_PERTURB = 0.02

MIN_MODEL_MACH = 2.0


VELOCITY_RATIO_STRATA = (
    (0.68, 0.85, 2),
    (0.85, 0.95, 2),
    (0.95, 1.00, 2),
    (1.00, 1.05, 2),
    (1.05, 1.15, 2),
)

MARGIN_DEFINITION = "1_minus_velocity_over_flutter_velocity"


def compute_flutter_margin(velocity: float, u_crit: float) -> float:
    if velocity <= 0.0:
        raise ValueError(f"velocity must be positive, got {velocity}")
    if u_crit <= 0.0:
        raise ValueError(f"u_crit must be positive, got {u_crit}")
    return 1.0 - velocity / u_crit


def _design_seed(design_id: str) -> int:
    h = hashlib.md5(design_id.encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def _sample_velocity_ratios(rng: np.random.Generator) -> list[float]:
    """Sample continuous velocity ratios from explicit strata."""
    ratios = []
    for lower, upper, count in VELOCITY_RATIO_STRATA:
        ratios.extend(rng.uniform(lower, upper, size=count).tolist())
    rng.shuffle(ratios)
    return ratios


def valid_ratios_for_design(ratios, *, u_crit, c_air):
    return [r for r in ratios if r * u_crit / c_air >= MIN_MODEL_MACH]


def _split_designs(design_ids: list[str], seed: int = 42):
    ids = sorted(design_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(0.7 * len(ids))
    n_val = int(0.15 * len(ids))
    return set(ids[:n_train]), set(ids[n_train:n_train + n_val]), set(ids[n_train + n_val:])


def _perturbed_design(design: DesignSample, realization_idx: int, rng: np.random.Generator) -> DesignSample:
    zeta = design.zeta * (1.0 + rng.uniform(-ZETA_PERTURB, ZETA_PERTURB))
    return DesignSample(
        design_id=design.design_id,
        L1=design.L1, L2=design.L2,
        face_thickness=design.face_thickness, core_thickness=design.core_thickness,
        face_E_mult=design.face_E_mult * (1.0 + rng.uniform(-FACE_E_PERTURB, FACE_E_PERTURB)),
        face_rho_mult=design.face_rho_mult * (1.0 + rng.uniform(-RHO_PERTURB, RHO_PERTURB)),
        core_G_mult=design.core_G_mult * (1.0 + rng.uniform(-CORE_G_PERTURB, CORE_G_PERTURB)),
        core_rho_mult=design.core_rho_mult * (1.0 + rng.uniform(-RHO_PERTURB, RHO_PERTURB)),
        zeta=zeta,
        rho_air=design.rho_air, c_air=design.c_air,
    )


def _worker_process_design(args):
    """Process one design. Module-scope for spawn pickling."""
    design, split = args
    t0 = time.time()

    try:
        with threadpool_limits(limits=1):
            solver = make_solver_from_design(design)
            v_lower = 2.0 * design.c_air * 1.01
            u_crit = compute_design_u_crit(solver, design, v_lower=v_lower)

        if u_crit is None:
            return None, f"{design.design_id}: u_crit not found"

        clips = []
        failures = []
        rng_base = np.random.default_rng(_design_seed(design.design_id))

        for r_idx in range(N_REALIZATIONS):
            design_pert = _perturbed_design(design, r_idx, rng_base)

            try:
                with threadpool_limits(limits=1):
                    solver_r = make_solver_from_design(design_pert)
                    v_lower_r = 2.0 * design_pert.c_air * 1.01
                    u_crit_r = compute_design_u_crit(solver_r, design_pert, v_lower=v_lower_r)
            except Exception as e:
                # Never fall back to nominal — skip this realization
                logger.warning("Realization %d failed for %s: %s", r_idx, design.design_id, e)
                continue

            if u_crit_r is None:
                continue

            # Compute velocities from perturbed u_crit for this realization
            vel_ratios = _sample_velocity_ratios(rng_base)
            vel_ratios = valid_ratios_for_design(vel_ratios, u_crit=u_crit_r, c_air=design_pert.c_air)
            # Resample if too few valid ratios remain
            if len(vel_ratios) < 4:
                min_ratio = MIN_MODEL_MACH * design_pert.c_air / u_crit_r
                lower = max(0.68, min_ratio)
                if lower >= 1.15:
                    failures.append({"phase": "velocity_filter", "error": "flutter boundary below Mach envelope"})
                    continue
                while len(vel_ratios) < 10:
                    vel_ratios.append(float(rng_base.uniform(lower, 1.15)))
            rng_base.shuffle(vel_ratios)
            all_vels = [float(r * u_crit_r) for r in vel_ratios]

            for vel in all_vels:
                try:
                    with threadpool_limits(limits=1):
                        eigs = compute_eigendecomposition(
                            solver_r, vel, n_modes=N_MODES, n_sensors=N_SENSORS,
                            rho=design_pert.rho_air, c_sound=design_pert.c_air,
                            zeta=design_pert.zeta,
                        )
                except Exception as e:
                    failures.append({
                        "velocity": vel,
                        "phase": "eigendecomposition",
                        "error": str(e),
                        "mach": vel / design_pert.c_air,
                        "margin": compute_flutter_margin(vel, u_crit_r) if u_crit_r else None,
                    })
                    continue

                for exc_idx in range(N_EXCITATIONS):
                    ss = np.random.SeedSequence(entropy=[
                        _design_seed(design.design_id),
                        r_idx,
                        int(round(vel * 1000)),
                        exc_idx,
                    ])
                    rng = np.random.default_rng(ss.generate_state(1)[0])

                    try:
                        clip = generate_clip_from_eigendecomposition(
                            eigs, rng, u_crit=u_crit_r,
                        )
                    except Exception as e:
                        failures.append({
                            "velocity": vel,
                            "phase": "clip_generation",
                            "error": str(e),
                            "mach": vel / design_pert.c_air,
                            "margin": compute_flutter_margin(vel, u_crit_r) if u_crit_r else None,
                        })
                        continue

                    clips.append({
                        "signals": clip.sensor_signals,
                        "margin": clip.margin,
                        "velocity": clip.velocity,
                        "design_id": design.design_id,
                        "realization_idx": r_idx,
                        "excitation_idx": exc_idx,
                        "split": split,
                    })

        elapsed = time.time() - t0
        return clips, {
            "design_id": design.design_id,
            "u_crit": u_crit,
            "n_clips": len(clips),
            "split": split,
            "elapsed": elapsed,
            "failures": failures,
        }
    except Exception as e:
        return None, f"{design.design_id}: {e}\n{traceback.format_exc()}"


def generate_accepted_designs(
    n_designs: int,
    seed: int,
) -> tuple[list[DesignSample], list[str]]:
    """Generate n_designs accepted designs with deterministic replacement."""
    accepted: list[DesignSample] = []
    rejection_reasons: list[str] = []
    batch_index = 0

    while len(accepted) < n_designs:
        remaining = n_designs - len(accepted)
        batch_seed = int(np.random.SeedSequence([seed, batch_index]).generate_state(1)[0])
        candidates = sample_designs(max(remaining * 2, 16), seed=batch_seed)
        for candidate in candidates:
            try:
                solver_c = make_solver_from_design(candidate)
                v_lower_c = 2.0 * candidate.c_air * 1.01
                u_crit_c = compute_design_u_crit(solver_c, candidate, v_lower=v_lower_c)
                if u_crit_c is None:
                    rejection_reasons.append(f"{candidate.design_id}: u_crit not found")
                    continue
                accepted.append(candidate)
            except Exception as e:
                rejection_reasons.append(f"{candidate.design_id}: {e}")
            if len(accepted) == n_designs:
                break
        batch_index += 1

    logger.info("Generated %d accepted designs after %d batches", len(accepted), batch_index)
    return accepted, rejection_reasons


def generate_dataset(
    n_designs: int,
    output_dir: str,
    seed: int,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {n_designs} designs...", flush=True)
    designs = sample_designs(n_designs, seed=seed)
    print(f"Sampled {len(designs)} designs.", flush=True)

    design_ids = [d.design_id for d in designs]
    train_ids, val_ids, test_ids = _split_designs(design_ids)
    split_map = {did: "train" for did in train_ids}
    split_map.update({did: "val" for did in val_ids})
    split_map.update({did: "test" for did in test_ids})

    work = [(d, split_map[d.design_id]) for d in designs]

    all_clips = []
    design_meta = []
    n_skipped = 0

    context = mp.get_context("spawn")
    worker_count = int(os.environ.get("DESIGN_WORKERS", "4"))

    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = {pool.submit(_worker_process_design, item): i for i, item in enumerate(work)}
        results_by_idx = [None] * len(work)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                clips_or_none, meta = future.result()
            except BaseException as exc:
                n_skipped += 1
                print(f"  Design {idx+1}/{n_designs}: BROKEN — {exc}", flush=True)
                results_by_idx[idx] = (None, str(exc)[:80])
                continue

            if clips_or_none is None:
                n_skipped += 1
                print(f"  Design {idx+1}/{n_designs}: SKIP — {meta[:80]}", flush=True)
                results_by_idx[idx] = (None, meta)
                continue

            results_by_idx[idx] = (clips_or_none, meta)

        for idx, result in enumerate(results_by_idx):
            if result is None:
                continue
            clips_or_none, meta = result
            if clips_or_none is None:
                continue
            all_clips.extend(clips_or_none)
            design_meta.append(meta)
            n_done = idx + 1
            if n_done % 10 == 0 or n_done == n_designs:
                print(
                    f"  Design {n_done}/{n_designs}: {meta['design_id']}, "
                    f"u_crit={meta['u_crit']:.1f}, "
                    f"{meta['n_clips']} clips ({meta['elapsed']:.1f}s)",
                    flush=True,
                )

    if not all_clips:
        print("No clips generated!", flush=True)
        return {"n_clips": 0, "output_dir": str(output_path)}

    n_clips = len(all_clips)
    T = all_clips[0]["signals"].shape[1]
    clips_arr = np.stack([c["signals"] for c in all_clips]).astype(np.float32)
    margins_arr = np.array([c["margin"] for c in all_clips], dtype=np.float32)
    velocities_arr = np.array([c["velocity"] for c in all_clips], dtype=np.float32)
    excitation_ids_arr = np.array([c["excitation_idx"] for c in all_clips])
    design_ids_arr = np.array([c["design_id"] for c in all_clips])
    realization_ids_arr = np.array([c["realization_idx"] for c in all_clips])

    # P1-31: Replace assert with ValueError for data validation
    if clips_arr.ndim != 3:
        raise ValueError(f"clips must have 3 dimensions, got {clips_arr.shape}")
    if not np.all(np.isfinite(clips_arr)):
        bad_count = int((~np.isfinite(clips_arr)).sum())
        raise ValueError(f"Dataset contains {bad_count} non-finite values")

    # P1-32: Low-energy and constant-channel detection
    channel_std = clips_arr.std(axis=2)
    clip_rms = np.sqrt(np.mean(clips_arr ** 2, axis=(1, 2)))
    invalid_low_energy = clip_rms < 1e-8
    invalid_constant_channel = np.any(channel_std < 1e-8, axis=1)
    if np.any(invalid_low_energy):
        indices = np.flatnonzero(invalid_low_energy)[:20]
        raise ValueError(f"Near-zero-energy clips detected at indices {indices.tolist()}")
    if np.any(invalid_constant_channel):
        indices = np.flatnonzero(invalid_constant_channel)[:20]
        raise ValueError(f"Clips with constant channels detected at indices {indices.tolist()}")

    # P1-33: Array alignment checks
    aligned_arrays = {
        "margins": margins_arr,
        "velocities": velocities_arr,
        "design_ids": design_ids_arr,
        "realization_ids": realization_ids_arr,
        "excitation_ids": excitation_ids_arr,
    }
    for name, array in aligned_arrays.items():
        if len(array) != n_clips:
            raise ValueError(f"{name} has length {len(array)}, expected {n_clips}")

    # Check for duplicate clip keys
    clip_keys = list(zip(
        design_ids_arr.tolist(),
        realization_ids_arr.tolist(),
        excitation_ids_arr.tolist(),
        [float(v) for v in velocities_arr.tolist()],
    ))
    if len(clip_keys) != len(set(clip_keys)):
        raise ValueError("Duplicate clip keys detected")

    # Filter out designs with zero clips from splits
    accepted_design_ids = set(c["design_id"] for c in all_clips)
    train_ids = train_ids & accepted_design_ids
    val_ids = val_ids & accepted_design_ids
    test_ids = test_ids & accepted_design_ids

    splits_of_clips = np.array([c["split"] for c in all_clips])
    n_train = int(np.sum(splits_of_clips == "train"))
    n_val = int(np.sum(splits_of_clips == "val"))
    n_test = int(np.sum(splits_of_clips == "test"))

    metadata = {
        "n_designs": n_designs,
        "n_realizations": N_REALIZATIONS,
        "n_excitations": N_EXCITATIONS,
        "n_sensors": N_SENSORS,
        "n_modes": N_MODES,
        "margin_definition": MARGIN_DEFINITION,
        "velocity_sampling": {
            "method": "stratified",
            "strata": [{"lower": lo, "upper": hi, "count": c} for lo, hi, c in VELOCITY_RATIO_STRATA],
            "n_total_velocities": sum(c for _, _, c in VELOCITY_RATIO_STRATA),
        },
        "n_clips": n_clips,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_skipped_designs": n_skipped,
        "clip_shape": list(clips_arr.shape),
        "designs": design_meta,
        "train_designs": sorted(train_ids),
        "val_designs": sorted(val_ids),
        "test_designs": sorted(test_ids),
    }

    # Atomic write: write to temp dir, then rename
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)

    # P2-1: Use .npy with memmap for clips instead of compressed .npz
    clips_path = tmp_path / "clips.npy"
    clips_memmap = np.lib.format.open_memmap(
        clips_path, mode="w+", dtype=np.float32, shape=clips_arr.shape,
    )
    clips_memmap[:] = clips_arr
    clips_memmap.flush()

    # Keep small metadata in NPZ
    np.savez_compressed(
        tmp_path / "metadata_arrays.npz",
        margins=margins_arr,
        velocities=velocities_arr,
        design_ids=design_ids_arr,
        realization_ids=realization_ids_arr,
        excitation_ids=excitation_ids_arr,
    )
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Atomic rename: remove old dir if exists, then rename tmp
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)

    print(f"\nDataset saved to {output_path}/", flush=True)
    for fname in sorted(output_path.iterdir()):
        if fname.is_file():
            size_mb = fname.stat().st_size / 1e6
            print(f"  {fname.name}: {size_mb:.1f} MB", flush=True)

    print(f"\nTotal: {n_clips} clips ({n_train} train, {n_val} val, {n_test} test)", flush=True)
    print(f"Skipped: {n_skipped} designs", flush=True)

    return {
        "n_clips": n_clips,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "designs": design_meta,
        "output_dir": str(output_path),
    }


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    parser = argparse.ArgumentParser(description="Generate P4 dataset")
    parser.add_argument("-n", "--n-designs", type=int, default=N_DESIGNS)
    parser.add_argument("-o", "--output-dir", type=str, default="p4_dataset")
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Generating P4 dataset: {args.n_designs} designs, output={args.output_dir}", flush=True)
    t0 = time.time()
    result = generate_dataset(args.n_designs, args.output_dir, args.seed)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)
