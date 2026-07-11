#!/usr/bin/env python3
"""Generate the expanded P4 dataset: 120 designs × 3 realizations × 14 velocities × 2 excitations = 10,080 clips.

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

N_REALIZATIONS = 3
VELOCITY_RATIOS = [0.88, 0.91, 0.94, 0.96, 0.98, 0.99, 1.00, 1.01, 1.03, 1.06]
FAR_SUBCRITICAL = np.linspace(0.85, 0.68, 4).tolist()
N_EXCITATIONS = 2
N_SENSORS = 8
N_MODES = 8

ZETA_PERTURB = 0.2
FACE_E_PERTURB = 0.03
CORE_G_PERTURB = 0.05
RHO_PERTURB = 0.02


def _design_seed(design_id: str) -> int:
    h = hashlib.md5(design_id.encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


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
        left_k=design.left_k, top_k=design.top_k,
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
            u_crit = compute_design_u_crit(solver, design)

        if u_crit is None:
            return None, f"{design.design_id}: u_crit not found"

        mach = u_crit / design.c_air
        if mach < 1.5 or mach > 9.0:
            return None, f"{design.design_id}: Mach {mach:.1f} outside [1.5, 9.0]"

        far_vels = [float(v * u_crit) for v in FAR_SUBCRITICAL]
        near_vels = [float(r * u_crit) for r in VELOCITY_RATIOS]
        all_vels = far_vels + near_vels

        clips = []
        rng_base = np.random.default_rng(_design_seed(design.design_id))

        for r_idx in range(N_REALIZATIONS):
            design_pert = _perturbed_design(design, r_idx, rng_base)

            try:
                with threadpool_limits(limits=1):
                    solver_r = make_solver_from_design(design_pert)
                    u_crit_r = compute_design_u_crit(solver_r, design_pert)
            except Exception:
                u_crit_r = u_crit

            for vel in all_vels:
                try:
                    with threadpool_limits(limits=1):
                        eigs = compute_eigendecomposition(
                            solver, vel, n_modes=N_MODES, n_sensors=N_SENSORS,
                        )
                except Exception:
                    continue

                for exc_idx in range(N_EXCITATIONS):
                    seed = (
                        _design_seed(design.design_id) * 10000
                        + r_idx * 1000
                        + int(round(vel))
                        + exc_idx
                    ) % (2**31)
                    rng = np.random.default_rng(seed)

                    try:
                        clip = generate_clip_from_eigendecomposition(
                            eigs, rng, u_crit=u_crit_r,
                        )
                    except Exception:
                        continue

                    clips.append({
                        "signals": clip.sensor_signals,
                        "margin": clip.margin,
                        "velocity": clip.velocity,
                        "design_id": design.design_id,
                        "realization_idx": r_idx,
                        "split": split,
                    })

        elapsed = time.time() - t0
        return clips, {
            "design_id": design.design_id,
            "u_crit": u_crit,
            "n_clips": len(clips),
            "velocities": all_vels,
            "split": split,
            "elapsed": elapsed,
        }
    except Exception as e:
        return None, f"{design.design_id}: {e}\n{traceback.format_exc()}"


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
        for i, future in enumerate(as_completed(futures)):
            try:
                clips_or_none, meta = future.result()
            except BaseException as exc:
                n_skipped += 1
                print(f"  Design {i+1}/{n_designs}: BROKEN — {exc}", flush=True)
                continue

            if clips_or_none is None:
                n_skipped += 1
                print(f"  Design {i+1}/{n_designs}: SKIP — {meta[:80]}", flush=True)
                continue

            all_clips.extend(clips_or_none)
            design_meta.append(meta)

            if (i + 1) % 10 == 0 or i + 1 == n_designs:
                print(
                    f"  Design {i+1}/{n_designs}: {meta['design_id']}, "
                    f"u_crit={meta['u_crit']:.1f}, "
                    f"{meta['n_clips']} clips ({meta['elapsed']:.1f}s)",
                    flush=True,
                )

    if not all_clips:
        print("No clips generated!", flush=True)
        return {"n_clips": 0, "output_dir": str(output_path)}

    n_clips = len(all_clips)
    T = all_clips[0]["signals"].shape[1]
    clips_arr = np.stack([c["signals"] for c in all_clips])
    margins_arr = np.array([c["margin"] for c in all_clips])
    velocities_arr = np.array([c["velocity"] for c in all_clips])
    design_ids_arr = np.array([c["design_id"] for c in all_clips])
    realization_ids_arr = np.array([c["realization_idx"] for c in all_clips])

    np.savez_compressed(output_path / "clips.npz", clips=clips_arr)
    np.save(output_path / "margins.npy", margins_arr)
    np.save(output_path / "velocities.npy", velocities_arr)
    np.save(output_path / "design_ids.npy", design_ids_arr)
    np.save(output_path / "realization_ids.npy", realization_ids_arr)

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
        "velocity_ratios": VELOCITY_RATIOS,
        "far_subcritical": FAR_SUBCRITICAL,
        "n_total_velocities": len(VELOCITY_RATIOS) + len(FAR_SUBCRITICAL),
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
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

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
    parser.add_argument("-n", "--n-designs", type=int, default=160)
    parser.add_argument("-o", "--output-dir", type=str, default="p4_dataset")
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Generating P4 dataset: {args.n_designs} designs, output={args.output_dir}", flush=True)
    t0 = time.time()
    result = generate_dataset(args.n_designs, args.output_dir, args.seed)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)
