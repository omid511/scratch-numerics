"""Config-driven parameter sweep data generation for FSDT honeycomb sandwich plates.

Usage:
    python data_generation.py config.yaml              # run sweep from config
    python data_generation.py config.yaml sensitivity  # run sensitivity analysis
    python data_generation.py                          # use default config
"""
from __future__ import annotations
import atexit
import json
import os
import signal
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

# Forked workers inherit BLAS/OpenMP threads → deadlock without this
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.stats import qmc

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver
from mechanics.config import ExperimentConfig, _get_git_hash


DEFAULT_CONFIG_PATH = "configs/default_sweep.yaml"


def _downsample_mode_shapes(mode_shapes: np.ndarray, target_grid: tuple[int, int]) -> np.ndarray:
    """Downsample mode shapes from solver grid to target grid via averaging."""
    from scipy.ndimage import zoom
    n_modes, ny, nx = mode_shapes.shape
    zoom_y = target_grid[0] / ny
    zoom_x = target_grid[1] / nx
    return zoom(mode_shapes, (1, zoom_y, zoom_x), order=1)


@dataclass
class SampleResult:
    params: dict
    frequencies: np.ndarray
    solve_time: float
    success: bool
    error: str | None = None
    mode_shapes: np.ndarray | None = None


def build_laminate(face_thickness: float, core_thickness: float) -> Laminate:
    E = 70e9; nu = 0.33; G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710)
    core = Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)
    h_total = 2 * face_thickness + core_thickness
    z = [-h_total / 2, -h_total / 2 + face_thickness,
         h_total / 2 - face_thickness, h_total / 2]
    return Laminate(materials=[face, core, face], angles=[0, 0, 0], z=z)


def solve_single(params: dict, cfg: ExperimentConfig, n_modes: int = 6, store_mode_shapes: bool = False) -> SampleResult:
    try:
        lam = build_laminate(params["face_thickness"], params["core_thickness"])
        solver = FSDTSolver(
            L1=params["L1"], L2=params["L2"],
            M=int(params.get("M", cfg.solver.M)),
            N=int(params.get("N", cfg.solver.N)),
            laminate=lam, k_stiffness=params["k_stiffness"],
        )
        solver.set_boundary(
            left={"type": cfg.plate.boundary.left.type},
            right={"type": cfg.plate.boundary.right.type},
            top={"type": cfg.plate.boundary.top.type},
            bottom={"type": cfg.plate.boundary.bottom.type},
        )
        t0 = time.time()
        result = solver.solve_modal(n_modes=n_modes)
        solve_time = time.time() - t0
        return SampleResult(
            params=params,
            frequencies=result.frequencies.real,
            solve_time=solve_time,
            success=True,
            mode_shapes=result.mode_shapes if store_mode_shapes else None,
        )
    except Exception as e:
        return SampleResult(
            params=params,
            frequencies=np.zeros(n_modes),
            solve_time=0.0,
            success=False,
            error=str(e)[:200],
        )


def _solve_wrapper(args):
    """Pickle-friendly wrapper for ProcessPoolExecutor."""
    params, cfg_dict, n_modes, store_mode_shapes = args
    cfg = ExperimentConfig._from_dict(cfg_dict)
    return solve_single(params, cfg, n_modes, store_mode_shapes=store_mode_shapes)


def _get_param_ranges(cfg: ExperimentConfig) -> list[tuple[str, float, float, bool]]:
    """Extract sweep parameter ranges from config. Fixed params excluded."""
    ranges = []
    for name, pconf in cfg.sweep.parameters.items():
        if name in cfg.sweep.sensitivity.fixed_params:
            continue  # skip fixed params
        ranges.append((name, float(pconf.range[0]), float(pconf.range[1]), pconf.log_scale))
    return ranges


def _apply_fixed_params(params: dict, cfg: ExperimentConfig) -> dict:
    """Inject fixed parameter values (from sensitivity analysis results)."""
    for name, val in cfg.sweep.sensitivity.fixed_params.items():
        params[name] = val
    return params


def generate_lhs_samples(cfg: ExperimentConfig) -> list[dict]:
    ranges = _get_param_ranges(cfg)
    n_params = len(ranges)
    sampler = qmc.LatinHypercube(d=n_params, seed=cfg.seed)
    raw = sampler.random(cfg.sweep.n_samples)

    eps = np.finfo(float).eps
    samples = []
    for i in range(cfg.sweep.n_samples):
        params = {}
        for j, (name, lo, hi, log_scale) in enumerate(ranges):
            u = np.clip(raw[i, j], eps, 1.0 - eps)
            if log_scale:
                val = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
            else:
                val = lo + u * (hi - lo)
            if name in ("M", "N"):
                val = int(round(val))
            params[name] = val
        params = _apply_fixed_params(params, cfg)
        samples.append(params)
    return samples


def _write_checkpoint(output_path, n_done, param_names, param_buffers, freq_buffer, time_buffer, success_buffer, mode_shape_dir, store_mode_shapes):
    """Save incremental checkpoint from pre-allocated buffers."""
    ckpt_path = output_path.replace(".npz", f"_ckpt_{n_done}.npz")
    ck = Path(ckpt_path)
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck_dict = {}
    for name in param_names:
        ck_dict[f"param_{name}"] = param_buffers[name][:n_done]
    ck_dict["frequencies"] = freq_buffer[:n_done]
    ck_dict["solve_times"] = time_buffer[:n_done]
    ck_dict["success"] = success_buffer[:n_done]
    # ponytail: skip mode shapes in checkpoints to avoid loading all .npy into memory
    np.savez_compressed(str(ck), **ck_dict)
    print(f"  Checkpoint: {ck}")


def run_sweep(cfg: ExperimentConfig | None = None, config_path: str | None = None) -> dict:
    if cfg is None:
        cfg = ExperimentConfig.from_yaml(config_path or DEFAULT_CONFIG_PATH)

    n_samples = cfg.sweep.n_samples
    n_modes = 6
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output.path).parent / f"sweep_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(run_dir / "sweep_output.npz")
    checkpoint_every = cfg.sweep.checkpoint_interval
    store_mode_shapes = cfg.sweep.store_mode_shapes
    mode_shape_grid = tuple(cfg.sweep.mode_shape_grid) if cfg.sweep.mode_shape_grid else None

    print(f"=== Parameter Sweep ===")
    print(f"Config: {cfg.name or 'unnamed'}")
    print(f"Output: {run_dir}")
    print(f"Samples: {n_samples}, Seed: {cfg.seed}")
    print(f"Store mode shapes: {store_mode_shapes}" + (f" (grid: {mode_shape_grid})" if mode_shape_grid else " (full solver grid)"))

    ranges = _get_param_ranges(cfg)
    param_names = [r[0] for r in ranges]
    print(f"Sweep params: {param_names}")
    if cfg.sweep.sensitivity.fixed_params:
        print(f"Fixed params: {list(cfg.sweep.sensitivity.fixed_params.keys())}")

    print(f"Generating LHS samples...")
    samples = generate_lhs_samples(cfg)

    # Pre-allocate incremental buffers (keep memory low for mode shapes)
    param_buffers = {name: np.empty(n_samples) for name in param_names}
    freq_buffer = np.empty((n_samples, n_modes))
    time_buffer = np.empty(n_samples)
    success_buffer = np.empty(n_samples, dtype=bool)
    error_buffer = []  # list of (index, msg)
    # ponytail: write mode shapes to run_dir (NOT /tmp which may be tmpfs = RAM-backed)
    mode_shape_dir = run_dir / "_mode_shapes_tmp" if store_mode_shapes else None
    if mode_shape_dir:
        mode_shape_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_failed = 0
    report_interval = max(1, n_samples // 20)

    # ponytail: parallel for large sweeps, sequential for small ones
    use_parallel = n_samples >= 100 and os.cpu_count() and os.cpu_count() > 1

    if use_parallel:
        cfg_dict = asdict(cfg)
        tasks = [
            (params, cfg_dict, n_modes, store_mode_shapes)
            for params in samples
        ]
        max_workers = cfg.sweep.max_workers if cfg.sweep.max_workers > 0 else min(os.cpu_count(), 3)
        print(f"Parallel sweep: {max_workers} workers")

        _pool_ref = [None]
        _cleaning = [False]
        def _cleanup_workers(signum=None, frame=None):
            if _cleaning[0]:
                return
            _cleaning[0] = True
            if _pool_ref[0] is not None:
                for proc in _pool_ref[0]._processes.values():
                    proc.kill()
            if mode_shape_dir and mode_shape_dir.exists():
                shutil.rmtree(mode_shape_dir, ignore_errors=True)
            os._exit(130)

        atexit.register(_cleanup_workers)
        signal.signal(signal.SIGINT, _cleanup_workers)
        signal.signal(signal.SIGTERM, _cleanup_workers)

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            _pool_ref[0] = pool
            future_to_idx = {pool.submit(_solve_wrapper, t): i for i, t in enumerate(tasks)}
            completed = 0
            try:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx.pop(future)  # ponytail: release Future + result immediately
                    result = future.result()
                    completed += 1

                    # Fill buffers incrementally
                    for name in param_names:
                        param_buffers[name][idx] = result.params.get(name, 0)
                    freq_buffer[idx] = result.frequencies
                    time_buffer[idx] = result.solve_time
                    success_buffer[idx] = result.success
                    if result.success:
                        n_success += 1
                    else:
                        n_failed += 1
                        error_buffer.append((idx, result.error))
                    if store_mode_shapes and result.mode_shapes is not None:
                        ms = _downsample_mode_shapes(result.mode_shapes, mode_shape_grid) if mode_shape_grid else result.mode_shapes
                        np.save(mode_shape_dir / f"{idx}.npy", ms)

                    if completed % report_interval == 0:
                        elapsed = time_buffer[:completed].sum()
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"  [{completed}/{n_samples}] solving... ({rate:.1f} pts/s)")

                    if checkpoint_every and completed % checkpoint_every == 0:
                        _write_checkpoint(output_path, n_samples, param_names, param_buffers, freq_buffer, time_buffer, success_buffer, mode_shape_dir, store_mode_shapes)
            finally:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                atexit.unregister(_cleanup_workers)
        del samples, tasks  # ponytail: free before final save
    else:
        print("Sequential sweep")
        for i, params in enumerate(samples):
            if (i + 1) % report_interval == 0 or i == 0:
                elapsed = time_buffer[:i].sum()
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{n_samples}] solving... ({rate:.1f} pts/s)")
            result = solve_single(params, cfg, n_modes, store_mode_shapes=store_mode_shapes)

            for name in param_names:
                param_buffers[name][i] = result.params.get(name, 0)
            freq_buffer[i] = result.frequencies
            time_buffer[i] = result.solve_time
            success_buffer[i] = result.success
            if result.success:
                n_success += 1
            else:
                n_failed += 1
                error_buffer.append((i, result.error))
            if store_mode_shapes and result.mode_shapes is not None:
                ms = _downsample_mode_shapes(result.mode_shapes, mode_shape_grid) if mode_shape_grid else result.mode_shapes
                np.save(mode_shape_dir / f"{i}.npy", ms)

            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                _write_checkpoint(output_path, i + 1, param_names, param_buffers, freq_buffer, time_buffer, success_buffer, mode_shape_dir, store_mode_shapes)

    # Provenance
    provenance = {
        "git_hash": _get_git_hash(),
        "solver_version": cfg.solver_version,
        "schema_version": str(cfg.schema_version),
        "config_name": cfg.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": n_samples,
        "n_success": int(n_success),
        "n_failed": int(n_failed),
        "seed": cfg.seed,
        "n_modes": n_modes,
        "sweep_method": cfg.sweep.method,
        "param_ranges": {name: {"min": lo, "max": hi, "log_scale": log}
                         for name, lo, hi, log in ranges},
        "fixed_params": cfg.sweep.sensitivity.fixed_params,
        "total_solve_time": float(time_buffer[:n_samples].sum()),
        "store_mode_shapes": store_mode_shapes,
        "parallel": use_parallel,
        "errors": [{"index": i, "error": e} for i, e in error_buffer[:20]],
    }

    # Save final output (without mode shapes — they go in a separate file)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for name in param_names:
        save_dict[f"param_{name}"] = param_buffers[name][:n_samples]
    save_dict["frequencies"] = freq_buffer[:n_samples]
    save_dict["solve_times"] = time_buffer[:n_samples]
    save_dict["success"] = success_buffer[:n_samples]
    save_dict["provenance"] = json.dumps(provenance)
    np.savez_compressed(str(out), **save_dict)

    # Save mode shapes to separate file (pre-allocate, load into slices — no reload loop)
    mode_shape_file = out.parent / "mode_shapes.npz"
    if store_mode_shapes and mode_shape_dir is not None:
        # peek at first shape
        first = next((f for f in mode_shape_dir.iterdir() if f.suffix == '.npy'), None)
        if first is not None:
            sample_shape = np.load(first).shape  # (n_modes, *grid)
            mode_shapes_out = np.zeros((n_samples, *sample_shape))
            for i in range(n_samples):
                f = mode_shape_dir / f"{i}.npy"
                if f.exists():
                    mode_shapes_out[i] = np.load(f)
            np.savez_compressed(str(mode_shape_file), mode_shapes=mode_shapes_out)
            del mode_shapes_out
            print(f"  Mode shapes: {mode_shape_file}")
        shutil.rmtree(mode_shape_dir, ignore_errors=True)

    # Summary
    print(f"\n--- Sweep Summary ---")
    print(f"Samples: {n_samples} total, {n_success} success, {n_failed} failed")
    print(f"Total solve time: {time_buffer[:n_samples].sum():.2f}s")
    for name in param_names:
        vals = param_buffers[name][:n_samples]
        print(f"  {name}: [{vals.min():.6g}, {vals.max():.6g}]")
    print(f"Frequencies: [{freq_buffer[:n_samples].min():.2f}, {freq_buffer[:n_samples].max():.2f}] Hz")
    if store_mode_shapes and mode_shape_file.exists():
        m = np.load(str(mode_shape_file))
        print(f"Mode shapes: {m['mode_shapes'].shape}")
        del m
    if error_buffer:
        print(f"Errors (first 5):")
        for i, e in error_buffer[:5]:
            print(f"  [{i}]: {e}")
    print(f"Saved to {out}")

    return {"n_samples": n_samples, "n_success": n_success, "n_failed": n_failed, "provenance": provenance, "output_path": output_path, "run_dir": str(run_dir)}


def sobol_sample(n_base: int, n_params: int, seed: int = 42) -> np.ndarray:
    sampler_a = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    sampler_b = qmc.Sobol(d=n_params, scramble=True, seed=seed + 10000)
    A = sampler_a.random(n_base)
    B = sampler_b.random(n_base)

    max_corr = max(abs(np.corrcoef(A[:, i], B[:, i])[0, 1]) for i in range(n_params))
    if max_corr > 0.1:
        rng = np.random.default_rng(seed)
        A = rng.random((n_base, n_params))
        B = rng.random((n_base, n_params))

    blocks = [A, B]
    for i in range(n_params):
        ABi = A.copy(); ABi[:, i] = B[:, i]; blocks.append(ABi)
    for i in range(n_params):
        BAi = B.copy(); BAi[:, i] = A[:, i]; blocks.append(BAi)
    return np.vstack(blocks)


def compute_sobol_indices(Y: np.ndarray, n_base: int, n_params: int, param_names: list[str]) -> dict:
    D, N = n_params, n_base
    Y_A = Y[:N]
    Y_B = Y[N:2*N]
    Y_AB = [Y[2*N + i*N : 2*N + (i+1)*N] for i in range(D)]
    Y_BA = [Y[2*N + D*N + i*N : 2*N + D*N + (i+1)*N] for i in range(D)]

    f0 = np.mean(Y_B, axis=0)
    V = np.var(Y_B, axis=0)
    nonzero = V > 1e-30
    S1 = np.zeros((D, Y.shape[1]))
    ST = np.zeros((D, Y.shape[1]))
    V_safe = np.where(nonzero, V, 1.0)

    for i in range(D):
        S1[i, nonzero] = (np.mean(Y_A * (Y_BA[i] - Y_B), axis=0)[nonzero] / V_safe[nonzero])
        ST[i, nonzero] = (1.0 - np.mean(Y_A * (Y_AB[i] - Y_B), axis=0)[nonzero] / V_safe[nonzero])

    return {"S1": np.clip(S1, 0, 1), "ST": np.clip(ST, 0, 1), "V": V, "f0": f0, "param_names": param_names}


def run_sensitivity_analysis(cfg: ExperimentConfig | None = None, config_path: str | None = None) -> dict:
    if cfg is None:
        cfg = ExperimentConfig.from_yaml(config_path or DEFAULT_CONFIG_PATH)

    ranges = _get_param_ranges(cfg)
    param_names = [r[0] for r in ranges]
    n_params = len(ranges)
    n_base = cfg.sweep.sensitivity.n_base_samples
    n_total = n_base * (2 * n_params + 2)
    n_modes = 6
    output_path = cfg.output.path + "_sensitivity.npz"

    print(f"=== Sobol Sensitivity Analysis ===")
    print(f"N_base={n_base}, D={n_params}, total samples={n_total}")

    samples_01 = sobol_sample(n_base, n_params, cfg.seed)
    Y = np.full((n_total, n_modes), np.nan)

    report_interval = max(1, n_total // 20)
    for i in range(n_total):
        params = {}
        for j, (name, lo, hi, log_scale) in enumerate(ranges):
            u = np.clip(samples_01[i, j], np.finfo(float).eps, 1.0 - np.finfo(float).eps)
            if log_scale:
                val = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
            else:
                val = lo + u * (hi - lo)
            if name in ("M", "N"):
                val = int(round(val))
            params[name] = val
        params = _apply_fixed_params(params, cfg)

        r = solve_single(params, cfg, n_modes)
        if r.success:
            Y[i] = r.frequencies

        if (i + 1) % report_interval == 0:
            print(f"  [{i+1}/{n_total}] solved...")

    indices = compute_sobol_indices(Y, n_base, n_params, param_names)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out),
        S1=indices["S1"], ST=indices["ST"], V=indices["V"], f0=indices["f0"],
        param_names=param_names, n_base=n_base, n_total=n_total, seed=cfg.seed,
        samples_01=samples_01, Y=Y,
    )

    print(f"\n--- Sobol Sensitivity Summary ---")
    for mode_idx in range(n_modes):
        print(f"\nMode {mode_idx + 1} (baseline: {indices['f0'][mode_idx]:.2f} Hz):")
        print(f"  {'Parameter':<18} {'S1':>8} {'ST':>8}")
        print(f"  {'-'*34}")
        ranking = sorted(range(n_params), key=lambda k: indices["ST"][k, mode_idx], reverse=True)
        for k in ranking:
            print(f"  {param_names[k]:<18} {indices['S1'][k, mode_idx]:8.4f} {indices['ST'][k, mode_idx]:8.4f}")

    print(f"\nSaved to {out}")
    return {"S1": indices["S1"], "ST": indices["ST"], "param_names": param_names, "output_path": output_path}


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, DictConfig
    import argparse

    parser = argparse.ArgumentParser(description="FSDT parameter sweep data generation")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. sweep.n_samples=500")
    parser.add_argument("--mode", choices=["sweep", "sensitivity"], default="sweep")
    parser.add_argument("--config-dir", default="configs", help="Config directory")
    parser.add_argument("--profile", action="store_true", help="Profile with memray")
    args = parser.parse_args()

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base=None):
        cfg = compose(config_name="default_sweep", overrides=args.overrides)
        cfg = OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg

    if args.mode == "sensitivity":
        from mechanics.config import ExperimentConfig
        ecfg = ExperimentConfig._from_dict(cfg) if isinstance(cfg, dict) else cfg
        run_sensitivity_analysis(cfg=ecfg)
    else:
        from mechanics.config import ExperimentConfig
        ecfg = ExperimentConfig._from_dict(cfg) if isinstance(cfg, dict) else cfg
        if args.profile:
            import memray
            with memray.Tracker("memray-sweep.bin"):
                run_sweep(cfg=ecfg)
            print(f"\nProfile saved to memray-sweep.bin")
            print(f"Run: uv run python -m memray flamegraph memray-sweep.bin")
        else:
            run_sweep(cfg=ecfg)
