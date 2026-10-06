#!/usr/bin/env python3
"""Run large P1 LF batches in fresh resumable Python processes.

The existing ``fsdt_mode_shapes.py`` is the single-run LF solver.  This
wrapper adds deterministic batching, optional parallel subprocesses, retries,
and bundle verification so a 10,000--20,000 design sweep can be resumed after
an interruption without touching the design IDs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np

from p1_design import read_design_table, parse_run_list

ROOT = Path(__file__).resolve().parent


def design_parameter_names(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        fields = tuple(next(csv.reader(stream)))
    names = tuple(field for field in fields if field not in ("run_id", "run"))
    if not names:
        raise ValueError(f"Design table has no parameter columns: {path}")
    return names


def _bundle_valid(path: Path, run_id: int, parameters: dict[str, float], modes: int, grid: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if (metadata.get("schema_version") != 2
                    or metadata.get("fidelity") != "lf"
                    or metadata.get("run_id") != run_id
                    or metadata.get("parameters") != parameters):
                return False
            if metadata.get("modes") != modes or metadata.get("grid") != grid:
                return False
            x, y = archive["x"], archive["y"]
            w, frequency = archive["w"], archive["frequencies"]
            fractions = archive["transverse_fraction"]
            indices = archive["eigen_mode_indices"]
            return bool(
                x.shape == (grid,) and y.shape == (grid,)
                and w.shape == (modes, grid, grid)
                and frequency.shape == (modes,)
                and fractions.shape == (modes,) and indices.shape == (modes,)
                and np.isfinite(w).all() and np.isfinite(frequency).all()
                and np.isfinite(fractions).all() and np.isfinite(indices).all()
                and np.all(frequency > 0) and np.all(np.diff(frequency) >= 0)
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False

def _batch_command(
    batch: Sequence[int], samples: Path, output: Path, modes: int,
    order: int, grid: int, driver: Path,
) -> list[str]:
    return [
        sys.executable, str(driver),
        "--samples", str(samples), "--output", str(output),
        "--runs", ",".join(str(run) for run in batch),
        "--modes", str(modes), "--order", str(order), "--grid", str(grid),
    ]


def _run_batch(
    batch: Sequence[int], samples: Path, output: Path, modes: int,
    order: int, grid: int, attempts: int, driver: Path,
) -> tuple[list[int], int]:
    command = _batch_command(batch, samples, output, modes, order, grid, driver)
    for attempt in range(1, attempts + 1):
        print(f"LF batch {list(batch)}: attempt {attempt}/{attempts}", flush=True)
        try:
            result = subprocess.run(command, cwd=str(ROOT), check=False)
        except OSError as error:
            print(f"LF batch process failed to start: {error}", flush=True)
            continue
        if result.returncode == 0:
            return list(batch), 0
    return list(batch), 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", type=Path,
                        default=ROOT / "fsdt_mode_shapes.py",
                        help="Single-run LF driver with the standard batch CLI")
    parser.add_argument("--runs", help="Comma-separated run IDs; default is every design")
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--order", type=int, default=15)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent fresh LF subprocesses; start with 1")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-recompute", action="store_true",
                        help="Permit replacing invalid existing bundles")
    args = parser.parse_args(argv)
    if min(args.modes, args.order, args.grid, args.batch_size, args.workers, args.attempts) < 1:
        parser.error("modes/order/grid/batch-size/workers/attempts must be positive")
    if args.grid < 3 or args.order < 3:
        parser.error("grid must be at least 3 and order at least 3")
    samples = args.samples.expanduser().resolve()
    output = args.output.expanduser().resolve()
    driver = args.driver.expanduser().resolve()
    if not samples.is_file():
        parser.error(f"Design table not found: {samples}")
    if not driver.is_file():
        parser.error(f"LF driver not found: {driver}")
    design_rows = read_design_table(samples, design_parameter_names(samples))
    by_id = dict(design_rows)
    try:
        requested = parse_run_list(args.runs)
    except ValueError as error:
        parser.error(str(error))
    runs = requested or sorted(by_id)
    if any(run not in by_id for run in runs):
        parser.error("Requested run ID is absent from the design table")
    output.mkdir(parents=True, exist_ok=True)
    lf_root = output / "lf"
    lf_root.mkdir(parents=True, exist_ok=True)

    valid = []
    invalid = []
    for run in runs:
        parameters = by_id[run]
        path = lf_root / f"run_{run:04d}.npz"
        if _bundle_valid(path, run, parameters, args.modes, args.grid):
            valid.append(run)
        else:
            invalid.append(run)
    print(f"LF status: {len(valid)} valid, {len(invalid)} missing/incompatible", flush=True)
    if args.verify_only:
        if invalid:
            raise SystemExit(f"Invalid LF bundles: {invalid[:20]}" + (" ..." if len(invalid) > 20 else ""))
        return 0
    if invalid and not args.allow_recompute:
        existing_invalid = [run for run in invalid if (lf_root / f"run_{run:04d}.npz").exists()]
        if existing_invalid:
            raise SystemExit(
                "Incompatible LF bundles already exist; use a new output root or "
                f"--allow-recompute. Runs: {existing_invalid[:20]}"
            )
    batches = [invalid[start:start + args.batch_size] for start in range(0, len(invalid), args.batch_size)]
    failed: list[int] = []
    if batches:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_batch, batch, samples, output, args.modes,
                    args.order, args.grid, args.attempts, driver,
                ): batch for batch in batches
            }
            for future in as_completed(futures):
                batch, returncode = future.result()
                if returncode:
                    failed.extend(batch)
        if failed:
            raise SystemExit(f"Unfinished LF runs after retries: {sorted(failed)}")
    remaining = [
        run for run in runs
        if not _bundle_valid(lf_root / f"run_{run:04d}.npz", run, by_id[run], args.modes, args.grid)
    ]
    if remaining:
        raise SystemExit(f"LF verification failed after execution: {remaining[:20]}")
    print(f"All {len(runs)} requested LF bundles are valid under {output}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
