#!/usr/bin/env python3
"""Run HF samples in independent Python/COMSOL processes.

MPh owns a process-wide JPype JVM and cannot restart a dead COMSOL client in
the same interpreter. This wrapper launches one worker process per run, so a
heap failure cannot poison subsequent samples. Valid NPZ bundles are skipped
by the worker.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import subprocess
import sys

PARAMETERS = ('alpha', 'beta', 'theta_c', 'eta1', 'eta2')


def read_ids(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not set(PARAMETERS).issubset(rows[0]):
        raise ValueError('Sample CSV must contain the five P1 parameters')
    result = []
    seen = set()
    for index, row in enumerate(rows, 1):
        run = int(row.get('run_id', row.get('run', index)))
        if run < 1 or run in seen:
            raise ValueError(f'Invalid or duplicate run ID {run}')
        result.append(run); seen.add(run)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', help='Comma-separated run IDs; default is every sample')
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--mesh-size', type=int, default=4, choices=range(1, 10))
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--reexport', action='store_true',
                        help='Re-export NPZ bundles from compatible saved COMSOL models')
    parser.add_argument('--replace', action='store_true',
                        help='Replace incompatible NPZ bundles in each worker')
    args = parser.parse_args(argv)
    if args.cores < 1 or args.attempts < 1:
        parser.error('--cores and --attempts must be positive')
    sample_path = args.samples.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not sample_path.is_file():
        parser.error(f'Sample table not found: {sample_path}')
    available = set(read_ids(sample_path))
    try:
        runs = (list(dict.fromkeys(int(v) for v in args.runs.split(',')))
                if args.runs else sorted(available))
    except ValueError:
        parser.error('--runs must contain comma-separated integers')
    if not runs or any(run not in available for run in runs):
        parser.error('Invalid run IDs')
    worker = root / 'hc_HighFidelity_LHS.py'
    failed = []
    for run in runs:
        success = False
        for attempt in range(1, args.attempts + 1):
            command = [
                sys.executable, str(worker), '--samples', str(sample_path),
                '--output', str(output_path), '--runs', str(run),
                '--cores', str(args.cores), '--mesh-size', str(args.mesh_size),
                '--worker',
            ]
            if args.reexport:
                command.append('--reexport')
            if args.replace:
                command.append('--replace')
            print(f'Worker run {run}: process attempt {attempt}/{args.attempts}', flush=True)
            result = subprocess.run(command, cwd=str(root))
            if result.returncode == 0:
                success = True
                break
        if not success:
            failed.append(run)
            print(f'Worker run {run}: failed after {args.attempts} fresh processes', flush=True)
    if failed:
        raise SystemExit(f'Unfinished HF runs: {failed}')
    print(f'All {len(runs)} requested runs have valid or newly-created NPZ bundles.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
