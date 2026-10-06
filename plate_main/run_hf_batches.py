#!/usr/bin/env python3
"""Run HF samples in independent Python/COMSOL processes.

MPh owns a process-wide JPype JVM and cannot restart a dead COMSOL client in
the same interpreter. This wrapper launches one worker process per run, so a
heap failure cannot poison subsequent samples. Windows workers use an
in-process COMSOL backend so the requested Java heap is not overridden by the
server launcher INI. Valid NPZ bundles are skipped by the worker.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import os
import re
import subprocess
import sys
import time
import math
import signal
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

PARAMETERS = ('alpha', 'beta', 'theta_c', 'eta1', 'eta2')

# Use MPh's sys.exit hook; a bare SystemExit bypasses its JVM exit-code capture.
# Import the existing solver unchanged to preserve completed-bundle provenance.
WORKER_ENTRYPOINT = (
    'import sys\n'
    'import mph\n'
    'if sys.platform == "win32":\n'
    '    mph.option("session", "stand-alone")\n'
    'import time\n'
    'import hc_HighFidelity_LHS as hf\n'
    'from simulations.scripts.hc_mesh_convergence import _element_count\n'
    '_build, _extract = hf.build_model, hf.solve_and_extract\n'
    'def build(*args, **kwargs):\n'
    '    started = time.monotonic()\n'
    '    print("PHASE geometry-and-mesh started", flush=True)\n'
    '    model = _build(*args, **kwargs)\n'
    '    print(f"PHASE geometry-and-mesh finished in {time.monotonic()-started:.1f}s; elements={_element_count(model)}; eigensolve-and-save started", flush=True)\n'
    '    return model\n'
    'def extract(*args, **kwargs):\n'
    '    started = time.monotonic()\n'
    '    print("PHASE field-extraction started", flush=True)\n'
    '    result = _extract(*args, **kwargs)\n'
    '    print(f"PHASE field-extraction finished in {time.monotonic()-started:.1f}s", flush=True)\n'
    '    return result\n'
    'hf.build_model, hf.solve_and_extract = build, extract\n'
    'sys.exit(hf.main())\n'
)


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


@contextmanager
def _parent_liveness():
    parent = os.getppid()
    if os.name != 'nt':
        yield lambda: os.getppid() == parent
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, parent)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    def alive():
        status = kernel.WaitForSingleObject(handle, 0)
        if status not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return status == 258
    try:
        yield alive
    finally:
        kernel.CloseHandle(handle)


def _stop_worker(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        result = subprocess.run(
            ['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
            capture_output=True, text=True, timeout=30)
        if result.returncode and process.poll() is None:
            raise RuntimeError(result.stdout + result.stderr)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def _supervise(command, root, env, log_path, timeout_s, heartbeat_s):
    started = time.monotonic()
    next_heartbeat = started + heartbeat_s
    with _parent_liveness() as parent_alive, log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt')
        print(f'Worker PID {process.pid}; durable log: {log_path}', flush=True)
        try:
            while True:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    _stop_worker(process)
                    print(f'Worker deadline reached after {timeout_s / 60:.1f} minutes; process tree stopped.', flush=True)
                    return 124
                try:
                    return process.wait(timeout=min(5.0, remaining))
                except subprocess.TimeoutExpired:
                    if not parent_alive():
                        _stop_worker(process)
                        print('Launching parent exited; owned worker tree stopped.', flush=True)
                        return 130
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        print(f'{datetime.now(timezone.utc).isoformat()} Worker PID {process.pid} still running; elapsed {(now-started)/60:.1f} minutes; log {log_path}', flush=True)
                        next_heartbeat = now + heartbeat_s
        except BaseException:
            _stop_worker(process)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', help='Comma-separated run IDs; default is every sample')
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--mesh-size', type=int, default=4, choices=range(1, 10))
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--java-heap-gb', type=int,
                        help='Maximum worker JVM heap in GiB; overrides inherited -Xmx')
    parser.add_argument('--timeout-minutes', type=float, default=360.0,
                        help='Per-worker deadline; timeout stops its process tree without retry')
    parser.add_argument('--heartbeat-seconds', type=float, default=60.0,
                        help='Progress-report interval; parent liveness is checked at least every 5 seconds')
    parser.add_argument('--scratch-root', type=Path,
                        help='Existing local directory for a fresh, worker-owned TEMP/TMP and Java temporary directory')
    parser.add_argument('--reexport', action='store_true',
                        help='Re-export NPZ bundles from compatible saved COMSOL models')
    parser.add_argument('--replace', action='store_true',
                        help='Replace incompatible NPZ bundles in each worker')
    args = parser.parse_args(argv)
    if (args.cores < 1 or args.attempts < 1
            or not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0
            or not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0
            or (args.java_heap_gb is not None and args.java_heap_gb < 1)):
        parser.error('Resource counts, timeout, and heartbeat interval must be positive and finite')
    if args.scratch_root is not None:
        args.scratch_root = args.scratch_root.expanduser().resolve()
        if not args.scratch_root.is_dir():
            parser.error('--scratch-root must be an existing directory')
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
    worker_env = None
    if args.java_heap_gb is not None:
        worker_env = os.environ.copy()
        options = re.sub(r'(?<!\S)-Xmx\S+', '',
                         worker_env.get('JAVA_TOOL_OPTIONS', '')).strip()
        worker_env['JAVA_TOOL_OPTIONS'] = (
            f'{options} -Xmx{args.java_heap_gb}g'.strip())
    log_root = output_path / 'worker_logs'
    log_root.mkdir(parents=True, exist_ok=True)
    failed = []
    for run in runs:
        success = False
        for attempt in range(1, args.attempts + 1):
            command = [
                sys.executable, '-u', '-c', WORKER_ENTRYPOINT,
                '--samples', str(sample_path),
                '--output', str(output_path), '--runs', str(run),
                '--cores', str(args.cores), '--mesh-size', str(args.mesh_size),
                '--worker',
            ]
            if args.reexport:
                command.append('--reexport')
            if args.replace:
                command.append('--replace')
            print(f'Worker run {run}: process attempt {attempt}/{args.attempts}', flush=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            log_path = log_root / f'run_{run:04d}_{stamp}_{os.getpid()}_attempt{attempt}.log'
            scratch_context = (TemporaryDirectory(prefix='p1_hf_', dir=args.scratch_root)
                               if args.scratch_root is not None else nullcontext(None))
            with scratch_context as scratch:
                env = worker_env
                if scratch is not None:
                    env = (worker_env if worker_env is not None else os.environ).copy()
                    env['TEMP'] = env['TMP'] = scratch
                    options = env.get('JAVA_TOOL_OPTIONS', '')
                    env['JAVA_TOOL_OPTIONS'] = (
                        f'{options} -Djava.io.tmpdir="{Path(scratch).as_posix()}"'.strip())
                    print(f'Worker-owned temporary directory: {scratch}', flush=True)
                status = _supervise(command, root, env, log_path,
                                    args.timeout_minutes * 60, args.heartbeat_seconds)
            if status == 0:
                success = True
                break
            print(f'Worker run {run}: exit {status}; details in {log_path}', flush=True)
            if status == 130:
                raise SystemExit(130)
            if status == 124:
                break
        if not success:
            failed.append(run)
            print(f'Worker run {run}: failed after {attempt} fresh processes', flush=True)
    if failed:
        raise SystemExit(f'Unfinished HF runs: {failed}')
    print(f'All {len(runs)} requested runs have valid or newly-created NPZ bundles.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
