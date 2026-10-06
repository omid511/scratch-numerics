#!/usr/bin/env python3
"""Execute the predeclared P1 HF plan without changing its run roles."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import math
from typing import Sequence

from p1_design import file_sha256, write_json
from run_hf_batches import _parent_liveness

ROOT = Path(__file__).resolve().parent


def design_run_ids(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        id_column = next((column for column in ("run_id", "run") if column in fields), None)
        result = set()
        for row_index, row in enumerate(reader, 1):
            run_id = int(row[id_column]) if id_column else row_index
            if run_id < 1 or run_id in result:
                raise ValueError(f"Duplicate or nonpositive design run ID {run_id}")
            result.add(run_id)
    if not result:
        raise ValueError(f"Empty design table: {path}")
    return result


def read_plan(path: Path) -> dict[str, list[int]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"run_id", "role"}.issubset(rows[0]):
        raise ValueError("HF plan must contain run_id and role")
    result = {role: [] for role in ("train", "calibration", "test")}
    seen = set()
    for row in rows:
        run = int(row["run_id"])
        role = row["role"].strip().lower()
        if role not in result or run in seen:
            raise ValueError(f"Invalid or duplicate HF plan row {run}")
        result[role].append(run)
        seen.add(run)
    return result

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", type=Path,
                        default=ROOT / "run_hf_batches.py",
                        help="HF batch driver with the repository CLI contract")
    parser.add_argument("--role", choices=("train", "calibration", "test", "all"), default="all")
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--java-heap-gb", type=int,
                        help="Maximum worker JVM heap in GiB")
    parser.add_argument("--mesh-size", type=int, default=4, choices=range(1, 10))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-minutes", type=float, default=360.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--reexport", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.java_heap_gb is not None and args.java_heap_gb < 1:
        parser.error("--java-heap-gb must be positive")
    if (not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0
            or not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0):
        parser.error("--timeout-minutes and --heartbeat-seconds must be positive and finite")
    design = design_run_ids(args.samples)
    plan = read_plan(args.plan)
    selected = sorted(
        run for role, runs in plan.items()
        if args.role == "all" or role == args.role
        for run in runs
    )
    if not selected or any(run not in design for run in selected):
        parser.error("HF plan contains a run absent from the design table")
    driver = args.driver.expanduser().resolve()
    if not driver.is_file():
        parser.error(f"HF driver not found: {driver}")
    command = [
        sys.executable, str(driver),
        "--samples", str(args.samples.resolve()), "--output", str(args.output.resolve()),
        "--runs", ",".join(str(run) for run in selected),
        "--cores", str(args.cores), "--mesh-size", str(args.mesh_size),
        "--attempts", str(args.attempts),
        "--timeout-minutes", str(args.timeout_minutes),
        "--heartbeat-seconds", str(args.heartbeat_seconds),
    ]
    if args.java_heap_gb is not None:
        command.extend(["--java-heap-gb", str(args.java_heap_gb)])
    if args.scratch_root is not None:
        command.extend(["--scratch-root", str(args.scratch_root.resolve())])
    if args.reexport:
        command.append("--reexport")
    if args.replace:
        command.append("--replace")
    manifest = {
        "schema_version": 1,
        "artifact": "p1_hf_execution",
        "samples_sha256": file_sha256(args.samples),
        "plan_sha256": file_sha256(args.plan),
        "driver": str(driver),
        "output": str(args.output.resolve()),
        "role": args.role,
        "run_ids": selected,
        "command": command,
        "mesh_size": args.mesh_size,
        "cores": args.cores,
        "java_heap_gb": args.java_heap_gb,
        "timeout_minutes": args.timeout_minutes,
        "heartbeat_seconds": args.heartbeat_seconds,
        "scratch_root": str(args.scratch_root.resolve()) if args.scratch_root is not None else None,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"hf_execution_{args.role}.json", manifest, overwrite=True)
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0
    with _parent_liveness() as parent_alive:
        driver = subprocess.Popen(command, cwd=str(ROOT))
        while True:
            try:
                returncode = driver.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                if not parent_alive():
                    # Exiting lets the driver stop its worker and clean its owned scratch.
                    print("HF plan parent exited; stopping the plan.", flush=True)
                    return 130
            except KeyboardInterrupt:
                # The same parent-death guard also covers a driver outside this console.
                return 130
    if returncode:
        raise SystemExit(f"HF plan execution failed with exit code {returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
