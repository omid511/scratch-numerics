#!/usr/bin/env python3
"""Build an immutable correction revision from a declared HF execution plan."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

from compute_corrections import build_targets


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_runs(path: Path, role: str) -> list[int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "run_id" not in rows[0]:
        raise ValueError("HF plan must contain run_id")
    selected = [
        int(row["run_id"])
        for row in rows
        if role == "all" or row.get("role") == role
    ]
    selected = sorted(set(selected))
    if not selected:
        raise ValueError(f"HF plan contains no runs for role {role!r}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Versioned root containing lf/ and hf/")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", choices=("pilot-five-variable", "full-proposal"),
                        default="full-proposal")
    parser.add_argument("--role", choices=("all", "train", "calibration", "test"),
                        default="all")
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--reference-run", type=int)
    parser.add_argument("--min-mac", type=float, default=.8)
    parser.add_argument("--min-margin", type=float, default=.05)
    parser.add_argument("--degeneracy-gap", type=float, default=.005)
    parser.add_argument("--min-hf-transverse-fraction", type=float, default=.5)
    parser.add_argument("--leave-p-out", type=int, default=1)
    args = parser.parse_args(argv)

    runs = plan_runs(args.plan, args.role)
    manifest, directory = build_targets(
        root=args.data,
        name=args.name,
        modes=args.modes,
        reference_run=args.reference_run,
        min_mac=args.min_mac,
        min_margin=args.min_margin,
        degeneracy_gap=args.degeneracy_gap,
        min_hf_transverse_fraction=args.min_hf_transverse_fraction,
        leave_p_out=args.leave_p_out,
        runs=runs,
        scope=args.scope,
    )
    manifest["hf_plan"] = {
        "path": args.plan.name,
        "sha256": file_sha256(args.plan),
        "role_selected": args.role,
        "run_ids": runs,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Accepted {manifest['accepted']}; quarantined {manifest['quarantined']}. Output: {directory}")
    if not manifest["accepted"]:
        raise SystemExit("No training targets accepted; inspect pairing.csv. Do not train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
