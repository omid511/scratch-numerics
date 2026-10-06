#!/usr/bin/env python3
"""Generate a proposal-scope LHS after physical parameter definitions are supplied.

The repository does not define the units, ranges, or meanings of k1--k5, nor
which face/core thicknesses are independent.  This generator therefore
requires an explicit JSON schema instead of inventing values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from p1_design import (
    PROPOSAL_PARAMETER_NAMES, file_sha256, write_design_table, write_json,
)


def read_schema(path: Path) -> list[dict[str, object]]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    parameters = schema.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("Schema must contain a parameters list")
    names = tuple(item.get("name") for item in parameters if isinstance(item, dict))
    if names != PROPOSAL_PARAMETER_NAMES:
        raise ValueError(
            f"Schema names must be exactly {PROPOSAL_PARAMETER_NAMES}; received {names}"
        )
    for item in parameters:
        if not isinstance(item.get("unit"), str) or not item["unit"].strip():
            raise ValueError(f"Parameter {item['name']} needs an explicit unit")
        lo, hi = float(item.get("min")), float(item.get("max"))
        if not np.isfinite([lo, hi]).all() or not lo < hi:
            raise ValueError(f"Parameter {item['name']} needs finite min < max")
    return parameters


def generate(schema: list[dict[str, object]], count: int, seed: int):
    unit = qmc.LatinHypercube(d=len(schema), seed=seed).random(count)
    rows = []
    for index, sample in enumerate(unit, 1):
        values = {}
        for coordinate, item in zip(sample, schema):
            lo, hi = float(item["min"]), float(item["max"])
            values[str(item["name"])] = lo + float(coordinate) * (hi - lo)
        rows.append((index, values))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("count must be positive")
    schema = read_schema(args.schema)
    rows = generate(schema, args.count, args.seed)
    write_design_table(args.output, rows, PROPOSAL_PARAMETER_NAMES)
    manifest = {
        "schema_version": 1,
        "artifact": "p1_design",
        "scope": "full-proposal",
        "parameter_names": list(PROPOSAL_PARAMETER_NAMES),
        "parameter_definitions": schema,
        "count": args.count,
        "seed": args.seed,
        "design_sha256": file_sha256(args.output),
        "schema_sha256": file_sha256(args.schema),
        "simulation_adapter_status": {
            "lf": "requires explicit adapter from proposal variables to FSDT solver",
            "hf": "requires explicit Abaqus/COMSOL model adapter",
        },
    }
    write_json(args.manifest or args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
