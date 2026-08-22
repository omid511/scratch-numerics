#!/usr/bin/env python3
"""Render the expanded P4 training results JSON into a markdown report.

Companion to ``train_p4_expanded.py`` (expanded 160-design dataset with
design-level splits). Produces ``P4_REPORT_EXPANDED.md`` in the style of the
legacy single-design ``P4_REPORT.md``, so numbers are directly comparable.

Usage:
    uv run python report_p4_expanded.py [--results p4_train_results.json] \
        [--out P4_REPORT_EXPANDED.md]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Model families in canonical display order (any extra models follow).
FAMILY_ORDER = [
    "quantile", "huber", "median",
    "constant_median", "velocity_linear", "growth_rate", "physics_ridge",
    "gru",
]

# Scalar metric keys rendered in each model detail block, beyond
# mae / mean_interval_width / coverage. All optional; absent -> omitted.
SCALAR_FIELDS = [
    ("false_safe_rate", "False-safe rate (median)"),
    ("false_safe_incidence", "False-safe incidence"),
    ("lower_bound_false_safe_rate", "Lower-bound false-safe rate"),
    ("false_unsafe_rate", "False-unsafe rate"),
    ("monotonicity_violation_rate", "Monotonicity violation rate"),
    ("near_flutter_mae", "Near-flutter MAE (margin<0.15)"),
    ("signed_bias_near_flutter", "Signed bias near flutter"),
]


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fmt(v, spec: str = ".4f") -> str:
    """Format a metric; 'n/a' for missing keys, non-numbers, NaN/Inf."""
    if v is None or not _is_num(v):
        return "n/a"
    return format(v, spec)


def get(d, *keys):
    """First present key in d, else None."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None


def sort_models(names: list[str]) -> list[str]:
    def key(name: str):
        base = name.rsplit("_s", 1)[0]
        try:
            seed = int(name.rsplit("_s", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        fam = FAMILY_ORDER.index(base) if base in FAMILY_ORDER else len(FAMILY_ORDER)
        return (fam, seed, name)

    return sorted(names, key=key)


def split_models(results: dict) -> tuple[dict[str, dict], dict]:
    """Separate model entries from top-level metadata blocks."""
    models, meta = {}, {}
    for k, v in results.items():
        if isinstance(v, dict) and any(
            kk in v for kk in ("mae", "coverage", "mean_interval_width")
        ):
            models[k] = v
        else:
            meta[k] = v
    return models, meta


def render_provenance(meta: dict) -> list[str]:
    lines = ["## Dataset Provenance", ""]
    flat: list[tuple[str, object]] = []

    def flatten(prefix: str, obj: dict) -> None:
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                flatten(key, v)
            elif isinstance(v, list) and len(v) > 8:
                flat.append((key, f"{len(v)} entries"))
            else:
                flat.append((key, json.dumps(v) if isinstance(v, (dict, list)) else v))

    flatten("", meta)

    if flat:
        lines += ["| Field | Value |", "|-------|-------|"]
        for k, v in flat:
            lines.append(f"| {k} | {v} |")
        lines.append("")
    else:
        lines += ["No dataset metadata recorded in the results file.", ""]
    return lines


def render_comparison(models: dict, names: list[str]) -> list[str]:
    lines = [
        "## Results Summary (test split, design-level)",
        "",
        "| Model | MAE | Mean Interval Width | Interval Coverage (q05-q95) |",
        "|-------|-----|---------------------|------------------------------|",
    ]
    cov_key = get_cov_key(models)
    for name in names:
        m = models[name]
        cov = m.get("coverage") or {}
        lines.append(
            f"| {name} | {fmt(m.get('mae'))} "
            f"| {fmt(m.get('mean_interval_width'))} "
            f"| {fmt(cov.get(cov_key), '.3f') if cov_key else 'n/a'} |"
        )
    lines.append("")
    if cov_key:
        lines.append(f"(Coverage column: `{cov_key}`.)")
        lines.append("")
    return lines


def get_cov_key(models: dict) -> str | None:
    """Pick the interval-coverage key shared by quantile models."""
    for m in models.values():
        cov = m.get("coverage")
        if isinstance(cov, dict):
            for k in sorted(cov):
                if k.startswith("interval_"):
                    return k
    return None


def render_model_details(models: dict, names: list[str]) -> list[str]:
    lines = ["## Per-Model Details", ""]
    for name in names:
        m = models[name]
        lines.append(f"### {name}")
        lines.append("")
        lines += ["| Metric | Value |", "|--------|-------|"]
        lines.append(f"| MAE | {fmt(m.get('mae'))} |")
        lines.append(f"| Mean interval width | {fmt(m.get('mean_interval_width'))} |")

        # Quantile coverages (q0.05 / q0.50 / q0.95, ...)
        cov = m.get("coverage") or {}
        for k in sorted(cov):
            if k.startswith("interval_"):
                label = f"Coverage {k.removeprefix('interval_').replace('_', '–')}"
                spec = ".3f"
            else:
                label = k.replace("_", " ")
                spec = ".3f" if k.startswith("q") else ".4f"
            lines.append(f"| {label} | {fmt(cov[k], spec)} |")

        # Optional CQR calibration block
        cqr = get(m, "cqr", "cqr_adjustment")
        if isinstance(cqr, dict):
            lines.append(
                f"| CQR adjusted coverage | {fmt(get(cqr, 'coverage', 'cqr_coverage'), '.3f')} |"
            )
            lines.append(
                f"| CQR interval width | {fmt(get(cqr, 'width', 'cqr_width'))} |"
            )
        else:
            ccw = get(m, "cqr_coverage"), get(m, "cqr_width")
            if any(_is_num(x) for x in ccw):
                lines.append(f"| CQR adjusted coverage | {fmt(ccw[0], '.3f')} |")
                lines.append(f"| CQR interval width | {fmt(ccw[1])} |")

        # Safety-critical scalars, monotonicity, etc.
        for key, label in SCALAR_FIELDS:
            if key in m:
                lines.append(f"| {label} | {fmt(m[key])} |")

        # Bootstrap CI on MAE, where present
        ci = get(m, "mae_bootstrap_ci", "bootstrap_mae_ci", "mae_ci")
        boot = m.get("bootstrap")
        if _is_list_pair(ci):
            lines.append(f"| MAE design-bootstrap 95% CI | [{fmt(ci[0])}, {fmt(ci[1])}] |")
        elif isinstance(boot, dict):
            lo = get(boot, "ci_low", "lo", "low")
            hi = get(boot, "ci_high", "hi", "high")
            obs = get(boot, "observed", "observed_mae")
            if _is_list_pair(boot.get("mae_ci")):
                lines.append(f"| MAE design-bootstrap 95% CI | "
                             f"[{fmt(boot['mae_ci'][0])}, {fmt(boot['mae_ci'][1])}] |")
            elif _is_num(lo) and _is_num(hi):
                lines.append(f"| MAE design-bootstrap 95% CI | [{fmt(lo)}, {fmt(hi)}]"
                             + (f" (observed {fmt(obs)})" if _is_num(obs) else "") + " |")

        lines.append("")

        pv = m.get("per_velocity")
        if isinstance(pv, dict) and pv:
            has_cov = any(isinstance(e.get("coverage"), dict) and e["coverage"]
                          for e in pv.values())
            head = "| Velocity (m/s) | MAE |" + (" Coverage (q05-q95) |" if has_cov else "")
            sep = "|---------------|-----|" + ("--------------------|" if has_cov else "")
            lines += [head, sep]
            for v in sorted(pv, key=lambda x: float(x)):
                e = pv[v] or {}
                row = f"| {v} | {fmt(e.get('mae'))} |"
                if has_cov:
                    row += f" {_vel_cov(e.get('coverage') or {})} |"
                lines.append(row)
            lines.append("")
    return lines


def _vel_cov(vcov: dict) -> str:
    """Interval coverage if present, else the q05/q95 pair."""
    ivals = [k for k in vcov if k.startswith("interval_")]
    if ivals:
        return fmt(vcov[ivals[0]], ".3f")
    lo, hi = vcov.get("q0.05"), vcov.get("q0.95")
    if _is_num(lo) and _is_num(hi):
        return f"{lo:.3f} / {hi:.3f}"
    return "n/a"


def _is_list_pair(v) -> bool:
    return (
        isinstance(v, (list, tuple)) and len(v) == 2
        and all(_is_num(x) for x in v)
    )


def render_per_velocity_matrix(models: dict, names: list[str]) -> list[str]:
    """Velocity rows x model columns of per-velocity MAE."""
    cols = [n for n in names if isinstance(models[n].get("per_velocity"), dict)
            and models[n]["per_velocity"]]
    if not cols:
        return []
    velocities = sorted(
        {str(v) for n in cols for v in models[n]["per_velocity"]},
        key=float,
    )
    lines = [
        "## Per-Velocity MAE Across Models",
        "",
        "| Velocity (m/s) | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for v in velocities:
        cells = [fmt((models[n]["per_velocity"].get(v) or {}).get("mae"))
                 for n in cols]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render(results_path: Path, out_path: Path) -> None:
    with open(results_path) as f:
        results = json.load(f)
    if not isinstance(results, dict):
        raise SystemExit(f"{results_path}: expected a top-level object, "
                         f"got {type(results).__name__}")

    models, meta = split_models(results)
    names = sort_models(list(models))

    lines = [
        "# P4 Aeroelastic Margin Estimation — Expanded-Dataset Report",
        "",
        "> **Generated from the expanded 160-design dataset** (design-level",
        "> train/val/test splits, domain-randomized sensor channels with",
        "> validity masks). This extends the legacy single-design",
        "> `P4_REPORT.md`; numbers are **not** directly comparable to it.",
        "",
        f"- Results file: `{results_path}`",
        "- Models trained by: `train_p4_expanded.py`",
        "",
    ]

    lines += render_provenance(meta)
    if names:
        lines += render_comparison(models, names)
        lines += render_model_details(models, names)
        lines += render_per_velocity_matrix(models, names)
    else:
        lines += ["No per-model results found in the results file.", ""]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="p4_train_results.json",
                    help="Path to the training results JSON")
    ap.add_argument("--out", default="P4_REPORT_EXPANDED.md",
                    help="Output markdown path")
    args = ap.parse_args()
    render(Path(args.results), Path(args.out))


if __name__ == "__main__":
    main()
