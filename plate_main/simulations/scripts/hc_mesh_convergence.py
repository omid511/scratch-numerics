#!/usr/bin/env python3
"""Run reference and extreme-point COMSOL mesh-convergence checks.

The module is import-safe: COMSOL starts only from ``main``.  The reference
case preserves the old paper comparison, while ``--extreme-runs`` checks the
production geometry at sampled parameter extremes.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_EIGS = 10

PRODUCTION_MESH_SIZE = 4


def refinement_pair(levels, production=PRODUCTION_MESH_SIZE):
    """Return production hauto and the finest requested finer hauto."""
    levels = sorted(set(levels))
    if production not in levels:
        raise ValueError(
            f'Extreme validation must include production hauto={production}')
    finer = [level for level in levels if level < production]
    if not finer:
        raise ValueError(
            f'Extreme validation needs a hauto below production {production}')
    return production, min(finer)
PAPER_FEM = np.array([
    1217.9, 2312.6, 2313.8, 3236.1, 3803.4,
    3838.7, 4594.7, 4597.6, 5575.4, 5584.7,
])
REFERENCE_SAMPLE = dict(
    alpha=0.8, beta=1.0, theta_c=30.0, eta1=1.0, eta2=0.2 / 3.0,
)


def parse_levels(text):
    try:
        levels = [int(value) for value in text.split(',') if value.strip()]
    except ValueError as exc:
        raise ValueError('Mesh levels must be comma-separated integers') from exc
    if not levels or any(level < 1 for level in levels):
        raise ValueError('Mesh levels must be positive')
    return list(dict.fromkeys(levels))


def read_samples(path):
    """Read explicit-ID or frozen implicit-row design tables."""
    from p1_geometry import PARAM_NAMES

    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    samples = {}
    for index, row in enumerate(rows, 1):
        run = int(row.get('run_id', row.get('run', index)))
        if run < 1 or run in samples:
            raise ValueError(f'Invalid or duplicate run ID {run}')
        samples[run] = {name: float(row[name]) for name in PARAM_NAMES}
    if not samples:
        raise ValueError('Sample table is empty')
    return samples


def _frequencies(model):
    """Return the first positive sorted eigenfrequencies from a solved model."""
    numerical = model.java.result().numerical()
    feature = numerical.create('mesh_freq', 'EvalGlobal')
    try:
        feature.set('data', 'dset1')
        feature.set('expr', ['freq'])
        feature.set('unit', ['Hz'])
        values = np.asarray(feature.getReal(), dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite eigenfrequencies')
        values = np.sort(values[values > 0], kind='stable')
        if len(values) < N_EIGS:
            raise ValueError(f'Expected {N_EIGS} positive modes, received {len(values)}')
        return values[:N_EIGS]
    finally:
        numerical.remove('mesh_freq')


def _element_count(model):
    try:
        stats = model.java.component('comp1').mesh('mesh1').stat()
        return int(str(stats.getNumElem()))
    except Exception:
        return None


def run_case(client, parameters, levels, save_models=None):
    """Run one geometry at each mesh level and return serializable records."""
    from hc_HighFidelity_LHS import EIGEN_SHIFT_HZ, build_model
    from p1_geometry import physical_parameters

    physical = physical_parameters(parameters)
    records = []
    for level in levels:
        model = None
        started = time.monotonic()
        record = dict(hauto=level, num_elements=None, mesh_time_s=None,
                      solve_time_s=None, frequencies=None, error=None)
        try:
            model = build_model(
                client, physical, mesh_size=level, n_eigs=N_EIGS,
                candidate_eigs=N_EIGS, eigen_shift_hz=EIGEN_SHIFT_HZ)
            record['mesh_time_s'] = time.monotonic() - started
            solve_started = time.monotonic()
            model.java.study('std1').run()
            record['solve_time_s'] = time.monotonic() - solve_started
            record['num_elements'] = _element_count(model)
            record['frequencies'] = _frequencies(model)
            if save_models is not None:
                target = Path(save_models) / f'hauto_{level}'
                target.mkdir(parents=True, exist_ok=True)
                model.save(str(target / 'model.mph'))
        except Exception as exc:
            record['error'] = str(exc)
        finally:
            if model is not None:
                try:
                    client.remove(model)
                except Exception:
                    pass
        records.append(record)
    return records


def _valid_records(records):
    return [record for record in records if record['frequencies'] is not None]


def write_convergence_csv(path, records, paper=PAPER_FEM):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['hauto', 'num_elements', 'mesh_time_s', 'solve_time_s']
    fields += [f'freq_mode{mode}' for mode in range(1, N_EIGS + 1)]
    fields += [f'error_mode{mode}_pct' for mode in range(1, N_EIGS + 1)]
    fields.append('error')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            frequencies = record['frequencies']
            if frequencies is not None:
                errors = 100 * np.abs(frequencies - paper) / paper
                row.update({f'freq_mode{mode}': float(value)
                            for mode, value in enumerate(frequencies, 1)})
                row.update({f'error_mode{mode}_pct': float(value)
                            for mode, value in enumerate(errors, 1)})
            writer.writerow(row)


def write_extreme_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['run_id', 'production_mesh', 'finer_mesh', 'max_relative_change',
              'status', 'error']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extreme_validation(client, samples, run_ids, levels, tolerance, output_dir):
    """Compare production hauto=4 to the finest requested finer hauto."""
    from p1_geometry import extreme_run_ids

    selected = extreme_run_ids(samples)
    if run_ids:
        selected = selected[:run_ids]
    production, finer = refinement_pair(levels)
    rows = []
    for run_id in selected:
        records = run_case(client, samples[run_id], [production, finer])
        valid = {record['hauto']: record for record in records
                 if record['frequencies'] is not None}
        row = dict(run_id=run_id, production_mesh=production,
                   finer_mesh=finer, max_relative_change=None,
                   status='failed', error=None)
        if production in valid and finer in valid:
            relative = np.abs(valid[production]['frequencies']
                              - valid[finer]['frequencies'])
            relative /= np.maximum(np.abs(valid[finer]['frequencies']),
                                   np.finfo(float).tiny)
            row['max_relative_change'] = float(np.max(relative))
            row['status'] = ('pass' if row['max_relative_change'] <= tolerance
                             else 'fail')
        else:
            row['error'] = '; '.join(record['error'] or 'unknown failure'
                                     for record in records)
        rows.append(row)
    write_extreme_csv(Path(output_dir) / 'mesh_extreme_validation.csv', rows)
    failures = [row for row in rows if row['status'] != 'pass']
    if failures:
        raise RuntimeError(f'Extreme mesh validation failed: {failures}')
    return rows


def plot_convergence(path, records, paper=PAPER_FEM):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = _valid_records(records)
    if not valid:
        raise ValueError('No valid mesh records to plot')
    levels = [record['hauto'] for record in valid]
    frequencies = np.array([record['frequencies'] for record in valid])
    elements = [record['num_elements'] or np.nan for record in valid]
    colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'olive']
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for mode in range(N_EIGS):
        axes[0].plot(levels, frequencies[:, mode], color=colors[mode],
                     marker=markers[mode], label=f'Mode {mode + 1}')
        axes[0].axhline(paper[mode], color=colors[mode], linestyle='--', alpha=.4)
        axes[1].plot(elements, frequencies[:, mode], color=colors[mode],
                     marker=markers[mode], label=f'Mode {mode + 1}')
        axes[1].axhline(paper[mode], color=colors[mode], linestyle='--', alpha=.4)
    axes[0].set(xlabel='hauto level (1=fine, 9=coarse)', ylabel='Eigenfrequency (Hz)',
                title='Reference mesh convergence')
    axes[1].set(xlabel='Number of elements', ylabel='Eigenfrequency (Hz)',
                title='Frequency versus element count')
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend(fontsize=8, loc='center right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=root / 'lhs_samples_v2.csv')
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('--levels', default='1,2,3,4,5,6,7,8,9')
    parser.add_argument('--extreme-levels', default='4,3')
    parser.add_argument('--extreme-runs', type=int, default=0,
                        help='Check this many deterministic parameter-extreme runs')
    parser.add_argument('--extreme-tolerance', type=float, default=2e-4,
                        help='Maximum relative frequency change between production and finer meshes')
    parser.add_argument('--output', type=Path, default=root)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    if args.cores < 1 or args.extreme_runs < 0 or args.extreme_tolerance < 0:
        parser.error('Invalid cores, extreme-run count, or tolerance')
    levels = parse_levels(args.levels)
    extreme_levels = parse_levels(args.extreme_levels)
    samples = read_samples(args.samples)

    import mph
    args.output.mkdir(parents=True, exist_ok=True)
    client = None
    try:
        client = mph.start(cores=args.cores)
        print('Running fixed reference mesh convergence')
        reference = run_case(client, REFERENCE_SAMPLE, levels,
                             save_models=args.output / 'reference')
        write_convergence_csv(args.output / 'mesh_convergence_final.csv', reference)
        if not args.no_plot:
            plot_convergence(args.output / 'mesh_convergence_final.png', reference)
        valid = _valid_records(reference)
        if valid:
            finest = min(valid, key=lambda record: record['hauto'])
            errors = 100 * np.abs(finest['frequencies'] - PAPER_FEM) / PAPER_FEM
            print(f"Reference finest hauto={finest['hauto']}; mean paper error={errors.mean():.3f}%")
        if args.extreme_runs:
            print(f'Running {args.extreme_runs} deterministic extreme designs')
            extreme_validation(client, samples, args.extreme_runs, extreme_levels,
                               args.extreme_tolerance, args.output)
            print('Extreme mesh validation passed')
    finally:
        if client is not None:
            try:
                client.clear()
            except Exception:
                pass
            try:
                client.disconnect()
            except Exception:
                pass


if __name__ == '__main__':
    main()
