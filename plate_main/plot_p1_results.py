#!/usr/bin/env python3
"""Create publication-ready diagnostics from audited P1 correction bundles.

Accepted-target diagnostics and strict HF test results are labelled separately.
Optional holdout and surrogate inputs reuse cached predictions and CV metrics;
this script never runs simulations or fits models. Fields are normalized shapes,
not dimensional displacements.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
from p1_design import file_sha256, read_design_table, write_json
from p1_holdout import Z_BY_COVERAGE
from p1_surrogate import P1Dataset, evaluate_predictions, write_records
mpl.use('Agg')
import matplotlib.pyplot as plt


mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, path):
    """Save a high-resolution raster and a vector copy for paper production."""
    path = Path(path)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def _error_summary(records):
    result = {
        'rows': len(records),
        'runs': len({int(row['run']) for row in records}),
    }
    for key in ('field_interior_rms', 'frequency_abs_error_pct'):
        values = np.asarray([float(row[key]) for row in records if key in row])
        if len(values):
            result.update({
                f'{key}_median': float(np.median(values)),
                f'{key}_p95': float(np.quantile(values, .95)),
                f'{key}_max': float(np.max(values)),
            })
    return result


def _holdout_results(dataset, prediction, report):
    keys = list(zip(prediction['run_ids'].astype(int), prediction['mode_ids'].astype(int)))
    lookup = dict(zip(zip(dataset.run_ids, dataset.mode_ids), range(dataset.n_rows)))
    expected = {
        key for key in lookup if int(key[0]) in report['role_runs']['test']
    }
    if not keys or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Holdout predictions must contain each accepted test pair exactly once')
    rows = np.asarray([lookup[key] for key in keys], dtype=int)
    if (not np.array_equal(prediction['f_hf_true'], dataset.f_hf[rows])
            or not np.array_equal(prediction['correction_true'], dataset.correction[rows])):
        raise ValueError('Cached holdout truth does not match the target revision')
    level = report.get('calibrated_coverage')
    if level not in (90, 95):
        raise ValueError('Use a corrected holdout report declaring calibrated_coverage')
    z = Z_BY_COVERAGE[level]
    raw = {
        'correction': prediction['correction_raw'],
        'correction_std': prediction['correction_raw_std'],
        'frequency': prediction['f_hf_raw'],
        'frequency_std': prediction['f_hf_raw_std'],
    }
    baseline = {
        'correction': np.zeros_like(raw['correction']), 'correction_std': None,
        'frequency': dataset.f_lf[rows], 'frequency_std': np.full(len(rows), np.nan),
    }
    records = {}
    for model, values in (('fsdt', baseline), ('latent_gp', raw)):
        records[model], _ = evaluate_predictions(dataset, rows, values, model, 0)
        for record in records[model]:
            record['frequency_abs_error_pct'] = abs(record['frequency_error_pct'])
    runs = dataset.run_ids[rows]
    field_error = np.abs(raw['correction'][:, dataset.interior_mask]
                         - dataset.correction[rows][:, dataset.interior_mask])
    frequency_error = np.abs(raw['frequency'] - dataset.f_hf[rows])
    uncertainty = {'nominal_coverage_pct': level}
    for label, field_std, frequency_std in (
        ('raw', prediction['correction_raw_std'], prediction['f_hf_raw_std']),
        ('calibrated', prediction['correction_calibrated_std'], prediction['f_hf_calibrated_std']),
    ):
        if (field_std.shape != raw['correction'].shape or frequency_std.shape != (len(rows),)
                or not np.isfinite(field_std).all() or not np.isfinite(frequency_std).all()
                or np.any(field_std < 0) or np.any(frequency_std < 0)):
            raise ValueError('Cached uncertainty must have matching shapes and finite nonnegative scales')
        field_half_width = z * np.maximum(field_std[:, dataset.interior_mask], 1e-12)
        frequency_half_width = z * np.maximum(frequency_std, 1e-12)
        field_covered = field_error <= field_half_width
        frequency_covered = frequency_error <= frequency_half_width
        unique_runs = np.unique(runs)
        field_run_count = sum(bool(field_covered[runs == run].all()) for run in unique_runs)
        frequency_run_count = sum(bool(frequency_covered[runs == run].all()) for run in unique_runs)
        uncertainty[label] = {
            'field_pointwise_coverage': float(field_covered.mean()),
            'frequency_mode_pair_coverage': float(frequency_covered.mean()),
            'field_simultaneous_run_coverage': field_run_count / len(unique_runs),
            'frequency_simultaneous_run_coverage': frequency_run_count / len(unique_runs),
            'field_covered_runs': field_run_count,
            'frequency_covered_runs': frequency_run_count,
            'evaluated_runs': len(unique_runs),
            'mean_field_interval_width': float(2 * field_half_width.mean()),
            'mean_frequency_interval_width_hz': float(2 * frequency_half_width.mean()),
        }
    improved = {
        quantity: sum(corrected[key] < low[key] for low, corrected in
                      zip(records['fsdt'], records['latent_gp']))
        for quantity, key in (('field', 'field_interior_rms'), ('frequency', 'frequency_abs_error_pct'))
    }
    summary = {
        'evaluation': 'strict predeclared HF test; accepted mode pairs only',
        'test_runs': int(len(np.unique(runs))),
        'test_mode_pairs': len(rows),
        'models': {model: _error_summary(values) for model, values in records.items()},
        'improved_mode_pairs': improved,
        'uncertainty': uncertainty,
    }
    return rows, records, summary


def _write_holdout_results(dataset, corr, holdout, surrogate, out):
    report = json.loads((holdout / 'holdout_metrics.json').read_text(encoding='utf-8'))
    if report['data_sha256'] != file_sha256(corr / 'training.npz'):
        raise ValueError('Holdout report belongs to a different target revision')
    with np.load(holdout / 'holdout_predictions.npz', allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    rows, records, summary = _holdout_results(dataset, prediction, report)
    write_records(out / 'test_metrics.csv', records['fsdt'] + records['latent_gp'])
    by_mode = [
        {'model': model, 'mode': int(mode),
         **_error_summary([row for row in values if row['mode'] == mode])}
        for model, values in records.items() for mode in np.unique(dataset.mode_ids[rows])
    ]
    write_records(out / 'test_by_mode.csv', by_mode)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    truth = dataset.f_hf[rows]
    lo = min(float(dataset.f_lf[rows].min()), float(prediction['f_hf_raw'].min()), float(truth.min()))
    hi = max(float(dataset.f_lf[rows].max()), float(prediction['f_hf_raw'].max()), float(truth.max()))
    for ax, frequencies, label in zip(axes, (dataset.f_lf[rows], prediction['f_hf_raw']), ('FSDT', 'Latent GP')):
        ax.scatter(truth, frequencies, c=dataset.mode_ids[rows], cmap='tab10', s=18, alpha=.75)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set(xlabel='COMSOL frequency (Hz)', ylabel=f'{label} frequency (Hz)', title=f'Strict HF test: {label}')
        ax.grid(alpha=.2)
    save_figure(fig, out / 'test_frequency_parity.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    modes = np.unique(dataset.mode_ids[rows])
    for ax, key, ylabel in zip(axes, ('frequency_abs_error_pct', 'field_interior_rms'),
                              ('Absolute frequency error (%)', 'Normalized interior field RMS')):
        for offset, (model, color, label) in zip((-.18, .18), (
            ('fsdt', '#c46b6b', 'FSDT'), ('latent_gp', '#4c78a8', 'Latent GP'),
        )):
            box = ax.boxplot(
                [[row[key] for row in records[model] if row['mode'] == mode] for mode in modes],
                positions=np.arange(len(modes)) + offset, widths=.3, patch_artist=True,
                showfliers=True, manage_ticks=False)
            for patch in box['boxes']:
                patch.set_facecolor(color)
            ax.plot([], [], color=color, lw=8, label=label)
        ax.set_xticks(np.arange(len(modes)), [str(mode) for mode in modes])
        ax.set(xlabel='Reference mode', ylabel=ylabel, title='Strict HF test errors')
        ax.grid(axis='y', alpha=.2); ax.legend()
    save_figure(fig, out / 'test_error_by_mode.png')

    uncertainty = summary['uncertainty']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, quantity, point_key, title in (
        (axes[0], 'field', 'field_pointwise_coverage', 'Normalized fields'),
        (axes[1], 'frequency', 'frequency_mode_pair_coverage', 'Frequencies'),
    ):
        for offset, label, color in ((-.18, 'raw', '#c46b6b'), (.18, 'calibrated', '#4c78a8')):
            values = [100 * uncertainty[label][point_key],
                      100 * uncertainty[label][f'{quantity}_simultaneous_run_coverage']]
            ax.bar(np.arange(2) + offset, values, width=.34, label=label, color=color)
        ax.axhline(uncertainty['nominal_coverage_pct'], color='k', ls='--', lw=1)
        ax.set_xticks(np.arange(2), ['Pointwise / mode pair', 'All pairs in a run'])
        ax.set(ylabel='Empirical coverage (%)', ylim=(0, 105), title=f'Strict HF test: {title}')
        ax.grid(axis='y', alpha=.2); ax.legend()
    save_figure(fig, out / 'test_uncertainty_coverage.png')

    errors = np.asarray([row['field_interior_rms'] for row in records['latent_gp']])
    ordered = np.argsort(errors)
    examples = (ordered[len(ordered) // 2], ordered[-1])
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    extent = [float(dataset.x[0]), float(dataset.x[-1]), float(dataset.y[0]), float(dataset.y[-1])]
    for row_axes, local, case in zip(axes, examples, ('Median-error case', 'Worst-error case')):
        row = rows[local]
        corrected = dataset.lf[row] + prediction['correction_raw'][local]
        fields = (dataset.hf[row], dataset.lf[row], corrected, corrected - dataset.hf[row])
        shape_scale = max(float(np.max(np.abs(field))) for field in fields[:3])
        for ax, field, label in zip(row_axes, fields, ('COMSOL', 'FSDT', 'Corrected FSDT', 'Corrected − COMSOL')):
            scale = max(float(np.max(np.abs(field))) if label == 'Corrected − COMSOL' else shape_scale, 1e-12)
            image = ax.imshow(field, origin='lower', extent=extent, aspect='equal',
                              cmap='RdBu_r', vmin=-scale, vmax=scale)
            ax.set(title=f'{label}\nrun {dataset.run_ids[row]}, mode {dataset.mode_ids[row]}',
                   xlabel='x (m)', ylabel='y (m)')
            fig.colorbar(image, ax=ax, shrink=.8)
        row_axes[0].set_ylabel(f'{case}\ny (m)')
    fig.suptitle('Strict HF test: normalized mode shapes (not dimensional displacement)')
    save_figure(fig, out / 'test_shape_examples.png')

    if surrogate is not None:
        cv = read_csv(surrogate / 'oof_metrics.csv')
        expected = {
            (int(run), int(mode)) for run, mode in zip(dataset.run_ids, dataset.mode_ids)
            if int(run) in report['role_runs']['train']
        }
        cv_summary = {}
        for model in sorted({row['model'] for row in cv}):
            values = [row for row in cv if row['model'] == model]
            keys = [(int(row['run']), int(row['mode'])) for row in values]
            if len(set(keys)) != len(keys) or set(keys) != expected:
                raise ValueError('CV metrics must cover only the frozen accepted training pairs')
            for row in values:
                if row.get('frequency_error_pct'):
                    row['frequency_abs_error_pct'] = abs(float(row['frequency_error_pct']))
            cv_summary[model] = _error_summary(values)
        write_records(out / 'cv_model_comparison.csv', [
            {'model': model, **values} for model, values in cv_summary.items()
        ])
        summary['training_run_cv'] = {
            'evaluation': 'grouped training-run CV; NOT the strict HF test',
            'models': cv_summary,
        }
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        names = list(cv_summary)
        for offset, statistic, label in ((-.18, 'median', 'Median'), (.18, 'p95', '95th percentile')):
            ax.bar(np.arange(len(names)) + offset,
                   [cv_summary[name][f'field_interior_rms_{statistic}'] for name in names],
                   width=.34, label=label)
        ax.set_xticks(np.arange(len(names)), [name.replace('_', '\n') for name in names])
        ax.set(ylabel='Normalized interior field RMS', title='Training-run CV comparison — not strict HF test')
        ax.grid(axis='y', alpha=.2); ax.legend()
        save_figure(fig, out / 'cv_model_comparison.png')
    summary.update({
        'artifact': 'p1_results_summary',
        'data_sha256': report['data_sha256'],
        'holdout_report_sha256': file_sha256(holdout / 'holdout_metrics.json'),
        'scope_warning': 'Accepted modes in the five-variable structural study; no universal error bound.',
    })
    write_json(out / 'results_summary.json', summary, overwrite=True)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='p1_data directory containing a corrections revision')
    parser.add_argument('--name', default='corrections_v5')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--example-run', type=int)
    parser.add_argument('--example-mode', type=int)
    parser.add_argument('--design', type=Path, help='Explicit design CSV for design-space plots')
    parser.add_argument('--holdout', type=Path, help='Corrected strict holdout directory; cached predictions only')
    parser.add_argument('--surrogate', type=Path, help='Training-run CV directory; requires --holdout')
    args = parser.parse_args(argv)
    if args.surrogate is not None and args.holdout is None:
        parser.error('--surrogate requires --holdout to verify the frozen training split')
    data = args.data.resolve(); corr = data/args.name
    out = (args.out or corr/'figures').resolve(); out.mkdir(parents=True, exist_ok=True)
    dataset = P1Dataset.load(corr)
    x, y = dataset.x, dataset.y
    lf, hf, delta = dataset.lf, dataset.hf, dataset.correction
    runs, modes = dataset.run_ids, dataset.mode_ids
    f_lf, f_hf = dataset.f_lf, dataset.f_hf
    interior = dataset.interior_mask
    rows = read_csv(corr/'pairing.csv')
    accepted = [r for r in rows if r['status'] == 'accepted']
    if len(accepted) != len(runs):
        raise ValueError('pairing.csv and training.npz accepted rows disagree')
    rel = 100*(f_hf-f_lf)/f_hf
    rms = np.sqrt(np.mean(delta[:, interior]**2, axis=1))
    edge_rms = np.sqrt(np.mean(delta[:, ~interior]**2, axis=1))
    unique_modes = sorted(set(modes.tolist()))

    fig, ax = plt.subplots(figsize=(7,6))
    for mode in unique_modes:
        ix = modes == mode
        ax.scatter(f_lf[ix], f_hf[ix], s=15, alpha=.65, label=f'Mode {mode}')
    lo, hi = min(f_lf.min(), f_hf.min()), max(f_lf.max(), f_hf.max())
    ax.plot([lo,hi], [lo,hi], 'k--', lw=1, label='Equality')
    ax.set(xlabel='FSDT frequency (Hz)', ylabel='COMSOL frequency (Hz)',
           title='Accepted LF/HF frequency pairs'); ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.25); fig.tight_layout(); save_figure(fig, out/'frequency_parity.png')

    fig, ax = plt.subplots(figsize=(8,5))
    values = [rel[modes == mode] for mode in unique_modes]
    ax.boxplot(values, tick_labels=[str(m) for m in unique_modes], showfliers=False)
    ax.axhline(0, color='k', lw=.8); ax.set(xlabel='Reference LF mode', ylabel='HF−LF frequency error (%)',
                                              title='Frequency correction by mode'); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); save_figure(fig, out/'frequency_error_by_mode.png')

    fig, ax = plt.subplots(figsize=(8,5))
    ax.boxplot([rms[modes == mode] for mode in unique_modes],
               tick_labels=[str(m) for m in unique_modes], showfliers=False)
    ax.set(xlabel='Reference LF mode', ylabel='Interior RMS(shape correction)',
           title='Max-normalized spatial correction by mode')
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout(); save_figure(fig, out/'correction_rms_by_mode.png')

    all_rows = rows
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    axes[0].hist([float(r['tracking_mac']) for r in all_rows], bins=25, alpha=.7, label='LF tracking MAC')
    axes[0].hist([float(r['pair_mac']) for r in all_rows], bins=25, alpha=.7, label='LF/HF pair MAC')
    axes[0].axvline(.8, color='k', ls='--', label='acceptance floor'); axes[0].set(xlabel='MAC',ylabel='Rows',title='Modal matching quality'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    status = [sum(r['status']=='accepted' for r in all_rows), sum(r['status']=='quarantined' for r in all_rows)]
    axes[1].bar(['Accepted','Quarantined'], status, color=['#4c9f70','#c46b6b'])
    axes[1].set(ylabel='Mode pairs',title='Target audit status'); axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout(); save_figure(fig, out/'matching_quality.png')

    # A row can carry multiple flags, so these are overlapping counts rather
    # than a partition of the unique quarantined rows.
    flags = [
        ('Near repeated\neigenvalue', 'near_repeated_eigenvalue_use_subspace_target', '#7b6fd0'),
        ('Low MAC', 'low_MAC', '#d9776c'),
        ('Ambiguous\nassignment', 'ambiguous_assignment', '#e3a84d'),
        ('Low HF\nw energy', 'low_HF_transverse_energy_fraction', '#8b6fb0'),
    ]
    flag_counts = [sum(token in r['reason'] for r in rows) for _, token, _ in flags]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar([name for name, _, _ in flags], flag_counts,
                  color=[color for _, _, color in flags], edgecolor='0.25', linewidth=.6)
    for bar, count in zip(bars, flag_counts):
        ax.text(bar.get_x() + bar.get_width()/2, count + .8, str(count),
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Mode-pair rows carrying flag')
    ax.set_title('Quarantine audit flags (flags may overlap)')
    ax.text(.99, .98, f"Unique quarantined rows: {sum(r['status'] == 'quarantined' for r in rows)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9, color='0.25')
    ax.grid(axis='y', alpha=.22)
    fig.tight_layout(); save_figure(fig, out/'quarantine_reasons.png')

    if args.design is not None:
        sample_rows = [
            values for _, values in read_design_table(args.design, dataset.parameter_names)
        ]
        alpha = np.array([r['alpha'] for r in sample_rows]); beta = np.array([r['beta'] for r in sample_rows])
        theta = np.array([r['theta_c'] for r in sample_rows]); eta1 = np.array([r['eta1'] for r in sample_rows])
        fig, ax = plt.subplots(figsize=(7,5.5)); sc=ax.scatter(alpha,beta,c=theta,s=20+20*eta1,cmap='viridis',edgecolors='k',lw=.25)
        fig.colorbar(sc,ax=ax,label='Cell angle θc (deg)'); ax.set(xlabel='Core ratio α',ylabel='Face ratio β',title='Five-variable Latin-hypercube design'); ax.grid(alpha=.2)
        fig.tight_layout(); save_figure(fig, out/'design_space.png')

        # Pairwise design matrix for supplementary material and reviewer checks.
        values = [alpha, beta, theta, eta1, np.array([float(r['eta2']) for r in sample_rows])]
        labels = [r'$\alpha$', r'$\beta$', r'$\theta_c$ (deg)', r'$\eta_1$', r'$\eta_2$']
        n = len(values)
        fig, axes = plt.subplots(n, n, figsize=(10, 10), constrained_layout=True)
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    ax.hist(values[i], bins=12, color='#4c78a8', alpha=.85, edgecolor='white')
                elif i > j:
                    ax.scatter(values[j], values[i], s=13, alpha=.7, color='#4c78a8', edgecolors='none')
                else:
                    ax.axis('off')
                if i == n - 1 and i > j:
                    ax.set_xlabel(labels[j])
                elif i != n - 1:
                    ax.set_xticklabels([])
                if j == 0 and i != j:
                    ax.set_ylabel(labels[i])
                elif j != 0:
                    ax.set_yticklabels([])
                ax.grid(alpha=.15)
        fig.suptitle(f'Pairwise coverage of the {len(sample_rows)}-point LHS design', y=1.02)
        save_figure(fig, out/'design_pairwise.png')

    target = 0
    if args.example_run is not None:
        found = np.flatnonzero((runs == args.example_run) & ((args.example_mode is None) | (modes == args.example_mode)))
        if not len(found): raise ValueError('Requested example is not an accepted target')
        target = int(found[0])
    elif len(runs):
        target = 0
    fig, axes = plt.subplots(1,3,figsize=(13,4.2), constrained_layout=True)
    extent=[float(x[0]),float(x[-1]),float(y[0]),float(y[-1])]
    vmax=max(np.max(np.abs(lf[target])),np.max(np.abs(hf[target])),1e-12)
    for ax, field, title in zip(axes,(lf[target],hf[target],delta[target]),('FSDT shape','COMSOL shape','Aligned HF−LF correction')):
        scale=vmax if title != 'Aligned HF−LF correction' else max(np.max(np.abs(field)),1e-12)
        im=ax.imshow(field,origin='lower',extent=extent,aspect='equal',cmap='RdBu_r',vmin=-scale,vmax=scale)
        ax.set(title=title,xlabel='x (m)',ylabel='y (m)'); fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(f'Accepted example: run {runs[target]}, reference mode {modes[target]}')
    save_figure(fig, out/f'example_run{runs[target]:04d}_mode{modes[target]:02d}.png')
    if args.holdout is not None:
        _write_holdout_results(dataset, corr, args.holdout, args.surrogate, out)
    print(f'Interior correction RMS median={np.median(rms):.6g}; '
          f'edge RMS median={np.median(edge_rms):.6g}')

    manifest=json.loads((corr/'manifest.json').read_text(encoding='utf-8'))
    print(f"Wrote {len(list(out.glob('*.png')))} PNG figures and {len(list(out.glob('*.pdf')))} PDF figures to {out}")
    print(f"Accepted {manifest['accepted']} / {manifest['accepted']+manifest['quarantined']} audited targets")


if __name__ == '__main__': main()
