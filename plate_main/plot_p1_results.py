#!/usr/bin/env python3
"""Create publication-ready diagnostics from audited P1 correction bundles.

Only accepted rows in the selected corrections revision are plotted.  Field
plots are max-normalized shape corrections, not dimensional displacements.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
from p1_pairing import boundary_mask
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='p1_data directory containing a corrections revision')
    parser.add_argument('--name', default='corrections_v5')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--example-run', type=int)
    parser.add_argument('--example-mode', type=int)
    args = parser.parse_args(argv)
    data = args.data.resolve(); corr = data/args.name
    out = (args.out or corr/'figures').resolve(); out.mkdir(parents=True, exist_ok=True)
    with np.load(corr/'training.npz', allow_pickle=False) as z:
        x, y = z['x'], z['y']
        lf, hf, delta = z['lf'], z['hf'], z['correction']
        runs, modes = z['run_ids'].astype(int), z['mode_ids'].astype(int)
        f_lf, f_hf = z['f_lf'], z['f_hf']
        mask = z['boundary_mask'] if 'boundary_mask' in z.files else boundary_mask(x, y)
        interior = (z['interior_mask'].astype(bool)
                    if 'interior_mask' in z.files else mask > 0)
    if mask.shape != (len(y), len(x)) or interior.shape != mask.shape:
        raise ValueError('Boundary mask does not match the training grid')
    if not np.any(interior) or not np.any(~interior):
        raise ValueError('Boundary mask must contain interior and edge points')
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

    samples_path = data.parent/'lhs_samples_v2.csv'
    if samples_path.exists():
        sample_rows = read_csv(samples_path)
        alpha = np.array([float(r['alpha']) for r in sample_rows]); beta = np.array([float(r['beta']) for r in sample_rows])
        theta = np.array([float(r['theta_c']) for r in sample_rows]); eta1 = np.array([float(r['eta1']) for r in sample_rows])
        fig, ax = plt.subplots(figsize=(7,5.5)); sc=ax.scatter(alpha,beta,c=theta,s=20+20*eta1,cmap='viridis',edgecolors='k',lw=.25)
        fig.colorbar(sc,ax=ax,label='Cell angle θc (deg)'); ax.set(xlabel='Core ratio α',ylabel='Face ratio β',title='P1-v2 Latin-hypercube design'); ax.grid(alpha=.2)
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
    print(f'Interior correction RMS median={np.median(rms):.6g}; '
          f'edge RMS median={np.median(edge_rms):.6g}')

    manifest=json.loads((corr/'manifest.json').read_text(encoding='utf-8'))
    print(f"Wrote {len(list(out.glob('*.png')))} PNG figures and {len(list(out.glob('*.pdf')))} PDF figures to {out}")
    print(f"Accepted {manifest['accepted']} / {manifest['accepted']+manifest['quarantined']} audited targets")


if __name__ == '__main__': main()
