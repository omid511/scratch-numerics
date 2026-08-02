#!/usr/bin/env python3
"""
Compute Correction Fields (COMSOL vs FSDT)
==========================================
Loads FSDT and COMSOL mode shapes, computes the difference for each sample/mode.
Saves frequency errors and spatial correction fields.

Outputs:
    - correction_freq.csv          : Frequency errors per sample (100 x 10)
    - correction_fields/           : Spatial correction fields (100 x 10, 80x80 grid)
    - correction_freq_analysis.png : Frequency error statistics
    - correction_field_stats.png   : Spatial correction field statistics
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SAMPLES = 100
N_MODES = 10
GRID_RES = 80

FSDT_FREQ_CSV = "D:/plate-main/lhs_fsdt_results.csv"
COMSOL_FREQ_CSV = "D:/plate-main/lhs_results_master.csv"
FSDT_DIR = "D:/plate-main/fsdt_mode_shapes"
COMSOL_DIR = "D:/plate-main/mode_shapes"
OUTPUT_DIR = "D:/plate-main/correction_fields"
OUTPUT_FREQ_CSV = "D:/plate-main/correction_freq.csv"


def load_frequencies():
    """Load FSDT and COMSOL frequencies, return aligned arrays."""
    fsdt_freqs = []
    with open(FSDT_FREQ_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fsdt_freqs.append([float(row[f'f{i}_fsdt']) for i in range(1, N_MODES + 1)])

    comsol_freqs = []
    with open(COMSOL_FREQ_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comsol_freqs.append([float(row[f'f{i}']) for i in range(1, N_MODES + 1)])

    return np.array(fsdt_freqs), np.array(comsol_freqs)


def load_mode_shape(path, normalize=True):
    """Load a single mode shape CSV, return (80,80) w-displacement grid."""
    data = np.loadtxt(path, skiprows=3, delimiter=",")
    w = data[:, 4].reshape(GRID_RES, GRID_RES)
    if normalize:
        max_abs = np.max(np.abs(w))
        if max_abs > 0:
            w = w / max_abs
    return w


def compute_frequency_corrections(fsdt_freqs, comsol_freqs):
    """Compute frequency errors: absolute, relative, and percentage."""
    abs_err = comsol_freqs - fsdt_freqs
    rel_err = abs_err / comsol_freqs * 100  # percentage
    return abs_err, rel_err


def compute_spatial_corrections():
    """Compute spatial correction fields for all samples and modes.
    
    Both FSDT and COMSOL mode shapes are normalized to [-1, 1] independently.
    The correction = normalized_COMSOL - normalized_FSDT represents shape difference.
    """
    corrections = np.zeros((N_SAMPLES, N_MODES, GRID_RES, GRID_RES))

    for run in range(1, N_SAMPLES + 1):
        for mode in range(1, N_MODES + 1):
            fsdt_path = os.path.join(FSDT_DIR, f"fsdt_run{run}_mode{mode}.csv")
            comsol_path = os.path.join(COMSOL_DIR, f"mode_shape_run{run}_mode{mode}.csv")

            if not os.path.exists(fsdt_path) or not os.path.exists(comsol_path):
                print(f"  MISSING: run{run} mode{mode}")
                continue

            fsdt_w = load_mode_shape(fsdt_path)
            comsol_w = load_mode_shape(comsol_path)

            # Both already normalized to [-1, 1] in their respective scripts
            corrections[run - 1, mode - 1] = comsol_w - fsdt_w

    return corrections


def save_frequency_corrections(fsdt_freqs, comsol_freqs, abs_err, rel_err):
    """Save frequency corrections to CSV."""
    header = ['run']
    for i in range(1, N_MODES + 1):
        header += [f'f{i}_fsdt', f'f{i}_comsol', f'f{i}_abs_err', f'f{i}_rel_err_pct']

    with open(OUTPUT_FREQ_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for run in range(N_SAMPLES):
            row = [run + 1]
            for mode in range(N_MODES):
                row += [fsdt_freqs[run, mode], comsol_freqs[run, mode],
                        abs_err[run, mode], rel_err[run, mode]]
            writer.writerow(row)


def save_spatial_corrections(corrections):
    """Save spatial correction fields to CSV files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xv = np.linspace(0, 0.3, GRID_RES)
    yv = np.linspace(0, 0.3, GRID_RES)
    Xg, Yg = np.meshgrid(xv, yv)

    for run in range(N_SAMPLES):
        for mode in range(N_MODES):
            out_path = os.path.join(OUTPUT_DIR, f"correction_run{run+1}_mode{mode+1}.csv")
            field = corrections[run, mode]
            flat = np.column_stack([Xg.ravel(), Yg.ravel(), field.ravel()])
            header = (f"Correction field: COMSOL - FSDT\n"
                      f"x,y,delta_w\n{GRID_RES}x{GRID_RES} grid")
            np.savetxt(out_path, flat, delimiter=",", header=header, comments="")


def plot_frequency_analysis(abs_err, rel_err, fsdt_freqs, comsol_freqs):
    """Plot frequency error statistics with best/worst case info."""
    mean_per_sample = np.mean(np.abs(rel_err), axis=1)
    worst_idx = int(np.argmax(mean_per_sample))
    best_idx = int(np.argmin(mean_per_sample))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FSDT vs COMSOL Frequency Error Analysis', fontsize=14, fontweight='bold')

    # Box plot of relative error per mode
    axes[0, 0].boxplot([rel_err[:, i] for i in range(N_MODES)], label=range(1, N_MODES + 1))
    axes[0, 0].set_xlabel('Mode')
    axes[0, 0].set_ylabel('Relative Error (%)')
    axes[0, 0].set_title('Error Distribution per Mode')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Mean error per mode
    mean_err = np.mean(rel_err, axis=0)
    std_err = np.std(rel_err, axis=0)
    axes[0, 1].bar(range(1, N_MODES + 1), mean_err, yerr=std_err, capsize=5,
                   color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Mode')
    axes[0, 1].set_ylabel('Mean Relative Error (%)')
    axes[0, 1].set_title('Mean Error per Mode')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Best vs Worst case comparison
    modes = range(1, N_MODES + 1)
    axes[1, 0].plot(modes, comsol_freqs[worst_idx], 'ro-', label='COMSOL', linewidth=2)
    axes[1, 0].plot(modes, fsdt_freqs[worst_idx], 'r^--', label='FSDT', linewidth=2)
    axes[1, 0].plot(modes, comsol_freqs[best_idx], 'bs-', label='COMSOL', linewidth=2)
    axes[1, 0].plot(modes, fsdt_freqs[best_idx], 'b^--', label='FSDT', linewidth=2)
    axes[1, 0].set_xlabel('Mode')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_title(f'Best (Run {best_idx+1}, {mean_per_sample[best_idx]:.1f}%) vs '
                         f'Worst (Run {worst_idx+1}, {mean_per_sample[worst_idx]:.1f}%)')
    axes[1, 0].legend([f'COMSOL (best)', f'FSDT (best)', f'COMSOL (worst)', f'FSDT (worst)'])
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram of all errors
    all_err = rel_err.flatten()
    axes[1, 1].hist(all_err, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=mean_per_sample[best_idx], color='b', linestyle='--', linewidth=2,
                       label=f'Best: {mean_per_sample[best_idx]:.1f}%')
    axes[1, 1].axvline(x=mean_per_sample[worst_idx], color='r', linestyle='--', linewidth=2,
                       label=f'Worst: {mean_per_sample[worst_idx]:.1f}%')
    axes[1, 1].set_xlabel('Relative Error (%)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'All Errors (mean={np.mean(all_err):.1f}%, std={np.std(all_err):.1f}%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_freq_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_freq_analysis.png")
    plt.close()


def plot_spatial_stats(corrections, rel_err):
    """Plot spatial correction field statistics with best/worst case info."""
    mean_per_sample = np.mean(np.abs(rel_err), axis=1)
    worst_idx = int(np.argmax(mean_per_sample))
    best_idx = int(np.argmin(mean_per_sample))

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle('Spatial Correction Field RMS per Mode (COMSOL − FSDT, normalized)', 
                 fontsize=14, fontweight='bold')

    for mode in range(N_MODES):
        ax = axes.flat[mode]
        rms_per_sample = np.sqrt(np.mean(corrections[:, mode] ** 2, axis=(1, 2)))
        ax.hist(rms_per_sample, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=rms_per_sample[best_idx], color='b', linestyle='--', linewidth=2,
                   label=f'Best' if mode == 0 else '')
        ax.axvline(x=rms_per_sample[worst_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Worst' if mode == 0 else '')
        ax.set_xlabel('RMS(Δw)')
        ax.set_ylabel('Count')
        ax.set_title(f'Mode {mode+1} (mean={rms_per_sample.mean():.4f})')
        ax.grid(True, alpha=0.3)
        if mode == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_field_stats.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_field_stats.png")
    plt.close()


def plot_fsdt_vs_comsol(fsdt_freqs, comsol_freqs, rel_err):
    """Plot FSDT vs COMSOL frequencies with diagonal line."""
    mean_per_sample = np.mean(np.abs(rel_err), axis=1)
    worst_idx = int(np.argmax(mean_per_sample))
    best_idx = int(np.argmin(mean_per_sample))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('FSDT vs COMSOL Natural Frequencies', fontsize=14, fontweight='bold')

    # All samples
    ax = axes[0]
    all_fsdt = fsdt_freqs.flatten()
    all_comsol = comsol_freqs.flatten()
    ax.scatter(all_comsol, all_fsdt, s=10, alpha=0.4, edgecolors='k', linewidths=0.3)
    lims = [min(all_fsdt.min(), all_comsol.min()) * 0.9,
            max(all_fsdt.max(), all_comsol.max()) * 1.1]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect agreement')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('COMSOL (Hz)')
    ax.set_ylabel('FSDT (Hz)')
    ax.set_title(f'All 1000 points (100 samples x 10 modes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Best vs Worst per mode
    ax = axes[1]
    modes = range(1, N_MODES + 1)
    ax.plot(modes, comsol_freqs[best_idx], 'bs-', linewidth=2, markersize=6)
    ax.plot(modes, fsdt_freqs[best_idx], 'b^--', linewidth=2, markersize=6)
    ax.plot(modes, comsol_freqs[worst_idx], 'rs-', linewidth=2, markersize=6)
    ax.plot(modes, fsdt_freqs[worst_idx], 'r^--', linewidth=2, markersize=6)
    ax.set_xlabel('Mode')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Best (Run {best_idx+1}) vs Worst (Run {worst_idx+1})')
    ax.legend([f'COMSOL best ({mean_per_sample[best_idx]:.1f}%)',
               f'FSDT best',
               f'COMSOL worst ({mean_per_sample[worst_idx]:.1f}%)',
               f'FSDT worst'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_fsdt_vs_comsol.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_fsdt_vs_comsol.png")
    plt.close()


def plot_spatial_correction_contours(corrections, rel_err):
    """Plot actual correction field contours for best/worst cases."""
    mean_per_sample = np.mean(np.abs(rel_err), axis=1)
    worst_idx = int(np.argmax(mean_per_sample))
    best_idx = int(np.argmin(mean_per_sample))

    xv = np.linspace(0, 0.3, GRID_RES)
    yv = np.linspace(0, 0.3, GRID_RES)
    X, Y = np.meshgrid(xv, yv)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f'Spatial Correction Field (COMSOL − FSDT) — Best Case (Run {best_idx+1})',
                 fontsize=14, fontweight='bold')

    for mode in range(N_MODES):
        ax = axes.flat[mode]
        field = corrections[best_idx, mode]
        vmax = np.max(np.abs(field))
        if vmax > 0:
            im = ax.contourf(X, Y, field, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.contourf(X, Y, field, levels=20, cmap='RdBu_r')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Mode {mode+1}', fontsize=10)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_spatial_best.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_spatial_best.png")
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f'Spatial Correction Field (COMSOL − FSDT) — Worst Case (Run {worst_idx+1})',
                 fontsize=14, fontweight='bold')

    for mode in range(N_MODES):
        ax = axes.flat[mode]
        field = corrections[worst_idx, mode]
        vmax = np.max(np.abs(field))
        if vmax > 0:
            im = ax.contourf(X, Y, field, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.contourf(X, Y, field, levels=20, cmap='RdBu_r')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Mode {mode+1}', fontsize=10)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_spatial_worst.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_spatial_worst.png")
    plt.close()


def plot_correction_vs_parameters(rel_err, corrections):
    """Plot correction magnitude vs each parameter."""
    param_names = ['alpha', 'beta', 'theta_c', 'eta1', 'eta2']
    param_labels = [r'$\alpha$', r'$\beta$', r'$\theta_c$', r'$\eta_1$', r'$\eta_2$']

    # Load parameters
    samples = []
    with open(FSDT_FREQ_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append([float(row[p]) for p in param_names])
    params = np.array(samples)

    mean_err_per_sample = np.mean(np.abs(rel_err), axis=1)

    # Frequency error vs parameters
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle('Frequency Error vs Parameters', fontsize=14, fontweight='bold')

    for p_idx, (ax, label) in enumerate(zip(axes, param_labels)):
        ax.scatter(params[:, p_idx], mean_err_per_sample, s=20, alpha=0.6,
                   edgecolors='k', linewidths=0.3)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_freq_vs_params.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_freq_vs_params.png")
    plt.close()

    # Spatial correction RMS vs parameters
    rms_all = np.zeros((N_SAMPLES, N_MODES))
    for mode in range(N_MODES):
        rms_all[:, mode] = np.sqrt(np.mean(corrections[:, mode] ** 2, axis=(1, 2)))

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle('Spatial Correction RMS vs Parameters', fontsize=14, fontweight='bold')

    for mode in range(N_MODES):
        ax = axes.flat[mode]
        # Color by mean parameter value
        sc = ax.scatter(params[:, 0], params[:, 1], c=rms_all[:, mode],
                       s=30, cmap='viridis', edgecolors='k', linewidths=0.3)
        ax.set_xlabel(r'$\alpha$', fontsize=10)
        ax.set_ylabel(r'$\beta$', fontsize=10)
        ax.set_title(f'Mode {mode+1}', fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_spatial_vs_params.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_spatial_vs_params.png")
    plt.close()


def plot_mode_shape_comparison(rel_err):
    """Side-by-side FSDT vs COMSOL mode shapes for best/worst cases."""
    mean_per_sample = np.mean(np.abs(rel_err), axis=1)
    worst_idx = int(np.argmax(mean_per_sample))
    best_idx = int(np.argmin(mean_per_sample))

    xv = np.linspace(0, 0.3, GRID_RES)
    yv = np.linspace(0, 0.3, GRID_RES)
    X, Y = np.meshgrid(xv, yv)

    for case_idx, case_label, case_color in [(best_idx, 'Best', 'Blues'), (worst_idx, 'Worst', 'Reds')]:
        fig, axes = plt.subplots(3, 10, figsize=(28, 9))
        fig.suptitle(f'{case_label} Case (Run {case_idx+1}, {mean_per_sample[case_idx]:.1f}% error) — FSDT vs COMSOL Mode Shapes',
                     fontsize=14, fontweight='bold')

        for mode in range(N_MODES):
            # FSDT
            ax_fsd = axes[0, mode]
            fsdt_path = os.path.join(FSDT_DIR, f"fsdt_run{case_idx+1}_mode{mode+1}.csv")
            fsdt_w = load_mode_shape(fsdt_path)
            vmax = max(np.max(np.abs(fsdt_w)), 0.01)
            im = ax_fsd.contourf(X, Y, fsdt_w, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax_fsd.set_title(f'Mode {mode+1}', fontsize=9)
            if mode == 0:
                ax_fsd.set_ylabel('FSDT', fontsize=10, fontweight='bold')
            ax_fsd.set_xticks([])

            # COMSOL
            ax_com = axes[1, mode]
            comsol_path = os.path.join(COMSOL_DIR, f"mode_shape_run{case_idx+1}_mode{mode+1}.csv")
            comsol_w = load_mode_shape(comsol_path)
            im = ax_com.contourf(X, Y, comsol_w, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            if mode == 0:
                ax_com.set_ylabel('COMSOL', fontsize=10, fontweight='bold')
            ax_com.set_xticks([])

            # Correction
            ax_corr = axes[2, mode]
            corr = comsol_w - fsdt_w
            vmax_corr = max(np.max(np.abs(corr)), 0.01)
            im = ax_corr.contourf(X, Y, corr, levels=20, cmap='RdBu_r', vmin=-vmax_corr, vmax=vmax_corr)
            if mode == 0:
                ax_corr.set_ylabel('Δw\n(COMSOL−FSDT)', fontsize=9, fontweight='bold')
            ax_corr.set_xlabel('x (m)', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'D:/plate-main/correction_mode_shapes_{case_label.lower()}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved correction_mode_shapes_{case_label.lower()}.png")
        plt.close()


def plot_correction_spatial_mean(corrections):
    """Plot mean |Δw| over plate surface for each sample and mode."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Spatial Correction Magnitude', fontsize=14, fontweight='bold')

    # Mean |Δw| per sample (averaged over all modes)
    mean_abs_per_sample = np.mean(np.abs(corrections), axis=(1, 2, 3))
    ax = axes[0]
    ax.bar(range(1, N_SAMPLES + 1), mean_abs_per_sample, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Mean |Δw|')
    ax.set_title('Mean Correction Magnitude per Sample (all modes)')
    ax.grid(True, alpha=0.3, axis='y')

    # Per mode
    ax = axes[1]
    mean_abs_per_mode = np.mean(np.abs(corrections), axis=(0, 2, 3))
    ax.bar(range(1, N_MODES + 1), mean_abs_per_mode, color='steelblue', edgecolor='black')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Mean |Δw|')
    ax.set_title('Mean Correction Magnitude per Mode (all samples)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_spatial_mean.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_spatial_mean.png")
    plt.close()


def plot_error_by_param_range(fsdt_freqs, comsol_freqs, rel_err):
    """Box plots of frequency error grouped by parameter quartiles."""
    param_names = ['alpha', 'beta', 'theta_c', 'eta1', 'eta2']
    param_labels = [r'$\alpha$', r'$\beta$', r'$\theta_c$', r'$\eta_1$', r'$\eta_2$']

    samples = []
    with open(FSDT_FREQ_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append([float(row[p]) for p in param_names])
    params = np.array(samples)

    mean_err_per_sample = np.mean(np.abs(rel_err), axis=1)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('Frequency Error by Parameter Range (low vs high)', fontsize=14, fontweight='bold')

    for p_idx, (ax, label) in enumerate(zip(axes, param_labels)):
        median_val = np.median(params[:, p_idx])
        low_mask = params[:, p_idx] <= median_val
        high_mask = ~low_mask

        data_low = mean_err_per_sample[low_mask]
        data_high = mean_err_per_sample[high_mask]

        bp = ax.boxplot([data_low, data_high], label=['Low', 'High'],
                       patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightsalmon')
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title(f'{label} (low: {data_low.mean():.1f}%, high: {data_high.mean():.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('D:/plate-main/correction_error_by_param_range.png', dpi=150, bbox_inches='tight')
    print("  Saved correction_error_by_param_range.png")
    plt.close()


def main():
    print("=" * 70)
    print("COMPUTE CORRECTION FIELDS (COMSOL − FSDT)")
    print("=" * 70)

    # Load frequencies
    print("\n[1/5] Loading frequencies...")
    fsdt_freqs, comsol_freqs = load_frequencies()
    print(f"  FSDT: {fsdt_freqs.shape}, COMSOL: {comsol_freqs.shape}")

    # Compute frequency corrections
    print("\n[2/5] Computing frequency corrections...")
    abs_err, rel_err = compute_frequency_corrections(fsdt_freqs, comsol_freqs)
    mean_per_sample = np.mean(np.abs(rel_err), axis=1)
    worst_idx = int(np.argmax(mean_per_sample))
    best_idx = int(np.argmin(mean_per_sample))
    print(f"  Mean relative error: {np.mean(np.abs(rel_err)):.1f}%")
    print(f"  Max relative error:  {np.max(np.abs(rel_err)):.1f}%")
    print(f"  Best:  Run {best_idx+1} ({mean_per_sample[best_idx]:.1f}%)")
    print(f"  Worst: Run {worst_idx+1} ({mean_per_sample[worst_idx]:.1f}%)")

    # Save frequency corrections
    print("\n[3/5] Saving frequency corrections...")
    save_frequency_corrections(fsdt_freqs, comsol_freqs, abs_err, rel_err)
    print(f"  Saved {OUTPUT_FREQ_CSV}")

    # Compute spatial corrections (normalized per pair)
    print("\n[4/5] Computing spatial correction fields...")
    corrections = compute_spatial_corrections()
    print(f"  Shape: {corrections.shape}")

    # Save spatial corrections
    print("\n[5/5] Saving spatial correction fields...")
    save_spatial_corrections(corrections)
    print(f"  Saved {N_SAMPLES * N_MODES} files to {OUTPUT_DIR}/")

    # Plots
    print("\nGenerating plots...")
    plot_frequency_analysis(abs_err, rel_err, fsdt_freqs, comsol_freqs)
    plot_spatial_stats(corrections, rel_err)
    plot_fsdt_vs_comsol(fsdt_freqs, comsol_freqs, rel_err)
    plot_spatial_correction_contours(corrections, rel_err)
    plot_correction_vs_parameters(rel_err, corrections)
    plot_mode_shape_comparison(rel_err)
    plot_correction_spatial_mean(corrections)
    plot_error_by_param_range(fsdt_freqs, comsol_freqs, rel_err)

    print("\nDone!")


if __name__ == "__main__":
    main()
