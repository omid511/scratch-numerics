#!/usr/bin/env python3
"""
Latin Hypercube Sampling for FSDT vs COMSOL Correction Study
=============================================================

This script generates 100 parameter combinations using Latin Hypercube Sampling,
runs the FSDT solver on each, and exports results for COMSOL comparison.

Parameters varied (5 total):
    alpha  : Core thickness ratio (h2/h)          [0.1, 0.9]
    beta   : Face sheet thickness ratio (h1/h3)   [0.1, 1.0]
    theta_c: Honeycomb cell angle (degrees)       [0, 75]
    eta1   : Edge length ratio (l2/l1)            [0.5, 3.0]
    eta2   : Wall thickness ratio (tc/l1)         [0.02, 0.15]

Outputs:
    - lhs_samples.csv              : 100 parameter combinations
    - lhs_fsdt_results.csv         : Parameters + FSDT natural frequencies (10 modes)
    - lhs_coverage.png             : Visual check of sample coverage
    - lhs_sensitivity_mode{1-10}.png: Scatter plots for each mode (2x3 grid each)
    - lhs_sensitivity_all_modes.png: Parameter correlation bar charts for all 10 modes
    - lhs_correlation_heatmap.png  : Heatmap of parameter correlation with each mode
    - lhs_distributions.png        : Parameter distributions
    - lhs_mode_shape_sensitivity.png : Mode shape RMS sensitivity for all 10 modes

"""

import sys
import os
import time
import numpy as np
from scipy.stats import qmc

# Add project root to path
sys.path.insert(0, '.')
import plate
import honeycomb

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42          # For reproducibility
N_SAMPLES = 100           # Number of LHS samples
N_MODES = 10              # Number of natural frequencies to compute
M_ORDER = 15              # Ritz basis order (M=N=15, matches paper)
N_BASIS = 50              # Legendre basis truncation order

# Fixed plate geometry (same as paper Table 1)
L1 = 0.3                  # Plate length x (m)
L2 = 0.3                  # Plate length y (m)
H_TOTAL = 0.01            # Total plate thickness (m)

# Face sheet material: Aluminium
E_face = 70e9             # Young's modulus (Pa)
rho_face = 2710           # Density (kg/m3)
nu_face = 0.33            # Poisson's ratio
G_face = E_face / (2 * (1 + nu_face))

# Boundary condition spring stiffness (clamped)
K_SPRING = 1e12           # Penalty spring stiffness

# Parameter ranges [lower, upper]
PARAM_RANGES = {
    'alpha':   [0.1, 0.9],       # Core thickness ratio
    'beta':    [0.1, 1.0],       # Face sheet ratio
    'theta_c': [0.0, 75.0],      # Cell angle (degrees)
    'eta1':    [0.5, 3.0],       # Edge length ratio
    'eta2':    [0.02, 0.15],     # Wall thickness ratio
}

# ============================================================================
# LHS GENERATION
# ============================================================================

def generate_lhs(n_samples, n_params, seed):
    """
    Generate Latin Hypercube samples in [0, 1]^n_params.
    
    Uses centered LHS for better space-filling properties.
    Scrambling is applied to break any residual grid patterns.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_params : int
        Number of parameters (dimensions).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray of shape (n_samples, n_params)
        Samples in [0, 1]^n_params.
    """
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    samples = sampler.random(n=n_samples)
    return samples


def apply_sensitivity_transform(lhs_unit):
    """
    Apply non-linear transformation to concentrate samples in high-sensitivity regions.
    
    Based on paper's parametric study findings:
    - α: sharp drop near α=1 → concentrate at upper end
    - β: maximum at β=1, high sensitivity at low β → concentrate at lower end
    - θc: flat until 60°, sharp drop after → concentrate at upper end
    - η1: peak near η1=1 → concentrate around 1
    - η2: peak near η2=0.05-0.10 → concentrate in middle range
    
    Parameters
    ----------
    lhs_unit : np.ndarray of shape (n_samples, 5)
        Unit LHS samples in [0, 1].
    
    Returns
    -------
    np.ndarray of shape (n_samples, 5)
        Transformed samples in [0, 1] with higher density in sensitive regions.
    """
    transformed = np.zeros_like(lhs_unit)
    
    # α (column 0): Power transform to concentrate at upper end (α > 0.8)
    # Exponent > 1 stretches the upper end
    transformed[:, 0] = lhs_unit[:, 0] ** 0.6  # More samples near α=0.8-0.9
    
    # β (column 1): Power transform to concentrate at lower end (β < 0.5)
    # Exponent < 1 stretches the lower end
    transformed[:, 1] = lhs_unit[:, 1] ** 1.8  # More samples near β=0.1-0.4
    
    # θc (column 2): Power transform to concentrate at upper end (θc > 50°)
    # Exponent > 1 stretches the upper end
    transformed[:, 2] = lhs_unit[:, 2] ** 0.5  # More samples near θc=50-75°
    
    # η1 (column 3): Beta-like transform centered around η1=1
    # Map [0,1] to peak at 0.2 (corresponds to η1=1.0 in [0.5, 3.0] range)
    x = lhs_unit[:, 3]
    # Use a transformation that peaks around x=0.2
    transformed[:, 3] = np.where(x < 0.5,
                                   0.5 * (2 * x) ** 0.5,   # Lower half: stretch
                                   0.5 + 0.5 * np.clip(2 * x - 1, 0, None) ** 2.0)  # Upper half: compress
    
    # η2 (column 4): Beta-like transform centered around η2=0.07
    # Map [0,1] to peak at 0.38 (corresponds to η2=0.07 in [0.02, 0.15] range)
    x = lhs_unit[:, 4]
    transformed[:, 4] = np.where(x < 0.5,
                                   0.5 * (2 * x) ** 0.7,   # Lower half: stretch
                                   0.5 + 0.5 * np.clip(2 * x - 1, 0, None) ** 1.5)  # Upper half: compress
    
    return transformed


def scale_to_physical(lhs_unit):
    """
    Scale LHS samples from [0, 1]^d to physical parameter ranges.
    Applies sensitivity-aware transformation first.
    
    Parameters
    ----------
    lhs_unit : np.ndarray of shape (n_samples, 5)
        Unit LHS samples.
    
    Returns
    -------
    np.ndarray of shape (n_samples, 5)
        Physical parameter values.
    """
    # Apply non-linear transform to concentrate in high-sensitivity regions
    lhs_transformed = apply_sensitivity_transform(lhs_unit)
    
    param_names = ['alpha', 'beta', 'theta_c', 'eta1', 'eta2']
    physical = np.zeros_like(lhs_transformed)
    for i, name in enumerate(param_names):
        lo, hi = PARAM_RANGES[name]
        physical[:, i] = lo + lhs_transformed[:, i] * (hi - lo)
    return physical


# ============================================================================
# FSDT SOLVER
# ============================================================================

def compute_fsdt_freq(alpha, beta, theta_c, eta1, eta2, return_mode_shapes=False):
    """
    Compute first N_MODES natural frequencies and optionally mode shapes.
    
    Parameters
    ----------
    alpha : float
        Core thickness ratio (h2/h).
    beta : float
        Face sheet ratio (h1/h3).
    theta_c : float
        Honeycomb cell angle in degrees.
    eta1 : float
        Edge length ratio (l2/l1).
    eta2 : float
        Wall thickness ratio (tc/l1).
    return_mode_shapes : bool
        If True, return mode shape data on a grid.
    
    Returns
    -------
    If return_mode_shapes is False:
        np.ndarray of shape (N_MODES,) - Natural frequencies in Hz
    If return_mode_shapes is True:
        tuple: (frequencies, mode_shapes_grid)
        - frequencies: np.ndarray of shape (N_MODES,)
        - mode_shapes_grid: list of N_MODES 2D arrays (N_GRID x N_GRID) of w displacement
    """
    try:
        # Honeycomb cell geometry
        t_core = eta2 * 3e-3       # Cell wall thickness (m)
        l1 = 3e-3                   # Cell edge length 1 (m)
        l2 = l1 * eta1              # Cell edge length 2 (m)
        theta = theta_c / 180 * np.pi  # Cell angle (rad)
        
        # Layer thicknesses from alpha and beta
        h2 = H_TOTAL * alpha                    # Core thickness
        h1 = (H_TOTAL - h2) / (1 + beta) * beta  # Top face
        h3 = H_TOTAL - h1 - h2                   # Bottom face
        
        # z-coordinates (centered at mid-plane)
        z = np.array([0, h1, h1 + h2, h1 + h2 + h3])
        z -= z[-1] / 2
        
        # Materials
        al = plate.Material(E_face, E_face, G_face, G_face, G_face,
                            nu_face, rho_face)
        core_props = honeycomb.material_property(
            E_face, G_face, rho_face, t_core, l1, l2, theta)
        core = plate.Material(*core_props)
        
        # Laminate profile
        profile = plate.Profile([al, core, al], [0, 0, 0], z)
        
        # Create plate with Ritz basis
        p = plate.Plate(L1, L2, M_ORDER, M_ORDER, profile)
        basis_x = plate.legendre_basis(N_BASIS, interval=[0, L1])
        basis_y = plate.legendre_basis(N_BASIS, interval=[0, L2])
        p.set_basis(basis_x, basis_y)
        
        # Clamped boundary conditions (penalty springs)
        p.add_spring(K_SPRING, plate.BC_C, x=0)
        p.add_spring(K_SPRING, plate.BC_C, x=L1)
        p.add_spring(K_SPRING, plate.BC_C, y=0)
        p.add_spring(K_SPRING, plate.BC_C, y=L2)
        
        # Assemble and solve
        p.assemble()
        solver = plate.ModalSolver(p)
        result = solver.solve(N_MODES * 2)  # Solve extra modes, take first N
        
        freqs = np.array([result[i].frequency for i in range(N_MODES)])
        
        if not return_mode_shapes:
            return freqs
        
        # Extract mode shapes on a grid
        N_GRID = 80
        x_grid = np.linspace(0, L1, N_GRID)
        y_grid = np.linspace(0, L2, N_GRID)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        mode_shapes = []
        for i in range(N_MODES):
            frame = result[i]
            w_grid = np.zeros((N_GRID, N_GRID))
            for ix, xi in enumerate(x_grid):
                for iy, yi in enumerate(y_grid):
                    w_grid[iy, ix] = frame.w(xi, yi)
            # Normalize mode shape
            max_abs = np.max(np.abs(w_grid))
            if max_abs > 0:
                w_grid = w_grid / max_abs
            mode_shapes.append(w_grid)
        
        return freqs, mode_shapes
    
    except Exception as e:
        print(f"  WARNING: Solver failed for alpha={alpha:.3f}, beta={beta:.3f}: {e}")
        if return_mode_shapes:
            return np.full(N_MODES, np.nan), [np.full((30, 30), np.nan) for _ in range(N_MODES)]
        return np.full(N_MODES, np.nan)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("Latin Hypercube Sampling for FSDT Correction Study")
    print("=" * 70)
    print()
    
    # Step 1: Generate LHS samples
    print("[1/4] Generating LHS samples...")
    t0 = time.time()
    lhs_unit = generate_lhs(N_SAMPLES, n_params=5, seed=RANDOM_SEED)
    lhs_physical = scale_to_physical(lhs_unit)
    print(f"  Generated {N_SAMPLES} samples in 5D parameter space")
    print(f"  Time: {time.time() - t0:.2f}s")
    print()
    
    # Step 2: Print sample summary
    print("[2/4] Parameter ranges in generated samples:")
    param_names = ['alpha', 'beta', 'theta_c', 'eta1', 'eta2']
    for i, name in enumerate(param_names):
        lo, hi = PARAM_RANGES[name]
        actual_lo = lhs_physical[:, i].min()
        actual_hi = lhs_physical[:, i].max()
        print(f"  {name:10s}: [{lo:.3f}, {hi:.3f}] -> sampled [{actual_lo:.3f}, {actual_hi:.3f}]")
    print()
    
    # Step 3: Run FSDT solver on each sample
    print(f"[3/4] Running FSDT solver on {N_SAMPLES} samples...")
    print(f"  Solving for {N_MODES} modes per sample (M=N={M_ORDER})")
    t0 = time.time()
    
    freq_results = np.zeros((N_SAMPLES, N_MODES))
    for idx in range(N_SAMPLES):
        alpha, beta, theta_c, eta1, eta2 = lhs_physical[idx]
        freqs = compute_fsdt_freq(alpha, beta, theta_c, eta1, eta2)
        freq_results[idx] = freqs
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (N_SAMPLES - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1:3d}/{N_SAMPLES}] f1={freqs[0]:8.1f} Hz | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.0f}s")
    
    total_time = time.time() - t0
    print(f"  Total time: {total_time:.1f}s ({total_time/N_SAMPLES:.2f}s per sample)")
    print()
    
    # Step 4: Export results
    print("[4/4] Exporting results...")
    
    import tempfile, shutil
    
    PROJECT_DIR = "D:/plate-main"
    
    def safe_save_csv(filepath, data, header):
        """Save CSV via temp file to avoid permission errors."""
        dir_name = PROJECT_DIR
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        os.close(fd)
        try:
            np.savetxt(tmp_path, data, delimiter=',', header=header, comments='', fmt='%.6f')
            shutil.move(tmp_path, filepath)
        except PermissionError:
            base, ext = os.path.splitext(filepath)
            alt_path = os.path.join(PROJECT_DIR, f"{os.path.basename(base)}_new{ext}")
            np.savetxt(alt_path, data, delimiter=',', header=header, comments='', fmt='%.6f')
            print(f"  WARNING: {filepath} locked. Saved as {alt_path}")
    
    # CSV: Parameter combinations only
    header_params = 'alpha,beta,theta_c,eta1,eta2'
    safe_save_csv(os.path.join(PROJECT_DIR, 'lhs_samples.csv'), lhs_physical, header_params)
    print(f"  Saved lhs_samples.csv ({N_SAMPLES} rows)")
    
    # CSV: Parameters + FSDT frequencies
    header_full = header_params + ',' + ','.join([f'f{i+1}_fsdt' for i in range(N_MODES)])
    full_results = np.hstack([lhs_physical, freq_results])
    safe_save_csv(os.path.join(PROJECT_DIR, 'lhs_fsdt_results.csv'), full_results, header_full)
    print(f"  Saved lhs_fsdt_results.csv ({N_SAMPLES} rows)")
    print()
    
    # Summary statistics
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    valid = ~np.isnan(freq_results[:, 0])
    n_valid = valid.sum()
    print(f"  Successful: {n_valid}/{N_SAMPLES}")
    if n_valid < N_SAMPLES:
        print(f"  Failed: {N_SAMPLES - n_valid}")
    print()
    
    if n_valid > 0:
        print("  Natural frequency statistics (Hz):")
        for i in range(N_MODES):
            col = freq_results[valid, i]
            print(f"    Mode {i+1}: mean={col.mean():.1f}, "
                  f"std={col.std():.1f}, "
                  f"min={col.min():.1f}, "
                  f"max={col.max():.1f}")
    print()
    
    # Visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
        
        param_names = ['alpha', 'beta', 'theta_c', 'eta1', 'eta2']
        param_labels = [r'$\alpha$', r'$\beta$', r'$\theta_c$', r'$\eta_1$', r'$\eta_2$']
        
        # =====================================================================
        # PLOT 1: LHS Coverage — all 10 pairwise combinations
        # =====================================================================
        from itertools import combinations
        
        param_names_list = ['alpha', 'beta', 'theta_c', 'eta1', 'eta2']
        param_labels_list = [r'$\alpha$', r'$\beta$', r'$\theta_c$', r'$\eta_1$', r'$\eta_2$']
        all_pairs = list(combinations(range(5), 2))
        
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        fig.suptitle('LHS Sample Coverage — All Parameter Pairs (100 samples)', fontsize=14)
        
        for plot_idx, (i, j) in enumerate(all_pairs):
            ax = axes.flat[plot_idx]
            ax.scatter(lhs_physical[:, i], lhs_physical[:, j],
                      s=15, alpha=0.7, edgecolors='k', linewidths=0.5)
            ax.set_xlabel(param_labels_list[i], fontsize=11)
            ax.set_ylabel(param_labels_list[j], fontsize=11)
            ax.set_xlim(PARAM_RANGES[param_names_list[i]])
            ax.set_ylim(PARAM_RANGES[param_names_list[j]])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lhs_coverage.png', dpi=150, bbox_inches='tight')
        print("  Saved lhs_coverage.png")
        plt.close()
        
        # =====================================================================
        # PLOT 2a: Parameter Sensitivity — Scatter for each Mode (10 images)
        # =====================================================================
        for mode_idx in range(N_MODES):
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            fig.suptitle(f'Parameter Sensitivity: Mode {mode_idx+1}', 
                         fontsize=14, fontweight='bold')
            
            freq_mode = freq_results[:, mode_idx]
            
            for p_idx, (ax, name, label) in enumerate(zip(axes.flat[:5], param_names, param_labels)):
                scatter = ax.scatter(lhs_physical[:, p_idx], freq_mode, 
                                   c=freq_mode, cmap='viridis', s=30, alpha=0.7, 
                                   edgecolors='k', linewidths=0.5)
                ax.set_xlabel(label, fontsize=12)
                ax.set_ylabel(f'f{mode_idx+1} (Hz)', fontsize=12)
                ax.set_xlim(PARAM_RANGES[name])
                ax.grid(True, alpha=0.3)
                
                z = np.polyfit(lhs_physical[:, p_idx], freq_mode, 2)
                p = np.poly1d(z)
                x_line = np.linspace(PARAM_RANGES[name][0], PARAM_RANGES[name][1], 100)
                ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
                ax.legend(loc='best')
            
            ax = axes.flat[5]
            correlations = []
            for p_idx in range(5):
                corr, _ = spearmanr(lhs_physical[:, p_idx], freq_mode)
                correlations.append(abs(corr))
            
            colors = ['red' if c == max(correlations) else 'steelblue' for c in correlations]
            bars = ax.bar(param_labels, correlations, color=colors, edgecolor='black')
            ax.set_ylabel('|Spearman Correlation|', fontsize=12)
            ax.set_title('Parameter Importance', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, correlations):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'lhs_sensitivity_mode{mode_idx+1}.png', dpi=150, bbox_inches='tight')
            print(f"  Saved lhs_sensitivity_mode{mode_idx+1}.png")
            plt.close()

        # =====================================================================
        # PLOT 2b: Parameter Sensitivity — Bar chart for all 10 Modes
        # =====================================================================
        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        fig.suptitle('Parameter Sensitivity: Spearman Correlation with Each Mode', 
                     fontsize=14, fontweight='bold')
        
        for mode_idx in range(N_MODES):
            ax = axes.flat[mode_idx]
            freq_mode = freq_results[:, mode_idx]
            correlations = []
            for p_idx in range(5):
                corr, _ = spearmanr(lhs_physical[:, p_idx], freq_mode)
                correlations.append(abs(corr))
            
            colors = ['red' if c == max(correlations) else 'steelblue' for c in correlations]
            bars = ax.bar(param_labels, correlations, color=colors, edgecolor='black')
            ax.set_ylabel('|Spearman Correlation|', fontsize=10)
            ax.set_title(f'Mode {mode_idx+1} (mean={freq_mode.mean():.0f}Hz)', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', labelsize=8)
            
            for bar, val in zip(bars, correlations):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('lhs_sensitivity_all_modes.png', dpi=150, bbox_inches='tight')
        print("  Saved lhs_sensitivity_all_modes.png")
        plt.close()
        
        # =====================================================================
        # PLOT 3: All Modes Sensitivity Heatmap (2x5)
        # =====================================================================
        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        fig.suptitle('Parameter Correlation with Each Mode', fontsize=14, fontweight='bold')
        
        corr_matrix = np.zeros((N_MODES, 5))
        for mode_idx in range(N_MODES):
            for param_idx in range(5):
                corr, _ = spearmanr(lhs_physical[:, param_idx], freq_results[:, mode_idx])
                corr_matrix[mode_idx, param_idx] = corr
        
        for mode_idx in range(N_MODES):
            row, col = divmod(mode_idx, 5)
            ax = axes[row, col]
            im = ax.imshow(corr_matrix[mode_idx, :].reshape(1, -1), 
                          cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_yticks([0])
            ax.set_yticklabels([f'Mode {mode_idx+1}'])
            ax.set_xticks(range(5))
            ax.set_xticklabels(param_labels, fontsize=9)
            
            for j in range(5):
                ax.text(j, 0, f'{corr_matrix[mode_idx, j]:.3f}', 
                       ha='center', va='center', fontsize=10,
                       color='white' if abs(corr_matrix[mode_idx, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        plt.savefig('lhs_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        print("  Saved lhs_correlation_heatmap.png")
        plt.close()
        
        # =====================================================================
        # PLOT 4: Parameter Distribution Comparison
        # =====================================================================
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        fig.suptitle('Parameter Distributions in LHS Samples', fontsize=14, fontweight='bold')
        
        for idx, (ax, name, label) in enumerate(zip(axes, param_names, param_labels)):
            ax.hist(lhs_physical[:, idx], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_xlim(PARAM_RANGES[name])
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = lhs_physical[:, idx].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'mean={mean_val:.3f}')
            ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('lhs_distributions.png', dpi=150, bbox_inches='tight')
        print("  Saved lhs_distributions.png")
        plt.close()
        
        # =====================================================================
        # PLOT 6: Mode Shape Sensitivity to Parameters
        # =====================================================================
        print("  Computing mode shape sensitivity...")
        
        def compute_mode_shape_rms(modes):
            """Compute RMS of mode shapes for each mode."""
            return [np.sqrt(np.mean(m**2)) for m in modes]
        
        print(f"  Computing mode shapes for {N_SAMPLES} samples...")
        mode_shapes_list = []
        for idx in range(N_SAMPLES):
            alpha, beta, theta_c, eta1, eta2 = lhs_physical[idx]
            freqs, modes = compute_fsdt_freq(alpha, beta, theta_c, eta1, eta2, 
                                              return_mode_shapes=True)
            mode_shapes_list.append(modes)
        
        mode_rms = np.zeros((N_SAMPLES, N_MODES))
        for idx in range(N_SAMPLES):
            if not np.isnan(mode_shapes_list[idx][0]).all():
                rms_vals = compute_mode_shape_rms(mode_shapes_list[idx])
                mode_rms[idx] = rms_vals
        
        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        fig.suptitle('Mode Shape RMS Sensitivity to Parameters (All Modes)', 
                     fontsize=14, fontweight='bold')
        
        for mode_idx in range(N_MODES):
            ax = axes.flat[mode_idx]
            correlations = []
            for p_idx in range(5):
                corr, _ = spearmanr(lhs_physical[:N_SAMPLES, p_idx], mode_rms[:, mode_idx])
                correlations.append(abs(corr))
            
            colors = ['red' if c == max(correlations) else 'steelblue' for c in correlations]
            bars = ax.bar(param_labels[:5], correlations, color=colors, edgecolor='black')
            ax.set_ylabel('|Spearman Correlation|', fontsize=10)
            ax.set_title(f'Mode {mode_idx+1}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', labelsize=8)
            
            for bar, val in zip(bars, correlations):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('lhs_mode_shape_sensitivity.png', dpi=150, bbox_inches='tight')
        print("  Saved lhs_mode_shape_sensitivity.png")
        plt.close()
        
    except Exception as e:
        print(f"  (Plotting skipped: {e})")
        import traceback
        traceback.print_exc()
    
    print()
    print("Done! Next steps:")
    print("  1. Check lhs_samples.csv for parameter combinations")
    print("  2. Run COMSOL with the same parameter combinations")
    print("  3. Compare COMSOL frequencies with FSDT frequencies in lhs_fsdt_results.csv")


if __name__ == '__main__':
    main()
