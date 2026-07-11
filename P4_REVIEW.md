# P4: Aeroelastic Margin Estimation — Comprehensive Review

## 1. Project Overview

**Repository**: `https://github.com/omid511/mechanics` (private, branch: `master`)

This project implements a fast FSDT (First-order Shear Deformation Theory) plate solver with four research proposals for aeroelastic analysis of honeycomb sandwich plates. The focus of this review is **Proposal 4 (P4): Stability Margin Estimation** — a machine-learning pipeline that predicts flutter margins from transient sensor signals.

### Physical System

- **Plate**: 1m × 1m honeycomb sandwich (Al face 5mm, honeycomb core 8mm)
- **Boundary conditions**: CFCF (clamped-free-clamped-free) — known to flutter
- **Flutter velocity**: u_crit ≈ 1455 m/s (Mach 4.28)
- **Solver**: FSDT with Legendre polynomial basis, penalty-spring boundary conditions

### P4 Pipeline

1. **Transient generation**: Propagate aeroelastic response via conjugate-pair modal superposition
2. **Domain randomization**: Noise, gain, resample perturbations for sim-to-real transfer
3. **TCN backbone**: Causal temporal convolutional network (4 layers, 32 channels)
4. **Quantile head**: Global average pooling → linear head → median-centered quantile output
5. **Training**: Pinball loss, Huber loss baseline, grouped train/val split

---

## 2. What Has Been Implemented

### 2.1 Core Solver (`src/mechanics/solver.py`)

**Generalized eigensystem** — uses `eig(A_comp, B_comp)` with companion pencil:
```
[0, I; -K, -C] z = s [I, 0; 0, M] z
```
Avoids explicit M⁻¹ and preserves conditioning.

**`_max_real_eigenvalue(velocity)`** — returns the least-stable physical eigenvalue real part. Filters:
1. Frequency range: 100–100000 rad/s (physical plate modes)
2. Transverse participation: η_w = ||q_w|| / ||q|| > 0.01
3. Eigenpair residual: r_k = ||(s²M + sC + K)q|| / (|s|²||Mq|| + |s|||Cq|| + ||Kq||) < 0.1

**`find_flutter_velocity()`** — scans velocity directly via spectral abscissa:
```
alpha(V) = max_k Re(lambda_k(V))
Locate first zero crossing: alpha(V_i) < 0 and alpha(V_{i+1}) >= 0
Bisect in velocity space to tol=1.0 m/s
```

**`_compute_D11()`** — bending stiffness from ABD matrix (ABD[3,3]).

**`assemble_aerodynamic()`** — piston theory stiffness and damping matrices. Only checks M>1.

### 2.2 Piston Theory (`src/mechanics/piston_theory.py`)

- `validate_mach()` with `min_mach=2.0` for strict validity
- `non_dimensional_lambda()` and `velocity_from_lambda()` for λ ↔ V conversion
- First-order supersonic piston theory pressure

### 2.3 Transient Generation (`src/mechanics/p4_margin_estimation/transient.py`)

**Full state propagation** via conjugate-pair modal superposition:
```python
# For each pair (λ, λ*), contribution is 2 Re[c_k v_k exp(λ_k t)]
# where c_k = (v_k^H x₀) / (v_k^H v_k) for right eigenvectors
```

**Eigenvalue filtering** (NOW consistent with `_max_real_eigenvalue`):
1. Frequency range: 100–100000 rad/s
2. Positive imaginary part (one-sided)
3. Transverse participation: η_w > 0.01
4. Eigenpair residual: r_k < 0.1

**Anti-aliasing**: Reject modes with freq > 0.4 × fs (fs=1024 Hz → cutoff=410 Hz)

**Unstable clip rejection**: `if np.any(eigvals.real > 0): raise ValueError(...)`

**Causal normalization**: Initial-window RMS (10% of clip)

**Margin calculation**: Uses `find_flutter_velocity()` for consistency

### 2.4 TCN (`src/mechanics/p4_margin_estimation/tcn.py`)

- Causal convolutions only (left padding, no future leakage)
- No normalization (GroupNorm removed — violated causality by normalizing over time axis)
- Residual connections
- 4 layers, dilation = 2^i

### 2.5 Quantile Head (`src/mechanics/p4_margin_estimation/quantile_head.py`)

**Global average pooling**: TCN output `(B, 32, 512)` → `.mean(dim=2)` → `(B, 32)` → linear head → `(B, n_q)`

**Median-centered parameterization**:
```
q0.50 = median (direct)
q0.05 = median - softplus(lower_raw)
q0.95 = median + softplus(upper_raw)
```

**Output**: Scalar margin prediction per clip, not time series.

### 2.6 Training (`src/mechanics/p4_margin_estimation/train.py`)

- Pinball loss for quantile regression
- NaN signal filtering
- Returns `(model, history)` tuple
- Cosine annealing LR scheduler

### 2.7 Domain Randomization (`src/mechanics/p4_margin_estimation/domain_randomization.py`)

- Additive Gaussian noise (sensor noise)
- Gain scaling (calibration error)
- Time resampling (variable sample rate) — resample to intermediate rate and back

### 2.8 Data Generation (`data_generation.py`)

**Bug fixes applied**:
1. **Parallel checkpoint corruption**: `as_completed` returns out-of-order; checkpoints now write full `n_samples` buffer
2. **Uninitialized mode shapes**: `np.empty` → `np.zeros`
3. **Sobol NaN→0 biasing**: Replaced with column median imputation
4. **Sensitivity CLI**: Was re-loading YAML from disk; now passes composed `cfg`

---

## 3. Current State and Test Results

### 3.1 P4 Tests

**Before eigenvalue filtering fix**: 28/28 pass

**After eigenvalue filtering fix**: 22/28 pass (6 fail)

**Failure analysis**: The test fixture uses a small 0.3m×0.3m CCCC plate with M=5, N=5 (25 basis functions). The companion pencil produces 250 eigenvalues. Of these:
- 25 pass `(residual < 0.1) & (eta_w > 0.01)` — mostly penalty-spring modes at high frequencies (700k–1.2M Hz)
- Only 9 pass the additional frequency filter (100–100000 rad/s)
- Of those 9, 3 are unstable (Re > 0) at V=800 m/s

The test velocities (800–1200 m/s) are too high for this plate under the new stricter filtering. The old filtering let some of these through; the new filtering (matching `_max_real_eigenvalue`) correctly identifies unstable modes and rejects clips.

**Root cause**: Test fixture needs lower velocities or a different plate configuration where all physical modes are stable.

### 3.2 P3 Tests

**40/40 pass** — all mode tracking, GP surrogate, active learning, and training tests pass.

### 3.3 Flutter Boundary

True flutter boundary for 1m CFCF plate:
- V=1450: α = -3.51 (stable)
- V=1460: α = -2.18 (stable)
- V=1470: α = +4.62 (unstable)
- Bisection: **u_crit ≈ 1455 m/s (Mach 4.28)**

The spectral abscissa oscillates at coalescence points (expected behavior — two modes swap dominance).

---

## 4. Issues Identified and Fixed

### 4.1 Eigenvalue Filtering Inconsistency (FIXED)

**Problem**: Transient generator used different eigenvalue selection criteria than `_max_real_eigenvalue`:
- Transient (old): `disp_norms > 1e-15`, `|Im| > 1`, `Im > 0`, `freq_hz <= 0.4*fs`
- `_max_real_eigenvalue`: `freq 100–100000 rad/s`, `η_w > 0.01`, `r_k < 0.1`

**Consequence**: At V=1376, `solve_complex_modal` showed Re=+22 for one mode, but `_max_real_eigenvalue` returned -25.54 (stable). The Re=+22 mode was non-physical — it passed transient filters but failed `_max_real_eigenvalue`'s η_w or residual check.

**Fix**: Updated `transient.py` to use the same filtering as `_max_real_eigenvalue` (frequency range, η_w, residual).

### 4.2 Aliased Modes (FIXED)

**Problem**: 1055.5 Hz mode aliased to ~32 Hz in the discrete signal (fs=1024 Hz).

**Fix**: Anti-aliasing filter rejects modes with freq > 0.4 × fs (410 Hz). Only the 4 physical modes below 410 Hz are propagated.

### 4.3 Unstable Clip Clipping (FIXED)

**Problem**: `np.clip(exponent, -500, 500)` produced numerical garbage for unstable modes.

**Fix**: Explicit rejection: `if np.any(eigvals.real > 0): raise ValueError(...)`. The caller catches and skips.

### 4.4 TCN Causality Violation (FIXED)

**Problem**: GroupNorm normalized over the time axis, violating causality (truncating input changed statistics for all timesteps).

**Fix**: Removed GroupNorm entirely. Small networks work fine without normalization.

### 4.5 Global Average Pooling (FIXED)

**Problem**: TCN output was time-series; quantile head operated per-timestep.

**Fix**: Global average pooling: `(B, 32, 512)` → `.mean(dim=2)` → `(B, 32)` → head → `(B, n_q)`. Scalar output.

### 4.6 Parallel Checkpoint Corruption (FIXED)

**Problem**: `as_completed` returns futures out-of-order, but checkpoints wrote rows as `completed`.

**Fix**: Checkpoints now write full `n_samples` buffer; `success` array marks valid rows.

### 4.7 Sobol NaN→0 Biasing (FIXED)

**Problem**: Sobol sequences can produce NaN values; replacing with 0 biased results toward zero.

**Fix**: Replaced with column median imputation.

---

## 5. Open Questions for Review

### 5.1 Physical Correctness

1. **Is the companion pencil formulation correct?** The generalized eigensystem `[0,I; -K,-C] z = s [I,0; 0,M] z` avoids M⁻¹, but the companion pencil has 2N eigenvalues for N physical DOFs. The spurious eigenvalues (infinite frequency, zero real part) are filtered out. Is this the standard approach for aeroelastic eigenvalue problems?

2. **Piston theory validity**: We enforce M ≥ 2.0 for strict validity. The flutter boundary at Mach 4.28 is well within this range. However, the training range extends down to Mach 2.0 (V=680 m/s). Is piston theory reliable at M=2.0 for this plate configuration?

3. **FSDT vs. CLT**: The solver uses First-order Shear Deformation Theory, which includes transverse shear deformation. For thick sandwich plates, this is more accurate than Classical Lamination Theory. However, the core material has very low shear moduli (G12=12 MPa, G13=G23=1.01 GPa). Is FSDT sufficient, or should we consider higher-order theories (e.g., Reddy's TSDT)?

4. **Penalty spring boundary conditions**: Clamped BCs are enforced via penalty springs (k=1e14). This is an approximation — true clamped BCs would constrain all DOFs at the boundary. How sensitive are the flutter results to the spring stiffness? Should we verify convergence with respect to k?

5. **Legendre basis completeness**: The solver uses M=6, N=6 Legendre polynomials (36 basis functions per DOF, 180 total DOFs for 5 DOFs). Is this sufficient for the mode shapes near flutter? The natural frequencies converge, but mode shapes near coalescence may require more basis functions.

### 5.2 Eigenvalue Filtering

6. **Frequency range (100–100000 rad/s)**: This is tuned for the 1m CFCF plate. For different plate sizes or materials, the physical frequency range would shift. Should the frequency range be adaptive (e.g., based on the undamped natural frequencies)?

7. **η_w threshold (0.01)**: Very low threshold — almost any mode with nonzero transverse displacement passes. Is this too permissive? Could it let through numerical artifacts?

8. **Residual threshold (0.1)**: Standard for eigenpair validation. But for poorly conditioned systems (high stiffness contrast between face and core), residuals may be systematically higher. Should we use a relative threshold?

9. **Anti-aliasing cutoff (0.4 × fs)**: Conservative — Nyquist is 0.5 × fs. The 0.4 factor provides margin for spectral leakage. Is this sufficient, or should we use a proper anti-aliasing filter before sampling?

### 5.3 Transient Generation

10. **Conjugate-pair propagation**: The propagation uses `2 Re[c_k v_k exp(λ_k t)]` for each conjugate pair. This assumes the eigenvectors are properly normalized. The coefficients are computed via pseudoinverse: `coeffs = pinv(eigvecs) @ x0`. Is this numerically stable for ill-conditioned eigenvector matrices?

11. **Initial conditions**: Random displacements in w-DOFs (0.1 × standard normal) and random velocities (0.01 × standard normal). The amplitude is arbitrary — the causal normalization scales to unit RMS in the initial window. Is this sufficient to excite all relevant modes?

12. **Sensor placement**: Fixed interior sensors via `default_sensor_xy()`. These avoid boundary nodes (where penalty springs dominate). But they may miss localized mode shapes near the clamped edges. Should we optimize sensor placement?

13. **Clip length**: 0.5 seconds, 512 timesteps (fs=1024 Hz). The shortest period is ~4.5 ms (220 Hz mode). 512 timesteps captures ~111 cycles of the highest-frequency mode. Is this sufficient for the TCN to learn the damping signature?

### 5.4 Machine Learning

14. **TCN architecture**: 4 layers, 32 channels, kernel=3, dilation=2^i. Receptive field = 30 timesteps. The clip has 512 timesteps. Is the receptive field sufficient to capture the exponential decay/growth envelope? A mode with Re=-50 has a time constant of 20 ms = 20 timesteps. The receptive field covers 1.5 time constants.

15. **Global average pooling**: All 512 timesteps contribute equally. But the early timesteps (transient) contain more information about damping than the late timesteps (steady-state or noise). Should we use attention or learned pooling instead?

16. **Quantile regression**: We predict q0.05, q0.50, q0.95. The median-centered parameterization ensures ordering. But the pinball loss treats all quantiles equally. Should we weight the lower quantile more heavily (conservative margin estimate)?

17. **Training data size**: 50 clips is small for a 4-layer TCN (~100K parameters). Domain randomization (noise, gain, resample) can augment the dataset. Should we apply it during training?

18. **Velocity range**: Training range is [680, ~1380] m/s. The margin range is [0.04, 0.53]. The eigenvalue Re changes from -20 to -25 across this range (only 25% relative change). Is this dynamic range sufficient for the TCN to learn?

19. **Huber vs. pinball loss**: AI Review #2 suggested starting with Huber loss for median prediction, then adding quantile head. We implemented pinball loss directly. Is this a problem?

20. **Grouped train/val split**: Clips from the same velocity should not appear in both train and val sets. The current implementation uses random split. Should we use grouped split by velocity level?

### 5.5 Data Generation

21. **Sobol sequence**: Used for quasi-random sampling of the velocity range. Sobol sequences have low discrepancy but can produce NaN values at singular points. The median imputation fix handles this, but could we use a better sampler (e.g., Latin Hypercube)?

22. **Parallel checkpoint corruption**: The `as_completed` pattern returns futures out-of-order. The fix writes the full buffer and marks valid rows. But this means each checkpoint rewrites the entire buffer. For large datasets, this is O(n²) in I/O. Should we use ordered completion or a lock-free data structure?

23. **Mode shape storage**: Mode shapes are stored as `(n_samples, M_eff, N_eff)` arrays. For M=6, N=6, grid=(30,30), this is 30×30×8 bytes = 7.2 KB per sample. For 10K samples, this is 72 MB. Is this acceptable?

### 5.6 Numerical Stability

24. **LU factorization caching**: The mass matrix LU factorization is cached lazily. But the aerodynamic matrices change with velocity. The companion pencil is rebuilt for each velocity. Is this efficient, or should we cache more?

25. **Eigenvector conditioning**: The generalized eigenvalue problem can produce poorly conditioned eigenvectors, especially near coalescence. The pseudoinverse (`pinv`) handles this, but may introduce numerical errors. Should we use SVD-based mode superposition instead?

26. **Exponent overflow**: The propagation computes `exp(λ_k.real * t)`. For unstable modes (Re > 0) at late times, this can overflow. The unstable clip rejection prevents this, but near the flutter boundary, Re is small and positive. Should we add a safety check for `max(Re) * t_end > 500`?

### 5.7 Validation

27. **No experimental data**: The entire pipeline is validated against the solver itself. There is no experimental data to compare against. How should we assess the real-world applicability?

28. **No Abaqus comparison**: P1 (multi-fidelity GP) was planned to use Abaqus HF data, but this doesn't exist yet. Should we generate Abaqus reference solutions for validation?

29. **Test coverage**: P4 has 28 tests (22 pass after filtering fix). P3 has 40 tests. P1 and P2 have their own test suites. But there are no integration tests for the full pipeline (data generation → training → evaluation). Should we add end-to-end tests?

### 5.8 Production Readiness

30. **Model serialization**: `torch.save(model.state_dict(), "p4_model.pt")` saves weights only. The model architecture is implicit. Should we save the full model or use a config file?

31. **Inference speed**: No benchmarking has been done. The TCN with 4 layers and 32 channels has ~100K parameters. Inference on a single clip should be fast (< 1 ms). But for real-time applications, we may need to optimize further.

32. **GPU support**: The training loop uses `device="cpu"`. Should we add CUDA support?

---

## 6. Next Steps

1. **Fix 6 failing P4 tests**: Adjust test velocities or plate configuration to work with the new eigenvalue filtering
2. **Run end-to-end P4 pipeline**: Generate data, train model, evaluate on held-out clips
3. **Implement cached eigensolution approach**: Solve once per velocity, generate multiple clips with randomized modal amplitudes/phases
4. **Restructure `run_p4.py`**: 24 velocity levels, 10 realizations each, grouped train/val split, Huber loss baseline
5. **Validate against known analytical solutions** (if available for simplified plate configurations)

---

## 7. File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `src/mechanics/solver.py` | FSDT solver, eigenvalue analysis, flutter boundary | 706 |
| `src/mechanics/piston_theory.py` | Supersonic piston theory | 129 |
| `src/mechanics/laminate.py` | Material, Laminate, ABD matrices | 190 |
| `src/mechanics/basis.py` | Legendre/trigonometric basis functions | — |
| `src/mechanics/boundary.py` | Penalty spring boundary conditions | — |
| `src/mechanics/p4_margin_estimation/transient.py` | Transient generation | 313 |
| `src/mechanics/p4_margin_estimation/tcn.py` | TCN backbone | 73 |
| `src/mechanics/p4_margin_estimation/quantile_head.py` | Quantile regression head | 87 |
| `src/mechanics/p4_margin_estimation/train.py` | Training loop, coverage evaluation | 159 |
| `src/mechanics/p4_margin_estimation/domain_randomization.py` | Sim-to-real perturbations | 80 |
| `src/mechanics/p4_margin_estimation/tests/test_p4.py` | P4 test suite | 488 |
| `src/mechanics/p3_robust_design/` | P3: sweep, mode tracking, GP, active learning | — |
| `src/mechanics/p3_robust_design/tests/test_p3.py` | P3 test suite | — |
| `data_generation.py` | Parallel data generation | — |
| `run_p4.py` | P4 execution script | 93 |

---

## 8. Constants and Configuration

```python
AIR_DENSITY = 1.2      # kg/m³
SOUND_SPEED = 340.0     # m/s
k_stiffness = 1e14      # penalty spring stiffness (N/m)

# Plate: 1m × 1m CFCF honeycomb sandwich
face = Material(E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9, nu12=0.33, rho=2710)
core = Material(E1=4.73e7, E2=4.73e7, G23=1.01e9, G13=1.01e9, G12=1.20e7, nu12=0.98, rho=278.15)
# Face thickness: 5mm, Core thickness: 8mm

# Solver: M=6, N=6 Legendre basis, grid=(30,30)
# True flutter: u_crit ≈ 1455 m/s (Mach 4.28)
# Training range: V ∈ [680, ~1400], margins ≈ [0.04, 0.53], M ≥ 2
```
