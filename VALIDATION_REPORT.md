# Validation Report: FSDT Supersonic Flutter Code Audit

## 1. Changes Made

### 1.1 Dense eigensolver (solver.py:400-421)
**Before:** `slinalg.eigs(A, n_eigs, sigma=0.1)` — ARPACK shift-invert finds eigenvalues near σ=0.1, contaminated by penalty boundary spring modes.

**After:** `linalg.eig(A)` for state dim ≤ 2000, ARPACK fallback for larger.

**Evidence of contamination:**
```
V=2000, 5mm CCCC Al plate:
  Dense eigvals:    max_re = +0.021 (40 modes with Re > 1e-6)
  ARPACK sigma=0.1: max_re = +686   (8 modes)
  ARPACK which=LR:  max_re = +2.3e12 (penalty eigenvalues)
```

### 1.2 Aerodynamic derivative fix (solver.py:283-284)
**Before:** `vv_sin = self._expanded[(1, 0)] * sin(α)` — ∫ N_{i,x}·N_j dA (x-derivative on test function)

**After:** `vv_sin = self._expanded[(0, 2)] * sin(α)` — ∫ N_i·N_{j,y} dA (y-derivative on trial function)

**Impact:** Silent for flow_angle=0 (sin(0)=0). Wrong for angled flow.

### 1.3 Stability tolerance (solver.py:437)
**Before:** `eigvals.real < 0`
**After:** `eigvals.real < 1e-6`

### 1.4 Precomputed aerodynamic bases (solver.py:283-295)
Replaced `expand_basis()` calls with `self._expanded[(0,1)]`, `self._expanded[(0,2)]`, `self._expanded_mass` — all precomputed in `__init__`.

---

## 2. Validation Against Literature

### 2.1 λ_cr for solid aluminum plates

| Reference | Config | λ_cr | Method |
|-----------|--------|------|--------|
| Song & Li (2013) | 0.1m×0.1m, 1mm, SSSS, E=210GPa | 512 | FEM |
| Song & Li (2013) | 0.1m×0.1m, 1mm, SCSC, E=210GPa | 546 | FEM |
| Song & Li (2013) | 0.1m×0.1m, 1mm, SFSF, E=210GPa | 336 | FEM |
| Nazemizadeh (2023) | 1m×1m, 5mm Al, CFCF | 615 | DQM |
| **Ours** | 0.3m×0.3m, 5mm Al, CCCF | **699** | **Dense eig** |

**Conclusion:** λ_cr values are consistent. The difference (615 vs 699) is expected: CCCF (3 clamped) is stiffer than CFCF (2 clamped).

### 2.2 Velocity scaling

V_cr ≈ λ_cr · D / (ρ · a · L³)

For 5mm Al, D=818.3 N·m:
- L=1m: V_cr ≈ 615·818.3/(1.225·343·1) ≈ 1198 m/s ≈ Mach 3.5 ✓ (matches literature)
- L=0.3m: V_cr ≈ 699·818.3/(1.225·343·0.027) ≈ 53,800 m/s ≈ Mach 157 ✓ (matches our code)

### 2.3 Aerodynamic damping limit

At high Mach: damp_coeff → ρ∞ · a∞

For 5mm Al: -ρ∞·a∞/(2·m_A) = -1.225·343/(2·13.55) = -15.50 s⁻¹

Observed: -14.88 to -14.93 s⁻¹ (within 4%)

For sandwich: -ρ∞·a∞/(2·m_A) = -27.48 s⁻¹

Observed: -26.3 s⁻¹ (within 5%)

---

## 3. Numerical Health

| Metric | Value | Concern? |
|--------|-------|----------|
| cond(K_base) | 1.76e11 | High (penalty k=1e14) |
| cond(M_mat) | 1.73e8 | Acceptable |
| Conjugate pairs | All complex, valid | No |
| C_air PSD | Yes (min eig ≥ 0) | No |
| Free-free rigid body modes | 2 exactly zero, 4 near-zero | No |

---

## 4. P4 Pipeline Architecture

### 4.1 Components

```
FSDTSolver.solve_complex_modal(v)
    ↓
generate_transient_clip(solver, v)
    ↓ eigenvalues + mode_shapes
    ↓ modal superposition: w(t) = Σ Re[q_k · exp(λ_k·t) · φ_k(sensor)]
    ↓
TransientClip (sensor_signals, margin)
    ↓
DomainRandomizer (noise, gain, resampling)
    ↓
TCNBackbone (causal conv → features)
    ↓
QuantileMarginModel (features → q0.05, q0.50, q0.95 per timestep)
    ↓
Pinball loss → training
    ↓
evaluate_coverage (MAE, interval width, per-quantile coverage)
```

### 4.2 What P4 tests verify (28 tests)

| Test | What it checks |
|------|---------------|
| Transient clip shapes | Output dimensions correct |
| Clip not all NaN | Signals are finite |
| generate_dataset | Batch generation works |
| margin value | (u_crit - v) / u_crit formula |
| Eigenvalue sign convention | Stable → decay, unstable → growth |
| Domain rand: shapes | Perturbation preserves dimensions |
| Domain rand: noise | Changes signal |
| Domain rand: gain | Scales signal |
| Domain rand: resample | Preserves shape |
| Domain rand: deterministic | Same seed → same output |
| TCN output shape | (B, hidden, T) |
| TCN causal constraint | No future leakage |
| TCN param count | < 200K |
| TCN gradient flows | Backprop works |
| Quantile head shape | (B, n_quantiles, T) |
| Quantile ordering | q0.05 ≤ q0.50 ≤ q0.95 |
| Pinball loss nonneg | Loss ≥ 0 |
| Pinball loss zero | pred=target → loss≈0 |
| evaluate_coverage | Returns dict with mae, coverage |
| Training convergence | Loss decreases |
| **P4 Behavioral tests** | |
| TCN improves with training | MAE decreases |
| Median ranking matches velocity | Lower v → higher margin prediction |
| Interval coverage ≥ 75% | Quantile intervals contain truth |
| Calibration q0.05 | ~5% below prediction |
| DR reduces perturbed MAE | Augmentation helps |
| Gain perturbation robustness | ±5% gain → MAE change < 20% |
| Pinball asymmetric response | Under-prediction penalised more at q0.95 |
| Quantile ordering preserved | After training |

### 4.3 What P4 tests DON'T verify (gaps)

1. **No test with real solver eigenvalues + real flutter margin** — Training tests use `_SyntheticClip` with synthetic signals, not real solver output. The `TestTransient` tests use real solver but force `clip.margin` manually.

2. **No test of `generate_transient_clip` with dense eigensolver** — The transient generation calls `solver.solve_complex_modal()` which now uses dense eigvals. No test verifies that the transient signals from dense eigvals are physically correct (decaying for stable, growing for unstable).

3. **No test of margin prediction accuracy on real data** — The `test_median_prediction_ranking_matches_velocity` test uses synthetic sine patterns, not real modal superposition signals. We haven't validated that a model trained on real solver output can predict margins accurately.

4. **No test of domain randomization on real solver signals** — DR tests use synthetic clips. Real solver signals have different statistics (exponential decay/growth, oscillatory patterns).

5. **No end-to-end test: solver → clips → train → predict margin** — The full pipeline has never been run with the fixed dense eigensolver on a config that actually flutters within a testable velocity range.

### 4.4 Recommendations for validation

To validate P4 end-to-end, we need a config that:
- Flutters at a testable velocity (Mach 2-5, not Mach 150+)
- Uses CFFF or CFCF boundary (not CCCC which is too stiff)
- Has enough modes for meaningful transient signals

From literature: 1m×1m, 5mm Al, CFCF → Mach 3.4 flutter. This is testable.

---

## 5. Open Questions for Advanced AI

### 5.1 Eigenvector phase normalization
Currently: `coeffs[k] = eigvecs_physical[:, k].real`
AI recommendation: phase-normalize first: `q *= exp(-j*angle(q[argmax(|q|)]))`
Question: Does taking `.real` without phase normalization affect the physical correctness of the transient signals? For a non-conservative system, the mode shape is complex-valued — the real part alone may not represent the actual mode shape at any particular instant.

### 5.2 Mode tracking across velocities
Currently: modes are re-sorted by frequency at each velocity independently. This causes mode labels to swap at crossings.
Question: For P4 training, does mode swapping matter? The model sees per-clip signals, not cross-velocity mode tracking. But if the same physical mode gets different indices at different velocities, the training data has inconsistent mode-to-signal mapping.

### 5.3 Transient signal normalization
Currently: `w = w / peak` normalizes per-sensor to [−1, 1].
Question: Does this normalization destroy the amplitude information that encodes the stability margin? The margin is encoded in the exponential growth/decay rate, not the absolute amplitude. But after normalization, a margin=0.1 signal looks identical to a margin=0.9 signal (both normalized to [−1, 1]).

### 5.4 Quantile head monotonicity
The quantile head uses differentiable monotonicity enforcement via `cumsum(softplus(diffs))`. Question: Is this the right inductive bias? For margin prediction, we expect q0.05 < q0.50 < q0.95 — but the intervals should widen near the flutter boundary (more uncertainty). Does the current architecture capture this?

### 5.5 Pinball loss at boundary cases
When margin=0 (at flutter), the target is 0. The model should predict q0.50≈0 with wide intervals. Question: Is pinball loss well-behaved at margin=0? The loss function is piecewise linear — does it have sufficient gradient signal at the boundary?

### 5.6 C_air at near-sonic speeds
At M≈1, `damp_coeff = A_dyn * (M²-2)/((M²-1)*V)` has a singularity at M=1. Our code raises ValueError for M≤1. Question: Is the M>1 check sufficient, or should we also check M > √2 (where damp_coeff changes sign)? For √2 > M > 1, the damping is negative (destabilizing), which is physically correct but may cause numerical issues in training.
