# Proposal 3: Mode-Aware Reliability Surrogate — Implementation Roadmap

## Project Charter

**Primary claim:** A mode-decomposition Gaussian process surrogate can predict
flutter safety margins (λ_cr) across stochastic boundary stiffnesses more
accurately than a standard GP, by explicitly tracking which physical mode is
dominant and learning each mode's response surface separately.

**MVP:** Gradient-enhanced GP surrogate that predicts λ_cr(x) for 2–4 varying
boundary stiffnesses, with reliability-oriented active learning. This is the
fallback and the minimum shippable product.

**Upgrade path:** Mode-decomposition surrogate where f_A(x), f_B(x) are
learned per-mode and λ_cr(x) = min(f_A(x), f_B(x)), with active learning
targeting the crossing locus f_A(x) = f_B(x).

**Dependencies:**
- `scikit-learn` (GaussianProcessRegressor) — already commonly available
- `scipy.spatial` (Latin Hypercube Sampling via `scipy.stats.qmc` — stdlib)
- No new heavy dependencies required

**Stop condition:** Gradient-enhanced GP outperforms standard GP on held-out
test set by ≥10% RMSE reduction, OR mode-decomposition variant outperforms
gradient-enhanced GP by ≥15%. If neither threshold is met after Phase 5,
the standard GP baseline is the deliverable.

---

## Module Breakdown

All new code lives under `src/mechanics/`. No existing files modified.

| File | Contents | Lines (est.) |
|------|----------|--------------|
| `surrogate/__init__.py` | Package init | ~5 |
| `surrogate/design_space.py` | Parameter definitions, LHS sampling, design point dataclass | ~120 |
| `surrogate/sweep.py` | Parameter sweep runner: calls `find_flutter_boundary` at each point, stores results | ~100 |
| `surrogate/mode_tracking.py` | MAC-optimal mode assignment across design parameter space (not just velocity) | ~150 |
| `surrogate/gp_standard.py` | Standard GP baseline (RBF kernel, no gradient enhancement) | ~80 |
| `surrogate/gp_gradient.py` | Gradient-enhanced GP with reliability-oriented acquisition | ~150 |
| `surrogate/mode_gp.py` | Mode-decomposition GP: per-mode surrogates, crossing detection | ~180 |
| `surrogate/active_learning.py` | Acquisition functions: expected improvement for reliability, uncertainty sampling at mode crossings | ~120 |
| `surrogate/baselines.py` | Brute-force Monte Carlo, polynomial chaos expansion baselines | ~100 |
| `surrogate/validate.py` | Coverage tests, comparison framework, hold-out validation | ~100 |
| `tests/test_surrogate_*.py` | Tests per module (see testing strategy) | ~50 each |

**Total new code:** ~1,300–1,500 lines across 10 source files + 5–6 test files.

---

## Implementation Phases

### Phase 1: Parameter Sweep Infrastructure

**Goal:** Run the solver at 200+ design points with varying boundary
stiffnesses and store results.

**Milestone:** `surrogate/sweep.py` runnable as standalone script, producing
`data/proposal3/sweep_results.npz` with full provenance.

**Steps:**
1. Create `surrogate/design_space.py`
   - `DesignPoint` dataclass: `k_left_bend: float, k_right_bend: float,
     k_left_shear: float, k_right_shear: float` (4 params, typical range
     1e6–1e12 N/m)
   - `DesignSpace` class: defines bounds, LHS sampling via
     `scipy.stats.qmc.LatinHypercube`
   - Log-scale support (sample in log10(k), transform to k)
   - Validation: bounds checking, condition number estimate

2. Create `surrogate/sweep.py`
   - `SweepRunner` class:
     - Takes `DesignSpace`, `FSDTSolver` template, velocity range
     - At each design point: set boundary springs, run
       `find_flutter_boundary`, store λ_cr + number of bisection iterations
     - Progress tracking, checkpoint/resume (save partial results every 50
       points)
     - Parallelism: sequential for now, `joblib` later if needed
   - `run_sweep(n_points=200)` entry point
   - Output format: structured numpy array with design coordinates + λ_cr +
     solve metadata

3. Integrate with existing `ExperimentConfig` for provenance

**Key design decision:** Use `find_flutter_boundary` (bisection) rather than
scanning velocity at each point. Bisection is ~10x faster per λ_cr estimate.

** ponytail:** No parallelism yet. Add when sweep takes >30 min.

---

### Phase 2: Mode Identity Extraction Feasibility

**Goal:** Determine if mode shapes can be consistently identified across the
design parameter space (not just velocity sweeps).

**Milestone:** Pass/fail verdict on mode-decomposition approach, documented in
`surrogate/mode_tracking.py`.

**Steps:**
1. Create `surrogate/mode_tracking.py`
   - Port MAC computation from `tests/test_mode_tracking.py`
   - `ModeTracker` class:
     - Maintains reference mode library from a fixed design point
     - At new design point: compute MAC matrix, assign modes via Hungarian
       algorithm
     - Track mode identity across 2D design space (not just 1D velocity
       sweep)
   - Feasibility test:
     - Run 50 design points in a small region of parameter space
     - Measure MAC consistency: what fraction of points have MAC > 0.8
       between assigned modes?
     - If MAC drops below 0.5 at >30% of points → mode-decomposition is
       unreliable → stick with gradient-enhanced GP

2. Decision gate:
   - MAC consistency ≥ 70% across design space → proceed to Phase 5
   - MAC consistency < 70% → skip Phase 5, go directly to Phase 6
   - Record decision in `surrogate/feasibility_report.json`

**Key finding from codebase:** `solve_complex_modal` returns `mode_shapes` as
(n_modes, grid_ny, grid_nx) arrays. The MAC test in
`test_mode_tracking.py:21-24` already works. The gap: no tool to track mode
identity across *design parameter* variations (only velocity variations exist
today).

** ponytail:** Feasibility check is 50 runs, not 200. Enough to see pattern.

---

### Phase 3: Gradient-Enhanced GP (Fallback Path)

**Goal:** Implement GP surrogate with gradient information for smooth
interpolation, including the gradient-enhanced kernel.

**Milestone:** GP predicts λ_cr(x) with RMSE < 5% of λ_cr range on hold-out
set.

**Steps:**
1. Create `surrogate/gp_standard.py`
   - Standard RBF-kernel GP via `sklearn.gaussian_process.GaussianProcessRegressor`
   - Kernel: `ConstantKernel * RBF + WhiteKernel`
   - Input: design parameters (log-transformed stiffnesses)
   - Output: λ_cr
   - Hyperparameter optimization via `marginal_log_likelihood`
   - `StandardGP` class with `fit()`, `predict()`, `predict_with_std()`

2. Create `surrogate/gp_gradient.py`
   - `GradientGP` class extending standard GP
   - Kernel: `GradientKernel(RBF)` — product kernel with derivative
     observations
   - Implementation approach: augment training data with finite-difference
     gradients
     - At each training point, perturb each parameter by δ = 1e-4 × range
     - Compute λ_cr at perturbed points (already in sweep data or computed
       on-the-fly)
     - Stack: Y = [λ_cr; ∂λ_cr/∂x₁; ...; ∂λ_cr/∂xₙ]
   - Alternatively: use `GPy` or manual implementation if sklearn doesn't
     support gradient observations cleanly
   - Acquisition function for active learning: predictive variance

3. Validate standard GP vs gradient-enhanced GP on sweep data from Phase 1
   - Hold-out 20% for test
   - Compare RMSE, MAE, max error, calibration (PICP, MPIW)

** ponytail:** Standard GP first. Gradient-enhancement only if standard GP
RMSE > 5% range. Sklearn's GP is fine at n<500.

---

### Phase 4: Active Learning with Reliability-Oriented Sampling

**Goal:** Select new sample points that improve surrogate accuracy near the
failure boundary (λ_cr ≈ threshold).

**Milestone:** Active learning loop reduces RMSE near failure boundary by ≥30%
compared to random sampling with same budget.

**Steps:**
1. Create `surrogate/active_learning.py`
   - `ReliabilityAcquisition` class:
     - Input: fitted GP, safety threshold λ_target
     - Computes acquisition value at candidate points:
       - **Uncertainty sampling:** σ(x) — high-variance regions
       - **Expected improvement for reliability:** P(λ_cr(x) < λ_target) ×
         (λ_target - μ(x))
       - **Boundary seeking:** μ(x) ≈ λ_target with high σ
   - `ActiveLearner` class:
     - Loop: fit GP → compute acquisition → sample top-k points → retrain
     - Max iterations: 10 (configurable)
     - Convergence: stop when acquisition max < 1e-3 or RMSE improvement < 1%
     - Batch acquisition: batch-size points per iteration (batch mode via
       kriging believer or simple top-k)

2. Integration with sweep:
   - Initial 200-point LHS from Phase 1
   - Active learning adds 20–50 points per iteration
   - Total budget: 400–700 solver calls

3. Failure probability estimation:
   - Given GP posterior, compute P_f = P(λ_cr(x) < λ_target)
   - Importance sampling or direct integration over parameter space
   - Compare with brute-force MC from Phase 6

** ponytail:** Top-k acquisition, no batch-optimal. Kriging believer is
overkill for 4 parameters.

---

### Phase 5: Mode-Decomposition Surrogate (Conditional on Phase 2)

**Goal:** Learn per-mode response surfaces and combine via min() for
λ_cr prediction.

**Milestone:** Mode-decomposition GP outperforms gradient-enhanced GP by ≥15%
RMSE reduction on held-out test set near mode crossing regions.

**Steps:**
1. Create `surrogate/mode_gp.py`
   - `ModeDecompositionGP` class:
     - Identify which modes are relevant (typically 2–3 lowest modes)
     - For each mode i, fit GP: f_i(x) → frequency or damping at fixed
       velocity
     - Or directly: f_i(x) → λ_cr contribution from mode i
     - Two sub-approaches:
       a. **Frequency-based:** Learn ω_i(x, V) for each mode, then find
          flutter crossing per mode → λ_cr_i = crossing of mode i
       b. **Direct λ_cr per mode:** At each sweep point, determine which
          mode caused flutter, fit GP per mode
     - Approach (b) is simpler — use if Phase 2 feasibility passes

2. Crossing detection:
   - At prediction: evaluate all mode GPs, compute λ_cr_i for each
   - λ_cr(x) = min_i(λ_cr_i(x))
   - Detect crossing: where |λ_cr_A(x) - λ_cr_B(x)| < ε
   - At crossings: flag for direct solver validation

3. Reliability analysis:
   - Same active learning framework from Phase 4
   - Additional acquisition: sample near predicted crossing locus
   - Compare with gradient-enhanced GP

** ponytail:** Direct λ_cr-per-mode (approach b). Frequency-based (a) needs
velocity-dependent GP — much more complex.

---

### Phase 6: Validation Framework

**Goal:** Rigorous comparison of all methods against brute-force reference.

**Milestone:** Validation report with quantitative comparison tables and
coverage analysis.

**Steps:**
1. Create `surrogate/baselines.py`
   - **Brute-force MC:** Sample 10,000 points from parameter space, compute
     λ_cr at each → reference P_f estimate
   - **Polynomial Chaos Expansion:** `scipy.interpolate` or manual
     implementation for comparison with GP methods
   - Both baselines consume same total solver budget as GP approach

2. Create `surrogate/validate.py`
   - `ValidationSuite` class:
     - Hold-out test set: 20% of Phase 1 sweep data
     - Metrics per method:
       - RMSE, MAE, max error on hold-out set
       - Failure probability estimate vs brute-force reference
       - Computational cost (solver calls)
     - Coverage analysis:
       - At what fraction of parameter space is surrogate error < 5%?
       - Where does it fail? Near mode crossings? Near boundaries?
   - `run_validation()` entry point: runs all methods, produces comparison
     table

3. Test coverage:
   - Unit tests for each module
   - Integration test: end-to-end sweep → GP → active learning → validation
   - Regression test: known λ_cr for specific parameter combinations

**Tests per phase:**
| Phase | Test file | Key tests |
|-------|-----------|-----------|
| 1 | `test_design_space.py` | LHS coverage, bounds checking, log-transform |
| 1 | `test_sweep.py` | Single-point solve, checkpoint/resume, provenance |
| 2 | `test_mode_tracking.py` | MAC computation, Hungarian assignment, feasibility |
| 3 | `test_gp_standard.py` | Fit/predict, hyperparameter optimization, hold-out RMSE |
| 3 | `test_gp_gradient.py` | Gradient kernel, augmented training data |
| 4 | `test_active_learning.py` | Acquisition functions, convergence, batch selection |
| 5 | `test_mode_gp.py` | Per-mode fitting, min() combination, crossing detection |
| 6 | `test_validate.py` | Comparison metrics, coverage, brute-force reference |

---

## Baselines

| Method | Solver calls | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Brute-force MC** | 10,000 | Reference truth, no modeling | Expensive, no interpolation |
| **Standard GP** | 200–400 | Fast inference, uncertainty | Fails at mode crossings |
| **Polynomial Chaos** | 200–400 | Fast, good for smooth | Poor at discontinuities |
| **Gradient-enhanced GP** | 200–400 | Better with derivatives | Same kernel limitation |
| **Mode-decomposition GP** | 200–400 | Handles mode crossings | Requires mode identity |

---

## File Structure

```
src/mechanics/
├── surrogate/
│   ├── __init__.py
│   ├── design_space.py
│   ├── sweep.py
│   ├── mode_tracking.py
│   ├── gp_standard.py
│   ├── gp_gradient.py
│   ├── mode_gp.py
│   ├── active_learning.py
│   ├── baselines.py
│   └── validate.py
tests/
├── test_design_space.py
├── test_sweep.py
├── test_mode_tracking.py
├── test_gp_standard.py
├── test_gp_gradient.py
├── test_active_learning.py
├── test_mode_gp.py
└── test_validate.py
data/proposal3/
├── sweep_results.npz
├── feasibility_report.json
└── validation_results.json
```

---

## Stopping Points

| Gate | Criterion | Action if met |
|------|-----------|---------------|
| After Phase 1 | Sweep fails to find λ_cr for >20% of points | Debug solver, adjust parameter ranges |
| After Phase 2 | MAC consistency < 70% | Skip Phase 5, gradient-enhanced GP is the product |
| After Phase 3 | Standard GP RMSE < 5% range | Gradient enhancement is optional polish |
| After Phase 4 | Active learning doesn't improve RMSE by ≥10% | Standard GP with initial 200 points is sufficient |
| After Phase 5 | Mode-decomposition doesn't beat gradient-enhanced GP | Gradient-enhanced GP is the product |
| After Phase 6 | No method beats brute-force MC by ≥2x cost reduction | Report: surrogate not worth the complexity |

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| 1: Sweep infrastructure | 2–3 days | None |
| 2: Mode tracking feasibility | 1–2 days | Phase 1 data |
| 3: Gradient-enhanced GP | 2–3 days | Phase 1 data |
| 4: Active learning | 2–3 days | Phase 3 |
| 5: Mode-decomposition GP | 3–4 days | Phases 2, 3 |
| 6: Validation | 1–2 days | All phases |
| **Total** | **11–17 days** | |

**Critical path:** Phase 1 → Phase 3 → Phase 4 → Phase 6
**Parallel:** Phase 2 can run alongside Phase 3 once Phase 1 data exists
**Conditional:** Phase 5 only if Phase 2 passes feasibility gate
