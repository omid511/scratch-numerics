# P3 Review R2: Robust Design Under Stochastic Boundary Conditions

## Summary

Re-review after first review flagged 2 critical issues. **Issue #1 (kernel sign)
was a false positive — the code is correct.** Issue #2 (hardcoded dimensions) is
partially fixed: sweep.py now uses solver attributes, but train.py still
hardcodes L1=0.3, L2=0.3 in three places. Tests pass (26/26) but remain mostly
shape/finiteness checks.

## Critical Issues Status

### 1. Kernel sign error — FALSE POSITIVE (no fix needed)

`gp_surrogate.py:82-87` — First review claimed the `(f,∇f)` and `(∇f,f)` blocks
need opposite signs. **This is wrong.** The current code is mathematically correct:

```
K_d = K(x,x') · (x - x') / l²    # ∂K/∂x' (derivative wrt 2nd argument)
K_aug[:n, n:] = K_d.T             # Cov(f(x), ∇f(x'))  ← correct
K_aug[n:, :n] = K_d               # Cov(∇f(x), f(x'))  ← correct
```

The two blocks are **transposes** of each other. A valid kernel matrix requires
`K_aug = K_aug^T`. With the first review's proposed negation:

```
K_aug[:n, n:] = -K_d.T    # would break symmetry
K_aug[n:, :n] = -K_d
```

This gives `K_aug^T ≠ K_aug` — not a valid kernel. The RBF antisymmetry
(∂K/∂x' = -∂K/∂x) is already encoded in K_d's definition, not in the block
signs. **The "fix" would break gradient-enhanced GP. Do not apply.**

Verified: `test_gradient_enhanced_fit` produces finite predictions. The test is
weak (checks finiteness, not accuracy) but the kernel itself is correct.

### 2. Hardcoded L1/L2 in train.py — NOT FIXED

`sweep.py:90` correctly uses `solver.L1, solver.L2`. But train.py still
hardcodes `0.3`:

| Location | Line | Code |
|----------|------|------|
| `active_learning_loop` | 91 | `FSDTSolver(L1=0.3, L2=0.3, ...)` |
| `active_learning_loop` | 95 | `build_boundary_springs(solver.L1, solver.L2, ...)` ← uses solver attrs for spring placement, but solver created with hardcoded dims |
| `brute_force_mc` | 174 | `FSDTSolver(L1=0.3, L2=0.3, ...)` |
| `brute_force_mc` | 178 | `build_boundary_springs(solver.L1, solver.L2, ...)` ← same pattern |

The solver is created with hardcoded 0.3, so `solver.L1`/`solver.L2` happen to
return 0.3 too — functionally identical but breaks if caller passes different
dimensions. **Fix: accept L1, L2 as function parameters.**

Also: `active_learning_loop` (line 91) hardcodes `M=8, N=8` — same issue.

## Remaining Issues from R1 (Still Open)

| # | Issue | Status |
|---|-------|--------|
| 3 | `active_learning_loop` accesses `gp._X_train`, `gp._y_train` (private) | OPEN |
| 4 | Candidate removal uses approximate nearest-neighbor (fragile) | OPEN |
| 5 | `select_batch` greedy without deduplication | OPEN |
| 6 | `log_marginal_likelihood` rebuilds K from scratch | OPEN |
| 7 | Dead code: `sweep.py:87` `springs = []` | OPEN |

## New Issues

### 8. GP accuracy not validated with increasing data

No test or demo shows RMSE decreasing as training points increase. The
`evaluate_gp_accuracy` function exists in train.py but has zero tests. This is
the core claim of P3 — that GP surrogate converges to solver accuracy — and it's
unverified.

### 9. `boundary_seeking` normalization questionable

`active_learning.py:58`: `exp(-0.5 * ((μ - target)/σ)²) * σ` — normalizing
distance by σ makes far-away high-σ points dominate over boundary-proximate
points with moderate σ. Consider `exp(-0.5 * (μ - target)²) * σ` (no division
by σ).

### 10. `select_batch` returns points in score order, not spatial order

`active_learning.py:91`: `argsort` returns highest-score first. If batch is
used sequentially, the first point is always the "best" — no diversity. For
batch active learning, consider k-DPP or simple max-distance deduplication.

## Test Coverage Assessment

**26/26 pass. Mostly shape/finiteness checks.**

| What's tested | What's missing |
|---------------|----------------|
| MAC identity/orthogonality | `track_modes_across_velocity` smoothness |
| Hungarian assignment | Mode label consistency across velocity steps |
| GP fit/predict shapes | GP accuracy on known function (RMSE vs. n_train) |
| GP interpolation (known linear) | GP accuracy improvement with more data |
| GP extrapolation uncertainty | Gradient-enhanced GP accuracy (only finiteness checked) |
| Acquisition function shapes | `select_batch` picks high-uncertainty points |
| Sweep output shapes | Flutter λ values physically sensible (>0) |
| | `evaluate_gp_accuracy` — zero tests |
| | `active_learning_loop` — RMSE decrease over iterations |
| | `brute_force_mc` — failure_probability in [0,1] |

### Behavioral tests needed

1. **GP convergence**: Fit GP on N=10,20,50,100 points from a known 2D function.
   Assert RMSE decreases monotonically.

2. **Mode tracking smoothness**: Sweep velocity 0→1000, assert |Δf|/Δv < threshold
   for each tracked mode (no frequency jumps).

3. **Sweep physics**: Assert all successful flutter λ > 0. Assert n_success/n_total > 0.5
   for well-conditioned configs.

4. **Active learning**: Run 3 iterations, assert total RMSE at end < initial RMSE.

5. **Gradient-enhanced accuracy**: Fit gradient GP on `f(x,y) = x + 0.5y`,
   predict at test points, assert RMSE < 0.01 (not just finite).

## Verdict

**One genuine unfixed issue** (hardcoded dimensions in train.py). **One false
positive** (kernel sign was correct all along). **Test coverage is weak** — shape
checks only, no behavioral validation of the core P3 claims (GP convergence,
mode tracking smoothness, active learning improvement).

Priority:
1. Fix hardcoded L1/L2/M/N in train.py (add parameters)
2. Add 3-5 behavioral tests (GP convergence, mode smoothness, sweep physics)
3. Add GP accuracy test for gradient-enhanced variant
