# P3 Review: Robust Design Under Stochastic Boundary Conditions

## Summary

Mode tracking (MAC + Hungarian) and GP surrogate are well-structured. The
active-learning loop and sweep pipeline are functional but have a critical
sign error in the gradient-enhanced kernel and hardcoded plate dimensions
that will break for non-0.3m plates.

## Critical Issues

### 1. Augmented kernel off-diagonal blocks have wrong sign

`gp_surrogate.py:82-87` — The `(f, ∇f)` and `(∇f, f)` blocks use `K_d.T`
and `K_d` respectively. For the RBF kernel, `∂K/∂x_i = -∂K/∂x_j`, so these
blocks should have **opposite** signs. The code gives them the same sign.

```
Correct:  [K, K_fg; -K_fg^T, K_gg]
Code:     [K, -K_fg^T; K_fg, K_gg]
```

This is `diag(I,-I) * M_correct * diag(I,-I)` — preserves eigenvalues so
Cholesky still works, but gradient-enhanced GP predictions will be wrong.
The `test_gradient_enhanced_fit` only checks finiteness, not accuracy, so
the bug is invisible.

**Fix**: `K_aug[:n, n:] = -K_d.T` and `K_aug[n:, :n] = -K_d`. Or equivalently
reshape `K_d` properly: `K_d.reshape(n,d,n).transpose(0,2,1).reshape(n,n*d)`
for the top-right, and its negative for bottom-left.

### 2. Hardcoded L1=0.3, L2=0.3 in solver construction

`sweep.py:89-96`, `train.py:91-98,174-181` — `build_boundary_springs` is
called with literal `0.3, 0.3` instead of `solver.L1, solver.L2`. If the
plate dimensions change, the springs are placed at wrong edge positions.

Also, `_make_solver` creates a solver with clamped BCs, then
`build_boundary_springs` + `set_boundary_springs` overrides everything —
the initial clamped BCs are dead setup.

## Improvements

### 3. `active_learning_loop` couples to GP internals

`train.py:78-79` — Accesses `gp._X_train` and `gp._y_train[:gp._n_train]`
(private attributes). If the GP's internal storage changes, this breaks.
Add a public `get_training_data()` method or pass training data explicitly.

### 4. Candidate removal uses approximate nearest-neighbor

`train.py:131-137` — Finds closest candidate by Euclidean distance to each
selected point. If two selected points are close to the same candidate, the
wrong one might be removed. Track by index instead:

```python
selected_idx = np.array([np.argmin(np.linalg.norm(cand_log10 - s, axis=1))
                          for s in selected])
mask[selected_idx] = False
```

### 5. `select_batch` greedy selection without deduplication

`active_learning.py:91` — Top-k by score with no deduplication. If the GP
has a flat uncertainty region, the top-k may be near-duplicates. Add
minimum-distance deduplication or use batch acquisition (e.g., k-DPP).

### 6. `log_marginal_likelihood` rebuilds kernel matrix

`gp_surrogate.py:231-234` — Rebuilds `K` from scratch when the Cholesky
factor from `fit()` already encodes `log(det(K))`. Use
`2 * sum(log(diag(cho[0])))` instead of `slogdet`.

### 7. Dead code in sweep loop

`sweep.py:87` — `springs = []` is assigned but never read.

## Nice-to-Haves

- `sweep.py:85` recreates solver per sample. Could hoist outside loop and
  only modify springs.
- `boundary_seeking` (`active_learning.py:57-59`) normalizes distance by σ
  in the exponent, which makes far-away high-σ points dominate over
  boundary-proximate points. Consider `exp(-0.5 * (μ - target)²) * σ`
  instead of `exp(-0.5 * ((μ - target)/σ)²) * σ`.

## Test Coverage Gaps

### Behavioral tests missing

| Module | What's untested |
|--------|----------------|
| `mode_tracking` | `track_modes_across_velocity` — no test that tracked frequencies/damping are smooth across velocity steps |
| `train.py` | `fit_gp` — no test that fitted GP has finite LML and reasonable RMSE on known function |
| `train.py` | `active_learning_loop` — no test that RMSE decreases over iterations |
| `train.py` | `brute_force_mc` — no test that failure probability is in [0,1] and n_success ≤ n_total |
| `train.py` | `evaluate_gp_accuracy` — no test at all |
| `active_learning` | No test that `select_batch` actually picks high-uncertainty or boundary-nearby points |
| `sweep` | No test that `flutter_lambda` values are physically sensible (>0) |
| `sweep` | No test that `n_success/n_total` ratio is reasonable |

### Edge cases missing

- Flutter boundary doesn't exist (no crossing in `[lambda_lower, lambda_upper]`) — does sweep report `n_success=0` gracefully?
- MAC matrix has ties (multiple assignments give same total cost)
- `select_batch` with `batch_size > len(X_candidates)`
- GP fit with 1 training point (degenerate kernel)
- GP predict at exact training point (variance should be ≈ noise_var)

### Specific code references

- `mode_tracking.py:83` — `freqs[assignment]` reorders correctly but
  `labels` logic at line 86-87 is confusing (not a bug, just unclear intent)
- `gp_surrogate.py:156` — jitter sequence `[0.0, 1e-8, ...]` starts with
  0.0 which will fail for ill-conditioned matrices; consider starting at 1e-8
- `train.py:197` — `lambdas < lambda_target` returns empty array if
  `n_success=0`, `np.sum` on empty is 0, so `n_fail=0` — correct but fragile
